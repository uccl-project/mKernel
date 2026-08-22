/**
 * @file ag_gemm.cu
 * @brief Multi-node All-Gather + GEMM - single fused kernel.
 *
 * Single kernel launch. Two CTA groups run concurrently:
 *
 *   Intra-comm CTAs [0, num_intra_comm):
 *     Phase 0 posts this rank's local A rows to the peer node via zero-copy
 *     RDMA as early as possible. Phase 1 gathers the local node's A shard into
 *     the multicast A buffer and signals per-(row,col) readiness for compute.
 *     Phase 2 waits for peer-node RDMA arrivals, republishes the received rows
 *     into a multicast A_recv buffer, and signals remote-row readiness.
 *
 *   Compute CTAs [num_intra_comm, 132):
 *     GEMM over local and remote halves. Local tiles wait on Phase-1 per-K
 *     signals; remote tiles wait on Phase-2 row signals. Tile order runs local
 *     work first, then remote work, giving RDMA more time to arrive.
 *
 * Coordination is fully device-side: multicast barriers for tile readiness,
 * arrival flags for RDMA completion, and an in-kernel reset before exit.
 *
 * Infrastructure (config, globals, helpers, host setup, entrypoint) lives in
 *   include/operators/ag_gemm/ag_gemm.cuh
 * Python/session glue + pybind module live in
 *   include/operators/ag_gemm/session.cuh
 */
#include "operators/ag_gemm/ag_gemm.cuh"
#include "operators/ag_gemm/ag_gemm_trace.cuh"

namespace ag_gemm_multinode {

__device__ inline void intra_comm_sm(const globals& G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(globals::NUM_COMM_CHUNKS < config::NUM_WARPS);
    typename globals::A_comm_tile (&A_smem)[globals::NUM_COMM_CHUNKS] =
        al.allocate<typename globals::A_comm_tile, globals::NUM_COMM_CHUNKS>();
    __shared__ kittens::semaphore inputs_arrived[globals::NUM_COMM_CHUNKS];

    const int comm_sm_id = blockIdx.x;
    const int warp_id = warp::groupid();
    const int lane_id = warp::laneid();
    const int global_row_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = G.A.cols() / (globals::RED_BLOCK * 2);
    const int num_local_blocks = local_row_blocks * col_blocks;
    uint32_t phasebits = 0xFFFF0000;

    if (warp_id < globals::NUM_COMM_CHUNKS && lane_id == 0) {
        init_semaphore(inputs_arrived[warp_id], 0, 1);

        // Gather own M_local shard from A[dev_idx] into A (multicast).
        if (G.debug_skip_phase1 == 0) {
            for (int task_id = comm_sm_id * globals::NUM_COMM_CHUNKS + warp_id;
                 task_id < num_local_blocks;
                 task_id += G.num_intra_comm * globals::NUM_COMM_CHUNKS) {
                const int row_idx = task_id / col_blocks;
                const int global_row_idx = row_idx + G.dev_idx * local_row_blocks;
                const int col_idx = task_id % col_blocks;
#ifdef AG_GEMM_TRACE
                const unsigned long long tr_t0 = trace::now_ns();
                const unsigned long long tr_c0 = clock64();
                unsigned long long tr_stall = 0;
#endif

                tma::expect_bytes(inputs_arrived[warp_id], sizeof(globals::A_comm_tile));
                tma::load_async(A_smem[warp_id], G.A[G.dev_idx], {global_row_idx, col_idx},
                                inputs_arrived[warp_id]);

                AG_TRACE_STALL_BEGIN(tr_w);
                wait(inputs_arrived[warp_id], get_phasebit<0>(phasebits, warp_id));
                AG_TRACE_STALL_END(tr_stall, tr_w);
                update_phasebit<0>(phasebits, warp_id);
                tma::store_async(G.A, A_smem[warp_id], {global_row_idx, col_idx});
                tma::store_async_wait();

                // Multicast store_async_wait only fences local-GPU completion;
                // cross-GPU visibility of the multicast write needs a system
                // fence before signaling compute (which lives on the same GPU
                // but reads via multicast). 
                __threadfence_system();

                // Plane 0 [row,col]: per-K-strip, count=1. Compute waits per red_idx
                // so it can stream tiles as cols arrive, not per whole row block.
                // Per-(row,col) count is 1 because each task_id is processed by
                // exactly one intra worker under the round-robin stripe.
                signal_all(G.barrier, {0, global_row_idx, col_idx}, 1);
#ifdef AG_GEMM_TRACE
                trace::emit(trace::ROLE_GATHER, task_id, tr_t0, trace::now_ns(),
                            tr_stall, 0ull, clock64() - tr_c0);
#endif
            }
        }
    }

    // Wait until every intra CTA has finished phase-1 multicast gather.
    // Must run outside the lane_id==0 branch so all threads in this CTA
    // execute __syncthreads (CUDA requires full-block participation).
    if (G.debug_skip_phase1_gate == 0 && threadIdx.x == 0) {
        int* counter = (int*)&G.barrier[G.dev_idx][{0, 1023, 1021}];
        const int my_arrival = atomicAdd(counter, 1);
        const int target = ((my_arrival / G.num_intra_comm) + 1) * G.num_intra_comm;
        while (atomicAdd(counter, 0) < target) {
            __nanosleep(50);
        }
    }
    __syncthreads();

#ifdef AG_GEMM_FASTPOLL
    // Publish "this CTA's phase-1 gather is complete" to every device. Placed
    // after __syncthreads() so all warps of this CTA have drained their task
    // loops (the gate above is thread-0 only and does not imply that). Each
    // task already did store_async_wait + __threadfence_system before its own
    // signal_all, so the multicast data is visible before this flag is.
    //
    // Compute reads this once per task: when it reaches NUM_DEVICES *
    // num_intra_comm, every plane-0 slot of the local shard is necessarily set,
    // so the per-K-strip readiness poll becomes pure overhead and is skipped.
    if (threadIdx.x == 0) {
        signal_all(G.barrier, {0, 1023, 1020}, 1);
    }
#endif

    if (G.debug_skip_phase2 != 0) {
        return;
    }

    // intra_comm_sm ranks r on the peer node have each RDMA-written their
    // M_local-row slice of peer A_half into THIS rank's recv_buf at the
    // corresponding A_half row offset [r*M_local, (r+1)*M_local). Exactly
    // one rank r's slice landed at each offset; OUR rank r 
    // workers fan out OUR slice via multicast into A_recv on all M ranks.

    const int K_val = G.A_recv_local_tensor.cols();
    const int chunks_per_inter_rb = max(1,
        (globals::ROW_BLOCK * K_val * (int)sizeof(bf16)) / CHUNK_BYTES);
    const int n_peers = G.num_nodes - 1;
    const int ring_steps = n_peers;
    const int rows_per_peer_slot = global_row_blocks;

    // Drain every recv_buf peer slot. ring_step is the hop order
    // (origin = node - 1 - step).
    for (int ring_step = 0; ring_step < ring_steps; ++ring_step) {
        const int origin_rank =
            ag_gemm_ring_origin_for_step(G.node_idx, G.num_nodes, ring_step);
        const int peer_slot =
            internode::slot_at_peer(origin_rank, G.node_idx, G.num_nodes);
        const int virt_arrival_slot = peer_slot + n_peers * ring_step;

        if (warp_id < globals::NUM_COMM_CHUNKS && lane_id == 0) {
            for (int task_id = comm_sm_id * globals::NUM_COMM_CHUNKS + warp_id;
                 task_id < num_local_blocks;
                 task_id += G.num_intra_comm * globals::NUM_COMM_CHUNKS) {

                const int row_idx = task_id / col_blocks;
                const int global_row_idx = row_idx + G.dev_idx * local_row_blocks;
                const int col_idx = task_id % col_blocks;
                const int slot_row_store =
                    peer_slot * rows_per_peer_slot + global_row_idx;
                const int slot_row_load = slot_row_store +
                    ring_step * (n_peers * rows_per_peer_slot);

                // Wait for the 2 underlying 128-row inter WRs that together fill
                // this 256-row intra_rb. post_merge_wrs_for_intra_row posts in
                // 128-row (ROW_BLOCK) rb units; global_row_idx is in 256-row
                // (ROW_BLOCK*2) units, so the two inter rbs are 2*global_row_idx
                // and 2*global_row_idx+1.
                //
                // Only wait once per intra_rb (on the first col task). Subsequent
                // col tasks for the same row land after the arrival flag already
                // cleared so the wait returns immediately, but hoisting is a
                // cheap correctness safeguard and matches how plane-0 flags on
                // col=0 already ratchet visibility for later cols.
                //
                // One intra tile is 256 rows but RDMA arrivals are tracked in
                // 128-row blocks — wait for both halves. Each inter rb is
                // posted as one WR (WR_SPLIT_CEILING=1) whose completion
                // raises the first chunk's arrival flag, so polling the
                // first chunk of each inter rb suffices.
                const int first_chunk_a = (2 * global_row_idx)     * chunks_per_inter_rb;
                const int first_chunk_b = (2 * global_row_idx + 1) * chunks_per_inter_rb;
                ag_gemm_wait_arrival_slot(G, virt_arrival_slot, first_chunk_a);
                ag_gemm_wait_arrival_slot(G, virt_arrival_slot, first_chunk_b);
                __threadfence_system();

                tma::expect_bytes(inputs_arrived[warp_id], sizeof(globals::A_comm_tile));
                tma::load_async(A_smem[warp_id], G.A_recv_local_tensor,
                                {slot_row_load, col_idx}, inputs_arrived[warp_id]);
                wait(inputs_arrived[warp_id], get_phasebit<0>(phasebits, warp_id));
                update_phasebit<0>(phasebits, warp_id);
                tma::store_async(G.A_recv, A_smem[warp_id], {slot_row_store, col_idx});
                tma::store_async_wait();
                __threadfence_system();

                if (G.remote_ready_per_col != 0) {
                    // Signal each republished A k-chunk. Remote compute waits
                    // on the matching chunk inside its red_idx loop.
                    signal_all(G.barrier, {2, slot_row_store, col_idx}, 1);
                } else {
                    // Default: count all k-chunks at row slot 0; remote compute
                    // waits for the whole row before consuming it.
                    signal_all(G.barrier, {2, slot_row_store, 0}, 1);
                }
            }
        }

        // Join all warps in this CTA before the ring cross-CTA gate (comm
        // subset runs TMA above; other warps must not enter that gate first).
        __syncthreads();

        // Each CTA forwards only the rows it owns after finishing its own
        // task loop. Rows are disjoint in the ring receive buffer, so an
        // unrelated CTA still publishing row Y does not block forwarding row X.

        if (warp_id < globals::NUM_COMM_CHUNKS && lane_id == 0) {
            if (G.ring_proxy_forward == 0 && ring_step + 1 < n_peers) {
                const int intra_col_blocks =
                    G.A_recv.cols() / (globals::RED_BLOCK * 2);
                for (int lr = comm_sm_id; lr < local_row_blocks;
                     lr += G.num_intra_comm) {
                    const int global_row_idx = lr + G.dev_idx * local_row_blocks;
                    const int slot_row_store =
                        peer_slot * rows_per_peer_slot + global_row_idx;
                    if (G.remote_ready_per_col != 0) {
                        for (int c = 0; c < intra_col_blocks; ++c) {
                            wait(G.barrier, {2, slot_row_store, c}, G.dev_idx, 1);
                        }
                    } else {
                        wait(G.barrier, {2, slot_row_store, 0}, G.dev_idx,
                             intra_col_blocks);
                    }
                    __threadfence_system();
                    post_ring_forward_wrs_for_intra_row(
                        G, peer_slot, origin_rank,
                        global_row_idx, chunks_per_inter_rb, ring_step + 1);
                }
            }
        }
    }

}

// ============================================================================
// Compute tile decode — shared between producer-load and producer-store warps
// ============================================================================
//
// Visit local tiles first, then remote tiles, using a SUPER_M row-major swizzle
// for L2 locality. Keeping local and remote phases separate gives RDMA more
// time to complete before remote tile consumption.
//
// `task_id` is logical, not global-shard ordered: shard_step=0 maps to this
// node's local shard on every node, then later shard_steps walk remote shards.
// This avoids node_idx>0 consuming remote tiles first and stalling on RDMA
// before doing independent local GEMM work.

__device__ inline comp_task decode_comp_task(int task_id,
                                             int super_rows,
                                             int final_rows,
                                             int super_blocks,
                                             int col_blocks,
                                             int total_local_tiles) {
    comp_task t;
    t.is_remote = (task_id >= total_local_tiles);
    const int flat = t.is_remote ? (task_id - total_local_tiles) : task_id;
    const int super_tile_limit = super_rows * col_blocks;
    if (flat < super_tile_limit) {
        t.rb      = globals::SUPER_M * (flat / super_blocks) + flat % globals::SUPER_M;
        t.col_idx = (flat % super_blocks) / globals::SUPER_M;
    } else {
        // Unreachable when final_rows==0 (then super_rows==node_row_blocks and
        // flat is always < super_tile_limit). Guard with max(1, ...) so the
        // div instruction the compiler emits is well-defined even on dead path.
        const int fr_safe = final_rows > 0 ? final_rows : 1;
        const int rem = flat - super_tile_limit;
        t.rb      = super_rows + rem % fr_safe;
        t.col_idx = rem / fr_safe;
    }
#ifdef AG_GEMM_ROWPERM
    // Match the order compute consumes rows to the order phase-1 produces them.
    //
    // Each GPU gathers only its own 1/8 shard -- intra rows [d*L, d*L+L) in
    // row-major order -- and all 8 GPUs run concurrently. So readiness sweeps
    // the intra-row space as {0, L, 2L, ...}, then {1, L+1, 2L+1, ...}: the
    // j-th row of every owner becomes available at roughly the same time.
    // The unpermuted task order instead walks rows 0,1,2,... so the first wave
    // of compute CTAs asks for 140 consecutive rows, of which only 8 (one per
    // owner) can possibly be ready. Measured: 86-92% of compute-CTA time is
    // spent stalled for the first 3.3 ms.
    //
    // Relabelling row r as (r % NUM_DEVICES) * L + (r / NUM_DEVICES) makes the
    // k-th row block consumed the k-th row block produced. It is a bijection,
    // so every tile is still computed exactly once; only the order changes.
    // It is also independent of dev_idx, so all GPUs keep walking tiles in
    // lockstep -- the multicast read sharing that made device rotation a
    // regression is preserved.
    if (!t.is_remote) {
        const int node_row_blocks = super_rows + final_rows;
        const int total_intra     = node_row_blocks / 2;
        const int local_intra     = total_intra / globals::NUM_DEVICES;
        if (local_intra > 0 && total_intra % globals::NUM_DEVICES == 0) {
            const int i     = t.rb >> 1;
            const int perm  = (i % globals::NUM_DEVICES) * local_intra
                            + (i / globals::NUM_DEVICES);
            t.rb = (perm << 1) | (t.rb & 1);
        }
    }
#endif
    return t;
}

__device__ __forceinline__ int ag_gemm_shard_rank_for_step(
    int node_idx, int num_nodes, int shard_step
) {
    return (node_idx + shard_step) % num_nodes;
}

// ============================================================================
// Comp SM: GEMM on both local and remote halves
// ============================================================================

#ifdef MKERNEL_TCGEN05
// Tensor memory holds exactly two 128x256 fp32 accumulators. Spend them on the
// two row blocks of a task rather than on double-buffering one: sharing a B
// tile across both is what raises arithmetic intensity, which measurement on
// gemm_rs showed matters (README_B300 s3.5).
static constexpr int ROW_BLOCKS_PER_TASK = globals::ROW_BLOCKS_PER_TASK;
#endif

__device__ inline void fused_comp_sm(const globals& G) {
    if (G.debug_skip_compute != 0) {
        return;
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
#ifdef MKERNEL_TCGEN05
    // Own allocation, so the loader never waits on the epilogue.
    globals::pipeline_outputs& outputs = allocator.allocate<globals::pipeline_outputs>();
#else
    globals::pipeline_outputs& outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(&inputs[globals::PIPELINE_STAGES - 1]);
#endif

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
#ifdef MKERNEL_TCGEN05
    // mma_done : MMA warp -> consumers, "tensor-memory accumulator is complete"
    // tmem_free: consumers -> MMA warp, "tensor memory drained, safe to reuse"
    // outputs_free: store warp -> consumers, between epilogue sub-rounds.
    __shared__ semaphore mma_done;
    __shared__ semaphore tmem_free[ROW_BLOCKS_PER_TASK];
    __shared__ semaphore outputs_free;
#endif
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
#ifdef MKERNEL_TCGEN05
            // Blackwell: the single tcgen05 MMA releases the stage (one
            // commit), instead of 8 consumer warps each arriving after wgmma.
            init_semaphore(inputs_finished[i], 0, 1);
#else
            init_semaphore(inputs_finished[i], 0, 8);
#endif
        }
        init_semaphore(outputs_arrived, 0, 1);   // one signal per sub-round
        init_semaphore(outputs_finished, 0, 1);
#ifdef MKERNEL_TCGEN05
        init_semaphore(mma_done,     0, 1);   // one commit covers both accumulators
        init_semaphore(outputs_free, 0, 1);   // one per epilogue sub-round
        #pragma unroll
        for (int i = 0; i < ROW_BLOCKS_PER_TASK; ++i)
            init_semaphore(tmem_free[i], 0, 2);   // one per consumer warpgroup
#endif
    }
    __syncthreads();

#ifdef MKERNEL_TCGEN05
    // Tensor-memory allocation is CTA-wide (the ctor runs tcgen05.alloc on warp
    // 0 then bar.sync 0), so every thread of a compute CTA must reach it and it
    // must sit outside the warp-role split below.
    tensor_allocator<1, 1> tm_alloc{};
    // All 512 columns: one 128x256 fp32 accumulator per row block of the task.
    auto d_tt_pool = tm_alloc.allocate<tt<float, globals::ROW_BLOCK,
                                                globals::COL_BLOCK * ROW_BLOCKS_PER_TASK>>(0);
#endif

    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    const int node_row_blocks = G.A_local.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int num_iters = G.A_local.cols() / globals::RED_BLOCK;

    const int super_rows = (node_row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = node_row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;

    const int num_node_blocks = node_row_blocks * col_blocks;
    const int total_blocks = num_node_blocks * G.num_nodes;
#ifdef MKERNEL_TCGEN05
    const int pairs_per_shard = num_node_blocks / globals::ROW_BLOCKS_PER_TASK;
    const int total_pairs     = pairs_per_shard * G.num_nodes;
#endif

    const int K_val = G.A_local.cols();
    const int chunks_per_rb = max(1, (globals::ROW_BLOCK * K_val * (int)sizeof(bf16)) / CHUNK_BYTES);

    // Task layout: shard-major. task_id < num_node_blocks → local shard;
    // task_id >= num_node_blocks → remote shards (one shard per shard_step).
    // Within each shard, flat index is SUPER_M-swizzled over
    // (node_row_blocks × col_blocks). See decode_comp_task() above for the
    // swizzle math.
    const int comp_idx = blockIdx.x - G.num_intra_comm;

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
#ifdef AG_GEMM_FASTPOLL
            // Sticky: once phase-1 is globally done it stays done for the epoch.
            bool all_gathered = false;
            const int gather_done_target = globals::NUM_DEVICES * G.num_intra_comm;
#endif
            // TMA load warp — CTA-stride over super-tile-swizzled tiles
            for (int pair_id = comp_idx; pair_id < total_pairs; pair_id += G.num_comp_sms) {
                // decode_comp_task varies rb fastest within a band, so flat and
                // flat+1 are adjacent row blocks at the same column. SUPER_M and
                // final_rows are both even, so flat=2k never straddles a band or
                // the super/tail boundary, and rb comes out even - which also
                // puts the pair inside one 256-row intra block, leaving the
                // per-K-strip barrier waits below untouched.
                const int shard_step = pair_id / pairs_per_shard;
                const int shard_rank = ag_gemm_shard_rank_for_step(
                    G.node_idx, G.num_nodes, shard_step);
                const int shard_task_id = 2 * (pair_id - shard_step * pairs_per_shard);
                const comp_task t = decode_comp_task(
                    shard_task_id, super_rows, final_rows, super_blocks, col_blocks,
                    num_node_blocks);
                const int rb = t.rb;
                const int col_idx = t.col_idx;
                const bool is_remote = (shard_rank != G.node_idx);
                if (is_remote && G.debug_skip_remote_compute != 0) {
                    continue;
                }
#ifdef AG_GEMM_TRACE
                const unsigned long long tr_t0 = trace::now_ns();
                const unsigned long long tr_c0 = clock64();
                unsigned long long tr_stall = 0;   // blocked on gather readiness
                unsigned long long tr_pipe  = 0;   // blocked on the compute pipeline
#endif
                int row_idx;
                int shard_rb = rb;
                int recv_peer_slot = 0;

                if (!is_remote) {
                    row_idx = rb;
                    // Local tiles: the fine per-(row,col) wait moves inside
                    // the red_idx loop below (keyed on red_idx/2 since one
                    // intra col_chunk = 2 compute K-strips).
                } else {
                    recv_peer_slot = internode::slot_at_peer(
                        shard_rank, G.node_idx, G.num_nodes);
                    row_idx = recv_peer_slot * node_row_blocks + rb;

                    // Remote tiles land in recv_buf via RDMA, then phase-2
                    // intra-AG republishes them into G.A_recv. Comp reads
                    // from the multicast-backed G.A_recv and waits on
                    // plane 2 once phase-2 has stored the row's tiles.
                    if (G.remote_ready_per_col == 0) {
                        // Plane 2 default is per-row count=col_blocks from all
                        // phase-2 workers for this intra row.
                        const int intra_rb = row_idx / 2;
                        const int intra_col_blocks = G.A_recv.cols() / (globals::RED_BLOCK * 2);
                        AG_TRACE_STALL_BEGIN(tr_w0);
                        wait(G.barrier, {2, intra_rb, 0}, G.dev_idx, intra_col_blocks);
                        AG_TRACE_STALL_END(tr_stall, tr_w0);
                        __threadfence_system();
                    }
                }

#ifdef AG_GEMM_OWNSHARD
                // Rows this device owns need no readiness wait at all. Phase 1
                // loads them from G.A[dev_idx] and multicast-stores them back
                // to G.A -- and A_local is built on that same local backing, so
                // for our own shard the copy writes identical bytes to the very
                // words compute reads. The data is already in place at kernel
                // start; the only thing the barrier was buying us was ordering
                // against a write that cannot change the value.
                bool own_shard = false;
                if (!is_remote) {
                    const int total_intra = (super_rows + final_rows) / 2;
                    const int local_intra = total_intra / globals::NUM_DEVICES;
                    if (local_intra > 0)
                        own_shard = ((row_idx >> 1) / local_intra) == G.dev_idx;
                }
#endif
#ifdef AG_GEMM_FASTPOLL
                if (!all_gathered) {
                    all_gathered = comm::atomic_u32::relaxed_load_s32_sys(
                        &G.barrier[G.dev_idx][{0, 1023, 1020}]) >= gather_done_target;
                }
#endif
                AG_TRACE_STALL_BEGIN(tr_w1);
                wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                AG_TRACE_STALL_END(tr_pipe, tr_w1);
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    // Per-K-strip wait on plane 0. Each intra col_chunk
                    // covers 2 compute K-strips, so wait when crossing the
                    // boundary.
#if defined(AG_GEMM_FASTPOLL) && defined(AG_GEMM_OWNSHARD)
                    if (!is_remote && !all_gathered && !own_shard && (red_idx & 1) == 0) {
#elif defined(AG_GEMM_FASTPOLL)
                    if (!is_remote && !all_gathered && (red_idx & 1) == 0) {
#elif defined(AG_GEMM_OWNSHARD)
                    if (!is_remote && !own_shard && (red_idx & 1) == 0) {
#else
                    if (!is_remote && (red_idx & 1) == 0) {
#endif
                        AG_TRACE_STALL_BEGIN(tr_w2);
                        wait(G.barrier, {0, row_idx / 2, red_idx / 2},
                             G.dev_idx, 1);
                        AG_TRACE_STALL_END(tr_stall, tr_w2);
                    }
                    if (is_remote && G.remote_ready_per_col != 0 && (red_idx & 1) == 0) {
                        AG_TRACE_STALL_BEGIN(tr_w3);
                        wait(G.barrier, {2, row_idx / 2, red_idx / 2},
                             G.dev_idx, 1);
                        AG_TRACE_STALL_END(tr_stall, tr_w3);
                        __threadfence_system();
                    }
                    AG_TRACE_STALL_BEGIN(tr_w4);
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    AG_TRACE_STALL_END(tr_pipe, tr_w4);
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
#ifdef MKERNEL_TCGEN05
                    // One 128-row A tile; coordinates are in units of A_tile,
                    // so the *2+i indexing of the two-half layout collapses.
                    // Both row blocks of the pair; they share the B tile below.
                    #pragma unroll
                    for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                        if (is_remote) {
                            tma::load_async(inputs[stage].A[h], G.A_recv[G.dev_idx],
                                            {recv_peer_slot * node_row_blocks + shard_rb + h, red_idx},
                                            inputs_arrived[stage]);
                        } else {
                            tma::load_async(inputs[stage].A[h], G.A_local,
                                            {row_idx + h, red_idx}, inputs_arrived[stage]);
                        }
                    }
#else
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        if (is_remote) {
                            // Remote tiles live in the multicast-backed
                            // A_recv dbuf after phase-2. Read from this
                            // rank's unicast view (G.A_recv[dev_idx]).
                            tma::load_async(inputs[stage].A[i], G.A_recv[G.dev_idx],
                                            {(recv_peer_slot * node_row_blocks + shard_rb) * 2 + i, red_idx}, inputs_arrived[stage]);
                        } else {
                            tma::load_async(inputs[stage].A[i], G.A_local,
                                            {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                        }
                    }
#endif
                    tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
#ifdef AG_GEMM_TRACE
                trace::emit(trace::ROLE_COMPUTE, pair_id, tr_t0, trace::now_ns(),
                            tr_stall, tr_pipe, clock64() - tr_c0);
#endif
            }
        }
#ifdef MKERNEL_TCGEN05
        else if (warp_id == 2 && lane_id == 0) {
            // Blackwell MMA issuer. tcgen05 MMAs are issued by one thread and
            // accumulate in tensor memory, so this warp replaces the wgmma the
            // consumers used to run. It walks the same task order as the loader
            // and consumers — including the identical remote-skip — so the
            // input pipeline stays in lockstep.
            for (int pair_id = comp_idx; pair_id < total_pairs; pair_id += G.num_comp_sms) {
                const int shard_step = pair_id / pairs_per_shard;
                const int shard_rank = ag_gemm_shard_rank_for_step(
                    G.node_idx, G.num_nodes, shard_step);
                if (shard_rank != G.node_idx && G.debug_skip_remote_compute != 0) {
                    continue;
                }
                // Tensor memory must be drained before the accumulate=0 MMA
                // overwrites it.
                using acc_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>;
                #pragma unroll
                for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                    wait(tmem_free[h], get_phasebit<1>(phasebits, h));
                    update_phasebit<1>(phasebits, h);
                }
                auto d0 = d_tt_pool.template subtile<acc_t>(0, 0);
                auto d1 = d_tt_pool.template subtile<acc_t>(0, globals::COL_BLOCK);
                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    // mm_AB zeroes the accumulator, mma_AB accumulates onto it.
                    // The semaphore fires once the MMA has consumed the shared
                    // operands, releasing the stage back to the loader.
                    // Two MMAs off one B tile. Only the second carries the
                    // semaphore: tcgen05.commit covers every MMA issued before
                    // it, so one commit releases the stage once both have read.
                    if (red_idx == 0) {
                        warp::mm_AB (d0, inputs[stage].A[0], inputs[stage].B);
                        warp::mm_AB (d1, inputs[stage].A[1], inputs[stage].B, inputs_finished[stage]);
                    } else {
                        warp::mma_AB(d0, inputs[stage].A[0], inputs[stage].B);
                        warp::mma_AB(d1, inputs[stage].A[1], inputs[stage].B, inputs_finished[stage]);
                    }
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
                kittens::tensor_commit<1>(mma_done);
            }
        }
#endif
        else if (warp_id == 1 && lane_id == 0) {
            // TMA store warp — same super-tile-swizzled task order as loader
            for (int pair_id = comp_idx; pair_id < total_pairs; pair_id += G.num_comp_sms) {
                // decode_comp_task varies rb fastest within a band, so flat and
                // flat+1 are adjacent row blocks at the same column. SUPER_M and
                // final_rows are both even, so flat=2k never straddles a band or
                // the super/tail boundary, and rb comes out even - which also
                // puts the pair inside one 256-row intra block, leaving the
                // per-K-strip barrier waits below untouched.
                const int shard_step = pair_id / pairs_per_shard;
                const int shard_rank = ag_gemm_shard_rank_for_step(
                    G.node_idx, G.num_nodes, shard_step);
                const int shard_task_id = 2 * (pair_id - shard_step * pairs_per_shard);
                const comp_task t = decode_comp_task(
                    shard_task_id, super_rows, final_rows, super_blocks, col_blocks,
                    num_node_blocks);
                const int rb = t.rb;
                const int col_idx = t.col_idx;
                const int row_idx = shard_rank * node_row_blocks + rb;
                if (shard_rank != G.node_idx && G.debug_skip_remote_compute != 0) {
                    continue;
                }

#ifdef MKERNEL_TCGEN05
                // One staging tile for four 64-row halves (two row blocks x two
                // halves), drained in sequence. outputs_free hands the tile back
                // between sub-rounds; outputs_finished closes the task.
                int sub = 0;
                #pragma unroll
                for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                    #pragma unroll
                    for (int hs = 0; hs < 2; hs++) {
                        wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                        update_phasebit<0>(phasebits, 0);
                        tma::store_async(G.C, outputs.C, {(row_idx + h) * 2 + hs, col_idx});
                        // store_async_wait waits for the global commit (not just
                        // smem reuse safety like read_wait). At large M,
                        // store-in-flight can race with downstream reads of C.
                        tma::store_async_wait();
                        if (++sub == 2 * globals::ROW_BLOCKS_PER_TASK) arrive(outputs_finished);
                        else                                           arrive(outputs_free);
                    }
                }
#else
                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::store_async(G.C, outputs.C[i], {row_idx * 2 + i, col_idx});
                tma::store_async_wait();
                arrive(outputs_finished);
#endif
            }
        }
    } else {
        // Consumer warpgroups: WGMMA — same tile count, same CTA-stride
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

#ifdef MKERNEL_TCGEN05
        for (int pair_id = comp_idx; pair_id < total_pairs; pair_id += G.num_comp_sms) {
            const int shard_step = pair_id / pairs_per_shard;
#else
        for (int task_id = comp_idx; task_id < total_blocks; task_id += G.num_comp_sms) {
            const int shard_step = task_id / num_node_blocks;
#endif
            const int shard_rank = ag_gemm_shard_rank_for_step(
                G.node_idx, G.num_nodes, shard_step);
            if (shard_rank != G.node_idx && G.debug_skip_remote_compute != 0) {
                continue;
            }
            rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
#ifdef MKERNEL_TCGEN05
            // Blackwell consumers do no MMA. They wait for the tensor-memory
            // accumulator, pull their 16-row slice into registers, release
            // tmem, then stage the tile in shared for the epilogue TMA warp.
            //
            // Row mapping follows ThunderKittens' group<8> tmem layout: the 8
            // consumer warps each own 16 of the 128 rows, at
            // `32*(w%4) + 16*(w/4)`. Two warps share each 32-lane tmem
            // sub-partition, which is the granularity tcgen05.ld requires.
            const int cw   = warpgroup_id * 4 + warp_id;      // consumer warp 0..7
            const int trow = 32 * (cw % 4) + 16 * (cw / 4);   // its tmem row

            // One signal covers both accumulators: tcgen05.commit fires only
            // after every MMA issued before it has completed.
            wait(mma_done, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            #pragma unroll
            for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                warp::load_async(C_accum,
                                 d_tt_pool.template subtile<tt<float, globals::ROW_BLOCK / 8,
                                                                    globals::COL_BLOCK>>(
                                     trow, globals::COL_BLOCK * h));
                kittens::tensor_load_wait();
                group<8>::sync(3);
                // Registers hold this accumulator, so tensor memory is free
                // even though the stores are still ahead.
                warpgroup::arrive(tmem_free[h]);

                // The single staging tile takes one 64-row half at a time.
                // Under the group<8> tmem layout warps {0,1,4,5} own rows < 64
                // and {2,3,6,7} own the rest, so each sub-round only four of
                // the eight warps write; the others just ride the barrier.
                #pragma unroll
                for (int hs = 0; hs < 2; hs++) {
                    if (h == 0 && hs == 0) {
                        // First sub-round of the task waits on the previous
                        // one. The <1> half starts signalled, so task 0 passes.
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        update_phasebit<1>(phasebits, 0);
                    } else {
                        wait(outputs_free, get_phasebit<0>(phasebits, 1));
                        update_phasebit<0>(phasebits, 1);
                    }
                    if (trow / 64 == hs) {
                        auto dst = outputs.C.template subtile<globals::ROW_BLOCK / 8,
                                                              globals::COL_BLOCK>(
                                       {(trow % 64) / (globals::ROW_BLOCK / 8), 0});
                        warp::store(dst, C_accum);
                    }
                    group<8>::sync(4);
                    if (warpgroup_id == 0 && warp_id == 0 && lane_id == 0)
                        arrive(outputs_arrived);
                }
            }
#else
            warp::zero(C_accum);

            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
                warpgroup::mma_async_wait();
                warp::arrive(inputs_finished[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }

            group<8>::sync(3);
            warpgroup::store(outputs.C[warpgroup_id], C_accum);
            warpgroup::sync(warpgroup_id + 1);
            warpgroup::arrive(outputs_arrived);
#endif
        }
    }
}

// ============================================================================
// Kernel entry + epilogue
// ============================================================================

// Grid-wide sync inside the persistent kernel using a monotonic counter at a
// reserved barrier slot {0, 1023, 1022} (per-device — barrier[dev_idx] is the
// local view, no multicast traffic). Each CTA derives its own per-iter target
// from its arrival number, so no host-side counter init is needed and the
// counter just grows monotonically. 
//
// Atomic polling forces L2 coherence on each check before the flag-plane reset.
__device__ inline void grid_sync_at_epoch(const globals& G) {
    __syncthreads();
    if (threadIdx.x == 0) {
        int* counter = (int*)&G.barrier[G.dev_idx][{0, 1023, 1022}];
        const int my_arrival = atomicAdd(counter, 1);
        const int target = ((my_arrival / (int)gridDim.x) + 1) * (int)gridDim.x;
        while (atomicAdd(counter, 0) < target) {
            __nanosleep(50);
        }
    }
    __syncthreads();
}

__device__ inline void barrier_reset(const globals& G);

__device__ inline void fused_kernel(const globals& G) {
    if (blockIdx.x < G.num_intra_comm) {
        intra_comm_sm(G);
    } else {
        fused_comp_sm(G);
    }

    // Inline the iter-end barrier reset so it shares the kernel launch
    // rather than incurring a separate cudaLaunchKernel. Grid-sync first
    // so no CTA clears a flag plane while another CTA is still signaling
    // into it.
    if (G.debug_skip_reset != 0) {
        return;
    }
    grid_sync_at_epoch(G);
    barrier_reset(G);
}

__device__ inline void barrier_reset(const globals& G) {
    // Barrier uses two active planes:
    //   plane 0:            [row_blocks, col_blocks] per-(row,col), count=1
    //   plane 2 default:    [row_blocks,          1] per-row,       count=num_cols
    //   plane 2 experiment: [row_blocks,  num_cols] per-(row,col),  count=1
    const int num_rows = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int num_cols = G.A.cols() / (globals::RED_BLOCK * 2);
    const int total_p0 = num_rows * num_cols;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < total_p0; i += stride) {
        int r = i / num_cols;
        int c = i % num_cols;
        G.barrier[G.dev_idx][{0, r, c}] = 0;
    }
    const int total_p2 = num_rows * (G.num_nodes - 1) *
        (G.remote_ready_per_col != 0 ? num_cols : 1);
    for (int i = offset; i < total_p2; i += stride) {
        if (G.remote_ready_per_col != 0) {
            const int r = i / num_cols;
            const int c = i % num_cols;
            G.barrier[G.dev_idx][{2, r, c}] = 0;
        } else {
            G.barrier[G.dev_idx][{2, i, 0}] = 0;
        }
    }
#ifdef AG_GEMM_FASTPOLL
    // Clear the phase-1-complete flag before the cross-device sync below, so
    // the next epoch starts from zero. grid_sync_at_epoch already ran, so no
    // CTA is still reading it.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        G.barrier[G.dev_idx][{0, 1023, 1020}] = 0;
#endif
    // Iter-end cross-device sync uses a slot outside the active data planes
    // (max shape is (num_rows<=128, num_cols<=1024) under validate_shapes).
    if (blockIdx.x == 0 && threadIdx.x == 0)
        barrier_all(G.barrier, {0, 1023, 1023}, G.dev_idx);
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void ag_gemm_fused_kernel_stub(const __grid_constant__ globals G) {
    fused_kernel(G);
}

// ============================================================================
// Prologue kernel
// ============================================================================
// Posts the inter-node RDMA WRs (zero-copy from A.data_ DMA-BUF MR) to the
// host proxy's D2H FIFO. The proxy can begin issuing post_send / waiting on
// CQE in parallel with the main kernel's launch + intra-AG phase, hiding
// kernel-launch + intra-comm-CTA-startup latency from the EFA critical path.
//
// Work distribution mirrors the original (one intra row per CTA, stride
// num_intra_comm). One CTA, one warp, one thread per CTA does the push —
// fifo.push() is thread-safe per queue.
__device__ inline void phase0_post_wrs(const globals& G) {
    const int comm_sm_id = blockIdx.x;
    const int warp_id = warp::groupid();
    const int lane_id = warp::laneid();
    const int global_row_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int K_val_for_merge = G.A_local.cols();
    const int chunks_per_rb_for_merge = max(1,
        (globals::ROW_BLOCK * K_val_for_merge * (int)sizeof(bf16)) / CHUNK_BYTES);
    if (warp_id == 0 && lane_id == 0) {
        for (int lr = comm_sm_id; lr < local_row_blocks; lr += G.num_intra_comm) {
            const int global_row_idx = lr + G.dev_idx * local_row_blocks;
            post_merge_wrs_for_intra_row(
                G, global_row_idx, chunks_per_rb_for_merge);
        }
    }
}

__global__ void ag_gemm_phase0_prologue_kernel(const __grid_constant__ globals G) {
    phase0_post_wrs(G);
}

void launch_fused_ag_gemm(const globals& G, unsigned int active_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        ag_gemm_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    // Side-stream prologue.
    //
    // The prologue (tiny kernel that pushes RDMA WRs to the host-proxy FIFO)
    // is launched on a SEPARATE non-blocking CUDA stream so its FIFO push can
    // race past the main-stream launch latency / inter-iter Python-side work.
    // The main fused kernel does NOT wait on the prologue at the device level
    // — the proxy thread reads the FIFO from host-visible memory independently
    // of the device-side scheduling. The bench's per-iter cuda.synchronize()
    // still drains both streams, so end-of-iter ordering is preserved.
    //
    // Cross-stream sync model:
    //   1. Record "main_pre" event on main stream (captures any prior
    //      main-stream work — e.g. local_A copies, prior iter completion).
    //   2. prologue stream waits on "main_pre" so the prologue cannot run
    //      before prior local-data writes are device-visible.
    //   3. Launch prologue on the side stream.
    //   4. Launch the main fused kernel on the main stream WITHOUT waiting on
    //      the prologue — that is the overlap. Both kernels then race; the
    //      proxy picks up FIFO entries as the prologue makes them visible.
    //
    // The side stream + events are session-lifetime singletons so we don't
    // pay creation cost per launch. A static-local guarded by a flag suffices
    // since launches are serialized on a single host thread per session.
    static cudaStream_t prologue_stream = nullptr;
    static cudaEvent_t main_pre_event = nullptr;
    static bool side_stream_inited = false;
    if (!side_stream_inited) {
        MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(
            &prologue_stream, cudaStreamNonBlocking));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(
            &main_pre_event, cudaEventDisableTiming));
        side_stream_inited = true;
    }

    const int prologue_blocks = G.num_intra_comm > 0 ? G.num_intra_comm : 1;
    // Single-node runs have no peers, so there are no RDMA work requests to
    // post. Skipping is not just an optimisation: the caller has no session,
    // so the d2h FIFO handles are null and the prologue would fault on them.
    const bool skip_prologue =
        G.num_nodes == 1 ||
        (std::getenv("AG_GEMM_SKIP_PROLOGUE") != nullptr &&
         std::getenv("AG_GEMM_SKIP_PROLOGUE")[0] == '1');
    if (skip_prologue) {
        ag_gemm_fused_kernel_stub<<<active_sms, config::NUM_THREADS,
                                    dynamic_shared_memory, stream>>>(G);
        return;
    }
    // Ring: post phase-0 merge WRs on the same stream as the fused kernel so
    // the prologue fully completes before intra_comm_sm begins. Side-stream
    // overlap would let the main kernel start while prologue CTAs are still
    // pushing FIFO entries, which can starve or reorder the ring merge vs
    // phase-2 arrival waits. Opt back to side-stream with
    // AG_GEMM_PROLOGUE_SIDE_STREAM=1.
    const bool force_side_stream =
        std::getenv("AG_GEMM_PROLOGUE_SIDE_STREAM") != nullptr &&
        std::getenv("AG_GEMM_PROLOGUE_SIDE_STREAM")[0] == '1';
    const bool prologue_main_stream = !force_side_stream;
    if (prologue_main_stream) {
        ag_gemm_phase0_prologue_kernel<<<prologue_blocks, WARP_THREADS, 0,
                                         stream>>>(G);
        ag_gemm_fused_kernel_stub<<<active_sms, config::NUM_THREADS,
                                    dynamic_shared_memory, stream>>>(G);
        return;
    }
    // Capture prior main-stream state.
    MKERNEL_CUDACHECK(cudaEventRecord(main_pre_event, stream));
    // Prologue stream waits on it.
    MKERNEL_CUDACHECK(cudaStreamWaitEvent(prologue_stream, main_pre_event, 0));
    // Launch prologue on the side stream.
    ag_gemm_phase0_prologue_kernel<<<prologue_blocks, WARP_THREADS, 0,
                                     prologue_stream>>>(G);
    // Launch main kernel — does NOT wait on prologue.
    ag_gemm_fused_kernel_stub<<<active_sms, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}

}  // namespace ag_gemm_multinode

#include "operators/ag_gemm/session.cuh"
