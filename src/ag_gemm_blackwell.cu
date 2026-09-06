/**
 * @file ag_gemm_blackwell.cu
 * @brief All-Gather + GEMM on Blackwell (tcgen05), split out of ag_gemm.cu.
 *
 * The Blackwell counterpart to ag_gemm.cu: 2-CTA tcgen05 clusters, B taken
 * N-major so the MMA issues as ABt, and a tensor-memory accumulator. Neither
 * file carries the other's MMA path, so there is no MKERNEL_TCGEN05 here.
 *
 * Python/session glue + pybind module live in
 *   include/operators/ag_gemm/ag_gemm_blackwell_session.cuh
 */
#include "operators/ag_gemm/ag_gemm_blackwell.cuh"

namespace ag_gemm_blackwell {

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

                tma::expect_bytes(inputs_arrived[warp_id], sizeof(globals::A_comm_tile));
                tma::load_async(A_smem[warp_id], G.A[G.dev_idx], {global_row_idx, col_idx},
                                inputs_arrived[warp_id]);

                wait(inputs_arrived[warp_id], get_phasebit<0>(phasebits, warp_id));
                update_phasebit<0>(phasebits, warp_id);
                tma::store_async(G.A, A_smem[warp_id], {global_row_idx, col_idx});
                tma::store_async_wait();

                // No fence here: signal_all below emits
                // multimem.red.release.sys, and a release orders this thread's
                // prior writes -- the multicast store included -- ahead of the
                // signal becoming observable. The __threadfence_system() that
                // used to sit here was asking for the guarantee the very next
                // instruction already provides. Dropping it is worth 14.4% at
                // M=4096, 6.3% at 8192, 2.7% at 16384, 1.8% at 32768, and it
                // tightens run-to-run spread (2.9% -> 0.2% at M=4096).
                // AG_GEMM_CHUNK_FENCE=1 restores it, as a one-flag bisect if a
                // corruption ever points back here.
#if AG_GEMM_CHUNK_FENCE
                __threadfence_system();
#endif

                // Plane 0 [row,col]: per-K-strip, count=1. Compute waits per red_idx
                // so it can stream tiles as cols arrive, not per whole row block.
                // Per-(row,col) count is 1 because each task_id is processed by
                // exactly one intra worker under the round-robin stripe.
                signal_all(G.barrier, {0, global_row_idx, col_idx}, 1);
#ifdef AG_GEMM_ROWPOLL
                // Plane 1 is otherwise unused: one counter per intra row.
                // Reaching col_blocks means every K chunk of that row is
                // published, which lets a compute task skip its per-chunk
                // polls entirely. Workers are strided across the flat
                // (row, col) space, so they finish a row's columns out of
                // order -- only the full count is a sound inference, not a
                // prefix.
                signal_all(G.barrier, {1, global_row_idx, 0}, 1);
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

}

// ============================================================================
// Compute tile decode — shared between producer-load and producer-store warps
// ============================================================================
//
// A SUPER_M row-major swizzle over this node's own shard, for L2 locality.
// ag_gemm.cu also walks remote shards after the local one; here there is only
// the local shard, so task_id indexes it directly.

__device__ inline comp_task decode_comp_task(int task_id,
                                             int super_rows,
                                             int final_rows,
                                             int super_blocks,
                                             int col_blocks) {
    comp_task t;
    const int flat = task_id;
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
    // Consume rows in the order phase-1 produces them. Each GPU gathers only
    // its own shard -- rows [d*L, d*L+L) in order, all GPUs concurrently -- so
    // readiness sweeps {0, L, 2L, ...} then {1, L+1, ...}, while the unpermuted
    // order asks for 0,1,2,... of which only one row per owner can be ready.
    // Relabelling r as (r % NUM_DEVICES) * L + r / NUM_DEVICES is a bijection
    // and independent of dev_idx, so the lockstep order that multicast read
    // sharing depends on survives. Worth 25% at M=32768.
    {
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

// ============================================================================
// Comp SM: GEMM on both local and remote halves
// ============================================================================

// Tensor memory holds exactly two 128x256 fp32 accumulators. Spend them on the
// two row blocks of a task rather than on double-buffering one: sharing a B
// tile across both is what raises arithmetic intensity, which measurement on
// gemm_rs showed matters (README_B300 s3.5).
static constexpr int ROW_BLOCKS_PER_TASK = globals::ROW_BLOCKS_PER_TASK;

__device__ inline void fused_comp_sm(const globals& G,
                                     tensor_allocator<1, config::CLUSTER_SIZE>& tm_alloc) {
    if (G.debug_skip_compute != 0) {
        return;
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    // Own allocation, so the loader never waits on the epilogue.
    globals::pipeline_outputs& outputs = allocator.allocate<globals::pipeline_outputs>();

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived[globals::OUTPUT_BUFFERS];
    __shared__ semaphore outputs_finished;
    // mma_done : MMA warp -> consumers, "tensor-memory accumulator is complete"
    // tmem_free: consumers -> MMA warp, "tensor memory drained, safe to reuse"
    // outputs_free: store warp -> consumers, between epilogue sub-rounds.
    __shared__ semaphore mma_done;
    __shared__ semaphore tmem_free[ROW_BLOCKS_PER_TASK];
    __shared__ semaphore outputs_free[globals::OUTPUT_BUFFERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            // One expect() per MMA warp on the leader CTA.
            init_semaphore(inputs_arrived[i], 0, globals::NUM_MMA_WARPS);
            // Blackwell: the tcgen05 MMA releases the stage (one commit per
            // MMA warp), instead of 8 consumer warps each arriving after wgmma.
            init_semaphore(inputs_finished[i], 0, globals::NUM_MMA_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < globals::OUTPUT_BUFFERS; ++i)
            init_semaphore(outputs_arrived[i], 0, 1);   // one signal per sub-round
        init_semaphore(outputs_finished, 0, 1);
        // One commit per MMA warp; at CLUSTER_SIZE 1 the single warp issues both.
        init_semaphore(mma_done,     0, globals::NUM_MMA_WARPS);
        #pragma unroll
        for (int i = 0; i < globals::OUTPUT_BUFFERS; ++i)
            init_semaphore(outputs_free[i], 0, 1);   // one per epilogue sub-round
        #pragma unroll
        for (int i = 0; i < ROW_BLOCKS_PER_TASK; ++i)
            init_semaphore(tmem_free[i], 0, 2 * config::CLUSTER_SIZE);   // warpgroups x CTAs
    }
    __syncthreads();

    // Tensor-memory allocation is CTA-wide (the ctor runs tcgen05.alloc on warp
    // 0 then bar.sync 0), so every thread of a compute CTA must reach it and it
    // must sit outside the warp-role split below.
    // tm_alloc is constructed by fused_kernel: at ncta 2 its destructor runs a
    // cluster barrier, so every CTA of every cluster -- gather CTAs included --
    // has to construct and destroy it, not just the compute ones.
    // All 512 columns: one 128x256 fp32 accumulator per row block of the task.
    auto d_tt_pool = tm_alloc.allocate<tt<float, globals::ROW_BLOCK,
                                                globals::COL_BLOCK * ROW_BLOCKS_PER_TASK>>(0);

    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    const int node_row_blocks = G.A_local.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.rows() / globals::COL_BLOCK;   // B is (N, K)
    const int num_iters = G.A_local.cols() / globals::RED_BLOCK;

    const int super_rows = (node_row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = node_row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;

    const int num_node_blocks = node_row_blocks * col_blocks;
    const int total_blocks = num_node_blocks;
    // Cluster task space. At CLUSTER_SIZE 1 these collapse onto the pair space
    // above and every index below is bit-identical to the pre-cluster code.
    const int clusters_per_shard = num_node_blocks / globals::ROW_BLOCKS_PER_CLUSTER;
    const int total_cluster_tasks = clusters_per_shard;
    const int ctarank = (config::CLUSTER_SIZE > 1) ? (int)cluster_ctarank() : 0;

    const int K_val = G.A_local.cols();
    const int chunks_per_rb = max(1, (globals::ROW_BLOCK * K_val * (int)sizeof(bf16)) / CHUNK_BYTES);

    // One shard. Flat index is SUPER_M-swizzled over
    // (node_row_blocks × col_blocks); see decode_comp_task() above.
    const int comp_idx = blockIdx.x - G.num_intra_comm;
    // Clusters are consecutive blockIdx.x. num_intra_comm is forced even at
    // CLUSTER_SIZE 2 (see the entrypoint), so comp_idx 0 is always a cluster
    // leader and no cluster straddles the gather/compute role split.
    const int cluster_id   = comp_idx / config::CLUSTER_SIZE;
    const int num_clusters = G.num_comp_sms / config::CLUSTER_SIZE;

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
#ifdef AG_GEMM_FASTPOLL
            // Sticky: once phase-1 is globally done it stays done for the epoch.
            bool all_gathered = false;
            const int gather_done_target = globals::NUM_DEVICES * G.num_intra_comm;
#endif
            // TMA load warp — CTA-stride over super-tile-swizzled tiles
            for (int pair_id = cluster_id; pair_id < total_cluster_tasks; pair_id += num_clusters) {
                // decode_comp_task varies rb fastest within a band, so flat and
                // flat+1 are adjacent row blocks at the same column. SUPER_M and
                // final_rows are both even, so flat=2k never straddles a band or
                // the super/tail boundary, and rb comes out even - which also
                // puts the pair inside one 256-row intra block, leaving the
                // per-K-strip barrier waits below untouched.
                const int shard_task_id =
                    globals::ROW_BLOCKS_PER_CLUSTER * pair_id
                  + globals::ROW_BLOCKS_PER_TASK * ctarank;
                const comp_task t = decode_comp_task(
                    shard_task_id, super_rows, final_rows, super_blocks, col_blocks);
                const int rb = t.rb;
                const int col_idx = t.col_idx;
                const int row_idx = rb;

#ifdef AG_GEMM_FASTPOLL
                if (!all_gathered) {
                    all_gathered = comm::atomic_u32::relaxed_load_s32_sys(
                        &G.barrier[G.dev_idx][{0, 1023, 1020}]) >= gather_done_target;
                }
#endif
#ifdef AG_GEMM_ROWPOLL
                // A task's own row is complete long before the last GPU has
                // finished gathering, so this fires far earlier than the global
                // all_gathered flag: one load instead of 256.
                bool row_ready = all_gathered;
                if (!row_ready) {
                    const int intra_cols = G.A.cols() / (globals::RED_BLOCK * 2);
                    row_ready = comm::atomic_u32::relaxed_load_s32_sys(
                        &G.barrier[G.dev_idx][{1, row_idx / 2, 0}]) >= intra_cols;
                }
#else
                const bool row_ready = all_gathered;
#endif
                wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    // Per-K-strip wait on plane 0. Each intra col_chunk
                    // covers 2 compute K-strips, so wait when crossing the
                    // boundary.
                    if (!row_ready && (red_idx & 1) == 0) {
                        // The writer is a *peer* GPU's multimem.red, so
                        // atomic_u32's own contract says this load must be
                        // acquire, not relaxed.
#if AG_GEMM_ACQUIRE_WAIT
                        wait_acquire(G.barrier, {0, row_idx / 2, red_idx / 2},
                                     G.dev_idx, 1);
#else
                        wait(G.barrier, {0, row_idx / 2, red_idx / 2},
                             G.dev_idx, 1);
#endif
                    }
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    // One 128-row A tile; coordinates are in units of A_tile,
                    // so the *2+i indexing of the two-half layout collapses.
                    // Both row blocks of the pair; they share the B tile below.
                    if constexpr (config::CLUSTER_SIZE > 1) {
                        // Each CTA loads its own slice (mask = 1 << ctarank) but
                        // points the completion at the leader's barrier
                        // (dst_mbar_cta = 0), which the MMA warps' single
                        // cluster::expect accounts for. No expect here.
                        #pragma unroll
                        for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                            tma::cluster::load_async(
                                inputs[stage].A[h], G.A_local,
                                {row_idx + h, red_idx},
                                inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
                        }
                    } else {
                        tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                        #pragma unroll
                        for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                            tma::load_async(inputs[stage].A[h], G.A_local,
                                            {row_idx + h, red_idx}, inputs_arrived[stage]);
                        }
                    }
                    // B is (N, K): tile row selects this CTA's column slab.
                    if constexpr (config::CLUSTER_SIZE > 1) {
                        tma::cluster::load_async(
                            inputs[stage].B, G.B,
                            {col_idx * config::CLUSTER_SIZE + ctarank, red_idx},
                            inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
                    } else
                    tma::load_async(inputs[stage].B, G.B,
                                    {col_idx * config::CLUSTER_SIZE + ctarank, red_idx},
                                    inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
        else if (warp_id >= 2 && warp_id < 2 + globals::NUM_MMA_WARPS
                 && lane_id == 0 && ctarank == 0) {
            // Blackwell MMA issuer: tcgen05 accumulates in tensor memory, so
            // one thread issues what the consumers' wgmma used to do. Walks the
            // same task order (and remote-skip) as loader and consumers.
            // At CLUSTER_SIZE 2 the MMA spans the pair -- A split by rows, B by
            // columns -- so only the leader issues, one warp per accumulator.
            for (int pair_id = cluster_id; pair_id < total_cluster_tasks; pair_id += num_clusters) {
                // Tensor memory must be drained before the accumulate=0 MMA
                // overwrites it.
                using acc_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>;
                if constexpr (config::CLUSTER_SIZE > 1) {
                    // One accumulator per MMA warp, so warp 2 is released as
                    // soon as accumulator 0 is drained cluster-wide.
                    const int acc = warp_id - 2;
                    tma::cluster::wait(tmem_free[acc], get_phasebit<1>(phasebits, acc));
                    update_phasebit<1>(phasebits, acc);
                } else {
                    #pragma unroll
                    for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                        wait(tmem_free[h], get_phasebit<1>(phasebits, h));
                        update_phasebit<1>(phasebits, h);
                    }
                }
                auto d0 = d_tt_pool.template subtile<acc_t>(0, 0);
                auto d1 = d_tt_pool.template subtile<acc_t>(
                    0, globals::COL_BLOCK * (globals::ROW_BLOCKS_PER_TASK - 1));
                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    if constexpr (config::CLUSTER_SIZE > 1) {
                        // One expect per MMA warp; together the two account for
                        // the six tile loads (three per CTA) the loaders
                        // redirected at the leader's barrier.
                        // Both CTAs point their transactions at the leader's
                        // barrier, so the expects must add up to the whole
                        // cluster's bytes: CLUSTER_SIZE x (RBPT A tiles + B).
                        // NUM_MMA_WARPS warps share that, one expect each.
                        if constexpr (globals::ROW_BLOCKS_PER_TASK == 2)
                            tma::cluster::expect(inputs_arrived[stage],
                                                 inputs[0].A[0], inputs[0].A[1], inputs[0].B);
                        else
                            tma::cluster::expect(inputs_arrived[stage],
                                                 inputs[0].A[0], inputs[0].B,
                                                 inputs[0].A[0], inputs[0].B);
                        tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    } else {
                        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    }
                    update_phasebit<0>(phasebits, stage);
                    // mm_AB zeroes the accumulator, mma_AB accumulates onto it.
                    // The semaphore fires once the MMA has consumed the shared
                    // operands, releasing the stage back to the loader.
                    // Two MMAs off one B tile. Only the second carries the
                    // semaphore: tcgen05.commit covers every MMA issued before
                    // it, so one commit releases the stage once both have read.
                    if constexpr (config::CLUSTER_SIZE > 1) {
                        // commit<2> multicasts, so each CTA's inputs_finished
                        // sees both MMAs and its loader is released in step.
                        auto d = (warp_id == 2) ? d0 : d1;   // d1 == d0 when RBPT == 1
                        if (red_idx == 0)
                            warp::mm2_ABt (d, inputs[stage].A[warp_id - 2],
                                           inputs[stage].B, inputs_finished[stage]);
                        else
                            warp::mma2_ABt(d, inputs[stage].A[warp_id - 2],
                                           inputs[stage].B, inputs_finished[stage]);
                    } else if constexpr (globals::ROW_BLOCKS_PER_TASK == 1) {
                        if (red_idx == 0)
                            warp::mm_ABt (d0, inputs[stage].A[0], inputs[stage].B,
                                          inputs_finished[stage]);
                        else
                            warp::mma_ABt(d0, inputs[stage].A[0], inputs[stage].B,
                                          inputs_finished[stage]);
                    } else if (red_idx == 0) {
                        warp::mm_ABt (d0, inputs[stage].A[0], inputs[stage].B);
                        warp::mm_ABt (d1, inputs[stage].A[1], inputs[stage].B, inputs_finished[stage]);
                    } else {
                        warp::mma_ABt(d0, inputs[stage].A[0], inputs[stage].B);
                        warp::mma_ABt(d1, inputs[stage].A[1], inputs[stage].B, inputs_finished[stage]);
                    }
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
                kittens::tensor_commit<config::CLUSTER_SIZE>(mma_done);
            }
        }
        else if (warp_id == 1 && lane_id == 0) {
            // TMA store warp — same super-tile-swizzled task order as loader
            for (int pair_id = cluster_id; pair_id < total_cluster_tasks; pair_id += num_clusters) {
                // decode_comp_task varies rb fastest within a band, so flat and
                // flat+1 are adjacent row blocks at the same column. SUPER_M and
                // final_rows are both even, so flat=2k never straddles a band or
                // the super/tail boundary, and rb comes out even - which also
                // puts the pair inside one 256-row intra block, leaving the
                // per-K-strip barrier waits below untouched.
                const int shard_task_id =
                    globals::ROW_BLOCKS_PER_CLUSTER * pair_id
                  + globals::ROW_BLOCKS_PER_TASK * ctarank;
                const comp_task t = decode_comp_task(
                    shard_task_id, super_rows, final_rows, super_blocks, col_blocks);
                const int rb = t.rb;
                const int col_idx = t.col_idx;
                const int row_idx = rb;

                // One staging tile for four 64-row halves (two row blocks x two
                // halves), drained in sequence. outputs_free hands the tile back
                // between sub-rounds; outputs_finished closes the task.
                int sub = 0;
                #pragma unroll
                for (int h = 0; h < globals::ROW_BLOCKS_PER_TASK; h++) {
                    #pragma unroll
                    for (int hs = 0; hs < 2; hs++) {
                        const int buf = (h * 2 + hs) % globals::OUTPUT_BUFFERS;
                        wait(outputs_arrived[buf], get_phasebit<0>(phasebits, buf));
                        update_phasebit<0>(phasebits, buf);
                        tma::store_async(G.C, outputs.C[buf], {(row_idx + h) * 2 + hs, col_idx});
#ifdef AG_GEMM_EPILOGUE_READ_WAIT
                        // read_wait only guarantees the TMA has drained the
                        // staging tile, which is all the sub-round needs to
                        // reuse it; the global commit is left in flight and the
                        // kernel-end sync retires it. gemm_rs uses this form.
                        tma::store_async_read_wait();
#else
                        // store_async_wait waits for the global commit (not just
                        // smem reuse safety like read_wait). At large M,
                        // store-in-flight can race with downstream reads of C.
                        tma::store_async_wait();
#endif
                        // A buffer only needs an explicit release if the
                        // consumers come back to it inside this task, which is
                        // OUTPUT_BUFFERS sub-rounds later. Beyond that,
                        // outputs_finished covers it: this warp stores in order
                        // and read_waits each one, so by the time it closes the
                        // task every earlier buffer has drained.
                        const int sub_idx = h * 2 + hs;
                        ++sub;
                        if (sub == 2 * globals::ROW_BLOCKS_PER_TASK)
                            arrive(outputs_finished);
                        else if (sub_idx + globals::OUTPUT_BUFFERS
                                 < 2 * globals::ROW_BLOCKS_PER_TASK)
                            arrive(outputs_free[buf]);
                        }
                }
            }
        }
    } else {
        // Consumer warpgroups: WGMMA — same tile count, same CTA-stride
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int pair_id = cluster_id; pair_id < total_cluster_tasks; pair_id += num_clusters) {
            rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
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
                if constexpr (config::CLUSTER_SIZE > 1) {
                    // The accumulator lives on the leader, so release it there.
                    if (warp_id == 0 && lane_id == 0)
                        tma::cluster::arrive(tmem_free[h], 0, 1);
                } else {
                    warpgroup::arrive(tmem_free[h]);
                }

                // The single staging tile takes one 64-row half at a time.
                // Under the group<8> tmem layout warps {0,1,4,5} own rows < 64
                // and {2,3,6,7} own the rest, so each sub-round only four of
                // the eight warps write; the others just ride the barrier.
                #pragma unroll
                for (int hs = 0; hs < 2; hs++) {
                    const int sub = h * 2 + hs;
                    const int buf = sub % globals::OUTPUT_BUFFERS;
                    if (sub == 0) {
                        // First sub-round of the task waits on the previous
                        // one. The <1> half starts signalled, so task 0 passes.
                        // It also implies every buffer is drained, which is why
                        // sub-rounds 1..OUTPUT_BUFFERS-1 need no wait at all --
                        // that is the whole point of the extra buffer.
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        update_phasebit<1>(phasebits, 0);
                    } else if (sub >= globals::OUTPUT_BUFFERS) {
                        wait(outputs_free[buf], get_phasebit<0>(phasebits, 1 + buf));
                        update_phasebit<0>(phasebits, 1 + buf);
                    }
                    if (trow / 64 == hs) {
                        auto dst = outputs.C[buf].template subtile<globals::ROW_BLOCK / 8,
                                                                   globals::COL_BLOCK>(
                                       {(trow % 64) / (globals::ROW_BLOCK / 8), 0});
                        warp::store(dst, C_accum);
                    }
                    group<8>::sync(4);
                    if (warpgroup_id == 0 && warp_id == 0 && lane_id == 0)
                        arrive(outputs_arrived[buf]);
                }
            }
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
    // Cluster-wide: every CTA's semaphores must exist before a peer redirects a
    // transaction or an arrival at them. Both this and the allocator's
    // destructor are cluster-scope, so they sit outside the role dispatch --
    // a CTA that skipped them would strand its cluster partner on the barrier.
    if constexpr (config::CLUSTER_SIZE > 1) everyone::tma::cluster::sync();
    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    if (blockIdx.x < G.num_intra_comm) {
        intra_comm_sm(G);
    } else {
        fused_comp_sm(G, tm_alloc);
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
#ifdef AG_GEMM_ROWPOLL
    for (int i = offset; i < num_rows; i += stride) {
        G.barrier[G.dev_idx][{1, i, 0}] = 0;
    }
#endif
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

#if defined(AG_GEMM_CLUSTER2)
__global__ __launch_bounds__(config::NUM_THREADS, 1)
__cluster_dims__(config::CLUSTER_SIZE, 1, 1)
#else
__global__ __launch_bounds__(config::NUM_THREADS, 1)
#endif
void ag_gemm_fused_kernel_stub(const __grid_constant__ globals G) {
    fused_kernel(G);
}

// ============================================================================
// Prologue kernel
void launch_fused_ag_gemm(const globals& G, unsigned int active_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        ag_gemm_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    // No phase-0 prologue here: it exists to push inter-node RDMA work
    // requests into the host proxy's FIFO, and a single node has no peers to
    // post to. ag_gemm.cu keeps it, along with the side-stream overlap it
    // needs.
    // Clusters are launched whole.
    const unsigned int cluster_grid =
        active_sms - active_sms % (unsigned int)config::CLUSTER_SIZE;
    ag_gemm_fused_kernel_stub<<<cluster_grid, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}

}  // namespace ag_gemm_blackwell

#include "operators/ag_gemm/ag_gemm_blackwell_session.cuh"
