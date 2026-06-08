/**
 * @file dispatch_gemm_glu_combine.cu
 * @brief Multi-node fused MoE: dispatch -> gemm1 -> SwiGLU -> gemm2 -> combine,
 *        in a single persistent kernel.
 *
 * During the dispatch phase, CTA roles are split by blockIdx.x:
 *
 *   Inter-send CTAs [0, num_send_sms):
 *     Push this node's pre-dispatch token buffer to the peer node through the
 *     D2H FIFO/RDMA path. Work is chunk-striped across warps and CTAs.
 *
 *   Inter-copy CTAs [..., ... + num_copy_sms):
 *     Poll peer arrival flags and publish per-chunk copy_ready flags. In
 *     zero-copy mode the RDMA destination is already the peer token buffer, so
 *     this role mainly turns NIC completion into device-visible readiness.
 *
 *   Dispatch CTAs [..., ... + num_dispatch_sms):
 *     Walk local tokens first, then peer tokens. Each token is TMA-loaded from
 *     its source GPU/node into post_tokens, with peer tokens gated by copy_ready.
 *
 *   Compute CTAs [..., 132):
 *     Run the fused gemm1+SwiGLU (post_tokens @ w1 -> silu(gate)*up -> act) once
 *     their dispatched row blocks are ready. SwiGLU is applied in the gemm1
 *     epilogue, so `act` is written directly with no h1[M,2I] HBM round-trip.
 *
 * After a grid barrier the role split dissolves and every CTA runs the rest in
 * lockstep, separated by grid + cross-GPU barriers:
 *   gemm2 (act @ w2 -> y_expert) -> combine.
 *
 * Combine has no dedicated CTA role: each CTA stores its weighted expert outputs
 * into the owner GPU's contribution buffer over IPC, then each owner GPU
 * gather-reduces its tokens into y_out. The inter-node tail reuses the dispatch
 * split (combine_phase2): the send CTAs RDMA the staged rows to peers, the rest
 * reduce the received contributions.
 *
 * Infrastructure (config, globals, helpers, host setup, entrypoint) lives in
 *   include/operators/dispatch_gemm_glu_combine/dispatch_gemm_glu_combine.cuh
 * Python/session glue + pybind module live in
 *   include/operators/dispatch_gemm_glu_combine/session.cuh
 */
#include "operators/dispatch_gemm_glu_combine/dispatch_gemm_glu_combine.cuh"

namespace moe_dispatch_gemm_glu_combine_multinode {

__device__ inline void fused_inter_send_sm(const fused_globals &G) {
    const int warps_per_cta = NUM_MAIN_THREADS / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int send_id = blockIdx.x;

    if (lane_id == 0) {
        int total_pushers = G.num_send_sms * warps_per_cta;
        int my_pusher = send_id * warps_per_cta + warp_id;
        for (int chunk_id = my_pusher; chunk_id < G.total_chunks; chunk_id += total_pushers) {
            uint32_t off = (uint32_t)(chunk_id * CHUNK_BYTES);
            uint32_t bytes = min(CHUNK_BYTES, G.pre_tokens_bytes - (int)off);
            const int n_peers = G.num_nodes - 1;
            // Per-peer slot offsets (zero at N == 2). single_peer_bytes is
            // this rank's pre_tokens_bytes (each sender contributes the
            // same chunk count to each peer).
            const int single_peer_bytes = G.pre_tokens_bytes;
            const int single_peer_tiles = G.total_chunks;
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint32_t)(sap * single_peer_tiles + chunk_id);
                cmd.bytes = bytes;
                cmd.local_offset = off;
                cmd.remote_offset = (uint64_t)sap * (uint32_t)single_peer_bytes + off;
                cmd.lane_id = (uint16_t)chunk_id;
                cmd.reserved0 = (uint8_t)(peer_slot * fused_globals::NUM_DEVICES + G.dev_idx);
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(
                        G.d2h_fifos, (uint32_t)cmd.lane_id);
                fifo.push(cmd);
            }
        }
    }
}

__device__ inline void fused_inter_copy_sm(const fused_globals &G) {
    int copy_id = blockIdx.x - G.num_send_sms;
    // Multi-peer: arrival_flags/copy_ready/peer_tokens are laid out by
    // sender slot, where slot_at_peer(sender, this_node) identifies the slot
    // into which that sender's data lands on this node.
    const int n_peers = G.num_nodes - 1;
    const int single_peer_chunks = G.total_chunks;
    for (int chunk_id = copy_id; chunk_id < G.total_chunks; chunk_id += G.num_copy_sms) {
        if (threadIdx.x == 0) {
            for (int slot = 0; slot < n_peers; ++slot) {
                const int flag_idx = slot * single_peer_chunks + chunk_id;
                uint32_t v;
                do {
                    // Proxy path: acquire load from the per-peer arrival
                    // slot. Pairs with the proxy's release-sys store.
                    v = comm::atomic_u32::acquire_load_sys(&G.arrival_flags[flag_idx]);
                    if (v == G.epoch) break;
                    __nanosleep(100);
                } while (true);
                // Zero-copy mode: peer_tokens IS the registered RDMA
                // destination, so this slot's chunk is already in place once
                // its arrival flag is observed.
                comm::atomic_u32::release_store_sys(
                    &G.copy_ready[G.dev_idx][{flag_idx}], 1u);
            }
        }
        __syncthreads();
    }
}


__device__ inline void dispatch_fused(const fused_globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    fused_globals::token_vec (&token)[fused_globals::TOKENS_PER_BLOCK] =
        al.allocate<fused_globals::token_vec, fused_globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[fused_globals::TOKENS_PER_BLOCK];


    const int lane_id = threadIdx.x;
    // Track which lanes had a valid token slot (regardless of src). Used after
    // the per-token TMA work to do ONE warp-level barrier atomic with the
    // popcount as the increment, replacing 16 per-lane atomics to the same
    // line. 16x fewer atomics + zero intra-warp serialization on the L2
    // atomic engine.
    bool valid_token_slot = false;
    if (lane_id < fused_globals::TOKENS_PER_BLOCK) {
        const int token_idx = sm_idx * fused_globals::TOKENS_PER_BLOCK + lane_id;
        if (token_idx < G.num_padded_local_tokens) {
            valid_token_slot = true;
            int src_node = G.pull_dispatch_indices[{token_idx, 0}];
            int src_dev_idx = G.pull_dispatch_indices[{token_idx, 1}];
            int src_token_idx = G.pull_dispatch_indices[{token_idx, 2}];

            if (src_node >= 0 && src_dev_idx >= 0 && src_token_idx >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                if (src_node == G.node_idx) {
                    ::dist::tma::expect_bytes(token_arrived[lane_id], sizeof(fused_globals::token_vec));
                    ::dist::tma::load_async(token[lane_id], G.pre_tokens[src_dev_idx],
                                    {src_token_idx, 0}, token_arrived[lane_id]);
                } else {
                    ::dist::tma::expect_bytes(token_arrived[lane_id], sizeof(fused_globals::token_vec));
                    // Dispatch/compute run concurrently with RDMA. Before
                    // TMA-loading peer_tokens[src_dev_idx] (which lives on
                    // local dev src_dev_idx via IPC), spin on the chunk_ready
                    // flags covering this token's byte range. Each peer dev's
                    // copy CTA sets copy_ready[dev][chunk]=1 after writing the
                    // chunk into its peer_tokens_local; we read across IPC.
                    const int sender_slot =
                        internode::slot_at_peer(src_node, G.node_idx, G.num_nodes);
                    const int byte_off = src_token_idx * fused_globals::H * 2;
                    const int byte_end = byte_off + fused_globals::H * 2 - 1;
                    const int first_chunk = byte_off / CHUNK_BYTES;
                    const int last_chunk = byte_end / CHUNK_BYTES;
                    for (int c = first_chunk; c <= last_chunk; c++) {
                        const int ready_idx = sender_slot * G.total_chunks + c;
                        int v;
                        do {
                            v = comm::atomic_u32::acquire_load_s32_sys(
                                &G.copy_ready[src_dev_idx][{ready_idx}]);
                            // Throttle: chunks arrive on a 50us+ timescale
                            // (RDMA bandwidth), so an unthrottled spin only
                            // generates IPC/PCIe traffic without reducing
                            // observed latency. 100ns sleep cuts poll rate
                            // ~30x. Mirrors the arrival_flags spin pattern
                            // in fused_inter_copy_sm.
                            if (v == 1) break;
                            __nanosleep(100);
                        } while (true);
                    }
                    const int peer_token_idx =
                        sender_slot * G.num_local_tokens + src_token_idx;
                    ::dist::tma::load_async(token[lane_id], G.peer_tokens[src_dev_idx],
                                    {peer_token_idx, 0}, token_arrived[lane_id]);
                }
                wait(token_arrived[lane_id], 0);
                ::dist::tma::store_async(G.post_tokens, token[lane_id], {token_idx, 0});
                ::dist::tma::store_async_wait();
            }
        }
    }

    // Warp-collapsed barrier increment. All 16 valid lanes targeted the same
    // barrier line (token_idx/ROW_BLOCK is identical for all 16 tokens in a
    // slice since TOKENS_PER_BLOCK=16 < ROW_BLOCK=128). Replace 16 per-lane
    // atomics-to-same-line (which serialize through L2's atomic engine) with
    // ONE atomic in lane 0, increment = popcount of valid lanes.
    //
    // Why this is safe: every valid lane previously did red.add(1) regardless
    // of src_node sign — the barrier counts "tokens accounted for", not
    // "tokens with sources". popcount preserves that semantic.
    //
    // All 384 threads of the CTA reach __ballot_sync (it is per-warp). For
    // warps 1..11 (lane_id >= 32), valid_token_slot is false → mask=0 → skip.
    unsigned int valid_mask =
        __ballot_sync(0xFFFFFFFFu, valid_token_slot);
    if (lane_id == 0 && valid_mask != 0u) {
        constexpr int SLICES_PER_RB_LOCAL =
            fused_globals::ROW_BLOCK / fused_globals::TOKENS_PER_BLOCK;
        const int row_block = sm_idx / SLICES_PER_RB_LOCAL;
        const int count = __popc(valid_mask);
        comm::atomic_u32::release_add_gpu(&G.barrier[G.dev_idx][{row_block}], count);
    }
}

// Row-major decode of `task_id ∈ [0, row_blocks*col_blocks)` into (row, col).
__device__ inline void dispatch_swizzle_decode(int task_id, int row_blocks, int col_blocks,
                                          int& row_in_grid, int& col_idx) {
    row_in_grid = task_id / col_blocks;
    col_idx = task_id % col_blocks;
}

// Super-M (L2 weight-reuse) decode of an EXPERT-LOCAL task_id ∈ [0, row_blocks*col_blocks).
// Rasterizes SUPER_M consecutive ROW-blocks at a fixed col before advancing col, so
// concurrent CTAs reuse the same weight col-tile in L2. Tail (row_blocks % super_m) rows
// fall back to row-major so coverage stays exact. super_m clamps to >=1.
// super_m==1 is provably byte-identical to dispatch_swizzle_decode (row-major).
__device__ inline void dispatch_super_m_decode(int task_id, int row_blocks, int col_blocks,
                                               int super_m, int& row_in_grid, int& col_idx) {
    if (super_m < 1) super_m = 1;
    const int super_rows   = (row_blocks / super_m) * super_m;   // rows covered by full bands
    const int super_blocks = super_m * col_blocks;               // tiles per full band
    if (task_id < super_rows * col_blocks) {
        const int band     = task_id / super_blocks;
        const int in_band  = task_id - band * super_blocks;
        col_idx     = in_band / super_m;
        row_in_grid = band * super_m + (in_band % super_m);
        return;
    }
    // Tail rows (row_blocks not a multiple of super_m): plain row-major within the tail.
    const int tail_rows = row_blocks - super_rows;              // in [0, super_m)
    const int tail_idx  = task_id - super_rows * col_blocks;
    col_idx     = tail_idx / tail_rows;
    row_in_grid = super_rows + (tail_idx % tail_rows);
}

// ============================================================================
// Tensor-core grouped GEMM (warpgroup MMA), parameterized to run as gemm1 then
// gemm2. Body is identical to dispatch_gemm.cu's group_gemm_fused; only the
// reduction depth (NUM_ITERS = K/RED_BLOCK), output width (COL_BLOCKS =
// N/COL_BLOCK), the A/B/C tensors, and whether to wait on the dispatch barrier
// (gemm1 only) are parameters.
template <int NUM_ITERS, int COL_BLOCKS, bool WAIT_BARRIER,
          typename AT, typename BT, typename CT>
__device__ inline void grouped_gemm(const fused_globals &G, const int sm_idx,
                                    const int num_sms, const AT &A_t,
                                    const BT &B_t, CT &C_t) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    fused_globals::pipeline_inputs (&inputs)[fused_globals::PIPELINE_STAGES] =
        allocator.allocate<fused_globals::pipeline_inputs, fused_globals::PIPELINE_STAGES>();
    fused_globals::pipeline_outputs &outputs =
        *reinterpret_cast<fused_globals::pipeline_outputs *>(&inputs[fused_globals::PIPELINE_STAGES - 1]);

    const int global_gpu_idx = G.node_idx * fused_globals::NUM_DEVICES + G.dev_idx;
    const int expert_offset = global_gpu_idx * fused_globals::NUM_EXPERTS_PER_DEV;
    __shared__ int padded_tokens_per_expert[fused_globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < fused_globals::NUM_EXPERTS_PER_DEV)
        padded_tokens_per_expert[threadIdx.x] =
            G.padded_tokens_per_expert[{expert_offset + (int)threadIdx.x}];
    __shared__ int local_rb_per_expert[fused_globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < fused_globals::NUM_EXPERTS_PER_DEV)
        local_rb_per_expert[threadIdx.x] = G.local_rb_per_expert[{(int)threadIdx.x}];

    __shared__ semaphore inputs_arrived[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    constexpr int num_iters = NUM_ITERS;
    constexpr int col_blocks = COL_BLOCKS;
    constexpr int NUM_PASSES = 2;

    if (wg_id == 2) {
        warpgroup::decrease_registers<40>();
        if (w_id == 0) {
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                #pragma unroll
                for (int expert_id = 0;
                     expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                    const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                    cum += padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _row_in_grid, _col_idx;
                        dispatch_super_m_decode(task_id, row_blocks, col_blocks, G.super_m, _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;
                        if (WAIT_BARRIER && l_id == 0) {
                            int bar_val = comm::atomic_u32::acquire_load_s32_gpu(
                                &G.barrier[G.dev_idx][{row_idx}]);
                            while (bar_val != fused_globals::ROW_BLOCK) {
                                __nanosleep(64);
                                bar_val = comm::atomic_u32::acquire_load_s32_gpu(
                                    &G.barrier[G.dev_idx][{row_idx}]);
                            }
                        }
                        __syncwarp();
                        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                            if (l_id == 0) {
                                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                                update_phasebit<1>(phasebits, stage);
                                if (red_idx == fused_globals::PIPELINE_STAGES - 1) {
                                    wait(outputs_finished, get_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES));
                                    update_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES);
                                }
                            }
                            __syncwarp();
                            if (l_id == 0) {
                                ::dist::tma::expect_bytes(inputs_arrived[stage], sizeof(fused_globals::pipeline_inputs));
                                #pragma unroll
                                for (int i = 0; i < 2; i++)
                                    ::dist::tma::load_async(inputs[stage].A[i], A_t,
                                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                                ::dist::tma::load_async(inputs[stage].B, B_t,
                                                {expert_id, red_idx, col_idx}, inputs_arrived[stage]);
                            }
                            __syncwarp();
                            stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                        }
                    }
                    task_id -= num_blocks;
                }
            }
        } else if (w_id == 1 && l_id == 0) {
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                #pragma unroll
                for (int expert_id = 0;
                     expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                    const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                    cum += padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _row_in_grid, _col_idx;
                        dispatch_super_m_decode(task_id, row_blocks, col_blocks, G.super_m, _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;
                        wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                        update_phasebit<0>(phasebits, 0);
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::store_async(C_t, outputs.C[i], {row_idx * 2 + i, col_idx});
                        ::dist::tma::store_async_read_wait();
                        arrive(outputs_finished);
                    }
                    task_id -= num_blocks;
                }
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        #pragma unroll 1
        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int task_id = sm_idx;
            int cum = 0;
            #pragma unroll
            for (int expert_id = 0;
                 expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                cum += padded_tokens_per_expert[expert_id];
                const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                const int total_rb = rb_end_e - rb_start_e;
                const int local_rb_e = local_rb_per_expert[expert_id];
                const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                const int num_blocks = row_blocks * col_blocks;
                for (; task_id < num_blocks; task_id += num_sms) {
                    rt_fl<fused_globals::ROW_BLOCK / 8, fused_globals::COL_BLOCK> C_accum;
                    warp::zero(C_accum);
                    // Depth-1 software-pipelined WGMMA: keep one committed group
                    // in flight so the next mma issue overlaps the prior's drain.
                    int prev_stage = -1;
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        warpgroup::mma_AB(C_accum, inputs[stage].A[wg_id], inputs[stage].B);
                        if (prev_stage >= 0) {
                            warpgroup::mma_async_wait<1>();
                            warp::arrive(inputs_finished[prev_stage]);
                        }
                        prev_stage = stage;
                        stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                    }
                    warpgroup::mma_async_wait<0>();
                    warp::arrive(inputs_finished[prev_stage]);
                    group<8>::sync(3);
                    warpgroup::store(outputs.C[wg_id], C_accum);
                    warpgroup::sync(wg_id + 1);
                    warpgroup::arrive(outputs_arrived);
                }
                task_id -= num_blocks;
            }
        }
    }
}

// ============================================================================
// gemm1 + SwiGLU FUSED. Same warpgroup-MMA pipeline as grouped_gemm, but each
// task computes BOTH the gate col tile (col c) and the up col tile (col c+CB1)
// for the same rows, fuses silu(gate)*up in registers, and TMA-stores the
// resulting `act` tile directly — eliminating the h1[M,2I] HBM write+read, the
// standalone swiglu_phase, and bar(2). gemm2 is unchanged (reads `act`).
//
// COL_BLOCK_1 = 128 keeps two fp32 accumulators (gate+up) within the register
// budget; col_blocks = I/COL_BLOCK_1 = 16. B is loaded twice per K step (gate
// col_idx=c, up col_idx=c+col_blocks). Waits the dispatch barrier (gemm1 input).
__device__ inline void gemm1_swiglu_fused(const fused_globals &G, const int sm_idx,
                                          const int num_sms) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    fused_globals::pipeline_inputs1 (&inputs)[fused_globals::PIPELINE_STAGES] =
        allocator.allocate<fused_globals::pipeline_inputs1, fused_globals::PIPELINE_STAGES>();
    fused_globals::pipeline_outputs1 &outputs =
        *reinterpret_cast<fused_globals::pipeline_outputs1 *>(&inputs[fused_globals::PIPELINE_STAGES - 1]);

    const int global_gpu_idx = G.node_idx * fused_globals::NUM_DEVICES + G.dev_idx;
    const int expert_offset = global_gpu_idx * fused_globals::NUM_EXPERTS_PER_DEV;
    __shared__ int padded_tokens_per_expert[fused_globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < fused_globals::NUM_EXPERTS_PER_DEV)
        padded_tokens_per_expert[threadIdx.x] =
            G.padded_tokens_per_expert[{expert_offset + (int)threadIdx.x}];
    __shared__ int local_rb_per_expert[fused_globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < fused_globals::NUM_EXPERTS_PER_DEV)
        local_rb_per_expert[threadIdx.x] = G.local_rb_per_expert[{(int)threadIdx.x}];

    __shared__ semaphore inputs_arrived[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    constexpr int num_iters  = fused_globals::H / fused_globals::RED_BLOCK;
    constexpr int col_blocks = fused_globals::I / fused_globals::COL_BLOCK_1;  // 16
    constexpr int NUM_PASSES = 2;

    if (wg_id == 2) {
        warpgroup::decrease_registers<40>();
        if (w_id == 0) {
            // Producer: load A[2] + gate-B + up-B per K step.
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                #pragma unroll
                for (int expert_id = 0;
                     expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                    const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                    cum += padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _row_in_grid, _col_idx;
                        dispatch_super_m_decode(task_id, row_blocks, col_blocks, G.super_m, _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;             // gate col (0..15)
                        if (l_id == 0) {
                            int bar_val = comm::atomic_u32::acquire_load_s32_gpu(
                                &G.barrier[G.dev_idx][{row_idx}]);
                            while (bar_val != fused_globals::ROW_BLOCK) {
                                __nanosleep(64);
                                bar_val = comm::atomic_u32::acquire_load_s32_gpu(
                                    &G.barrier[G.dev_idx][{row_idx}]);
                            }
                        }
                        __syncwarp();
                        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                            if (l_id == 0) {
                                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                                update_phasebit<1>(phasebits, stage);
                                if (red_idx == fused_globals::PIPELINE_STAGES - 1) {
                                    wait(outputs_finished, get_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES));
                                    update_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES);
                                }
                            }
                            __syncwarp();
                            if (l_id == 0) {
                                ::dist::tma::expect_bytes(inputs_arrived[stage], sizeof(fused_globals::pipeline_inputs1));
                                #pragma unroll
                                for (int i = 0; i < 2; i++)
                                    ::dist::tma::load_async(inputs[stage].A[i], G.post_tokens,
                                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                                ::dist::tma::load_async(inputs[stage].Bg, G.w1,
                                                {expert_id, red_idx, col_idx}, inputs_arrived[stage]);
                                ::dist::tma::load_async(inputs[stage].Bu, G.w1,
                                                {expert_id, red_idx, col_idx + col_blocks}, inputs_arrived[stage]);
                            }
                            __syncwarp();
                            stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                        }
                    }
                    task_id -= num_blocks;
                }
            }
        } else if (w_id == 1 && l_id == 0) {
            // Store warp: TMA-store the fused act tiles.
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                #pragma unroll
                for (int expert_id = 0;
                     expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                    const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                    cum += padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _row_in_grid, _col_idx;
                        dispatch_super_m_decode(task_id, row_blocks, col_blocks, G.super_m, _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;
                        wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                        update_phasebit<0>(phasebits, 0);
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::store_async(G.act, outputs.C[i], {row_idx * 2 + i, col_idx});
                        ::dist::tma::store_async_read_wait();
                        arrive(outputs_finished);
                    }
                    task_id -= num_blocks;
                }
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        #pragma unroll 1
        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int task_id = sm_idx;
            int cum = 0;
            #pragma unroll
            for (int expert_id = 0;
                 expert_id < fused_globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                const int rb_start_e = cum / fused_globals::ROW_BLOCK;
                cum += padded_tokens_per_expert[expert_id];
                const int rb_end_e = (cum + fused_globals::ROW_BLOCK - 1) / fused_globals::ROW_BLOCK;
                const int total_rb = rb_end_e - rb_start_e;
                const int local_rb_e = local_rb_per_expert[expert_id];
                const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                const int num_blocks = row_blocks * col_blocks;
                for (; task_id < num_blocks; task_id += num_sms) {
                    rt_fl<fused_globals::ROW_BLOCK / 8, fused_globals::COL_BLOCK_1> acc_g;
                    rt_fl<fused_globals::ROW_BLOCK / 8, fused_globals::COL_BLOCK_1> acc_u;
                    warp::zero(acc_g);
                    warp::zero(acc_u);
                    int prev_stage = -1;
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        warpgroup::mma_AB(acc_g, inputs[stage].A[wg_id], inputs[stage].Bg);
                        warpgroup::mma_AB(acc_u, inputs[stage].A[wg_id], inputs[stage].Bu);
                        if (prev_stage >= 0) {
                            warpgroup::mma_async_wait<1>();
                            warp::arrive(inputs_finished[prev_stage]);
                        }
                        prev_stage = stage;
                        stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                    }
                    warpgroup::mma_async_wait<0>();
                    warp::arrive(inputs_finished[prev_stage]);
                    // SwiGLU fuse in registers: acc_g <- silu(gate) * up. acc_g and
                    // acc_u share the identical rt layout, so element-wise indices
                    // correspond. data[k] is a float2 (packed fp32 pair).
                    #pragma unroll
                    for (int i = 0; i < decltype(acc_g)::height; i++) {
                        #pragma unroll
                        for (int j = 0; j < decltype(acc_g)::width; j++) {
                            #pragma unroll
                            for (int k = 0; k < acc_g.packed_per_tile; k++) {
                                float gx = acc_g.tiles[i][j].data[k].x;
                                float gy = acc_g.tiles[i][j].data[k].y;
                                float ux = acc_u.tiles[i][j].data[k].x;
                                float uy = acc_u.tiles[i][j].data[k].y;
                                acc_g.tiles[i][j].data[k].x = (gx / (1.0f + __expf(-gx))) * ux;
                                acc_g.tiles[i][j].data[k].y = (gy / (1.0f + __expf(-gy))) * uy;
                            }
                        }
                    }
                    group<8>::sync(3);
                    warpgroup::store(outputs.C[wg_id], acc_g);
                    warpgroup::sync(wg_id + 1);
                    warpgroup::arrive(outputs_arrived);
                }
                task_id -= num_blocks;
            }
        }
    }
}

// gemm2 (act @ w2 -> y_expert, no dispatch-barrier wait). Compile-time K/N from
// the operator dims. gemm1 is handled by the fused gemm1_swiglu_fused above.
__device__ inline void ffn_gemm2(const fused_globals &G, int sm_idx, int num_sms) {
    grouped_gemm<fused_globals::I / fused_globals::RED_BLOCK,
                 fused_globals::H / fused_globals::COL_BLOCK, false>(
        G, sm_idx, num_sms, G.act, G.w2, G.y_expert);
}

// Intra-node combine: scatter-add weight*y_expert into the token-owner GPU's
// y_out via IPC, for rows whose dispatch source is on THIS node.
__device__ inline void combine_intra(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int sn = G.pull_dispatch_indices[{r, 0}];
        const int sd = G.pull_dispatch_indices[{r, 1}];
        const int st = G.pull_dispatch_indices[{r, 2}];
        if (sn != G.node_idx || sd < 0 || st < 0) continue;  // local-source only
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        for (int h = threadIdx.x; h < H; h += blockDim.x)
            atomicAdd(&G.y_out[sd][{st, h}], w * __bfloat162float(y[h]));
    }
}
// fused combine scatter: ONE row walk that branches on sn==node_idx.
// sn == node_idx  -> INTRA path: scalar fp32 atomic into y_out (verbatim from
// combine_intra). Local-source rows.
// sn != node_idx  -> STAGE path: bf162 atomic into pre_tokens UPPER (verbatim
// from combine_stage), with the row_expert guard.
// Replaces the standalone combine_intra + combine_stage passes (one grid sweep
// instead of two; both NVLink atomic streams interleave on the same SM).
__device__ inline void combine_scatter_fused(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    const int base = G.num_local_tokens;   // pre_tokens UPPER half (stage path)
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int sn = G.pull_dispatch_indices[{r, 0}];
        const int sd = G.pull_dispatch_indices[{r, 1}];
        const int st = G.pull_dispatch_indices[{r, 2}];
        if (sd < 0 || st < 0) continue;                 // padding
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        if (sn == G.node_idx) {
            // INTRA path. when use_gather, intra is handled by the
            // atomic-free producer slot-write + owner gather; skip here.
            if (G.use_gather) continue;
            // (verbatim from combine_intra :461-462).
            for (int h = threadIdx.x; h < H; h += blockDim.x)
                atomicAdd(&G.y_out[sd][{st, h}], w * __bfloat162float(y[h]));
        } else {
            // STAGE path (verbatim from combine_stage :649-664).
            // when use_inter_gather, inter rows are handled by the
            // atomic-free producer slot-write + owner gather; skip here.
            if (G.use_inter_gather) continue;
            const int e = G.row_expert_ptr[r];
            if (e < 0 || e >= fused_globals::NUM_EXPERTS_PER_DEV) continue;
            for (int h2 = threadIdx.x; h2 < H / 2; h2 += blockDim.x) {
                const int h = 2 * h2;
                __nv_bfloat162 v = __floats2bfloat162_rn(w * __bfloat162float(y[h]),
                                                         w * __bfloat162float(y[h + 1]));
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(
                              &G.pre_tokens_flat[sd][{base + st, h}]), v);
            }
        }
    }
}

// atomic-free GATHER combine (INTRA). Two phases:
// producer: each intra dispatched row r stores w*y_expert[r] (pre-weighted),
// non-atomically, into the OWNER GPU's comb_buf slot row_to_slot[r].
// One writer per slot => no atomics, fully coalesced bf162 store.
// owner   : one warp-group strided over output tokens; each token gather-reduces
// its [owner_offset[t], owner_offset[t+1]) comb_buf slots (already
// weighted) in fp32 registers and does ONE coalesced store to y_out.
__device__ inline void combine_producer_intra(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int own = G.row_to_owner_ptr[r];          // owner local dev, -1 = padding/inter
        if (own < 0) continue;
        const int slot = G.row_to_slot_ptr[r];
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        // 128-bit-friendly bf162 store into owner GPU's comb_buf[slot] over IPC.
        __nv_bfloat16* dst = &G.comb_buf[own][{slot, 0}];
        for (int h2 = threadIdx.x; h2 < H / 2; h2 += blockDim.x) {
            const int h = 2 * h2;
            reinterpret_cast<__nv_bfloat162*>(dst)[h2] =
                __floats2bfloat162_rn(w * __bfloat162float(y[h]),
                                      w * __bfloat162float(y[h + 1]));
        }
    }
}

__device__ inline void combine_gather_owner(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    constexpr int VEC = 8;                 // 8 bf16 (one float4) per group
    const int HV = H / VEC;                // groups per row
    const __nv_bfloat16* cb = &G.comb_buf[G.dev_idx][{0, 0}];   // own comb_buf
    float* y_base = &G.y_out[G.dev_idx][{0, 0}];
    const long total = (long)G.num_local_tokens * HV;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const int t = idx / HV, j = idx % HV;     // j-th 8-element group in token t
        const int base = G.owner_offset_ptr[t];
        const int kt = G.owner_offset_ptr[t + 1] - base;
        float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 a1 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < kt; ++k) {
            const __nv_bfloat16* row = cb + (size_t)(base + k) * H + j * VEC;
            float4 pv = reinterpret_cast<const float4*>(row)[0];
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&pv);
            a0.x += __low2float(p2[0]); a0.y += __high2float(p2[0]);
            a0.z += __low2float(p2[1]); a0.w += __high2float(p2[1]);
            a1.x += __low2float(p2[2]); a1.y += __high2float(p2[2]);
            a1.z += __low2float(p2[3]); a1.w += __high2float(p2[3]);
        }
        float* yp = y_base + (size_t)t * H + j * VEC;
        reinterpret_cast<float4*>(yp)[0] = a0;
        reinterpret_cast<float4*>(yp)[1] = a1;
    }
}

// atomic-free GATHER combine (INTER). Sender-side pre-sum, mirrors the
// intra producer/gather but on the rows the intra path SKIPS (sn != node_idx):
// producer: each inter dispatched row r stores w*y_expert[r] (pre-weighted),
// non-atomically, into the OWNER-STAGING-dev's inter_comb_buf slot
// inter_row_to_slot[r] over IPC (the staging dev = sd, same dev the
// atomic scatter targets; combine_send ships it to remote GPU sd).
// owner   : this gpu gather-reduces its inter_comb_buf CSR range in fp32 and
// writes the dense pre_tokens UPPER send row (overwrite). The wire
// then carries the byte-identical 1x dense [L,H] (send/reduce unchanged).
__device__ inline void combine_producer_inter(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int own = G.inter_row_to_owner_ptr[r];    // owner-staging dev, -1 = intra/pad
        if (own < 0) continue;
        const int slot = G.inter_row_to_slot_ptr[r];
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        __nv_bfloat16* dst = &G.inter_comb_buf[own][{slot, 0}];
        for (int h2 = threadIdx.x; h2 < H / 2; h2 += blockDim.x) {
            const int h = 2 * h2;
            reinterpret_cast<__nv_bfloat162*>(dst)[h2] =
                __floats2bfloat162_rn(w * __bfloat162float(y[h]),
                                      w * __bfloat162float(y[h + 1]));
        }
    }
}

// single-pass fuse of combine_producer_intra + combine_producer_inter.
// Each padded row r is intra XOR inter (disjoint by construction: row_to_owner>=0
// iff sn==node_idx; inter_row_to_owner>=0 iff sn!=node_idx — the index build
// guarantees this). One grid-stride sweep handles both planes by
// branching on the (mutually exclusive) owner index, removing the second pass's
// loop-control + index loads. Store body is byte-identical to the two originals.
__device__ inline void combine_producer_fused(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    const bool do_inter = G.use_inter_gather;       // hoist out of the hot loop
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int own_i = G.row_to_owner_ptr[r];                    // intra owner local dev, else -1
        const int own_x = do_inter ? G.inter_row_to_owner_ptr[r] : -1; // inter staging dev, else -1
        if (own_i < 0 && own_x < 0) continue;       // padding / other plane
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        // exactly one of (own_i>=0, own_x>=0) is true (disjoint by construction).
        __nv_bfloat16* dst = (own_i >= 0)
            ? &G.comb_buf[own_i][{G.row_to_slot_ptr[r], 0}]
            : &G.inter_comb_buf[own_x][{G.inter_row_to_slot_ptr[r], 0}];
        for (int h2 = threadIdx.x; h2 < H / 2; h2 += blockDim.x) {
            const int h = 2 * h2;
            reinterpret_cast<__nv_bfloat162*>(dst)[h2] =
                __floats2bfloat162_rn(w * __bfloat162float(y[h]),
                                      w * __bfloat162float(y[h + 1]));
        }
    }
}

__device__ inline void combine_gather_inter(const fused_globals &G) {
    constexpr int H = fused_globals::H;
    constexpr int VEC = 8;                 // 8 bf16 (one float4) per group
    const int HV = H / VEC;                // groups per row
    const __nv_bfloat16* cb = &G.inter_comb_buf[G.dev_idx][{0, 0}];   // own inter plane
    const int L = G.num_local_tokens;
    const int n_peers = G.num_nodes - 1;
    // pre_tokens UPPER holds one dense [L,H] send region per destination peer,
    // laid out after the L-row dispatch region: peer p -> rows [L + p*L, L + (p+1)*L).
    // inter_owner_offset_ptr is a flat [n_peers*(L+1)] CSR; peer p uses table p.
    __nv_bfloat16* ubase = &G.pre_tokens_flat[G.dev_idx][{L, 0}];
    const long total = (long)n_peers * L * HV;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const long srow = idx / HV;               // dense send row across peers
        const int j = (int)(idx % HV);            // j-th 8-element group
        const int p = (int)(srow / L);            // destination peer slot
        const int t = (int)(srow % L);            // owner token within the peer
        const int* off_tbl = G.inter_owner_offset_ptr + (long)p * (L + 1);
        const int off = off_tbl[t];
        const int kt = off_tbl[t + 1] - off;
        float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f);   // start at 0 (overwrite)
        float4 a1 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < kt; ++k) {
            const __nv_bfloat16* cbrow = cb + (size_t)(off + k) * H + j * VEC;
            float4 pv = reinterpret_cast<const float4*>(cbrow)[0];
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&pv);
            a0.x += __low2float(p2[0]); a0.y += __high2float(p2[0]);
            a0.z += __low2float(p2[1]); a0.w += __high2float(p2[1]);
            a1.x += __low2float(p2[2]); a1.y += __high2float(p2[2]);
            a1.z += __low2float(p2[3]); a1.w += __high2float(p2[3]);
        }
        // Pack fp32 a0/a1 -> 8 bf16 and store one coalesced bf162x4 into send row.
        __nv_bfloat16* sp = ubase + (size_t)srow * H + j * VEC;
        __nv_bfloat162* sp2 = reinterpret_cast<__nv_bfloat162*>(sp);
        sp2[0] = __floats2bfloat162_rn(a0.x, a0.y);
        sp2[1] = __floats2bfloat162_rn(a0.z, a0.w);
        sp2[2] = __floats2bfloat162_rn(a1.x, a1.y);
        sp2[3] = __floats2bfloat162_rn(a1.z, a1.w);
    }
}

// Two-pass dispatch walker, parameterized by (sm_idx, stride).
// Stride controls how widely workers spread across the per-expert task lists.
// Real dispatch CTAs use sm_idx ∈ [0, num_dispatch_sms) with stride =
// num_dispatch_sms.
__device__ inline void dispatch_two_pass_walk(const fused_globals &G, int sm_idx, int stride) {
    constexpr int SLICES_PER_RB =
        fused_globals::ROW_BLOCK / fused_globals::TOKENS_PER_BLOCK;
    const int global_gpu_idx_d =
        G.node_idx * fused_globals::NUM_DEVICES + G.dev_idx;
    const int expert_offset_d =
        global_gpu_idx_d * fused_globals::NUM_EXPERTS_PER_DEV;
    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        int task_id = sm_idx;
        int cum_d = 0;
        #pragma unroll
        for (int e = 0; e < fused_globals::NUM_EXPERTS_PER_DEV; ++e) {
            const int rb_start_e = cum_d / fused_globals::ROW_BLOCK;
            cum_d += G.padded_tokens_per_expert[{expert_offset_d + e}];
            const int rb_end_e =
                (cum_d + fused_globals::ROW_BLOCK - 1)
                / fused_globals::ROW_BLOCK;
            const int total_rb = rb_end_e - rb_start_e;
            const int local_rb_e = G.local_rb_per_expert[{e}];
            const int rb_offset_e =
                (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
            const int row_blocks_e =
                (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
            const int num_slices_e = row_blocks_e * SLICES_PER_RB;
            for (; task_id < num_slices_e; task_id += stride) {
                const int rb_in_pass = task_id / SLICES_PER_RB;
                const int slice = task_id % SLICES_PER_RB;
                const int row_block = rb_offset_e + rb_in_pass;
                const int sm_idx_real = row_block * SLICES_PER_RB + slice;
                dispatch_fused(G, sm_idx_real);
            }
            task_id -= num_slices_e;
        }
    }
}

__device__ inline void combine_phase2(const fused_globals &G);         // defined below
__device__ inline void combine_zero_staging(const fused_globals &G);   // defined below
__device__ inline void combine_global_barrier(const fused_globals &G, int bi);  // below
__device__ inline void combine_grid_barrier(const fused_globals &G, int bi);    // below

__global__ __launch_bounds__(NUM_MAIN_THREADS, 1)
void fused_kernel(const __grid_constant__ fused_globals G) {
    const int block = (int)blockIdx.x;
    const int copy_phase_blocks = G.num_send_sms + G.num_copy_sms;

    // Dispatch/compute CTAs proceed immediately. Each peer token is gated
    // per-chunk by copy_ready inside dispatch_fused(). Local tokens
    // (src_node == G.node_idx) need no gate — pre_tokens is always available.

    if (block < G.num_send_sms) {
        // Pass G by const reference. Taking a local mutable copy here causes
        // ptxas to materialize the 3KB+ fused_globals struct in per-thread
        // local memory across all 384 threads (~1MB/CTA) and spills into
        // STACK:7024 even though the dispatch/gemm branches never touch the
        // send fields. Const reference keeps the struct in constant memory
        // (since G is __grid_constant__).
        fused_inter_send_sm(G);
    } else if (block < copy_phase_blocks) {
        fused_inter_copy_sm(G);
    } else if (block < copy_phase_blocks + G.num_dispatch_sms) {
        int dispatch_id = block - copy_phase_blocks;
        // Dispatch walks LOCAL row_blocks first (no copy_ready wait, no
        // chunk-arrival dependency), then PEER row_blocks (gated on
        // copy_ready inside dispatch_fused).
        const int dispatch_stride = G.num_dispatch_sms;
        dispatch_two_pass_walk(G, dispatch_id, dispatch_stride);
    } else {
        int comp_idx = block - copy_phase_blocks - G.num_dispatch_sms;
        // Fused gemm1+SwiGLU: post @ w1 -> silu(gate)*up -> act (waits dispatch
        // barrier). Writes `act` directly, skipping the h1[M,2I] HBM round-trip,
        // the standalone swiglu_phase, and bar(2).
        gemm1_swiglu_fused(G, comp_idx, G.num_comp_sms);
    }

    // FFN tail (all CTAs, separated by grid + cross-GPU barriers):
    // gemm1+SwiGLU done (act written) -> gemm2 -> fused intra+stage combine.
    // bar(2) (h1-ready) and swiglu_phase are ELIMINATED by the fusion: act is the
    // gemm1 output, read only by this GPU's gemm2 (local producer->consumer). The
    // grid-local bar(3) below already orders gemm1's act store before gemm2.
    // hoist the pre_tokens-UPPER zero before gemm2 (gemm2 never touches
    // pre upper). When the zero-fill is needed (!use_inter_gather), bar(3) must
    // be cross-GPU + __threadfence_system so every GPU's pre-upper is peer-visible
    // (Invariant Z) before the stage scatter. In the default path
    // (use_inter_gather=1) combine_gather_inter OVERWRITES every send row, so the
    // zero-fill is skipped and bar(3) only needs to guarantee THIS GPU's act
    // (gemm2 input, local) is ready -> a grid-LOCAL barrier suffices.
    if (G.num_nodes > 1 && !G.use_inter_gather) {
        combine_zero_staging(G);
        combine_global_barrier(G, 3);             // act ready + pre-upper zeroed cross-GPU
    } else {
        combine_grid_barrier(G, 3);               // act ready (grid-local)
    }
    ffn_gemm2(G, block, gridDim.x);               // gemm2: act @ w2 -> y_expert (all CTAs)
    // bar(4) fences THIS GPU's gemm2 output (local y_expert) before its producers
    // read it; the producers' peer writes are fenced cross-GPU by bar(5), so a
    // grid-local barrier suffices.
    combine_grid_barrier(G, 4);                   // y_expert ready (grid-local)
    // in the default build (use_gather=1 && use_inter_gather=1)
    // combine_scatter_fused continue's on every row (intra gated by use_gather,
    // stage by use_inter_gather) — a pure no-op sweep. Skip it; only run when an
    // opt-out path is active, where it stays byte-identical.
    if (!G.use_gather || !G.use_inter_gather)
        combine_scatter_fused(G);                 // stage (+ intra when !use_gather)
    if (G.use_gather) {
        // single-pass producer fuse (intra+inter, disjoint rows).
        // When inter gather is off, inter rows go through the scatter above, so
        // fall back to the intra-only producer.
        if (G.use_inter_gather) combine_producer_fused(G);
        else                    combine_producer_intra(G);
        combine_global_barrier(G, 5);             // producer comb_buf writes visible
        combine_gather_owner(G);                  // intra: all CTAs write y_out
        // atomic-free inter gather: same phase as intra gather, writes the
        // dense pre_tokens UPPER send row (overwrite). bar(1) below fences it
        // before combine_send reads it, exactly as it fenced the atomic scatter.
        if (G.use_inter_gather) combine_gather_inter(G);
    }

    // Phase 2: inter-node combine. zero hoisted above gemm2 and stage fused into
    // combine_scatter_fused; only bar(1) -> send/reduce remains.
    // No-op for single node. Uses barrier slot 1 internally.
    combine_phase2(G);

    // Last-arriving CTA clears per-row-block barriers in-kernel; all other CTAs
    // only contribute to the cleanup counter.
    __shared__ int is_last_cta;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int prev = atomicAdd(G.cleanup_done, 1u);
        is_last_cta = (prev + 1 == (unsigned int)gridDim.x) ? 1 : 0;
        if (is_last_cta) atomicExch(G.cleanup_done, 0u);
    }
    __syncthreads();
    if (is_last_cta) {
        const int num_row_blocks =
            (G.num_padded_local_tokens + fused_globals::ROW_BLOCK - 1)
            / fused_globals::ROW_BLOCK;
        for (int i = threadIdx.x; i < num_row_blocks; i += blockDim.x)
            G.barrier[G.dev_idx][{i}] = 0;
    }
}

__global__ void fused_cleanup_kernel(__grid_constant__ const fused_cleanup_globals G) {
    for (int row_idx = threadIdx.x; row_idx < G.num_row_blocks; row_idx += blockDim.x) {
        G.barrier[G.dev_idx][{row_idx}] = 0;
    }
}

// Launch wrapper: kept in this TU so the kernel body stays out of the .cuh.
void launch_fused_dispatch_gemm(const fused_globals& G, cudaStream_t stream) {
    cudaFuncSetAttribute(fused_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    fused_kernel<<<SM_COUNT, NUM_MAIN_THREADS, DYNAMIC_SHARED_MEMORY, stream>>>(G);
}

// ============================================================================
// Inter-node combine — Phase 2 of the single fused_kernel.
//
// pre_tokens and peer_tokens are sized 2x: lower half = dispatch region (Phase
// 1), upper half = combine region, in the SAME registered RDMA MR. Combine
// never touches the dispatch data, so it needs NO cross-node barrier — only
// intra-node grid + cross-GPU barriers.
//
// zero  : clear this GPU's pre_tokens UPPER (combine staging) region.
// stage : each expert-GPU IPC-atomic-adds weight*y_expert for its remote-
// source rows into the matching-index GPU's pre_tokens UPPER slot.
// send  : GPU d RDMA-sends pre_tokens upper -> peer same-index peer_tokens
// upper, targeting the combine (upper) arrival-flag slots.
// reduce: wait the combine arrival flags, add received peer_tokens upper into
// y_out (which already holds the intra-node sums from Phase 1).
// Correct for num_nodes==2 (one peer's worth in the upper half).
// ============================================================================

// Grid barrier (all local CTAs) + cross-GPU barrier_all (all M GPUs), using
// barrier slot bi in {0,1}. combine_bar_count[bi]/ready[bi] are zero on entry.
__device__ inline void combine_global_barrier(const fused_globals &G, int bi) {
    __syncthreads();
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        __threadfence_system();
        unsigned int prev = atomicAdd(&G.combine_bar_count[bi], 1u);
        is_last = (prev + 1 == (unsigned int)gridDim.x) ? 1 : 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        atomicExch(&G.combine_bar_count[bi], 0u);
        // Distinct grid-counter slot per barrier (bi); only 2 cross-GPU planes
        // exist, reused round-robin (barriers are sequential so a plane is fully
        // reset before its next use).
        dist::barrier_all<fused_globals::NUM_DEVICES>(G.sync_barrier, {bi % 2, 0}, G.dev_idx);
        __threadfence_system();
        atomicExch(&G.combine_bar_ready[bi], 1u);
    }
    if (threadIdx.x == 0)
        while (atomicAdd(&G.combine_bar_ready[bi], 0u) == 0u) __nanosleep(64);
    __syncthreads();
}

// grid-LOCAL barrier (all local CTAs only — NO cross-GPU plane, NO
// __threadfence_system). Used ONLY for bar(4): between bar(4) and bar(5) the
// producers read ONLY this GPU's local y_expert (gemm2 output) and write into
// PEER comb_buf/inter_comb_buf; the first cross-GPU READ of those peer buffers is
// the gathers, AFTER bar(5) (still a full cross-GPU combine_global_barrier). So
// bar(4) need only guarantee THIS GPU's gemm2 CTAs finished before THIS GPU's
// producers read local y_expert — a grid-local guarantee. Reuses combine_bar_*[bi].
__device__ inline void combine_grid_barrier(const fused_globals &G, int bi) {
    __syncthreads();
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        __threadfence();                              // grid-local visibility (NOT _system)
        unsigned int prev = atomicAdd(&G.combine_bar_count[bi], 1u);
        is_last = (prev + 1 == (unsigned int)gridDim.x) ? 1 : 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        atomicExch(&G.combine_bar_count[bi], 0u);
        atomicExch(&G.combine_bar_ready[bi], 1u);     // no dist::barrier_all
    }
    if (threadIdx.x == 0)
        while (atomicAdd(&G.combine_bar_ready[bi], 0u) == 0u) __nanosleep(64);
    __syncthreads();
}

__device__ inline void combine_zero_staging(const fused_globals &G) {
    const int H = fused_globals::H;
    const int base = G.num_local_tokens;   // upper half (token rows) starts here
    // Vectorized zero-fill: 8 bf16 (one float4) per thread. The staging region is
    // contiguous [base*H, 2*base*H) in the flat pre_tokens buffer.
    __nv_bfloat16* dst0 = &G.pre_tokens_flat[G.dev_idx][{base, 0}];
    const int total_vec = (G.num_local_tokens * H) / 8;
    const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_vec;
         idx += gridDim.x * blockDim.x) {
        reinterpret_cast<float4*>(dst0)[idx] = zero4;
    }
}

__device__ inline void combine_stage(const fused_globals &G) {
    const int H = fused_globals::H;
    const int base = G.num_local_tokens;   // pre_tokens UPPER half
    for (int r = blockIdx.x; r < G.num_padded_local_tokens; r += gridDim.x) {
        const int e = G.row_expert_ptr[r];
        if (e < 0 || e >= fused_globals::NUM_EXPERTS_PER_DEV) continue;
        const int sn = G.pull_dispatch_indices[{r, 0}];
        const int sd = G.pull_dispatch_indices[{r, 1}];
        const int st = G.pull_dispatch_indices[{r, 2}];
        if (sn < 0 || sd < 0 || st < 0) continue;   // padding
        if (sn == G.node_idx) continue;             // intra-node: done in Phase 1
        const float w = G.topk_weights_ptr[r];
        const __nv_bfloat16* y = G.y_expert_ptr + (size_t)r * H;
        for (int h2 = threadIdx.x; h2 < H / 2; h2 += blockDim.x) {
            const int h = 2 * h2;
            __nv_bfloat162 v = __floats2bfloat162_rn(w * __bfloat162float(y[h]),
                                                     w * __bfloat162float(y[h + 1]));
            atomicAdd(reinterpret_cast<__nv_bfloat162*>(
                          &G.pre_tokens_flat[sd][{base + st, h}]), v);
        }
    }
}

// RDMA-send pre_tokens UPPER -> peer same-index peer_tokens UPPER. Mirrors
// fused_inter_send_sm but offsets every address into the combine (upper) region
// and targets the combine arrival-flag slots (same session epoch as dispatch).
__device__ inline void combine_send(const fused_globals &G) {
    const int warps_per_cta = NUM_MAIN_THREADS / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (lane_id != 0) return;
    const int n_peers = G.num_nodes - 1;
    const int total_pushers = G.num_send_sms * warps_per_cta;
    const int my_pusher = blockIdx.x * warps_per_cta + warp_id;
    const int single_peer_bytes = G.pre_tokens_bytes;
    const int single_peer_tiles = G.total_chunks;
    const int dispatch_flags = n_peers * G.total_chunks;        // combine flags start here
    const uint64_t local_base  = (uint64_t)G.pre_tokens_bytes;             // pre_tokens upper
    const uint64_t remote_base = (uint64_t)n_peers * G.pre_tokens_bytes;   // peer_tokens upper
    for (int chunk_id = my_pusher; chunk_id < G.total_chunks; chunk_id += total_pushers) {
        uint32_t off = (uint32_t)(chunk_id * CHUNK_BYTES);
        uint32_t bytes = min(CHUNK_BYTES, G.pre_tokens_bytes - (int)off);
        for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
            const int peer_rank = internode::peer_rank_for_slot(G.node_idx, G.num_nodes, peer_slot);
            const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
            internode::TransferCmd cmd{};
            cmd.cmd_type = internode::CmdType::WRITE;
            cmd.dst_rank = (uint8_t)peer_rank;
            cmd.tile_id = (uint32_t)(dispatch_flags + sap * single_peer_tiles + chunk_id);
            cmd.bytes = bytes;
            // local_base is peer-slot 0's region; each peer reads its own dense [L,H].
            cmd.local_offset = local_base + (uint64_t)peer_slot * (uint32_t)single_peer_bytes + off;
            cmd.remote_offset = remote_base + (uint64_t)sap * (uint32_t)single_peer_bytes + off;
            cmd.lane_id = (uint16_t)chunk_id;
            cmd.reserved0 = (uint8_t)(peer_slot * fused_globals::NUM_DEVICES + G.dev_idx);
            internode::D2HFifoDevice fifo =
                internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)cmd.lane_id);
            fifo.push(cmd);
        }
    }
}

__device__ inline void combine_reduce(const fused_globals &G) {
    const int H = fused_globals::H;
    const int reduce_id = blockIdx.x - G.num_send_sms;
    const int num_reduce = gridDim.x - G.num_send_sms;
    const int n_peers = G.num_nodes - 1;
    const int dispatch_flags = n_peers * G.total_chunks;
    const int recv_base_tok = n_peers * G.num_local_tokens;   // peer_tokens UPPER (tokens)
    // Wait for every peer slot's chunks to arrive before reducing.
    if (threadIdx.x == 0) {
        for (int f = 0; f < n_peers * G.total_chunks; ++f) {
            // Monotonic >= (not ==): a peer that has advanced stamps a higher
            // epoch (still valid), and a freshly-reset flag (0) stays < epoch.
            while (comm::atomic_u32::acquire_load_sys(
                       &G.arrival_flags[dispatch_flags + f]) < G.epoch)
                __nanosleep(100);
        }
    }
    __syncthreads();
    // Vectorized reduce: each thread handles 8 contiguous h-values (two float4 of
    // y_out + one float4 of bf16 peer data per slot). H is a multiple of 8.
    constexpr int VEC = 8;
    const int HV = H / VEC;                       // 8-wide groups per row
    const int total = G.num_local_tokens * HV;
    float* y_base = &G.y_out[G.dev_idx][{0, 0}];
    for (int idx = reduce_id * blockDim.x + threadIdx.x; idx < total;
         idx += num_reduce * blockDim.x) {
        const int t = idx / HV, j = idx % HV;     // j-th 8-element group in row t
        float* yp = y_base + (size_t)t * H + j * VEC;
        float4 a0 = reinterpret_cast<float4*>(yp)[0];
        float4 a1 = reinterpret_cast<float4*>(yp)[1];
        for (int slot = 0; slot < n_peers; ++slot) {
            const __nv_bfloat16* pk =
                G.peer_tokens_local + (size_t)(recv_base_tok + slot * G.num_local_tokens) * H;
            float4 pv = reinterpret_cast<const float4*>(pk + (size_t)t * H + j * VEC)[0];
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&pv);
            a0.x += __low2float(p2[0]); a0.y += __high2float(p2[0]);
            a0.z += __low2float(p2[1]); a0.w += __high2float(p2[1]);
            a1.x += __low2float(p2[2]); a1.y += __high2float(p2[2]);
            a1.z += __low2float(p2[3]); a1.w += __high2float(p2[3]);
        }
        reinterpret_cast<float4*>(yp)[0] = a0;
        reinterpret_cast<float4*>(yp)[1] = a1;
    }
}

// In-kernel reset of this node's combine arrival flags (upper half of the flag
// array). Runs after combine_reduce has consumed them and before the cross-node
// barrier push, so the peer's next-iteration RDMA writes cannot clobber the
// clear. Replaces the host-side reset_arrival_flags in commit_epoch, which
// races in-flight RDMA from a peer that has already advanced.
__device__ inline void combine_iter_end_reset_arrival_flags(const fused_globals &G) {
    const int n_peers = G.num_nodes - 1;
    const int dispatch_flags = n_peers * G.total_chunks;   // combine flags start here
    const int total_combine_flags = n_peers * G.total_chunks;
    const int stride = gridDim.x * blockDim.x;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    volatile uint32_t* flag_ptr = G.arrival_flags;
    for (int i = offset; i < total_combine_flags; i += stride) {
        flag_ptr[dispatch_flags + i] = 0u;
    }
    __threadfence_system();
}

// Cross-node completion handshake: the kernel does not exit until both nodes
// have finished combine for this epoch, so y_out is complete and visible to a
// downstream kernel on the same stream. One driver CTA pushes BARRIER_NOTIFY
// (the proxy drains its in-flight writes, then RDMA-writes its epoch into the
// peer's barrier slot) and spins the host-mapped flag; the other CTAs spin an
// HBM mirror to keep the PCIe flag down to a single reader.
__device__ inline void combine_xnode_barrier(const fused_globals &G, int driver_block_id) {
    if (G.cross_node_barrier == nullptr) return;

    if (blockIdx.x == driver_block_id) {
        if (threadIdx.x == 0) {
            internode::TransferCmd cmd{};
            cmd.cmd_type = internode::CmdType::BARRIER_NOTIFY;
            cmd.enqueue_device_ns = comm::globaltimer();
            internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, 0).push(cmd);

            // Monotonic >= (flag is never cleared): a peer ahead satisfies
            // immediately, a flag still at 0 keeps us waiting.
            while (comm::atomic_u32::acquire_load_sys(
                       const_cast<uint32_t*>(G.cross_node_barrier)) < G.epoch) {
                __nanosleep(100);
            }
            if (G.xnode_ready_device != nullptr) {
                comm::atomic_u32::release_store_gpu(G.xnode_ready_device, G.epoch);
            }
        }
    } else if (G.xnode_ready_device != nullptr) {
        if (threadIdx.x == 0) {
            uint32_t v;
            do {
                v = comm::atomic_u32::acquire_load_gpu(G.xnode_ready_device);
                if (v < G.epoch) __nanosleep(100);
            } while (v < G.epoch);
        }
    } else {
        if (threadIdx.x == 0) {
            while (comm::atomic_u32::acquire_load_sys(
                       const_cast<uint32_t*>(G.cross_node_barrier)) < G.epoch) {
                __nanosleep(100);
            }
        }
    }
    __syncthreads();
}

// Phase 2 (inter-node combine), no-op for single node:
//   barrier -> send + reduce -> converge -> reset flags -> cross-node barrier.
// The zero-fill is hoisted before gemm2 (folded into bar(3)) and the stage
// scatter is fused into combine_scatter_fused, so only bar(1) -> send/reduce
// remains here. On return y_out is complete and visible cross-node, so no host
// dist.barrier is needed between back-to-back epoch-advancing launches (the host
// must set MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET=1 so it does not race the
// in-kernel flag reset).
__device__ inline void combine_phase2(const fused_globals &G) {
    if (G.num_nodes <= 1) return;
    // zero + stage already done (zero hoisted before gemm2; stage fused into
    // combine_scatter_fused).
    // bar(1): ensure the inter gather's local writes to pre_tokens UPPER are
    // complete before combine_send RDMA-reads them. In the default gather path
    // send reads only THIS GPU's pre_tokens UPPER (the gather wrote them locally;
    // the gather's own __threadfence_system already made them NIC-visible), so a
    // grid-LOCAL barrier suffices — no peer buffer is read here. (In the legacy
    // !use_inter_gather scatter path the staging atomics target PEER pre-upper,
    // which DOES need cross-GPU visibility, so keep the global barrier there.)
    if (G.use_inter_gather) combine_grid_barrier(G, 1);
    else                    combine_global_barrier(G, 1);
    if (blockIdx.x < G.num_send_sms) combine_send(G);
    else                             combine_reduce(G);

    // Converge all CTAs (reuse the otherwise-idle bar(0)) so every reduce's
    // y_out write and every send's RDMA push is issued before we reset/notify.
    // Grid-LOCAL: between here and the notify nothing reads a PEER buffer — the
    // reset clears THIS node's local arrival_flags and reduce wrote local y_out —
    // so the NVLink barrier_all is unnecessary. The flag reset's own
    // __threadfence_system still orders it before the driver's notify push.
    combine_grid_barrier(G, 0);

    // Reset this node's combine flags before the notify push; the peer's
    // next-iteration writes are gated on the notify, so they can't race it.
    combine_iter_end_reset_arrival_flags(G);

    // Cross-node completion handshake; driver = first reduce CTA.
    combine_xnode_barrier(G, G.num_send_sms);
}

}  // namespace moe_dispatch_gemm_glu_combine_multinode

#include "operators/dispatch_gemm_glu_combine/session.cuh"
