/**
 * @file dispatch_gemm.cu
 * @brief Multi-node MoE Dispatch + Group GEMM - single fused kernel.
 *
 * Single kernel launch. CTA roles are split by blockIdx.x:
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
 *   GEMM CTAs [..., 132):
 *     Run grouped expert GEMMs after the dispatched row blocks are ready.
 *
 * The kernel overlaps RDMA, token dispatch, and expert GEMM, then lets the
 * last CTA clear per-row dispatch barriers before exit.
 *
 * Infrastructure (config, globals, helpers, host setup, entrypoint) lives in
 *   include/operators/dispatch_gemm/dispatch_gemm.cuh
 * Python/session glue + pybind module live in
 *   include/operators/dispatch_gemm/session.cuh
 */
#include "operators/dispatch_gemm/dispatch_gemm.cuh"

namespace moe_dispatch_gemm_multinode {

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

__device__ inline void group_gemm_fused(const fused_globals &G, const int sm_idx, const int num_sms) {
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
    // Per-expert count of pure-local row_blocks (first N rb of each expert
    // are all-local under DISPATCH_LOCAL_FIRST). Used by DISPATCH_GEMM_LOCAL_FIRST for
    // its two-pass schedule.
    __shared__ int local_rb_per_expert[fused_globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < fused_globals::NUM_EXPERTS_PER_DEV)
        local_rb_per_expert[threadIdx.x] =
            G.local_rb_per_expert[{(int)threadIdx.x}];

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
    constexpr int num_iters = fused_globals::H / fused_globals::RED_BLOCK;
    constexpr int col_blocks = fused_globals::I / fused_globals::COL_BLOCK;

    // Under DISPATCH_GEMM_LOCAL_FIRST we walk all experts' pure-local row_blocks
    // first (pass 0), then all experts' peer row_blocks (pass 1). The pass=0
    // barrier waits never see RDMA latency, so compute drains ~50% of its
    // work during the 2.8ms RDMA tail at 131K. Pipeline stage/phasebits state
    // is persistent across passes — total iterations equal the single-pass
    // count, just reordered.
    constexpr int NUM_PASSES = 2;

    if (wg_id == 2) {
        warpgroup::decrease_registers<40>();
        if (w_id == 0) {
            // Loader warp. Lane 0 issues big-TMA A+B loads.
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
                        dispatch_swizzle_decode(task_id, row_blocks, col_blocks,
                                           _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;
                        if (l_id == 0) {
                            // ld.acquire.gpu.global pairs with the producer's
                            // red.release.gpu.global.add in dispatch_fused
                            // (same-GPU producer/consumer for post_tokens). The
                            // acquire makes prior dispatch ::dist::tma::store_async +
                            // store_async_wait writes to post_tokens visible
                            // before our subsequent TMA load — was ld.relaxed,
                            // which left a memory-model gap that worked only
                            // because L2 turnover hid the staleness.
                            //
                            // Nanosleep 16 -> 64ns: dispatch fan-in is ~128
                            // atomic adds per row_block barrier; many compute
                            // CTAs (col_blocks of the same row_block) poll the
                            // same line in tight 16ns intervals, generating
                            // ~3-5 GB/s of L2 contention per slot. 64ns cuts
                            // that 4x with no wall impact (waits are 100s of
                            // ns to us anyway).
                            int bar_val;
                            bar_val = comm::atomic_u32::acquire_load_s32_gpu(
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
                                    ::dist::tma::load_async(inputs[stage].A[i], G.post_tokens,
                                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                                ::dist::tma::load_async(inputs[stage].B, G.weights,
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
                        dispatch_swizzle_decode(task_id, row_blocks, col_blocks,
                                           _row_in_grid, _col_idx);
                        const int row_idx = _row_in_grid + row_offset;
                        const int col_idx = _col_idx;
                        wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                        update_phasebit<0>(phasebits, 0);
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::store_async(G.outputs, outputs.C[i], {row_idx * 2 + i, col_idx});
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
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        warpgroup::mma_AB(C_accum, inputs[stage].A[wg_id], inputs[stage].B);
                        warpgroup::mma_async_wait();
                        warp::arrive(inputs_finished[stage]);
                        stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                    }
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
        group_gemm_fused(G, comp_idx, G.num_comp_sms);
    }

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

}  // namespace moe_dispatch_gemm_multinode

#include "operators/dispatch_gemm/session.cuh"
