/**
 * @file gemm_ar_multinode.cu
 * @brief Multi-node GEMM + All-Reduce — single fused kernel.
 *
 * Single kernel launch. Four CTA groups run concurrently:
 *
 *   Compute CTAs [0, num_comp_sms):
 *     GEMM A@B, TMA-store result tiles into C_distributed_tensor (IPC multicast buffer).
 *     Signals per-tile barrier so intra-AR CTAs know each tile is ready.
 *     Tile visit order is selectable: default slice-major, optional
 *     slice-interleaved (GEMM_AR_GEMM_INTERLEAVE_SLICES) to balance intra-AR,
 *     or device-relative slice interleaving (GEMM_AR_GEMM_INTERLEAVE_DEVREL)
 *     to front-load each GPU's own outbound rows.
 *
 *   Intra-AR CTAs [num_comp_sms, num_comp_sms + num_intra_ar_sms):
 *     Wait per-tile barrier, run IPC multicast all-reduce on the tile.
 *     Publishes tile readiness through multicast barrier slots so every GPU's
 *     inter-send CTA can observe the handoff.
 *
 *   Inter-send CTAs [..., ... + num_inter_send_sms):
 *     Wait the multicast-ready slots for tiles in my slice (dev_idx's rows),
 *     copy to staging buffer, push RDMA via D2H FIFO. Fires RDMA as soon as
 *     each row-block's tiles are all-reduced, overlapping with ongoing
 *     intra-AR on other tiles.
 *
 *   Inter-reduce-and-publish CTAs [..., 132):
 *     Poll arrival_flags for RDMA data, reduce C_local + C_recv into a
 *     multicast-backed final tensor for my slice. After all my-slice tiles are
 *     done, publish slice completion through multicast barrier slots so every
 *     local GPU's final-output view is complete without a peer-read all-gather.
 *
 * No host synchronization anywhere — all coordination via device-side flags.
 *
 * Infrastructure (config, structs, helpers, host entrypoint) lives in
 *   include/operators/gemm_ar/gemm_ar.cuh
 * Python/session glue + pybind module live in
 *   include/operators/gemm_ar/session.cuh
 */

#include "common/cuda_checks.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/ready_queue.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace kittens;

#include "operators/gemm_ar/gemm_ar.cuh"

namespace gemm_ar_multinode {

// ============================================================================
// 4 CTA roles + fused-kernel dispatcher.
//
// Each CTA picks its role based on blockIdx.x, then runs one of:
//   compute       fused_comp_sm
//   intra-AR      fused_intra_ar_sm
//   inter-send    fused_inter_send_sm
//   inter-reduce  fused_inter_reduce_and_publish_sm
//
// Plus fused_epilogue_kernel for tail-side multicast cleanup.
// ============================================================================

__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    fused_globals::pipeline_inputs (&inputs)[fused_globals::PIPELINE_STAGES] =
        al.allocate<fused_globals::pipeline_inputs, fused_globals::PIPELINE_STAGES>();
    fused_globals::pipeline_outputs& outputs =
        *reinterpret_cast<fused_globals::pipeline_outputs*>(&inputs[fused_globals::PIPELINE_STAGES - 1]);
    __shared__ semaphore inputs_arrived[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived, outputs_finished;
    if (threadIdx.x == 0) {
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();
    int warpgroup_id = warpgroup::groupid(), warp_id = warpgroup::warpid(), lane_id = warp::laneid();
    int stage = 0; uint32_t phasebits = 0xFFFF0000;
    int row_blocks = G.A.rows() / fused_globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / fused_globals::COL_BLOCK;
    int num_blocks = row_blocks * col_blocks;
    int num_iters = G.A.cols() / fused_globals::RED_BLOCK;
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
        if (warp_id == 0 && lane_id == 0) {
            // Producer warp: always per-tile claiming (good load balance at all M).
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
                int row_idx, col_idx;
                gemm_ar_decode_comp_task(
                    task_id, G.row_blocks_per_slice, col_blocks, G.dev_idx, row_idx, col_idx);
                {
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        ::dist::tma::expect_bytes(inputs_arrived[stage], sizeof(fused_globals::pipeline_inputs));
                        if (red_idx == fused_globals::PIPELINE_STAGES - 1) {
                            wait(outputs_finished, get_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES));
                            update_phasebit<1>(phasebits, fused_globals::PIPELINE_STAGES);
                        }
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::load_async(inputs[stage].A[i], G.A, {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                        ::dist::tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);
                        stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                    }
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            // TMA store warp — write to C_distributed_tensor, signal barrier for intra-AR
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
                int row_idx, col_idx;
                gemm_ar_decode_comp_task(
                    task_id, G.row_blocks_per_slice, col_blocks, G.dev_idx, row_idx, col_idx);
                {
                    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                    update_phasebit<0>(phasebits, 0);
                    for (int i = 0; i < 2; i++)
                        ::dist::tma::store_async(G.C[G.dev_idx], outputs.C[i], {row_idx * 2 + i, col_idx});
                    ::dist::tma::store_async_read_wait();
                    arrive(outputs_finished);
                    // Counter-based chunk signaling: per-tile claiming for load
                    // balance, per-chunk NVSwitch signals for reduced traffic.
                    // Each GPU counts completed tiles per chunk via local atomicAdd.
                    // The last CTA to finish a tile in a chunk fires one signal.
                    {
                        const int chunk_col = col_idx / G.chunk_tiles;
                        const int flat_chunk = row_idx * G.chunks_per_row + chunk_col;
                        const int chunk_start = chunk_col * G.chunk_tiles;
                        const int tiles_this_chunk = min(G.chunk_tiles, col_blocks - chunk_start);
                        uint32_t prev = atomicAdd(G.comp_chunk_tiles_done + flat_chunk, 1u);
                        if (prev == (uint32_t)(tiles_this_chunk - 1)) {
                            int slice_owner = row_idx / G.row_blocks_per_slice;
                            signal(G.barrier, {row_idx, chunk_col}, slice_owner, 1);
                        }
                    }
                }
            }
        }
    } else {
        // Consumer warpgroups: WGMMA — always per-tile claiming
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
            int row_idx, col_idx;
            gemm_ar_decode_comp_task(
                task_id, G.row_blocks_per_slice, col_blocks, G.dev_idx, row_idx, col_idx);
            {
                rt_fl<fused_globals::ROW_BLOCK / 8, fused_globals::COL_BLOCK> C_accum;
                warp::zero(C_accum);
                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
                    warpgroup::mma_async_wait();
                    warp::arrive(inputs_finished[stage]);
                    stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
                }
                group<8>::sync(3);
                warpgroup::store(outputs.C[warpgroup_id], C_accum);
                warpgroup::sync(warpgroup_id + 1);
                warpgroup::arrive(outputs_arrived);
            }
        }
    }
}

// ============================================================================
// Pipelined intra-node all-reduce tile helper
// ============================================================================
//
// The vanilla group::all_reduce issues each multimem.ld_reduce with a "memory"
// ASM clobber, which serializes every NVSwitch round-trip (~600 ns each).
// With 42 sequential loads per warp × 4 tiles this creates ~100-180 µs of
// intra-AR latency.
//
// Fix: issue GEMM_AR_AR_UNROLL loads (all independent, different output registers)
// BEFORE any stores, so the SM can have multiple NVSwitch requests in-flight
// simultaneously. The loads have no register dependency between them, so the
// hardware warp scheduler can overlap their NVLink round-trips.
//
// Safety: the caller holds __syncthreads() after the per-tile barrier wait,
// which provides the acquire fence guaranteeing all compute CTAs' writes to
// G.C are visible. No additional "memory" clobber is needed on the loads.

// UNROLL=8: issue 8 independent multimem.ld_reduce loads before any stores,
// hiding 8× the NVSwitch round-trip latency per outer loop iteration.
// For shapes where intra-AR CTA count is very high (M=8192: 32 CTAs → 3072
// simultaneous NVSwitch ops), NVSwitch congestion can be avoided instead by
// reducing num_intra_comm_sms at launch time (Python-side O3 formula).
constexpr int GEMM_AR_AR_UNROLL = 8;
__device__ __forceinline__ void gemm_ar_pipelined_ar_tile(
    const fused_globals::C_distributed_tensor& dbuf, int tile_row, int tile_col
) {
    // dbuf uses element-level row/col indexing in mc_ptr_at.
    constexpr int TILE_ROWS   = fused_globals::ROW_BLOCK;          // 128
    constexpr int TILE_COLS   = fused_globals::COL_BLOCK;          // 256
    constexpr int GROUP_WARPS = config::NUM_WARPS;                  // 12
    constexpr int ELEMS_PER_TH = 2;
    const int warps_per_row   = TILE_COLS / (WARP_THREADS * ELEMS_PER_TH);  // 4
    const int total_iters     = TILE_ROWS * warps_per_row;                  // 512
    const int row_base        = tile_row * TILE_ROWS;               // element row start
    const int col_base        = tile_col * TILE_COLS;               // element col start
    const int warp_laneid     = threadIdx.x % WARP_THREADS;

    for (int i = warpid(); i < total_iters; i += GROUP_WARPS * GEMM_AR_AR_UNROLL) {
        // Compute all pointers for this batch.
        bf16_2* ptrs[GEMM_AR_AR_UNROLL];
        uint32_t tmps[GEMM_AR_AR_UNROLL];

        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                const int ri = row_base + ii / warps_per_row;
                const int ci = col_base + (ii % warps_per_row) * WARP_THREADS * ELEMS_PER_TH + warp_laneid * ELEMS_PER_TH;
                ptrs[u] = reinterpret_cast<bf16_2*>(dbuf.mc_ptr_at({0, 0, ri, ci}));
            }
        }

        // Issue all UNROLL loads without "memory" clobber between them.
        // Independent output registers allow the hardware to pipeline NVLink
        // round-trips (UNROLL requests in-flight vs 1).
        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(
                    tmps[u], ptrs[u]);  // No memory clobber — loads are independent post-syncthreads fence
            }
        }

        // All stores after all loads.
        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(ptrs[u], tmps[u]);
            }
        }
    }
}

// ============================================================================
// Pipelined intra-node reduce-scatter tile helper (iter31: RS variant)
// ============================================================================
//
// Replaces the broadcast multimem.st with a regular local store to C_local.
// After reduce-scatter:
//   - THIS GPU has the correct intra-node sum for its own slice rows.
//   - Other GPUs have NOT received this GPU's sum (no broadcast).
// The send CTA reads C_local (same rows) and sends via IB to the remote node.
// The inter-reduce CTA adds received data + local RS result and publishes via
// multicast to C_final (which IS the all-gather step for local GPUs).
//
// Expected benefit: removes ~half of NVSwitch traffic (no multimem.st),
// reducing intra-AR time for small M where NVSwitch ops dominate.

// Reference's dual-store pattern (gemm_ar_ref_kernel.cu:215-247): the
// multimem.ld_reduce produces results in registers — write them to BOTH
// C_local and a staging slot in the same loop. Eliminates the separate
// pack-to-staging pass. tx_tile_base (when non-null) is the contiguous start
// of this tile's TILE_ROWS×TILE_COLS staging region (G.staging_buf +
// tile_id*TILE_ELEMS in our layout). Null = legacy single-store behavior.
__device__ __forceinline__ void gemm_ar_pipelined_rs_tile(
    const fused_globals::C_distributed_tensor& dbuf, int tile_row, int tile_col,
    bf16* C_local, int N, bf16* tx_tile_base = nullptr
) {
    constexpr int TILE_ROWS   = fused_globals::ROW_BLOCK;          // 128
    constexpr int TILE_COLS   = fused_globals::COL_BLOCK;          // 256
    constexpr int GROUP_WARPS = config::NUM_WARPS;                  // 12
    constexpr int ELEMS_PER_TH = 2;
    const int warps_per_row   = TILE_COLS / (WARP_THREADS * ELEMS_PER_TH);  // 4
    const int total_iters     = TILE_ROWS * warps_per_row;                  // 512
    const int row_base        = tile_row * TILE_ROWS;               // element row start
    const int col_base        = tile_col * TILE_COLS;               // element col start
    const int warp_laneid     = threadIdx.x % WARP_THREADS;

    for (int i = warpid(); i < total_iters; i += GROUP_WARPS * GEMM_AR_AR_UNROLL) {
        bf16_2* mc_ptrs[GEMM_AR_AR_UNROLL];    // multicast addresses for ld_reduce
        bf16_2* local_ptrs[GEMM_AR_AR_UNROLL]; // local GPU addresses for scalar store
        uint32_t tmps[GEMM_AR_AR_UNROLL];

        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                const int ri = row_base + ii / warps_per_row;
                const int ci = col_base + (ii % warps_per_row) * WARP_THREADS * ELEMS_PER_TH + warp_laneid * ELEMS_PER_TH;
                mc_ptrs[u]    = reinterpret_cast<bf16_2*>(dbuf.mc_ptr_at({0, 0, ri, ci}));
                local_ptrs[u] = reinterpret_cast<bf16_2*>(C_local + (long)ri * N + (long)ci);
            }
        }

        // Issue all UNROLL multimem.ld_reduce loads (pipelined, same as AR version).
        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(
                    tmps[u], mc_ptrs[u]);  // No memory clobber — same reasoning as AR version
            }
        }

        // Dual store: results in `tmps[u]` go to C_local AND (when
        // tx_tile_base != null) to the staging slot. Reference pattern.
        #pragma unroll
        for (int u = 0; u < GEMM_AR_AR_UNROLL; u++) {
            const int ii = i + u * GROUP_WARPS;
            if (ii < total_iters) {
                bf16_2* tx_ptr_u = nullptr;
                if (tx_tile_base != nullptr) {
                    const int ri_in_tile = ii / warps_per_row;
                    const int ci_in_tile =
                        (ii % warps_per_row) * WARP_THREADS * ELEMS_PER_TH
                        + warp_laneid * ELEMS_PER_TH;
                    tx_ptr_u = reinterpret_cast<bf16_2*>(
                    tx_tile_base
                        + (long)ri_in_tile * fused_globals::COL_BLOCK
                        + (long)ci_in_tile);
                }
                comm::multimem<comm::bf16_2>::st_global_bits_no_clobber(local_ptrs[u], tmps[u]);
                if (tx_ptr_u != nullptr) {
                    comm::multimem<comm::bf16_2>::st_global_bits_no_clobber(tx_ptr_u, tmps[u]);
                }
            }
        }
    }
}

// ============================================================================
// Role 2: Intra-node all-reduce SM — wait per-tile, reduce, publish ready slot
// ============================================================================

__device__ inline void fused_intra_ar_sm(const fused_globals& G) {

    // Per-tile claiming: iterate over individual tiles (not chunks) for maximum
    // parallelism. Aligns with single-node kernel design where each comm CTA
    // processes one tile independently. tile_id = rb_in_slice * col_blocks + col_idx.
    // At small M (2048/4096), this lets many CTAs process tiles in parallel
    // instead of serializing chunk_tiles tiles per chunk in one CTA.
    const int total_tiles = G.total_tiles_per_device;
    const int my_offset = blockIdx.x - G.num_comp_sms;
    const int stride = G.num_intra_ar_sms;

    for (int raw_step = 0; raw_step * stride < total_tiles; raw_step++) {
        int tile_id = my_offset + raw_step * stride;
        if (tile_id >= total_tiles) break;
        // Decode tile_id → (rb_in_slice, row_idx, col_idx, chunk_id)
        const int rb_in_slice = tile_id / G.col_blocks;
        const int col_idx = tile_id - rb_in_slice * G.col_blocks;
        const int row_idx = G.dev_idx * G.row_blocks_per_slice + rb_in_slice;
        const int chunk_col = col_idx / G.chunk_tiles;
        const int chunk_id = rb_in_slice * G.chunks_per_row + chunk_col;


        // Hardware barrier waits for all local devices to signal this chunk.
        // The acquire-vs-relaxed choice is outside the spin loop.
        if (G.use_acquire_poll) {
            wait_acquire(G.barrier, {row_idx, chunk_col}, G.dev_idx, fused_globals::NUM_DEVICES);
        } else {
            wait(G.barrier, {row_idx, chunk_col}, G.dev_idx, fused_globals::NUM_DEVICES);
        }
        __syncthreads();

        bf16* _tx_tile_base =
            G.staging_buf + (long)tile_id * fused_globals::TILE_ELEMS;
        gemm_ar_pipelined_rs_tile(G.C, row_idx, col_idx, G.C_local, G.N,
                             _tx_tile_base);
        __syncthreads();

        // Per-chunk completion: atomicAdd on intra_chunk_tiles_done[chunk_id].
        // Last tile in chunk sets local_done_flag for downstream sender/reducer.
        // The last CTA also resets the per-chunk barrier slot so that under
        // GEMM_AR_STEADY_STATE_BENCH we do not rely on host-side `barrier.zero_()`,
        // which races with in-flight peer p2p signals during iter-transition
        // (barrier plane 0 is the only cross-replica slot at risk — peers
        // `signal()` directly into slice_owner's replica via NVSwitch, outside
        // of any CUDA stream ordering).
        if (threadIdx.x == 0) {
            __threadfence();
            const int col_start = chunk_col * G.chunk_tiles;
            const int tiles_this_chunk = min(G.chunk_tiles, G.col_blocks - col_start);
            uint32_t prev = atomicAdd(G.intra_chunk_tiles_done + chunk_id, 1u);
            if (prev + 1u == (uint32_t)tiles_this_chunk) {
                // Reset barrier slot for next iter: slot now at NUM_DEVICES,
                // decrement back to 0. Safe because all tiles_this_chunk CTAs
                // have already passed wait() (wait precedes the reduce that
                // precedes this atomicAdd), so no reader will miss the full count.
                comm::atomic_u32::release_add_sys(
                    &G.barrier[G.dev_idx][{row_idx, chunk_col}],
                    -(int)fused_globals::NUM_DEVICES);
                // Also reset the tile-done counter so next iter starts from 0
                // (ar_done.zero_() is skipped under steady-state in Python).
                gemm_ar_release_store_u32(G.intra_chunk_tiles_done + chunk_id, 0u);
                gemm_ar_release_store_u32(G.local_done_flag + chunk_id, 1u);
                const uint32_t done_count = atomicAdd(G.local_ar_done_chunks, 1u) + 1u;
                gemm_ar_debug_log_transition(G, "local_ar_done", chunk_id, done_count);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Role 3: Inter-node send SM — wait multicast-ready slots, copy, push RDMA
// ============================================================================
__device__ inline void gemm_ar_pack_tiles_to_staging(
    const fused_globals& G, int row_idx, int first_tile_id, int col_start, int run_tiles
) {
    const int dst_cols = run_tiles * fused_globals::COL_BLOCK;
    const bf16* src_base = G.C_local
                           + (long)row_idx * fused_globals::ROW_BLOCK * G.N
                           + (long)col_start * fused_globals::COL_BLOCK;
    bf16* dst_base = G.staging_buf + (long)first_tile_id * fused_globals::TILE_ELEMS;
    const int dst_vecs_per_row = dst_cols / 8;
    const int total_vecs = fused_globals::ROW_BLOCK * dst_vecs_per_row;
    if (dst_cols == G.N) {
        // Whole-row pack: contiguous copy from C_local into staging.
        const int total_elems = fused_globals::ROW_BLOCK * dst_cols;
        const int total_u4 = total_elems / 8;
        for (int idx = threadIdx.x; idx < total_u4; idx += config::NUM_THREADS) {
            reinterpret_cast<uint4*>(dst_base)[idx] = reinterpret_cast<const uint4*>(src_base)[idx];
        }
    } else {
        for (int vec_idx = threadIdx.x; vec_idx < total_vecs; vec_idx += config::NUM_THREADS) {
            const int r = vec_idx / dst_vecs_per_row;
            const int c = vec_idx - r * dst_vecs_per_row;
            const uint4* src4 = reinterpret_cast<const uint4*>(src_base + (long)r * G.N);
            uint4* dst4 = reinterpret_cast<uint4*>(dst_base + (long)r * dst_cols);
            dst4[c] = src4[c];
        }
    }
    __syncthreads();
}

__device__ inline void fused_inter_send_sm(const fused_globals& G) {
    const int tid = threadIdx.x;
    const int send_base = G.num_comp_sms + G.num_intra_ar_sms;
    const int send_id = blockIdx.x - send_base;


    // Phase 2 (Step 2c): static row-block ownership. CTA at offset send_id
    // owns row-blocks {send_id, send_id + stride, ...}. Wait on each owned
    // row-block's chunks via local_done_flag (acquire), then either coalesce
    // them into one whole-row RDMA or send each chunk individually.
    const int row_blocks = G.row_blocks_per_slice;
    const int stride = G.num_inter_send_sms;
    const int chunks_per_row = G.chunks_per_row;


    for (int rb_in_slice = send_id; rb_in_slice < row_blocks; rb_in_slice += stride) {
        const int first_chunk = rb_in_slice * chunks_per_row;


        // Per-chunk send, optionally grouping adjacent chunks.
        constexpr int kCoalesceK = 1;
        int c = first_chunk;
        while (c < first_chunk + chunks_per_row) {
            // How many consecutive chunks to bundle into a single RDMA write.
            int group_k = chunks_per_row - (c - first_chunk);
            if (group_k > kCoalesceK) group_k = kCoalesceK;

            if (tid == 0) {
                // Wait for all group_k chunks to be locally ready. The group is
                // a contiguous prefix of this row-block's remaining chunks, so
                // serial waits do not hurt pipelining relative to the non-
                // coalesced path (we would wait on each chunk in turn anyway).
                for (int gi = 0; gi < group_k; ++gi) {
                    const int cc = c + gi;
                    while (gemm_ar_acquire_load_u32(G.local_done_flag + cc) == 0u) {
                        gemm_ar_debug_maybe_dump_stuck(G, "send_wait_local", send_id, cc);
                        __nanosleep(50);
                    }
                }
            }
            __syncthreads();

            // Decode the first chunk in the group; coalesced tiles are
            // contiguous in column-space within a single row-block.
            int sub_rb_in_slice, sub_row_idx, sub_col_start, sub_cols_this_chunk;
            gemm_ar_chunk_decode(
                c, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
                sub_rb_in_slice, sub_row_idx, sub_col_start, sub_cols_this_chunk);
            // Compute the total tile count spanned by the group. Only the last
            // chunk in a row may be partial; all earlier chunks are full
            // (chunk_tiles). So group_tiles = (group_k-1)*chunk_tiles +
            // tiles_of_last_chunk.
            const int last_c = c + group_k - 1;
            int last_rb_dummy, last_row_dummy, last_col_start_dummy, last_tiles;
            gemm_ar_chunk_decode(
                last_c, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
                last_rb_dummy, last_row_dummy, last_col_start_dummy, last_tiles);
            const int group_tiles = (group_k - 1) * G.chunk_tiles + last_tiles;
            const int pack_first_tile = sub_rb_in_slice * G.col_blocks + sub_col_start;
            const int logical_q = gemm_ar_send_logical_queue(G, sub_rb_in_slice, sub_col_start);

            __syncthreads();  // intra_ar wrote staging during reduce

            if (tid == 0) {
                const uint32_t offset = (uint32_t)((long)pack_first_tile * TILE_BYTES);
                // Per-peer slot offsets: at N == 2, sap == 0 so the offsets
                // are zero (bit-identical). At N > 2 they partition the
                // receiver's recv_buf / arrival flag space by sender slot.
                const long single_peer_bytes =
                    (long)G.row_blocks_per_slice * (long)G.col_blocks * TILE_BYTES;
                const int  single_peer_tiles =
                    G.row_blocks_per_slice * G.col_blocks;
                const int n_peers = G.num_nodes - 1;
                for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint16_t)(sap * single_peer_tiles + pack_first_tile);
                    cmd.bytes = (uint32_t)((long)group_tiles * TILE_BYTES);
                    cmd.local_offset = offset;
                    cmd.src_view = 0;
                    // Encode the grouped tile count so the proxy can publish all
                    // arrivals covered by this send.
                    cmd.row_count = (uint16_t)group_tiles;
                    cmd.remote_offset = (uint32_t)((long)sap * single_peer_bytes) + offset;
                    cmd.lane_id = (uint16_t)logical_q;
                    cmd.enqueue_device_ns = gemm_ar_globaltimer();
                    // v2 default: per-WR fence + DB. Safe across all shapes.
                    gemm_ar_post_send_cmd(G, send_id, logical_q, cmd);
                }
                atomicAdd(G.send_issued_chunks, 1u);
                __threadfence();
            }
            __syncthreads();
            c += group_k;
        }
    }

}

// ============================================================================
// Receiver path: claim remote-arrived chunks and publish final output
// ============================================================================

// ============================================================================
// Out-of-order work-stealing reduce (GEMM_AR_INTER_REDUCE_WORKSTEAL).
//
// Only the dedicated inter-reduce CTAs enter; recycled CTAs exit
// immediately. The dedicated CTAs use a global cursor + bounded scan
// window to process chunks in any-ready order, eliminating the
// head-of-line blocking in the static path where a late RDMA arrival
// stalls all subsequent chunks owned by that CTA.
//
// Claiming uses atomicCAS on local_done_flag (1→2), which makes claimed
// chunks invisible to future scans (they read as 2u != 1u).
//
// Queue draining (RDMA arrival flag publication) remains with CTAs
// that own a queue (red_id < num_remote_queues).
__device__ inline void shared_reduce_my_slice_worksteal(
    const fused_globals& G
) {
    const int tid = threadIdx.x;
    const long slice_row_offset = (long)G.dev_idx * G.slice_rows;
    const long slice_elem_offset = slice_row_offset * G.N;
    const int reduce_base = G.num_comp_sms + G.num_intra_ar_sms + G.num_inter_send_sms;
    const int red_id = blockIdx.x - reduce_base;

    // Only dedicated inter-reduce CTAs enter. Recycled CTAs (compute,
    // intra-AR, inter-send that finished primary work) exit immediately —
    // having 132 CTAs compete on the scan/CAS/cursor creates L2 contention
    // that outweighs the parallelism benefit.
    if (red_id < 0) return;

    // Queue draining: only CTAs in the dedicated inter-reduce range own a queue.
    const int my_queue = (red_id < G.num_remote_queues) ? red_id : -1;


    // Bounded scan window with cursor-diversified origin. Each iteration
    // scans at most SCAN_WIN chunks starting from a globally advancing
    // cursor position. CAS on local_done_flag (1→2) makes claimed chunks
    // invisible to subsequent scans, so bounded windows cannot lose chunks.
    constexpr int SCAN_WIN = 16;
    __shared__ int s_chunk_id;

    while (true) {
        if (tid == 0) {
            if (my_queue >= 0) {
                gemm_ar_drain_arrival_queue_publish_flags(G, my_queue);
            }

            // Get a diversified scan origin via global cursor.
            const uint32_t cursor = atomicAdd(G.reduce_cursor, 1u);
            const int origin = (int)(cursor % (unsigned)G.total_chunks);
            const int win = (G.total_chunks < SCAN_WIN)
                          ? G.total_chunks : SCAN_WIN;

            s_chunk_id = -1;
            for (int i = 0; i < win; ++i) {
                const int c = (origin + i) % G.total_chunks;
                if (gemm_ar_acquire_load_u32(G.remote_arrived_flag + c) == 0u) continue;
                if (gemm_ar_acquire_load_u32(G.local_done_flag + c) != 1u) continue;
                const uint32_t old = atomicCAS(
                    reinterpret_cast<unsigned int*>(G.local_done_flag + c), 1u, 2u);
                if (old == 1u) {
                    s_chunk_id = c;
                    break;
                }
            }

            if (s_chunk_id < 0) {
                if ((int)gemm_ar_acquire_load_u32(G.published_chunks) >= G.total_chunks) {
                    s_chunk_id = -2;
                } else {
                    __nanosleep(50);
                }
            }
        }
        __syncthreads();

        // All threads see the same s_chunk_id via shared memory.
        if (s_chunk_id == -2) break;  // all done — uniform exit
        if (s_chunk_id < 0) continue; // nothing found, retry



        const int chunk_id = s_chunk_id;
        int rb_in_slice, row_idx, col_start, tiles_this_chunk;
        gemm_ar_chunk_decode(
            chunk_id, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
            rb_in_slice, row_idx, col_start, tiles_this_chunk);

        for (int local_col = 0; local_col < tiles_this_chunk; ++local_col) {
            const int col_idx    = col_start + local_col;
            const int tile_id    = rb_in_slice * G.col_blocks + col_idx;
            const bf16* recv     = G.C_recv + (long)tile_id * fused_globals::TILE_ELEMS;
            constexpr int ELEMS_PER_VEC = 8;
            constexpr int VECS_PER_ROW  = fused_globals::COL_BLOCK / ELEMS_PER_VEC; // 32
            constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;        // 12
            const int vec_lane  = tid % VECS_PER_ROW;
            const int row_lane  = tid / VECS_PER_ROW;
            const int col_elem  = vec_lane * ELEMS_PER_VEC;
            const int global_col = col_idx * fused_globals::COL_BLOCK + col_elem;
            const long tile_col_offset = (long)col_idx * fused_globals::COL_BLOCK;
            for (int r = row_lane; r < fused_globals::ROW_BLOCK; r += ROWS_PER_WAVE) {
                const int local_row  = rb_in_slice * fused_globals::ROW_BLOCK + r;
                const int global_row = (int)(slice_row_offset + (long)local_row);
                const long global_elem =
                    slice_elem_offset
                    + (long)local_row * G.N
                    + tile_col_offset
                    + (long)col_elem;
                const uint4 lv = *reinterpret_cast<const uint4*>(G.C_local + global_elem);
                const uint4 rv = *reinterpret_cast<const uint4*>(
                    recv + r * fused_globals::COL_BLOCK + col_elem);
                __nv_bfloat162 o0 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.x),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
                __nv_bfloat162 o1 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.y),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
                __nv_bfloat162 o2 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.z),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
                __nv_bfloat162 o3 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.w),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
                uint4 ov;
                ov.x = *reinterpret_cast<unsigned int*>(&o0);
                ov.y = *reinterpret_cast<unsigned int*>(&o1);
                ov.z = *reinterpret_cast<unsigned int*>(&o2);
                ov.w = *reinterpret_cast<unsigned int*>(&o3);
                gemm_ar_publish_final_vec8(G, global_row, global_col, ov);
            }
        }

        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            const uint32_t published = atomicAdd(G.published_chunks, 1u) + 1u;
            gemm_ar_debug_log_transition(G, "published", chunk_id, published);
        }
        __syncthreads();
    }
}

// ============================================================================
// Static inter-reduce for small chunk counts. Each dedicated reduce CTA owns
// chunks {red_id, red_id + stride, ...}, so no CAS is needed to claim work.
// CTAs without chunks still service arrival queues.
// ============================================================================
__device__ inline void shared_reduce_my_slice_static(
    const fused_globals& G
) {
    const int tid = threadIdx.x;
    const long slice_row_offset = (long)G.dev_idx * G.slice_rows;
    const long slice_elem_offset = slice_row_offset * G.N;
    const int reduce_base = G.num_comp_sms + G.num_intra_ar_sms + G.num_inter_send_sms;
    const int red_id = blockIdx.x - reduce_base;
    const bool consumer_only = (G.num_comp_sms == 0 && G.num_intra_ar_sms == 0
                                && G.num_inter_send_sms == 0);

    // Recycled CTAs (comp/intra/send that re-enter after primary work) have no
    // role in the static path — return immediately.
    if (!consumer_only && red_id < 0) return;
    // CTAs beyond the dedicated reduce range also exit.
    if (red_id >= G.num_inter_reduce_store_sms) return;


    const int my_queue = (red_id >= 0 && red_id < G.num_remote_queues) ? red_id : -1;
    const int stride   = G.num_inter_reduce_store_sms;

    // Process statically owned chunks: poll flags directly, no CAS.
    for (int chunk_id = red_id; chunk_id < G.total_chunks; chunk_id += stride) {
        // Spin until both the intra-AR result (local_done_flag == 1) and the
        // remote RDMA arrival (remote_arrived_flag != 0) are visible.
        // The owning queue drainer continues servicing arrivals in the loop.
        if (tid == 0) {
            while (true) {
                if (my_queue >= 0) gemm_ar_drain_arrival_queue_publish_flags(G, my_queue);
                const bool remote_done =
                    gemm_ar_acquire_load_u32(G.remote_arrived_flag + chunk_id) != 0u;
                const bool local_done =
                    gemm_ar_acquire_load_u32(G.local_done_flag + chunk_id) == 1u;
                if (remote_done && local_done) {
                    break;
                }
                __nanosleep(50);
            }
        }
        __syncthreads();

        // Reduce C_local (intra-node AR result) + C_recv (remote node's data)
        // and publish the element-wise sum to C_final via NVLink multicast.
        int rb_in_slice, row_idx, col_start, tiles_this_chunk;
        gemm_ar_chunk_decode(
            chunk_id, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
            rb_in_slice, row_idx, col_start, tiles_this_chunk);

        for (int local_col = 0; local_col < tiles_this_chunk; ++local_col) {
            const int col_idx    = col_start + local_col;
            const int tile_id    = rb_in_slice * G.col_blocks + col_idx;
            const bf16* recv     = G.C_recv + (long)tile_id * fused_globals::TILE_ELEMS;
            constexpr int ELEMS_PER_VEC = 8;
            constexpr int VECS_PER_ROW  = fused_globals::COL_BLOCK / ELEMS_PER_VEC; // 32
            constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;        // 12
            const int vec_lane  = tid % VECS_PER_ROW;
            const int row_lane  = tid / VECS_PER_ROW;
            const int col_elem  = vec_lane * ELEMS_PER_VEC;
            const int global_col = col_idx * fused_globals::COL_BLOCK + col_elem;
            const long tile_col_offset = (long)col_idx * fused_globals::COL_BLOCK;
            for (int r = row_lane; r < fused_globals::ROW_BLOCK; r += ROWS_PER_WAVE) {
                const int local_row  = rb_in_slice * fused_globals::ROW_BLOCK + r;
                const int global_row = (int)(slice_row_offset + (long)local_row);
                const long global_elem =
                    slice_elem_offset
                    + (long)local_row * G.N
                    + tile_col_offset
                    + (long)col_elem;
                const uint4 lv = *reinterpret_cast<const uint4*>(G.C_local + global_elem);
                const uint4 rv = *reinterpret_cast<const uint4*>(
                    recv + r * fused_globals::COL_BLOCK + col_elem);
                __nv_bfloat162 o0 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.x),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
                __nv_bfloat162 o1 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.y),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
                __nv_bfloat162 o2 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.z),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
                __nv_bfloat162 o3 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.w),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
                uint4 ov;
                ov.x = *reinterpret_cast<unsigned int*>(&o0);
                ov.y = *reinterpret_cast<unsigned int*>(&o1);
                ov.z = *reinterpret_cast<unsigned int*>(&o2);
                ov.w = *reinterpret_cast<unsigned int*>(&o3);
                gemm_ar_publish_final_vec8(G, global_row, global_col, ov);
            }
        }

        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(G.published_chunks, 1u);
        }
        __syncthreads();
    }



    // After all statically owned chunks are published, spin until every chunk
    // across all reduce CTAs is published (needed before multicast finalization).
    // CTAs with red_id >= total_chunks (no owned chunks) also land here and
    // continue draining their arrival queue so RDMA flags get published promptly.
    if (tid == 0) {
        while ((int)gemm_ar_acquire_load_u32(G.published_chunks) < G.total_chunks) {
            if (my_queue >= 0) gemm_ar_drain_arrival_queue_publish_flags(G, my_queue);
            else __nanosleep(50);
        }
    }
    __syncthreads();
}

__device__ inline void shared_reduce_my_slice(
    const fused_globals& G
) {
    // Large chunk counts use work stealing to absorb RDMA arrival jitter.
    if (G.total_chunks >= 512) {
        shared_reduce_my_slice_worksteal(G
        );
        return;
    }

    // Small/medium chunk counts use static ownership to avoid CAS contention.
    if (G.total_chunks <= 64) {
        shared_reduce_my_slice_static(G
        );
        return;
    }

    const int tid = threadIdx.x;
    const long slice_row_offset = (long)G.dev_idx * G.slice_rows;
    const long slice_elem_offset = slice_row_offset * G.N;
    const int reduce_base = G.num_comp_sms + G.num_intra_ar_sms + G.num_inter_send_sms;
    const int red_id = blockIdx.x - reduce_base;
    const bool consumer_only = (G.num_comp_sms == 0 && G.num_intra_ar_sms == 0 && G.num_inter_send_sms == 0);
    // Recycled compute/intra/send CTAs only join the scan loop for tiny chunk
    // counts; otherwise they add cache pressure without enough work to claim.
    const bool allow_recycled = (G.total_chunks <= 8);
    if (!consumer_only && red_id < 0 && !allow_recycled) return;

    const int my_queue = (red_id >= 0 && red_id < G.num_remote_queues) ? red_id : -1;

    __shared__ int s_chunk_id;
    // Spread scan starting positions across CTAs using blockIdx.x.
    int scan_start = (int)((unsigned)blockIdx.x % (unsigned)max(1, G.total_chunks));

    while (true) {
        // Drain our arrival queue if we own one.
        if (my_queue >= 0 && tid == 0) {
            (void)gemm_ar_drain_arrival_queue_publish_flags(G, my_queue);
        }

        // Scan for a ready, unclaimed chunk. Claim via CAS(local_done_flag, 1→2).
        // Bounded scan window: scan at most CAS_SCAN_WIN entries per iteration
        // to reduce L2 contention from 8 CTAs all reading the same 256-entry
        // flag array. Each CTA starts from a diversified scan_start (seeded by
        // blockIdx.x) and advances by CAS_SCAN_WIN on miss, wrapping around
        // to cover the full array after total_chunks/CAS_SCAN_WIN misses.
        // scan_start is per-CTA local, so misses do not add atomic contention.
        constexpr int CAS_SCAN_WIN = 32;
        if (tid == 0) {
            s_chunk_id = -1;
            const int win = (G.total_chunks < CAS_SCAN_WIN)
                          ? G.total_chunks : CAS_SCAN_WIN;
            for (int i = 0; i < win; ++i) {
                const int c = (scan_start + i) % G.total_chunks;
                if (gemm_ar_acquire_load_u32(G.remote_arrived_flag + c) == 0u) continue;
                if (gemm_ar_acquire_load_u32(G.local_done_flag + c) != 1u) continue;
                // Both flags set; try to claim.
                const uint32_t old = atomicCAS(
                    reinterpret_cast<unsigned int*>(G.local_done_flag + c), 1u, 2u);
                if (old == 1u) {
                    s_chunk_id = c;
                    scan_start = (c + 1) % G.total_chunks;
                    break;
                }
            }
            // On miss, advance window to scan fresh territory next iteration.
            if (s_chunk_id < 0) {
                scan_start = (scan_start + win) % G.total_chunks;
            }
        }
        __syncthreads();

        if (s_chunk_id < 0) {
            if ((int)gemm_ar_acquire_load_u32(G.published_chunks) >= G.total_chunks) break;
            if (tid == 0) {
                gemm_ar_debug_maybe_dump_stuck(G, "recv_idle", red_id, scan_start, 1);
                __nanosleep(50);
            }
            __syncthreads();
            continue;
        }


        const int chunk_id = s_chunk_id;
        int rb_in_slice, row_idx, col_start, tiles_this_chunk;
        gemm_ar_chunk_decode(
            chunk_id, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
            rb_in_slice, row_idx, col_start, tiles_this_chunk);

        for (int local_col = 0; local_col < tiles_this_chunk; ++local_col) {
            const int col_idx = col_start + local_col;
            const int tile_id = rb_in_slice * G.col_blocks + col_idx;
            const bf16* recv = G.C_recv + (long)tile_id * fused_globals::TILE_ELEMS;
            constexpr int ELEMS_PER_VEC = 8;
            constexpr int VECS_PER_ROW = fused_globals::COL_BLOCK / ELEMS_PER_VEC; // 32
            constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;      // 12
            const int vec_lane = tid % VECS_PER_ROW;
            const int row_lane = tid / VECS_PER_ROW;
            const int col_elem = vec_lane * ELEMS_PER_VEC;
            const int global_col = col_idx * fused_globals::COL_BLOCK + col_elem;
            const long tile_col_offset = (long)col_idx * fused_globals::COL_BLOCK;
            for (int r = row_lane; r < fused_globals::ROW_BLOCK; r += ROWS_PER_WAVE) {
                const int local_row = rb_in_slice * fused_globals::ROW_BLOCK + r;
                const int global_row = (int)(slice_row_offset + (long)local_row);
                const long global_elem =
                    slice_elem_offset
                    + (long)local_row * G.N
                    + tile_col_offset
                    + (long)col_elem;
                const uint4 lv = *reinterpret_cast<const uint4*>(G.C_local + global_elem);
                const uint4 rv = *reinterpret_cast<const uint4*>(recv + r * fused_globals::COL_BLOCK + col_elem);
                __nv_bfloat162 o0 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&lv.x),
                                            *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
                __nv_bfloat162 o1 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&lv.y),
                                            *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
                __nv_bfloat162 o2 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&lv.z),
                                            *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
                __nv_bfloat162 o3 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&lv.w),
                                            *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
                uint4 ov;
                ov.x = *reinterpret_cast<unsigned int*>(&o0);
                ov.y = *reinterpret_cast<unsigned int*>(&o1);
                ov.z = *reinterpret_cast<unsigned int*>(&o2);
                ov.w = *reinterpret_cast<unsigned int*>(&o3);
                gemm_ar_publish_final_vec8(G, global_row, global_col, ov);
            }
        }

        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            const uint32_t published = atomicAdd(G.published_chunks, 1u) + 1u;
            gemm_ar_debug_log_transition(G, "published", chunk_id, published);
        }
        __syncthreads();

    }
}

__device__ inline void fused_inter_reduce_and_publish_sm(const fused_globals& G) {
    const int red_id = blockIdx.x - gemm_ar_inter_reduce_store_base(G);


    // --- Sub-phase A: chunk-owned recv-side reduction ---
    shared_reduce_my_slice(G
    );


    // --- Sub-phase B: finalize multicast publication once all chunks are done ---
    if (!G.defer_final_multicast_finish) {
        gemm_ar_finish_multicast_publication(G, red_id);
    }

    // --- Sub-phase C: no peer-read gather ---
}

// ============================================================================
// Fused kernel entry point
// ============================================================================

__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < G.num_comp_sms) {
        fused_comp_sm(G);
        shared_reduce_my_slice(G);
    } else if (blockIdx.x < G.num_comp_sms + G.num_intra_ar_sms) {
        fused_intra_ar_sm(G);
        shared_reduce_my_slice(G);
    } else if (blockIdx.x < G.num_comp_sms + G.num_intra_ar_sms + G.num_inter_send_sms) {
        fused_inter_send_sm(G);
        shared_reduce_my_slice(G);
    } else {
        fused_inter_reduce_and_publish_sm(G);
    }
    // Small shapes (defer_final_multicast_finish == 0) skip the epilogue launch;
    // do the iter-end barrier here. Crucially, only the dedicated reduce CTAs
    // are guaranteed to execute after gemm_ar_finish_multicast_publication() has
    // completed; the rest of the grid can reach the main-kernel tail earlier.
    // Using the whole grid here lets non-reduce CTAs reset arrival flags and
    // push the cross-node barrier before final multicast publication is fully
    // established, which is exactly the small-shape steady-state race.
    if (!G.defer_final_multicast_finish) {
        const int reduce_base = gemm_ar_inter_reduce_store_base(G);
        const bool is_reduce_cta =
            (blockIdx.x >= reduce_base &&
             blockIdx.x < reduce_base + G.num_inter_reduce_store_sms);
        if (is_reduce_cta) {
            // Reset arrival flags on-stream BEFORE the barrier — the barrier then
            // gates peer's next-iter send, so no clobber race with peer RDMA.
            gemm_ar_iter_end_reset_arrival_flags(G, reduce_base, G.num_inter_reduce_store_sms);
            if (!G.skip_final_barrier) {
                gemm_ar_hierarchical_xnode_barrier(G, reduce_base);
            }
        }
    }
}

__device__ inline void fused_epilogue_kernel(const fused_globals& G) {
    const int reduce_base = gemm_ar_inter_reduce_store_base(G);
    const bool is_reduce_cta =
        (blockIdx.x >= reduce_base && blockIdx.x < reduce_base + G.num_inter_reduce_store_sms);
    if (is_reduce_cta) {
        gemm_ar_finish_multicast_publication(G, blockIdx.x - reduce_base);
    }
    // 16-way iter-end barrier: the 8-way intra-node finish_multicast above
    // combined with a cross-node RDMA barrier here gives a true 16-rank sync
    // at iter boundary — enables honest steady-state benchmarking without
    // host-side dist.barrier/synchronize between iters.
    if (is_reduce_cta) {
        // Reset arrival flags on-stream BEFORE the barrier; gating peer's
        // next-iter RDMA writes behind our push means no clobber race.
        gemm_ar_iter_end_reset_arrival_flags(G, reduce_base, G.num_inter_reduce_store_sms);
        if (!G.skip_final_barrier) {
            gemm_ar_hierarchical_xnode_barrier(G, reduce_base);
        }
    }
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void gemm_ar_fused_epilogue_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_epilogue_kernel(G);
}

// Launch wrappers stay in this TU so the kernel bodies stay out of the .cuh.
void launch_fused_gemm_ar(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        gemm_ar_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    gemm_ar_fused_kernel_stub<<<config::NUM_BLOCKS, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}
void launch_fused_gemm_ar_epilogue(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        gemm_ar_fused_epilogue_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    gemm_ar_fused_epilogue_kernel_stub<<<config::NUM_BLOCKS, config::NUM_THREADS,
                                         dynamic_shared_memory, stream>>>(G);
}

}  // namespace gemm_ar_multinode

#include "operators/gemm_ar/session.cuh"
