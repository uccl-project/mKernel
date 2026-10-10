#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#endif

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/atomic_u32.cuh"
#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/ag_gemm/ag_gemm_warp_specialized.cuh"

// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
#include "memory/tk_ops_group_util_util.cuh"
// clang-format on

using namespace kittens;

namespace ag_gemm_warp_specialized {

namespace {

// Process-lifetime resources, one set per local CUDA device. This extension
// uses one host process per GPU, and launches are serialized by that process.
struct ACopyPipelineState {
    cudaStream_t stream = nullptr;
    cudaEvent_t main_pre_event = nullptr;
    cudaEvent_t copy_completion = nullptr;
    bool initialized = false;
};

inline ACopyPipelineState& get_A_copy_state(int dev_idx) {
    static ACopyPipelineState states[INTRA_NUM_DEVICES];
    ACopyPipelineState& state = states[dev_idx];
    if (!state.initialized) {
        // The first call may happen inside a CUDA graph capture. None of these
        // calls enqueue stream work, so relax capture mode during resource setup.
        cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeRelaxed;
        MKERNEL_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&capture_mode));
        MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&state.main_pre_event, cudaEventDisableTiming));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&state.copy_completion, cudaEventDisableTiming));
        MKERNEL_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&capture_mode));
        state.initialized = true;
    }
    return state;
}

// tcgen05 MMA with the CTA group taken from the config. mm2_AB / mma2_AB are
// hardwired to a 2-CTA group, so spell out the ncta template argument instead.
// B_tile is stored [N, K] (see fused_globals::B_tile), so this is really an
// ABt-shaped MMA -- transpose::T on the B operand tells the tensor core to
// read the [N, K] tile as B^T, producing the same A @ B result.
template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mm_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 0, NUM_CTA>(d, a, b, sem);
}

template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mma_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 1, NUM_CTA>(d, a, b, sem);
}

template <typename ST>
__device__ inline void store_async_3d(
    const CUtensorMap& map, const ST& src, int column, int row, int device) {
    uint64_t map_ptr = reinterpret_cast<uint64_t>(&map);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(map_ptr), "r"(src_ptr), "r"(column), "r"(row), "r"(device)
        : "memory");

    kittens::tma::store_commit_group();
}

}  // namespace

// traverse the grid in a snake like pattern to raise L2 cache reuse
// https://github.com/HazyResearch/ThunderKittens/blob/0230013a72b51338a137b50f69538ec69d4d4675/include/common/util.cuh#L367
template <int SUPERGROUP_WIDTH = 5>
__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int num_rows,
                                                                   int num_cols,
                                                                   int tile_idx) {
    static_assert(SUPERGROUP_WIDTH > 0, "SUPERGROUP_SIZE must be greater than 0");
    const int supergroup_numel = num_rows * SUPERGROUP_WIDTH;
    const int supergroup_idx = tile_idx / supergroup_numel;
    const int supersection_cols = (num_cols / SUPERGROUP_WIDTH) * SUPERGROUP_WIDTH;
    const int supersection_numel = num_rows * supersection_cols;
    const int finalsection_cols = num_cols - supersection_cols;
    int row_idx, col_idx;
    if (tile_idx < supersection_numel) {
        row_idx = (tile_idx % supergroup_numel) / SUPERGROUP_WIDTH;
        col_idx = supergroup_idx * SUPERGROUP_WIDTH + tile_idx % SUPERGROUP_WIDTH;
    } else {
        const int remainder_task_id = tile_idx - supersection_numel;
        row_idx = remainder_task_id / finalsection_cols;
        col_idx = supersection_cols + remainder_task_id % finalsection_cols;
    }
    return {(supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx};
};

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA,
          int SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS,
          AgStrategy STRATEGY>
__device__ __forceinline__ void ag_gemm_warp_specialized(
    const fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>& G) {
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>;

    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A[G.dev_idx].template prefetch_tma<typename fg::A_tile>();
        G.B.template prefetch_tma<typename fg::B_tile>();
        G.C.prefetch_tma();
    }

    const int cluster_idx = blockIdx.x / fg::NUM_CLUSTERS;
    const int local_m = G.A.rows();
    const int row_tiles_per_device = (local_m + fg::ROW_BLOCK - 1) / fg::ROW_BLOCK;
    const int cluster_rows_per_device =
        (row_tiles_per_device + fg::NUM_CLUSTERS * fg::CONSUMER_WARPS - 1) /
        (fg::NUM_CLUSTERS * fg::CONSUMER_WARPS);
    const int num_comp_clusters = fg::NUM_BLOCKS / fg::NUM_CLUSTERS;

    // round up to the nearest multiple of the COL_BLOCK
    const int num_col_tiles = (G.N + _COL_BLOCK - 1) / _COL_BLOCK;
    const int cluster_tiles_per_device = cluster_rows_per_device * num_col_tiles;
    const int total_num_tiles = cluster_tiles_per_device * fg::NUM_DEVICES;

    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    typename fg::pipeline_inputs(&inputs_smem)[fg::PRODUCER_CONSUMER_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::pipeline_inputs, fg::PRODUCER_CONSUMER_PIPELINE_STAGES>();
    typename fg::C_tile(&C_smem)[fg::EPILOGUE_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::C_tile, fg::EPILOGUE_PIPELINE_STAGES>();

    __shared__ semaphore tma_load[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];
    __shared__ semaphore mma_finish[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];

    __shared__ semaphore epilogue_ready[fg::NUM_TMEM_SLOTS];
    __shared__ semaphore epilogue_tmem_finished[fg::NUM_TMEM_SLOTS];

    __shared__ semaphore tmem_allocated;
    __shared__ semaphore tmem_finished;

    __shared__ uint32_t tmem_addr;
    tensor_allocator<1, fg::NUM_CLUSTERS> tm_alloc{};

    uint32_t phasebits = fg::PHASE_BITS_INIT;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fg::PRODUCER_CONSUMER_PIPELINE_STAGES; i++) {
            // tma finish has to be broadcasted to mma warp
            init_semaphore(tma_load[i], 0, fg::NUM_CLUSTERS);
            // mma warp will broadcast finish
            init_semaphore(mma_finish[i], 0, fg::CONSUMER_WARPS);
        }

#pragma unroll
        for (int i = 0; i < fg::NUM_TMEM_SLOTS; i++) {
            init_semaphore(epilogue_ready[i], 0, 1);

            // tmem finish has to be broadcasted back
            init_semaphore(epilogue_tmem_finished[i], WARPGROUP_WARPS * fg::NUM_CLUSTERS);
        }

        // every CTA in the cluster arrives here, itself included
        init_semaphore(tmem_allocated, 1);
        init_semaphore(tmem_finished, fg::NUM_CTA);
    }

    // flush to ensure the mbarriers are visible
    everyone::tma::cluster::arrive_aligned();

    auto load = [&](int tile_row_idx, int tile_col_idx, int target_device, int& input_stage_id) {
        const int actual_target_device = (target_device + G.dev_idx) % fg::NUM_DEVICES;
        const bool is_local = actual_target_device == G.dev_idx;

        // Multicast push polls the source rank's ordinary ready allocation via
        // its imported peer pointer. Pull polls a locally committed flag after
        // each copy engine transfer into this rank's gathered buffer.
        if (!is_local) {
            if constexpr (STRATEGY == AgStrategy::MULTICAST_PUSH) {
                while (comm::atomic_u32::acquire_load_sys(
                           G.A_copy_ready[actual_target_device].raw_ptr) < fg::A_copy_epoch) {
                    __nanosleep(16);
                }
            } else {
                while (comm::atomic_u32::acquire_load_gpu(G.A_copy_ready[G.dev_idx].raw_ptr +
                                                          actual_target_device) <
                       fg::A_copy_epoch) {
                    __nanosleep(16);
                }
            }
        }

        // tile_row_idx is consumer 0's row tile; each additional consumer
        // warp c owns the row tile NUM_CLUSTERS further along -- see the
        // matching offset in the consumer/epilogue loops below.
        auto a_tile_row_idx_for = [&](int c) { return tile_row_idx + c * fg::NUM_CLUSTERS; };

        for (int iter_k = 0; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile(&A_smem)[fg::CONSUMER_WARPS] = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(mma_finish[input_stage_id], (phasebits >> 0 & 0b1));

            if constexpr (_NUM_CTA == 2) {
                tma::cluster::expect_bytes(
                    tma_load[input_stage_id],
                    sizeof(fg::A_tile) * fg::CONSUMER_WARPS + sizeof(fg::B_tile),
                    0);

                // B_tile is [N, K], so the TMA coordinate is (n_tile, k_tile),
                // not (k_tile, n_tile).
                tma::cluster::load_async(B_smem,
                                         G.B,
                                         {tile_col_idx * fg::NUM_CLUSTERS + cta_rank, iter_k},
                                         tma_load[input_stage_id],
                                         (uint16_t)(1 << cta_rank),
                                         0);

#pragma unroll
                for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                    tma::cluster::load_async(A_smem[c],
                                             G.A[G.dev_idx],
                                             {actual_target_device, a_tile_row_idx_for(c), iter_k},
                                             tma_load[input_stage_id],
                                             (uint16_t)(1 << cta_rank),
                                             0);
                }

            } else {
                // A 1-CTA cluster has nothing to multicast to and nothing to
                // map: the barrier is this CTA's own. The cluster overloads
                // would still emit cta_group::2.multicast::cluster loads, so
                // take the plain CTA-scope TMA path instead.
                tma::expect_bytes(tma_load[input_stage_id],
                                  sizeof(fg::A_tile) * fg::CONSUMER_WARPS + sizeof(fg::B_tile));

                // B_tile is [N, K], so the TMA coordinate is (n_tile, k_tile),
                // not (k_tile, n_tile).
                tma::load_async(B_smem,
                                G.B,
                                {tile_col_idx * fg::NUM_CLUSTERS + cta_rank, iter_k},
                                tma_load[input_stage_id]);

#pragma unroll
                for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                    tma::load_async(A_smem[c],
                                    G.A[G.dev_idx],
                                    {actual_target_device, a_tile_row_idx_for(c), iter_k},
                                    tma_load[input_stage_id]);
                }
            }

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 0b1;
            }
        }
    };
    auto consume = [&](typename fg::C_tt_tile* tmem,
                       int& input_stage_id,
                       int& epilogue_stage_id,
                       const int consumer_id) {
        const int consumer_semaphore_idx = consumer_id + 2;
        wait(epilogue_tmem_finished[epilogue_stage_id * fg::CONSUMER_WARPS + consumer_id],
             (phasebits >> consumer_semaphore_idx) & 0b1);

        {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            mm_ABt_ncta<fg::NUM_CTA>(
                tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int iter_k = 1; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            mma_ABt_ncta<fg::NUM_CTA>(
                tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<fg::NUM_CLUSTERS>(
            epilogue_ready[epilogue_stage_id * fg::CONSUMER_WARPS + consumer_id]);
        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;

        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << consumer_semaphore_idx);
        }
    };

    auto epilogue = [&](int target_device,
                        int tile_row_idx,
                        int tile_col_idx,
                        typename fg::C_tt_tile* tmem,
                        const int consumer_id,
                        int& epilogue_stage_id,
                        int& epilogue_transfer_stage_id,
                        bool is_last_tile) {
        constexpr int C_CHUNK_COLS = fg::COL_BLOCK / fg::C_TILE_DIVISOR;
        rt_bf<fg::ROW_BLOCK / WARPGROUP_WARPS, C_CHUNK_COLS> c_reg[fg::C_TILE_DIVISOR];

        // Mirrors consume()'s consumer_semaphore_idx (bits [2, 2+CONSUMER_WARPS)):
        // this is the other end of the same per-consumer double-buffer, using
        // the next CONSUMER_WARPS bits so the two phases never alias.
        const int epilogue_semaphore_idx = 2 + fg::CONSUMER_WARPS + consumer_id;
        wait(epilogue_ready[epilogue_stage_id * fg::CONSUMER_WARPS + consumer_id],
             (phasebits >> epilogue_semaphore_idx) & 0b1);

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            warpgroup::load_async(
                c_reg[i],
                // TODO: review this indexing
                tmem[epilogue_stage_id].template subtile<tt<float, fg::ROW_BLOCK, C_CHUNK_COLS>>(
                    i * C_CHUNK_COLS));
        }

        tensor_load_wait();

        if (elect_warp_leader()) {
            if constexpr (_NUM_CTA == 2) {
                tma::cluster::arrive(
                    epilogue_tmem_finished[epilogue_stage_id * fg::CONSUMER_WARPS + consumer_id],
                    0);
            } else {
                arrive(epilogue_tmem_finished[epilogue_stage_id]);
            }
        }

        // Only the last consumer warp's pass over the final tile should
        // signal PDL -- firing it once per consumer would be redundant.
        if (is_last_tile && consumer_id == fg::CONSUMER_WARPS - 1) {
            warpgroup::sync(1);
            pdl::arrive();
        }

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fg::EPILOGUE_PIPELINE_STAGES - 1>();
            warpgroup::sync(1);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[epilogue_transfer_stage_id], c_reg[i]);
            warpgroup::sync(1);

            if (warpgroup::laneid() == 0) {
                const int chunk = tile_col_idx * fg::C_TILE_DIVISOR + i;
                const int column = chunk * C_CHUNK_COLS;
                const int row = tile_row_idx * fg::ROW_BLOCK;
                store_async_3d(
                    G.C.map, C_smem[epilogue_transfer_stage_id], column, row, target_device);
            }

            epilogue_transfer_stage_id =
                (epilogue_transfer_stage_id + 1) % fg::EPILOGUE_PIPELINE_STAGES;
        }

        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (0b1 << epilogue_semaphore_idx);
        }
    };

    if (warpgroup_id >= fg::EPILOGUE_WARPGROUPS) {
        if (warp_id == 4) {
            pdl::wait();
            everyone::tma::cluster::wait();

            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int tile_id = cluster_idx; tile_id < total_num_tiles;
                     tile_id += num_comp_clusters) {
                    // work should be partitioned based on the rank tile size. M = GLOBAL_M / TP
                    auto [local_row_id, tile_col_idx] = calculate_tile_idx<SUPERGROUP_WIDTH>(
                        cluster_rows_per_device, num_col_tiles, tile_id % cluster_tiles_per_device);

                    int target_device = tile_id / cluster_tiles_per_device;

                    load(local_row_id * fg::NUM_CLUSTERS * fg::CONSUMER_WARPS + cta_rank,
                         tile_col_idx,
                         target_device,
                         input_stage_id);
                }
            }

            pdl::arrive();
        } else if (warp_id < 5 + _NUM_CONSUMER_WARPS) {
            int input_stage_id = 0;
            int epilogue_stage_id = 0;
            typename fg::C_tt_tile tmem[fg::TMEM_PIPELINE_STAGES];
            // warp 4 is the producer, so the first consumer warp is warp 5.
            const int consumer_warp_id = warp_id - 5;

            // wait for PDL
            pdl::wait();
            everyone::tma::cluster::wait();

            if (warp_id == 5) {
                tm_alloc.provision(tmem_addr);
                if (elect_warp_leader()) {
                    arrive(tmem_allocated);
                }
                tm_alloc.set_addr(tmem_addr);
            } else {
                wait(tmem_allocated, 0);
                tm_alloc.set_addr(tmem_addr);
            }

            if (cta_rank == 0 && elect_warp_leader()) {
#pragma unroll
                for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
                    // Each consumer warp owns its own contiguous
                    // TMEM_PIPELINE_STAGES * COL_BLOCK column range so the
                    // two consumers' pipelines never alias in tensor memory.
                    tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(
                        (consumer_warp_id * fg::TMEM_PIPELINE_STAGES + i) * fg::COL_BLOCK);
                }

                for (int tile_id = cluster_idx; tile_id < total_num_tiles;
                     tile_id += num_comp_clusters) {
                    consume(tmem, input_stage_id, epilogue_stage_id, consumer_warp_id);
                }
            }

            pdl::arrive();
        }
    } else {
        // Each consumer warp has its own TMEM_PIPELINE_STAGES cycle (they
        // produce independently), so track their stage counters separately;
        // the SMEM transfer FIFO is a single shared resource, so one counter
        // for it is correct regardless of consumer count.
        int epilogue_stage_id[fg::CONSUMER_WARPS] = {};
        int epilogue_transfer_stage_id = 0;
        typename fg::C_tt_tile tmem[fg::NUM_TMEM_SLOTS];

        // wait for PDL and tmem
        everyone::tma::cluster::wait();
        wait(tmem_allocated, 0);

#pragma unroll
        for (int i = 0; i < fg::NUM_TMEM_SLOTS; i++) {
            // Matches the consumer warps' allocation: consumer c's stages
            // live at columns [c * TMEM_PIPELINE_STAGES, (c+1) * TMEM_PIPELINE_STAGES).
            tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(i * fg::COL_BLOCK);
        }

        for (int tile_id = cluster_idx; tile_id < total_num_tiles; tile_id += num_comp_clusters) {
            // work should be partitioned based on the rank tile size. M = GLOBAL_M / TP
            auto [local_row_id, tile_col_idx] = calculate_tile_idx<SUPERGROUP_WIDTH>(
                cluster_rows_per_device, num_col_tiles, tile_id % cluster_tiles_per_device);
            const int target_device =
                (tile_id / cluster_tiles_per_device + G.dev_idx) % fg::NUM_DEVICES;

#pragma unroll
            for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                // Matches the load()/consume() row offset: consumer c owns
                // the row tile NUM_CLUSTERS further along than consumer 0.
                const int local_cta_row =
                    (local_row_id * fg::CONSUMER_WARPS + c) * fg::NUM_CLUSTERS + cta_rank;
                epilogue(target_device,
                         local_cta_row,
                         tile_col_idx,
                         &tmem[c * fg::TMEM_PIPELINE_STAGES],
                         c,
                         epilogue_stage_id[c],
                         epilogue_transfer_stage_id,
                         tile_id + num_comp_clusters >= total_num_tiles);
            }
        }

        // wait for store to complete before deallocation of tmem
        if (warpgroup::laneid() == 0) {
            dist::tma::store_async_wait();
        }

        tensor_before_thread_sync();
        group<fg::EPILOGUE_WARPS>::sync(1);

        if (group<fg::EPILOGUE_WARPS>::warpid() == 0) {
            if (elect_warp_leader()) {
                if constexpr (_NUM_CTA == 1) {
                    arrive(tmem_finished);
                } else {
#pragma unroll
                    for (int peer = 0; peer < fg::NUM_CTA; peer++) {
                        tma::cluster::arrive(tmem_finished, peer);
                    }
                }
            }
            // Only reach here if we finish with our tmem. Other party as well
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA,
          int SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS,
          AgStrategy STRATEGY>
__global__ __cluster_dims__(
    fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(
        fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>::NUM_THREADS,
        1) void fused_kernel_stub(const __grid_constant__ fused_globals<_ROW_BLOCK,
                                                                        _COL_BLOCK,
                                                                        _NUM_CTA,
                                                                        _NUM_CONSUMER_WARPS> G) {
    ag_gemm_warp_specialized<_ROW_BLOCK,
                             _COL_BLOCK,
                             _NUM_CTA,
                             SUPERGROUP_WIDTH,
                             _NUM_CONSUMER_WARPS,
                             STRATEGY>(G);
}

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA,
          int SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS,
          AgStrategy STRATEGY>
inline void launch_ag_gemm_warp_specialized(
    const fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>& G) {
    cudaStream_t stream = G.stream;

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!G.stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
#endif
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>;
    static_assert(fg::SMEM_FITS, "SMEM allocation too large for this config");
    ACopyPipelineState& copy_state = get_A_copy_state(G.dev_idx);

    // NOTE: must prepare and the caller's cross-rank barrier precede this fork on stream.
    MKERNEL_CUDACHECK(cudaEventRecord(copy_state.main_pre_event, stream));
    MKERNEL_CUDACHECK(cudaStreamWaitEvent(copy_state.stream, copy_state.main_pre_event, 0));

    const size_t shard_elements = static_cast<size_t>(G.A.rows()) * G.K;
    const size_t shard_bytes = shard_elements * sizeof(typename fg::A_local_tensor::dtype);

    if constexpr (STRATEGY == AgStrategy::MULTICAST_PUSH) {
        auto* dst = G.A.mc_ptr + static_cast<size_t>(G.dev_idx) * shard_elements;
        const auto* src = G.A[G.dev_idx].raw_ptr + static_cast<size_t>(G.dev_idx) * shard_elements;
        MKERNEL_CUDACHECK(
            cudaMemcpyAsync(dst, src, shard_bytes, cudaMemcpyDeviceToDevice, copy_state.stream));
        MKERNEL_CUCHECK(
            cuStreamWriteValue32(reinterpret_cast<CUstream>(copy_state.stream),
                                 reinterpret_cast<CUdeviceptr>(G.A_copy_ready[G.dev_idx].raw_ptr),
                                 fg::A_copy_epoch,
                                 CU_STREAM_WRITE_VALUE_DEFAULT));
    } else {
        // Stage one complete shard per remote device in the same ring order
        // used by the persistent kernel. The local shard is read directly.
#pragma unroll
        for (int distance = 1; distance < fg::NUM_DEVICES; ++distance) {
            const int peer = (G.dev_idx + distance) % fg::NUM_DEVICES;
            auto* dst = G.A[G.dev_idx].raw_ptr + static_cast<size_t>(peer) * shard_elements;
            const auto* src = G.A[peer].raw_ptr + static_cast<size_t>(peer) * shard_elements;

            MKERNEL_CUDACHECK(cudaMemcpyAsync(
                dst, src, shard_bytes, cudaMemcpyDeviceToDevice, copy_state.stream));

            // Keep the default pre-write barrier: it publishes the copied shard
            // before the completion epoch. The kernel-side load only needs GPU
            // scope because it reads a flag and payload resident on this device.
            MKERNEL_CUCHECK(cuStreamWriteValue32(
                reinterpret_cast<CUstream>(copy_state.stream),
                reinterpret_cast<CUdeviceptr>(G.A_copy_ready[G.dev_idx].raw_ptr + peer),
                fg::A_copy_epoch,
                CU_STREAM_WRITE_VALUE_DEFAULT));
        }
    }

    constexpr int smem_size = fg::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = fg::NUM_THREADS;
    constexpr int grid = fg::NUM_BLOCKS;

    auto this_kernel = fused_kernel_stub<_ROW_BLOCK,
                                         _COL_BLOCK,
                                         _NUM_CTA,
                                         SUPERGROUP_WIDTH,
                                         _NUM_CONSUMER_WARPS,
                                         STRATEGY>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchAttribute pdl_attr = {};
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = num_threads;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;
    launch_config.attrs = &pdl_attr;
    launch_config.numAttrs = 1;

    MKERNEL_CUDACHECK(cudaLaunchKernelEx(&launch_config, this_kernel, G));

    // Have the copy stream join the main stream again for graph capture.
    MKERNEL_CUDACHECK(cudaEventRecord(copy_state.copy_completion, copy_state.stream));
    MKERNEL_CUDACHECK(cudaStreamWaitEvent(stream, copy_state.copy_completion));

    MKERNEL_CUDACHECK(cudaGetLastError());
}
};  // namespace ag_gemm_warp_specialized

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include "operators/ag_gemm/ag_gemm_warp_specialized_session.cuh"
#endif
