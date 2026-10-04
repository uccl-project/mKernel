#pragma once

#include <cassert>
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/parallel_buffer.cuh"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "operators/ag_gemm/ag_gemm_timing.cuh"

#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"

namespace ag_gemm_warp_specialized {

// CTAs per cluster, i.e. the tcgen05 MMA CTA group. 2 splits COL_BLOCK across
// the pair so each CTA stages half the B tile; 1 gives every CTA its own MMA.
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = 1;

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
struct fused_globals;

// Number of tile columns visited before the snake pattern steps to the next
// supergroup; wider supergroups trade B-tile reuse for A-tile reuse in L2.
static constexpr int DEFAULT_SUPERGROUP_WIDTH = 5;

// Optional host-side diagnostics for the submission bubble between the eager
// and late copy batches. Times use the host steady clock and therefore are not
// directly aligned with the device's %globaltimer timestamps.
struct HostLaunchTiming {
    uint64_t attribute_ns = 0;
    uint64_t launch_ns = 0;
    uint64_t submission_gap_ns = 0;
    bool attribute_configured_now = false;
    bool launch_was_captured = false;
#ifdef PROFILE_TIMINGS
    // Timed events do not consume an SM. They replace the one-thread marker
    // kernels that could be starved behind the persistent GEMM and delay the
    // memcpy they were supposed to observe.
    cudaEvent_t memcpy_anchor_event = nullptr;
    cudaEvent_t memcpy_begin_events[INTRA_NUM_DEVICES] = {};
    cudaEvent_t memcpy_complete_events[INTRA_NUM_DEVICES] = {};
    bool memcpy_events_initialized = false;
#endif
};

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int SUPERGROUP_WIDTH = DEFAULT_SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
void launch_ag_gemm_warp_specialized(
    const fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>& G,
    void* timing_records = nullptr,
    HostLaunchTiming* host_timing = nullptr);

// for M < 512, this should be 128
template <int _ROW_BLOCK, int _COL_BLOCK, int _NUM_CTA, int _NUM_CONSUMER_WARPS>
struct fused_globals {
    // config items
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;

    // not sure if I want to use a warp specialized or sm specialized strategy yet
    static constexpr int NUM_BLOCKS = AG_GEMM_NUM_BLOCKS;
    static constexpr int CONSUMER_WARPS = _NUM_CONSUMER_WARPS;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = 1;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;
    // CTAs per cluster. 2-CTA MMA is preferred for shapes that divide cleanly;
    // NUM_CLUSTERS is the cluster dimension the kernel launches with.
    static constexpr int NUM_CTA = _NUM_CTA;
    static constexpr int NUM_CLUSTERS = NUM_CTA;
    static_assert(NUM_CTA == 1 || NUM_CTA == 2, "tcgen05 only has 1- and 2-CTA MMA groups");
    static_assert(NUM_BLOCKS % NUM_CTA == 0, "NUM_BLOCKS must be a whole number of clusters");
    static_assert(_COL_BLOCK % NUM_CTA == 0, "COL_BLOCK must split evenly across the cluster");
    static constexpr int NUM_THREADS = (CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS) * 32;

    // this is pipelining along the reduction dimension
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = []() {
        if constexpr (_NUM_CONSUMER_WARPS == 2) {
            return 4;
        } else if constexpr (_NUM_CTA == 1 || _COL_BLOCK == 256) {
            return 6;
        } else {
            return 7;
        }
    }();
    // this is pipelining among different MMAs
    static constexpr int TMEM_PIPELINE_STAGES =
        kittens::MAX_TENSOR_COLS / _COL_BLOCK / CONSUMER_WARPS;
    static constexpr int NUM_TMEM_SLOTS = TMEM_PIPELINE_STAGES * CONSUMER_WARPS;
    // this is the number of epilogue stages that can be in flight at any time
    static constexpr int EPILOGUE_PIPELINE_STAGES = _COL_BLOCK == 128 ? 3 : 2;
    // this is the number of partitions for the epilogue tile in SMEM
    static constexpr int C_TILE_DIVISOR = 4;

    static constexpr int ROW_BLOCK = _ROW_BLOCK;
    static constexpr int COL_BLOCK = _COL_BLOCK;
    static constexpr int RED_BLOCK = 64;

    using A_tile = kittens::st_bf<ROW_BLOCK, RED_BLOCK>;
    // B is stored [N, K] (not [K, N]) so the reduction dimension is contiguous
    // in HBM -- the tile shape here mirrors that: rows are the N-chunk, cols
    // are the K-chunk. The MMA call reads it back with transpose::T, and the
    // TMA load coordinate is (n_tile, k_tile) to match.
    using B_tile = kittens::st_bf<COL_BLOCK / NUM_CLUSTERS, RED_BLOCK>;

    using C_tt_tile = kittens::tt<float, ROW_BLOCK, COL_BLOCK>;
    // for smem staging -- keep at least
    using C_tile = kittens::st_bf<ROW_BLOCK, COL_BLOCK / C_TILE_DIVISOR>;

    static constexpr int MAX_DYNAMIC_SHARED_MEMORY = 227 * 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (sizeof(A_tile) * CONSUMER_WARPS + sizeof(B_tile)) * PRODUCER_CONSUMER_PIPELINE_STAGES +
        sizeof(C_tile) * EPILOGUE_PIPELINE_STAGES + 1024;
    // Deliberately not a static_assert: the tuner instantiates fused_globals
    // for every candidate so it can ask which ones fit. The hard check lives
    // in launch_ag_gemm_warp_specialized, so nothing oversized can actually launch.
    static constexpr bool SMEM_FITS = DYNAMIC_SHARED_MEMORY <= MAX_DYNAMIC_SHARED_MEMORY;

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, NUM_DEVICES, -1, -1, A_tile>;
    using A_distributed_tensor = dist::distributed_tensor<A_local_tensor, NUM_DEVICES, true>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;

    // NOTE: TK rounds up the tensor map to the nearest multiple of the swizzle
    // we dont want that behavior as that would give us OOB writes
    // The solution is therefore to create our own tensormap, with the same
    // swizzle as C_tile, while maintaining correctness
    struct C_local_tensor {
        comm::bf16* data;
        CUtensorMap map;

        __device__ inline void prefetch_tma() const { dist::tma::prefetch_tensormap(&map); }
    };

    A_distributed_tensor A;
    B_local_tensor B;
    C_local_tensor C;

    // Copy-engine completion is published into local HBM.
    uint32_t* A_copy_ready;
    static constexpr uint32_t A_copy_epoch = 1;

#ifdef PROFILE_TIMINGS
    TimingRecord* timings;
#endif

    int dev_idx;
    int M;
    int N;
    int K;

    cudaStream_t stream = nullptr;

    struct pipeline_inputs {
        A_tile A[_NUM_CONSUMER_WARPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    /*
     * bit 0: TMA producer -- starts with 1 (PRODUCER WARP)
     * bit 1: TMA consumer -- starts with 0 (CONSUMER WARP)
     * bits [2, 2+CONSUMER_WARPS): TMEM producer, one per consumer warp --
     *   starts with 1 (CONSUMER WARP), since there is no prior epilogue
     *   readout for the first pipeline fill to wait on
     * bits [2+CONSUMER_WARPS, 2+2*CONSUMER_WARPS): TMEM consumer, one per
     *   consumer warp -- starts with 0 (EPILOGUE WARP)
     */
    static constexpr int TMA_PRODUCER_BIT = 0b1;
    static constexpr int TMA_CONSUMER_BITS = 0b000;
    static constexpr int TMEM_PRODUCER_BITS = ((1 << CONSUMER_WARPS) - 1) << 2;
    static constexpr int TMEM_CONSUMER_BITS = 0;
    static constexpr int PHASE_BITS_INIT =
        TMA_PRODUCER_BIT | TMA_CONSUMER_BITS | TMEM_PRODUCER_BITS | TMEM_CONSUMER_BITS;
};

// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2593r1.html#valid-workaround
// to allow the else branch of template deductions to accept static_assert(0)
template <typename>
inline constexpr bool always_false_v = false;

template <typename DistributedTensor,
          typename LocalTensor,
          int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
__host__ inline fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>
ag_gemm_warp_specialized_make_globals(DistributedTensor& A,
                                      const LocalTensor& B,
                                      LocalTensor& C,
                                      int dev_idx,
                                      int M,
                                      int N,
                                      int K,
                                      cudaStream_t stream) {
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>;

    // create the C tensor map
    // NOTE: requriement is that M % NUM_DEVICES == 0
    const int local_m = M / fg::NUM_DEVICES;
    typename fg::C_local_tensor C_tensor;

    uint64_t global_dim[3] = {
        static_cast<uint64_t>(N),        // columns
        static_cast<uint64_t>(local_m),  // rows per device
        fg::NUM_DEVICES,                 // devices
    };

    uint64_t global_stride[2] = {
        N * sizeof(comm::bf16),
        local_m * N * sizeof(comm::bf16),
    };

    uint32_t box_dim[3] = {
        fg::C_tile::cols,
        fg::C_tile::rows,
        1,
    };

    uint32_t element_stride[3] = {1, 1, 1};

    // Must match the swizzle TK chose for the smem C_tile (128B for 64 bf16
    // columns, 64B for 32), else the TMA store decodes the staged tile wrongly.
    constexpr CUtensorMapSwizzle c_swizzle = fg::C_tile::swizzle_bytes == 128
        ? CU_TENSOR_MAP_SWIZZLE_128B
        : fg::C_tile::swizzle_bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                          : CU_TENSOR_MAP_SWIZZLE_32B;

    // currently, we want to accomodate both bf16* and at::Tensors
    if constexpr (std::is_same_v<LocalTensor, comm::bf16*> &&
                  dist::RawDistributedMulticastTensorLike<DistributedTensor, comm::bf16>) {
        C_tensor.data = C;
        cuTensorMapEncodeTiled(&C_tensor.map,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               3,
                               C_tensor.data,
                               global_dim,
                               global_stride,
                               box_dim,
                               element_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               c_swizzle,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        return {
            .A = ::dist::make_dbuf<typename fg::A_distributed_tensor>(
                reinterpret_cast<uint64_t>(A.mc),
                reinterpret_cast<uint64_t*>(A.uc_ptrs),
                1,
                fg::NUM_DEVICES,
                M / fg::NUM_DEVICES,
                K),
            .B = ::dist::make_local_tensor<typename fg::B_local_tensor>(
                reinterpret_cast<uint64_t>(B), 1, 1, N, K),
            .C = C_tensor,
            .A_copy_ready = nullptr,
#ifdef PROFILE_TIMINGS
            .timings = nullptr,
#endif
            .dev_idx = dev_idx,
            .M = M,
            .N = N,
            .K = K,
            .stream = stream,
        };
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    } else if constexpr (std::is_same_v<LocalTensor, at::Tensor> &&
                         std::is_same_v<DistributedTensor, dist::ParallelBuffer>) {
        C_tensor.data = reinterpret_cast<comm::bf16*>(C.data_ptr());
        cuTensorMapEncodeTiled(&C_tensor.map,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               3,
                               C_tensor.data,
                               global_dim,
                               global_stride,
                               box_dim,
                               element_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               c_swizzle,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        return {
            .A = ::dist::distributed_tensor_from_buffer<typename fg::A_distributed_tensor>(A),
            .B = ::dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
            .C = C_tensor,
            .A_copy_ready = nullptr,
#ifdef PROFILE_TIMINGS
            .timings = nullptr,
#endif
            .dev_idx = dev_idx,
            .M = M,
            .N = N,
            .K = K,
            .stream = stream,
        };
#endif
    } else {
        static_assert(
            always_false_v<LocalTensor>,
            "LocalTensor must be either __nv_bfloat16* or at::Tensor, while DistributedTensor must "
            "either satisfy dist::RawDistributedMulticastTensorLike or ParallelBuffer");
    }
}

template <typename DistributedTensor, typename LocalTensor>
void entrypoint(DistributedTensor& A,
                const LocalTensor& B,
                LocalTensor& C,
                int M = -1,
                int N = -1,
                int K = -1,
                int dev_idx = -1,
                cudaStream_t stream = nullptr,
                void* timing_records = nullptr,
                HostLaunchTiming* host_timing = nullptr) {
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    dev_idx = dev_idx == -1 ? A.local_rank_ : dev_idx;
    M = M == -1 ? C.size(0) * C.size(1) : M;
    N = N == -1 ? B.size(0) : N;
    K = K == -1 ? B.size(1) : K;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                "A.local_world_size must match the compiled INTRA_NUM_DEVICES");
#endif

    // TODO: this only works for TP == 8
    constexpr int MIN_LARGE_GEMM_N = 6288;

    // use size of N to check which projection is being done
    if (N >= MIN_LARGE_GEMM_N) {
        if (M <= 2048) {
            using fg = fused_globals<128, 128, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 128, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 2, 15>(
                globals, timing_records, host_timing);
        } else if (M <= 3072) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 15>(
                globals, timing_records, host_timing);
        } else if (M <= 3584) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 20>(
                globals, timing_records, host_timing);
        } else if (M <= 4096) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5>(
                globals, timing_records, host_timing);
        } else if (M <= 8192) {
            using fg = fused_globals<128, 256, 2, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor,
                                                      LocalTensor,
                                                      128,
                                                      256,
                                                      2,
                                                      2>(A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5, 2>(
                globals, timing_records, host_timing);
        } else if (M <= 16384) {
            using fg = fused_globals<128, 256, 2, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor,
                                                      LocalTensor,
                                                      128,
                                                      256,
                                                      2,
                                                      2>(A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5, 2>(
                globals, timing_records, host_timing);
        } else {
            using fg = fused_globals<128, 256, 2, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor,
                                                      LocalTensor,
                                                      128,
                                                      256,
                                                      2,
                                                      2>(A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5, 2>(
                globals, timing_records, host_timing);
        }
    } else {
        if (M <= 2048) {
            using fg = fused_globals<128, 128, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 128, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 2, 25>(
                globals, timing_records, host_timing);
        } else if (M <= 3072) {
            using fg = fused_globals<128, 128, 1>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 128, 1>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 1, 20>(
                globals, timing_records, host_timing);
        } else if (M <= 3584) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 10>(
                globals, timing_records, host_timing);
        } else if (M <= 4096) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 10>(
                globals, timing_records, host_timing);
        } else if (M <= 8192) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 10>(
                globals, timing_records, host_timing);
        } else if (M <= 16384) {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 15>(
                globals, timing_records, host_timing);
        } else {
            using fg = fused_globals<128, 256, 2>;
            fg globals =
                ag_gemm_warp_specialized_make_globals<DistributedTensor, LocalTensor, 128, 256, 2>(
                    A, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 15>(
                globals, timing_records, host_timing);
        }
    }
}
};  // namespace ag_gemm_warp_specialized
