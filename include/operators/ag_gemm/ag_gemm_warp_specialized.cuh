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
#include <type_traits>
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
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"

namespace ag_gemm_warp_specialized {

// CTAs per cluster, i.e. the tcgen05 MMA CTA group. 2 splits COL_BLOCK across
// the pair so each CTA stages half the B tile; 1 gives every CTA its own MMA.
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = 1;
static constexpr int MIN_LARGE_GEMM_N = 6288;
enum class AgStrategy {
    PULL,
    MULTICAST_PUSH,
};

// Shared by prepare and launch so the reset matches the dispatched transport.
constexpr AgStrategy strategy_for_shape(int M, int N) {
    return N < MIN_LARGE_GEMM_N && M <= 4096 ? AgStrategy::MULTICAST_PUSH : AgStrategy::PULL;
}

template <int _ROW_BLOCK, int _COL_BLOCK, int _NUM_CTA, int _NUM_CONSUMER_WARPS>
struct fused_globals;

// Number of tile columns visited before the snake pattern steps to the next
// supergroup; wider supergroups trade B-tile reuse for A-tile reuse in L2.
static constexpr int DEFAULT_SUPERGROUP_WIDTH = 5;

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA,
          int SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS,
          AgStrategy STRATEGY>
void launch_ag_gemm_warp_specialized(
    const fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>& G);

// for M < 512, this should be 128
template <int _ROW_BLOCK, int _COL_BLOCK, int _NUM_CTA, int _NUM_CONSUMER_WARPS>
struct fused_globals {
    // config items
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;

    // not sure if I want to use a warp specialized or sm specialized strategy yet
    static constexpr int NUM_BLOCKS = 148;
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

    // Keep C's real N stride. The generic local_tensor helper rounds the
    // tensor-map width to the swizzle granularity, which can produce OOB or
    // shifted stores when N is not a multiple of C_tile::cols.
    struct C_local_tensor {
        comm::bf16* data;
        CUtensorMap map;

        __device__ inline void prefetch_tma() const { dist::tma::prefetch_tensormap(&map); }
    };

    A_distributed_tensor A;
    B_local_tensor B;
    C_local_tensor C;

    // One distributed allocation holds all copy-engine completion flags.
    using A_ready_local_tensor = dist::local_tensor<uint32_t, 1, 1, 1, NUM_DEVICES>;
    using A_ready_distributed_tensor =
        dist::distributed_tensor<A_ready_local_tensor, NUM_DEVICES, false>;
    A_ready_distributed_tensor A_copy_ready;
    static constexpr uint32_t A_copy_epoch = 1;

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

template <int ROW_BLOCK,
          int COL_BLOCK,
          int NUM_CTA,
          int NUM_CONSUMER_WARPS,
          typename DistributedTensor,
          typename ReadyTensor,
          typename LocalTensor>
__host__ inline fused_globals<ROW_BLOCK, COL_BLOCK, NUM_CTA, NUM_CONSUMER_WARPS>
ag_gemm_warp_specialized_make_globals(DistributedTensor& A,
                                      ReadyTensor& A_copy_ready,
                                      const LocalTensor& B,
                                      LocalTensor& C,
                                      int dev_idx,
                                      int M,
                                      int N,
                                      int K,
                                      cudaStream_t stream) {
    using fg = fused_globals<ROW_BLOCK, COL_BLOCK, NUM_CTA, NUM_CONSUMER_WARPS>;

    const int local_m = M / fg::NUM_DEVICES;
    typename fg::C_local_tensor C_tensor;
    if constexpr (std::is_same_v<LocalTensor, comm::bf16*>) {
        C_tensor.data = C;
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    } else if constexpr (std::is_same_v<LocalTensor, at::Tensor>) {
        C_tensor.data = reinterpret_cast<comm::bf16*>(C.data_ptr());
#endif
    } else {
        static_assert(always_false_v<LocalTensor>, "Unsupported local tensor type");
    }

    uint64_t global_dim[3] = {
        static_cast<uint64_t>(N),
        static_cast<uint64_t>(local_m),
        static_cast<uint64_t>(fg::NUM_DEVICES),
    };
    uint64_t global_stride[2] = {
        static_cast<uint64_t>(N) * sizeof(comm::bf16),
        static_cast<uint64_t>(local_m) * N * sizeof(comm::bf16),
    };
    uint32_t box_dim[3] = {
        fg::C_tile::cols,
        fg::C_tile::rows,
        1,
    };
    uint32_t element_stride[3] = {1, 1, 1};

    constexpr CUtensorMapSwizzle c_swizzle = fg::C_tile::swizzle_bytes == 128
        ? CU_TENSOR_MAP_SWIZZLE_128B
        : fg::C_tile::swizzle_bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                          : CU_TENSOR_MAP_SWIZZLE_32B;

    MKERNEL_CUCHECK(cuTensorMapEncodeTiled(&C_tensor.map,
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
                                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    if constexpr (std::is_same_v<LocalTensor, comm::bf16*> &&
                  dist::RawDistributedMulticastTensorLike<DistributedTensor, comm::bf16> &&
                  dist::RawDistributedMulticastTensorLike<ReadyTensor, comm::bf16>) {
        return {
            .A = dist::make_distributed_tensor<typename fg::A_distributed_tensor>(
                reinterpret_cast<uint64_t>(A.mc),
                reinterpret_cast<uint64_t*>(A.uc_ptrs),
                1,
                fg::NUM_DEVICES,
                local_m,
                K),
            .B = dist::make_local_tensor<typename fg::B_local_tensor>(
                reinterpret_cast<uint64_t>(B), 1, 1, N, K),
            .C = C_tensor,
            .A_copy_ready = dist::make_distributed_tensor<typename fg::A_ready_distributed_tensor>(
                reinterpret_cast<uint64_t*>(A_copy_ready.uc_ptrs), 1, 1, 1, fg::NUM_DEVICES),
            .dev_idx = dev_idx,
            .M = M,
            .N = N,
            .K = K,
            .stream = stream,
        };
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    } else if constexpr (std::is_same_v<LocalTensor, at::Tensor> &&
                         std::is_same_v<DistributedTensor, dist::ParallelBuffer> &&
                         std::is_same_v<ReadyTensor, dist::ParallelBuffer>) {
        return {
            .A = dist::distributed_tensor_from_buffer<typename fg::A_distributed_tensor>(A),
            .B = dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
            .C = C_tensor,
            .A_copy_ready =
                dist::distributed_tensor_from_buffer<typename fg::A_ready_distributed_tensor>(
                    A_copy_ready, 1, 1, 1, fg::NUM_DEVICES),
            .dev_idx = dev_idx,
            .M = M,
            .N = N,
            .K = K,
            .stream = stream,
        };
#endif
    } else {
        static_assert(always_false_v<LocalTensor>, "Unsupported distributed/local tensor types");
    }
}

// Enqueue the local reset. Before launch, callers must establish a stream-ordered
// cross-rank barrier AFTER every rank's prepare (and input writes). The previous
// invocation must have finished on every rank before reusing its flags/buffers.
template <AgStrategy STRATEGY, typename ReadyTensor>
static void prepare_impl(ReadyTensor& A_copy_ready, int dev_idx, cudaStream_t stream) {
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!stream)
        stream = at::cuda::getCurrentCUDAStream().stream();
#endif
    void* local_flags;
    if constexpr (dist::RawDistributedMulticastTensorLike<ReadyTensor, comm::bf16>) {
        local_flags = A_copy_ready.uc_ptrs[dev_idx];
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    } else if constexpr (std::is_same_v<ReadyTensor, dist::ParallelBuffer>) {
        local_flags = A_copy_ready.data_.data_ptr();
#endif
    } else {
        static_assert(always_false_v<ReadyTensor>, "Unsupported ready tensor type");
    }
    constexpr size_t flag_count = STRATEGY == AgStrategy::MULTICAST_PUSH ? 1 : INTRA_NUM_DEVICES;
    MKERNEL_CUDACHECK(cudaMemsetAsync(local_flags, 0, flag_count * sizeof(uint32_t), stream));
}

// Use the same physical (padded) M and N that will be passed to launch.
template <typename ReadyTensor>
void prepare(ReadyTensor& A_copy_ready, int M, int N, int dev_idx, cudaStream_t stream) {
    if (strategy_for_shape(M, N) == AgStrategy::MULTICAST_PUSH)
        prepare_impl<AgStrategy::MULTICAST_PUSH>(A_copy_ready, dev_idx, stream);
    else
        prepare_impl<AgStrategy::PULL>(A_copy_ready, dev_idx, stream);
}

// A contains every rank's shard; callers initialize only A[rank] on each device.
// M/N/K are physical tensor dimensions, including padding. Does not reset flags:
// prepare -> cross-rank barrier -> launch -> cross-rank completion barrier.
template <typename DistributedTensor, typename ReadyTensor, typename LocalTensor>
void launch(DistributedTensor& A,
            ReadyTensor& A_copy_ready,
            const LocalTensor& B,
            LocalTensor& C,
            int M,
            int N,
            int K,
            int dev_idx,
            cudaStream_t stream) {
    // use size of N to check which projection is being done
    if (N >= MIN_LARGE_GEMM_N) {
        if (M <= 2048) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 128, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 2, 15, 1, AgStrategy::PULL>(globals);
        } else if (M <= 3072) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 15, 1, AgStrategy::PULL>(globals);
        } else if (M <= 3584) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 20, 1, AgStrategy::PULL>(globals);
        } else if (M <= 4096) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5, 1, AgStrategy::PULL>(globals);
        } else {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 2>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 5, 2, AgStrategy::PULL>(globals);
        }
    } else {
        if (M <= 2048) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 128, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 2, 25, 1, AgStrategy::MULTICAST_PUSH>(
                globals);
        } else if (M <= 3072) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 128, 1, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 128, 1, 20, 1, AgStrategy::MULTICAST_PUSH>(
                globals);
        } else if (M <= 4096) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 10, 1, AgStrategy::MULTICAST_PUSH>(
                globals);
        } else if (M <= 8192) {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 1>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 10, 1, AgStrategy::PULL>(globals);
        } else {
            auto globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, 2>(
                A, A_copy_ready, B, C, dev_idx, M, N, K, stream);
            launch_ag_gemm_warp_specialized<128, 256, 2, 15, 2, AgStrategy::PULL>(globals);
        }
    }
}
};  // namespace ag_gemm_warp_specialized
