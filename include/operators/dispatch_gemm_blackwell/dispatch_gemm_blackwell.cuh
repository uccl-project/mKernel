#pragma once

/**
 * @file dispatch_gemm_blackwell.cuh
 * @brief Direct MoE token gather inside one NVLink domain.
 *
 * pre_tokens is a ParallelBuffer whose VMM mappings and peer TMA descriptors
 * cover all GPUs in the domain. pull_dispatch_indices is [N, 2], with columns
 * (source GPU, source token row). No token data is replicated before dispatch.
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_memory_vec_tma.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 4
#endif

namespace moe_dispatch_gemm_blackwell {

static constexpr int NUM_DISPATCH_THREADS = 32;
// 16 * 7168 * sizeof(bf16) = 224 KiB, plus a small static mbarrier array.
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

struct dispatch_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int H = TK_MOE_H;
    static constexpr int TOKENS_PER_BLOCK = 16;
    static constexpr int ROW_BLOCK = 128;

    using token_vec = sv_bf<H>;
    using pre_tokens_distributed_tensor = dist::distributed_tensor<
        dist::local_tensor<bf16, 1, 1, -1, H, token_vec>,
        NUM_DEVICES, false>;
    using post_tokens_local_tensor =
        dist::local_tensor<bf16, 1, 1, -1, H, token_vec>;
    using pull_dispatch_indices_local_tensor =
        dist::local_tensor<int, 1, 1, -1, 2>;
    using row_ready_local_tensor =
        dist::local_tensor<int, 1, 1, 1, -1>;

    pre_tokens_distributed_tensor pre_tokens;
    post_tokens_local_tensor post_tokens;
    pull_dispatch_indices_local_tensor pull_dispatch_indices;
    row_ready_local_tensor row_ready;
    int num_output_tokens;
    int num_dispatch_sms;
};

void launch_dispatch_lsa(const dispatch_globals& G, cudaStream_t stream);

struct weight_load_globals {
    using weight_tile = st_bf<64, 256>;
    using matrix_tensor =
        dist::local_tensor<bf16, 1, 1, -1, -1, weight_tile>;

    matrix_tensor weights;
    matrix_tensor output;
    int row_tile;
    int col_tile;
};

void launch_dummy_weight_load(const weight_load_globals& G,
                              cudaStream_t stream);

inline void dispatch(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &row_ready,
    int num_dispatch_sms
) {
    const int dev_idx = pre_tokens.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    TORCH_CHECK(pre_tokens.local_world_size_ == dispatch_globals::NUM_DEVICES,
                "pre_tokens world size must match INTRA_NUM_DEVICES");
    TORCH_CHECK(post_tokens.dim() == 2 &&
                post_tokens.size(1) == dispatch_globals::H,
                "post_tokens must have shape [N, H]");
    TORCH_CHECK(pull_dispatch_indices.dim() == 2 &&
                pull_dispatch_indices.size(0) == post_tokens.size(0) &&
                pull_dispatch_indices.size(1) == 2,
                "pull_dispatch_indices must have shape [N, 2]");
    const int num_row_blocks =
        (static_cast<int>(post_tokens.size(0)) + dispatch_globals::ROW_BLOCK - 1) /
        dispatch_globals::ROW_BLOCK;
    TORCH_CHECK(row_ready.dim() == 1 &&
                row_ready.size(0) == num_row_blocks &&
                row_ready.scalar_type() == at::kInt &&
                row_ready.is_contiguous(),
                "row_ready must be contiguous int32 [ceil(N / 128)]");
    TORCH_CHECK(num_dispatch_sms > 0, "num_dispatch_sms must be positive");

    dispatch_globals G{
        .pre_tokens = ::dist::distributed_tensor_from_buffer<
            dispatch_globals::pre_tokens_distributed_tensor>(pre_tokens),
        .post_tokens = ::dist::local_tensor_from_tensor<
            dispatch_globals::post_tokens_local_tensor>(post_tokens),
        .pull_dispatch_indices = ::dist::local_tensor_from_tensor<
            dispatch_globals::pull_dispatch_indices_local_tensor>(
                pull_dispatch_indices),
        .row_ready = ::dist::local_tensor_from_tensor<
            dispatch_globals::row_ready_local_tensor>(row_ready),
        .num_output_tokens = static_cast<int>(post_tokens.size(0)),
        .num_dispatch_sms = num_dispatch_sms,
    };
    launch_dispatch_lsa(G, stream);
}

inline void dummy_weight_load(
    at::Tensor &weights,
    at::Tensor &output,
    int row_tile,
    int col_tile
) {
    c10::cuda::CUDAGuard device_guard(weights.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(weights.dim() == 2 && weights.scalar_type() == at::kBFloat16,
                "weights must be a 2D bf16 tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({64, 256}) &&
                output.scalar_type() == at::kBFloat16 && output.is_contiguous(),
                "output must be contiguous bf16 [64, 256]");
    TORCH_CHECK(row_tile >= 0 && (row_tile + 1) * 64 <= weights.size(0),
                "row tile is out of bounds");
    TORCH_CHECK(col_tile >= 0 && (col_tile + 1) * 256 <= weights.size(1),
                "column tile is out of bounds");

    weight_load_globals G{
        .weights = ::dist::local_tensor_from_tensor<
            weight_load_globals::matrix_tensor>(weights),
        .output = ::dist::local_tensor_from_tensor<
            weight_load_globals::matrix_tensor>(output),
        .row_tile = row_tile,
        .col_tile = col_tile,
    };
    launch_dummy_weight_load(G, stream);
}

}  // namespace moe_dispatch_gemm_blackwell
