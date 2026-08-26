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
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"
#include "profiling/timings.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 4
#endif

namespace moe_dispatch_gemm_blackwell {

#ifdef PROFILE_TIMINGS
enum TimingEvent : uint32_t {
    EV_CTA_START = 0,
    EV_CTA_END,
    EV_DISPATCH_SLICE_BEGIN,
    EV_DISPATCH_PULL_DONE,
    EV_DISPATCH_SLICE_DONE,
    EV_GEMM_ROW_WAIT_BEGIN,
    EV_GEMM_ROW_WAIT_DONE,
    EV_GEMM_TMA_DONE,
    EV_GEMM_COMPUTE_WAIT_BEGIN,
    EV_GEMM_COMPUTE_BEGIN,
    EV_GEMM_COMPUTE_DONE,
    EV_GEMM_EPILOGUE_WAIT_BEGIN,
    EV_GEMM_EPILOGUE_BEGIN,
    EV_GEMM_EPILOGUE_DONE,
};
#endif

// 16 * 7168 * sizeof(bf16) = 224 KiB, plus allocator alignment headroom.
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int TOKENS_PER_BLOCK = 16;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int OUTPUT_STAGES = 2;
    static constexpr int SUPER_M = 4;
    using token_vec = sv_bf<H>;
    using dispatch_tile = st_bf<TOKENS_PER_BLOCK, 256, false>;
    using A_tile = st_bf<ROW_BLOCK, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK, 32>;

    using pre_tokens_tensor = dist::distributed_tensor<
        dist::local_tensor<bf16, 1, 1, -1, H, token_vec, dispatch_tile>,
        NUM_DEVICES, false>;
    using post_tokens_tensor =
        dist::local_tensor<bf16, 1, 1, -1, H,
                           token_vec, dispatch_tile, A_tile>;
    using routes_tensor = dist::local_tensor<int, 1, 1, -1, 2>;
    using row_ready_tensor = dist::local_tensor<int, 1, 1, 1, -1>;
    using weights_tensor =
        dist::local_tensor<bf16, 1, -1, H, I, B_tile>;
    using outputs_tensor =
        dist::local_tensor<bf16, 1, 1, -1, I, C_tile>;
    using padded_tensor = dist::local_tensor<int, 1, 1, 1, -1>;

    pre_tokens_tensor pre_tokens;
    post_tokens_tensor post_tokens;
    routes_tensor pull_dispatch_indices;
    row_ready_tensor row_ready;
    weights_tensor weights;
    outputs_tensor outputs;
    padded_tensor padded_tokens_per_expert;

    int num_output_tokens;
    int num_local_experts;
    int num_dispatch_sms;
    int num_gemm_sms;
#ifdef PROFILE_TIMINGS
    ::mkernel::timing::TimingRecord *timings;
#endif

    struct pipeline_input {
        A_tile A;
        B_tile B;
    };
};

void launch_fused(const fused_globals& G, cudaStream_t stream);

inline void dispatch_gemm_impl(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &row_ready,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    int num_dispatch_sms,
    int num_gemm_sms
#ifdef PROFILE_TIMINGS
    , ::mkernel::timing::TimingRecord *timings = nullptr
#endif
) {
    const int dev_idx = pre_tokens.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    TORCH_CHECK(pre_tokens.local_world_size_ == fused_globals::NUM_DEVICES,
                "pre_tokens world size must match INTRA_NUM_DEVICES");
    TORCH_CHECK(post_tokens.dim() == 2 &&
                post_tokens.size(1) == fused_globals::H &&
                post_tokens.scalar_type() == at::kBFloat16 &&
                post_tokens.is_contiguous(),
                "post_tokens must be contiguous bf16 [N, H]");
    TORCH_CHECK(pull_dispatch_indices.sizes() ==
                    at::IntArrayRef({post_tokens.size(0), 2}) &&
                pull_dispatch_indices.scalar_type() == at::kInt &&
                pull_dispatch_indices.is_contiguous(),
                "pull_dispatch_indices must be contiguous int32 [N, 2]");
    const int num_row_blocks =
        (static_cast<int>(post_tokens.size(0)) + fused_globals::ROW_BLOCK - 1) /
        fused_globals::ROW_BLOCK;
    TORCH_CHECK(row_ready.sizes() == at::IntArrayRef({num_row_blocks}) &&
                row_ready.scalar_type() == at::kInt && row_ready.is_contiguous(),
                "row_ready must be contiguous int32 [ceil(N / 128)]");
    TORCH_CHECK(weights.dim() == 3 && weights.size(1) == fused_globals::H &&
                weights.size(2) == fused_globals::I &&
                weights.scalar_type() == at::kBFloat16 && weights.is_contiguous(),
                "weights must be contiguous bf16 [E, H, I]");
    TORCH_CHECK(outputs.sizes() ==
                    at::IntArrayRef({post_tokens.size(0), fused_globals::I}) &&
                outputs.scalar_type() == at::kBFloat16 && outputs.is_contiguous(),
                "outputs must be contiguous bf16 [N, I]");
    TORCH_CHECK(padded_tokens_per_expert.dim() == 1 &&
                padded_tokens_per_expert.size(0) == weights.size(0) &&
                padded_tokens_per_expert.scalar_type() == at::kInt &&
                padded_tokens_per_expert.is_contiguous(),
                "padded_tokens_per_expert must be contiguous int32 [E]");
    TORCH_CHECK(num_dispatch_sms > 0 && num_gemm_sms > 0,
                "dispatch and GEMM SM counts must be positive");
    TORCH_CHECK(num_dispatch_sms + num_gemm_sms <= kittens::num_sms(dev_idx),
                "dispatch + GEMM SM counts exceed the device SM count");
    fused_globals G{
        .pre_tokens = ::dist::distributed_tensor_from_buffer<
            fused_globals::pre_tokens_tensor>(pre_tokens),
        .post_tokens = ::dist::local_tensor_from_tensor<
            fused_globals::post_tokens_tensor>(post_tokens),
        .pull_dispatch_indices = ::dist::local_tensor_from_tensor<
            fused_globals::routes_tensor>(pull_dispatch_indices),
        .row_ready = ::dist::local_tensor_from_tensor<
            fused_globals::row_ready_tensor>(row_ready),
        .weights = ::dist::local_tensor_from_tensor<
            fused_globals::weights_tensor>(weights),
        .outputs = ::dist::local_tensor_from_tensor<
            fused_globals::outputs_tensor>(outputs),
        .padded_tokens_per_expert = ::dist::local_tensor_from_tensor<
            fused_globals::padded_tensor>(padded_tokens_per_expert),
        .num_output_tokens = static_cast<int>(post_tokens.size(0)),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .num_dispatch_sms = num_dispatch_sms,
        .num_gemm_sms = num_gemm_sms,
#ifdef PROFILE_TIMINGS
        .timings = timings,
#endif
    };
    launch_fused(G, stream);
}

#ifdef PROFILE_TIMINGS
inline void dispatch_gemm_profile(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &row_ready,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    at::Tensor &timings,
    int num_dispatch_sms,
    int num_gemm_sms
) {
    const int64_t required_records =
        static_cast<int64_t>(num_dispatch_sms + num_gemm_sms) *
        ::mkernel::timing::EVENTS_PER_BLOCK;
    TORCH_CHECK(timings.is_cuda() && timings.is_contiguous() &&
                    timings.scalar_type() == at::kLong,
                "timings must be a contiguous CUDA int64 tensor");
    TORCH_CHECK(timings.get_device() == pre_tokens.local_rank_,
                "timings must be on the same device as pre_tokens");
    TORCH_CHECK(timings.numel() >= required_records * 2,
                "timings needs two int64 values per timing record");
    dispatch_gemm_impl(pre_tokens, post_tokens, pull_dispatch_indices,
                       row_ready, weights, outputs,
                       padded_tokens_per_expert, num_dispatch_sms,
                       num_gemm_sms,
                       reinterpret_cast<::mkernel::timing::TimingRecord *>(
                           timings.data_ptr<int64_t>()));
}
#endif

inline void dispatch_gemm(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &row_ready,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    int num_dispatch_sms,
    int num_gemm_sms
) {
    dispatch_gemm_impl(pre_tokens, post_tokens, pull_dispatch_indices,
                       row_ready, weights, outputs,
                       padded_tokens_per_expert, num_dispatch_sms,
                       num_gemm_sms);
}

}  // namespace moe_dispatch_gemm_blackwell
