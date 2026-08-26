#pragma once

/**
 * @file dispatch_gemm_warp_specialization.cuh
 * @brief warp-specialized NVLink dispatch + BF16 FC1 for Blackwell.
 *
 * Every resident CTA contains both a dispatch warpgroup and a tcgen05 GEMM
 * pipeline.  Dispatch streams remote rows through small per-warp shared-memory
 * bounce buffers into a bounded global-memory ring.  GEMM consumes ready ring
 * blocks and releases each physical slot after all N tiles have completed.
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"

#ifdef PROFILE_TIMINGS
#include "profiling/timings.cuh"
#endif

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace moe_dispatch_gemm_warp_specialization {

#ifdef PROFILE_TIMINGS
enum TimingEvent : uint32_t {
    EV_CTA_START = 0,
    EV_CTA_END,
    EV_DISPATCH_BEGIN,
    EV_DISPATCH_END,
    EV_EPILOGUE_BEGIN,
    EV_EPILOGUE_END,
    EV_TMA_PRODUCER_BEGIN,
    EV_TMA_PRODUCER_END,
    EV_MMA_ISSUER_BEGIN,
    EV_MMA_ISSUER_END,
    EV_DISPATCH_WAIT_BEGIN,
    EV_DISPATCH_WAIT_END,
    EV_DISPATCH_WORK_DONE,
    EV_TMA_RING_WAIT_BEGIN,
    EV_TMA_RING_WAIT_END,
    EV_TMA_K_LOOP_BEGIN,
    EV_TMA_K_LOOP_END,
    EV_MMA_K_LOOP_BEGIN,
    EV_MMA_K_LOOP_END,
    EV_EPILOGUE_WAIT_BEGIN,
    EV_EPILOGUE_WAIT_END,
    EV_EPILOGUE_TASK_END,
    EV_DISPATCH_GROUP_SYNC_DONE,
    EV_DISPATCH_PUBLISH_DONE,
    EV_DISPATCH_ROUND_SYNC_DONE,
    // Summary-only events: timestamp stores an accumulated duration in ns.
    EV_EPILOGUE_TMEM_WAIT_TOTAL,
    EV_EPILOGUE_STORE_WAIT_TOTAL,
};
#endif

// Four full-token buffers leave enough shared memory for the existing four-stage
// BF16 tcgen05 pipeline. each H=7168 token uses one pull.
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

struct warp_specialization_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;

    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 256;
    static constexpr int BLOCK_K = 64;
    static constexpr int NUM_STAGES = 3;
    static constexpr int NUM_OUTPUT_STAGES = 2;
    static constexpr int NUM_DISPATCH_WARPS = 4;
    static constexpr int DISPATCH_CTAS_PER_BLOCK = 16;
    static constexpr int PULL_BYTES = 14336;
    static constexpr int THREADS = 384;

    static_assert((H * static_cast<int>(sizeof(bf16))) % PULL_BYTES == 0,
                  "H must split evenly into dispatch pull chunks");
    static_assert(I % BLOCK_N == 0, "I must be divisible by BLOCK_N");

    using dispatch_chunk = uint4[PULL_BYTES / sizeof(uint4)];
    using A_tile = st_bf<BLOCK_M, BLOCK_K>;
    using B_tile = st_bf<BLOCK_K, BLOCK_N>;
    using C_tile = st_bf<BLOCK_M, 32>;

    using pre_tokens_tensor = dist::distributed_tensor<
        dist::local_tensor<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    using ring_tokens_tensor =
        dist::local_tensor<bf16, 1, 1, -1, H, A_tile>;
    using routes_tensor = dist::local_tensor<int, 1, 1, -1, 2>;
    using weights_tensor =
        dist::local_tensor<bf16, 1, -1, H, I, B_tile>;
    using outputs_tensor =
        dist::local_tensor<bf16, 1, 1, -1, I, C_tile>;

    pre_tokens_tensor pre_tokens;
    ring_tokens_tensor ring_tokens;
    routes_tensor pull_dispatch_indices;
    weights_tensor weights;
    outputs_tensor outputs;

    int *ring_full_epoch;
    int *ring_empty_epoch;
    int *ring_done_tiles;
    int *row_block_to_expert;

    int num_output_tokens;
    int num_row_blocks;
    int num_ring_blocks;
    int num_local_experts;
    int num_sms;
#ifdef PROFILE_TIMINGS
    ::mkernel::timing::TimingRecord *timings;
#endif

    struct pipeline_input {
        A_tile A;
        B_tile B;
    };
};

void launch_warp_specialization(const warp_specialization_globals &G, cudaStream_t stream);

inline void dispatch_gemm_warp_specialization_impl(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &ring_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &ring_full_epoch,
    at::Tensor &ring_empty_epoch,
    at::Tensor &ring_done_tiles,
    at::Tensor &row_block_to_expert,
    at::Tensor &weights,
    at::Tensor &outputs,
    int num_sms
#ifdef PROFILE_TIMINGS
    , ::mkernel::timing::TimingRecord *timings = nullptr
#endif
) {
    const int dev_idx = pre_tokens.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    TORCH_CHECK(pre_tokens.local_world_size_ == warp_specialization_globals::NUM_DEVICES,
                "pre_tokens world size must match INTRA_NUM_DEVICES");
    TORCH_CHECK(pull_dispatch_indices.dim() == 2 &&
                pull_dispatch_indices.size(1) == 2 &&
                pull_dispatch_indices.scalar_type() == at::kInt &&
                pull_dispatch_indices.is_contiguous(),
                "pull_dispatch_indices must be contiguous int32 [M, 2]");

    const int64_t num_output_tokens = pull_dispatch_indices.size(0);
    TORCH_CHECK(num_output_tokens > 0 &&
                num_output_tokens % warp_specialization_globals::BLOCK_M == 0,
                "M must be positive and padded to BLOCK_M=128");
    const int64_t num_row_blocks = num_output_tokens / warp_specialization_globals::BLOCK_M;

    TORCH_CHECK(ring_tokens.dim() == 2 &&
                ring_tokens.size(1) == warp_specialization_globals::H &&
                ring_tokens.size(0) % warp_specialization_globals::BLOCK_M == 0 &&
                ring_tokens.scalar_type() == at::kBFloat16 &&
                ring_tokens.is_contiguous(),
                "ring_tokens must be contiguous bf16 [RING_BLOCKS*128, H]");
    const int64_t num_ring_blocks =
        ring_tokens.size(0) / warp_specialization_globals::BLOCK_M;
    TORCH_CHECK(num_ring_blocks > 0,
                "ring_tokens must contain at least one ring block");

    const auto check_ring_state = [&](const at::Tensor &t,
                                      const char *name) {
        TORCH_CHECK(t.dim() == 1 && t.size(0) == num_ring_blocks &&
                    t.scalar_type() == at::kInt && t.is_contiguous(),
                    name, " must be contiguous int32 [RING_BLOCKS]");
    };
    check_ring_state(ring_full_epoch, "ring_full_epoch");
    check_ring_state(ring_empty_epoch, "ring_empty_epoch");
    check_ring_state(ring_done_tiles, "ring_done_tiles");

    TORCH_CHECK(row_block_to_expert.dim() == 1 &&
                row_block_to_expert.size(0) == num_row_blocks &&
                row_block_to_expert.scalar_type() == at::kInt &&
                row_block_to_expert.is_contiguous(),
                "row_block_to_expert must be contiguous int32 [M/128]");
    TORCH_CHECK(weights.dim() == 3 &&
                weights.size(1) == warp_specialization_globals::H &&
                weights.size(2) == warp_specialization_globals::I &&
                weights.scalar_type() == at::kBFloat16 &&
                weights.is_contiguous(),
                "weights must be contiguous bf16 [E, H, I]");
    TORCH_CHECK(outputs.sizes() ==
                    at::IntArrayRef({num_output_tokens,
                                     static_cast<int64_t>(warp_specialization_globals::I)}) &&
                outputs.scalar_type() == at::kBFloat16 &&
                outputs.is_contiguous(),
                "outputs must be contiguous bf16 [M, I]");
    TORCH_CHECK(
        num_sms >= warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK &&
        num_sms <= kittens::num_sms(dev_idx),
        "num_sms must cover one dispatch CTA group and not exceed the device");

    warp_specialization_globals G{
        .pre_tokens = ::dist::distributed_tensor_from_buffer<
            warp_specialization_globals::pre_tokens_tensor>(pre_tokens),
        .ring_tokens = ::dist::local_tensor_from_tensor<
            warp_specialization_globals::ring_tokens_tensor>(ring_tokens),
        .pull_dispatch_indices = ::dist::local_tensor_from_tensor<
            warp_specialization_globals::routes_tensor>(pull_dispatch_indices),
        .weights = ::dist::local_tensor_from_tensor<
            warp_specialization_globals::weights_tensor>(weights),
        .outputs = ::dist::local_tensor_from_tensor<
            warp_specialization_globals::outputs_tensor>(outputs),
        .ring_full_epoch = ring_full_epoch.data_ptr<int>(),
        .ring_empty_epoch = ring_empty_epoch.data_ptr<int>(),
        .ring_done_tiles = ring_done_tiles.data_ptr<int>(),
        .row_block_to_expert = row_block_to_expert.data_ptr<int>(),
        .num_output_tokens = static_cast<int>(num_output_tokens),
        .num_row_blocks = static_cast<int>(num_row_blocks),
        .num_ring_blocks = static_cast<int>(num_ring_blocks),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .num_sms = num_sms,
#ifdef PROFILE_TIMINGS
        .timings = timings,
#endif
    };
    launch_warp_specialization(G, stream);
}

#ifdef PROFILE_TIMINGS
inline void dispatch_gemm_warp_specialization_profile(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &ring_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &ring_full_epoch,
    at::Tensor &ring_empty_epoch,
    at::Tensor &ring_done_tiles,
    at::Tensor &row_block_to_expert,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &timings,
    int num_sms
) {
    const int64_t required_records =
        static_cast<int64_t>(num_sms) * ::mkernel::timing::EVENTS_PER_BLOCK;
    TORCH_CHECK(timings.is_cuda() && timings.is_contiguous() &&
                    timings.scalar_type() == at::kLong,
                "timings must be a contiguous CUDA int64 tensor");
    TORCH_CHECK(timings.get_device() == pre_tokens.local_rank_,
                "timings must be on the same device as pre_tokens");
    TORCH_CHECK(timings.numel() >= required_records * 2,
                "timings needs two int64 values per timing record");
    dispatch_gemm_warp_specialization_impl(
        pre_tokens, ring_tokens, pull_dispatch_indices,
        ring_full_epoch, ring_empty_epoch, ring_done_tiles,
        row_block_to_expert, weights, outputs, num_sms,
        reinterpret_cast<::mkernel::timing::TimingRecord *>(
            timings.data_ptr<int64_t>()));
}
#endif

inline void dispatch_gemm_warp_specialization(
    dist::ParallelBuffer &pre_tokens,
    at::Tensor &ring_tokens,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &ring_full_epoch,
    at::Tensor &ring_empty_epoch,
    at::Tensor &ring_done_tiles,
    at::Tensor &row_block_to_expert,
    at::Tensor &weights,
    at::Tensor &outputs,
    int num_sms) {
    dispatch_gemm_warp_specialization_impl(
        pre_tokens, ring_tokens, pull_dispatch_indices,
        ring_full_epoch, ring_empty_epoch, ring_done_tiles,
        row_block_to_expert, weights, outputs, num_sms);
}

}  // namespace moe_dispatch_gemm_warp_specialization
