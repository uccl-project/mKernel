#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
#ifdef PROFILE_TIMINGS
    using namespace ag_gemm_warp_specialized;
    m.def(
        "ag_gemm_warp_specialized",
        [](dist::ParallelBuffer& A,
           const at::Tensor& B,
           at::Tensor& C,
           int logical_global_m,
           const at::Tensor& timings) {
            void* timing_records = nullptr;
            if (timings.defined()) {
                TORCH_CHECK(timings.is_cuda(), "timings must be a CUDA tensor");
                TORCH_CHECK(timings.scalar_type() == at::kLong,
                            "timings must have dtype torch.int64");
                TORCH_CHECK(timings.is_contiguous(), "timings must be contiguous");
                TORCH_CHECK(timings.device() == C.device(),
                            "timings and C must be on the same CUDA device");
                constexpr int64_t required_elements =
                    static_cast<int64_t>(TIMING_NUM_BLOCKS) * EVENTS_PER_BLOCK * 2;
                TORCH_CHECK(timings.numel() >= required_elements,
                            "timings needs at least ",
                            required_elements,
                            " int64 elements, got ",
                            timings.numel());
                timing_records = timings.data_ptr<int64_t>();
            }
            ag_gemm_warp_specialized::entrypoint<dist::ParallelBuffer, at::Tensor>(
                A, B, C, logical_global_m, -1, -1, -1, nullptr, timing_records);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"),
        pybind11::arg("timings") = at::Tensor());

    pybind11::dict events;
    events["RANK_ENTRY_WAIT_BEGIN"] = static_cast<int>(EV_RANK_ENTRY_WAIT_BEGIN);
    events["RANK_ENTRY_WAIT_DONE"] = static_cast<int>(EV_RANK_ENTRY_WAIT_DONE);
    events["MEMCPY_BEGIN"] = static_cast<int>(EV_MEMCPY_BEGIN);
    events["MEMCPY_COMPLETE"] = static_cast<int>(EV_MEMCPY_COMPLETE);
    events["COPY_WAIT_BEGIN"] = static_cast<int>(EV_COPY_WAIT_BEGIN);
    events["COPY_WAIT_DONE"] = static_cast<int>(EV_COPY_WAIT_DONE);
    events["GEMM_BEGIN"] = static_cast<int>(EV_GEMM_BEGIN);
    events["GEMM_DONE"] = static_cast<int>(EV_GEMM_DONE);
    events["RANK_EXIT_WAIT_BEGIN"] = static_cast<int>(EV_RANK_EXIT_WAIT_BEGIN);
    events["RANK_EXIT_WAIT_DONE"] = static_cast<int>(EV_RANK_EXIT_WAIT_DONE);
    m.attr("TIMING_EVENTS") = events;
    m.attr("EVENTS_PER_BLOCK") = EVENTS_PER_BLOCK;
    m.attr("TIMING_RECORD_SIZE") = static_cast<int>(sizeof(TimingRecord));
    m.attr("TIMING_NUM_BLOCKS") = TIMING_NUM_BLOCKS;
    m.attr("TIMING_CONTROL_BLOCK") = TIMING_CONTROL_BLOCK;
#else
    m.def(
        "ag_gemm_warp_specialized",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C, int logical_global_m) {
            ag_gemm_warp_specialized::entrypoint<dist::ParallelBuffer, at::Tensor>(
                A, B, C, logical_global_m);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));
#endif
}
