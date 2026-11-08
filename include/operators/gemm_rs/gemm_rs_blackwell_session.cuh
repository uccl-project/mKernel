#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    // Shape of the int32 `sync` DistBuffer (allocate with multicast=False). It is
    // fixed, so one buffer serves every M, N and config.
    m.attr("gemm_rs_blackwell_sync_shape") =
        pybind11::make_tuple(gemm_rs_blackwell::SYNC_BUFFER_WORDS);
    m.def(
        "gemm_rs_blackwell_prepare",
        [](at::Tensor A, at::Tensor B, dist::ParallelBuffer& C, dist::ParallelBuffer& sync) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::prepare(A, B, C, sync);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("sync"),
        "Initialize local counters once. Synchronize the stream and all ranks before launch.");
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A, at::Tensor B, dist::ParallelBuffer& C, dist::ParallelBuffer& sync) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::entrypoint(A, B, C, sync);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("sync"),
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "The kernel configuration is chosen from a per-M tuned table (M <= 32768).");
}
