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
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& sync) {
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
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& sync,
           int SCHEDULER_SWIZZLE_SIZE,
           int signals_per_tile,
           int rs_strip_rows,
           int mma_warps) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A,
                                      B,
                                      C,
                                      sync,
                                      SCHEDULER_SWIZZLE_SIZE,
                                      signals_per_tile,
                                      rs_strip_rows,
                                      mma_warps);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("sync"),
        pybind11::arg("SCHEDULER_SWIZZLE_SIZE") = gemm_rs_blackwell::DEFAULT_SCHEDULER_SWIZZLE_SIZE,
        pybind11::arg("signals_per_tile") = gemm_rs_blackwell::DEFAULT_SIGNALS_PER_TILE,
        pybind11::arg("rs_strip_rows") = gemm_rs_blackwell::DEFAULT_RS_STRIP_ROWS,
        pybind11::arg("mma_warps") = gemm_rs_blackwell::DEFAULT_NUM_CONSUMER_WARPS,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "SCHEDULER_SWIZZLE_SIZE (1, 2, 4, or 8) is the tile-column width of each rasterization "
        "supergroup. signals_per_tile (1, 2, or 4) is how many ready-counter "
        "publications each tile is split into; rs_strip_rows (8, 16, or 32) is the "
        "height of each reduce-scatter strip. mma_warps (1 or 2) is the MMA warps per "
        "CTA, giving 256- or 512-row cluster tiles; 1 allows signals_per_tile 1 or 2.");
}
