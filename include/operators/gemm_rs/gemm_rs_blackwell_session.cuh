#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    {
        // One ready counter per READY_ROWS x READY_COLS region, for every signal
        // granularity: size `ready` as (M / world_size / rows, N / cols).
        using fg = gemm_rs_blackwell::fused_globals<gemm_rs_blackwell::DEFAULT_SWIZZLE_SIZE,
                                                    gemm_rs_blackwell::DEFAULT_SIGNALS_PER_TILE,
                                                    gemm_rs_blackwell::DEFAULT_RS_STRIP_ROWS,
                                                    gemm_rs_blackwell::DEFAULT_NUM_CONSUMER_WARPS>;
        m.attr("gemm_rs_blackwell_ready_rows") = fg::READY_ROWS;
        m.attr("gemm_rs_blackwell_ready_cols") = fg::READY_COLS;
    }
    m.def(
        "gemm_rs_blackwell_prepare",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::prepare(A, B, C, ready, completion_epochs);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        "Initialize local counters once. Synchronize the stream and all ranks before launch.");
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
           int swizzle_size,
           int signals_per_tile,
           int rs_strip_rows,
           int mma_warps) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A,
                                      B,
                                      C,
                                      ready,
                                      completion_epochs,
                                      swizzle_size,
                                      signals_per_tile,
                                      rs_strip_rows,
                                      mma_warps);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("swizzle_size") = gemm_rs_blackwell::DEFAULT_SWIZZLE_SIZE,
        pybind11::arg("signals_per_tile") = gemm_rs_blackwell::DEFAULT_SIGNALS_PER_TILE,
        pybind11::arg("rs_strip_rows") = gemm_rs_blackwell::DEFAULT_RS_STRIP_ROWS,
        pybind11::arg("mma_warps") = gemm_rs_blackwell::DEFAULT_NUM_CONSUMER_WARPS,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "swizzle_size (1, 2, 4, or 8) is the tile-column width of each rasterization "
        "supergroup. signals_per_tile (1, 2, or 4) is how many ready-counter "
        "publications each tile is split into; rs_strip_rows (8, 16, or 32) is the "
        "height of each reduce-scatter strip. mma_warps (1 or 2) is the MMA warps per "
        "CTA, giving 256- or 512-row cluster tiles; 1 allows signals_per_tile 1 or 2.");
}
