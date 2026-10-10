#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def(
        "ag_gemm_warp_specialized_prepare",
        [](dist::ParallelBuffer& A_copy_ready, int M, int N) {
            const int dev_idx = A_copy_ready.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
        },
        pybind11::arg("A_copy_ready"),
        pybind11::arg("M"),
        pybind11::arg("N"));
    m.def(
        "ag_gemm_warp_specialized_launch",
        [](dist::ParallelBuffer& A,
           dist::ParallelBuffer& A_copy_ready,
           const at::Tensor& B,
           at::Tensor& C) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            const int M = C.size(0) * C.size(1);
            ag_gemm_warp_specialized::launch<dist::ParallelBuffer,
                                             dist::ParallelBuffer,
                                             at::Tensor>(A,
                                                         A_copy_ready,
                                                         B,
                                                         C,
                                                         M,
                                                         B.size(0),
                                                         B.size(1),
                                                         dev_idx,
                                                         at::cuda::getCurrentCUDAStream().stream());
        },
        pybind11::arg("A"),
        pybind11::arg("A_copy_ready"),
        pybind11::arg("B"),
        pybind11::arg("C"));
}
