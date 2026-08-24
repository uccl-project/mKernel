#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("gemm_ar_intranode_blackwell",
          &gemm_ar_intranode_blackwell::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("C_final"),
          pybind11::arg("epoch"),
          pybind11::arg("gemm_to_ar_signal_strategy"));
}