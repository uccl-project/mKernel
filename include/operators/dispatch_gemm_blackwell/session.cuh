#pragma once

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("moe_dispatch_blackwell", &moe_dispatch_gemm_blackwell::dispatch,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("row_ready"),
          pybind11::arg("num_dispatch_sms"));
    m.def("dummy_weight_load", &moe_dispatch_gemm_blackwell::dummy_weight_load,
          pybind11::arg("weights"),
          pybind11::arg("output"),
          pybind11::arg("row_tile"),
          pybind11::arg("col_tile"));
    m.def("tcgen05_micro", &moe_dispatch_gemm_blackwell::tcgen05_micro,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"));
    m.def("moe_dispatch_gemm_blackwell",
          &moe_dispatch_gemm_blackwell::dispatch_gemm,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("row_ready"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("num_dispatch_sms"),
          pybind11::arg("num_gemm_sms"));
}
