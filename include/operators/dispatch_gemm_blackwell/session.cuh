#pragma once

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
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
    m.def("moe_dispatch_gemm_blackwell_profile",
          &moe_dispatch_gemm_blackwell::dispatch_gemm_profile,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("row_ready"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("profile"),
          pybind11::arg("num_dispatch_sms"),
          pybind11::arg("num_gemm_sms"));
}
