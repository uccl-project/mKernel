#pragma once

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("moe_dispatch_fc1_mega_blackwell",
          &moe_dispatch_fc1_mega_blackwell::dispatch_fc1_mega,
          pybind11::arg("pre_tokens"),
          pybind11::arg("ring_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("ring_full_epoch"),
          pybind11::arg("ring_empty_epoch"),
          pybind11::arg("ring_done_tiles"),
          pybind11::arg("row_block_to_expert"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("num_sms"));
}

