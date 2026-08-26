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
          pybind11::arg("num_dispatch_sms") = 28,
          pybind11::arg("num_gemm_sms") = 124);
#ifdef PROFILE_TIMINGS
    m.def("moe_dispatch_gemm_blackwell_profile",
          &moe_dispatch_gemm_blackwell::dispatch_gemm_profile,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("row_ready"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("timings"),
          pybind11::arg("num_dispatch_sms") = 28,
          pybind11::arg("num_gemm_sms") = 124);

    pybind11::dict events;
    events["CTA_START"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_CTA_START);
    events["CTA_END"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_CTA_END);
    events["DISPATCH_SLICE_BEGIN"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_DISPATCH_SLICE_BEGIN);
    events["DISPATCH_PULL_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_DISPATCH_PULL_DONE);
    events["DISPATCH_SLICE_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_DISPATCH_SLICE_DONE);
    events["GEMM_ROW_WAIT_BEGIN"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_ROW_WAIT_BEGIN);
    events["GEMM_ROW_WAIT_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_ROW_WAIT_DONE);
    events["GEMM_TMA_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_TMA_DONE);
    events["GEMM_COMPUTE_WAIT_BEGIN"] =
        static_cast<int>(
            moe_dispatch_gemm_blackwell::EV_GEMM_COMPUTE_WAIT_BEGIN);
    events["GEMM_COMPUTE_BEGIN"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_COMPUTE_BEGIN);
    events["GEMM_COMPUTE_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_COMPUTE_DONE);
    events["GEMM_EPILOGUE_WAIT_BEGIN"] =
        static_cast<int>(
            moe_dispatch_gemm_blackwell::EV_GEMM_EPILOGUE_WAIT_BEGIN);
    events["GEMM_EPILOGUE_BEGIN"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_EPILOGUE_BEGIN);
    events["GEMM_EPILOGUE_DONE"] =
        static_cast<int>(moe_dispatch_gemm_blackwell::EV_GEMM_EPILOGUE_DONE);
    m.attr("TIMING_EVENTS") = events;
    m.attr("EVENTS_PER_BLOCK") = ::mkernel::timing::EVENTS_PER_BLOCK;
    m.attr("TIMING_RECORD_SIZE") =
        static_cast<int>(sizeof(::mkernel::timing::TimingRecord));
#endif
}
