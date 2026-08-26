#pragma once

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("moe_dispatch_gemm_warp_specialization",
          &moe_dispatch_gemm_warp_specialization::dispatch_gemm_warp_specialization,
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
#ifdef PROFILE_TIMINGS
    m.def("moe_dispatch_gemm_warp_specialization_profile",
          &moe_dispatch_gemm_warp_specialization::dispatch_gemm_warp_specialization_profile,
          pybind11::arg("pre_tokens"),
          pybind11::arg("ring_tokens"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("ring_full_epoch"),
          pybind11::arg("ring_empty_epoch"),
          pybind11::arg("ring_done_tiles"),
          pybind11::arg("row_block_to_expert"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("timings"),
          pybind11::arg("num_sms"));

    pybind11::dict events;
    events["CTA_START"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_CTA_START);
    events["CTA_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_CTA_END);
    events["DISPATCH_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_BEGIN);
    events["DISPATCH_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_END);
    events["EPILOGUE_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_BEGIN);
    events["EPILOGUE_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_END);
    events["TMA_PRODUCER_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_PRODUCER_BEGIN);
    events["TMA_PRODUCER_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_PRODUCER_END);
    events["MMA_ISSUER_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_MMA_ISSUER_BEGIN);
    events["MMA_ISSUER_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_MMA_ISSUER_END);
    events["DISPATCH_WAIT_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_WAIT_BEGIN);
    events["DISPATCH_WAIT_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_WAIT_END);
    events["DISPATCH_WORK_DONE"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_WORK_DONE);
    events["TMA_RING_WAIT_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_RING_WAIT_BEGIN);
    events["TMA_RING_WAIT_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_RING_WAIT_END);
    events["TMA_K_LOOP_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_K_LOOP_BEGIN);
    events["TMA_K_LOOP_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_TMA_K_LOOP_END);
    events["MMA_K_LOOP_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_MMA_K_LOOP_BEGIN);
    events["MMA_K_LOOP_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_MMA_K_LOOP_END);
    events["EPILOGUE_WAIT_BEGIN"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_WAIT_BEGIN);
    events["EPILOGUE_WAIT_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_WAIT_END);
    events["EPILOGUE_TASK_END"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_TASK_END);
    events["DISPATCH_GROUP_SYNC_DONE"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_GROUP_SYNC_DONE);
    events["DISPATCH_PUBLISH_DONE"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_PUBLISH_DONE);
    events["DISPATCH_ROUND_SYNC_DONE"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_DISPATCH_ROUND_SYNC_DONE);
    events["EPILOGUE_TMEM_WAIT_TOTAL"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_TMEM_WAIT_TOTAL);
    events["EPILOGUE_STORE_WAIT_TOTAL"] = static_cast<int>(
        moe_dispatch_gemm_warp_specialization::EV_EPILOGUE_STORE_WAIT_TOTAL);
    m.attr("TIMING_EVENTS") = events;
    m.attr("EVENTS_PER_BLOCK") = ::mkernel::timing::EVENTS_PER_BLOCK;
    m.attr("TIMING_RECORD_SIZE") =
        static_cast<int>(sizeof(::mkernel::timing::TimingRecord));
#endif
}
