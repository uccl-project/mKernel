#pragma once

#include <cstdint>

#ifdef PROFILE_TIMINGS
#include "common/timings.cuh"
#endif

namespace ag_gemm_warp_specialized {

static constexpr int AG_GEMM_NUM_BLOCKS = 148;

#ifdef PROFILE_TIMINGS
using TimingRecord = timing_profile::TimingRecord;
static constexpr int EVENTS_PER_BLOCK = timing_profile::EVENTS_PER_BLOCK;

// Kernel CTAs own [0, AG_GEMM_NUM_BLOCKS). The final partition is reserved
// for the two cross-rank barrier kernels that bracket AG-GEMM.
static constexpr int TIMING_CONTROL_BLOCK = AG_GEMM_NUM_BLOCKS;
static constexpr int TIMING_NUM_BLOCKS = AG_GEMM_NUM_BLOCKS + 1;
// A single pre-copy anchor lives outside the dense control-record prefix. The
// host uses it to align timed CUDA events with %globaltimer, then discards it.
static constexpr uint32_t TIMING_ANCHOR_INDEX = EVENTS_PER_BLOCK - 1;

// These values are persisted in profile .npz files. Each user-visible timing
// category has a begin/end endpoint so the renderer can build exact spans.
enum TimingEvent : uint32_t {
    EV_RANK_ENTRY_WAIT_BEGIN = 0,
    EV_RANK_ENTRY_WAIT_DONE = 1,
    EV_MEMCPY_BEGIN = 2,
    EV_MEMCPY_COMPLETE = 3,
    EV_COPY_WAIT_BEGIN = 4,
    EV_COPY_WAIT_DONE = 5,
    EV_GEMM_BEGIN = 6,
    EV_GEMM_DONE = 7,
    EV_RANK_EXIT_WAIT_BEGIN = 8,
    EV_RANK_EXIT_WAIT_DONE = 9,
};
#endif

}  // namespace ag_gemm_warp_specialized
