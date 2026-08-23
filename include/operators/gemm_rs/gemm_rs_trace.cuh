/**
 * Activity-trace bindings for gemm_rs (build with GEMM_RS_TRACE=1).
 *
 * gemm_rs has no gather phase on the single-node path -- A and B are already
 * local -- so the question the trace answers here is different from ag_gemm's:
 * the reduce-scatter is fused into the epilogue as a cross-GPU store_add, and
 * what we want to know is whether that write, or the MMA, sets the pace.
 *
 * One record per task per warp role, so the three can be compared directly.
 */
#pragma once

#ifdef GEMM_RS_TRACE
#define MKERNEL_ACTIVITY_TRACE
#endif
#include "common/mkernel_activity_trace.cuh"

#ifdef GEMM_RS_TRACE
namespace gemm_rs_multinode {
namespace trace {
using namespace ::mkernel::trace;

// Roles. Each emits one record per task, with a/b as named below.
//   LOADER : a = ticks blocked on inputs_finished (pipeline backpressure)
//            b = 0
//   MMA    : a = ticks blocked on inputs_arrived  (waiting for operands)
//            b = ticks blocked on tmem_free       (waiting for the epilogue)
//   STORE  : a = ticks blocked on outputs_arrived (waiting for consumers)
//            b = ticks inside store_add + read_wait (the cross-GPU push)
static constexpr unsigned long long ROLE_LOADER = 0;
static constexpr unsigned long long ROLE_MMA    = 1;
static constexpr unsigned long long ROLE_STORE  = 2;

// Live progress slots, for diagnosing a hang.
static constexpr int SLOT_LOADER = 0;
static constexpr int SLOT_MMA    = 1;
static constexpr int SLOT_STORE  = 2;
static constexpr int SLOT_END    = 3;
}  // namespace trace
}  // namespace gemm_rs_multinode
#endif
