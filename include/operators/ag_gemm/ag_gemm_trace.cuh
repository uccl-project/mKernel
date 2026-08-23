/**
 * Activity-trace bindings for ag_gemm (build with AG_GEMM_TRACE=1).
 *
 * Roles and slots only; the machinery is in common/mkernel_activity_trace.cuh.
 */
#pragma once

#ifdef AG_GEMM_TRACE
#define MKERNEL_ACTIVITY_TRACE
#endif
#include "common/mkernel_activity_trace.cuh"

#ifdef AG_GEMM_TRACE
namespace ag_gemm_multinode {
namespace trace {
using namespace ::mkernel::trace;

// Records. a = ticks blocked on gather readiness, b = ticks blocked on the
// compute pipeline.
static constexpr unsigned long long ROLE_GATHER  = 0;  // phase-1 multicast gather
static constexpr unsigned long long ROLE_COMPUTE = 1;  // GEMM task (one row-block pair)

// Live progress, for diagnosing a hang. Each slot holds the task a role last
// started, or DONE once it drains its loop; SLOT_END tracks the kernel tail.
// Deliberately coarse -- it is enough to say *which* role is stuck, and the
// 2-CTA deadlock was localised by adding finer marks temporarily and deleting
// them again.
static constexpr int SLOT_GATHER   = 0;
static constexpr int SLOT_LOADER   = 1;
static constexpr int SLOT_MMA      = 2;
static constexpr int SLOT_STORE    = 3;
static constexpr int SLOT_CONSUMER = 4;
static constexpr int SLOT_END      = 5;   // 1 role done, 2 grid-synced, 3 reset
static constexpr int DONE = 999999;       // role drained its task loop
}  // namespace trace
}  // namespace ag_gemm_multinode
#endif
