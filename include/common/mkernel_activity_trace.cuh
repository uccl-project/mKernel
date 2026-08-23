/**
 * Per-task activity trace, shared by the Blackwell kernels. Compiled out unless
 * MKERNEL_ACTIVITY_TRACE is defined.
 *
 * Records are read back after the kernel finishes and build a timeline.
 * g_progress is written in place as work advances and can be read *during* a
 * hang with cudaMemcpyFromSymbolAsync on a non-blocking stream, since the copy
 * engines run independently of SM execution.
 *
 * Two clocks: %globaltimer is device-wide, so it is the only thing that puts
 * two SMs on one axis, but slow -- read twice per task. clock64() is per-SM and
 * nearly free, so it does the inner-loop stall accounting. busy self-calibrates
 * each record: ns_per_tick = (t2 - t0) / busy.
 *
 * Roles and progress slots are named by the including kernel.
 */
#pragma once

#ifdef MKERNEL_ACTIVITY_TRACE

#include <cuda_runtime.h>

namespace mkernel {
namespace trace {

static constexpr int MAX_RECS   = 1 << 16;   // 4 MB; the largest shape emits ~6K
static constexpr int FIELDS     = 10;
static constexpr int MAX_CTAS   = 256;
static constexpr int ROLE_SLOTS = 9;

struct rec {
    unsigned long long smid, block, role, task;
    unsigned long long t0_ns, t2_ns;   // %globaltimer at task start / end
    unsigned long long a, b;           // caller-assigned tick counters
    unsigned long long busy, spare;    // ticks for the whole task
};
static_assert(sizeof(rec) == FIELDS * sizeof(unsigned long long), "rec must stay flat");

__device__ rec g_recs[MAX_RECS];
__device__ int g_count;
__device__ int g_progress[MAX_CTAS * ROLE_SLOTS];

__device__ __forceinline__ unsigned int smid() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void emit(
    unsigned long long role, int task,
    unsigned long long t0_ns, unsigned long long t2_ns,
    unsigned long long a, unsigned long long b, unsigned long long busy
) {
    const int i = atomicAdd(&g_count, 1);
    if (i >= MAX_RECS) return;   // drop rather than scribble
    g_recs[i] = { smid(), blockIdx.x, role, (unsigned long long)task,
                  t0_ns, t2_ns, a, b, busy, 0 };
}

__device__ __forceinline__ void mark(int slot, int value) {
    // Latest value, not a history.
    if (blockIdx.x < MAX_CTAS)
        reinterpret_cast<volatile int*>(g_progress)[blockIdx.x * ROLE_SLOTS + slot] = value;
}

}  // namespace trace
}  // namespace mkernel

#define MKERNEL_TRACE_MARK(slot, value) ::mkernel::trace::mark((slot), (value))
#define MKERNEL_TRACE_TICK_BEGIN(var) unsigned long long var = clock64()
#define MKERNEL_TRACE_TICK_END(acc, var) (acc) += clock64() - (var)

#else

#define MKERNEL_TRACE_MARK(slot, value) do {} while (0)
#define MKERNEL_TRACE_TICK_BEGIN(var) do {} while (0)
#define MKERNEL_TRACE_TICK_END(acc, var) do {} while (0)

#endif
