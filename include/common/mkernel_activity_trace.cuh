/**
 * @file
 * @brief Per-task activity trace shared by the Blackwell kernels.
 *
 * Two independent facilities, both compiled out unless MKERNEL_ACTIVITY_TRACE
 * is defined:
 *
 *   1. Records. One per task per warp role, carrying an absolute start/end
 *      timestamp plus three tick counters the caller assigns meaning to. Read
 *      back after the kernel finishes; this is what a timeline is built from.
 *
 *   2. A live progress array, written in place as work advances and read back
 *      with cudaMemcpyFromSymbolAsync on a non-blocking stream. The copy
 *      engines run independently of SM execution, so it comes out even while
 *      every CTA is spinning -- which is the only case where the records above
 *      are unavailable, i.e. a deadlock.
 *
 * Two clocks on purpose:
 *   - %globaltimer is device-wide and nanosecond-based, so it is the only thing
 *     that can put two SMs on one axis. It is also slow to read, so it is read
 *     exactly twice per task.
 *   - clock64() is a per-SM cycle counter -- useless across SMs, but nearly
 *     free and perfectly good for accumulating a duration inside one task on
 *     one SM. All the inner-loop stall accounting uses it.
 * busy lets each record self-calibrate: ns_per_tick = (t2 - t0) / busy.
 *
 * Roles and progress slots are named by the including kernel; nothing here
 * assumes what they mean.
 */
#pragma once

#ifdef MKERNEL_ACTIVITY_TRACE

#include <cuda_runtime.h>

namespace mkernel {
namespace trace {

// 64K records = 4 MB. The largest shape profiled so far emits ~6K.
static constexpr int MAX_RECS = 1 << 16;
static constexpr int FIELDS   = 10;

struct rec {
    unsigned long long smid;
    unsigned long long block;
    unsigned long long role;
    unsigned long long task;
    unsigned long long t0_ns;   // %globaltimer at task start
    unsigned long long t2_ns;   // %globaltimer at task end
    unsigned long long a;       // caller-assigned tick counter
    unsigned long long b;       // caller-assigned tick counter
    unsigned long long busy;    // ticks for the whole task
    unsigned long long spare;
};
static_assert(sizeof(rec) == FIELDS * sizeof(unsigned long long), "rec must stay flat");

__device__ rec g_recs[MAX_RECS];
__device__ int g_count;

static constexpr int MAX_CTAS   = 256;
static constexpr int ROLE_SLOTS = 9;

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
    rec &r  = g_recs[i];
    r.smid  = smid();
    r.block = blockIdx.x;
    r.role  = role;
    r.task  = (unsigned long long)task;
    r.t0_ns = t0_ns;
    r.t2_ns = t2_ns;
    r.a     = a;
    r.b     = b;
    r.busy  = busy;
    r.spare = 0;
}

__device__ __forceinline__ void mark(int slot, int value) {
    if (blockIdx.x < MAX_CTAS)
        // Plain volatile store: we want the latest value, not a history.
        reinterpret_cast<volatile int*>(g_progress)[blockIdx.x * ROLE_SLOTS + slot] = value;
}

}  // namespace trace
}  // namespace mkernel

#define MKERNEL_TRACE_MARK(slot, value) ::mkernel::trace::mark((slot), (value))
// Inner-loop stall accounting. Cheap enough to leave in a k-loop.
#define MKERNEL_TRACE_TICK_BEGIN(var) unsigned long long var = clock64()
#define MKERNEL_TRACE_TICK_END(acc, var) (acc) += clock64() - (var)

#else  // !MKERNEL_ACTIVITY_TRACE

#define MKERNEL_TRACE_MARK(slot, value) do {} while (0)
#define MKERNEL_TRACE_TICK_BEGIN(var) do {} while (0)
#define MKERNEL_TRACE_TICK_END(acc, var) do {} while (0)

#endif  // MKERNEL_ACTIVITY_TRACE
