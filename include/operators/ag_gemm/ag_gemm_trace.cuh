/**
 * Per-task activity trace for ag_gemm (build with AG_GEMM_TRACE=1).
 *
 * Answers one question: do the gather CTAs and the compute CTAs actually
 * interleave in time, or does compute sit blocked until gather has run ahead?
 * Every gather task and every compute task emits one record, and the records
 * carry both an absolute timestamp (so tasks on different SMs land on a common
 * axis) and a stall count (so a bar can be split into "waiting" vs "working").
 *
 * Two clocks on purpose:
 *   - %globaltimer is device-wide and nanosecond-based, so it is the only thing
 *     that can put two SMs on one axis. It is also slow to read, so it is read
 *     exactly twice per task.
 *   - clock64() is a per-SM cycle counter -- useless across SMs, but nearly
 *     free and perfectly good for accumulating a duration inside one task on
 *     one SM. All the inner-loop stall accounting uses it.
 * busy_clk lets each record self-calibrate: ns_per_tick = (t2-t0)/busy_clk.
 */
#pragma once

#ifdef AG_GEMM_TRACE

#include <cuda_runtime.h>

namespace ag_gemm_multinode {
namespace trace {

// Sized for the largest shape we profile: M=32768 gives 4096 gather tasks and
// 2048 compute tasks. 64K records = 4 MB of static device memory.
static constexpr int MAX_RECS = 1 << 16;
static constexpr int FIELDS   = 10;

// Roles.
static constexpr unsigned long long ROLE_GATHER  = 0;  // phase-1 multicast gather
static constexpr unsigned long long ROLE_COMPUTE = 1;  // GEMM task (one row-block pair)

struct rec {
    unsigned long long smid;
    unsigned long long block;
    unsigned long long role;
    unsigned long long task;
    unsigned long long t0_ns;      // %globaltimer at task start
    unsigned long long t2_ns;      // %globaltimer at task end
    unsigned long long stall_clk;  // ticks blocked on data readiness (gather -> compute)
    unsigned long long pipe_clk;   // ticks blocked on pipeline backpressure (compute-bound)
    unsigned long long busy_clk;   // ticks for the whole task
    unsigned long long spare;
};
static_assert(sizeof(rec) == FIELDS * sizeof(unsigned long long), "rec must stay flat");

__device__ rec g_recs[MAX_RECS];
__device__ int g_count;

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
    unsigned long long stall_clk, unsigned long long pipe_clk,
    unsigned long long busy_clk
) {
    const int i = atomicAdd(&g_count, 1);
    if (i >= MAX_RECS) return;   // drop rather than scribble
    rec &r    = g_recs[i];
    r.smid    = smid();
    r.block   = blockIdx.x;
    r.role    = role;
    r.task    = (unsigned long long)task;
    r.t0_ns   = t0_ns;
    r.t2_ns   = t2_ns;
    r.stall_clk = stall_clk;
    r.pipe_clk  = pipe_clk;
    r.busy_clk  = busy_clk;
    r.spare     = 0;
}

}  // namespace trace
}  // namespace ag_gemm_multinode

// Inner-loop stall accounting. Cheap enough to leave in the k-loop.
#define AG_TRACE_STALL_BEGIN(var) unsigned long long var = clock64()
#define AG_TRACE_STALL_END(acc, var) (acc) += clock64() - (var)

#else  // !AG_GEMM_TRACE

#define AG_TRACE_STALL_BEGIN(var) do {} while (0)
#define AG_TRACE_STALL_END(acc, var) do {} while (0)

#endif  // AG_GEMM_TRACE
