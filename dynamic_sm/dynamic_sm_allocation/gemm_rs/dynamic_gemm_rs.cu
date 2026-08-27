#include "policies/scheduler_base.cuh"

#include "pyutils/torchutils.cuh"
#include "../common/dynamic_sm_allocation_utils.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <vector>

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace dynamic_gemm_rs {

namespace scheduler = dynamic_sm_allocation::scheduler;

static constexpr unsigned int COMM_FLOOR_HELPERS = 1;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
#ifdef GEMM_RS_NUM_BLOCKS
    static constexpr int NUM_BLOCKS = GEMM_RS_NUM_BLOCKS;
#else
    static constexpr int NUM_BLOCKS = 132;  // H100 default: one CTA per SM
#endif

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS =
        CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

struct globals {
    static constexpr unsigned int TRACE_MAX = 8192;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using workspace_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using output_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;

    A_gl A;
    B_gl B;
    workspace_gl workspace;
    output_pgl output;
    int *ready;
    const int dev_idx;
    unsigned int *next_compute;
    unsigned int *next_comm;
    unsigned int *active_comm_helpers;
    unsigned int *done_compute;
    unsigned int *done_comm;
    unsigned int *ready_comm;
    unsigned int *last_ready_comm;
    unsigned int *last_done_compute;
    unsigned int *last_done_comm;
    unsigned int *target_comm_helpers;
    // --- FT cycle + controller_lock counters ---
    unsigned int* compute_cycles;
    unsigned int* comm_cycles;
    unsigned int* last_compute_cycles;
    unsigned int* last_comm_cycles;
    unsigned int* controller_lock;
    unsigned int* stable_count;
    unsigned int* ema_cost_primary;
    unsigned int* ema_cost_secondary;
#ifdef GEMM_RS_PRINT_SCHED_STATS
    unsigned int* trace_buf;
    unsigned int* trace_idx;
    unsigned long long* trace_time_buf;
#endif
#ifdef GEMM_RS_PRINT_TASK_TIMES
    unsigned int* compute_task_times;
    unsigned int* comm_task_times;
#endif

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

using scheduler::load_u32;



__device__ inline bool is_ready_local(const int *ready, const int task_id);
__device__ inline int map_task_id(
    const int logical_task_id,
    const int num_blocks,
    const int dev_idx
);

#ifdef GEMM_RS_PRINT_SCHED_STATS
__device__ inline void record_trace(
    const globals& G,
    unsigned int prev_target,
    unsigned int next_target,
    const scheduler::WorkSample& sample
) {
    if (G.trace_buf == nullptr || G.trace_idx == nullptr) {
        return;
    }
    unsigned int slot = atomicAdd(G.trace_idx, 1u);
    if (slot >= globals::TRACE_MAX) {
        return;
    }
    const unsigned int active = scheduler::load_u32(G.active_comm_helpers);
    const unsigned int ready_secondary = scheduler::load_u32(G.ready_comm);
    const unsigned int backlog =
        ready_secondary > sample.curr_done_secondary
            ? ready_secondary - sample.curr_done_secondary
            : 0u;
    G.trace_buf[slot * 8 + 0] = sample.curr_done_primary;
    G.trace_buf[slot * 8 + 1] = sample.curr_done_secondary;
    G.trace_buf[slot * 8 + 2] = next_target;
    G.trace_buf[slot * 8 + 3] = active;
    G.trace_buf[slot * 8 + 4] = prev_target;
    G.trace_buf[slot * 8 + 5] = backlog;
    // Per-task cost (float-as-uint): matches work_balance WINDOWED computation.
    float trace_C_p = (sample.delta_primary > 0)
        ? fmaxf((float)sample.delta_primary_cycles / (float)sample.delta_primary, 1.0f)
        : 0.0f;
    unsigned int delta_s = max(sample.delta_secondary, sample.delta_ready_secondary);
    float trace_C_s = (delta_s > 0)
        ? fmaxf((float)sample.delta_secondary_cycles / (float)delta_s, 1.0f)
        : 0.0f;
    G.trace_buf[slot * 8 + 6] = __float_as_uint(trace_C_p);
    G.trace_buf[slot * 8 + 7] = __float_as_uint(trace_C_s);
    if (G.trace_time_buf != nullptr) {
        G.trace_time_buf[slot] = scheduler::read_globaltimer_ns();
    }
}
#endif

// Work-balance policy variant of the controller update.
__device__ inline void maybe_update_comm_controller(
    const globals &G,
    const scheduler::WorkBalanceConfig& wb_config) {
    const scheduler::HelperCounters counters{
        .active_helpers = G.active_comm_helpers,
        .done_primary = G.done_compute,
        .done_secondary = G.done_comm,
        .last_done_primary = G.last_done_compute,
        .last_done_secondary = G.last_done_comm,
        .target_helpers = G.target_comm_helpers,
        .blocked_primary = nullptr,
        .last_blocked_primary = nullptr,
        .primary_cycles = G.compute_cycles,
        .secondary_cycles = G.comm_cycles,
        .last_primary_cycles = G.last_compute_cycles,
        .last_secondary_cycles = G.last_comm_cycles,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
        .controller_lock = G.controller_lock,
        .stable_count = G.stable_count,
    };
    scheduler::update_target_any_cta_with_trace(
        counters,
        [&] __device__(unsigned int prev_target,
                       const scheduler::WorkSample& sample) {
#ifdef FIXED_DYNAMIC_TARGET
            (void)prev_target; (void)sample;
            return (unsigned int)FIXED_DYNAMIC_TARGET;
#else
            return scheduler::update_target_work_balance(prev_target, sample, wb_config);
#endif
        },
        [&] __device__(unsigned int prev_target,
                       unsigned int next_target,
                       const scheduler::WorkSample& sample) {
#ifdef GEMM_RS_PRINT_SCHED_STATS
            record_trace(G, prev_target, next_target, sample);
#endif
        });
}

__device__ inline bool try_join_comm_helpers(
    const globals &G,
    const unsigned int comm_helper_cap
) {
    return scheduler::try_join_helpers(
        G.active_comm_helpers, G.target_comm_helpers, comm_helper_cap);
}

__device__ inline bool try_leave_comm_helpers(const globals &G) {
    return scheduler::try_leave_helpers(
        G.active_comm_helpers, G.target_comm_helpers, COMM_FLOOR_HELPERS);
}

__device__ inline bool try_claim_ready_comm_task(
    const globals &G,
    const int num_blocks,
    int &claimed_task
) {
    const unsigned int candidate = load_u32(G.next_comm);
    if ((int)candidate >= num_blocks) {
        claimed_task = -1;
        return false;
    }
    const int task_id = map_task_id((int)candidate, num_blocks, G.dev_idx);
    if (!is_ready_local(G.ready, task_id)) {
        claimed_task = -1;
        return false;
    }
    const unsigned int prior = atomicCAS(G.next_comm, candidate, candidate + 1u);
    if (prior != candidate) {
        claimed_task = -1;
        return false;
    }
    claimed_task = (int)candidate;
    return true;
}

__device__ inline void signal_ready(const int *ready, const int task_id) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" :: "l"(ready + task_id), "r"(1u)
                 : "memory");
}

__device__ inline bool is_ready_local(const int *ready, const int task_id) {
    unsigned int val;
    asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                 : "=r"(val)
                 : "l"(ready + task_id)
                 : "memory");
    return val == 1u;
}

__device__ inline void wait_ready(const int *ready, const int task_id) {
    while (!is_ready_local(ready, task_id)) {
        __nanosleep(16);
    }
}

__device__ inline int map_task_id(
    const int logical_task_id,
    const int num_blocks,
    const int dev_idx
) {
    const int dev_task_offset =
        ((dev_idx + 1) * (num_blocks / globals::NUM_DEVICES)) % num_blocks;
    return (logical_task_id + dev_task_offset) % num_blocks;
}

__device__ inline void compute_tile(
    const globals &G,
    const int task_id,
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES],
    globals::pipeline_outputs &outputs,
    semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    semaphore &outputs_arrived,
    semaphore &outputs_finished,
    int &stage,
    uint32_t &phasebits,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks,
    const int num_iters
) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();

    int row_idx;
    int col_idx;
    decode_task_id<globals>(
        task_id,
        row_blocks,
        col_blocks,
        super_rows,
        final_rows,
        super_blocks,
        row_idx,
        col_idx
    );

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if (warp_id == 0 && lane_id == 0) {
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                if (red_idx == globals::PIPELINE_STAGES - 1) {
                    wait(
                        outputs_finished,
                        get_phasebit<1>(phasebits, globals::PIPELINE_STAGES)
                    );
                    update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                }
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::load_async(
                        inputs[stage].A[i],
                        G.A,
                        {row_idx * 2 + i, red_idx},
                        inputs_arrived[stage]
                    );
                }
                tma::load_async(
                    inputs[stage].B,
                    G.B,
                    {red_idx, col_idx},
                    inputs_arrived[stage]
                );
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) {
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                tma::store_async(
                    G.workspace,
                    outputs.C[i],
                    {row_idx * 2 + i, col_idx}
                );
            }
            tma::store_async_read_wait();
            signal_ready(G.ready, task_id);
            atomicAdd(G.ready_comm, 1u);
            arrive(outputs_finished);
        }
    } else {
        rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
        warp::zero(C_accum);

        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
            warpgroup::mma_async_wait();
            warp::arrive(inputs_finished[stage]);
            stage = (stage + 1) % globals::PIPELINE_STAGES;
        }

        group<8>::sync(3);
        warpgroup::store(outputs.C[warpgroup_id], C_accum);
        warpgroup::sync(warpgroup_id + 1);
        warpgroup::arrive(outputs_arrived);
    }
}

__device__ inline void comm_tile(
    const globals &G,
    const int task_id,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    globals::C_tile (&partials)[2] =
        allocator.allocate<globals::C_tile, 2>();

    __shared__ semaphore partials_arrived[2];
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            init_semaphore(partials_arrived[i], 0, 1);
        }
    }
    __syncthreads();

    int row_idx;
    int col_idx;
    decode_task_id<globals>(
        task_id,
        row_blocks,
        col_blocks,
        super_rows,
        final_rows,
        super_blocks,
        row_idx,
        col_idx
    );

    const int row_blocks_per_dev = row_blocks / globals::NUM_DEVICES;
    const int owner_dev_idx = row_idx / row_blocks_per_dev;
    const int local_row_idx = row_idx % row_blocks_per_dev;

    const int warp_id = warp::groupid();
    if (warp_id == 0 && laneid() == 0) {
        wait_ready(G.ready, task_id);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            tma::expect_bytes(partials_arrived[i], sizeof(globals::C_tile));
            tma::load_async(
                partials[i],
                G.workspace,
                {row_idx * 2 + i, col_idx},
                partials_arrived[i]
            );
        }
    } else if (warp_id == 1 && laneid() == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            wait(partials_arrived[i], 0);
            tma::store_add_async(
                G.output[owner_dev_idx],
                partials[i],
                {local_row_idx * 2 + i, col_idx}
            );
        }
        tma::store_async_read_wait();
    }

    __syncthreads();
}

__device__ inline void main_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);

    static_assert(
        sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) +
                sizeof(globals::pipeline_outputs) <=
            config::DYNAMIC_SHARED_MEMORY
    );
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs =
        *reinterpret_cast<globals::pipeline_outputs *>(
            &inputs[globals::PIPELINE_STAGES - 1]
        );

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ bool claimed_ready_comm;
    __shared__ int claimed_comm_task;
    __shared__ int claimed_compute_task;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);

    }
    __syncthreads();

    const int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int super_rows = (row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;
    const int num_blocks = row_blocks * col_blocks;
    const int num_iters = G.A.cols() / globals::RED_BLOCK;
    const unsigned int comm_helper_cap = min(
        (unsigned int)num_blocks,
        (unsigned int)(config::NUM_BLOCKS - 1)
    );
    const scheduler::WorkBalanceConfig wb_config{
        .total_ctas = (unsigned int)config::NUM_BLOCKS,
        .total_primary = (unsigned int)num_blocks,
        .total_secondary = (unsigned int)num_blocks,
        .helper_floor = COMM_FLOOR_HELPERS,
        .helper_cap = comm_helper_cap,
        .secondary_blocks_primary = false,  // compute→comm dependency
        .ema_alpha = 0.3f,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
    };
    bool helping_comm = false;

    while (true) {
        maybe_update_comm_controller(G, wb_config);
        __syncthreads();

        if (threadIdx.x == 0) {
            claimed_ready_comm = false;
            claimed_comm_task = -1;
            claimed_compute_task = -1;
            // Target-based role assignment: read target with relaxed load,
            // decide role based on blockIdx.  No CAS needed — eliminates
            // try_join/try_leave atomic contention entirely.
            const unsigned int target = scheduler::load_u32(G.target_comm_helpers);
            helping_comm = ((unsigned int)blockIdx.x < target &&
                            (unsigned int)blockIdx.x < (unsigned int)comm_helper_cap);
            if (helping_comm) {
                claimed_ready_comm =
                    try_claim_ready_comm_task(G, num_blocks, claimed_comm_task);
            }
        }
        __syncthreads();
        if (claimed_ready_comm) {
            unsigned long long _comm_t0 = clock64();
            comm_tile(
                G,
                map_task_id(claimed_comm_task, num_blocks, G.dev_idx),
                row_blocks,
                col_blocks,
                super_rows,
                final_rows,
                super_blocks
            );
            if (threadIdx.x == 0) {
                const unsigned int _comm_elapsed =
                    (unsigned int)((clock64() - _comm_t0) >> 10);
                atomicAdd(G.done_comm, 1u);
                if (G.comm_cycles != nullptr)
                    atomicAdd(G.comm_cycles, _comm_elapsed);
#ifdef GEMM_RS_PRINT_TASK_TIMES
                if (G.comm_task_times != nullptr && claimed_comm_task >= 0) {
                    G.comm_task_times[claimed_comm_task] = _comm_elapsed;
                }
#endif
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            claimed_compute_task = (int)atomicAdd(G.next_compute, 1u);
        }
        __syncthreads();
        if (claimed_compute_task < num_blocks) {
            unsigned long long _comp_t0 = clock64();
            int stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            compute_tile(
                G,
                map_task_id(claimed_compute_task, num_blocks, G.dev_idx),
                inputs,
                outputs,
                inputs_arrived,
                inputs_finished,
                outputs_arrived,
                outputs_finished,
                stage,
                phasebits,
                row_blocks,
                col_blocks,
                super_rows,
                final_rows,
                super_blocks,
                num_iters
            );
            if (threadIdx.x == 0) {
                const unsigned int _comp_elapsed =
                    (unsigned int)((clock64() - _comp_t0) >> 10);
                atomicAdd(G.done_compute, 1u);
                if (G.compute_cycles != nullptr)
                    atomicAdd(G.compute_cycles, _comp_elapsed);
#ifdef GEMM_RS_PRINT_TASK_TIMES
                if (G.compute_task_times != nullptr && claimed_compute_task >= 0) {
                    G.compute_task_times[claimed_compute_task] = _comp_elapsed;
                }
#endif
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            // No try_leave needed with target-based assignment.
            // All compute tiles done; claim remaining comm tasks directly.
            claimed_comm_task = (int)atomicAdd(G.next_comm, 1u);
        }
        __syncthreads();
        if (claimed_comm_task >= num_blocks) {
            break;
        }
        unsigned long long _comm_t0b = clock64();
        comm_tile(
            G,
            map_task_id(claimed_comm_task, num_blocks, G.dev_idx),
            row_blocks,
            col_blocks,
            super_rows,
            final_rows,
            super_blocks
        );
        if (threadIdx.x == 0) {
            const unsigned int comm_cycles =
                (unsigned int)((clock64() - _comm_t0b) >> 10);
            atomicAdd(G.done_comm, 1u);
            if (G.comm_cycles != nullptr)
                atomicAdd(G.comm_cycles, comm_cycles);
#ifdef GEMM_RS_PRINT_TASK_TIMES
            if (G.comm_task_times != nullptr && claimed_comm_task >= 0) {
                G.comm_task_times[claimed_comm_task] = comm_cycles;
            }
#endif
        }
        __syncthreads();
    }

}

// Cache-line padded counter struct — each hot counter gets its own 128-byte
// L2 cache line to prevent false sharing between writer and reader CTAs.
struct alignas(128) SchedulerCounters {
    unsigned int next_compute;          char _pad0[124];   // Slot 0: hot write (all CTAs claiming tiles)
    unsigned int next_comm;             char _pad1[124];   // Slot 1: hot write (comm CTAs)
    unsigned int active_comm_helpers;   char _pad2[124];   // Slot 2: hot write (role changes)
    unsigned int done_compute;          char _pad3[124];   // Slot 3: hot write (all CTAs)
    unsigned int done_comm;             char _pad4[124];   // Slot 4: hot write (comm CTAs)
    unsigned int ready_comm;            char _pad5[124];   // Slot 5: ready comm tasks
    unsigned int last_ready_comm;       char _pad6[124];   // Slot 6: snapshot for controller
    unsigned int last_done_compute;     char _pad7[124];   // Slot 7: snapshot for controller
    unsigned int last_done_comm;        char _pad8[124];   // Slot 8: snapshot for controller
    unsigned int target_comm_helpers;   char _pad9[124];   // Slot 9: controller output (read by all)
    unsigned int compute_cycles;        char _pad10[124];  // Slot 10: cycle accumulator
    unsigned int comm_cycles;           char _pad11[124];  // Slot 11: cycle accumulator
    unsigned int last_compute_cycles;   char _pad12[124];  // Slot 12: snapshot for controller
    unsigned int last_comm_cycles;      char _pad13[124];  // Slot 13: snapshot for controller
    unsigned int stable_count;          char _pad14[124];  // Slot 14: controller backoff
    unsigned int ema_cost_primary;      char _pad15[124];  // Slot 15: EMA cost primary (float-as-uint)
    unsigned int ema_cost_secondary;    char _pad16[124];  // Slot 16: EMA cost secondary (float-as-uint)
};
static_assert(sizeof(SchedulerCounters) == 17 * 128, "SchedulerCounters size mismatch");

static SchedulerCounters* g_sched_counters[globals::NUM_DEVICES] = {nullptr};
#ifdef GEMM_RS_PRINT_SCHED_STATS
static unsigned int *g_trace_buf[globals::NUM_DEVICES] = {nullptr};
static unsigned int *g_trace_idx[globals::NUM_DEVICES] = {nullptr};
static unsigned long long *g_trace_time_buf[globals::NUM_DEVICES] = {nullptr};
#endif
#ifdef GEMM_RS_PRINT_TASK_TIMES
static unsigned int *g_compute_task_times[globals::NUM_DEVICES] = {nullptr};
static unsigned int *g_comm_task_times[globals::NUM_DEVICES] = {nullptr};
static unsigned int g_task_time_capacity[globals::NUM_DEVICES] = {0};
#endif

#ifdef GEMM_RS_PRINT_TASK_TIMES
static void print_task_time_summary(
    const char* label,
    const unsigned int* buf,
    unsigned int count) {
    std::vector<unsigned int> vals;
    vals.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        if (buf[i] != 0u) {
            vals.push_back(buf[i]);
        }
    }
    if (vals.empty()) {
        printf("[gemm_rs %s times] no samples\n", label);
        return;
    }
    std::sort(vals.begin(), vals.end());
    const auto at_pct = [&](double pct) -> unsigned int {
        const size_t idx = static_cast<size_t>(pct * (vals.size() - 1));
        return vals[idx];
    };
    unsigned long long sum = 0;
    for (unsigned int v : vals) sum += v;
    printf(
        "[gemm_rs %s times] samples=%zu avg=%.1f cyc>>10 min=%u p50=%u p90=%u p99=%u max=%u\n",
        label,
        vals.size(),
        (double)sum / (double)vals.size(),
        vals.front(),
        at_pct(0.50),
        at_pct(0.90),
        at_pct(0.99),
        vals.back()
    );
}
#endif

#ifdef GEMM_RS_PRINT_TASK_TIMES
static at::Tensor get_compute_task_times_py(int dev_idx, int n) {
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kInt));
    if (g_compute_task_times[dev_idx] != nullptr && n > 0) {
        cudaMemcpy(out.data_ptr<int>(), g_compute_task_times[dev_idx],
                   (size_t)n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}
static at::Tensor get_comm_task_times_py(int dev_idx, int n) {
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kInt));
    if (g_comm_task_times[dev_idx] != nullptr && n > 0) {
        cudaMemcpy(out.data_ptr<int>(), g_comm_task_times[dev_idx],
                   (size_t)n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif
#ifdef GEMM_RS_PRINT_SCHED_STATS
static at::Tensor get_trace_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, g_trace_idx[dev_idx], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, (unsigned int)globals::TRACE_MAX);
    auto out = at::zeros({n, 8}, at::TensorOptions().dtype(at::kInt));
    if (n > 0 && g_trace_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int>(), g_trace_buf[dev_idx],
                   (size_t)n * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}

static at::Tensor get_trace_time_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, g_trace_idx[dev_idx], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, (unsigned int)globals::TRACE_MAX);
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kLong));
    if (n > 0 && g_trace_time_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int64_t>(), g_trace_time_buf[dev_idx],
                   (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &workspace,
    kittens::py::TKParallelTensor &output,
    const at::Tensor &ready
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(
        workspace.is_cuda() && ready.is_cuda(),
        "workspace/ready must be CUDA tensors"
    );
    TORCH_CHECK(
        A.is_contiguous() && B.is_contiguous() && workspace.is_contiguous() &&
            ready.is_contiguous(),
        "A/B/workspace/ready must be contiguous"
    );
    TORCH_CHECK(A.dtype() == at::ScalarType::BFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");
    TORCH_CHECK(
        workspace.dtype() == at::ScalarType::BFloat16,
        "workspace must be bf16"
    );
    TORCH_CHECK(
        ready.dtype() == at::ScalarType::Int,
        "ready tensor dtype must be int32"
    );
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be rank-2");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul shape mismatch: A[M,K], B[K,N]");

    kittens::py::parallel_tensor_check(output);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));
    const int row_blocks = M / globals::ROW_BLOCK;
    const int col_blocks = N / globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;

    TORCH_CHECK(
        M % globals::ROW_BLOCK == 0,
        "M must be divisible by ROW_BLOCK (", globals::ROW_BLOCK, ")"
    );
    TORCH_CHECK(
        K % globals::RED_BLOCK == 0,
        "K must be divisible by RED_BLOCK (", globals::RED_BLOCK, ")"
    );
    TORCH_CHECK(
        N % globals::COL_BLOCK == 0,
        "N must be divisible by COL_BLOCK (", globals::COL_BLOCK, ")"
    );
    TORCH_CHECK(
        workspace.size(0) == M && workspace.size(1) == N,
        "workspace shape must be [M, N]"
    );
    TORCH_CHECK(
        ready.dim() == 1 && ready.size(0) == num_blocks,
        "ready shape must be [M/ROW_BLOCK * N/COL_BLOCK]"
    );
    TORCH_CHECK(
        output.data_.dtype() == at::ScalarType::BFloat16,
        "output must be bf16"
    );

    const int dev_idx = output.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

#ifdef GEMM_RS_PRINT_SCHED_STATS
    if (g_trace_buf[dev_idx] == nullptr) {
        cudaMalloc(&g_trace_buf[dev_idx], globals::TRACE_MAX * 8 * sizeof(unsigned int));
        cudaMalloc(&g_trace_idx[dev_idx], sizeof(unsigned int));
        cudaMalloc(&g_trace_time_buf[dev_idx], globals::TRACE_MAX * sizeof(unsigned long long));
    }
    cudaMemsetAsync(g_trace_buf[dev_idx], 0, globals::TRACE_MAX * 8 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_idx[dev_idx], 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_time_buf[dev_idx], 0, globals::TRACE_MAX * sizeof(unsigned long long), stream);
#endif
#ifdef GEMM_RS_PRINT_TASK_TIMES
    if (g_task_time_capacity[dev_idx] < (unsigned int)num_blocks) {
        if (g_compute_task_times[dev_idx] != nullptr) cudaFree(g_compute_task_times[dev_idx]);
        if (g_comm_task_times[dev_idx] != nullptr) cudaFree(g_comm_task_times[dev_idx]);
        cudaMalloc(&g_compute_task_times[dev_idx], num_blocks * sizeof(unsigned int));
        cudaMalloc(&g_comm_task_times[dev_idx], num_blocks * sizeof(unsigned int));
        g_task_time_capacity[dev_idx] = (unsigned int)num_blocks;
    }
    cudaMemsetAsync(g_compute_task_times[dev_idx], 0, num_blocks * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_comm_task_times[dev_idx], 0, num_blocks * sizeof(unsigned int), stream);
#endif

    if (g_sched_counters[dev_idx] == nullptr) {
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    }
    cudaMemsetAsync(g_sched_counters[dev_idx], 0, sizeof(SchedulerCounters), stream);
    SchedulerCounters* sc = g_sched_counters[dev_idx];
    {
        // Compute initial target from task ratio for faster convergence.
        const unsigned int comm_cap_h = min(
            (unsigned int)num_blocks,
            (unsigned int)(config::NUM_BLOCKS - 1));
        const scheduler::WorkBalanceConfig wb_init{
            .total_ctas = (unsigned int)config::NUM_BLOCKS,
            .total_primary = (unsigned int)num_blocks,
            .total_secondary = (unsigned int)num_blocks,
            .helper_floor = COMM_FLOOR_HELPERS,
            .helper_cap = comm_cap_h,
            .secondary_blocks_primary = false,
            .ema_alpha = 0.3f,
            .ema_cost_primary = nullptr,
            .ema_cost_secondary = nullptr,
        };
        const unsigned int initial_target_comm_helpers =
            scheduler::compute_initial_target_wb(wb_init);
        cudaMemcpyAsync(
            &sc->target_comm_helpers,
            &initial_target_comm_helpers,
            sizeof(unsigned int),
            cudaMemcpyHostToDevice,
            stream
        );
    }

    globals G{
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .workspace = kittens::py::tensor_to_gl<globals::workspace_gl>(workspace),
        .output =
            kittens::py::parallel_tensor_to_pgl<globals::output_pgl>(output),
        .ready = ready.data_ptr<int>(),
        .dev_idx = dev_idx,
        .next_compute = &sc->next_compute,
        .next_comm = &sc->next_comm,
        .active_comm_helpers = &sc->active_comm_helpers,
        .done_compute = &sc->done_compute,
        .done_comm = &sc->done_comm,
        .ready_comm = &sc->ready_comm,
        .last_ready_comm = &sc->last_ready_comm,
        .last_done_compute = &sc->last_done_compute,
        .last_done_comm = &sc->last_done_comm,
        .target_comm_helpers = &sc->target_comm_helpers,
        .compute_cycles = &sc->compute_cycles,
        .comm_cycles = &sc->comm_cycles,
        .last_compute_cycles = &sc->last_compute_cycles,
        .last_comm_cycles = &sc->last_comm_cycles,
        .controller_lock = nullptr,
        .stable_count = &sc->stable_count,
        .ema_cost_primary = &sc->ema_cost_primary,
        .ema_cost_secondary = &sc->ema_cost_secondary,
#ifdef GEMM_RS_PRINT_SCHED_STATS
        .trace_buf = g_trace_buf[dev_idx],
        .trace_idx = g_trace_idx[dev_idx],
        .trace_time_buf = g_trace_time_buf[dev_idx],
#endif
#ifdef GEMM_RS_PRINT_TASK_TIMES
        .compute_task_times = g_compute_task_times[dev_idx],
        .comm_task_times = g_comm_task_times[dev_idx],
#endif
    };

    kittens::py::launch_kernel<config, globals, main_kernel>(G);
#ifdef GEMM_RS_PRINT_SCHED_STATS
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    SchedulerCounters h;
    cudaMemcpy(&h, g_sched_counters[dev_idx], sizeof(SchedulerCounters), cudaMemcpyDeviceToHost);
    if (dev_idx == 0) {
        printf("[gemm_rs sched] dev=%d done_compute=%u/%d ready_comm=%u/%d done_comm=%u/%d final_target=%u active_helpers=%u\n",
               dev_idx, h.done_compute, num_blocks, h.ready_comm, num_blocks, h.done_comm, num_blocks, h.target_comm_helpers, h.active_comm_helpers);
        unsigned int tidx = 0;
        cudaMemcpy(&tidx, g_trace_idx[dev_idx], sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int n_entries = tidx < globals::TRACE_MAX ? tidx : globals::TRACE_MAX;
        if (n_entries > 0) {
            unsigned int* tbuf = new unsigned int[n_entries * 8];
            cudaMemcpy(tbuf, g_trace_buf[dev_idx], n_entries * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("[gemm_rs trace] %u entries\n", n_entries);
            printf("  %6s  %6s  %6s  %6s  %6s  %6s  %6s\n",
                   "d_comp", "d_comm", "target", "active", "prev_t", "backlg", "comp_ct");
            for (unsigned int i = 0; i < n_entries; i++) {
                unsigned int act = tbuf[i * 8 + 3];
                printf("  %6u  %6u  %6u  %6u  %6u  %6u  %6u\n",
                       tbuf[i * 8 + 0], tbuf[i * 8 + 1], tbuf[i * 8 + 2], act,
                       tbuf[i * 8 + 4], tbuf[i * 8 + 5], 132u - act);
            }
            delete[] tbuf;
        }
    }
#endif
#ifdef GEMM_RS_PRINT_TASK_TIMES
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (dev_idx == 0) {
        std::vector<unsigned int> compute_times(num_blocks, 0u);
        std::vector<unsigned int> comm_times(num_blocks, 0u);
        cudaMemcpy(
            compute_times.data(),
            g_compute_task_times[dev_idx],
            num_blocks * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            comm_times.data(),
            g_comm_task_times[dev_idx],
            num_blocks * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );
        print_task_time_summary("compute", compute_times.data(), (unsigned int)num_blocks);
        print_task_time_summary("comm", comm_times.data(), (unsigned int)num_blocks);
    }
#endif
}

} // namespace dynamic_gemm_rs

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def(
        "matmul_reduce_scatter_dynamic",
        &dynamic_gemm_rs::entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("workspace"),
        pybind11::arg("output"),
        pybind11::arg("ready")
    );
#ifdef GEMM_RS_PRINT_TASK_TIMES
    m.def("get_compute_task_times", &dynamic_gemm_rs::get_compute_task_times_py,
          pybind11::arg("dev_idx"), pybind11::arg("n"));
    m.def("get_comm_task_times", &dynamic_gemm_rs::get_comm_task_times_py,
          pybind11::arg("dev_idx"), pybind11::arg("n"));
#endif
#ifdef GEMM_RS_PRINT_SCHED_STATS
    m.def("get_trace_buf", &dynamic_gemm_rs::get_trace_buf_py,
          pybind11::arg("dev_idx"));
    m.def("get_trace_time_buf", &dynamic_gemm_rs::get_trace_time_buf_py,
          pybind11::arg("dev_idx"));
#endif
}
