#include "policies/scheduler_base.cuh"
#include "pyutils/torchutils.cuh"
#include "../common/dynamic_sm_allocation_utils.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace dynamic_gemm_ar {

namespace scheduler = dynamic_sm_allocation::scheduler;

static constexpr unsigned int COMM_FLOOR_HELPERS = 1;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
#ifdef GEMM_AR_NUM_BLOCKS
    static constexpr int NUM_BLOCKS = GEMM_AR_NUM_BLOCKS;
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
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;
    using barrier_pgl = barrier_t<NUM_DEVICES>;

    A_gl A;
    B_gl B;
    C_pgl C;
    barrier_pgl barrier;
    const int dev_idx;
    unsigned int* next_compute;
    unsigned int* next_comm;
    unsigned int* active_comm_helpers;
    unsigned int* done_compute;
    unsigned int* done_comm;
    unsigned int* last_done_compute;
    unsigned int* last_done_comm;
    unsigned int* target_comm_helpers;
    // --- FT cycle + controller_lock counters (only used with FT policy) ---
    unsigned int* compute_cycles;
    unsigned int* comm_cycles;
    unsigned int* last_compute_cycles;
    unsigned int* last_comm_cycles;
    unsigned int* controller_lock;
    unsigned int* stable_count;
    unsigned int* ema_cost_primary;
    unsigned int* ema_cost_secondary;
    unsigned int* successful_comm_claims;
    unsigned int* failed_comm_claims;
    unsigned int* last_successful_comm_claims;
    unsigned int* last_failed_comm_claims;
    unsigned int* total_comm_starts;
    unsigned int* last_total_comm_starts;
    unsigned int* max_failed_ready_value;
    unsigned int* cas_failed_claims;
#ifdef GEMM_AR_PRINT_SCHED_STATS
    unsigned int* trace_buf;
    unsigned int* trace_idx;
    unsigned long long* trace_time_buf;
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

__device__ inline int comm_linear_to_task_id(const int comm_linear, const int dev_idx);

__device__ inline void maybe_update_comm_controller(
    const globals& G,
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
        .successful_secondary_claims = G.successful_comm_claims,
        .failed_secondary_claims = G.failed_comm_claims,
        .last_successful_secondary_claims = G.last_successful_comm_claims,
        .last_failed_secondary_claims = G.last_failed_comm_claims,
    };
    scheduler::update_target_any_cta(
        counters,
        [&] __device__(unsigned int prev_target,
                       const scheduler::WorkSample& sample) {
#ifdef FIXED_DYNAMIC_TARGET
            // Override: use a fixed target to isolate infrastructure overhead
            // from policy error.  All infrastructure (sample_work, commit_sample,
            // controller lock, try_join/try_leave) still runs.
            unsigned int next = FIXED_DYNAMIC_TARGET;
#else
            unsigned int next = scheduler::update_target_work_balance(
                prev_target, sample, wb_config);
#endif
#ifdef GEMM_AR_PRINT_SCHED_STATS
            if (G.trace_buf != nullptr && G.trace_idx != nullptr) {
                unsigned int slot = atomicAdd(G.trace_idx, 1u);
                if (slot < globals::TRACE_MAX) {
                    const unsigned int curr_total_comm_starts =
                        scheduler::load_u32(G.total_comm_starts);
                    const unsigned int last_total_comm_starts =
                        scheduler::load_u32(G.last_total_comm_starts);
                    G.trace_buf[slot * 11 + 0] = sample.curr_done_primary;
                    G.trace_buf[slot * 11 + 1] = sample.curr_done_secondary;
                    G.trace_buf[slot * 11 + 2] = next;
                    G.trace_buf[slot * 11 + 3] = scheduler::load_u32(G.active_comm_helpers);
                    G.trace_buf[slot * 11 + 4] = prev_target;
                    G.trace_buf[slot * 11 + 5] = sample.delta_secondary;
                    G.trace_buf[slot * 11 + 6] = sample.delta_successful_secondary_claims;
                    G.trace_buf[slot * 11 + 7] = sample.delta_failed_secondary_claims;
                    G.trace_buf[slot * 11 + 8] =
                        curr_total_comm_starts - last_total_comm_starts;
                    // Per-task cost (float-as-uint): matches work_balance WINDOWED computation.
                    float trace_C_p = (sample.delta_primary > 0)
                        ? fmaxf((float)sample.delta_primary_cycles / (float)sample.delta_primary, 1.0f)
                        : 0.0f;
                    unsigned int delta_s = max(sample.delta_secondary, sample.delta_ready_secondary);
                    float trace_C_s = (delta_s > 0)
                        ? fmaxf((float)sample.delta_secondary_cycles / (float)delta_s, 1.0f)
                        : 0.0f;
                    G.trace_buf[slot * 11 + 9] = __float_as_uint(trace_C_p);
                    G.trace_buf[slot * 11 + 10] = __float_as_uint(trace_C_s);
                    if (G.trace_time_buf != nullptr) {
                        G.trace_time_buf[slot] = scheduler::read_globaltimer_ns();
                    }
                }
            }
            atomicExch(G.last_total_comm_starts, scheduler::load_u32(G.total_comm_starts));
#endif
            return next;
        });
}

__device__ inline bool try_join_comm_helpers(
    const globals& G,
    const unsigned int comm_helper_cap) {
    return scheduler::try_join_helpers(
        G.active_comm_helpers, G.target_comm_helpers, comm_helper_cap);
}

__device__ inline bool try_leave_comm_helpers(const globals& G) {
    return scheduler::try_leave_helpers(
        G.active_comm_helpers, G.target_comm_helpers, COMM_FLOOR_HELPERS);
}

__device__ inline bool try_claim_ready_comm_task(
    const globals& G,
    const int total_comm_tasks,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks,
    int& claimed_task) {
    const unsigned int candidate = load_u32(G.next_comm);
    if ((int)candidate >= total_comm_tasks) {
        claimed_task = -1;
        return false;
    }

    const int task_id = comm_linear_to_task_id((int)candidate, G.dev_idx);
    int row_idx;
    int col_idx;
    decode_task_id<globals>(task_id, row_blocks, col_blocks, super_rows,
                   final_rows, super_blocks, row_idx, col_idx);
    int ready_val;
    asm volatile("{ld.relaxed.sys.global.s32 %0, [%1];}"
                 : "=r"(ready_val)
                 : "l"(&G.barrier[G.dev_idx][{row_idx, col_idx}])
                 : "memory");
    if (ready_val != globals::NUM_DEVICES) {
        atomicMax(G.max_failed_ready_value, (unsigned int)ready_val);
        claimed_task = -1;
        return false;
    }

    const unsigned int prior = atomicCAS(G.next_comm, candidate, candidate + 1u);
    if (prior != candidate) {
        atomicAdd(G.cas_failed_claims, 1u);
        claimed_task = -1;
        return false;
    }
    claimed_task = (int)candidate;
    atomicAdd(G.successful_comm_claims, 1u);
    return true;
}

__host__ __device__ inline int total_comm_tasks_for_device(const int num_blocks, const int dev_idx) {
    if (num_blocks <= dev_idx) {
        return 0;
    }
    return ((num_blocks - 1 - dev_idx) / globals::NUM_DEVICES) + 1;
}

__device__ inline int comm_linear_to_task_id(const int comm_linear, const int dev_idx) {
    return comm_linear * globals::NUM_DEVICES + dev_idx;
}

__device__ inline void comm_tile(
    const globals& G,
    const int task_id,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks) {
    int row_idx;
    int col_idx;
    decode_task_id<globals>(task_id, row_blocks, col_blocks, super_rows, final_rows,
                   super_blocks, row_idx, col_idx);

    if (threadIdx.x == 0) {
        wait(G.barrier, {row_idx, col_idx}, G.dev_idx, globals::NUM_DEVICES);
    }
    __syncthreads();

    group<config::NUM_WARPS>::all_reduce<globals::ROW_BLOCK, globals::COL_BLOCK,
                                         reduce_op::ADD>(G.C, {row_idx, col_idx});
}

__device__ inline void compute_tile(
    const globals& G,
    const int task_id,
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES],
    globals::pipeline_outputs& outputs,
    semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    semaphore& outputs_arrived,
    semaphore& outputs_finished,
    int& stage,
    uint32_t& phasebits,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks,
    const int num_iters) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();

    int row_idx;
    int col_idx;
    decode_task_id<globals>(task_id, row_blocks, col_blocks, super_rows, final_rows,
                   super_blocks, row_idx, col_idx);

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if (warp_id == 0 && lane_id == 0) {
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage],
                                  sizeof(globals::pipeline_inputs));
                if (red_idx == globals::PIPELINE_STAGES - 1) {
                    wait(outputs_finished,
                         get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                    update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                }
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::load_async(inputs[stage].A[i], G.A,
                                    {row_idx * 2 + i, red_idx},
                                    inputs_arrived[stage]);
                }
                tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx},
                                inputs_arrived[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) {
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                tma::store_async(G.C[G.dev_idx], outputs.C[i],
                                 {row_idx * 2 + i, col_idx});
            }
            tma::store_async_read_wait();
            arrive(outputs_finished);

            const int signal_dev_idx = task_id % globals::NUM_DEVICES;
            signal(G.barrier, {row_idx, col_idx}, signal_dev_idx, 1);
        }
    } else {
        rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
        warp::zero(C_accum);

        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id],
                              inputs[stage].B);
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

__device__ inline void main_kernel(const globals& G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    static_assert(
        sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) +
            sizeof(globals::pipeline_outputs) <=
        config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs(&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs& outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(
            &inputs[globals::PIPELINE_STAGES - 1]);

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ bool claimed_ready_comm;
    __shared__ int claimed_comm_task;
    __shared__ int next_comp_task;

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
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    const int total_comm_tasks =
        total_comm_tasks_for_device(num_blocks, G.dev_idx);
    const unsigned int comm_helper_cap = min(
        (unsigned int)total_comm_tasks,
        (unsigned int)(config::NUM_BLOCKS - 1)
    );
    const scheduler::WorkBalanceConfig wb_config{
        .total_ctas = (unsigned int)config::NUM_BLOCKS,
        .total_primary = (unsigned int)num_blocks,
        .total_secondary = (unsigned int)total_comm_tasks,
        .helper_floor = COMM_FLOOR_HELPERS,
        .helper_cap = comm_helper_cap,
        .secondary_blocks_primary = false,  // compute→comm dependency
        .ema_alpha = 0.3f,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
    };
    bool helping_comm = false;

    // Batched done atomicAdd — accumulate locally, flush every 8 tiles.
    static constexpr unsigned int DONE_FLUSH_INTERVAL = 8;
    unsigned int local_done_compute_accum = 0;
    unsigned int local_done_comm_accum = 0;

    while (true) {
        maybe_update_comm_controller(G, wb_config);
        __syncthreads();

        if (threadIdx.x == 0) {
            claimed_ready_comm = false;
            claimed_comm_task = -1;
            // Target-based role assignment: read target with relaxed load,
            // decide role based on blockIdx.  No CAS needed.
            const unsigned int target = scheduler::load_u32(G.target_comm_helpers);
            helping_comm = ((unsigned int)blockIdx.x < target &&
                            (unsigned int)blockIdx.x < (unsigned int)comm_helper_cap);
            if (helping_comm) {
                claimed_ready_comm = try_claim_ready_comm_task(
                    G,
                    total_comm_tasks,
                    row_blocks,
                    col_blocks,
                    super_rows,
                    final_rows,
                    super_blocks,
                    claimed_comm_task
                );
                if (!claimed_ready_comm) {
                    atomicAdd(G.failed_comm_claims, 1u);
                }
            }
        }
        __syncthreads();
        if (claimed_ready_comm) {
            const int task_id =
                comm_linear_to_task_id(claimed_comm_task, G.dev_idx);
            if (threadIdx.x == 0) {
                atomicAdd(G.total_comm_starts, 1u);
            }
            unsigned long long _ct0 = clock64();
            comm_tile(G, task_id, row_blocks, col_blocks, super_rows,
                      final_rows, super_blocks);
            if (threadIdx.x == 0 && G.comm_cycles != nullptr)
                atomicAdd(G.comm_cycles, (unsigned int)((clock64() - _ct0) >> 10));
            if (threadIdx.x == 0) {
                ++local_done_comm_accum;
                if (local_done_comm_accum >= DONE_FLUSH_INTERVAL) {
                    atomicAdd(G.done_comm, local_done_comm_accum);
                    local_done_comm_accum = 0;
                }
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            next_comp_task = atomicAdd(G.next_compute, 1u);
        }
        __syncthreads();
        if (next_comp_task < num_blocks) {
            unsigned long long _ct1 = clock64();
            compute_tile(G, next_comp_task, inputs, outputs, inputs_arrived,
                         inputs_finished, outputs_arrived, outputs_finished, stage,
                         phasebits, row_blocks, col_blocks, super_rows, final_rows,
                         super_blocks, num_iters);
            if (threadIdx.x == 0 && G.compute_cycles != nullptr)
                atomicAdd(G.compute_cycles, (unsigned int)((clock64() - _ct1) >> 10));
            if (threadIdx.x == 0) {
                ++local_done_compute_accum;
                if (local_done_compute_accum >= DONE_FLUSH_INTERVAL) {
                    atomicAdd(G.done_compute, local_done_compute_accum);
                    local_done_compute_accum = 0;
                }
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            // No try_leave needed with target-based assignment.
            claimed_comm_task = atomicAdd(G.next_comm, 1u);
        }
        __syncthreads();
        if (claimed_comm_task >= total_comm_tasks) {
            break;
        }
        const int task_id = comm_linear_to_task_id(claimed_comm_task, G.dev_idx);
        if (threadIdx.x == 0) {
            atomicAdd(G.total_comm_starts, 1u);
        }
        unsigned long long _ct2 = clock64();
        comm_tile(G, task_id, row_blocks, col_blocks, super_rows, final_rows,
                  super_blocks);
        if (threadIdx.x == 0 && G.comm_cycles != nullptr)
            atomicAdd(G.comm_cycles, (unsigned int)((clock64() - _ct2) >> 10));
        if (threadIdx.x == 0) {
            ++local_done_comm_accum;
            if (local_done_comm_accum >= DONE_FLUSH_INTERVAL) {
                atomicAdd(G.done_comm, local_done_comm_accum);
                local_done_comm_accum = 0;
            }
        }
        __syncthreads();
    }

    // Flush remaining done counters after loop exit.
    if (threadIdx.x == 0) {
        if (local_done_compute_accum > 0)
            atomicAdd(G.done_compute, local_done_compute_accum);
        if (local_done_comm_accum > 0)
            atomicAdd(G.done_comm, local_done_comm_accum);
    }
}

__device__ inline void epilogue_kernel(const globals& G) {
    const int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride) {
        G.barrier[G.dev_idx][{i / col_blocks, i % col_blocks}] = 0;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        barrier_all(G.barrier, {1, 0, 0}, G.dev_idx);
    }
}

// Cache-line padded counter struct — each hot counter gets its own 128-byte
// L2 cache line to prevent false sharing, mirroring the AG+GEMM counters.
struct alignas(128) SchedulerCounters {
    unsigned int next_compute;              char _pad0[124];   // Slot 0: hot write (all CTAs claiming tiles)
    unsigned int next_comm;                 char _pad1[124];   // Slot 1: hot write (comm CTAs)
    unsigned int active_comm_helpers;       char _pad2[124];   // Slot 2: hot write (role changes)
    unsigned int done_compute;              char _pad3[124];   // Slot 3: hot write (all CTAs per tile)
    unsigned int done_comm;                 char _pad4[124];   // Slot 4: hot write (comm CTAs per tile)
    unsigned int last_done_compute;         char _pad5[124];   // Slot 5: controller read
    unsigned int last_done_comm;            char _pad6[124];   // Slot 6: controller read
    unsigned int target_comm_helpers;       char _pad7[124];   // Slot 7: hot read (all CTAs per iter)
    unsigned int stable_count;              char _pad8[124];   // Slot 8: controller backoff
    // Diagnostic counters (less hot, but still padded for consistency)
    unsigned int successful_comm_claims;    char _pad9[124];   // Slot 9
    unsigned int failed_comm_claims;        char _pad10[124];  // Slot 10
    unsigned int last_successful_comm_claims; char _pad11[124]; // Slot 11
    unsigned int last_failed_comm_claims;   char _pad12[124];  // Slot 12
    unsigned int total_comm_starts;         char _pad13[124];  // Slot 13
    unsigned int last_total_comm_starts;    char _pad14[124];  // Slot 14
    unsigned int max_failed_ready_value;    char _pad15[124];  // Slot 15
    unsigned int cas_failed_claims;         char _pad16[124];  // Slot 16
    unsigned int compute_cycles;            char _pad17[124];  // Slot 17: cycle counter for compute tiles
    unsigned int comm_cycles;               char _pad18[124];  // Slot 18: cycle counter for comm tiles
    unsigned int last_compute_cycles;       char _pad19[124];  // Slot 19: snapshot for controller
    unsigned int last_comm_cycles;          char _pad20[124];  // Slot 20: snapshot for controller
    unsigned int ema_cost_primary;           char _pad21[124];  // Slot 21: EMA cost primary (float-as-uint)
    unsigned int ema_cost_secondary;         char _pad22[124];  // Slot 22: EMA cost secondary (float-as-uint)
};
static_assert(sizeof(SchedulerCounters) == 23 * 128, "SchedulerCounters size mismatch");

static SchedulerCounters* g_sched_counters[globals::NUM_DEVICES] = {nullptr};
#ifdef GEMM_AR_PRINT_SCHED_STATS
static unsigned int* g_trace_buf[globals::NUM_DEVICES] = {nullptr};
static unsigned int* g_trace_idx[globals::NUM_DEVICES] = {nullptr};
static unsigned long long* g_trace_time_buf[globals::NUM_DEVICES] = {nullptr};
#endif

void entrypoint(
    const at::Tensor& A,
    const at::Tensor& B,
    kittens::py::TKParallelTensor& C,
    kittens::py::TKParallelTensor& barrier) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");
    TORCH_CHECK(A.dtype() == at::ScalarType::BFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be rank-2");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul shape mismatch: A[M,K], B[K,N]");

    kittens::py::parallel_tensor_check(C, barrier);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    TORCH_CHECK(M % globals::ROW_BLOCK == 0,
                "M must be divisible by ROW_BLOCK (", globals::ROW_BLOCK, ")");
    TORCH_CHECK(K % globals::RED_BLOCK == 0,
                "K must be divisible by RED_BLOCK (", globals::RED_BLOCK, ")");
    TORCH_CHECK(N % globals::COL_BLOCK == 0,
                "N must be divisible by COL_BLOCK (", globals::COL_BLOCK, ")");
    TORCH_CHECK(C.data_.dtype() == at::ScalarType::BFloat16, "C must be bf16");
    TORCH_CHECK(C.data_.size(0) == M && C.data_.size(1) == N,
                "C shape must be [M, N]");
    TORCH_CHECK(barrier.data_.dtype() == at::ScalarType::Int,
                "barrier tensor dtype must be int32");

    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

#ifdef GEMM_AR_PRINT_SCHED_STATS
    if (g_trace_buf[dev_idx] == nullptr) {
        cudaMalloc(&g_trace_buf[dev_idx], globals::TRACE_MAX * 11 * sizeof(unsigned int));
        cudaMalloc(&g_trace_idx[dev_idx], sizeof(unsigned int));
        cudaMalloc(&g_trace_time_buf[dev_idx], globals::TRACE_MAX * sizeof(unsigned long long));
    }
    cudaMemsetAsync(g_trace_buf[dev_idx], 0, globals::TRACE_MAX * 11 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_idx[dev_idx], 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_time_buf[dev_idx], 0, globals::TRACE_MAX * sizeof(unsigned long long), stream);
#endif

    if (g_sched_counters[dev_idx] == nullptr)
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    cudaMemsetAsync(g_sched_counters[dev_idx], 0, sizeof(SchedulerCounters), stream);
    SchedulerCounters* sc = g_sched_counters[dev_idx];

    const int row_blocks_h = M / globals::ROW_BLOCK;
    const int col_blocks_h = N / globals::COL_BLOCK;
    const int num_blocks_h = row_blocks_h * col_blocks_h;
    const int total_comm_h = total_comm_tasks_for_device(num_blocks_h, dev_idx);
    {
        const unsigned int comm_cap_h = min(
            (unsigned int)total_comm_h,
            (unsigned int)(config::NUM_BLOCKS - 1));
        const scheduler::WorkBalanceConfig wb_init{
            .total_ctas = (unsigned int)config::NUM_BLOCKS,
            .total_primary = (unsigned int)num_blocks_h,
            .total_secondary = (unsigned int)total_comm_h,
            .helper_floor = COMM_FLOOR_HELPERS,
            .helper_cap = comm_cap_h,
            .secondary_blocks_primary = false,
            .ema_alpha = 0.3f,
            .ema_cost_primary = nullptr,
            .ema_cost_secondary = nullptr,
        };
        const unsigned int initial_target_comm_helpers =
            scheduler::compute_initial_target_wb(wb_init);
        cudaMemcpyAsync(&sc->target_comm_helpers, &initial_target_comm_helpers,
                        sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    }

    globals G{
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<globals::C_pgl>(C),
        .barrier =
            kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .next_compute = &sc->next_compute,
        .next_comm = &sc->next_comm,
        .active_comm_helpers = &sc->active_comm_helpers,
        .done_compute = &sc->done_compute,
        .done_comm = &sc->done_comm,
        .last_done_compute = &sc->last_done_compute,
        .last_done_comm = &sc->last_done_comm,
        .target_comm_helpers = &sc->target_comm_helpers,
        .compute_cycles = &sc->compute_cycles,
        .comm_cycles = &sc->comm_cycles,
        .last_compute_cycles = &sc->last_compute_cycles,
        .last_comm_cycles = &sc->last_comm_cycles,
        // CTA-0-only controller.
        .controller_lock = nullptr,
        .stable_count = &sc->stable_count,  // O-C: enable exponential backoff
        .ema_cost_primary = &sc->ema_cost_primary,
        .ema_cost_secondary = &sc->ema_cost_secondary,
        .successful_comm_claims = &sc->successful_comm_claims,
        .failed_comm_claims = &sc->failed_comm_claims,
        .last_successful_comm_claims = &sc->last_successful_comm_claims,
        .last_failed_comm_claims = &sc->last_failed_comm_claims,
        .total_comm_starts = &sc->total_comm_starts,
        .last_total_comm_starts = &sc->last_total_comm_starts,
        .max_failed_ready_value = &sc->max_failed_ready_value,
        .cas_failed_claims = &sc->cas_failed_claims,
#ifdef GEMM_AR_PRINT_SCHED_STATS
        .trace_buf = g_trace_buf[dev_idx],
        .trace_idx = g_trace_idx[dev_idx],
        .trace_time_buf = g_trace_time_buf[dev_idx],
#endif
    };

    kittens::py::launch_kernel<config, globals, main_kernel>(G);
    kittens::py::launch_kernel<config, globals, epilogue_kernel>(G);
#ifdef GEMM_AR_PRINT_SCHED_STATS
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (dev_idx == 0) {
        SchedulerCounters h;
        cudaMemcpy(&h, g_sched_counters[dev_idx], sizeof(SchedulerCounters), cudaMemcpyDeviceToHost);
        printf("[gemm_ar sched] dev=%d done_compute=%u done_comm=%u final_target=%u active_helpers=%u succ_claims=%u failed_claims=%u total_comm_starts=%u cas_failed_claims=%u max_failed_ready_value=%u\n",
               dev_idx, h.done_compute, h.done_comm, h.target_comm_helpers, h.active_comm_helpers, h.successful_comm_claims, h.failed_comm_claims, h.total_comm_starts, h.cas_failed_claims, h.max_failed_ready_value);
        unsigned int tidx = 0;
        cudaMemcpy(&tidx, g_trace_idx[dev_idx], sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int n_entries = tidx < globals::TRACE_MAX ? tidx : globals::TRACE_MAX;
        if (n_entries > 0) {
            unsigned int* tbuf = new unsigned int[n_entries * 11];
            cudaMemcpy(tbuf, g_trace_buf[dev_idx], n_entries * 11 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("[gemm_ar trace] %u entries\n", n_entries);
            printf("  %6s  %6s  %6s  %6s  %6s  %6s  %8s  %10s  %10s  %10s  %11s  %11s\n",
                   "d_comp", "d_comm", "target", "active", "prev_t", "comp_ct",
                   "done_sr", "succ_clm", "fail_clm", "start_clm", "claim_sr", "eff_comm");
            for (unsigned int i = 0; i < n_entries; i++) {
                const unsigned int active = tbuf[i * 11 + 3];
                const unsigned int delta_done_secondary = tbuf[i * 11 + 5];
                const unsigned int delta_success_claims = tbuf[i * 11 + 6];
                const unsigned int delta_failed_claims = tbuf[i * 11 + 7];
                const unsigned int delta_total_comm_starts = tbuf[i * 11 + 8];
                const unsigned int total_claims =
                    delta_success_claims + delta_failed_claims;
                const float claim_success_rate = total_claims > 0
                    ? (float)delta_success_claims / (float)total_claims
                    : 0.0f;
                const float effective_comm_progress =
                    claim_success_rate * (float)delta_done_secondary;
                printf("  %6u  %6u  %6u  %6u  %6u  %6u  %8u  %10u  %10u  %10u  %11.3f  %11.3f\n",
                       tbuf[i * 11 + 0], tbuf[i * 11 + 1], tbuf[i * 11 + 2],
                       active, tbuf[i * 11 + 4], 132u - active,
                       delta_done_secondary, delta_success_claims,
                       delta_failed_claims, delta_total_comm_starts,
                       claim_success_rate,
                       effective_comm_progress);
            }
            delete[] tbuf;
        }
    }
#endif
}

}  // namespace dynamic_gemm_ar

#include <torch/csrc/utils/pybind.h>

#ifdef GEMM_AR_PRINT_SCHED_STATS
static at::Tensor get_trace_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_gemm_ar::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_gemm_ar::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_gemm_ar::globals::TRACE_MAX);
    auto out = at::zeros({n, 11}, at::TensorOptions().dtype(at::kInt));
    if (n > 0 && dynamic_gemm_ar::g_trace_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int>(), dynamic_gemm_ar::g_trace_buf[dev_idx],
                   (size_t)n * 11 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}

static at::Tensor get_trace_time_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_gemm_ar::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_gemm_ar::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_gemm_ar::globals::TRACE_MAX);
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kLong));
    if (n > 0 && dynamic_gemm_ar::g_trace_time_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int64_t>(), dynamic_gemm_ar::g_trace_time_buf[dev_idx],
                   (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def("matmul_all_reduce_dynamic", &dynamic_gemm_ar::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"));
#ifdef GEMM_AR_PRINT_SCHED_STATS
    m.def("get_trace_buf", &get_trace_buf_py, pybind11::arg("dev_idx"));
    m.def("get_trace_time_buf", &get_trace_time_buf_py, pybind11::arg("dev_idx"));
#endif
}
