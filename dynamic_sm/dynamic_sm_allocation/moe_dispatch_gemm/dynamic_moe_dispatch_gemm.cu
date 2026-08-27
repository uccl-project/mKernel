// Dynamic SM partitioning for MoE Dispatch+GEMM.
//
// Each CTA owns a strided sequence of GEMM tiles, and may temporarily help
// with dispatch when its next GEMM tile is blocked on row readiness.
//
// The scheduler is regulated by a small device-wide completion-based
// controller:
//   - done_dispatch counts completed dispatch blocks;
//   - done_gemm counts completed GEMM tiles;
//   - CTA 0 / thread 0 periodically updates a global target number of active
//     dispatch helpers from the observed completion ratio;
//   - blocked CTAs only join dispatch-help mode when the active helper count is
//     below that global target.
//
// GEMM keeps priority whenever a CTA's next tile is ready; dispatch help exists
// only to unblock row-ready GEMM progress.

#include "policies/scheduler_base.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

#ifndef TK_MOE_H
#define TK_MOE_H 7168
#endif

#ifndef TK_MOE_I
#define TK_MOE_I 2048
#endif

#ifndef TK_MOE_TOP_K
#define TK_MOE_TOP_K 8
#endif

#ifndef TK_MOE_NUM_EXPERTS
#define TK_MOE_NUM_EXPERTS 256
#endif

namespace dynamic_moe {

namespace scheduler = dynamic_sm_allocation::scheduler;

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int NUM_EPILOGUE_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;
static constexpr unsigned int DISPATCH_FLOOR_HELPERS = 1;
static constexpr unsigned int TRACE_MAX = 8192;

struct globals {
    static constexpr int NUM_DEVICES     = TK_NUM_DEVICES;
    static constexpr int H               = TK_MOE_H;
    static constexpr int I               = TK_MOE_I;
    static constexpr int TOP_K           = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS     = TK_MOE_NUM_EXPERTS;
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / NUM_DEVICES;

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M         = 12;
    static constexpr int ROW_BLOCK       = 128;
    static constexpr int COL_BLOCK       = 256;
    static constexpr int RED_BLOCK       = 64;

    using token_vec  = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = 16;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using pre_tokens_pgl              = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl              = gl<bf16, 1, 1, -1, H, token_vec, A_tile>;
    using weights_gl                  = gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_gl                  = gl<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_gl = gl<int,  1, 1,  1, NUM_EXPERTS>;
    using pull_dispatch_indices_gl    = gl<int,  1, 1, -1, 2>;
    using barrier_pgl                 = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

    pre_tokens_pgl              pre_tokens;
    post_tokens_gl              post_tokens;
    weights_gl                  weights;
    outputs_gl                  outputs;
    padded_tokens_per_expert_gl padded_tokens_per_expert;
    pull_dispatch_indices_gl    pull_dispatch_indices;
    barrier_pgl                 barrier;

    const int dev_idx;
    const int num_padded_local_tokens;
    const int comp_only_ctas;
    const int reserved_comm_ctas;
    unsigned int* next_dispatch; // global dispatch-block counter
    unsigned int* next_gemm;     // global flattened GEMM task counter
    unsigned int* active_dispatch_helpers;
    unsigned int* done_dispatch;
    unsigned int* done_gemm;
    unsigned int* last_done_dispatch;
    unsigned int* last_done_gemm;
    unsigned int* target_dispatch_helpers;
    // --- FT cycle + controller_lock counters ---
    unsigned int* gemm_cycles;
    unsigned int* dispatch_cycles;
    unsigned int* last_gemm_cycles;
    unsigned int* last_dispatch_cycles;
    unsigned int* controller_lock;
    unsigned int* stable_count;
    unsigned int* ema_cost_primary;
    unsigned int* ema_cost_secondary;
#ifdef MOE_PRINT_SCHED_STATS
    unsigned int* trace_buf;
    unsigned int* trace_idx;
    unsigned long long* trace_time_buf;
#endif

    struct pipeline_inputs  { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

using scheduler::load_u32;

__device__ inline int load_active_dispatch_helpers(const globals& G) {
    return (int)load_u32(G.active_dispatch_helpers);
}

__device__ inline int load_target_dispatch_helpers(const globals& G) {
    return (int)load_u32(G.target_dispatch_helpers);
}

#ifdef MOE_PRINT_SCHED_STATS
__device__ inline void record_trace(
    const globals& G,
    const unsigned int prev_target,
    const unsigned int next_target,
    const scheduler::WorkSample& sample) {
    if (G.trace_buf == nullptr || G.trace_idx == nullptr) {
        return;
    }
    const unsigned int slot = atomicAdd(G.trace_idx, 1u);
    if (slot >= TRACE_MAX) {
        return;
    }
    G.trace_buf[slot * 8 + 0] = sample.curr_done_primary;
    G.trace_buf[slot * 8 + 1] = sample.curr_done_secondary;
    G.trace_buf[slot * 8 + 2] = next_target;
    G.trace_buf[slot * 8 + 3] = scheduler::load_u32(G.active_dispatch_helpers);
    G.trace_buf[slot * 8 + 4] = prev_target;
    G.trace_buf[slot * 8 + 5] = 0u;  // reserved (was blocked_gemm_observations, never incremented)
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

// Work-balance policy variant of the dispatch controller update.
__device__ inline void maybe_update_dispatch_controller(
    const globals& G,
    const scheduler::WorkBalanceConfig& wb_config) {
    const scheduler::HelperCounters counters{
        .active_helpers = G.active_dispatch_helpers,
        .done_primary = G.done_gemm,
        .done_secondary = G.done_dispatch,
        .last_done_primary = G.last_done_gemm,
        .last_done_secondary = G.last_done_dispatch,
        .target_helpers = G.target_dispatch_helpers,
        .blocked_primary = nullptr,
        .last_blocked_primary = nullptr,
        .primary_cycles = G.gemm_cycles,
        .secondary_cycles = G.dispatch_cycles,
        .last_primary_cycles = G.last_gemm_cycles,
        .last_secondary_cycles = G.last_dispatch_cycles,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
        .controller_lock = nullptr,
        .stable_count = G.stable_count,
    };
#ifdef MOE_PRINT_SCHED_STATS
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
            record_trace(G, prev_target, next_target, sample);
        });
#else
    scheduler::update_target_any_cta(
        counters,
        [&] __device__(unsigned int prev_target,
                       const scheduler::WorkSample& sample) {
#ifdef FIXED_DYNAMIC_TARGET
            (void)prev_target; (void)sample;
            return (unsigned int)FIXED_DYNAMIC_TARGET;
#else
            return scheduler::update_target_work_balance(prev_target, sample, wb_config);
#endif
        });
#endif
}

// ---------- dispatch ----------------------------------------------------------
// claim a block index from the global counter and dispatch those tokens.
__device__ inline void dispatch_block(const globals& G, const int dispatch_block_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];

    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int token_idx = dispatch_block_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (token_idx < G.num_padded_local_tokens) {
            int src_dev_idx   = G.pull_dispatch_indices[{token_idx, 0}];
            int src_token_idx = G.pull_dispatch_indices[{token_idx, 1}];

            if (src_dev_idx >= 0 && src_token_idx >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens[src_dev_idx], {src_token_idx, 0}, token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.post_tokens, token[lane_id], {token_idx, 0});
                tma::store_async_wait();
            }
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                :: "l"(&G.barrier[G.dev_idx][{token_idx / globals::ROW_BLOCK}]), "r"(1)
                : "memory");
        }
    }

    // Dynamic mode reuses a CTA across multiple dispatch tasks, unlike the
    // static kernel's one-dispatch-CTA-per-block mapping. Make sure all
    // threads have finished using the shared token buffer and semaphores
    // before the caller claims another dispatch task.
    __syncthreads();
}

// ---------- group GEMM --------------------------------------------------------
// Each CTA claims one (flattened) GEMM tile at a time from next_gemm.
// A tile is identified by a flattened index over all experts x row_blocks x col_blocks.
__device__ inline void group_gemm_task(
    const globals& G,
    const int flat_task,
    int* padded_tokens_per_expert,
    globals::pipeline_inputs  (&inputs) [globals::PIPELINE_STAGES],
    globals::pipeline_outputs& outputs,
    semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    semaphore& outputs_arrived,
    semaphore& outputs_finished,
    int&      stage,
    uint32_t& phasebits)
{
    constexpr int num_iters   = globals::H / globals::RED_BLOCK;
    constexpr int col_blocks  = globals::I / globals::COL_BLOCK;

    // Decode flat_task → (expert_id, row_idx, col_idx) by walking the experts.
    int remaining = flat_task;
    int expert_id = 0;
    int num_tokens_cum = 0;
    int row_block_start = 0, row_block_end = 0, num_blocks_expert = 0;
    for (; expert_id < globals::NUM_EXPERTS_PER_DEV; ++expert_id) {
        const int n = padded_tokens_per_expert[expert_id];
        row_block_start = num_tokens_cum / globals::ROW_BLOCK;
        num_tokens_cum += n;
        row_block_end   = (num_tokens_cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
        num_blocks_expert = (row_block_end - row_block_start) * col_blocks;
        if (remaining < num_blocks_expert) break;
        remaining -= num_blocks_expert;
    }
    if (expert_id >= globals::NUM_EXPERTS_PER_DEV) return; // out of range

    const int local_row = remaining / col_blocks;
    const int col_idx   = remaining % col_blocks;
    const int row_idx   = row_block_start + local_row;

    // Wait for the dispatch barrier on this row block
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id      = warpgroup::warpid();
    const int lane_id      = warp::laneid();

    if (warpgroup_id == 2) {
        warpgroup::decrease_registers<40>();
        if (warp_id == 0 && lane_id == 0) {
            // TMA load producer
            int bar_val;
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
            while (bar_val != globals::ROW_BLOCK) {
                __nanosleep(16);
                asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                    : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
            }
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                if (red_idx == globals::PIPELINE_STAGES - 1) {
                    wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                    update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::load_async(inputs[stage].A[i], G.post_tokens,
                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                tma::load_async(inputs[stage].B, G.weights,
                                {expert_id, red_idx, col_idx}, inputs_arrived[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) {
            // Store epilogue producer
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
#pragma unroll
            for (int i = 0; i < 2; i++)
                tma::store_async(G.outputs, outputs.C[i], {row_idx * 2 + i, col_idx});
            tma::store_async_read_wait();
            arrive(outputs_finished);
        }
    } else {
        // Consumer warpgroups
        warpgroup::increase_registers<232>();
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

// Fully adaptive variant: every CTA owns a strided GEMM lane and helps with
// dispatch only when its next GEMM tile is still blocked on row readiness.
__global__ __launch_bounds__(NUM_MAIN_THREADS, 1)
void adaptive_dispatch_group_gemm_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        al.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs& outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(&inputs[globals::PIPELINE_STAGES - 1]);

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ unsigned int my_dispatch_task;
    __shared__ bool barrier_ready;
    __shared__ bool have_dispatch;
    __shared__ int padded_ppe[globals::NUM_EXPERTS_PER_DEV];
    __shared__ int s_row_idx;

    if (threadIdx.x == 0) {
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }

    if (threadIdx.x < globals::NUM_EXPERTS_PER_DEV) {
        padded_ppe[threadIdx.x] = G.padded_tokens_per_expert[
            {G.dev_idx * globals::NUM_EXPERTS_PER_DEV + (int)threadIdx.x}];
    }
    __syncthreads();

    const int num_dispatch_blocks =
        (G.num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int col_blocks = globals::I / globals::COL_BLOCK;

    int total_gemm_tasks = 0;
    {
        int cum = 0;
        for (int e = 0; e < globals::NUM_EXPERTS_PER_DEV; ++e) {
            const int rbs = cum / globals::ROW_BLOCK;
            cum += padded_ppe[e];
            const int rbe = (cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
            total_gemm_tasks += (rbe - rbs) * col_blocks;
        }
    }
    const unsigned int dispatch_helper_cap = min(
        (unsigned int)(SM_COUNT - 1),
        max(1u, (unsigned int)num_dispatch_blocks)
    );
    const scheduler::WorkBalanceConfig wb_config{
        .total_ctas = (unsigned int)SM_COUNT,
        .total_primary = (unsigned int)total_gemm_tasks,
        .total_secondary = (unsigned int)num_dispatch_blocks,
        .helper_floor = DISPATCH_FLOOR_HELPERS,
        .helper_cap = dispatch_helper_cap,
        .secondary_blocks_primary = true,  // GEMM depends on dispatch (comm→compute)
        .ema_alpha = 0.3f,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
    };

    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    bool helping_dispatch = false;

    // Batch done_gemm/done_dispatch atomicAdds to reduce L2 contention.
    static constexpr unsigned int DONE_GEMM_FLUSH_INTERVAL = 8;
    static constexpr unsigned int DONE_DISPATCH_FLUSH_INTERVAL = 8;
    unsigned int local_done_accum = 0;
    unsigned int local_dispatch_done_accum = 0;

    // Per-CTA stride: each CTA owns tiles blockIdx.x, blockIdx.x+132, ...
    // Unlike AG+GEMM, these CTAs don't permanently leave the compute path —
    // they only temporarily help dispatch during barrier waits, then return
    // to their next strided GEMM tile. No tail effect from stranded tiles.
    for (int gemm_task = (int)blockIdx.x; gemm_task < total_gemm_tasks; gemm_task += SM_COUNT) {
        // Update controller once per GEMM task (not during barrier-wait)
        maybe_update_dispatch_controller(G, wb_config);
        __syncthreads();

        if (threadIdx.x == 0) {
            int remaining = gemm_task;
            int cum = 0;
            s_row_idx = 0;
            for (int e = 0; e < globals::NUM_EXPERTS_PER_DEV; ++e) {
                const int rbs = cum / globals::ROW_BLOCK;
                cum += padded_ppe[e];
                const int rbe = (cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
                const int nb = (rbe - rbs) * col_blocks;
                if (remaining < nb) {
                    s_row_idx = rbs + remaining / col_blocks;
                    break;
                }
                remaining -= nb;
            }
            int bar_val;
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{s_row_idx}]) : "memory");
            barrier_ready = (bar_val == globals::ROW_BLOCK);
        }
        __syncthreads();

        while (!barrier_ready) {
            // Try dispatch work only if tasks remain (speculative check avoids contention)
            if (threadIdx.x == 0) {
                have_dispatch = false;
                unsigned int cur = scheduler::load_u32(G.next_dispatch);
                if ((int)cur < num_dispatch_blocks) {
                    unsigned int d = atomicAdd(G.next_dispatch, 1u);
                    have_dispatch = ((int)d < num_dispatch_blocks);
                    if (have_dispatch) {
                        my_dispatch_task = d;
                    }
                }
            }
            __syncthreads();

            if (have_dispatch) {
                // Measure dispatch task cost for the cost-balance formula.
                unsigned long long _disp_t0 = clock64();
                dispatch_block(G, (int)my_dispatch_task);
                if (threadIdx.x == 0) {
                    if (G.dispatch_cycles != nullptr)
                        atomicAdd(G.dispatch_cycles,
                                  (unsigned int)((clock64() - _disp_t0) >> 10));
                    ++local_dispatch_done_accum;
                    if (local_dispatch_done_accum >= DONE_DISPATCH_FLUSH_INTERVAL) {
                        atomicAdd(G.done_dispatch, local_dispatch_done_accum);
                        local_dispatch_done_accum = 0;
                    }
                }
            } else {
                if (threadIdx.x == 0) {
                    __nanosleep(16);
                }
            }
            // Re-check barrier (merged sync)
            if (threadIdx.x == 0) {
                int bar_val;
                asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                    : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{s_row_idx}]) : "memory");
                barrier_ready = (bar_val == globals::ROW_BLOCK);
            }
            __syncthreads();
        }

        // Measure GEMM task cost for the cost-balance formula.
        unsigned long long _gemm_t0 = clock64();
        group_gemm_task(G, gemm_task, padded_ppe,
                        inputs, outputs, inputs_arrived, inputs_finished,
                        outputs_arrived, outputs_finished, stage, phasebits);
        // Batched done_gemm — accumulate locally, flush every N tiles.
        if (threadIdx.x == 0) {
            const unsigned int gemm_cost = (unsigned int)((clock64() - _gemm_t0) >> 10);
            if (G.gemm_cycles != nullptr)
                atomicAdd(G.gemm_cycles, gemm_cost);
            ++local_done_accum;
            if (local_done_accum >= DONE_GEMM_FLUSH_INTERVAL) {
                atomicAdd(G.done_gemm, local_done_accum);
                local_done_accum = 0;
            }
        }
        __syncthreads();
    }
    // Flush any remaining done_gemm count.
    if (threadIdx.x == 0 && local_done_accum > 0) {
        atomicAdd(G.done_gemm, local_done_accum);
    }

    while (true) {
        if (threadIdx.x == 0) {
            my_dispatch_task = atomicAdd(G.next_dispatch, 1u);
        }
        __syncthreads();
        if ((int)my_dispatch_task >= num_dispatch_blocks) {
            break;
        }
        unsigned long long _disp_t0_tail = clock64();
        dispatch_block(G, (int)my_dispatch_task);
        if (threadIdx.x == 0) {
            if (G.dispatch_cycles != nullptr)
                atomicAdd(G.dispatch_cycles,
                          (unsigned int)((clock64() - _disp_t0_tail) >> 10));
            ++local_dispatch_done_accum;
            if (local_dispatch_done_accum >= DONE_DISPATCH_FLUSH_INTERVAL) {
                atomicAdd(G.done_dispatch, local_dispatch_done_accum);
                local_dispatch_done_accum = 0;
            }
        }
        __syncthreads();
    }
    // Flush any remaining done_dispatch count.
    if (threadIdx.x == 0 && local_dispatch_done_accum > 0) {
        atomicAdd(G.done_dispatch, local_dispatch_done_accum);
    }
}

// ---------- epilogue ----------------------------------------------------------
__global__ __launch_bounds__(NUM_EPILOGUE_THREADS)
void epilogue_kernel(const __grid_constant__ globals G) {
    const int num_blocks = (G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x  * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        G.barrier[G.dev_idx][{i}] = 0;
}

// ---------- host counters ------------------------------------------------------
// Cache-line padded counter struct — each hot counter gets its own 128-byte
// L2 cache line to prevent false sharing.  Mirrors AG+GEMM's SchedulerCounters.
struct alignas(128) SchedulerCounters {
    unsigned int next_dispatch;                char _pad0[124];   // Slot 0: hot write (dispatch CTAs)
    unsigned int next_gemm;                    char _pad1[124];   // Slot 1: hot write (all CTAs claiming tiles)
    unsigned int active_dispatch_helpers;      char _pad2[124];   // Slot 2: hot write (role changes)
    unsigned int done_dispatch;                char _pad3[124];   // Slot 3: (proxy via next_dispatch)
    unsigned int done_gemm;                    char _pad4[124];   // Slot 4: hot write (all CTAs per tile)
    unsigned int last_done_dispatch;           char _pad5[124];   // Slot 5: controller read
    unsigned int last_done_gemm;               char _pad6[124];   // Slot 6: controller read
    unsigned int target_dispatch_helpers;      char _pad7[124];   // Slot 7: hot read (all CTAs per iter)
    unsigned int stable_count;                 char _pad8[124];   // Slot 8: controller backoff
    unsigned int gemm_cycles;                  char _pad9[124];   // Slot 9: cycle-weighted GEMM cost
    unsigned int dispatch_cycles;              char _pad10[124];  // Slot 10: cycle-weighted dispatch cost
    unsigned int last_gemm_cycles;             char _pad11[124];  // Slot 11: controller snapshot
    unsigned int last_dispatch_cycles;         char _pad12[124];  // Slot 12: controller snapshot
    unsigned int ema_cost_primary;             char _pad13[124];  // Slot 13: EMA cost primary (float-as-uint)
    unsigned int ema_cost_secondary;           char _pad14[124];  // Slot 14: EMA cost secondary (float-as-uint)
};
static_assert(sizeof(SchedulerCounters) == 15 * 128, "SchedulerCounters size mismatch");

static SchedulerCounters* g_sched_counters[globals::NUM_DEVICES] = {nullptr};
#ifdef MOE_PRINT_SCHED_STATS
static unsigned int* g_trace_buf[globals::NUM_DEVICES] = {nullptr};
static unsigned int* g_trace_idx[globals::NUM_DEVICES] = {nullptr};
static unsigned long long* g_trace_time_buf[globals::NUM_DEVICES] = {nullptr};
#endif

} // namespace dynamic_moe

// ---------- Python entrypoint -------------------------------------------------
void entrypoint_dynamic_moe(
    kittens::py::TKParallelTensor& pre_tokens,
    at::Tensor& post_tokens,
    at::Tensor& weights,
    at::Tensor& outputs,
    at::Tensor& padded_tokens_per_expert,
    at::Tensor& pull_dispatch_indices,
    kittens::py::TKParallelTensor& barrier,
    const int num_padded_local_tokens)
{
    using namespace dynamic_moe;

    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_sched_counters[dev_idx] == nullptr)
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    cudaMemsetAsync(g_sched_counters[dev_idx], 0, sizeof(SchedulerCounters), stream);
    SchedulerCounters* sc = g_sched_counters[dev_idx];
#ifdef MOE_PRINT_SCHED_STATS
    if (g_trace_buf[dev_idx] == nullptr) {
        cudaMalloc(&g_trace_buf[dev_idx], TRACE_MAX * 8 * sizeof(unsigned int));
        cudaMalloc(&g_trace_idx[dev_idx], sizeof(unsigned int));
        cudaMalloc(&g_trace_time_buf[dev_idx], TRACE_MAX * sizeof(unsigned long long));
    }
    cudaMemsetAsync(g_trace_buf[dev_idx], 0, TRACE_MAX * 8 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_idx[dev_idx], 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_time_buf[dev_idx], 0, TRACE_MAX * sizeof(unsigned long long), stream);
#endif

    const int num_dispatch_blocks_h =
        (num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int col_blocks_h = globals::I / globals::COL_BLOCK;
    const int approx_gemm_tasks_h =
        ((num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK) * col_blocks_h;
    {
        // H-adaptive initial dispatch target — analogous to AG+GEMM's K-adaptive
        // comm target.  The GEMM reduction depth (H / RED_BLOCK) determines how
        // long each CTA is occupied per tile.  Deeper reduction → CTAs stuck in
        // GEMM longer → need more initial dispatch helpers to fill barriers.
        //
        //   H_blocks = H / (RED_BLOCK * 16)          (compute cost proxy, = 7)
        //   h_adaptive_cap = max(4, min(dispatch_blocks, SM_COUNT) / (1 + H_blocks))
        //
        // For H=7168 this gives cap = max(4, 132/8) = 16; the policy then ramps
        // toward the steady-state optimum of roughly 44.
        const unsigned int dispatch_cap_h = min(
            (unsigned int)(SM_COUNT - 1),
            max(1u, (unsigned int)num_dispatch_blocks_h));
        constexpr unsigned int H_blocks = globals::H / (globals::RED_BLOCK * 16u);
        const unsigned int h_adaptive_cap = max(
            4u,
            min((unsigned int)num_dispatch_blocks_h, (unsigned int)SM_COUNT) / (1u + H_blocks));
        const unsigned int initial_target_dispatch_helpers =
            min(h_adaptive_cap, dispatch_cap_h);
        cudaMemcpyAsync(&sc->target_dispatch_helpers, &initial_target_dispatch_helpers, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, stream);
    }

    globals G {
        .pre_tokens   = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .post_tokens  = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights      = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs      = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert =
            kittens::py::tensor_to_gl<globals::padded_tokens_per_expert_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices =
            kittens::py::tensor_to_gl<globals::pull_dispatch_indices_gl>(pull_dispatch_indices),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_padded_local_tokens = num_padded_local_tokens,
        .comp_only_ctas  = 0,
        .reserved_comm_ctas = 0,
        .next_dispatch = &sc->next_dispatch,
        .next_gemm     = &sc->next_gemm,
        .active_dispatch_helpers = &sc->active_dispatch_helpers,
        .done_dispatch = &sc->done_dispatch,
        .done_gemm = &sc->done_gemm,
        .last_done_dispatch = &sc->last_done_dispatch,
        .last_done_gemm = &sc->last_done_gemm,
        .target_dispatch_helpers = &sc->target_dispatch_helpers,
        .gemm_cycles = &sc->gemm_cycles,
        .dispatch_cycles = &sc->dispatch_cycles,
        .last_gemm_cycles = &sc->last_gemm_cycles,
        .last_dispatch_cycles = &sc->last_dispatch_cycles,
        .controller_lock = nullptr,
        .stable_count = &sc->stable_count,  // enable exponential backoff
        .ema_cost_primary = &sc->ema_cost_primary,
        .ema_cost_secondary = &sc->ema_cost_secondary,
#ifdef MOE_PRINT_SCHED_STATS
        .trace_buf = g_trace_buf[dev_idx],
        .trace_idx = g_trace_idx[dev_idx],
        .trace_time_buf = g_trace_time_buf[dev_idx],
#endif
    };

    cudaFuncSetAttribute(adaptive_dispatch_group_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    adaptive_dispatch_group_gemm_kernel<<<SM_COUNT, NUM_MAIN_THREADS,
                                          DYNAMIC_SHARED_MEMORY, stream>>>(G);

    const int epilogue_blocks =
        ((G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK
         + NUM_EPILOGUE_THREADS - 1) / NUM_EPILOGUE_THREADS;
    epilogue_kernel<<<epilogue_blocks, NUM_EPILOGUE_THREADS, 0, stream>>>(G);
}

void entrypoint_adaptive_moe(
    kittens::py::TKParallelTensor& pre_tokens,
    at::Tensor& post_tokens,
    at::Tensor& weights,
    at::Tensor& outputs,
    at::Tensor& padded_tokens_per_expert,
    at::Tensor& pull_dispatch_indices,
    kittens::py::TKParallelTensor& barrier,
    const int num_padded_local_tokens)
{
    using namespace dynamic_moe;

    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_sched_counters[dev_idx] == nullptr)
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    cudaMemsetAsync(g_sched_counters[dev_idx], 0, sizeof(SchedulerCounters), stream);
    SchedulerCounters* sc2 = g_sched_counters[dev_idx];

    const int num_dispatch_blocks_h2 =
        (num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int col_blocks_h2 = globals::I / globals::COL_BLOCK;
    const int approx_gemm_tasks_h2 =
        ((num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK) * col_blocks_h2;
    {
        // H-adaptive initial dispatch target (same formula as entrypoint_dynamic_moe).
        const unsigned int dispatch_cap_h2 = min(
            (unsigned int)(SM_COUNT - 1),
            max(1u, (unsigned int)num_dispatch_blocks_h2));
        constexpr unsigned int H_blocks2 = globals::H / (globals::RED_BLOCK * 16u);
        const unsigned int h_adaptive_cap2 = max(
            4u,
            min((unsigned int)num_dispatch_blocks_h2, (unsigned int)SM_COUNT) / (1u + H_blocks2));
        const unsigned int initial_target_dispatch_helpers =
            min(h_adaptive_cap2, dispatch_cap_h2);
        cudaMemcpyAsync(&sc2->target_dispatch_helpers, &initial_target_dispatch_helpers, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, stream);
    }

    globals G {
        .pre_tokens   = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .post_tokens  = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights      = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs      = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert =
            kittens::py::tensor_to_gl<globals::padded_tokens_per_expert_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices =
            kittens::py::tensor_to_gl<globals::pull_dispatch_indices_gl>(pull_dispatch_indices),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_padded_local_tokens = num_padded_local_tokens,
        .comp_only_ctas = 0,
        .reserved_comm_ctas = 0,
        .next_dispatch = &sc2->next_dispatch,
        .next_gemm = &sc2->next_gemm,
        .active_dispatch_helpers = &sc2->active_dispatch_helpers,
        .done_dispatch = &sc2->done_dispatch,
        .done_gemm = &sc2->done_gemm,
        .last_done_dispatch = &sc2->last_done_dispatch,
        .last_done_gemm = &sc2->last_done_gemm,
        .target_dispatch_helpers = &sc2->target_dispatch_helpers,
        .gemm_cycles = &sc2->gemm_cycles,
        .dispatch_cycles = &sc2->dispatch_cycles,
        .last_gemm_cycles = &sc2->last_gemm_cycles,
        .last_dispatch_cycles = &sc2->last_dispatch_cycles,
        .controller_lock = nullptr,
        .stable_count = &sc2->stable_count,  // enable exponential backoff
        .ema_cost_primary = &sc2->ema_cost_primary,
        .ema_cost_secondary = &sc2->ema_cost_secondary,
#ifdef MOE_PRINT_SCHED_STATS
        .trace_buf = nullptr,
        .trace_idx = nullptr,
        .trace_time_buf = nullptr,
#endif
    };

    cudaFuncSetAttribute(adaptive_dispatch_group_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    adaptive_dispatch_group_gemm_kernel<<<SM_COUNT, NUM_MAIN_THREADS,
                                          DYNAMIC_SHARED_MEMORY, stream>>>(G);

    const int epilogue_blocks =
        ((G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK
         + NUM_EPILOGUE_THREADS - 1) / NUM_EPILOGUE_THREADS;
    epilogue_kernel<<<epilogue_blocks, NUM_EPILOGUE_THREADS, 0, stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

#ifdef MOE_PRINT_SCHED_STATS
static at::Tensor get_trace_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_moe::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_moe::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_moe::TRACE_MAX);
    auto out = at::zeros({n, 8}, at::TensorOptions().dtype(at::kInt));
    if (n > 0 && dynamic_moe::g_trace_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int>(), dynamic_moe::g_trace_buf[dev_idx],
                   (size_t)n * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}

static at::Tensor get_trace_time_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_moe::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_moe::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_moe::TRACE_MAX);
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kLong));
    if (n > 0 && dynamic_moe::g_trace_time_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int64_t>(), dynamic_moe::g_trace_time_buf[dev_idx],
                   (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def("moe_dispatch_gemm_dynamic", &entrypoint_dynamic_moe,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("barrier"),
          pybind11::arg("num_padded_local_tokens"));
    m.def("moe_dispatch_gemm_adaptive", &entrypoint_adaptive_moe,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("barrier"),
          pybind11::arg("num_padded_local_tokens"));
#ifdef MOE_PRINT_SCHED_STATS
    m.def("get_trace_buf", &get_trace_buf_py, pybind11::arg("dev_idx"));
    m.def("get_trace_time_buf", &get_trace_time_buf_py, pybind11::arg("dev_idx"));
#endif
}
