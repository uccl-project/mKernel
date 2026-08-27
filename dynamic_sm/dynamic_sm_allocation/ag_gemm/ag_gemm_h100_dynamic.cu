#include "ag_gemm_h100_common.cuh"
#include "policies/scheduler_base.cuh"
#include "policies/work_balance.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

namespace ag_gemm_dynamic {

namespace scheduler = dynamic_sm_allocation::scheduler;

// Flush done_compute/done_comm local accumulators every this many tiles.
// Must be < RECHECK_MASK+1 (=64) so the backoff recheck gate stays accurate.
static constexpr unsigned int DONE_COMPUTE_FLUSH_INTERVAL = 8;
static constexpr unsigned int DONE_COMM_FLUSH_INTERVAL = 8;

// Maximum trace entries for scheduler profiling.
static constexpr unsigned int TRACE_MAX = 8192;

struct dynamic_globals {
    static constexpr int NUM_DEVICES = globals::NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = globals::PIPELINE_STAGES;
    static constexpr int SUPER_M = globals::SUPER_M;
    static constexpr int ROW_BLOCK = globals::ROW_BLOCK;
    static constexpr int COL_BLOCK = globals::COL_BLOCK;
    static constexpr int RED_BLOCK = globals::RED_BLOCK;
    static constexpr int NUM_CHUNKS = globals::NUM_CHUNKS;
    static constexpr int COMM_NUM_WORKERS = globals::COMM_NUM_WORKERS;

    using A_tile = globals::A_tile;
    using A_comm_tile = globals::A_comm_tile;
    using B_tile = globals::B_tile;
    using C_tile = globals::C_tile;
    using A_pgl = globals::A_pgl;
    using B_gl = globals::B_gl;
    using C_gl = globals::C_gl;
    using barrier_pgl = globals::barrier_pgl;
    using pipeline_inputs = globals::pipeline_inputs;
    using pipeline_outputs = globals::pipeline_outputs;



    A_pgl A;
    B_gl B;
    C_gl C;
    barrier_pgl barrier;

    const int dev_idx;
    const unsigned int active_sms;  // Runtime CTA count (default: config::NUM_BLOCKS)
    unsigned int* comp_next_tile;
    unsigned int* comm_next_tile;
    unsigned int* active_comm_helpers;
    unsigned int* done_compute;
    unsigned int* done_comm;
    unsigned int* last_done_compute;
    unsigned int* last_done_comm;
    unsigned int* target_comm_helpers;
    unsigned int* blocked_remote_observations;
    unsigned int* last_blocked_remote_observations;
    unsigned int* blocked_remote_cycles;
    unsigned int* last_blocked_remote_cycles;
    unsigned int* ready_comm_tiles;
    unsigned int* last_ready_comm_tiles;
    // --- FT cycle + controller_lock counters ---
    unsigned int* compute_cycles;
    unsigned int* comm_cycles;
    unsigned int* last_compute_cycles;
    unsigned int* last_comm_cycles;
    unsigned int* controller_lock;
    unsigned int* stable_count;
    unsigned int* ema_cost_primary;
    unsigned int* ema_cost_secondary;
#ifdef AG_GEMM_PRINT_SCHED_STATS
    unsigned int* trace_buf;
    unsigned int* trace_idx;
    unsigned long long* trace_time_buf;
#endif
};


__device__ inline void epilogue_kernel_wrapper(const dynamic_globals& G) {
    epilogue_kernel<dynamic_globals>(G);
}

using scheduler::load_u32;


// Work-balance policy variant of the comm controller update for AG+GEMM.
// With CTA-0-only controller (controller_lock == nullptr), only CTA-0 runs this.
// Uses done_comm (finish rate) for accurate cost measurement.
__device__ __forceinline__ void maybe_update_comm_controller(
    const dynamic_globals& G,
    const scheduler::WorkBalanceConfig& wb_config,
    const int local_row_blocks_comm,
    const int col_blocks_comm) {
    // Backoff: once the target has been stable for BACKOFF_THRESH consecutive
    // evaluations, skip ALL controller work (including the ready_comm scan) until
    // done_compute crosses the next RECHECK_MASK+1 boundary.  This cuts ~20
    // global memory ops per iteration down to 2 reads during stable periods.
    if (threadIdx.x == 0 &&
        G.stable_count != nullptr &&
        scheduler::load_u32(G.stable_count) >= scheduler::BACKOFF_THRESH) {
        if ((scheduler::load_u32(G.done_compute) & scheduler::RECHECK_MASK) != 0u)
            return;
    }

    const scheduler::HelperCounters counters{
        .active_helpers = G.active_comm_helpers,
        .done_primary = G.done_compute,
        .done_secondary = G.done_comm,
        .last_done_primary = G.last_done_compute,
        .last_done_secondary = G.last_done_comm,
        .target_helpers = G.target_comm_helpers,
        .blocked_primary = G.blocked_remote_observations,
        .last_blocked_primary = G.last_blocked_remote_observations,
        .blocked_primary_cycles = G.blocked_remote_cycles,
        .last_blocked_primary_cycles = G.last_blocked_remote_cycles,
        .ready_secondary = G.ready_comm_tiles,
        .last_ready_secondary = G.last_ready_comm_tiles,
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
            return scheduler::update_target_work_balance(prev_target, sample, wb_config);
        },
        [&] __device__(unsigned int prev_target,
                       unsigned int next_target,
                       const scheduler::WorkSample& sample) {
#ifdef AG_GEMM_PRINT_SCHED_STATS
            if (G.trace_buf != nullptr && G.trace_idx != nullptr) {
                unsigned int slot = atomicAdd(G.trace_idx, 1u);
                if (slot < TRACE_MAX) {
                    G.trace_buf[slot * 7 + 0] = sample.curr_done_primary;
                    G.trace_buf[slot * 7 + 1] = sample.curr_done_secondary;
                    G.trace_buf[slot * 7 + 2] = next_target;
                    G.trace_buf[slot * 7 + 3] = scheduler::load_u32(G.active_comm_helpers);
                    G.trace_buf[slot * 7 + 4] = prev_target;
                    // Per-task cost (float-as-uint): matches work_balance WINDOWED computation.
                    float trace_C_p = (sample.delta_primary > 0)
                        ? fmaxf((float)sample.delta_primary_cycles / (float)sample.delta_primary, 1.0f)
                        : 0.0f;
                    // Comm: done_comm batches flushes (interval=8) but each CTA
                    // processes <8 rows, so done_comm stays 0 mid-kernel.  Use
                    // comm_next_tile (claimed rows) as a progress proxy — it's
                    // updated every claim via atomicAdd.  comm_cycles is also
                    // updated per-task, so cumulative avg = cycles / claimed_rows.
                    unsigned int delta_s = max(sample.delta_secondary, sample.delta_ready_secondary);
                    float trace_C_s;
                    if (delta_s > 0) {
                        trace_C_s = fmaxf((float)sample.delta_secondary_cycles / (float)delta_s, 1.0f);
                    } else {
                        // Fallback: cumulative cost from comm_next_tile (claimed rows).
                        unsigned int claimed = scheduler::load_u32(G.comm_next_tile);
                        trace_C_s = (claimed > 0 && sample.curr_secondary_cycles > 0)
                            ? (float)sample.curr_secondary_cycles / (float)claimed
                            : 0.0f;
                    }
                    G.trace_buf[slot * 7 + 5] = __float_as_uint(trace_C_p);
                    G.trace_buf[slot * 7 + 6] = __float_as_uint(trace_C_s);
                    if (G.trace_time_buf != nullptr) {
                        G.trace_time_buf[slot] = scheduler::read_globaltimer_ns();
                    }
                }
            }
#else
            (void)prev_target; (void)next_target; (void)sample;
#endif
        });
}

__device__ inline void main_kernel(const dynamic_globals& G) {
    // CTAs beyond active_sms exit early — supports multi-tenant scenarios
    // where fewer than 132 SMs are available to this kernel.
    if (blockIdx.x >= G.active_sms) return;

    extern __shared__ int __shm[];

    const int num_iters = G.A.cols() / globals::RED_BLOCK;
    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int warp_id = warp::groupid();
    const int lane_id = warp::laneid();

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore comm_inputs_arrived[globals::NUM_CHUNKS];
    __shared__ unsigned int next_tile;
    __shared__ globals::pipeline_inputs* comp_inputs_ptr;
    __shared__ globals::pipeline_outputs* comp_outputs_ptr;
    __shared__ typename globals::A_comm_tile* comm_A_smem_ptr;
    __shared__ unsigned int comm_task_id;
    __shared__ bool helping_comm;
    __shared__ bool comm_exhausted;
    // Local accumulators — flushed every FLUSH_INTERVAL tiles to reduce atomicAdd pressure.
    __shared__ unsigned int local_done_accum;       // done_compute
    __shared__ unsigned int local_done_comm_accum;  // done_comm
    uint32_t comp_phasebits = 0xFFFF0000;
    uint32_t comm_phasebits = 0xFFFF0000;
    const int global_row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const unsigned int total_comp_tiles =
        (unsigned int)global_row_blocks * (unsigned int)col_blocks;
    const int global_row_blocks_comm = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int local_row_blocks_comm = global_row_blocks_comm / globals::NUM_DEVICES;
    const int col_blocks_comm = G.A.cols() / (globals::RED_BLOCK * 2);
    // Row-based comm: comm_next_tile indexes ROWS (0..local_row_blocks_comm-1).
    // Each comm CTA claims 1 row and processes all col_blocks_comm columns using
    // 3 warpgroups in parallel with TMA store pipelining (comm_sm_row).
    // Signals once per row with count=col_blocks_comm (1 NVLink multicast op).
    const unsigned int total_comm_tiles = (unsigned int)local_row_blocks_comm;
    // Cap: max comm CTAs = rows (1 CTA per row), bounded at 3/4 * active_sms
    const unsigned int comm_helper_cap = min(
        total_comm_tiles > 0 ? total_comm_tiles : 1u,
        max(1u, ((G.active_sms - 1u) * 3u) / 4u));
    // K-adaptive initial target: for small K, more comm CTAs are needed to
    // prevent AG blocking compute (TMA store to NVLink is slow).  For large K
    // (expensive GEMM tiles), compute dominates so fewer comm CTAs suffice.
    // K-adaptive cap = max(4, NUM_BLOCKS / (1 + K / (RED_BLOCK×16))):
    //   K=2048: cap=44   K=4096: cap=26   K=8192: cap=14   K=16384: cap=7   K=32768: cap=4
    // Floor = min(local_row_blocks_comm, k_adaptive_cap): 1 CTA per row (parallel).
    const unsigned int K_blocks = (unsigned int)G.A.cols() / (globals::RED_BLOCK * 16u);
    const unsigned int k_adaptive_cap =
        max(4u, G.active_sms / (1u + K_blocks));
    // comm_floor = min(rows, K-adaptive cap): want ≥1 CTA per row but bounded.
    const unsigned int comm_floor_dynamic =
        min((unsigned int)local_row_blocks_comm, k_adaptive_cap);
    const scheduler::WorkBalanceConfig wb_config{
        .total_ctas = G.active_sms,
        .total_primary = total_comp_tiles,
        .total_secondary = total_comm_tiles,
        .helper_floor = comm_floor_dynamic,
        .helper_cap = comm_helper_cap,
        .secondary_blocks_primary = true,  // comm→compute dependency
        .ema_alpha = 0.3f,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
    };
    int stage = 0;

    if (threadIdx.x == 0) {
        comm_exhausted = false;
        // All CTAs start as compute; comm helpers are recruited dynamically
        // via the normal try_join / try_leave mechanism.
        helping_comm = false;
        local_done_accum = 0;
        local_done_comm_accum = 0;

        // (comp_base_tile removed: tiles now claimed via global atomicAdd)
#pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    if (warp_id < globals::NUM_CHUNKS && lane_id == 0) {
        init_semaphore(comm_inputs_arrived[warp_id], 0, 1);
    }
    __syncthreads();

    {
        tma_swizzle_allocator al((int*)&__shm[0]);
        comp_layout comp_layout_result = setup_comp_path(
            al, inputs_arrived, inputs_finished, outputs_arrived, outputs_finished);
        if (threadIdx.x == 0) {
            comp_inputs_ptr = comp_layout_result.inputs;
            comp_outputs_ptr = comp_layout_result.outputs;
        }
        __syncthreads();
        tma_swizzle_allocator al_comm((int*)&__shm[0]);
        comm_layout comm_layout_result =
            setup_comm_path(al_comm, comm_inputs_arrived);
        if (threadIdx.x == 0) {
            comm_A_smem_ptr = comm_layout_result.A_smem;
        }
        __syncthreads();
    }

    globals::pipeline_inputs(&inputs_ref)[globals::PIPELINE_STAGES] =
        *reinterpret_cast<globals::pipeline_inputs(*)[globals::PIPELINE_STAGES]>(
            comp_inputs_ptr);
    typename globals::A_comm_tile(&A_smem)[globals::NUM_CHUNKS] =
        *reinterpret_cast<typename globals::A_comm_tile(*)[globals::NUM_CHUNKS]>(
            comm_A_smem_ptr);

    __syncthreads();

    for (;;) {
        maybe_update_comm_controller(
            G, wb_config, local_row_blocks_comm, col_blocks_comm);

#ifdef AG_GEMM_SCHEDULER_ONLY_CTA0
        // CTA 0 is scheduler-only: it runs the controller but does NO
        // compute or comm work.  This isolates scheduling overhead from
        // work interference.  CTA 0 just spins the controller loop until
        // all compute is done, then exits.
        if (blockIdx.x == 0) {
            // Broadcast done signal through shared memory so ALL threads in CTA-0
            // see the same decision.  If only thread-0 calls break, threads 1-511
            // would hang at the __syncthreads() below — kernel deadlock.
            if (threadIdx.x == 0) {
                helping_comm = (load_u32(G.done_compute) >= total_comp_tiles);
            }
            __syncthreads();
            if (helping_comm) break;  // all threads exit together
            // Yield briefly to avoid hammering the scheduler bus.
            if (threadIdx.x == 0) {
                __nanosleep(100);
            }
            continue;
        }
#endif
        // Target-based role assignment: blockIdx.x < target → comm helper.
        // CTA-0 is both the controller and a comm/compute worker.
        // Compute CTAs never overflow to comm — they just wait on barriers.
        //
        // Thread 0 runs the controller update and the role decision together, so
        // both share the single __syncthreads() below.  The decision has to reach
        // the rest of the CTA through shared memory: a per-thread read of the
        // target is unsafe because load_u32 uses ld.relaxed.gpu and can return a
        // stale L1 value relative to thread-0's atomicExch.  Warps disagreeing on
        // helping_comm would diverge into mismatched __syncthreads() and deadlock.
        if (threadIdx.x == 0) {
            const unsigned int target =
                min(load_u32(G.target_comm_helpers),
                    G.active_sms - 1u);
            // Once all comm is exhausted, no CTA should enter comm path.
            // Also check the global comm_next_tile counter so that pure-compute
            // CTAs (which never set comm_exhausted themselves) can detect global
            // exhaustion and enter the fast path on the same iteration.
            if (!comm_exhausted) {
                // comm_next_tile is a row index (0..local_row_blocks_comm-1).
                if (load_u32((unsigned int*)G.comm_next_tile) >=
                    (unsigned int)local_row_blocks_comm) {
                    comm_exhausted = true;
                }
            }
            if (!comm_exhausted) {
                helping_comm = ((unsigned int)blockIdx.x < target);
            } else {
                helping_comm = false;
            }
        }
        __syncthreads();  // REQUIRED: Ensures all threads see synchronized decision about roles.


        if (helping_comm) {
            // Row-based comm: one row per CTA.  A larger batch would let a single
            // CTA claim every row at small M and leave the other comm CTAs idle.
            // CTAs past the row count exhaust immediately and fall back to compute.
            //
            // Each CTA drives all col_blocks columns of its row with
            // COMM_NUM_WORKERS warpgroups (stride=3) and pipelined TMA stores,
            // waited once at the end rather than per column, then signals the row
            // with a single NVLink multicast of count=col_blocks_comm.
            if (threadIdx.x == 0) {
                comm_task_id = atomicAdd((unsigned int*)G.comm_next_tile, 1u);
            }
            __syncthreads();
            if (comm_task_id >= (unsigned int)local_row_blocks_comm) {
                // No more rows to claim — mark exhausted and fall through to compute.
                if (threadIdx.x == 0) {
                    comm_exhausted = true;
                    helping_comm = false;
                }
                __syncthreads();
                continue;
            }
            unsigned long long _comm_t0 = clock64();
            comm_sm_row(G, (int)comm_task_id, A_smem, comm_inputs_arrived, comm_phasebits);
            if (threadIdx.x == 0) {
                if (G.comm_cycles != nullptr)
                    atomicAdd(G.comm_cycles, (unsigned int)((clock64() - _comm_t0) >> 10));
                ++local_done_comm_accum;
                if (local_done_comm_accum >= DONE_COMM_FLUSH_INTERVAL) {
                    atomicAdd(G.done_comm, local_done_comm_accum);
                    local_done_comm_accum = 0;
                }
            }
            if (warpgroup_id == 0 && warp_in_wg == 0 && lane_id == 0) {
                signal_all(G.barrier,
                           coord<ducks::default_type>(
                               0, 0, 0,
                               (int)comm_task_id + G.dev_idx * local_row_blocks_comm),
                           col_blocks_comm);
            }
            __syncthreads();
            continue;
        }

        // Compute path — handles all compute tiles, both while comm is active and
        // after it is exhausted.  Tiles are claimed globally via atomicAdd on
        // comp_next_tile rather than by a per-CTA stride, so a CTA delayed by comm
        // simply processes fewer tiles instead of stranding its share.  One
        // atomicAdd per tile (~100-200 cycles) is negligible against an HGMMA tile.
        if (threadIdx.x == 0) {
            next_tile = atomicAdd((unsigned int*)G.comp_next_tile, 1u);
        }
        __syncthreads();
        if (next_tile >= total_comp_tiles) {
            break;
        }

        int row_idx_fixed, col_idx_fixed;
        const int super_rows = (local_row_blocks / globals::SUPER_M) * globals::SUPER_M;
        const int final_rows = local_row_blocks - super_rows;
        const int super_blocks = globals::SUPER_M * col_blocks;
        const int num_local_blocks = local_row_blocks * col_blocks;
        const int task_id = (int)next_tile;
        if (task_id < num_local_blocks) {
            int local_row_idx;
            if (task_id < super_rows * col_blocks) {
                local_row_idx =
                    globals::SUPER_M * (task_id / super_blocks) +
                    task_id % globals::SUPER_M;
                col_idx_fixed = (task_id % super_blocks) / globals::SUPER_M;
            } else {
                const int remainder_id = task_id - super_rows * col_blocks;
                local_row_idx = super_rows + remainder_id % final_rows;
                col_idx_fixed = remainder_id / final_rows;
            }
            row_idx_fixed = local_row_idx + G.dev_idx * local_row_blocks;
        } else {
            const int num_peer_devices = globals::NUM_DEVICES - 1;
            const int amortized_task_id = task_id - num_local_blocks;
            const int target_shard =
                amortized_task_id / (num_peer_devices * col_blocks);
            const int idx_in_shard =
                amortized_task_id % (num_peer_devices * col_blocks);
            const int shard_row_idx = idx_in_shard % num_peer_devices;
            row_idx_fixed = (shard_row_idx >= G.dev_idx ? shard_row_idx + 1
                                                        : shard_row_idx) *
                                local_row_blocks +
                            target_shard;
            col_idx_fixed = idx_in_shard / num_peer_devices;
            // Wait for the remote AG row — no __syncthreads() needed: the producer
            // warpgroup thread that calls wait() drives TMA loads inside comp_sm_tile.
            // Comm signals once per row with count=col_blocks_comm (single NVLink op),
            // so wait for count=col_blocks_comm to unblock after all row data is stored.
            if (warpgroup_id == config::NUM_WARPGROUPS - 1 &&
                warp_id == 0 && lane_id == 0) {
                wait(G.barrier,
                     coord<ducks::default_type>(0, 0, 0, row_idx_fixed / 2),
                     G.dev_idx, col_blocks_comm);
            }
        }
        unsigned long long _comp_t0 = clock64();
        comp_sm_tile(
            G, row_idx_fixed, col_idx_fixed, false, inputs_ref, *comp_outputs_ptr,
            inputs_arrived, inputs_finished, outputs_arrived,
            outputs_finished, stage, comp_phasebits, num_iters, 1);
        if (threadIdx.x == 0 && G.compute_cycles != nullptr)
            atomicAdd(G.compute_cycles, (unsigned int)((clock64() - _comp_t0) >> 10));
        if (threadIdx.x == 0) {
            ++local_done_accum;
            if (local_done_accum >= DONE_COMPUTE_FLUSH_INTERVAL) {
                atomicAdd(G.done_compute, local_done_accum);
                local_done_accum = 0;
            }
        }
        continue;
    }
    // Flush any remaining batched done counts.
    if (threadIdx.x == 0) {
        if (local_done_accum > 0)
            atomicAdd(G.done_compute, local_done_accum);
        if (local_done_comm_accum > 0)
            atomicAdd(G.done_comm, local_done_comm_accum);
    }
}

// Cache-line padded scheduler counters.
// Each hot counter lives on its own 128-byte L2 cache line to prevent false
// sharing between writers (done_compute: 128 CTAs write every 8 tiles) and
// readers (target_comm_helpers, comm_next_tile: all CTAs read every iteration).
// Without padding, L2 invalidations from done_compute writes evict the
// read-mostly values, adding round-trip L2 latency on every iteration.
struct alignas(128) SchedulerCounters {
    // Slot 0: next compute tile claimed (written by each CTA once per tile)
    unsigned int comp_next_tile;  char _pad0[124];
    // Slot 1: next comm row claimed (written by comm CTAs; read by all for exhaustion check)
    unsigned int comm_next_tile;  char _pad1[124];
    // Slot 2: active comm helpers (written by comm CTAs on join/leave)
    unsigned int active_comm_helpers; char _pad2[124];
    // Slot 3: done_compute (hot write: 128 CTAs atomicAdd every DONE_COMPUTE_FLUSH_INTERVAL)
    unsigned int done_compute;    char _pad3[124];
    // Slot 4: done_comm (written by controller only)
    unsigned int done_comm;       char _pad4[124];
    // Slot 5: last_done_compute (written by controller only)
    unsigned int last_done_compute; char _pad5[124];
    // Slot 6: last_done_comm (written by controller only)
    unsigned int last_done_comm;  char _pad6[124];
    // Slot 7: target_comm_helpers (hot read: all CTAs read every iteration; written by controller)
    unsigned int target_comm_helpers; char _pad7[124];
    // Slot 8: stable_count (written/read by controller only)
    unsigned int stable_count;    char _pad8[124];
    // Slot 9: compute_cycles (hot write: all compute CTAs atomicAdd after each tile)
    unsigned int compute_cycles;  char _pad9[124];
    // Slot 10: comm_cycles (hot write: all comm CTAs atomicAdd after each row)
    unsigned int comm_cycles;     char _pad10[124];
    // Slot 11: last_compute_cycles (written by controller only)
    unsigned int last_compute_cycles; char _pad11[124];
    // Slot 12: last_comm_cycles (written by controller only)
    unsigned int last_comm_cycles; char _pad12[124];
    // Slot 13: EMA cost primary (float-as-uint, written by controller)
    unsigned int ema_cost_primary; char _pad13[124];
    // Slot 14: EMA cost secondary (float-as-uint, written by controller)
    unsigned int ema_cost_secondary; char _pad14[124];
};
static_assert(sizeof(SchedulerCounters) == 15 * 128,
              "SchedulerCounters size mismatch");

static SchedulerCounters* g_sched_counters[globals::NUM_DEVICES] = {nullptr};

#ifdef AG_GEMM_PRINT_SCHED_STATS
static unsigned int* g_trace_buf[globals::NUM_DEVICES] = {nullptr};
static unsigned int* g_trace_idx[globals::NUM_DEVICES] = {nullptr};
static unsigned long long* g_trace_time_buf[globals::NUM_DEVICES] = {nullptr};
#endif

void entrypoint(kittens::py::TKParallelTensor& A,
                const at::Tensor& B,
                at::Tensor& C,
                kittens::py::TKParallelTensor& barrier,
                const int forced_comm_target = -1,
                const int active_sms = config::NUM_BLOCKS) {
    TORCH_CHECK(active_sms >= 2 && active_sms <= config::NUM_BLOCKS,
                "active_sms must be in [2, ", config::NUM_BLOCKS, "], got ", active_sms);
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_sched_counters[dev_idx] == nullptr) {
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    }
    SchedulerCounters* sc = g_sched_counters[dev_idx];
    unsigned int* comp_next_tile = &sc->comp_next_tile;
    unsigned int* comm_next_tile = &sc->comm_next_tile;

    cudaMemsetAsync(sc, 0, sizeof(SchedulerCounters), stream);

    const int M_h = static_cast<int>(C.size(0));
    const int K_h = static_cast<int>(B.size(0));
    const int N_h = static_cast<int>(B.size(1));
    const unsigned int total_comp_h =
        (unsigned int)(M_h / globals::ROW_BLOCK) *
        (unsigned int)(N_h / globals::COL_BLOCK);
    // Row-based comm: total_comm_h = local AG row count (not flat task count).
    // comm_next_tile tracks row index, matching device-side total_comm_tiles = rows.
    const unsigned int local_row_blocks_comm_h =
        (unsigned int)((M_h / (globals::ROW_BLOCK * 2)) / globals::NUM_DEVICES);
    const unsigned int total_comm_h = local_row_blocks_comm_h;  // row count only
    {
        const unsigned int comm_cap_h = min(
            total_comm_h > 0 ? total_comm_h : 1u,
            max(1u, (((unsigned int)active_sms - 1u) * 3u) / 4u));
        // K-adaptive cap: small K (cheap compute tiles) needs more comm helpers
        // to avoid AG blocking compute. For large K (expensive GEMM), fewer helpers needed.
        // cap = max(4, NUM_BLOCKS / (1 + K_blocks)) where K_blocks = K / (RED_BLOCK*16)
        // Examples: K=2048→cap=44, K=8192→cap=14, K=16384→cap=7, K=32768→cap=4
        // Small M has too few tiles for the policy to converge from a bad initial
        // target, so the formula matters most there; large M is insensitive to it.
        const unsigned int K_blocks_h = (unsigned int)K_h / ((unsigned int)globals::RED_BLOCK * 16u);
        const unsigned int k_adaptive_cap_h = max(4u, (unsigned int)active_sms / (1u + K_blocks_h));
        // Floor = min(rows, K-adaptive cap): ideally 1 CTA per row for parallel AG.
        // For M=2048 (2 rows): floor=2 → 2 CTAs handle 2 rows in parallel.
        // For M=32768 (32 rows, cap=4): floor=4 → 4 CTAs handle rows in 8 passes.
        const unsigned int comm_floor_h =
            min((unsigned int)local_row_blocks_comm_h, k_adaptive_cap_h);
        const unsigned int initial_target_comm_helpers =
            (forced_comm_target >= 0)
                ? min((unsigned int)forced_comm_target, comm_cap_h)
                : min(comm_floor_h, comm_cap_h);
        cudaMemcpyAsync(&sc->target_comm_helpers, &initial_target_comm_helpers,
                        sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        // All CTAs start as compute, so active comm helpers begins at 0.
        const unsigned int initial_active = 0;
        cudaMemcpyAsync(&sc->active_comm_helpers, &initial_active,
                        sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    }

#ifdef AG_GEMM_PRINT_SCHED_STATS
    // Allocate trace buffers on first call, reset on every call.
    if (g_trace_buf[dev_idx] == nullptr) {
        cudaMalloc(&g_trace_buf[dev_idx], TRACE_MAX * 7 * sizeof(unsigned int));
        cudaMalloc(&g_trace_idx[dev_idx], sizeof(unsigned int));
        cudaMalloc(&g_trace_time_buf[dev_idx], TRACE_MAX * sizeof(unsigned long long));
    }
    cudaMemsetAsync(g_trace_buf[dev_idx], 0, TRACE_MAX * 7 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_idx[dev_idx], 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_time_buf[dev_idx], 0, TRACE_MAX * sizeof(unsigned long long), stream);
#endif

    dynamic_globals G{
        .A = kittens::py::parallel_tensor_to_pgl<globals::A_pgl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::tensor_to_gl<globals::C_gl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(
            barrier),
        .dev_idx = dev_idx,
        .active_sms = (unsigned int)active_sms,
        .comp_next_tile = comp_next_tile,
        .comm_next_tile = comm_next_tile,
        .active_comm_helpers = &sc->active_comm_helpers,
        .done_compute = &sc->done_compute,
        .done_comm = &sc->done_comm,
        .last_done_compute = &sc->last_done_compute,
        .last_done_comm = &sc->last_done_comm,
        .target_comm_helpers = &sc->target_comm_helpers,
        .blocked_remote_observations = nullptr,
        .last_blocked_remote_observations = nullptr,
        .blocked_remote_cycles = nullptr,
        .last_blocked_remote_cycles = nullptr,
        .ready_comm_tiles = nullptr,
        .last_ready_comm_tiles = nullptr,
        // Cycle tracking: wired to SchedulerCounters slots for work-balance policy.
        .compute_cycles = &sc->compute_cycles,
        .comm_cycles = &sc->comm_cycles,
        .last_compute_cycles = &sc->last_compute_cycles,
        .last_comm_cycles = &sc->last_comm_cycles,
        // CTA-0-only controller: nullptr disables controller_lock CAS contention.
        .controller_lock = nullptr,
        // stable_count enables exponential backoff.
        .stable_count = &sc->stable_count,
        .ema_cost_primary = &sc->ema_cost_primary,
        .ema_cost_secondary = &sc->ema_cost_secondary,
#ifdef AG_GEMM_PRINT_SCHED_STATS
        .trace_buf = g_trace_buf[dev_idx],
        .trace_idx = g_trace_idx[dev_idx],
        .trace_time_buf = g_trace_time_buf[dev_idx],
#endif
    };
    kittens::py::launch_kernel<config, dynamic_globals, main_kernel>(G, (unsigned int)active_sms);
    kittens::py::launch_kernel<config, dynamic_globals, epilogue_kernel_wrapper>(G);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace ag_gemm_dynamic

#ifndef AG_GEMM_DYNAMIC_ONLY
namespace ag_gemm_static {
extern void entrypoint(kittens::py::TKParallelTensor& A,
                       const at::Tensor& B,
                       at::Tensor& C,
                       kittens::py::TKParallelTensor& barrier,
                       const int num_comm_sms,
                       const int active_sms);
}
#endif

#ifdef AG_GEMM_PRINT_SCHED_STATS
// Return the scheduler trace buffer as a (N, 7) int tensor.
// Columns: [done_comp, done_comm, next_target, active_helpers, prev_target, cost_p_bits, cost_s_bits].
static at::Tensor get_trace_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (ag_gemm_dynamic::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, ag_gemm_dynamic::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, ag_gemm_dynamic::TRACE_MAX);
    auto out = at::zeros({n, 7}, at::TensorOptions().dtype(at::kInt));
    if (n > 0 && ag_gemm_dynamic::g_trace_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int>(), ag_gemm_dynamic::g_trace_buf[dev_idx],
                   (size_t)n * 7 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}

// Return the scheduler trace timestamps as a (N,) long tensor (nanoseconds).
static at::Tensor get_trace_time_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (ag_gemm_dynamic::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, ag_gemm_dynamic::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, ag_gemm_dynamic::TRACE_MAX);
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kLong));
    if (n > 0 && ag_gemm_dynamic::g_trace_time_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int64_t>(), ag_gemm_dynamic::g_trace_time_buf[dev_idx],
                   (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def("all_gather_matmul_dynamic", &ag_gemm_dynamic::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("forced_comm_target") = -1,
          pybind11::arg("active_sms") = 132);
#ifndef AG_GEMM_DYNAMIC_ONLY
    m.def("all_gather_matmul", &ag_gemm_static::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("num_comm_sms"),
          pybind11::arg("active_sms") = 132);
#endif
#ifdef AG_GEMM_PRINT_SCHED_STATS
    m.def("get_trace_buf", &get_trace_buf_py, pybind11::arg("dev_idx"));
    m.def("get_trace_time_buf", &get_trace_time_buf_py, pybind11::arg("dev_idx"));
#endif
}
