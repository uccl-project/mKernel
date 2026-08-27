// Dynamic SM partitioning for Ring Attention.
//
// A fully dynamic "everyone interleaves" policy starves KV forwarding when
// compute work is abundant, so the scheduling here is service-regulated:
//   1) a small fixed comm floor guarantees forward progress for KV forwarding;
//   2) additional CTAs are elastically recruited into comm service when the
//      pending comm backlog grows beyond a per-CTA threshold;
//   3) recruited CTAs fall back to compute only after the backlog drops below a
//      lower exit threshold (hysteresis).
//
// For ring attention the forwarded KV path is on the next stage's critical
// path, so preserving comm service latency matters more than perfectly
// balancing average CTA utilization. The online signal is pending comm backlog,
// which works here because all comm slots are ready from stage start.

#include "policies/scheduler_base.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

using namespace kittens;
namespace scheduler = dynamic_sm_allocation::scheduler;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace dynamic_ring {

struct config {
    static constexpr int CLUSTER_SIZE         = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 3;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS   = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 32;
    static constexpr int CONSUMER_REGISTERS = 160;
    static constexpr int COMM_FLOOR_CTAS = 4;  // policy helper_floor (was hardcoded)
    static constexpr float COMM_RATE_GAMMA = 1.15f;
    static constexpr float COMM_RATE_HYSTERESIS = 0.10f;
    static constexpr unsigned int WORK_SAMPLE_WINDOW = 8;
};

static constexpr unsigned int TRACE_MAX = 8192;

struct globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int D           = 128;
    static constexpr int QO_BLOCK    = 64;
    static constexpr int KV_BLOCK    = 128;
    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile     = st_bf<QO_BLOCK, D>;
    using K_tile     = st_bf<KV_BLOCK, D>;
    using V_tile     = st_bf<KV_BLOCK, D>;
    using L_vec      = col_vec<st_fl<QO_BLOCK, D>>;
    using O_tile     = st_bf<QO_BLOCK, D>;
    using L_vec_2x   = col_vec<st_fl<2 * QO_BLOCK, D>>;
    using O_tile_2x  = st_bf<2 * QO_BLOCK, D>;

    using Q_gl       = gl<bf16, -1, -1, -1, D, Q_tile>;
    using K_pgl      = pgl<gl<bf16, -1, -1, -1, D, K_tile>, NUM_DEVICES, false>;
    using V_pgl      = pgl<gl<bf16, -1, -1, -1, D, V_tile>, NUM_DEVICES, false>;
    using L_gl       = gl<float, 1, -1, -1, -1, L_vec, L_vec_2x>;
    using O_gl       = gl<bf16, -1, -1, -1, D, O_tile, O_tile_2x>;
    using barrier_pgl = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

    Q_gl      Q;
    K_pgl     K0, K1;
    V_pgl     V0, V1;
    L_gl      L_block, L;
    O_gl      O_block, O;
    barrier_pgl barrier;

    int ring_stage;
    const int dev_idx;
    const int n_partial_blocks;    // precomputed on host
    const int n_reduction_blocks;  // precomputed on host

    unsigned int* next_comm;    // global comm-slot counter (pairs: even=K, odd=V)
    unsigned int* next_partial; // global compute-block counter
    unsigned int* active_comm_helpers; // recruited comm CTAs beyond the floor
    unsigned int* done_comm;          // completed comm slots
    unsigned int* done_partial;       // completed partial-attention blocks
    unsigned int* last_done_comm;     // controller snapshot
    unsigned int* last_done_partial;  // controller snapshot
    unsigned int* target_helpers;     // global helper target chosen by CTA0
    // --- FT cycle + controller_lock counters ---
    unsigned int* partial_cycles;
    unsigned int* comm_slot_cycles;
    unsigned int* last_partial_cycles;
    unsigned int* last_comm_slot_cycles;
    unsigned int* controller_lock;
    unsigned int* stable_count;
    unsigned int* ema_cost_primary;
    unsigned int* ema_cost_secondary;
    unsigned int* kernel_phase;     // debug: 0=comm_partial, 1=reduction, 2=barrier, 3=done
#ifdef RING_ATTN_PRINT_SCHED_STATS
    unsigned int* trace_buf;
    unsigned int* trace_idx;
    unsigned long long* trace_time_buf;
#endif

    __host__ static int compute_partial_blocks(const Q_gl& Q) {
        return Q.batch() * Q.depth() * Q.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK);
    }
    __host__ static int compute_reduction_blocks(const O_gl& O) {
        return O.batch() * O.depth() * O.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK * 2);
    }
    __host__ int num_partial_blocks() const { return n_partial_blocks; }
    __host__ int num_reduction_blocks() const { return n_reduction_blocks; }
};

// ---------- KV ring communication --------------------------------------------
// comm_slot: even → send K tiles, odd → send V tiles.
// The number of KV blocks transferred per comm_slot is determined by the total
// comm pairs: each pair covers (num_batches * num_heads * KV_blocks) tiles
// split over NUM_CHUNKS warps inside the CTA.
static constexpr int NUM_CHUNKS = 7;

__device__ inline void attn_comm_slot(const globals& G, const int comm_slot) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(globals::K_tile) * NUM_CHUNKS <= config::DYNAMIC_SHARED_MEMORY);
    globals::K_tile (&KV_smem)[NUM_CHUNKS] = al.allocate<globals::K_tile, NUM_CHUNKS>();

    const int warp_id = warp::groupid();
    const int num_batches = G.Q.batch();
    const int num_heads   = G.Q.depth();
    const int KV_blocks   = G.K0.rows() / globals::KV_BLOCK;
    const int num_blocks  = num_batches * num_heads * KV_blocks;
    const int dst_dev     = (G.dev_idx + 1) % globals::NUM_DEVICES;

    // Total comm pairs = ceil(num_blocks / NUM_CHUNKS); slot maps to pair index
    const int pair_idx = comm_slot / 2;  // which pair
    const bool send_K  = (comm_slot % 2 == 0);

    __shared__ semaphore inputs_arrived[NUM_CHUNKS];
    __shared__ semaphore inputs_finished[NUM_CHUNKS];
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < NUM_CHUNKS; i++) {
            init_semaphore(inputs_arrived[i],  0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
    }
    __syncthreads();

    uint32_t phasebits = 0xFFFF0000;
    if (warp_id < NUM_CHUNKS && laneid() == 0) {
        int chunk_id = warp_id;
        // Each comm slot covers one pair_idx; warp covers its chunk within that pair
        const int task_id = NUM_CHUNKS * pair_idx + chunk_id;
        if (task_id < num_blocks) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx  = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx    = task_id % KV_blocks;

            wait(inputs_finished[chunk_id], get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);

            tma::expect_bytes(inputs_arrived[chunk_id], sizeof(globals::K_tile));
            if (send_K) {
                if (G.ring_stage % 2 == 0)
                    tma::load_async(KV_smem[chunk_id], G.K0[G.dev_idx],
                                    {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    tma::load_async(KV_smem[chunk_id], G.K1[G.dev_idx],
                                    {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            } else {
                if (G.ring_stage % 2 == 0)
                    tma::load_async(KV_smem[chunk_id], G.V0[G.dev_idx],
                                    {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    tma::load_async(KV_smem[chunk_id], G.V1[G.dev_idx],
                                    {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            }
        }
    } else if (NUM_CHUNKS <= warp_id && warp_id < 2 * NUM_CHUNKS && laneid() == 0) {
        int chunk_id = warp_id - NUM_CHUNKS;
        const int task_id = NUM_CHUNKS * pair_idx + chunk_id;
        if (task_id < num_blocks) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx  = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx    = task_id % KV_blocks;

            wait(inputs_arrived[chunk_id], get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            if (send_K) {
                if (G.ring_stage % 2 == 0)
                    tma::store_async(G.K1[dst_dev], KV_smem[chunk_id],
                                     {batch_idx, head_idx, KV_idx, 0});
                else
                    tma::store_async(G.K0[dst_dev], KV_smem[chunk_id],
                                     {batch_idx, head_idx, KV_idx, 0});
            } else {
                if (G.ring_stage % 2 == 0)
                    tma::store_async(G.V1[dst_dev], KV_smem[chunk_id],
                                     {batch_idx, head_idx, KV_idx, 0});
                else
                    tma::store_async(G.V0[dst_dev], KV_smem[chunk_id],
                                     {batch_idx, head_idx, KV_idx, 0});
            }
            tma::store_async_read_wait();
            arrive(inputs_finished[chunk_id]);
        }
    }
}

// ---------- partial attention (identical to static) ---------------------------
__device__ inline void attn_partial(const globals& G, const int block_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    globals::Q_tile  (&Q_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::Q_tile, config::CONSUMER_WARPGROUPS>();
    globals::K_tile  (&K_smem)[globals::PIPELINE_STAGES] =
        al.allocate<globals::K_tile, globals::PIPELINE_STAGES>();
    globals::V_tile  (&V_smem)[globals::PIPELINE_STAGES] =
        al.allocate<globals::V_tile, globals::PIPELINE_STAGES>();
    globals::L_vec   (&L_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::L_vec, config::CONSUMER_WARPGROUPS>();
    globals::O_tile  (&O_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::O_tile, config::CONSUMER_WARPGROUPS>();

    const int num_heads = G.Q.depth();
    const int QO_blocks = G.Q.rows() / (config::CONSUMER_WARPGROUPS * globals::QO_BLOCK);
    const int KV_blocks = G.K0.rows() / globals::KV_BLOCK;
    const int batch_idx = block_idx / (QO_blocks * num_heads);
    const int head_idx  = (block_idx % (QO_blocks * num_heads)) / QO_blocks;
    const int QO_idx    = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS;
    const int warpgroup_id = warpgroup::groupid();

    __shared__ semaphore Q_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ semaphore L_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ semaphore O_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore compute_done[globals::PIPELINE_STAGES];

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(Q_arrived[i], 0, 1);
            init_semaphore(L_arrived[i], 0, 1);
            init_semaphore(O_arrived[i], 0, 1);
        }
#pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; i++) {
            init_semaphore(K_arrived[i], 0, 1);
            init_semaphore(V_arrived[i], 0, 1);
            init_semaphore(compute_done[i], config::CONSUMER_WARPGROUPS, 0);
        }
    }
    __syncthreads();

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
        for (int KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
            wait(compute_done[KV_idx % globals::PIPELINE_STAGES],
                 (KV_idx / globals::PIPELINE_STAGES + 1) % 2);
            if (G.ring_stage % 2 == 0) {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES],
                                             sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES],
                                           G.K0[G.dev_idx],
                                           {batch_idx, head_idx, KV_idx, 0},
                                           K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES],
                                             sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES],
                                           G.V0[G.dev_idx],
                                           {batch_idx, head_idx, KV_idx, 0},
                                           V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            } else {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES],
                                             sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES],
                                           G.K1[G.dev_idx],
                                           {batch_idx, head_idx, KV_idx, 0},
                                           K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES],
                                             sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES],
                                           G.V1[G.dev_idx],
                                           {batch_idx, head_idx, KV_idx, 0},
                                           V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        rt_fl<16, globals::KV_BLOCK>  att_block;
        rt_bf<16, globals::KV_BLOCK>  att_block_mma;
        rt_fl<16, globals::D>         o_reg;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

        warpgroup::tma::expect_bytes(Q_arrived[warpgroup_id], sizeof(Q_smem[warpgroup_id]));
        warpgroup::tma::load_async(Q_smem[warpgroup_id], G.Q,
                                   {batch_idx, head_idx, QO_idx + warpgroup_id, 0},
                                   Q_arrived[warpgroup_id]);

        warp::zero(norm_vec);
        warp::zero(o_reg);
        warp::neg_infty(max_vec);
        wait(Q_arrived[warpgroup_id], 0);

        for (int KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
            wait(K_arrived[KV_idx % globals::PIPELINE_STAGES],
                 (KV_idx / globals::PIPELINE_STAGES) % 2);
            warpgroup::mm_ABt(att_block, Q_smem[warpgroup_id],
                              K_smem[KV_idx % globals::PIPELINE_STAGES]);

            warp::copy(max_vec_last_scaled, max_vec);
            warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.08838834764f);
            warpgroup::mma_async_wait();
            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
            warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
            warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

            wait(V_arrived[KV_idx % globals::PIPELINE_STAGES],
                 (KV_idx / globals::PIPELINE_STAGES) % 2);
            warpgroup::mma_AB(o_reg, att_block_mma,
                              V_smem[KV_idx % globals::PIPELINE_STAGES]);
            warpgroup::mma_async_wait();
            warpgroup::arrive(compute_done[KV_idx % globals::PIPELINE_STAGES], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(O_smem[warpgroup_id], o_reg);
        warpgroup::sync(warpgroup_id + 4);
        if (G.ring_stage == 0)
            warpgroup::tma::store_async(G.O, O_smem[warpgroup_id],
                                        {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
        else
            warpgroup::tma::store_async(G.O_block, O_smem[warpgroup_id],
                                        {batch_idx, head_idx, QO_idx + warpgroup_id, 0});

        warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);
        warpgroup::store(L_smem[warpgroup_id], norm_vec);
        warpgroup::sync(warpgroup_id + 4);
        if (warpgroup::laneid() == 0) {
            if (G.ring_stage == 0)
                tma::store_async(G.L, L_smem[warpgroup_id],
                                 {batch_idx, head_idx, QO_idx + warpgroup_id});
            else
                tma::store_async(G.L_block, L_smem[warpgroup_id],
                                 {batch_idx, head_idx, QO_idx + warpgroup_id});
        }
    }
}

// ---------- reduction (unchanged from static) ---------------------------------
__device__ inline void attn_reduction(const globals& G, const int block_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    globals::O_tile_2x (&O_block_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    globals::O_tile_2x (&O_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    globals::L_vec_2x (&L_block_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();
    globals::L_vec_2x (&L_smem)[config::CONSUMER_WARPGROUPS] =
        al.allocate<globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();

    const int warpgroup_id = warpgroup::groupid();
    const int num_heads    = G.O.depth();
    const int QO_blocks    = G.O.rows() / (2 * globals::QO_BLOCK * config::CONSUMER_WARPGROUPS);
    const int batch_idx    = block_idx / (QO_blocks * num_heads);
    const int head_idx     = (block_idx % (QO_blocks * num_heads)) / QO_blocks;
    const int QO_idx       = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS;

    __shared__ semaphore inputs_arrived[config::CONSUMER_WARPGROUPS];
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            tma::expect_bytes(inputs_arrived[i],
                              (sizeof(globals::L_vec_2x) + sizeof(globals::O_tile_2x)) * 2);
            tma::load_async(L_smem[i], G.L, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            tma::load_async(O_smem[i], G.O, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
            tma::load_async(L_block_smem[i], G.L_block, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            tma::load_async(O_block_smem[i], G.O_block, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
        }
    }
    __syncthreads();

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        wait(inputs_arrived[warpgroup_id], 0);

        rt_fl<32, globals::D> O_reg, O_block_reg;
        col_vec<rt_fl<32, globals::D>> L_reg, L_block_reg, L_new_reg;

        warpgroup::load(L_reg, L_smem[warpgroup_id]);
        warpgroup::load(L_block_reg, L_block_smem[warpgroup_id]);
        warp::sub(L_new_reg, L_block_reg, L_reg);
        warp::exp(L_new_reg, L_new_reg);
        warp::add(L_new_reg, L_new_reg, 1.f);
        warp::log(L_new_reg, L_new_reg);
        warp::add(L_new_reg, L_new_reg, L_reg);
        warpgroup::store(L_smem[warpgroup_id], L_new_reg);

        warp::sub(L_reg, L_reg, L_new_reg);
        warp::exp(L_reg, L_reg);
        warp::sub(L_block_reg, L_block_reg, L_new_reg);
        warp::exp(L_block_reg, L_block_reg);
        warpgroup::load(O_reg, O_smem[warpgroup_id]);
        warp::mul_row(O_reg, O_reg, L_reg);
        warpgroup::load(O_block_reg, O_block_smem[warpgroup_id]);
        warp::mul_row(O_block_reg, O_block_reg, L_block_reg);
        warp::add(O_reg, O_reg, O_block_reg);
        warpgroup::store(O_smem[warpgroup_id], O_reg);

        warpgroup::sync(warpgroup_id + 4);
        if (warpgroup::laneid() == 0) {
            tma::store_async(G.O, O_smem[warpgroup_id],
                             {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
            tma::store_async(G.L, L_smem[warpgroup_id],
                             {batch_idx, head_idx, QO_idx + warpgroup_id});
        }
    }
}

// ---------- device kernels ---------------------------------------------------
using scheduler::load_u32;

__device__ inline int load_active_comm_helpers(const globals& G) {
    return (int)load_u32(G.active_comm_helpers);
}

__device__ inline int load_target_helpers(const globals& G) {
    return (int)load_u32(G.target_helpers);
}

#ifdef RING_ATTN_PRINT_SCHED_STATS
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
    G.trace_buf[slot * 8 + 3] = scheduler::load_u32(G.active_comm_helpers);
    G.trace_buf[slot * 8 + 4] = prev_target;
    G.trace_buf[slot * 8 + 5] = 0u;
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

// Work-balance policy variant of the comm controller update for ring attention.
__device__ inline void maybe_update_comm_controller(
    const globals& G,
    const scheduler::WorkBalanceConfig& wb_config) {
    const scheduler::HelperCounters counters{
        .active_helpers = G.active_comm_helpers,
        .done_primary = G.done_partial,
        .done_secondary = G.done_comm,
        .last_done_primary = G.last_done_partial,
        .last_done_secondary = G.last_done_comm,
        .target_helpers = G.target_helpers,
        .blocked_primary = nullptr,
        .last_blocked_primary = nullptr,
        // Explicitly initialize ALL fields to avoid NVCC device-code
        // designated-init zero-initialization gaps.
        .blocked_primary_cycles = nullptr,
        .last_blocked_primary_cycles = nullptr,
        .ready_secondary = nullptr,
        .last_ready_secondary = nullptr,
        .claimed_secondary = nullptr,
        .primary_cycles = G.partial_cycles,
        .secondary_cycles = G.comm_slot_cycles,
        .last_primary_cycles = G.last_partial_cycles,
        .last_secondary_cycles = G.last_comm_slot_cycles,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
        .controller_lock = nullptr,
        .stable_count = G.stable_count,
        .successful_secondary_claims = nullptr,
        .failed_secondary_claims = nullptr,
        .last_successful_secondary_claims = nullptr,
        .last_failed_secondary_claims = nullptr,
    };
#ifdef RING_ATTN_PRINT_SCHED_STATS
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

__device__ inline bool try_join_comm_helpers(const globals& G, const int assist_cap) {
    return scheduler::try_join_helpers(
        G.active_comm_helpers, G.target_helpers, (unsigned int)max(0, assist_cap));
}

__device__ inline bool try_leave_comm_helpers(const globals& G) {
    return scheduler::try_leave_helpers(
        G.active_comm_helpers, G.target_helpers, 0u);
}

__device__ inline void comm_partial_kernel(const globals& G) {
    extern __shared__ int __unused[];

    const int num_batches   = G.Q.batch();
    const int num_heads     = G.Q.depth();
    const int KV_blocks     = G.K0.rows() / globals::KV_BLOCK;
    const int total_pairs   = (num_batches * num_heads * KV_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
    const int total_comm_slots = total_pairs * 2;  // K slots + V slots
    const int total_partial    = G.n_partial_blocks;
    const int comm_helper_cap = min((int)gridDim.x - 1,
                                    max(1, total_comm_slots));
    const scheduler::WorkBalanceConfig wb_config{
        .total_ctas = (unsigned int)gridDim.x,
        .total_primary = (unsigned int)total_partial,
        .total_secondary = (unsigned int)total_comm_slots,
        .helper_floor = (unsigned int)min((int)gridDim.x,
                                          config::COMM_FLOOR_CTAS),
        .helper_cap = (unsigned int)comm_helper_cap,
        .secondary_blocks_primary = false,  // compute→comm dependency
        .ema_alpha = 0.3f,
        .ema_cost_primary = G.ema_cost_primary,
        .ema_cost_secondary = G.ema_cost_secondary,
    };

    __shared__ unsigned int my_comm_slot;
    __shared__ unsigned int my_partial;
    __shared__ bool is_comm;
    __shared__ bool comm_exhausted;
    unsigned int iter_count = 0;
    // Batch done_comm/done_partial atomicAdds to reduce L2 contention.
    static constexpr unsigned int DONE_FLUSH_INTERVAL = 8;
    unsigned int local_done_comm_accum = 0;
    unsigned int local_done_partial_accum = 0;
    if (threadIdx.x == 0) {
        is_comm = false;
        comm_exhausted = false;
    }
    __syncthreads();

    while (true) {
        // Controller update: CTA-0 only, every 8th iteration
        if ((iter_count & 7u) == 0u) {
            maybe_update_comm_controller(G, wb_config);
        }
        ++iter_count;
        __syncthreads();

        // Unified target-based role assignment, as in GEMM+AR and GEMM+RS.
        if (threadIdx.x == 0 && !comm_exhausted) {
            const unsigned int tgt = scheduler::load_u32(G.target_helpers);
            is_comm = (blockIdx.x < tgt);
        }
        __syncthreads();

        if (is_comm) {
            if (threadIdx.x == 0) {
                my_comm_slot = atomicAdd(G.next_comm, 1u);
            }
            __syncthreads();
            if ((int)my_comm_slot < total_comm_slots) {
                unsigned long long _comm_t0 = clock64();
                attn_comm_slot(G, (int)my_comm_slot);
                if (threadIdx.x == 0) {
                    if (G.comm_slot_cycles != nullptr)
                        atomicAdd(G.comm_slot_cycles,
                                  (unsigned int)((clock64() - _comm_t0) >> 10));
                    ++local_done_comm_accum;
                    if (local_done_comm_accum >= DONE_FLUSH_INTERVAL) {
                        atomicAdd(G.done_comm, local_done_comm_accum);
                        local_done_comm_accum = 0;
                    }
                }
                __syncthreads();
                continue;
            }
            // All comm slots exhausted — flush pending done counts so the
            // policy sees accurate remaining work instead of a phantom backlog.
            if (threadIdx.x == 0) {
                if (local_done_comm_accum > 0) {
                    atomicAdd(G.done_comm, local_done_comm_accum);
                    local_done_comm_accum = 0;
                }
                comm_exhausted = true;
                is_comm = false;
            }
            __syncthreads();
        }

        // Compute path.
        if (threadIdx.x == 0) {
            my_partial = atomicAdd(G.next_partial, 1u);
        }
        __syncthreads();
        if ((int)my_partial >= total_partial) {
            // Compute done — drain remaining comm slots.
            if (threadIdx.x == 0) {
                my_comm_slot = atomicAdd(G.next_comm, 1u);
            }
            __syncthreads();
            if ((int)my_comm_slot >= total_comm_slots) {
                break;
            }
            {
                unsigned long long _comm_t0_tail = clock64();
                attn_comm_slot(G, (int)my_comm_slot);
                if (threadIdx.x == 0) {
                    if (G.comm_slot_cycles != nullptr)
                        atomicAdd(G.comm_slot_cycles,
                                  (unsigned int)((clock64() - _comm_t0_tail) >> 10));
                    ++local_done_comm_accum;
                    if (local_done_comm_accum >= DONE_FLUSH_INTERVAL) {
                        atomicAdd(G.done_comm, local_done_comm_accum);
                        local_done_comm_accum = 0;
                    }
                }
            }
            __syncthreads();
            continue;
        }
        {
            unsigned long long _partial_t0 = clock64();
            attn_partial(G, (int)my_partial);
            if (threadIdx.x == 0) {
                if (G.partial_cycles != nullptr)
                    atomicAdd(G.partial_cycles,
                              (unsigned int)((clock64() - _partial_t0) >> 10));
                ++local_done_partial_accum;
                if (local_done_partial_accum >= DONE_FLUSH_INTERVAL) {
                    atomicAdd(G.done_partial, local_done_partial_accum);
                    local_done_partial_accum = 0;
                }
            }
        }
        __syncthreads();
    }
    // Flush remaining done counts.
    if (threadIdx.x == 0) {
        if (local_done_comm_accum > 0)
            atomicAdd(G.done_comm, local_done_comm_accum);
        if (local_done_partial_accum > 0)
            atomicAdd(G.done_partial, local_done_partial_accum);
    }
}

__device__ inline void reduction_kernel(const globals& G) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && G.kernel_phase != nullptr)
        atomicExch(G.kernel_phase, 1u);  // phase 1 = reduction
    attn_reduction(G, blockIdx.x);
}

struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};
__device__ inline void barrier_kernel(const globals& G) {
    if (G.kernel_phase != nullptr) atomicExch(G.kernel_phase, 2u);  // phase 2 = barrier
    barrier_all(G.barrier, {1, 0, 0}, G.dev_idx);
    if (G.kernel_phase != nullptr) atomicExch(G.kernel_phase, 3u);  // phase 3 = done
}

// ---------- host-side counters -----------------------------------------------
// Cache-line padded counter struct — each hot counter gets its own 128-byte
// L2 cache line to prevent false sharing between writer and reader CTAs.
struct alignas(128) SchedulerCounters {
    unsigned int next_comm;             char _pad0[124];   // Slot 0
    unsigned int next_partial;          char _pad1[124];   // Slot 1
    unsigned int active_comm_helpers;   char _pad2[124];   // Slot 2
    unsigned int done_comm;             char _pad3[124];   // Slot 3
    unsigned int done_partial;          char _pad4[124];   // Slot 4
    unsigned int last_done_comm;        char _pad5[124];   // Slot 5
    unsigned int last_done_partial;     char _pad6[124];   // Slot 6
    unsigned int target_helpers;        char _pad7[124];   // Slot 7
    unsigned int partial_cycles;        char _pad8[124];   // Slot 8: cycle accumulator
    unsigned int comm_slot_cycles;      char _pad9[124];   // Slot 9: cycle accumulator
    unsigned int last_partial_cycles;   char _pad10[124];  // Slot 10: snapshot
    unsigned int last_comm_slot_cycles; char _pad11[124];  // Slot 11: snapshot
    unsigned int stable_count;          char _pad12[124];  // Slot 12: backoff
    unsigned int ema_cost_primary;      char _pad13[124];  // Slot 13: EMA cost primary (float-as-uint)
    unsigned int ema_cost_secondary;    char _pad14[124];  // Slot 14: EMA cost secondary (float-as-uint)
    unsigned int kernel_phase;          char _pad15[124];  // Slot 15: debug
};
static_assert(sizeof(SchedulerCounters) == 16 * 128, "SchedulerCounters size mismatch");

static SchedulerCounters* g_sched_counters[globals::NUM_DEVICES] = {nullptr};
#ifdef RING_ATTN_PRINT_SCHED_STATS
static unsigned int* g_trace_buf[globals::NUM_DEVICES] = {nullptr};
static unsigned int* g_trace_idx[globals::NUM_DEVICES] = {nullptr};
static unsigned long long* g_trace_time_buf[globals::NUM_DEVICES] = {nullptr};
#endif

} // namespace dynamic_ring

// ---------- Python entrypoint ------------------------------------------------
void entrypoint_dynamic_ring(
    const at::Tensor& Q,
    kittens::py::TKParallelTensor& K0,
    kittens::py::TKParallelTensor& K1,
    kittens::py::TKParallelTensor& V0,
    kittens::py::TKParallelTensor& V1,
    at::Tensor& L,
    at::Tensor& L_block,
    at::Tensor& O,
    at::Tensor& O_block,
    kittens::py::TKParallelTensor& barrier,
    const int ring_stage)
{
    using namespace dynamic_ring;
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_sched_counters[dev_idx] == nullptr)
        cudaMalloc(&g_sched_counters[dev_idx], sizeof(SchedulerCounters));
    cudaMemsetAsync(g_sched_counters[dev_idx], 0, sizeof(SchedulerCounters), stream);
    SchedulerCounters* sc = g_sched_counters[dev_idx];
#ifdef RING_ATTN_PRINT_SCHED_STATS
    if (g_trace_buf[dev_idx] == nullptr) {
        cudaMalloc(&g_trace_buf[dev_idx], TRACE_MAX * 8 * sizeof(unsigned int));
        cudaMalloc(&g_trace_idx[dev_idx], sizeof(unsigned int));
        cudaMalloc(&g_trace_time_buf[dev_idx], TRACE_MAX * sizeof(unsigned long long));
    }
    cudaMemsetAsync(g_trace_buf[dev_idx], 0, TRACE_MAX * 8 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_idx[dev_idx], 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(g_trace_time_buf[dev_idx], 0, TRACE_MAX * sizeof(unsigned long long), stream);
#endif

    {
        auto Q_gl_tmp = kittens::py::tensor_to_gl<globals::Q_gl>(Q);
        const int n_partial_h = globals::compute_partial_blocks(Q_gl_tmp);
        const int nb = Q_gl_tmp.batch();
        const int nh = Q_gl_tmp.depth();
        const int kvb = K0.data_.size(-2) / globals::KV_BLOCK;
        const int total_pairs_h = (nb * nh * kvb + NUM_CHUNKS - 1) / NUM_CHUNKS;
        const int total_comm_h = total_pairs_h * 2;
        const int total_ctas_h = min(132, total_comm_h + n_partial_h);
        const int comm_cap_h = min(total_ctas_h - 1, max(1, total_comm_h));
        const scheduler::WorkBalanceConfig wb_init{
            .total_ctas = (unsigned int)total_ctas_h,
            .total_primary = (unsigned int)n_partial_h,
            .total_secondary = (unsigned int)total_comm_h,
            .helper_floor = (unsigned int)min(total_ctas_h,
                                              (int)config::COMM_FLOOR_CTAS),
            .helper_cap = (unsigned int)comm_cap_h,
            .secondary_blocks_primary = false,
            .ema_alpha = 0.3f,
            .ema_cost_primary = nullptr,
            .ema_cost_secondary = nullptr,
        };
        unsigned int dynamic_init = scheduler::compute_initial_target_wb(wb_init);
        cudaMemcpyAsync(&sc->target_helpers, &dynamic_init,
                        sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    }

    auto Q_gl   = kittens::py::tensor_to_gl<globals::Q_gl>(Q);
    auto O_gl   = kittens::py::tensor_to_gl<globals::O_gl>(O);

    globals G {
        .Q       = Q_gl,
        .K0      = kittens::py::parallel_tensor_to_pgl<globals::K_pgl>(K0),
        .K1      = kittens::py::parallel_tensor_to_pgl<globals::K_pgl>(K1),
        .V0      = kittens::py::parallel_tensor_to_pgl<globals::V_pgl>(V0),
        .V1      = kittens::py::parallel_tensor_to_pgl<globals::V_pgl>(V1),
        .L_block = kittens::py::tensor_to_gl<globals::L_gl>(L_block),
        .L       = kittens::py::tensor_to_gl<globals::L_gl>(L),
        .O_block = kittens::py::tensor_to_gl<globals::O_gl>(O_block),
        .O       = O_gl,
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .ring_stage = ring_stage,
        .dev_idx    = dev_idx,
        .n_partial_blocks    = globals::compute_partial_blocks(Q_gl),
        .n_reduction_blocks  = globals::compute_reduction_blocks(O_gl),
        .next_comm    = &sc->next_comm,
        .next_partial = &sc->next_partial,
        .active_comm_helpers = &sc->active_comm_helpers,
        .done_comm = &sc->done_comm,
        .done_partial = &sc->done_partial,
        .last_done_comm = &sc->last_done_comm,
        .last_done_partial = &sc->last_done_partial,
        .target_helpers = &sc->target_helpers,
        .partial_cycles = &sc->partial_cycles,
        .comm_slot_cycles = &sc->comm_slot_cycles,
        .last_partial_cycles = &sc->last_partial_cycles,
        .last_comm_slot_cycles = &sc->last_comm_slot_cycles,
        .controller_lock = nullptr,
        .stable_count = &sc->stable_count,
        .ema_cost_primary = &sc->ema_cost_primary,
        .ema_cost_secondary = &sc->ema_cost_secondary,
        .kernel_phase = &sc->kernel_phase,
#ifdef RING_ATTN_PRINT_SCHED_STATS
        .trace_buf = g_trace_buf[dev_idx],
        .trace_idx = g_trace_idx[dev_idx],
        .trace_time_buf = g_trace_time_buf[dev_idx],
#endif
    };

    // Total grid = max across all possible tasks to keep all SMs busy
    const int num_batches  = G.Q.batch();
    const int num_heads    = G.Q.depth();
    const int KV_blocks    = G.K0.rows() / globals::KV_BLOCK;
    const int total_pairs  = (num_batches * num_heads * KV_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
    const int total_comm_slots  = total_pairs * 2;
    const int total_partial     = G.num_partial_blocks();
    const int total_ctas = min(132, total_comm_slots + total_partial);

    CUDACHECK(cudaFuncSetAttribute(
        kittens::py::global_kernel_unclustered<config, globals, comm_partial_kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY));
    kittens::py::global_kernel_unclustered<config, globals, comm_partial_kernel>
        <<<total_ctas, config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY, stream>>>(G);

    if (ring_stage > 0) {
        CUDACHECK(cudaFuncSetAttribute(
            kittens::py::global_kernel_unclustered<config, globals, reduction_kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY));
        kittens::py::global_kernel_unclustered<config, globals, reduction_kernel>
            <<<G.num_reduction_blocks(), config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY, stream>>>(G);
    }

    CUDACHECK(cudaFuncSetAttribute(
        kittens::py::global_kernel_unclustered<barrier_config, globals, barrier_kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, barrier_config::DYNAMIC_SHARED_MEMORY));
    kittens::py::global_kernel_unclustered<barrier_config, globals, barrier_kernel>
        <<<1, 1, 0, stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

#ifdef RING_ATTN_PRINT_SCHED_STATS
static at::Tensor get_trace_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_ring::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_ring::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_ring::TRACE_MAX);
    auto out = at::zeros({n, 8}, at::TensorOptions().dtype(at::kInt));
    if (n > 0 && dynamic_ring::g_trace_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int>(), dynamic_ring::g_trace_buf[dev_idx],
                   (size_t)n * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return out;
}

static at::Tensor get_trace_time_buf_py(int dev_idx) {
    unsigned int tidx = 0;
    if (dynamic_ring::g_trace_idx[dev_idx] != nullptr) {
        cudaMemcpy(&tidx, dynamic_ring::g_trace_idx[dev_idx], sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    }
    int n = (int)min(tidx, dynamic_ring::TRACE_MAX);
    auto out = at::zeros({n}, at::TensorOptions().dtype(at::kLong));
    if (n > 0 && dynamic_ring::g_trace_time_buf[dev_idx] != nullptr) {
        cudaMemcpy(out.data_ptr<int64_t>(), dynamic_ring::g_trace_time_buf[dev_idx],
                   (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    return out;
}
#endif

// Read the 14 scheduler counters from a non-blocking stream so it works
// even while the main kernel is hung.
static at::Tensor get_counters_py(int dev_idx) {
    // Return 14 counters in a flat tensor for backward-compatible Python access.
    auto out = at::zeros({14}, at::TensorOptions().dtype(at::kInt));
    if (dynamic_ring::g_sched_counters[dev_idx] != nullptr) {
        dynamic_ring::SchedulerCounters h;
        cudaStream_t nb_stream;
        cudaStreamCreateWithFlags(&nb_stream, cudaStreamNonBlocking);
        cudaMemcpyAsync(&h, dynamic_ring::g_sched_counters[dev_idx],
                        sizeof(dynamic_ring::SchedulerCounters),
                        cudaMemcpyDeviceToHost, nb_stream);
        cudaStreamSynchronize(nb_stream);
        cudaStreamDestroy(nb_stream);
        int* p = out.data_ptr<int>();
        p[0]  = (int)h.next_comm;
        p[1]  = (int)h.next_partial;
        p[2]  = (int)h.active_comm_helpers;
        p[3]  = (int)h.done_comm;
        p[4]  = (int)h.done_partial;
        p[5]  = (int)h.last_done_comm;
        p[6]  = (int)h.last_done_partial;
        p[7]  = (int)h.target_helpers;
        p[8]  = (int)h.partial_cycles;
        p[9]  = (int)h.comm_slot_cycles;
        p[10] = (int)h.last_partial_cycles;
        p[11] = (int)h.last_comm_slot_cycles;
        p[12] = (int)h.stable_count;
        p[13] = (int)h.kernel_phase;
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def("tk_mha_fwd_d128_dynamic", &entrypoint_dynamic_ring,
          pybind11::arg("Q"),
          pybind11::arg("K0"), pybind11::arg("K1"),
          pybind11::arg("V0"), pybind11::arg("V1"),
          pybind11::arg("L"), pybind11::arg("L_block"),
          pybind11::arg("O"), pybind11::arg("O_block"),
          pybind11::arg("barrier"),
          pybind11::arg("ring_stage"));
    m.def("get_counters", &get_counters_py, pybind11::arg("dev_idx"));
#ifdef RING_ATTN_PRINT_SCHED_STATS
    m.def("get_trace_buf", &get_trace_buf_py, pybind11::arg("dev_idx"));
    m.def("get_trace_time_buf", &get_trace_time_buf_py, pybind11::arg("dev_idx"));
#endif
}
