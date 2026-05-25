/**
 * @file ring_attention.cu
 * @brief Multi-node Ring Attention - staged kernels.
 *
 * The host entrypoint launches a short sequence of kernels rather than one
 * persistent mega-kernel, keeping register live ranges small and avoiding
 * ptxas spills in the attention path:
 *
 *   KV send prologue:
 *     Stages local K/V data and posts RDMA writes to every remote peer node.
 *     It returns after WRs are queued so network transfer overlaps the early
 *     ring stages.
 *
 *   Per-ring comm+partial kernels:
 *     Comm CTAs exchange K/V tiles around the local node while compute CTAs run
 *     partial attention for one Q block and the current ring stage.
 *
 *   Per-ring reduction kernels:
 *     Merge stage outputs with online softmax state when the partial path does
 *     not fuse the reduction directly.
 *
 *   KV copy epilogue:
 *     Waits for peer-node RDMA arrivals and publishes received K/V into the
 *     local buffers used by later stages.
 *
 * Device-side barriers coordinate local ring handoff; RDMA arrival flags gate
 * peer-node K/V visibility.
 *
 * Infrastructure (config/globals/structs, grid-barrier helpers, host setup,
 * entrypoint) lives in
 *   include/operators/ring_attention/ring_attention.cuh
 * Python/session glue + pybind module live in
 *   include/operators/ring_attention/session.cuh
 *
 * This TU contains the device role functions (attn_partial / attn_reduction /
 * attn_comm / kv_stage_and_send_sm / kv_send_sm / kv_copy_sm) and the
 * __global__ stage stubs that the host entrypoint launches.
 */
#include "operators/ring_attention/ring_attention.cuh"

namespace ring_attn_multinode {

// ============================================================================
// Stage device functions: attn_partial, attn_comm, attn_reduction
// ============================================================================
//
// These device functions live in this TU rather than the header because they
// are passed by name to __global__ stubs in the same TU; ptxas needs the
// definition visible at the point each stub is instantiated.

/**
 * Compute partial attention for one tile block.
 * @tparam SKIP_REG_ALLOC If true, skip increase_registers/decrease_registers
 *         (caller already did the register setup, e.g. persistent kernel).
 * @tparam FUSE_REDUCE If true, at ring_stage>0 read previous (L, O) from G.L/G.O,
 *         merge via online softmax in registers, and write back to G.L/G.O —
 *         eliminates the separate attn_reduction pass + intra-grid barrier.
 *         Stage 0 writes directly to G.L/G.O unchanged.
 * @param ring_stage Current ring stage (determines K0/K1, V0/V1, O/O_block).
 */
template <bool SKIP_REG_ALLOC = false, bool FUSE_REDUCE = false>
__device__ inline void attn_partial(const globals &G, const int block_idx, const int ring_stage) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(globals::Q_tile) * config::CONSUMER_WARPGROUPS +
                  sizeof(globals::K_tile) * globals::PIPELINE_STAGES +
                  sizeof(globals::V_tile) * globals::PIPELINE_STAGES +
                  sizeof(globals::L_vec) * config::CONSUMER_WARPGROUPS +
                  sizeof(globals::O_tile) * config::CONSUMER_WARPGROUPS <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::Q_tile (&Q_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::Q_tile, config::CONSUMER_WARPGROUPS>();
    typename globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = al.allocate<typename globals::K_tile, globals::PIPELINE_STAGES>();
    typename globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = al.allocate<typename globals::V_tile, globals::PIPELINE_STAGES>();
    typename globals::L_vec (&L_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec, config::CONSUMER_WARPGROUPS>();
    typename globals::O_tile (&O_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile, config::CONSUMER_WARPGROUPS>();

    const int num_heads = G.Q.depth();
    const int QO_blocks = G.Q.rows() / (config::CONSUMER_WARPGROUPS * globals::QO_BLOCK);
    const int KV_blocks = G.K0.rows() / globals::KV_BLOCK;
    const int batch_idx = block_idx / (QO_blocks * num_heads);
    const int head_idx = (block_idx % (QO_blocks * num_heads)) / QO_blocks;
    const int QO_idx = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS;
    const int warpgroup_id = warpgroup::groupid();

    __shared__ kittens::semaphore Q_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore L_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore O_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ kittens::semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ kittens::semaphore compute_done[globals::PIPELINE_STAGES];
    __shared__ kittens::semaphore reduce_arrived[config::CONSUMER_WARPGROUPS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(Q_arrived[i], 0, 1);
            init_semaphore(L_arrived[i], 0, 1);
            init_semaphore(O_arrived[i], 0, 1);
            init_semaphore(reduce_arrived[i], 0, 1);
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
        if constexpr (!SKIP_REG_ALLOC) { warpgroup::decrease_registers<config::PRODUCER_REGISTERS>(); }
        for (int KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
            wait(compute_done[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES + 1) % 2);
            if (ring_stage % 2 == 0) {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES], G.K0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES], G.V0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            } else {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES], G.K1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES], G.V1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            }
        }
    } else {
        if constexpr (!SKIP_REG_ALLOC) { warpgroup::increase_registers<config::CONSUMER_REGISTERS>(); }

        rt_fl<16, globals::KV_BLOCK> att_block;
        rt_bf<16, globals::KV_BLOCK> att_block_mma;
        rt_fl<16, globals::D> o_reg;

        col_vec<rt_fl<16, globals::KV_BLOCK>> norm_vec;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec_last_scaled;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec_scaled;

        warpgroup::tma::expect_bytes(Q_arrived[warpgroup_id], sizeof(Q_smem[warpgroup_id]));
        warpgroup::tma::load_async(Q_smem[warpgroup_id], G.Q, {batch_idx, head_idx, QO_idx + warpgroup_id, 0}, Q_arrived[warpgroup_id]);

        warp::zero(norm_vec);
        warp::zero(o_reg);
        warp::neg_infty(max_vec);

        wait(Q_arrived[warpgroup_id], 0);

        for (auto KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
            wait(K_arrived[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES) % 2);
            warpgroup::mm_ABt(att_block, Q_smem[warpgroup_id], K_smem[KV_idx % globals::PIPELINE_STAGES]);

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

            wait(V_arrived[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES) % 2);

            warpgroup::mma_AB(o_reg, att_block_mma, V_smem[KV_idx % globals::PIPELINE_STAGES]);
            warpgroup::mma_async_wait();

            warpgroup::arrive(compute_done[KV_idx % globals::PIPELINE_STAGES], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);

        // Convert scaled-log-base-2 max back to natural log, fold into L.
        warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);
        // norm_vec now holds L_stage (log-sum-exp in natural log, softmax-scaled)

        if constexpr (FUSE_REDUCE) {
            if (ring_stage > 0) {
                // Fused reduction: load previous (L, O) from G.L/G.O, merge via
                // online softmax in registers, overwrite G.L/G.O. Reuses
                // Q_smem (Q is done) and L_smem (not yet written) as scratch.
                // Note: vec TMA must be issued from a single thread per WG —
                // warpgroup::tma::load_async for vecs distributes across all
                // 128 threads' warp-lanes (laneid & 31), causing 4× redundant
                // TMA commands that overflow the mbarrier byte counter.
                if (warpgroup::laneid() == 0) {
                    ::dist::tma::expect_bytes(reduce_arrived[warpgroup_id],
                                      sizeof(globals::L_vec) + sizeof(globals::O_tile));
                    ::dist::tma::load_async(L_smem[warpgroup_id], G.L,
                                    {batch_idx, head_idx, QO_idx + warpgroup_id},
                                    reduce_arrived[warpgroup_id]);
                    ::dist::tma::load_async(Q_smem[warpgroup_id], G.O,
                                    {batch_idx, head_idx, QO_idx + warpgroup_id, 0},
                                    reduce_arrived[warpgroup_id]);
                }
                wait(reduce_arrived[warpgroup_id], 0);

                col_vec<rt_fl<16, globals::KV_BLOCK>> L_prev_reg;
                col_vec<rt_fl<16, globals::KV_BLOCK>> L_new_reg;
                rt_fl<16, globals::D> O_prev_reg;

                warpgroup::load(L_prev_reg, L_smem[warpgroup_id]);
                warpgroup::load(O_prev_reg, Q_smem[warpgroup_id]);

                // L_new = logaddexp(L_prev, L_stage), with L_stage in norm_vec
                warp::sub(L_new_reg, norm_vec, L_prev_reg);
                warp::exp(L_new_reg, L_new_reg);
                warp::add(L_new_reg, L_new_reg, 1.f);
                warp::log(L_new_reg, L_new_reg);
                warp::add(L_new_reg, L_new_reg, L_prev_reg);

                // O_final = exp(L_prev - L_new) * O_prev + exp(L_stage - L_new) * O_stage
                warp::sub(L_prev_reg, L_prev_reg, L_new_reg);
                warp::exp(L_prev_reg, L_prev_reg);
                warp::sub(norm_vec, norm_vec, L_new_reg);
                warp::exp(norm_vec, norm_vec);
                warp::mul_row(O_prev_reg, O_prev_reg, L_prev_reg);
                warp::mul_row(o_reg, o_reg, norm_vec);
                warp::add(o_reg, o_reg, O_prev_reg);

                warp::copy(norm_vec, L_new_reg);
            }
        }

        warpgroup::store(O_smem[warpgroup_id], o_reg);
        warpgroup::sync(warpgroup_id + 4);
        if constexpr (FUSE_REDUCE) {
            warpgroup::tma::store_async(G.O, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
        } else if (ring_stage == 0) {
            warpgroup::tma::store_async(G.O, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
        } else {
            warpgroup::tma::store_async(G.O_block, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
        }

        warpgroup::store(L_smem[warpgroup_id], norm_vec);
        warpgroup::sync(warpgroup_id + 4);
        if (warpgroup::laneid() == 0) {
            if constexpr (FUSE_REDUCE) {
                ::dist::tma::store_async(G.L, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
            } else if (ring_stage == 0) {
                ::dist::tma::store_async(G.L, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
            } else {
                ::dist::tma::store_async(G.L_block, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
            }
        }
    }
}

/**
 * Merge O + O_block via online softmax reduction.
 * @tparam SKIP_REG_ALLOC If true, skip register reallocation.
 */
template <bool SKIP_REG_ALLOC = false>
__device__ inline void attn_reduction(const globals &G, const int block_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(globals::O_tile_2x) * config::CONSUMER_WARPGROUPS * 2 +
                  sizeof(globals::L_vec_2x) * config::CONSUMER_WARPGROUPS * 2 <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::O_tile_2x (&O_block_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::O_tile_2x (&O_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::L_vec_2x (&L_block_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::L_vec_2x (&L_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();

    const int warpgroup_id = warpgroup::groupid();
    const int num_heads = G.O.depth();
    const int QO_blocks = G.O.rows() / (2 * globals::QO_BLOCK * config::CONSUMER_WARPGROUPS);
    const int batch_idx = block_idx / (QO_blocks * num_heads);
    const int head_idx = (block_idx % (QO_blocks * num_heads)) / QO_blocks;
    const int QO_idx = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore inputs_arrived[config::CONSUMER_WARPGROUPS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            ::dist::tma::expect_bytes(inputs_arrived[i], (sizeof(globals::L_vec_2x) + sizeof(globals::O_tile_2x)) * 2);
            ::dist::tma::load_async(L_smem[i], G.L, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            ::dist::tma::load_async(O_smem[i], G.O, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
            ::dist::tma::load_async(L_block_smem[i], G.L_block, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            ::dist::tma::load_async(O_block_smem[i], G.O_block, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
        }
    }
    __syncthreads();

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if constexpr (!SKIP_REG_ALLOC) { warpgroup::decrease_registers<config::PRODUCER_REGISTERS>(); }
    } else {
        if constexpr (!SKIP_REG_ALLOC) { warpgroup::increase_registers<config::CONSUMER_REGISTERS>(); }

        wait(inputs_arrived[warpgroup_id], 0);

        rt_fl<32, globals::D> O_reg;
        rt_fl<32, globals::D> O_block_reg;
        col_vec<rt_fl<32, globals::D>> L_reg;
        col_vec<rt_fl<32, globals::D>> L_block_reg;
        col_vec<rt_fl<32, globals::D>> L_new_reg;

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
            ::dist::tma::store_async(G.O, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
            ::dist::tma::store_async(G.L, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
        }
    }
}

/**
 * Communicate KV tiles to the next GPU in the ring.
 * @param ring_stage Current ring stage (determines source/dest buffers).
 */
__device__ inline void attn_comm(const globals &G, const int block_idx, const int ring_stage) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    // NUM_CHUNKS splits intra-node K/V rotation work into chunks × 2 sides
    // (send/recv). Requires 2*NUM_CHUNKS ≤ warps-per-CTA — otherwise warps
    // 2*NUM_CHUNKS..NUM_WARPS-1 do real work that never gets scheduled and
    // the kernel deadlocks. Default warps/CTA = 4 warpgroups × 4 warps = 16
    // (supports NUM_CHUNKS ≤ 8). With CONSUMER_WARPGROUPS=2 (3 wg total),
    // warps/CTA drops to 12 → NUM_CHUNKS must be ≤ 6.
    static constexpr int NUM_CHUNKS = config::CONSUMER_WARPGROUPS >= 3 ? 7 : 6;
    static_assert(2 * NUM_CHUNKS <= config::NUM_WARPS,
                  "attn_comm needs 2*NUM_CHUNKS warps to schedule send+recv; "
                  "reduce NUM_CHUNKS or increase NUM_WARPGROUPS.");
    static_assert(sizeof(globals::K_tile) * NUM_CHUNKS <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::K_tile (&KV_smem)[NUM_CHUNKS] = al.allocate<typename globals::K_tile, NUM_CHUNKS>();

    const int warp_id = warp::groupid();
    const int num_batches = G.Q.batch();
    const int num_heads = G.Q.depth();
    const int KV_blocks = G.K0.rows() / globals::KV_BLOCK;
    const int num_blocks = num_batches * num_heads * KV_blocks;
    const int dst_dev_idx = (G.dev_idx + 1) % globals::NUM_DEVICES;

    __shared__ kittens::semaphore inputs_arrived[NUM_CHUNKS];
    __shared__ kittens::semaphore inputs_finished[NUM_CHUNKS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_CHUNKS; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
    }
    __syncthreads();

    uint32_t phasebits = 0xFFFF0000;

    if (warp_id < NUM_CHUNKS && laneid() == 0) {
        int chunk_id = warp_id;
        for (int task_id = NUM_CHUNKS * (block_idx / 2) + chunk_id; task_id < num_blocks; task_id += NUM_CHUNKS * (G.num_comm_sms / 2)) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx = task_id % KV_blocks;

            wait(inputs_finished[chunk_id], get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);

            ::dist::tma::expect_bytes(inputs_arrived[chunk_id], sizeof(globals::K_tile));
            if (block_idx % 2 == 0) {
                if (ring_stage % 2 == 0)
                    ::dist::tma::load_async(KV_smem[chunk_id], G.K0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    ::dist::tma::load_async(KV_smem[chunk_id], G.K1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            } else {
                if (ring_stage % 2 == 0)
                    ::dist::tma::load_async(KV_smem[chunk_id], G.V0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    ::dist::tma::load_async(KV_smem[chunk_id], G.V1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            }
        }
    } else if (NUM_CHUNKS <= warp_id && warp_id < 2 * NUM_CHUNKS && laneid() == 0) {
        int chunk_id = warp_id - NUM_CHUNKS;
        for (int task_id = NUM_CHUNKS * (block_idx / 2) + chunk_id; task_id < num_blocks; task_id += NUM_CHUNKS * (G.num_comm_sms / 2)) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx = task_id % KV_blocks;

            wait(inputs_arrived[chunk_id], get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            if (block_idx % 2 == 0) {
                if (ring_stage % 2 == 0)
                    ::dist::tma::store_async(G.K1[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
                else
                    ::dist::tma::store_async(G.K0[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
            } else {
                if (ring_stage % 2 == 0)
                    ::dist::tma::store_async(G.V1[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
                else
                    ::dist::tma::store_async(G.V0[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
            }

            ::dist::tma::store_async_read_wait();
            arrive(inputs_finished[chunk_id]);
        }
    }
}

// ============================================================================
// Inter-node KV exchange role functions
// ============================================================================
//
// Zero-copy KV send: skips the K0/V0 → send_buf pack. K0 is registered as
// the session's local_gpu_buf (src_view=0), V0 as clocal_gpu_buf (src_view=1),
// both via DMA-BUF (`direct_dmabuf_enabled=true`). The proxy posts a single-SGE
// RDMA write straight from the registered VMM tensor, so no GPU-memory copy
// is required on the sender side.
//
// Layout invariants:
//   - K0/V0 are row-major contiguous DistBuffers of size K_bytes/V_bytes.
//   - chunk_id ∈ [0, total_chunks_K) → K side, local_offset = chunk*CHUNK_BYTES
//     within K0; remote_offset = sap*single_peer_bytes + chunk*CHUNK_BYTES
//     (lands at this rank's slot within the peer's recv_buf K region).
//   - chunk_id ∈ [total_chunks_K, ...) → V side, local_offset = chunk*CHUNK_BYTES
//     within V0; remote_offset = sap*single_peer_bytes + K_bytes +
//     chunk*CHUNK_BYTES (lands in the peer's recv_buf V region for this slot).
__device__ inline void kv_stage_and_send_sm(const kv_exchange_globals &G) {
    int send_id = blockIdx.x;
    int total_chunks = G.total_chunks_K + G.total_chunks_V;
    if (threadIdx.x == 0) {
        for (int chunk_id = send_id; chunk_id < total_chunks; chunk_id += G.num_send_sms) {
            bool is_v = (chunk_id >= G.total_chunks_K);
            uint32_t off, bytes, remote_off;
            uint8_t src_view;
            if (!is_v) {
                off = (uint32_t)(chunk_id * CHUNK_BYTES);
                bytes = min(CHUNK_BYTES, G.K_bytes - (int)off);
                remote_off = off;
                src_view = 0;
            } else {
                int v_chunk = chunk_id - G.total_chunks_K;
                off = (uint32_t)(v_chunk * CHUNK_BYTES);
                bytes = min(CHUNK_BYTES, G.V_bytes - (int)off);
                remote_off = (uint32_t)(G.K_bytes) + off;
                src_view = 1;
            }
            const int n_peers = G.num_nodes - 1;
            // Per-peer slot offsets (zero at N == 2). single_peer_bytes is
            // this rank's K + V combined byte count; single_peer_tiles is
            // the chunk count for one peer's worth.
            const int single_peer_bytes = G.K_bytes + G.V_bytes;
            const int single_peer_tiles = G.total_chunks_K + G.total_chunks_V;
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint32_t)(sap * single_peer_tiles + chunk_id);
                cmd.bytes = bytes;
                cmd.local_offset = off;
                cmd.remote_offset = (uint64_t)sap * (uint32_t)single_peer_bytes + remote_off;
                cmd.src_view = src_view;
                cmd.lane_id = (uint16_t)chunk_id;
                cmd.reserved0 = (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)chunk_id);
                fifo.push(cmd);
            }
        }
    }
}

__device__ inline void kv_send_sm(const kv_exchange_globals &G) {
    // Pack K then V into the FIFO. K uses chunk_id [0, total_chunks_K).
    // V uses chunk_id [total_chunks_K, total_chunks_K + total_chunks_V).
    int comm_id = blockIdx.x;
    int warps_per_cta = config::NUM_THREADS / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int total_pushers = G.num_send_sms * warps_per_cta;
    int my_pusher = comm_id * warps_per_cta + warp_id;
    int total_chunks = G.total_chunks_K + G.total_chunks_V;

    if (lane_id == 0) {
        for (int chunk_id = my_pusher; chunk_id < total_chunks; chunk_id += total_pushers) {
            uint32_t off, bytes;
            bool is_v = (chunk_id >= G.total_chunks_K);
            const int n_peers = G.num_nodes - 1;
            // Per-peer slot offsets (zero at N == 2).
            const int single_peer_bytes = G.K_bytes + G.V_bytes;
            const int single_peer_tiles = G.total_chunks_K + G.total_chunks_V;
            if (!is_v) {
                off = (uint32_t)(chunk_id * CHUNK_BYTES);
                bytes = min(CHUNK_BYTES, G.K_bytes - (int)off);
                // Local K starts at offset 0 in send_buf, V starts at K_bytes
                for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint32_t)(sap * single_peer_tiles + chunk_id);
                    cmd.bytes    = bytes;
                    cmd.local_offset  = off;
                    cmd.remote_offset = (uint64_t)sap * (uint32_t)single_peer_bytes + off;
                    cmd.lane_id = (uint16_t)chunk_id;
                    cmd.reserved0 = (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                    internode::D2HFifoDevice fifo =
                        internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)chunk_id);
                    fifo.push(cmd);
                }
            } else {
                int v_chunk = chunk_id - G.total_chunks_K;
                off = (uint32_t)(v_chunk * CHUNK_BYTES);
                bytes = min(CHUNK_BYTES, G.V_bytes - (int)off);
                for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint32_t)(sap * single_peer_tiles + chunk_id);
                    cmd.bytes    = bytes;
                    cmd.local_offset = (uint64_t)(G.K_bytes) + off;
                    cmd.remote_offset = (uint64_t)sap * (uint32_t)single_peer_bytes
                                        + (uint32_t)(G.K_bytes) + off;
                    cmd.lane_id = (uint16_t)chunk_id;
                    cmd.reserved0 = (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                    internode::D2HFifoDevice fifo =
                        internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)chunk_id);
                    fifo.push(cmd);
                }
            }
        }
    }
}

__device__ inline void kv_copy_sm(const kv_exchange_globals &G, int peer_slot) {
    // Wait for the requested peer slot's KV to arrive, then D2D copy from
    // recv_buf[peer_slot * single_peer_bytes ..] into this rank's K0/V0.
    int copy_id = blockIdx.x - G.num_send_sms;
    int total_chunks = G.total_chunks_K + G.total_chunks_V;

    // arrival_flags is laid out [peer_slot * total_chunks + chunk_id].
    const int single_peer_chunks = total_chunks;
    const int single_peer_bytes = G.K_bytes + G.V_bytes;
    bf16 *K_src_base = G.recv_buf_base + (peer_slot * single_peer_bytes) / 2;
    bf16 *V_src_base = K_src_base + G.K_bytes / 2;

    for (int chunk_id = copy_id; chunk_id < total_chunks; chunk_id += G.num_copy_sms) {
        if (threadIdx.x == 0) {
            const int flag_idx = peer_slot * single_peer_chunks + chunk_id;
            uint32_t v;
            do {
                v = comm::atomic_u32::volatile_load(&G.arrival_flags[flag_idx]);
                if (v == G.epoch) break;
                __nanosleep(100);
            } while (true);
        }
        __syncthreads();
        __threadfence_system();

        bool is_v = (chunk_id >= G.total_chunks_K);
        const bf16 *src;
        bf16 *dst;
        uint32_t off, bytes;
        if (!is_v) {
            off = (uint32_t)(chunk_id * CHUNK_BYTES);
            bytes = min(CHUNK_BYTES, G.K_bytes - (int)off);
            src = K_src_base + off / 2;
            dst = G.K0_local + off / 2;
        } else {
            int v_chunk = chunk_id - G.total_chunks_K;
            off = (uint32_t)(v_chunk * CHUNK_BYTES);
            bytes = min(CHUNK_BYTES, G.V_bytes - (int)off);
            src = V_src_base + off / 2;
            dst = G.V0_local + off / 2;
        }

        const uint4 *src4 = reinterpret_cast<const uint4*>(src);
        uint4       *dst4 = reinterpret_cast<uint4*>(dst);
        int n_vec = bytes / 16;
        for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            dst4[i] = src4[i];
        }
        __syncthreads();
    }
}

// ============================================================================
// __global__ kernel stubs (entrypoint launches these via plain <<<...>>>)
// ============================================================================

// Per-stage kernels. Splitting comm+partial / reduction / RDMA into independent
// __global__ launches with one CTA per work-block keeps each kernel's register
// live-range short, yielding STACK:0 (vs the persistent kernel's STACK:600
// spills) and avoiding ptxas C7510/C7512 wgmma serialization that costs ~5×
// per-block compute time.
//
// Launch grid: num_comm_sms + num_partial_blks. CTA roles split by blockIdx.x.
// Comm CTAs do attn_comm (one per comm slot); compute CTAs do exactly one
// attn_partial block (oversubscribed → multiple waves on H100's 132 SMs).
// No outer reg setup here; the called functions do their own warpgroup
// register reallocation. Passing SKIP_REG_ALLOC=false lets ptxas split the
// register frame at the function boundary (STACK:0 / STACK:48).
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void attn_comm_partial_stage_kernel(
    const __grid_constant__ globals G,
    const int stage
) {
    if ((int)blockIdx.x < G.num_comm_sms) {
        attn_comm(G, blockIdx.x, stage);
    } else {
        attn_partial<false>(G, blockIdx.x - G.num_comm_sms, stage);
    }
}

// Launch grid: num_reduction_blks. One CTA per reduction block.
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void attn_reduction_stage_kernel(
    const __grid_constant__ globals G
) {
    attn_reduction<false>(G, blockIdx.x);
}

// Send-only: posts RDMA WRs for local K0/V0 → every remote peer node.
// Non-blocking; kernel returns once WRs are queued. Used as prologue so RDMA
// overlaps with round-0 stages.
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kv_send_kernel(
    const __grid_constant__ kv_exchange_globals KE
) {
    if ((int)blockIdx.x < KE.num_send_sms) {
        kv_stage_and_send_sm(KE);
    }
}

// Copy-only: waits for peer arrivals + D2D-copies recv buffers into K0/V0.
// kv_copy_sm internally indexes by (blockIdx.x - num_send_sms), so launch
// num_send + num_copy CTAs and gate the active range to keep the indexing.
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kv_copy_kernel(
    const __grid_constant__ kv_exchange_globals KE,
    const int peer_slot
) {
    if ((int)blockIdx.x >= KE.num_send_sms &&
        (int)blockIdx.x < KE.num_send_sms + KE.num_copy_sms) {
        kv_copy_sm(KE, peer_slot);
    }
}

// Cross-GPU barrier only. One CTA, one thread does barrier_all.
__global__ void barrier_only_kernel(
    const __grid_constant__ globals::barrier_distributed_tensor barrier,
    const int dev_idx
) {
    if (threadIdx.x == 0) {
        barrier_all(barrier, {1, 0, 0}, dev_idx);
    }
}

}  // namespace ring_attn_multinode

#include "operators/ring_attention/session.cuh"
