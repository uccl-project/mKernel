#pragma once
/**
 * @file gemm.cuh
 * @brief Modular, fused_globals-free grouped/per-expert bf16 warpgroup-MMA GEMM
 *        for the dispatch_gemm_glu_combine kernel (Gen-15 rewrite).
 *
 * This is a faithful refactor of `grouped_gemm` from
 * src/dispatch_gemm_glu_combine.cu, with four things made explicit that are
 * implicit/hard-coded in the original:
 *   (1) A and B TMA stage depths are separate template ints (STAGES_A/STAGES_B);
 *   (2) the epilogue strategy is a template bool (REG_EPILOGUE):
 *         false = today's SMEM-C buffer unioned onto the last input stage,
 *         true  = a small *transient* SMEM-C buffer decoupled from the input
 *                 pipeline (register accumulator -> warpgroup::store transient
 *                 -> TMA store), so C no longer steals an input pipeline stage;
 *   (3) the dispatch barrier is a caller-supplied Gate functor (decouples the
 *       module from fused_globals);
 *   (4) the SMEM budget is a template parameter with a static_assert, so an
 *       over-budget instantiation fails to *compile*.
 *
 * The module owns ONLY the GEMM. It knows nothing about dispatch, combine,
 * RDMA, or the globals struct. The fused kernel passes plain tensor views, a
 * per-expert row-block layout (GroupLayout), and an optional Gate.
 *
 * The super-M L2-rasterization decode that won 1.5290->1.5763 is kept verbatim
 * and driven by GroupLayout.super_m (a first-class config knob).
 */

#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"

namespace gemm_sm90 {

using namespace kittens;

// ---- compile-time config (the tunable surface) -----------------------------
template <int ROW_BLOCK_, int COL_BLOCK_, int RED_BLOCK_,
          int STAGES_B_,          // depth of the B (weight) TMA pipeline
          int STAGES_A_,          // depth of the A (activation) TMA pipeline (>= STAGES_B)
          int MMA_INFLIGHT_,      // committed WGMMA groups kept live (1 = today's depth-1)
          bool REG_EPILOGUE_,     // true: transient (decoupled) SMEM-C, no input-stage union
          int SMEM_BUDGET_>       // hard byte cap the instantiation must fit
struct Config {
    static constexpr int ROW_BLOCK = ROW_BLOCK_, COL_BLOCK = COL_BLOCK_, RED_BLOCK = RED_BLOCK_;
    static constexpr int STAGES_B = STAGES_B_, STAGES_A = STAGES_A_;
    static constexpr int MMA_INFLIGHT = MMA_INFLIGHT_;
    static constexpr bool REG_EPILOGUE = REG_EPILOGUE_;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;   // 8192 B  (two row-halves per row-block)
    using B_tile = st_bf<RED_BLOCK,   COL_BLOCK>;     // 32768 B
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;   // 32768 B

    // For the (legacy) union path STAGES_A must equal STAGES_B: A[2]+B share a
    // single `pipeline_inputs` stage and C[2] unions onto the last stage.
    static_assert(!( !REG_EPILOGUE_ ) || (STAGES_A_ == STAGES_B_),
                  "gemm_sm90: union epilogue requires STAGES_A == STAGES_B");
    static_assert(STAGES_A_ >= STAGES_B_, "gemm_sm90: STAGES_A must be >= STAGES_B");

    // ---- the byte budget, proven at compile time ---------------------------
    // Union path: footprint = (S-1) input stages + max(input_stage, 2*C).
    //   input_stage = 2*A + B; today S=4 -> 3*49152 + 65536 = 212992.
    // Reg path: A and B staged independently, plus ONE transient C buffer
    //   (TMA needs a SMEM source; the win is C no longer steals an input stage,
    //   and STAGES_A can run deeper than STAGES_B).
    static constexpr int sA = (int)sizeof(A_tile);
    static constexpr int sB = (int)sizeof(B_tile);
    static constexpr int sC = (int)sizeof(C_tile);
    static constexpr int input_stage = 2 * sA + sB;

    // Reg path needs TWO transient C tiles (one per math warpgroup half, wg0/wg1
    // each store a 64-row half), exactly like the union path's C[2].
    static constexpr int smem_total = REG_EPILOGUE_
        ? (STAGES_A_ * 2 * sA + STAGES_B_ * sB + 2 * sC)   // decoupled A/B + 2 transient C
        : ((STAGES_B_ - 1) * input_stage                    // legacy union: S-1 full input stages
           + (input_stage > 2 * sC ? input_stage : 2 * sC)); // + max(last input stage, C[2])

    static_assert(smem_total + 2048 <= SMEM_BUDGET_,
                  "gemm_sm90: instantiation exceeds SMEM budget");
};

// ---- per-expert row-block layout the caller supplies (no globals leak) -----
struct GroupLayout {
    const int* padded_tokens_per_expert;   // [E] SMEM-resident, caller-loaded
    const int* local_rb_per_expert;        // [E] local-first split
    int  num_experts;                      // E
    int  col_blocks;                       // N / COL_BLOCK
    int  num_iters;                        // K / RED_BLOCK
    int  super_m;                          // L2-rasterization factor (1 = row-major)
};

// ---- optional gate the consumer waits on before issuing a tile (gemm1 only)-
struct NullGate {
    __device__ inline bool wait(int /*row_idx*/, int /*lane*/) const { return true; }
};

// Super-M (L2 weight-reuse) decode of an EXPERT-LOCAL task_id. Verbatim from
// the original dispatch_super_m_decode (src/...cu:220-237). super_m==1 is
// provably byte-identical to row-major.
__device__ inline void super_m_decode(int task_id, int row_blocks, int col_blocks,
                                      int super_m, int& row_in_grid, int& col_idx) {
    if (super_m < 1) super_m = 1;
    const int super_rows   = (row_blocks / super_m) * super_m;
    const int super_blocks = super_m * col_blocks;
    if (task_id < super_rows * col_blocks) {
        const int band     = task_id / super_blocks;
        const int in_band  = task_id - band * super_blocks;
        col_idx     = in_band / super_m;
        row_in_grid = band * super_m + (in_band % super_m);
        return;
    }
    const int tail_rows = row_blocks - super_rows;
    const int tail_idx  = task_id - super_rows * col_blocks;
    col_idx     = tail_idx / tail_rows;
    row_in_grid = super_rows + (tail_idx % tail_rows);
}

// ============================================================================
// LEGACY UNION EPILOGUE (REG_EPILOGUE == false)
// Byte-for-byte behavioral equivalent of the original grouped_gemm. C[2] is
// unioned onto the last input stage; STAGES_A == STAGES_B.
// ============================================================================
template <class Cfg, class AT, class BT, class CT, class Gate>
__device__ inline void run_union(int* scratch, int sm_idx, int num_sms,
                                 const AT& A_t, const BT& B_t, CT& C_t,
                                 const GroupLayout& L, const Gate& gate) {
    constexpr int STAGES   = Cfg::STAGES_B;            // == STAGES_A here
    constexpr int NUM_PASSES = 2;
    using A_tile = typename Cfg::A_tile;
    using B_tile = typename Cfg::B_tile;
    using C_tile = typename Cfg::C_tile;
    struct pipeline_inputs  { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };

    tma_swizzle_allocator allocator(scratch);
    pipeline_inputs (&inputs)[STAGES] =
        allocator.allocate<pipeline_inputs, STAGES>();
    pipeline_outputs &outputs =
        *reinterpret_cast<pipeline_outputs*>(&inputs[STAGES - 1]);

    __shared__ semaphore inputs_arrived[STAGES];
    __shared__ semaphore inputs_finished[STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    const int num_iters  = L.num_iters;
    const int col_blocks = L.col_blocks;
    const int E = L.num_experts;

    if (wg_id == 2) {
        warpgroup::decrease_registers<40>();
        if (w_id == 0) {
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                for (int expert_id = 0; expert_id < E; expert_id++) {
                    const int rb_start_e = cum / Cfg::ROW_BLOCK;
                    cum += L.padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = L.local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _r, _c;
                        super_m_decode(task_id, row_blocks, col_blocks, L.super_m, _r, _c);
                        const int row_idx = _r + row_offset;
                        const int col_idx = _c;
                        if (l_id == 0) gate.wait(row_idx, l_id);
                        __syncwarp();
                        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                            if (l_id == 0) {
                                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                                update_phasebit<1>(phasebits, stage);
                                if (red_idx == STAGES - 1) {
                                    wait(outputs_finished, get_phasebit<1>(phasebits, STAGES));
                                    update_phasebit<1>(phasebits, STAGES);
                                }
                            }
                            __syncwarp();
                            if (l_id == 0) {
                                ::dist::tma::expect_bytes(inputs_arrived[stage], sizeof(pipeline_inputs));
                                #pragma unroll
                                for (int i = 0; i < 2; i++)
                                    ::dist::tma::load_async(inputs[stage].A[i], A_t,
                                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                                ::dist::tma::load_async(inputs[stage].B, B_t,
                                                {expert_id, red_idx, col_idx}, inputs_arrived[stage]);
                            }
                            __syncwarp();
                            stage = (stage + 1) % STAGES;
                        }
                    }
                    task_id -= num_blocks;
                }
            }
        } else if (w_id == 1 && l_id == 0) {
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                for (int expert_id = 0; expert_id < E; expert_id++) {
                    const int rb_start_e = cum / Cfg::ROW_BLOCK;
                    cum += L.padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = L.local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _r, _c;
                        super_m_decode(task_id, row_blocks, col_blocks, L.super_m, _r, _c);
                        const int row_idx = _r + row_offset;
                        const int col_idx = _c;
                        wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                        update_phasebit<0>(phasebits, 0);
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::store_async(C_t, outputs.C[i], {row_idx * 2 + i, col_idx});
                        ::dist::tma::store_async_read_wait();
                        arrive(outputs_finished);
                    }
                    task_id -= num_blocks;
                }
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        #pragma unroll 1
        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int task_id = sm_idx;
            int cum = 0;
            for (int expert_id = 0; expert_id < E; expert_id++) {
                const int rb_start_e = cum / Cfg::ROW_BLOCK;
                cum += L.padded_tokens_per_expert[expert_id];
                const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                const int total_rb = rb_end_e - rb_start_e;
                const int local_rb_e = L.local_rb_per_expert[expert_id];
                const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                const int num_blocks = row_blocks * col_blocks;
                for (; task_id < num_blocks; task_id += num_sms) {
                    rt_fl<Cfg::ROW_BLOCK / 8, Cfg::COL_BLOCK> C_accum;
                    warp::zero(C_accum);
                    int prev_stage = -1;
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        warpgroup::mma_AB(C_accum, inputs[stage].A[wg_id], inputs[stage].B);
                        if (prev_stage >= 0) {
                            warpgroup::mma_async_wait<1>();
                            warp::arrive(inputs_finished[prev_stage]);
                        }
                        prev_stage = stage;
                        stage = (stage + 1) % STAGES;
                    }
                    warpgroup::mma_async_wait<0>();
                    warp::arrive(inputs_finished[prev_stage]);
                    group<8>::sync(3);
                    warpgroup::store(outputs.C[wg_id], C_accum);
                    warpgroup::sync(wg_id + 1);
                    warpgroup::arrive(outputs_arrived);
                }
                task_id -= num_blocks;
            }
        }
    }
}

// ============================================================================
// REGISTER (DECOUPLED) EPILOGUE (REG_EPILOGUE == true)
// A and B stage independently (STAGES_A may exceed STAGES_B). The C output is a
// SINGLE transient SMEM buffer placed AFTER the input pipeline (no union), so C
// no longer steals an input stage. The math warpgroup keeps the accumulator in
// registers, warpgroup::store's into the transient, and the storer warp TMAs it
// out; tma_store_wait is deferred so the next row-block's WGMMA issue overlaps
// the prior C store. MMA_INFLIGHT controls committed-group depth.
// ============================================================================
template <class Cfg, class AT, class BT, class CT, class Gate>
__device__ inline void run_reg(int* scratch, int sm_idx, int num_sms,
                              const AT& A_t, const BT& B_t, CT& C_t,
                              const GroupLayout& L, const Gate& gate) {
    constexpr int SA = Cfg::STAGES_A;
    constexpr int SB = Cfg::STAGES_B;
    constexpr int NUM_PASSES = 2;
    constexpr int INFLIGHT = Cfg::MMA_INFLIGHT;
    using A_tile = typename Cfg::A_tile;
    using B_tile = typename Cfg::B_tile;
    using C_tile = typename Cfg::C_tile;
    struct a_stage { A_tile A[2]; };

    // Decoupled allocation: SA A-stages, SB B-stages, then ONE transient C[2]
    // (one per math warpgroup half) placed after the input pipeline.
    tma_swizzle_allocator allocator(scratch);
    a_stage (&As)[SA]    = allocator.allocate<a_stage, SA>();
    B_tile  (&Bs)[SB]    = allocator.allocate<B_tile, SB>();
    C_tile  (&Cs)[2]     = allocator.allocate<C_tile, 2>();

    __shared__ semaphore A_arrived[SA];
    __shared__ semaphore A_finished[SA];
    __shared__ semaphore B_arrived[SB];
    __shared__ semaphore B_finished[SB];
    __shared__ semaphore C_arrived;
    __shared__ semaphore C_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < SA; ++i) { init_semaphore(A_arrived[i], 0, 1); init_semaphore(A_finished[i], 0, 8); }
        #pragma unroll
        for (int i = 0; i < SB; ++i) { init_semaphore(B_arrived[i], 0, 1); init_semaphore(B_finished[i], 0, 8); }
        init_semaphore(C_arrived, 0, 2);
        init_semaphore(C_finished, 0, 1);
    }
    __syncthreads();

    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();
    const int num_iters  = L.num_iters;
    const int col_blocks = L.col_blocks;
    const int E = L.num_experts;

    if (wg_id == 2) {
        warpgroup::decrease_registers<40>();
        if (w_id == 0) {
            // Producer: independent A and B pipelines, separate stage cursors.
            int a_stage_i = 0, b_stage_i = 0;
            uint32_t a_phase = 0xFFFF0000, b_phase = 0xFFFF0000;
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                for (int expert_id = 0; expert_id < E; expert_id++) {
                    const int rb_start_e = cum / Cfg::ROW_BLOCK;
                    cum += L.padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = L.local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _r, _c;
                        super_m_decode(task_id, row_blocks, col_blocks, L.super_m, _r, _c);
                        const int row_idx = _r + row_offset;
                        const int col_idx = _c;
                        if (l_id == 0) gate.wait(row_idx, l_id);
                        __syncwarp();
                        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                            if (l_id == 0) {
                                // A pipeline
                                wait(A_finished[a_stage_i], get_phasebit<1>(a_phase, a_stage_i));
                                update_phasebit<1>(a_phase, a_stage_i);
                                ::dist::tma::expect_bytes(A_arrived[a_stage_i], 2 * (int)sizeof(A_tile));
                                #pragma unroll
                                for (int i = 0; i < 2; i++)
                                    ::dist::tma::load_async(As[a_stage_i].A[i], A_t,
                                                    {row_idx * 2 + i, red_idx}, A_arrived[a_stage_i]);
                                // B pipeline
                                wait(B_finished[b_stage_i], get_phasebit<1>(b_phase, b_stage_i));
                                update_phasebit<1>(b_phase, b_stage_i);
                                ::dist::tma::expect_bytes(B_arrived[b_stage_i], (int)sizeof(B_tile));
                                ::dist::tma::load_async(Bs[b_stage_i], B_t,
                                                {expert_id, red_idx, col_idx}, B_arrived[b_stage_i]);
                            }
                            __syncwarp();
                            a_stage_i = (a_stage_i + 1) % SA;
                            b_stage_i = (b_stage_i + 1) % SB;
                        }
                    }
                    task_id -= num_blocks;
                }
            }
        } else if (w_id == 1 && l_id == 0) {
            // Storer: TMA the transient C[2] out, deferred read-wait.
            uint32_t c_phase = 0xFFFF0000;
            #pragma unroll 1
            for (int pass = 0; pass < NUM_PASSES; ++pass) {
                int task_id = sm_idx;
                int cum = 0;
                for (int expert_id = 0; expert_id < E; expert_id++) {
                    const int rb_start_e = cum / Cfg::ROW_BLOCK;
                    cum += L.padded_tokens_per_expert[expert_id];
                    const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                    const int total_rb = rb_end_e - rb_start_e;
                    const int local_rb_e = L.local_rb_per_expert[expert_id];
                    const int row_offset = (pass == 0) ? rb_start_e : (rb_start_e + local_rb_e);
                    const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                    const int num_blocks = row_blocks * col_blocks;
                    for (; task_id < num_blocks; task_id += num_sms) {
                        int _r, _c;
                        super_m_decode(task_id, row_blocks, col_blocks, L.super_m, _r, _c);
                        const int row_idx = _r + row_offset;
                        const int col_idx = _c;
                        wait(C_arrived, get_phasebit<0>(c_phase, 0));
                        update_phasebit<0>(c_phase, 0);
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            ::dist::tma::store_async(C_t, Cs[i], {row_idx * 2 + i, col_idx});
                        ::dist::tma::store_async_read_wait();
                        arrive(C_finished);
                    }
                    task_id -= num_blocks;
                }
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        int a_stage_i = 0, b_stage_i = 0;
        uint32_t a_phase = 0xFFFF0000, b_phase = 0xFFFF0000, c_phase = 0xFFFF0000;
        #pragma unroll 1
        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int task_id = sm_idx;
            int cum = 0;
            for (int expert_id = 0; expert_id < E; expert_id++) {
                const int rb_start_e = cum / Cfg::ROW_BLOCK;
                cum += L.padded_tokens_per_expert[expert_id];
                const int rb_end_e = (cum + Cfg::ROW_BLOCK - 1) / Cfg::ROW_BLOCK;
                const int total_rb = rb_end_e - rb_start_e;
                const int local_rb_e = L.local_rb_per_expert[expert_id];
                const int row_blocks = (pass == 0) ? local_rb_e : (total_rb - local_rb_e);
                const int num_blocks = row_blocks * col_blocks;
                for (; task_id < num_blocks; task_id += num_sms) {
                    rt_fl<Cfg::ROW_BLOCK / 8, Cfg::COL_BLOCK> C_accum;
                    warp::zero(C_accum);
                    int prev_a = -1, prev_b = -1;
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(A_arrived[a_stage_i], get_phasebit<0>(a_phase, a_stage_i));
                        update_phasebit<0>(a_phase, a_stage_i);
                        wait(B_arrived[b_stage_i], get_phasebit<0>(b_phase, b_stage_i));
                        update_phasebit<0>(b_phase, b_stage_i);
                        warpgroup::mma_AB(C_accum, As[a_stage_i].A[wg_id], Bs[b_stage_i]);
                        if (prev_a >= 0) {
                            warpgroup::mma_async_wait<INFLIGHT>();
                            warp::arrive(A_finished[prev_a]);
                            warp::arrive(B_finished[prev_b]);
                        }
                        prev_a = a_stage_i; prev_b = b_stage_i;
                        a_stage_i = (a_stage_i + 1) % SA;
                        b_stage_i = (b_stage_i + 1) % SB;
                    }
                    warpgroup::mma_async_wait<0>();
                    warp::arrive(A_finished[prev_a]);
                    warp::arrive(B_finished[prev_b]);
                    // Wait the storer drained the transient C[wg_id] from the
                    // PREVIOUS row-block before overwriting it.
                    if (wg_id == 0 && l_id == 0) {
                        wait(C_finished, get_phasebit<1>(c_phase, 0));
                        update_phasebit<1>(c_phase, 0);
                    }
                    group<8>::sync(3);
                    warpgroup::store(Cs[wg_id], C_accum);
                    warpgroup::sync(wg_id + 1);
                    warpgroup::arrive(C_arrived);
                }
                task_id -= num_blocks;
            }
        }
    }
}

// ---- the entry point -------------------------------------------------------
template <class Cfg, class AT, class BT, class CT, class Gate = NullGate>
__device__ inline void run(int* scratch, int sm_idx, int num_sms,
                          const AT& A, const BT& B, CT& C,
                          const GroupLayout& L, const Gate& gate = {}) {
    if constexpr (Cfg::REG_EPILOGUE)
        run_reg<Cfg>(scratch, sm_idx, num_sms, A, B, C, L, gate);
    else
        run_union<Cfg>(scratch, sm_idx, num_sms, A, B, C, L, gate);
}

} // namespace gemm_sm90
