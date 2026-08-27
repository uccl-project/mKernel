#pragma once

#include "pyutils/torchutils.cuh"

using namespace kittens;

// When AG_GEMM_BARRIER_READBACK_DEBUG is defined, barrier readback verification
// is enabled around signal/clear sites. This helps catch shape-sensitive
// accounting bugs that would otherwise manifest as deadlocks.

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

#ifndef AG_GEMM_SKIP_COMMON_CONFIG_GLOBALS
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 132;

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
    static constexpr int DEFAULT_REGISTERS = 128;
};

struct globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using A_comm_tile = st_bf<ROW_BLOCK * 2, RED_BLOCK * 2>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    static constexpr int NUM_CHUNKS =
        config::DYNAMIC_SHARED_MEMORY / sizeof(globals::A_comm_tile);
    static constexpr int COMM_NUM_WORKERS =
        (config::NUM_WARPGROUPS < NUM_CHUNKS ? config::NUM_WARPGROUPS
                                             : NUM_CHUNKS);

    using A_pgl = pgl<gl<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES,
                      true, A_comm_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using barrier_pgl = barrier_t<NUM_DEVICES>;

    A_pgl A;
    B_gl B;
    C_gl C;
    barrier_pgl barrier;

    const int dev_idx;
    const unsigned int active_sms;  // Runtime CTA count (default: config::NUM_BLOCKS)

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};
#endif

template <typename G_t>
__device__ inline void comm_sm_row(
    const G_t& G,
    int local_row_idx,
    typename globals::A_comm_tile (&A_smem)[globals::NUM_CHUNKS],
    kittens::semaphore (&inputs_arrived)[globals::NUM_CHUNKS],
    uint32_t& phasebits) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane_id = warp::laneid();
    if (warpgroup_id >= globals::COMM_NUM_WORKERS || warp_in_wg != 0 ||
        lane_id != 0)
        return;

    const int global_row_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int col_blocks = G.A.cols() / (globals::RED_BLOCK * 2);
    const int global_row_idx =
        local_row_idx + G.dev_idx * (global_row_blocks / globals::NUM_DEVICES);

    for (int col_idx = warpgroup_id; col_idx < col_blocks;
         col_idx += globals::COMM_NUM_WORKERS) {
        tma::expect_bytes(inputs_arrived[warpgroup_id],
                          sizeof(globals::A_comm_tile));
        tma::load_async(A_smem[warpgroup_id], G.A[G.dev_idx],
                        {global_row_idx, col_idx},
                        inputs_arrived[warpgroup_id]);
        wait(inputs_arrived[warpgroup_id], get_phasebit<0>(phasebits, warpgroup_id));
        update_phasebit<0>(phasebits, warpgroup_id);
        tma::store_async(G.A, A_smem[warpgroup_id], {global_row_idx, col_idx});
    }
    tma::store_async_wait();
}

template <typename G_t>
__device__ inline void comm_sm_row(
    const G_t& G,
    int local_row_idx,
    typename globals::A_comm_tile (&A_smem)[globals::NUM_CHUNKS],
    kittens::semaphore (&inputs_arrived)[globals::NUM_CHUNKS],
    uint32_t& phasebits,
    const int num_comm_workers_per_stage) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane_id = warp::laneid();
    if (warpgroup_id >= globals::COMM_NUM_WORKERS || warp_in_wg != 0 ||
        lane_id != 0)
        return;

    const int global_row_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = G.A.cols() / (globals::RED_BLOCK * 2);
    const int global_row_idx = local_row_idx + G.dev_idx * local_row_blocks;

    for (int col_idx = 0; col_idx < col_blocks; col_idx++) {
        tma::expect_bytes(inputs_arrived[warpgroup_id],
                          sizeof(globals::A_comm_tile));
        tma::load_async(A_smem[warpgroup_id], G.A[G.dev_idx],
                        {global_row_idx, col_idx},
                        inputs_arrived[warpgroup_id]);
        wait(inputs_arrived[warpgroup_id], get_phasebit<0>(phasebits, warpgroup_id));
        update_phasebit<0>(phasebits, warpgroup_id);
        tma::store_async(G.A, A_smem[warpgroup_id], {global_row_idx, col_idx});
        tma::store_async_wait();
    }
    signal_all(G.barrier, coord<ducks::default_type>(0, 0, 0, global_row_idx),
               num_comm_workers_per_stage);
}

template <typename G_t>
__device__ inline void comp_sm_row(
    const G_t& G,
    int row_idx,
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES],
    globals::pipeline_outputs& outputs,
    kittens::semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    kittens::semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    kittens::semaphore& outputs_arrived,
    kittens::semaphore& outputs_finished,
    int& stage,
    uint32_t& phasebits,
    const int num_iters,
    const int num_comm_workers_per_stage,
    const int local_row_blocks,
    const int col_blocks) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    const int local_row_start = G.dev_idx * local_row_blocks;
    const bool is_remote_row =
        row_idx < local_row_start || row_idx >= local_row_start + local_row_blocks;

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if (warp_id == 0 && lane_id == 0) {
            if (is_remote_row) {
                wait(G.barrier, coord<ducks::default_type>(0, 0, 0, row_idx / 2),
                     G.dev_idx, 1);
            }

            for (int col_idx = 0; col_idx < col_blocks; col_idx++) {
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
                        tma::load_async(inputs[stage].A[i], G.A[G.dev_idx],
                                        {row_idx * 2 + i, red_idx},
                                        inputs_arrived[stage]);
                    }
                    tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx},
                                    inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            for (int col_idx = 0; col_idx < col_blocks; col_idx++) {
                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::store_async(G.C, outputs.C[i], {row_idx * 2 + i, col_idx});
                }
                tma::store_async_read_wait();
                arrive(outputs_finished);
            }
        }
    } else {
        for (int col_idx = 0; col_idx < col_blocks; col_idx++) {
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
}

#ifndef AG_GEMM_SKIP_COMMON_CONFIG_GLOBALS
__device__ __forceinline__ void adjust_registers_for_comp() {
    if (warpgroup::groupid() == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
    }
}

__device__ __forceinline__ void adjust_registers_for_comm() {
    if (warpgroup::groupid() == config::NUM_WARPGROUPS - 1) {
        warpgroup::increase_registers<config::DEFAULT_REGISTERS -
                                      config::PRODUCER_REGISTERS>();
    } else {
        warpgroup::decrease_registers<config::CONSUMER_REGISTERS -
                                      config::DEFAULT_REGISTERS>();
    }
}

struct comp_layout {
    globals::pipeline_inputs* inputs;
    globals::pipeline_outputs* outputs;
};

__device__ __forceinline__ comp_layout setup_comp_path(
    tma_swizzle_allocator& al,
    semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    semaphore& outputs_arrived,
    semaphore& outputs_finished) {
    adjust_registers_for_comp();

    static_assert(sizeof(globals::pipeline_inputs) *
                          (globals::PIPELINE_STAGES - 1) +
                      sizeof(globals::pipeline_outputs) <=
                  config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs(&inputs)[globals::PIPELINE_STAGES] =
        al.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs& outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(
            &inputs[globals::PIPELINE_STAGES - 1]);
    return comp_layout{&inputs[0], &outputs};
}

struct comm_layout {
    typename globals::A_comm_tile* A_smem;
};

template <int NumChunks = globals::NUM_CHUNKS>
__device__ __forceinline__ comm_layout setup_comm_path(
    tma_swizzle_allocator& al,
    semaphore (&comm_inputs_arrived)[globals::NUM_CHUNKS]) {
    static_assert(NumChunks >= config::NUM_WARPGROUPS,
                  "comm is warpgroup-based: need at least one chunk per warpgroup");
    typename globals::A_comm_tile(&A_smem)[NumChunks] =
        al.allocate<typename globals::A_comm_tile, NumChunks>();
    return comm_layout{&A_smem[0]};
}
#endif

template <typename G_t>
__device__ __forceinline__ void comm_sm_tile(
    const G_t& G,
    int global_row_idx,
    int col_idx,
    bool should_signal,
    typename globals::A_comm_tile& A_smem_chunk,
    kittens::semaphore& sem,
    uint32_t& phasebits,
    int phasebit_idx) {
    tma::expect_bytes(sem, sizeof(globals::A_comm_tile));
    tma::load_async(A_smem_chunk, G.A[G.dev_idx], {global_row_idx, col_idx}, sem);
    wait(sem, get_phasebit<0>(phasebits, phasebit_idx));
    update_phasebit<0>(phasebits, phasebit_idx);
    tma::store_async(G.A, A_smem_chunk, {global_row_idx, col_idx});
    tma::store_async_wait();
    if (should_signal) {
        auto slot = coord<ducks::default_type>(0, 0, 0, global_row_idx);
        signal_all(G.barrier, slot, 1);
#ifdef AG_GEMM_BARRIER_READBACK_DEBUG
        int readback;
        asm volatile(
            "multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
            : "=r"(readback)
            : "l"(G.barrier.mc_ptr_at(slot))
            : "memory");
        (void)readback;
#endif
    }
}

template <typename G_t>
__device__ __forceinline__ void comp_sm_tile(
    const G_t& G,
    int row_idx,
    int col_idx,
    bool is_remote_row,
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES],
    globals::pipeline_outputs& outputs,
    kittens::semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    kittens::semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    kittens::semaphore& outputs_arrived,
    kittens::semaphore& outputs_finished,
    int& stage,
    uint32_t& phasebits,
    const int num_iters,
    const int num_comm_workers_per_stage) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if (warp_id == 0 && lane_id == 0) {
            if (false && is_remote_row) {
                wait_mc(G.barrier, coord<ducks::default_type>(0, 0, 0, row_idx / 2),
                        num_comm_workers_per_stage);
            }

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
                    tma::load_async(inputs[stage].A[i], G.A[G.dev_idx],
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
                tma::store_async(G.C, outputs.C[i], {row_idx * 2 + i, col_idx});
            }
            tma::store_async_read_wait();
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

template <typename G_t, int ExpectedBeforeClear = -1>
__device__ inline void epilogue_kernel(const G_t& G) {
    const int num_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int col_blocks_comm = G.A.cols() / (globals::RED_BLOCK * 2);
    const int expected_before =
        (ExpectedBeforeClear < 0) ? col_blocks_comm : ExpectedBeforeClear;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        barrier_all(G.barrier, coord<ducks::default_type>(0, 0, 0, 1023),
                    G.dev_idx);
    }
    __syncthreads();

    for (int i = offset; i < num_blocks; i += stride) {
        auto slot = coord<ducks::default_type>(0, 0, 0, i);
#ifdef AG_GEMM_BARRIER_READBACK_DEBUG
        int readback_before;
        asm volatile(
            "multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
            : "=r"(readback_before)
            : "l"(G.barrier.mc_ptr_at(slot))
            : "memory");
        if (readback_before != expected_before) {
            printf(
                "[adaptive ag_gemm epilogue] dev=%d slot=%d readback_before=%d "
                "expected_before=%d col_blocks_comm=%d\n",
                G.dev_idx, i, readback_before, expected_before, col_blocks_comm);
            asm volatile("trap;");
        }
#endif
        clear_slot_mc(G.barrier, slot);
#ifdef AG_GEMM_BARRIER_READBACK_DEBUG
        int readback_after;
        asm volatile(
            "multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
            : "=r"(readback_after)
            : "l"(G.barrier.mc_ptr_at(slot))
            : "memory");
        if (readback_after != 0) {
            printf(
                "[adaptive ag_gemm epilogue] dev=%d slot=%d readback_after=%d\n",
                G.dev_idx, i, readback_after);
            asm volatile("trap;");
        }
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        barrier_all(G.barrier, coord<ducks::default_type>(0, 0, 0, 1023),
                    G.dev_idx);
    }
    __syncthreads();
}
