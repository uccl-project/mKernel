#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/atomic_u32.cuh"
#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor_tensor.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/gemm_ar/gemm_ar_blackwell.cuh"
// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
#include "memory/tk_ops_group_util_util.cuh"
// clang-format on

using namespace kittens;

namespace gemm_ar_intranode_blackwell {

// use snake-like pattern, but without the safety path for odd numbered shapes referenced from:
// https://github.com/HazyResearch/ThunderKittens/blob/0230013a72b51338a137b50f69538ec69d4d4675/include/common/util.cuh#L367
template <int SUPERGROUP_WIDTH = 8>
__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int num_rows,
                                                                   int num_cols,
                                                                   int tile_idx) {
    const int supergroup_numel = num_rows * SUPERGROUP_WIDTH;
    const int supergroup_idx = tile_idx / supergroup_numel;

    const int row_idx = (tile_idx % supergroup_numel) / SUPERGROUP_WIDTH;
    const int col_idx = supergroup_idx * SUPERGROUP_WIDTH + tile_idx % SUPERGROUP_WIDTH;

    return {(supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx};
};

template <int SUPERGROUP_WIDTH>
__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.prefetch_tma<fused_globals::A_tile>();
        G.B.prefetch_tma<fused_globals::B_tile>();
        G.C_dist[G.dev_idx].prefetch_tma<fused_globals::C_tile>();
    }

    // Each TMA load loads 2 128 * 64 A tiiles and 64 * 256 B tile
    // These tiles will then participate in 2CTA MMA, to produce a 256 * 256 tile per SM
    const int num_row_tiles = G.M / (fused_globals::ROW_BLOCK * config::NUM_CLUSTERS);
    const int num_col_tiles = G.N / fused_globals::COL_BLOCK;
    const int num_tiles_total = num_row_tiles * num_col_tiles;
    const int cluster_idx = blockIdx.x / config::NUM_CLUSTERS;
    const int num_comp_clusters = config::NUM_COMP_SM / config::NUM_CLUSTERS;

    // allocate smem and tmem
    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    fused_globals::pipeline_inputs(&inputs_smem)[fused_globals::PIPELINE_STAGES] =
        smem_allocator.allocate<fused_globals::pipeline_inputs, fused_globals::PIPELINE_STAGES>();
    fused_globals::C_tile(&C_smem)[config::CONSUMER_WARPS][fused_globals::NUM_C_TILES] =
        smem_allocator
            .allocate<fused_globals::C_tile, config::CONSUMER_WARPS, fused_globals::NUM_C_TILES>();

    __shared__ semaphore tma_load[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore mma_finish[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore epilogue_ready[config::CONSUMER_WARPS];
    __shared__ semaphore epilogue_tmem_finished[config::CONSUMER_WARPS];

    tensor_allocator<1, config::NUM_CLUSTERS> tm_alloc{};

    // combined phasebits, one bit per barrier array (see the PHASE_BIT_*
    // layout in config). A bit is toggled once per full ring traversal of its
    // array, i.e. when the stage index wraps to 0 -- the epilogue rings hold a
    // single barrier per consumer, so their bits flip on every iteration.
    uint32_t phasebits = config::PHASE_BITS_INIT;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; i++) {
            // TMA load needs to wait for both clusters to finish
            init_semaphore(tma_load[i], 0, config::NUM_CLUSTERS);
            // tma in each CTA needs to wait for both consumers to finish
            init_semaphore(mma_finish[i], 0, config::CONSUMER_WARPS);
        }

#pragma unroll
        for (int c = 0; c < config::CONSUMER_WARPS; c++) {
            init_semaphore(epilogue_ready[c], 0, 1);
            // broadcasted back to the mma thread to signal that tmem is ready
            init_semaphore(epilogue_tmem_finished[c], WARPGROUP_WARPS * config::NUM_CLUSTERS, 0);
        }
    }

    // flush to ensure the mbarriers are visible
    everyone::tma::cluster::sync();

    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(mma_finish[input_stage_id], (phasebits >> config::PHASE_BIT_MMA_FINISH) & 0b1);

            tma::cluster::expect_bytes(tma_load[input_stage_id],
                                       sizeof(fused_globals::A_tile) * config::CONSUMER_WARPS +
                                           sizeof(fused_globals::B_tile),
                                       0);

#pragma unroll
            for (int c = 0; c < config::CONSUMER_WARPS; c++) {
                tma::cluster::load_async(inputs_smem[input_stage_id].A[c],
                                         G.A,
                                         {tile_row_idx + c, i},
                                         tma_load[input_stage_id],
                                         (uint16_t)(1 << cta_rank),
                                         0);
            }

            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {i,
                                      (tile_col_idx * config::NUM_CLUSTERS +
                                       cta_rank)},  // this has to be here becuase the epilogue
                                                    // warp loads the tiles in col units of 256
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << config::PHASE_BIT_MMA_FINISH);
            }
        }
    };

    auto consume = [&](int& input_stage_id, fused_globals::C_tt_tile* tmem, const int consumer_id) {
        wait(epilogue_tmem_finished[consumer_id],
             (phasebits >> (config::PHASE_BIT_EPILOGUE_FINISHED + consumer_id)) & 0b1);

        {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> config::PHASE_BIT_TMA_LOAD) & 0b1);

            mm2_AB(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;

            if (input_stage_id == 0) {
                phasebits ^= (1 << config::PHASE_BIT_TMA_LOAD);
            }
        }

        for (int i = 1; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> config::PHASE_BIT_TMA_LOAD) & 0b1);

            mma2_AB(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << config::PHASE_BIT_TMA_LOAD);
            }
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready[consumer_id]);

        // This consumer owns exactly one accumulator, so epilogue_finished
        // completes once per output tile and its phase flips every iteration.
        phasebits ^= (1 << (config::PHASE_BIT_EPILOGUE_FINISHED + consumer_id));
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        fused_globals::C_tt_tile* tmem,
                        const int warpgroup_id,
                        bool is_last_tile) {
        wait(epilogue_ready[warpgroup_id],
             (phasebits >> (config::PHASE_BIT_EPILOGUE_READY + warpgroup_id)) & 0b1);

        const auto& C_out = G.C_dist[G.dev_idx];
        constexpr int C_CHUNK_COLS = fused_globals::COL_BLOCK / fused_globals::EPILOGUE_STAGES;
        rt_bf<fused_globals::ROW_BLOCK / (4 * config::CONSUMER_WARPS), C_CHUNK_COLS>
            c_reg[fused_globals::EPILOGUE_STAGES];
        // + 1 because __syncthreads makes use of id = 0
        const int epilogue_barrier = warpgroup_id + 1;

#pragma unroll
        for (int i = 0; i < fused_globals::EPILOGUE_STAGES; i++) {
            warpgroup::load_async(
                c_reg[i],
                tmem[0]
                    .template subtile<
                        tt<float, fused_globals::ROW_BLOCK / config::CONSUMER_WARPS, C_CHUNK_COLS>>(
                        i * C_CHUNK_COLS));
        }
        tensor_load_wait();
        warpgroup::sync(epilogue_barrier);

        // signal tmem empty
        if (elect_warp_leader()) {
            tma::cluster::arrive(epilogue_tmem_finished[warpgroup_id], 0);
        }

#pragma unroll
        for (int i = 0; i < fused_globals::EPILOGUE_STAGES; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fused_globals::NUM_C_TILES - 1>();
            warpgroup::sync(epilogue_barrier);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[warpgroup_id][i % fused_globals::NUM_C_TILES], c_reg[i]);
            warpgroup::sync(epilogue_barrier);

            if (warpgroup::laneid() == 0) {
                // C_tile is only COL_BLOCK / EPILOGUE_STAGES wide, so the TMA
                // column coordinate counts chunks, not COL_BLOCK tiles.
                dist::tma::store_async(
                    C_out,
                    C_smem[warpgroup_id][i % fused_globals::NUM_C_TILES],
                    {tile_row_idx, tile_col_idx * fused_globals::EPILOGUE_STAGES + i});
            }
        }

        // Same single-accumulator ring as the consumer side: epilogue_ready
        // completes once per output tile, so this flips every iteration.
        phasebits ^= (1 << (config::PHASE_BIT_EPILOGUE_READY + warpgroup_id));
    };

    // Row tiles are A_tile/C_tile sized (ROW_BLOCK / CONSUMER_WARPS rows), so a
    // cluster block spans NUM_CLUSTERS * CONSUMER_WARPS of them: this CTA owns
    // the CONSUMER_WARPS tiles starting here, one per consumer.
    const int cta_row_tile_base = cta_rank * config::CONSUMER_WARPS;

    // producer + consumers share the tail warpgroup(s)
    if (warpgroup_id >= config::EPILOGUE_WARPGROUPS) {
        warpgroup::decrease_registers<config::MAINLOOP_REGISTERS>();

        if (warp_id == config::PRODUCER_WARP_ID) {
            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int tile_id = cluster_idx; tile_id < num_tiles_total;
                     tile_id += num_comp_clusters) {
                    auto [tile_row_id, tile_col_id] =
                        calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
                    // This specifies the 256 * 256 chunk that has to be loaded
                    load(tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                             cta_row_tile_base,
                         tile_col_id,
                         input_stage_id);
                }
            }
        } else if (warp_id >= config::FIRST_CONSUMER_WARP_ID &&
                   warp_id < config::FIRST_CONSUMER_WARP_ID + config::CONSUMER_WARPS) {
            if (cta_rank == 0 && elect_warp_leader()) {
                const int consumer_id = warp_id - config::FIRST_CONSUMER_WARP_ID;

                // give each warp its own view of tmem
                fused_globals::C_tt_tile tmem[1];
                tmem[0] = tm_alloc.allocate<fused_globals::C_tt_tile>(consumer_id *
                                                                      fused_globals::COL_BLOCK);

                int input_stage_id = 0;
                for (int iter = cluster_idx; iter < num_tiles_total; iter += num_comp_clusters) {
                    consume(input_stage_id, tmem, consumer_id);
                }
            }
        }
    } else {
        warpgroup::increase_registers<config::EPILOGUE_REGISTERS>();
        // give each warpgroup its own view of tmem
        fused_globals::C_tt_tile tmem[1];
        tmem[0] =
            tm_alloc.allocate<fused_globals::C_tt_tile>(warpgroup_id * fused_globals::COL_BLOCK);

        for (int tile_id = cluster_idx; tile_id < num_tiles_total; tile_id += num_comp_clusters) {
            // this returns an index in the 512 * 256 tile
            auto [tile_row_id, tile_col_id] =
                calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
            const int c_row_tile = tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                cta_row_tile_base + warpgroup_id;
            // This specifies the 128 * 256 tile that should be epilogu-ed --
            // the same row tile the producer loaded into A[warpgroup_id].
            epilogue(c_row_tile,
                     tile_col_id,
                     tmem,
                     warpgroup_id,
                     tile_id + num_comp_clusters >= num_tiles_total);

            // TMA store visible in GMEM
            dist::tma::store_async_wait();

            // Along a vertical cluster block, CTA0 holds sub-rows 0,1 and CTA1
            // holds 2,3
            const int device_to_signal = (c_row_tile % 4) + (tile_col_id % 2) * 4;

            // NOTE: there is no need to use a warpgroup::sync() here, becuase the thread issuing
            // the TMA store is also the thread doing the TMA wait, so the signal will not be sent
            // until the TMA wait is completed
            if (warpgroup::laneid() == 0) {
                // https://github.com/NVIDIA/cutlass/issues/3117#issuecomment-5179892505
                // only need a GPU scope, because the data only needs to be present in local L2
                // for peer to read over nvlink
                comm::atomic_u32::release_store_gpu(
                    &G.comp_comm_barrier[G.dev_idx][{c_row_tile, tile_col_id}], G.epoch);
            }
        }
    }
}

// ============================================================================
// Intranode All-Reduce
// ============================================================================
//
// Ported from gemm_ar.cu's gemm_ar_pipelined_ar_tile
template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void pipelined_ar_tile(const fused_globals& G,
                                                  int row_base,
                                                  int col_base);

template <int SUPERGROUP_WIDTH, int AR_UNROLL>
__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    const int iter_gate_value = G.epoch * config::NUM_DEVICES;

    // we would like to handle tiles on a 128*256 basis, so the for loop should go based on that
    const int num_tiles_per_row = G.N / fused_globals::COL_BLOCK;
    const int num_row_tiles = G.M / (fused_globals::ROW_BLOCK / config::CONSUMER_WARPS);
    const int num_tiles_total = num_row_tiles * num_tiles_per_row;
    const int comm_block_idx = blockIdx.x - config::NUM_COMP_SM;
    constexpr int NUM_DEVICES_PER_TILE = 4;

    const int num_comm_row_tiles = num_row_tiles / (config::CONSUMER_WARPS * config::NUM_CLUSTERS);
    const int comm_row_idx = G.dev_idx % NUM_DEVICES_PER_TILE;
    static_assert(config::NUM_DEVICES >= NUM_DEVICES_PER_TILE,
                  "Intranode design was not made for < 4 devices");

    const int tile_id_stride = config::NUM_DEVICES * config::NUM_COMM_SM;
    for (int tile_id = G.dev_idx + comm_block_idx * config::NUM_DEVICES; tile_id < num_tiles_total;
         tile_id += tile_id_stride) {
        // 4 devices handle 1 comp tile, stacked vertically, since completions come in 512 * 256
        // groups
        const int comm_level_tile_id = tile_id / NUM_DEVICES_PER_TILE;
        auto [tile_row_idx, tile_col_idx] = calculate_tile_idx<SUPERGROUP_WIDTH>(
            num_comm_row_tiles, num_tiles_per_row, comm_level_tile_id);

        // we can use a relaxed wait here, since every operation after this is multimem, which does
        // not go through the L1 cache + signal from before is a release add operation
        const int actual_tile_row = tile_row_idx * NUM_DEVICES_PER_TILE + comm_row_idx;
        if (threadIdx.x == 0) {
            int val;
            do {
                comm::multimem<int>::ld_reduce<comm::reduce_op::MIN, comm::memory_model::STRONG>(
                    val,
                    reinterpret_cast<const int*>(
                        G.comp_comm_barrier.mc_ptr_at({actual_tile_row, tile_col_idx})));
            } while (val < (int)G.epoch);
        }
        __syncthreads();

        const int row_base = actual_tile_row * (fused_globals::ROW_BLOCK / config::CONSUMER_WARPS);
        const int col_base = tile_col_idx * fused_globals::COL_BLOCK;
        pipelined_ar_tile<AR_UNROLL,
                          fused_globals::ROW_BLOCK / config::CONSUMER_WARPS,
                          fused_globals::COL_BLOCK>(G, row_base, col_base);
    }
}

namespace ar_detail {
// create variations for how much caching can be done based on the unroll factor, to stay within
// the register limits

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void ar_unroll_no_cache(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base);

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void ar_unroll_ld_cached(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base);

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void ar_unroll_cached(const fused_globals::C_distributed_tensor& C_dist,
                                                 const fused_globals::C_final_tensor& C_final,
                                                 int row_base,
                                                 int col_base);
}  // namespace ar_detail

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void pipelined_ar_tile(const fused_globals& G,
                                                  int row_base,
                                                  int col_base) {
    if constexpr (AR_UNROLL >= 32) {
        ar_detail::ar_unroll_ld_cached<AR_UNROLL, SUBTILE_M, SUBTILE_N>(
            G.C_dist, G.C_final, row_base, col_base);
    } else {
        ar_detail::ar_unroll_cached<AR_UNROLL, SUBTILE_M, SUBTILE_N>(
            G.C_dist, G.C_final, row_base, col_base);
    }
}

template <int SUPERGROUP_WIDTH, int AR_UNROLL>
__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < config::NUM_COMP_SM) {
        fused_comp_sm<SUPERGROUP_WIDTH>(G);
    } else {
        fused_intranode_sm<SUPERGROUP_WIDTH, AR_UNROLL>(G);
    }
}

template <int SUPERGROUP_WIDTH, int AR_UNROLL>
__global__ __cluster_dims__(config::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(config::NUM_THREADS,
                      1) void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel<SUPERGROUP_WIDTH, AR_UNROLL>(G);
}

template <int SUPERGROUP_WIDTH, int AR_UNROLL>
void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int smem_size = fused_globals::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = config::NUM_THREADS;
    constexpr int grid = config::NUM_BLOCKS;  // set aside 20 SMs for comm

    auto this_kernel = gemm_ar_fused_kernel_stub<SUPERGROUP_WIDTH, AR_UNROLL>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    this_kernel<<<grid, num_threads, smem_size, stream>>>(G);
}

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N>
__device__ __forceinline__ void ar_unroll_ld_cached(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base) {
    constexpr int UNITS_PER_ROW = SUBTILE_N / 2;            // 128
    constexpr int TOTAL_UNITS = SUBTILE_M * UNITS_PER_ROW;  // 16384
    constexpr int NT = config::NUM_THREADS;
    constexpr int BATCH = AR_UNROLL * NT;

    for (int base = threadIdx.x; base < TOTAL_UNITS; base += BATCH) {
        comm::bf16_2* ld_ptrs[AR_UNROLL];
        uint32_t tmps[AR_UNROLL];

        // Consecutive threads take consecutive bf16_2 units, so each warp's
        // requests coalesce into contiguous 128B chunks.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * config::NUM_THREADS;
            if (j < TOTAL_UNITS) {
                const int r = row_base + j / UNITS_PER_ROW;
                const int c = col_base + (j % UNITS_PER_ROW) * 2;
                ld_ptrs[u] = reinterpret_cast<comm::bf16_2*>(C_dist.mc_ptr_at({r, c}));
            }
        }

        // All loads before any store — this is the whole point of the helper.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * config::NUM_THREADS;
            if (j < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(tmps[u],
                                                                                 ld_ptrs[u]);
            }
        }

        const ptrdiff_t st_delta = C_final.mc_ptr - C_dist.mc_ptr;  // outside the loop
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            if (base + u * config::NUM_THREADS < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(
                    reinterpret_cast<comm::bf16_2*>(reinterpret_cast<comm::bf16*>(ld_ptrs[u]) +
                                                    st_delta),
                    tmps[u]);
            }
        }
    }
}
};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"