#include "memory/tk_ops_thread_mma_tcgen05.cuh"
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#endif

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cassert>
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
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/gemm_rs/gemm_rs_blackwell.cuh"

// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
#include "memory/tk_ops_group_util_util.cuh"
// clang-format on

using namespace kittens;

namespace gemm_rs_blackwell {
namespace {
template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mm_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 0, NUM_CTA>(d, a, b, sem);
}

template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mma_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 1, NUM_CTA>(d, a, b, sem);
}

template <int TILE_ROWS, int TILE_COLS, int STORES_PER_BATCH>
__device__ __forceinline__ void reduce_scatter_tile(const comm::bf16* multicast_tile,
                                                    comm::bf16* local_tile,
                                                    size_t row_stride,
                                                    int lane) {
    constexpr int ELEMENTS_PER_STORE = 128 / 8 / sizeof(comm::bf16);
    constexpr int LANES_PER_ROW = TILE_COLS / ELEMENTS_PER_STORE;
    constexpr int ROWS_PER_STORE = 32 / LANES_PER_ROW;
    constexpr int ROWS_PER_BATCH = STORES_PER_BATCH * ROWS_PER_STORE;
    // A partial batch would reduce and store rows past the end of the tile.
    static_assert(TILE_ROWS % ROWS_PER_BATCH == 0, "TILE_ROWS must be a whole number of batches");

    const int lane_col = (lane % LANES_PER_ROW) * ELEMENTS_PER_STORE;
    const int lane_row = lane / LANES_PER_ROW;

    for (int batch_row = 0; batch_row < TILE_ROWS; batch_row += ROWS_PER_BATCH) {
        uint32_t reduced[STORES_PER_BATCH][4];

#pragma unroll
        for (int store_idx = 0; store_idx < STORES_PER_BATCH; ++store_idx) {
            const int row = batch_row + store_idx * ROWS_PER_STORE + lane_row;
            const comm::bf16* src = multicast_tile + size_t(row) * row_stride + lane_col;
            reduce_load_16B_no_clobber(reduced[store_idx], src);
        }

#pragma unroll
        for (int store_idx = 0; store_idx < STORES_PER_BATCH; ++store_idx) {
            const int row = batch_row + store_idx * ROWS_PER_STORE + lane_row;
            comm::bf16* dst = local_tile + size_t(row) * row_stride + lane_col;
            store_16B(dst, reduced[store_idx]);
        }
    }
}

template <int SUPERGROUP_WIDTH>
__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int num_rows,
                                                                   int num_cols,
                                                                   int tile_idx) {
    static_assert(SUPERGROUP_WIDTH > 0, "SUPERGROUP_SIZE must be greater than 0");
    const int supergroup_numel = num_rows * SUPERGROUP_WIDTH;
    const int supergroup_idx = tile_idx / supergroup_numel;
    const int supersection_cols = (num_cols / SUPERGROUP_WIDTH) * SUPERGROUP_WIDTH;
    const int supersection_numel = num_rows * supersection_cols;
    const int finalsection_cols = num_cols - supersection_cols;
    int row_idx, col_idx;
    if (tile_idx < supersection_numel) {
        row_idx = (tile_idx % supergroup_numel) / SUPERGROUP_WIDTH;
        col_idx = supergroup_idx * SUPERGROUP_WIDTH + tile_idx % SUPERGROUP_WIDTH;
    } else {
        const int remainder_task_id = tile_idx - supersection_numel;
        row_idx = remainder_task_id / finalsection_cols;
        col_idx = supersection_cols + remainder_task_id % finalsection_cols;
    }
    return {(supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx};
}

__host__ __device__ __forceinline__ bool epoch_reached(uint32_t observed, uint32_t expected) {
    return int32_t(observed - expected) >= 0;
}

}  // namespace

static constexpr int EXIT_BARRIER_THREADS = 256;

__global__ void exit_epoch_barrier(int* local_ready,
                                   int ready_numel,
                                   completion_epoch_dist_tensor epochs,
                                   int dev_idx) {
    auto* local_epoch = epochs[dev_idx].raw_ptr;
    // Every thread reads the epoch before the barrier, so none can see the bump.
    const uint32_t completed_epoch = comm::atomic_u32::acquire_load_gpu(local_epoch) + 1u;

    // we choose to clear the buffer rather than increment an epoch so that
    // we are not bound to the same input length (so we can signal and poll different lengths of
    // the ready buffer)
    for (int i = threadIdx.x; i < ready_numel; i += blockDim.x) {
        local_ready[i] = 0;
    }
    // Order every thread's zeros before the cumulative release.
    __syncthreads();
    if (threadIdx.x == 0) {
        comm::atomic_u32::release_store_sys(local_epoch, completed_epoch);
    }

    for (int peer = threadIdx.x; peer < INTRA_NUM_DEVICES; peer += blockDim.x) {
        while (!epoch_reached(comm::atomic_u32::acquire_load_sys(epochs[peer].raw_ptr),
                              completed_epoch)) {
            __nanosleep(16);
        }
    }
}

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
__device__ __forceinline__ void gemm_rs_blackwell(const fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                                                                      _SIGNALS_PER_TILE,
                                                                      _RS_STRIP_ROWS,
                                                                      _CONSUMER_WARPS>& G) {
    using fg =
        fused_globals<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>;

    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.template prefetch_tma<typename fg::A_tile>();
        G.B.template prefetch_tma<typename fg::B_tile>();
        G.C[G.dev_idx].template prefetch_tma<typename fg::C_tile>();
    }

    const int cluster_idx = blockIdx.x / fg::NUM_CLUSTERS;
    constexpr int num_clusters = fg::NUM_BLOCKS / fg::NUM_CLUSTERS;

    const int num_tile_rows_total = G.M / (fg::A_tile::rows);
    const int num_tile_rows_per_cta = num_tile_rows_total / fg::CONSUMER_WARPS;
    const int num_tile_rows_per_cluster = num_tile_rows_per_cta / fg::NUM_CLUSTERS;

    const int num_tile_cols = G.N / G.COL_BLOCK;

    const int total_num_tiles = num_tile_rows_per_cluster * num_tile_cols;

    // RS calculations
    const int rows_per_device = G.M / fg::NUM_DEVICES;

    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    typename fg::pipeline_inputs(&inputs_smem)[fg::PRODUCER_CONSUMER_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::pipeline_inputs, fg::PRODUCER_CONSUMER_PIPELINE_STAGES>();
    typename fg::C_tile(&C_smem)[fg::EPILOGUE_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::C_tile, fg::EPILOGUE_PIPELINE_STAGES>();

    __shared__ semaphore mma_finish[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];
    __shared__ semaphore tma_load[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];

    __shared__ semaphore epilogue_ready[fg::NUM_TMEM_SLOTS];
    __shared__ semaphore epilogue_tmem_finished[fg::NUM_TMEM_SLOTS];

    // Epilogue -> signal warp: a tile's stores have drained and it may be published.
    __shared__ semaphore signal_full[fg::SIGNAL_PIPELINE_STAGES];
    __shared__ semaphore signal_empty[fg::SIGNAL_PIPELINE_STAGES];

    __shared__ semaphore tmem_allocated;
    __shared__ semaphore tmem_finished;

    __shared__ uint32_t tmem_addr;
    tensor_allocator<1, fg::NUM_CLUSTERS> tm_alloc{};

    uint32_t phasebits = fg::PHASE_BITS_INIT;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fg::PRODUCER_CONSUMER_PIPELINE_STAGES; i++) {
            init_semaphore(mma_finish[i], fg::CONSUMER_WARPS);
            init_semaphore(tma_load[i], fg::NUM_CLUSTERS);
        }

#pragma unroll
        for (int i = 0; i < fg::NUM_TMEM_SLOTS; i++) {
            init_semaphore(epilogue_ready[i], 1);
            init_semaphore(epilogue_tmem_finished[i], WARPGROUP_WARPS * fg::NUM_CLUSTERS);
        }

#pragma unroll
        for (int i = 0; i < fg::SIGNAL_PIPELINE_STAGES; i++) {
            init_semaphore(signal_full[i], 1);
            init_semaphore(signal_empty[i], 1);
        }

        init_semaphore(tmem_allocated, 1);
        init_semaphore(tmem_finished, fg::NUM_CLUSTERS);
    }

    // flush to ensure the mbarriers are visible
    everyone::tma::cluster::arrive_aligned();

    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int iter_k = 0; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile(&A_smem)[fg::CONSUMER_WARPS] = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(mma_finish[input_stage_id], (phasebits & 0b1));

            tma::cluster::expect_bytes(tma_load[input_stage_id],
                                       sizeof(fg::A_tile) * fg::CONSUMER_WARPS + sizeof(fg::B_tile),
                                       0);
            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {tile_col_idx * fg::NUM_CLUSTERS + cta_rank, iter_k},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);
#pragma unroll
            for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                tma::cluster::load_async(A_smem[c],
                                         G.A,
                                         {tile_row_idx + c, iter_k},
                                         tma_load[input_stage_id],
                                         (uint16_t)(1 << cta_rank),
                                         0);
            }

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 0b1;
            }
        }
    };

    // Built from the published base rather than kept in an array: with two TMEM
    // stages, indexing an array by the runtime stage puts it on the stack.
    auto tmem_slot = [&](int consumer_id, int stage) {
        return tm_alloc.template allocate<typename fg::C_tt_tile>(
            (consumer_id * fg::TMEM_PIPELINE_STAGES + stage) * fg::COL_BLOCK);
    };

    auto consume = [&](int& input_stage_id, int& epilogue_stage_id, const int consumer_id) {
        const int consumer_semaphore_idx = consumer_id + 2;
        // One handoff barrier pair per TMEM slot; the phase bit flips once per lap.
        const int tmem_slot_idx = consumer_id * fg::TMEM_PIPELINE_STAGES + epilogue_stage_id;
        wait(epilogue_tmem_finished[tmem_slot_idx], (phasebits >> consumer_semaphore_idx) & 0b1);
        auto accumulator = tmem_slot(consumer_id, epilogue_stage_id);

        {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            mm_ABt_ncta<fg::NUM_CLUSTERS>(accumulator, A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int iter_k = 1; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            mma_ABt_ncta<fg::NUM_CLUSTERS>(accumulator, A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<fg::NUM_CLUSTERS>(epilogue_ready[tmem_slot_idx]);
        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;

        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << consumer_semaphore_idx);
        }
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        const int consumer_id,
                        int& epilogue_stage_id,
                        int& epilogue_smem_idx,
                        bool is_last_tile,
                        int& signal_stage,
                        int& signal_phase) {
        constexpr int C_CHUNK_COLS = fg::COL_BLOCK / fg::C_TILE_DIVISOR;
        rt_bf<fg::A_tile::rows / WARPGROUP_WARPS, C_CHUNK_COLS> c_reg[fg::C_TILE_DIVISOR];

        const int epilogue_semaphore_idx = consumer_id + 4;

        // This consumer closes a signal region's rows; its chunks close the columns.
        const bool closes_region_rows = (consumer_id + 1) % fg::CONSUMERS_PER_SIGNAL == 0;

        // Drain, then hand the region to the signal warp. The drain is what makes the
        // region visible, so each region adds one.
        auto publish_region = [&](int region_col) {
            dist::tma::store_async_wait();
            // Only the elected thread issued and drained the TMA stores, so it alone
            // hands the region off.
            if (warp_id == 0 && elect_warp_leader()) {
                wait(signal_empty[signal_stage], signal_phase);
                arrive(signal_full[signal_stage]);
            }
            signal_stage = (signal_stage + 1) % fg::SIGNAL_PIPELINE_STAGES;
            if (signal_stage == 0) {
                signal_phase ^= 1;
            }
        };

        const int tmem_slot_idx = consumer_id * fg::TMEM_PIPELINE_STAGES + epilogue_stage_id;
        wait(epilogue_ready[tmem_slot_idx], (phasebits >> epilogue_semaphore_idx) & 0b1);

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            warpgroup::load_async(
                c_reg[i],
                tmem_slot(consumer_id, epilogue_stage_id)
                    .template subtile<tt<float, fg::A_tile::rows, C_CHUNK_COLS>>(i * C_CHUNK_COLS));
        }

        tensor_load_wait();
        if (elect_warp_leader()) {
            tma::cluster::arrive(epilogue_tmem_finished[tmem_slot_idx], 0);
        }

        if (is_last_tile && consumer_id == 0) {
            warpgroup::sync(1);
            pdl::arrive();
        }

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fg::EPILOGUE_PIPELINE_STAGES - 1>();
            warpgroup::sync(1);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[epilogue_smem_idx], c_reg[i]);
            warpgroup::sync(1);

            if (warp_id == 0 && elect_warp_leader()) {
                dist::tma::store_async(G.C[G.dev_idx],
                                       C_smem[epilogue_smem_idx],
                                       {tile_row_idx, tile_col_idx * fg::C_TILE_DIVISOR + i});
            }

            epilogue_smem_idx = (epilogue_smem_idx + 1) % fg::EPILOGUE_PIPELINE_STAGES;

            // Column regions before the last are published mid-loop, as soon as their
            // chunks are stored; the last is published below, once all stores are issued.
            if (closes_region_rows && (i + 1) % fg::C_CHUNKS_PER_SIGNAL == 0 &&
                i + 1 < fg::C_TILE_DIVISOR) {
                publish_region((i + 1) / fg::C_CHUNKS_PER_SIGNAL - 1);
            }
        }
        if (closes_region_rows) {
            publish_region(fg::SIGNAL_COL_SPLITS - 1);
        }

        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (0b1 << epilogue_semaphore_idx);
        }
    };

    if (warpgroup_id == fg::MAINLOOP_WARPGROUP) {
        warpgroup::decrease_registers<fg::MAINLOOP_REGISTERS>();

        if (warp_id == fg::PRODUCER_WARP_ID) {
            pdl::wait();
            everyone::tma::cluster::wait();

            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int i = cluster_idx; i < total_num_tiles; i += num_clusters) {
                    auto [tile_row_idx, tile_col_idx] =
                        calculate_tile_idx<fg::SCHEDULER_SWIZZLE_SIZE>(
                            num_tile_rows_per_cluster, num_tile_cols, i);
                    load(tile_row_idx * fg::NUM_CLUSTERS * fg::CONSUMER_WARPS +
                             cta_rank * fg::CONSUMER_WARPS,
                         tile_col_idx,
                         input_stage_id);
                }
            }

            pdl::arrive();
        } else if (warp_id >= fg::FIRST_CONSUMER_WARP_ID &&
                   warp_id < fg::FIRST_CONSUMER_WARP_ID + fg::CONSUMER_WARPS) {
            pdl::wait();
            everyone::tma::cluster::wait();

            // The two-CTA allocation requires a complete warp in each CTA,
            // before restricting MMA issue to the leader in CTA 0.
            if (warp_id == fg::FIRST_CONSUMER_WARP_ID) {
                tm_alloc.provision(tmem_addr);
                if (elect_warp_leader()) {
                    arrive(tmem_allocated);
                }
            } else {
                wait(tmem_allocated, 0);
            }
            tm_alloc.set_addr(tmem_addr);

            if (cta_rank == 0 && elect_warp_leader()) {
                int input_stage_id = 0;
                int epilogue_stage_id = 0;
                const int consumer_id = warp_id - fg::FIRST_CONSUMER_WARP_ID;

                for (int i = cluster_idx; i < total_num_tiles; i += num_clusters) {
                    consume(input_stage_id, epilogue_stage_id, consumer_id);
                }
            }

            pdl::arrive();
        } else if (warp_id == fg::SIGNAL_WARP_ID) {
            everyone::tma::cluster::wait();

            // Walks the epilogue's tile sequence and publishes each tile once its
            // stores drain. The release fence before the red waits on this SM's
            // outstanding traffic, which is why it is kept off the epilogue.
            if (elect_warp_leader()) {
                int signal_stage = 0;
                int signal_phase = 0;
                for (int tile_id = cluster_idx; tile_id < total_num_tiles;
                     tile_id += num_clusters) {
                    auto [tile_row_idx, tile_col_idx] =
                        calculate_tile_idx<fg::SCHEDULER_SWIZZLE_SIZE>(
                            num_tile_rows_per_cluster, num_tile_cols, tile_id);
#pragma unroll
                    for (int s = 0; s < fg::SIGNALS_PER_TILE; s++) {
                        wait(signal_full[signal_stage], signal_phase);
                        comm::atomic_u32::fence_acq_rel_gpu();

                        const int cta_row_start =
                            (tile_row_idx * fg::NUM_CLUSTERS + cta_rank) * fg::ROW_BLOCK;
                        const int device_to_signal = cta_row_start / rows_per_device;
                        const int local_tile_row =
                            (cta_row_start % rows_per_device) / fg::ROW_BLOCK;

                        comm::atomic_u32::relaxed_add_gpu(
                            G.C_barrier[device_to_signal].raw_ptr +
                                fg::ready_idx(local_tile_row, tile_col_idx, num_tile_cols, s),
                            1);
                        arrive(signal_empty[signal_stage]);

                        signal_stage = (signal_stage + 1) % fg::SIGNAL_PIPELINE_STAGES;
                        if (signal_stage == 0) {
                            signal_phase ^= 1;
                        }
                    }
                }
            }
        }
    } else if (warpgroup_id == fg::RS_WARPGROUP) {
        warpgroup::decrease_registers<fg::RS_REGISTERS>();
        const int lane = laneid();
        const int tile_rows_per_device = rows_per_device / fg::ROW_BLOCK;
        const int tiles_per_device = tile_rows_per_device * num_tile_cols;

        const int device_row_start = G.dev_idx * rows_per_device;

        pdl::wait();
        everyone::tma::cluster::wait();

        constexpr uint32_t ready_count = uint32_t(fg::NUM_DEVICES);

        const int rs_warp = warp_id - fg::FIRST_RS_WARP_ID;
        for (int unit_idx = blockIdx.x + rs_warp * fg::NUM_BLOCKS;
             unit_idx < tiles_per_device * fg::RS_STRIPS_PER_TILE;
             unit_idx += fg::NUM_BLOCKS * fg::REDUCE_SCATTER_WARPS) {
            // split the tile into (tile, strip)
            const int strip_idx = unit_idx % fg::RS_STRIPS_PER_TILE;

            // split the tile into (tile, region, strip_row, strip_col)
            const int region =
                strip_idx / fg::RS_STRIPS_PER_SIGNAL;  // each region is a RS::ROW x RS::COL chunk
            const int strip_row = (region / fg::SIGNAL_COL_SPLITS) * fg::SIGNAL_ROWS +
                (strip_idx % fg::RS_STRIPS_PER_SIGNAL) * fg::RS_M_STRIP_SIZE;
            const int strip_col = (region % fg::SIGNAL_COL_SPLITS) * fg::SIGNAL_COLS;

            // Walk this device's tiles in the GEMM's rasterization
            const auto [local_tile_row, tile_col] = calculate_tile_idx<fg::SCHEDULER_SWIZZLE_SIZE>(
                tile_rows_per_device, num_tile_cols, unit_idx / fg::RS_STRIPS_PER_TILE);

            // These match the coordinates used by the epilogue's signal.
            if (elect_warp_leader()) {
                // Seeing the region's counter complete also proves all of its adds
                // have landed, which the exit barrier's reset relies on.
                const auto* ready_counter = G.C_barrier[G.dev_idx].raw_ptr +
                    fg::ready_idx(local_tile_row, tile_col, num_tile_cols, region);
                while (comm::atomic_u32::acquire_load_gpu(ready_counter) < ready_count) {
                    __nanosleep(16);
                }
            }
            __syncwarp();

            const int tile_row_start =
                device_row_start + local_tile_row * fg::ROW_BLOCK + strip_row;
            const int tile_col_start = tile_col * fg::COL_BLOCK + strip_col;
            reduce_scatter_tile<fg::RS_M_STRIP_SIZE, fg::RS_COLS, fg::RS_STORES_PER_BATCH>(
                G.C.mc_ptr_at({tile_row_start, tile_col_start}),
                &G.C[G.dev_idx][{tile_row_start, tile_col_start}],
                G.C[G.dev_idx].cols(),
                lane);
        }

        pdl::arrive();
    } else {
        warpgroup::increase_registers<fg::EPILOGUE_REGISTERS>();

        int epilogue_stage_id[fg::CONSUMER_WARPS] = {};
        int epilogue_smem_idx = 0;
        int signal_stage = 0;
        // Fresh mbarriers complete the phase before parity 0, so the first lap
        // of signal_empty passes without waiting.
        int signal_phase = 1;

        everyone::tma::cluster::wait();
        wait(tmem_allocated, 0);
        // Tile views and the eventual deallocation use the published base.
        tm_alloc.set_addr(tmem_addr);

        for (int tile_id = cluster_idx; tile_id < total_num_tiles; tile_id += num_clusters) {
            auto [tile_row_idx, tile_col_idx] = calculate_tile_idx<fg::SCHEDULER_SWIZZLE_SIZE>(
                num_tile_rows_per_cluster, num_tile_cols, tile_id);

            // The signal warp derives the same row from tile_id; keep the two in step.
            const int cluster_level_tile_id = tile_row_idx * fg::NUM_CLUSTERS * fg::CONSUMER_WARPS +
                cta_rank * fg::CONSUMER_WARPS;
#pragma unroll
            for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                const int warp_level_tile_id = cluster_level_tile_id + c;
                epilogue(warp_level_tile_id,
                         tile_col_idx,
                         c,
                         epilogue_stage_id[c],
                         epilogue_smem_idx,
                         tile_id + num_clusters >= total_num_tiles,
                         signal_stage,
                         signal_phase);
            }
        }

        tensor_before_thread_sync();
        group<fg::EPILOGUE_WARPS>::sync(1);

        if (group<fg::EPILOGUE_WARPS>::warpid() == 0) {
            if (elect_warp_leader()) {
#pragma unroll
                for (int peer = 0; peer < fg::NUM_CLUSTERS; peer++) {
                    tma::cluster::arrive(tmem_finished, peer);
                }
            }
            // Only reach here if we finish with our tmem. Other party as well
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
__global__ __cluster_dims__(
    fused_globals<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>::
        NUM_CLUSTERS,
    1,
    1)
    __launch_bounds__(
        fused_globals<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>::
            NUM_THREADS,
        1) void fused_kernel_stub(const __grid_constant__ fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                                                                        _SIGNALS_PER_TILE,
                                                                        _RS_STRIP_ROWS,
                                                                        _CONSUMER_WARPS> G) {
    gemm_rs_blackwell<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>(
        G);
}

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
inline void launch_gemm_rs_blackwell(const fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                                                         _SIGNALS_PER_TILE,
                                                         _RS_STRIP_ROWS,
                                                         _CONSUMER_WARPS>& G) {
    cudaStream_t stream = G.stream;

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!G.stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
#endif

    using fg =
        fused_globals<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>;
    constexpr int smem_size = fg::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = fg::NUM_THREADS;
    constexpr int grid = fg::NUM_BLOCKS;

    auto this_kernel = fused_kernel_stub<_SCHEDULER_SWIZZLE_SIZE,
                                         _SIGNALS_PER_TILE,
                                         _RS_STRIP_ROWS,
                                         _CONSUMER_WARPS>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchAttribute pdl_attr = {};
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = num_threads;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;
    launch_config.attrs = &pdl_attr;
    launch_config.numAttrs = 1;

    MKERNEL_CUDACHECK(cudaLaunchKernelEx(&launch_config, this_kernel, G));
    MKERNEL_CUDACHECK(cudaGetLastError());

    // Full completion dependency, including in captured graphs. The next use
    // of these buffers on this stream cannot pass until every rank is done.
    // Clear only the prefix of the ready grid this launch's M and N used.
    exit_epoch_barrier<<<1, EXIT_BARRIER_THREADS, 0, stream>>>(
        G.C_barrier[G.dev_idx].raw_ptr, fg::ready_numel(G.M, G.N), G.completion_epochs, G.dev_idx);
    MKERNEL_CUDACHECK(cudaGetLastError());
}
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
inline void prepare(at::Tensor& A,
                    at::Tensor& B,
                    dist::ParallelBuffer& C,
                    dist::ParallelBuffer& sync) {
    // Shape checks on A, B and C run at launch, when the config is known.
    TORCH_CHECK(
        sync.data_.element_size() == sizeof(uint32_t) && sync.data_.numel() == SYNC_BUFFER_WORDS,
        "sync must be an int32 buffer of shape gemm_rs_blackwell_sync_shape");
    MKERNEL_CUDACHECK(cudaMemsetAsync(
        sync.data_.data_ptr(), 0, sync.data_.nbytes(), at::cuda::getCurrentCUDAStream().stream()));
}

// Picks the tuned <swizzle, signals_per_tile, rs_strip_rows, mma_warps> for M.
// Keep in sync with BlackwellBenchConfig.mkernel_candidates in gemm_rs_bench.py.
inline void entrypoint(at::Tensor& A,
                       at::Tensor& B,
                       dist::ParallelBuffer& C,
                       dist::ParallelBuffer& sync,
                       cudaStream_t stream = nullptr) {
    const int M = A.size(0);

    if (!stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    if (M <= 2048) {
        auto globals = gemm_rs_blackwell_make_globals<2, 2, 32, 1>(
            A, B, C, sync, C.local_rank_, A.size(0), B.size(0), A.size(1), stream);
        launch_gemm_rs_blackwell(globals);
    } else if (M <= 4096) {
        auto globals = gemm_rs_blackwell_make_globals<2, 2, 32, 1>(
            A, B, C, sync, C.local_rank_, A.size(0), B.size(0), A.size(1), stream);
        launch_gemm_rs_blackwell(globals);
    } else if (M <= 8192) {
        auto globals = gemm_rs_blackwell_make_globals<1, 2, 16, 1>(
            A, B, C, sync, C.local_rank_, A.size(0), B.size(0), A.size(1), stream);
        launch_gemm_rs_blackwell(globals);
    } else if (M <= 16384) {
        auto globals = gemm_rs_blackwell_make_globals<1, 2, 16, 1>(
            A, B, C, sync, C.local_rank_, A.size(0), B.size(0), A.size(1), stream);
        launch_gemm_rs_blackwell(globals);
    } else if (M <= 32768) {
        auto globals = gemm_rs_blackwell_make_globals<2, 2, 16, 1>(
            A, B, C, sync, C.local_rank_, A.size(0), B.size(0), A.size(1), stream);
        launch_gemm_rs_blackwell(globals);
    } else {
        TORCH_CHECK(false, "gemm_rs_blackwell is not tuned for M > 32768, got M=", M);
    }
}

#endif

template void launch_gemm_rs_blackwell<DEFAULT_SCHEDULER_SWIZZLE_SIZE,
                                       DEFAULT_SIGNALS_PER_TILE,
                                       DEFAULT_RS_STRIP_ROWS,
                                       DEFAULT_NUM_CONSUMER_WARPS>(
    const fused_globals<DEFAULT_SCHEDULER_SWIZZLE_SIZE,
                        DEFAULT_SIGNALS_PER_TILE,
                        DEFAULT_RS_STRIP_ROWS,
                        DEFAULT_NUM_CONSUMER_WARPS>& G);
};  // namespace gemm_rs_blackwell

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include "operators/gemm_rs/gemm_rs_blackwell_session.cuh"
#endif
