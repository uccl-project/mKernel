#pragma once

#include <cassert>
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/parallel_buffer.cuh"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"

namespace gemm_rs_blackwell {
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = 2;
static constexpr int DEFAULT_SCHEDULER_SWIZZLE_SIZE = 4;
static constexpr int DEFAULT_SIGNALS_PER_TILE = 1;
static constexpr int DEFAULT_RS_STRIP_ROWS = 32;

// Each rank's `sync` buffer (int32, no multicast) holds its completion epoch in
// word 0, padded to one 32B L2 sector, followed by the ready-counter grid.
static constexpr int SYNC_HEADER_BYTES = 32;
static constexpr int SYNC_HEADER_WORDS = SYNC_HEADER_BYTES / sizeof(uint32_t);
// The buffer has a fixed size of one 2MB VMM granule, which every allocation is
// rounded up to anyway. It holds ~524K ready counters, so any practical M and N fit.
static constexpr int SYNC_BUFFER_BYTES = 2 << 20;
static constexpr int SYNC_BUFFER_WORDS = SYNC_BUFFER_BYTES / sizeof(uint32_t);

// Only the owning rank writes its completed epoch. Peers acquire-load it.
using completion_epoch_local_tensor = dist::local_tensor<uint32_t, 1, 1, 1, 1>;
using completion_epoch_dist_tensor =
    dist::distributed_tensor<completion_epoch_local_tensor, INTRA_NUM_DEVICES, false>;

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
struct fused_globals;

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
void launch_gemm_rs_blackwell(const fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                                                  _SIGNALS_PER_TILE,
                                                  _RS_STRIP_ROWS,
                                                  _CONSUMER_WARPS>& G);

// Call once before capture/replay, with no work using these buffers in flight.
// Wait for this stream's initialization to finish locally, then synchronize all
// ranks before the first launch. Serialize launches on G.stream and keep the sync
// buffer alive across all graph replays. M and N may change between launches:
// each launch's exit barrier re-zeroes the ready counters it used.
template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
void initialize_gemm_rs_blackwell_state(const fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                                                            _SIGNALS_PER_TILE,
                                                            _RS_STRIP_ROWS,
                                                            _CONSUMER_WARPS>& G);

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS>
struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int NUM_BLOCKS = 148;
    // MMA warps per CTA, each owning a 128-row slice of the CTA's tile. One warp
    // halves the tile height (256x256 per cluster instead of 512x256).
    static constexpr int CONSUMER_WARPS = _CONSUMER_WARPS;
    static_assert(CONSUMER_WARPS == 1 || CONSUMER_WARPS == 2, "CONSUMER_WARPS must be 1 or 2");
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = 1;
    static constexpr int REDUCE_SCATTER_WARPS = 4;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;

    // Warp layout: [epilogue WG(s)] [producer, MMA warp(s), signal] [RS WG]. setmaxnreg
    // is .sync.aligned over a whole warpgroup, so the producer/MMA tail is filled
    // out by the signal warp (and an idle warp with one MMA warp) and the RS warps
    // get a warpgroup of their own.
    static constexpr int PRODUCER_WARP_ID = EPILOGUE_WARPS;
    static constexpr int FIRST_CONSUMER_WARP_ID = PRODUCER_WARP_ID + PRODUCER_WARPS;
    static constexpr int MAINLOOP_WARPGROUP = EPILOGUE_WARPGROUPS;
    static constexpr int RS_WARPGROUP = MAINLOOP_WARPGROUP + 1;
    static constexpr int FIRST_RS_WARP_ID = RS_WARPGROUP * kittens::WARPGROUP_WARPS;
    // The warp that pads the producer/MMA warpgroup publishes tile arrivals, so
    // the epilogue never stalls on the release fence of the signal. It stays the
    // warpgroup's last warp for any MMA warp count.
    static constexpr int SIGNAL_WARP_ID = FIRST_RS_WARP_ID - 1;
    static_assert(FIRST_CONSUMER_WARP_ID + CONSUMER_WARPS <= SIGNAL_WARP_ID,
                  "Producer/MMA/signal warps must fit in one warpgroup");
    static_assert(REDUCE_SCATTER_WARPS == kittens::WARPGROUP_WARPS,
                  "The RS warps must form exactly one warpgroup");

    static constexpr int NUM_WARPS = FIRST_RS_WARP_ID + REDUCE_SCATTER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

    static constexpr int MAINLOOP_REGISTERS = 40;
    static constexpr int RS_REGISTERS = 160;
    static constexpr int EPILOGUE_REGISTERS = 224;
    // guard against register overallocation
    static_assert((MAINLOOP_REGISTERS + RS_REGISTERS) * kittens::WARPGROUP_THREADS +
                          EPILOGUE_REGISTERS * EPILOGUE_WARPS * kittens::WARP_THREADS <=
                      65536,
                  "setmaxnreg budget exceeds the register file");
    // setmaxnreg.dec can only lower a warp's count from the launch allocation.
    static constexpr int REGISTER_CEILING = (65536 / NUM_THREADS / 8) * 8;
    static_assert(MAINLOOP_REGISTERS <= REGISTER_CEILING && RS_REGISTERS <= REGISTER_CEILING,
                  "Decreased register counts must not exceed the launch allocation");
    static constexpr int NUM_CLUSTERS = DEFAULT_NUM_CTA;

    // A 2-CTA MMA covers 128 rows per CTA.
    static constexpr int ROW_BLOCK = 128 * CONSUMER_WARPS;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int C_TILE_DIVISOR = 4;

    // Publications per tile, in epilogue store order: 1 is the whole tile, 2 is each
    // consumer's half (column halves with one consumer), and 4 also splits each half
    // by columns, published midway through its stores. Each covers SIGNAL_ROWS x
    // SIGNAL_COLS. One consumer with 4 signals (64-column regions) is not enabled.
    static constexpr int SIGNALS_PER_TILE = _SIGNALS_PER_TILE;
    static_assert(SIGNALS_PER_TILE == 1 || SIGNALS_PER_TILE == 2 || SIGNALS_PER_TILE == 4,
                  "SIGNALS_PER_TILE must be 1, 2, or 4");

    static constexpr int SIGNAL_ROW_SPLITS =
        SIGNALS_PER_TILE < CONSUMER_WARPS ? SIGNALS_PER_TILE : CONSUMER_WARPS;

    static constexpr int SIGNAL_COL_SPLITS = SIGNALS_PER_TILE / SIGNAL_ROW_SPLITS;
    static constexpr int SIGNAL_ROWS = ROW_BLOCK / SIGNAL_ROW_SPLITS;
    static constexpr int SIGNAL_COLS = COL_BLOCK / SIGNAL_COL_SPLITS;
    static constexpr int CONSUMERS_PER_SIGNAL = CONSUMER_WARPS / SIGNAL_ROW_SPLITS;
    static constexpr int C_CHUNKS_PER_SIGNAL = C_TILE_DIVISOR / SIGNAL_COL_SPLITS;
    static_assert(C_TILE_DIVISOR % SIGNAL_COL_SPLITS == 0,
                  "A signal region must end on a C chunk boundary");

    __host__ __device__ static constexpr int ready_numel(int M, int N) {
        return (M / NUM_DEVICES / ROW_BLOCK) * (N / COL_BLOCK) * SIGNALS_PER_TILE;
    }

    //  [tile_row, tile_col, region] indexing for the ready buffer
    __host__ __device__ static constexpr int ready_idx(int local_tile_row,
                                                       int tile_col,
                                                       int num_tile_cols,
                                                       int region) {
        return (local_tile_row * num_tile_cols + tile_col) * SIGNALS_PER_TILE + region;
    }

    // RS units are RS_M_STRIP_SIZE x RS_COLS strips inside one signal region, so each
    // waits on its region's ready counter.
    static constexpr int RS_M_STRIP_SIZE = _RS_STRIP_ROWS;
    static constexpr int RS_COLS = SIGNAL_COLS;
    static_assert(SIGNAL_ROWS % RS_M_STRIP_SIZE == 0,
                  "An RS strip must not straddle two signal regions");
    // Loads in flight per lane: a whole strip, capped at 32, so a batch never does a larger than
    // 512B load, which would exceed the width of the tensor
    static constexpr int RS_ROWS_PER_STORE = kittens::WARP_THREADS / (RS_COLS / 8);
    static constexpr int RS_STORES_PER_BATCH =
        RS_M_STRIP_SIZE / RS_ROWS_PER_STORE < 32 ? RS_M_STRIP_SIZE / RS_ROWS_PER_STORE : 32;
    static constexpr int RS_STRIPS_PER_SIGNAL = SIGNAL_ROWS / RS_M_STRIP_SIZE;
    // Partition a single tile into strips as well, so we can have more CTAs active per
    // completed RS tile
    static constexpr int RS_STRIPS_PER_TILE = RS_STRIPS_PER_SIGNAL * SIGNALS_PER_TILE;

    static constexpr int TMEM_PIPELINE_STAGES =
        kittens::MAX_TENSOR_COLS / COL_BLOCK / CONSUMER_WARPS;
    static constexpr int NUM_TMEM_SLOTS = TMEM_PIPELINE_STAGES * CONSUMER_WARPS;

    // used for SMEM allocation
    // One MMA warp halves each stage's A bytes, so two more stages fit in SMEM.
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = CONSUMER_WARPS == 1 ? 6 : 4;
    static constexpr int EPILOGUE_PIPELINE_STAGES = 2;
    // Drained tiles the epilogue may hand off before the signal warp catches up.
    static constexpr int SIGNAL_PIPELINE_STAGES = 4;

    // Tile columns per supergroup in the tile rasterization.
    static constexpr int SCHEDULER_SWIZZLE_SIZE = _SCHEDULER_SWIZZLE_SIZE;
    static_assert(SCHEDULER_SWIZZLE_SIZE > 0, "SCHEDULER_SWIZZLE_SIZE must be greater than 0");

    using A_tile = kittens::st_bf<ROW_BLOCK / CONSUMER_WARPS, RED_BLOCK>;  // [M, K]
    using B_tile = kittens::st_bf<COL_BLOCK / NUM_CLUSTERS, RED_BLOCK>;    // [N, K]

    using C_tt_tile = kittens::tt<float, ROW_BLOCK / CONSUMER_WARPS, COL_BLOCK>;  // [M, N]
    using C_tile = kittens::st_bf<ROW_BLOCK / CONSUMER_WARPS, COL_BLOCK / C_TILE_DIVISOR>;

    static constexpr int MAX_DYNAMIC_SHARED_MEMORY = 227 * 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (sizeof(A_tile) * CONSUMER_WARPS + sizeof(B_tile)) * PRODUCER_CONSUMER_PIPELINE_STAGES +
        sizeof(C_tile) * EPILOGUE_PIPELINE_STAGES + 1024;
    static constexpr bool SMEM_FITS = DYNAMIC_SHARED_MEMORY <= MAX_DYNAMIC_SHARED_MEMORY;
    static_assert(SMEM_FITS, "Pipeline stages exceed dynamic shared memory");

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, C_tile>;
    using C_dist_tensor = dist::distributed_tensor<C_local_tensor, NUM_DEVICES, true>;

    // Peers bump counters through unicast pointers, so no multicast VA is needed.
    using C_ready_barrier =
        dist::distributed_tensor<dist::local_tensor<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

    int dev_idx;
    int M;
    int N;
    int K;

    A_local_tensor A;
    B_local_tensor B;
    C_dist_tensor C;

    C_ready_barrier C_barrier;
    completion_epoch_dist_tensor completion_epochs;

    cudaStream_t stream = nullptr;

    struct pipeline_inputs {
        A_tile A[CONSUMER_WARPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    // bit 0: TMA producer
    // bit 1: TMA consumer
    // bit 2-3: TMEM producer
    // bit 4-5: TMEM consumer
    // bit 6: TMEM allocate
    // bit 7: TMEM deallocate
    static constexpr int PHASE_BITS_INIT = 0b00001101;
};

template <typename>
inline constexpr bool always_false_v = false;

template <int _SCHEDULER_SWIZZLE_SIZE,
          int _SIGNALS_PER_TILE,
          int _RS_STRIP_ROWS,
          int _CONSUMER_WARPS,
          typename DistributedTensor,
          typename LocalTensor>
__host__ inline fused_globals<_SCHEDULER_SWIZZLE_SIZE,
                              _SIGNALS_PER_TILE,
                              _RS_STRIP_ROWS,
                              _CONSUMER_WARPS>
gemm_rs_blackwell_make_globals(LocalTensor& A,
                               LocalTensor& B,
                               DistributedTensor& C,
                               DistributedTensor& sync,
                               int dev_idx,
                               int M,
                               int N,
                               int K,
                               cudaStream_t stream) {
    using fg =
        fused_globals<_SCHEDULER_SWIZZLE_SIZE, _SIGNALS_PER_TILE, _RS_STRIP_ROWS, _CONSUMER_WARPS>;
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if constexpr (std::is_same_v<LocalTensor, at::Tensor> &&
                  std::is_same_v<DistributedTensor, dist::ParallelBuffer>) {
        // A CTA tile must not straddle two ranks' rows: the signal warp sends a whole
        // region to one rank, and the RS warps enumerate whole tiles per rank.
        TORCH_CHECK((M / fg::NUM_DEVICES) % fg::ROW_BLOCK == 0 && N % fg::COL_BLOCK == 0,
                    "M / world_size must be a multiple of ",
                    fg::ROW_BLOCK,
                    " and N a multiple of ",
                    fg::COL_BLOCK);
        const int ready_numel = fg::ready_numel(M, N);
        TORCH_CHECK(sync.data_.element_size() == sizeof(uint32_t) &&
                        sync.data_.numel() == SYNC_BUFFER_WORDS,
                    "sync must be an int32 buffer of shape gemm_rs_blackwell_sync_shape");
        TORCH_CHECK(SYNC_HEADER_WORDS + ready_numel <= SYNC_BUFFER_WORDS,
                    "M=",
                    M,
                    ", N=",
                    N,
                    " needs more ready counters than the sync buffer holds");
        TORCH_CHECK(int(sync.raw_ptrs_.size()) == fg::NUM_DEVICES,
                    "sync must span ",
                    fg::NUM_DEVICES,
                    " ranks");
        // Slice each rank's sync buffer into the epoch word and the ready grid.
        uint64_t epoch_ptrs[fg::NUM_DEVICES];
        uint64_t ready_ptrs[fg::NUM_DEVICES];
        for (int i = 0; i < fg::NUM_DEVICES; i++) {
            epoch_ptrs[i] = reinterpret_cast<uint64_t>(sync.raw_ptrs_[i]);
            ready_ptrs[i] = epoch_ptrs[i] + SYNC_HEADER_BYTES;
        }
        return {.dev_idx = dev_idx,
                .M = M,
                .N = N,
                .K = K,
                .A = dist::local_tensor_from_tensor<typename fg::A_local_tensor>(A),
                .B = dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
                .C = dist::distributed_tensor_from_buffer<typename fg::C_dist_tensor>(C),
                .C_barrier = dist::make_distributed_tensor<typename fg::C_ready_barrier>(
                    ready_ptrs, 1, 1, 1, ready_numel),
                .completion_epochs = dist::make_distributed_tensor<completion_epoch_dist_tensor>(
                    epoch_ptrs, 1, 1, 1, 1),
                .stream = stream};
    } else {
        static_assert(always_false_v<LocalTensor>, "Unsupported LocalTensorType");
    }
#else
    static_assert(always_false_v<LocalTensor>, "Construct fused_globals directly without Torch");
#endif
}

namespace {
__device__ __forceinline__ void reduce_load_16B_no_clobber(uint32_t* out, const comm::bf16* src) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "l"(src));
}

__device__ __forceinline__ void store_16B(comm::bf16* dst, const uint32_t* in) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]));
}
}  // namespace
};  // namespace gemm_rs_blackwell
