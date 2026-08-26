#pragma once

/**
 * @file gemm_rs.cuh
 * @brief Multi-node GEMM + Reduce-Scatter.
 *
 * Uses the intra-node M-GPU pattern (output_distributed_tensor +
 * tma::store_add_async + per-task ready flags + atomic task claiming) plus
 * an inter-node phase that exchanges each GPU's owned rows with the
 * same-index GPU on every remote node.
 *
 * Let M = INTRA_NUM_DEVICES (GPUs per node) and N = num_nodes. Data flow
 * per GPU (g ∈ [0, M) within node n ∈ [0, N)):
 *   Initial: A[g] (M_rows, K), B[g] (K, N_cols)   ← partial inputs.
 *   Phase A: local workspace = A[g] @ B[g]
 *            tma::store_add_async to output_distributed_tensor[owner_in_node].
 *            After A: output[g] holds M_rows/M rows summed over M local GPUs.
 *   Sync 1:  intra-node barrier_all (all M local GPUs done with Phase A).
 *   Phase B: copy output[g] → staging_buf, push RDMA to GPU g on every
 *            remote node. Peers land in recv_buf at per-sender offsets.
 *            Set sender_done[tile].
 *   Phase C: poll arrival_flags + sender_done, add each peer slot of
 *            recv_buf to output[g] in-place. Final: output[g] holds
 *            M_rows/M rows summed over all N*M GPUs.
 *
 * Output shape per GPU: (M_rows/M, N_cols) — a true N*M-way reduce-scatter
 * partitioned 1/M per node.
 *
 * Hot path is a single CTA-specialized kernel:
 *   compute CTAs -> intra-node RS CTAs -> inter-node send CTAs -> inter-node reduce CTAs
 * with fine-grained per-tile handoff between each phase.
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "common/cuda_checks.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "comm/comm.cuh"
#include "comm/atomic_u32.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <assert.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace gemm_rs_multinode {

// ============================================================================
// Config
// ============================================================================

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 132;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

// ============================================================================
// Acquire/release memory helpers (match gemm_ar's gemm_ar_acquire_load_u32 /
// gemm_ar_release_store_u32 but namespaced for gemm_rs).
// ============================================================================

// PTX wrappers come from include/comm/atomic_u32.cuh (shared with gemm_ar
// and any future kernel). Kept as gemm_rs_-prefixed inline aliases so existing
// call sites read unchanged.
template <typename PtrT>
__device__ inline uint32_t gemm_rs_acquire_load_u32(PtrT* ptr) {
    return comm::atomic_u32::acquire_load_gpu(ptr);
}
template <typename PtrT>
__device__ inline void gemm_rs_release_store_u32(PtrT* ptr, uint32_t val) {
    comm::atomic_u32::release_store_gpu(ptr, val);
}

// GPU-scope arrival-flag polling primitives used by the reducer CTA.
// - acquire.gpu — best when a single chunk is the critical-path gate.
// - relaxed.gpu — better throughput when many chunks polled in parallel.
// The caller chooses at runtime via Rt.use_acquire_poll.
__device__ __forceinline__ uint32_t gemm_rs_poll_arrival_acquire(volatile uint32_t* p) {
    return comm::atomic_u32::acquire_load_gpu(const_cast<uint32_t*>(p));
}
__device__ __forceinline__ uint32_t gemm_rs_poll_arrival_relaxed(volatile uint32_t* p) {
    return comm::atomic_u32::relaxed_load_gpu(const_cast<uint32_t*>(p));
}

// ============================================================================
// Kernel 1: intra-node compute + 8-GPU reduce-scatter via dbuf atomic-add
// (Direct port of dynamic_sm_allocation/question5_gemm_rs/gemm_rs.cu)
// ============================================================================

struct intra_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using A_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, B_tile>;
    // Workspace is a multicast dbuf under GEMM_RS_MULTIMEM_RS: each GPU has its own
    // replica, and multimem.ld_reduce via workspace.mc_ptr_at pulls the M-way
    // sum in one HW op (matching gemm_ar's gemm_ar_pipelined_rs_tile). The
    // compute/intra-RS TMA paths index workspace[dev_idx] for the local gl,
    // which doesn't need a multicast binding — so when GEMM_RS_MULTIMEM_RS is off
    // we gate the multicast flag to match the Python-side DistBuffer.
    // This is required at M=65536 where the 8 GiB workspace exceeds the
    // cuMulticastBindMem single-binding granularity (CUDA_ERROR_ILLEGAL_STATE).
    using workspace_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;
    using output_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<NUM_DEVICES>;
    // GEMM_RS_READY_VIA_MULTIMEM: per-chunk "this GPU's compute done" flag. Each
    // peer writes `epoch` to its own replica on the last tile of a chunk.
    // Owner polls multimem.ld_reduce.min across all M replicas — returns
    // `epoch` iff every peer has stamped this chunk. Replaces the M-way dbuf
    // barrier (M sys atomics per chunk) with M local stores + 1 HW op.
    using ready_chunk_distributed_tensor = dist::distributed_tensor<dist::local_tensor<int, 1, 1, -1, -1>, NUM_DEVICES, true>;

    A_local_tensor A;
    B_local_tensor B;
    workspace_distributed_tensor workspace;
    output_distributed_tensor output;
    // Cross-GPU TMA atomic-add target for intra-RS. Python allocates this
    // as (m_local, n); the host entrypoint reinterprets it as chunk-major
    // tiles so C_tile stores land at the inter-node send layout.
    output_distributed_tensor staging;
    barrier_distributed_tensor barrier;
    int *ready;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
    unsigned int *next_compute;
    unsigned int *next_comm;
    unsigned int *kernel_done;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

struct fused_globals {
    static constexpr int NUM_DEVICES = intra_globals::NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = intra_globals::PIPELINE_STAGES;
    static constexpr int SUPER_M = intra_globals::SUPER_M;
    static constexpr int ROW_BLOCK = intra_globals::ROW_BLOCK;
    static constexpr int COL_BLOCK = intra_globals::COL_BLOCK;
    static constexpr int RED_BLOCK = intra_globals::RED_BLOCK;
    // Inter-node send/reduce granularity: N column-tiles per chunk.
    // RDMA sends are posted per ready chunk.
    static constexpr int CHUNK_TILES = 4;

    using A_tile = intra_globals::A_tile;
    using B_tile = intra_globals::B_tile;
    using C_tile = intra_globals::C_tile;

    using A_local_tensor = intra_globals::A_local_tensor;
    using B_local_tensor = intra_globals::B_local_tensor;
    using workspace_distributed_tensor = intra_globals::workspace_distributed_tensor;
    using output_distributed_tensor = intra_globals::output_distributed_tensor;
    using barrier_distributed_tensor = intra_globals::barrier_distributed_tensor;

    struct runtime_state {
        int *inter_ready;
        bf16 *output_local;
        bf16 *recv_buf;
        bf16 *staging_buf;
        int M_local;
        int N;
        int node_idx;
        int num_nodes;  // total node count (>= 2); recv_buf/staging are per-peer.
        unsigned int *sender_done;
        internode::D2HFifoDeviceBundle d2h_fifos;
        volatile uint32_t *arrival_flags;
        uint32_t epoch;
        int total_tiles;
        unsigned int *next_send;       // work-stealing cursor for per-chunk send
        unsigned int *next_reduce;     // work-stealing cursor for per-chunk reduce
        unsigned int *chunks_processed; // incremented per completed chunk; reducer
                                        // pollers exit when it reaches total_chunks.
        unsigned int *remote_arrived_chunks;
        uint32_t *chunk_tiles_done;    // per-chunk atomic counter: how many tiles finished intra-RS

        // Chunk barrier: per-chunk counter for batching dbuf signals (matches gemm_ar)
        uint32_t *comp_chunk_tiles_done;   // size: row_blocks * chunks_per_row (global row indexing)
        // GEMM_RS_MULTIMEM_RS: compute-side chunk-done counter (compute CTAs
        // increment this; the last to hit tiles_this_chunk signals the dbuf
        // barrier so the owner's intra-RS can begin its multimem pull).
        uint32_t *comp_gemm_done;
        // GEMM_RS_MULTIMEM_RS: per-chunk "ready to send" flag set by the owner's
        // intra-RS after multimem.ld_reduce + local store for all tiles of
        // the chunk. SFR sender polls this (instead of the dbuf barrier) to
        // avoid aliasing the compute-side M-way barrier signal.
        uint32_t *chunk_sendable;
        // Per-chunk claim bitmap. A CTA must win atomicCAS(0->1) before
        // posting a chunk, so each chunk is sent once.
        uint32_t *chunk_send_claimed;
        // GEMM_RS_READY_VIA_MULTIMEM: multicast u32 per-chunk "my compute done"
        // flag (per GPU stamps `epoch` on last-tile-in-chunk). Non-nullptr
        // when the flag path is compiled in.
        intra_globals::ready_chunk_distributed_tensor *ready_chunk;
        // Arrival queue: per-chunk flag published by reducer after RDMA arrival detected
        uint32_t *remote_arrived_flag;     // 0=pending, 1=remote arrived, 2=ready-queue claimed
        uint32_t *remote_arrived_peer_mask;
        uint32_t *arrival_queue_head;
        uint32_t *ready_reduce_queue;      // optional queue of chunks ready to reduce (+1 encoded)
        uint32_t *ready_reduce_head;
        uint32_t *ready_reduce_tail;
        uint32_t *ready_reduce_scan;
        uint32_t *peer_accum_queue;        // encoded (peer_slot, chunk_id) work items
        uint32_t *peer_accum_head;
        uint32_t *peer_accum_tail;
        uint32_t *peer_accum_done_count;   // per chunk completed peer accumulations
        uint32_t *chunk_reduce_done;        // per chunk final accumulator ready
        uint32_t *chunk_accum_lock;         // simple per-chunk spin lock for staging_buf
        // Cached layout values for device-side use
        int chunks_per_row;
        int chunk_tiles_val;    // min(CHUNK_TILES, col_blocks)
        int col_blocks_val;
        int row_blocks_per_slice;  // M_local / ROW_BLOCK
        int num_remote_queues;
        int remote_queue_stride;
        int num_recv_progress_sms;
        int owner_chunks_total;
        // Arrival poll tuning (T2a/T2b). Selected host-side based on shape:
        //   use_acquire_poll=1 -> ld.acquire.sys  (lower latency at small/mid M)
        //   use_acquire_poll=0 -> ld.relaxed.sys  (better throughput at large M)
        // reduce_poll_sleep_ns: backoff between arrival polls. At M>=16K the RDMA
        // round-trip is many ms; 1000ns reduces cache thrash without visibly
        // adding wake latency. Default 100ns preserves prior behavior.
        uint8_t use_acquire_poll;
        // Runtime gate for writing intra-RS outputs directly into staging.
        // When false, sender CTAs pack from output_local into staging.
        uint8_t use_intra_rs_dual_write;
        uint8_t use_ready_reduce_queue;
        uint8_t use_transport_arrival_queue;
        uint8_t use_incremental_peer_reduce;
        uint8_t use_receiver_owner_rs;
        uint8_t _pad1[3];
        uint32_t reduce_poll_sleep_ns;
    };

    intra_globals intra;
    int num_send_sms;
    int num_reduce_sms;
    runtime_state *rt;
};

// ============================================================================
// Tile visit order decode functions (matches gemm_ar gemm_ar_multinode_common.cuh).
// Four variants selectable via compile-time #define.
// ============================================================================

// Default: all tiles of slice 0, then slice 1, etc. Within each slice: row-major.
__device__ inline void gemm_rs_slice_row_major_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int& row_idx, int& col_idx
) {
    const int tiles_per_slice = row_blocks_per_slice * col_blocks;
    const int slice_idx = task_id / tiles_per_slice;
    const int local_idx = task_id - slice_idx * tiles_per_slice;
    const int rb_in_slice = local_idx / col_blocks;
    col_idx = local_idx - rb_in_slice * col_blocks;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Round-robin across GPU slices by row-block index. Balances barrier progress.
__device__ inline void gemm_rs_slice_interleaved_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int num_slices,
    int& row_idx, int& col_idx
) {
    const int tiles_per_round = num_slices * col_blocks;
    const int round_idx = task_id / tiles_per_round;
    const int within_round = task_id - round_idx * tiles_per_round;
    const int slice_idx = within_round / col_blocks;
    col_idx = within_round - slice_idx * col_blocks;
    const int rb_in_slice = round_idx;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Device-relative interleaving: each GPU starts its round with its own slice.
// Front-loads RDMA-relevant tiles so inter-node sends start earlier.
__device__ inline void gemm_rs_slice_interleaved_devrel_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int num_slices,
    int dev_idx, int& row_idx, int& col_idx
) {
    const int tiles_per_round = num_slices * col_blocks;
    const int round_idx = task_id / tiles_per_round;
    const int within_round = task_id - round_idx * tiles_per_round;
    const int raw_slice = within_round / col_blocks;
    const int slice_idx = (raw_slice + dev_idx) % num_slices;
    col_idx = within_round - raw_slice * col_blocks;
    const int rb_in_slice = round_idx;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Super-M interleaving within each slice (existing gemm_rs pattern, matches gemm_ar variant).
__device__ inline void gemm_rs_slice_super_m_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int super_m,
    int& row_idx, int& col_idx
) {
    const int tiles_per_slice = row_blocks_per_slice * col_blocks;
    const int slice_idx = task_id / tiles_per_slice;
    const int local_idx = task_id - slice_idx * tiles_per_slice;
    const int super_rows = (row_blocks_per_slice / super_m) * super_m;
    const int super_blocks = super_m * col_blocks;
    if (local_idx < super_rows * col_blocks) {
        const int band = local_idx / super_blocks;
        const int in_band = local_idx - band * super_blocks;
        const int rb_in_band = in_band % super_m;
        col_idx = in_band / super_m;
        row_idx = slice_idx * row_blocks_per_slice + band * super_m + rb_in_band;
        return;
    }
    const int tail_rows = row_blocks_per_slice - super_rows;
    const int tail_idx = local_idx - super_rows * col_blocks;
    const int rb_in_tail = tail_idx % tail_rows;
    col_idx = tail_idx / tail_rows;
    row_idx = slice_idx * row_blocks_per_slice + super_rows + rb_in_tail;
}

// Dispatcher: compile-time selection of tile visit order (matches gemm_ar's gemm_ar_decode_comp_task).
template <typename G>
__device__ inline void gemm_rs_decode_comp_task(
    int task_id, int row_blocks_per_slice, int col_blocks, int dev_idx,
    int& row_idx, int& col_idx
) {
    gemm_rs_slice_interleaved_devrel_decode(task_id, row_blocks_per_slice, col_blocks,
                                       G::NUM_DEVICES, dev_idx, row_idx, col_idx);
}

// Compute -> intra-RS ready signalling. Default is per-tile (batch=1): each
// compute tile writes 1 to ready[task_id]. With GEMM_RS_COMPUTE_SIGNAL_BATCH=N>1
// tiles are grouped in consecutive N-ID runs; each tile atomic-adds 1 to the
// group's first-tile slot, and wait_ready spins until that slot reaches N.
// This reduces release-store count by N and amortizes the ld.acquire loop but
// serializes intra-RS start on the SLOWEST tile within a group, so batch=2/4
// is the practical sweet range.
#define GEMM_RS_COMPUTE_SIGNAL_BATCH 1

__device__ inline void signal_ready(int *ready, int task_id) {
    comm::atomic_u32::release_store_gpu(ready + task_id, 1u);
}

// gemm_rs-local variant of TK's signal(): performs a cross-GPU atomic-add release
// on the peer's barrier slot at GPU scope rather than system scope. Cross-GPU
// dbuf memory on Hopper is hardware-coherent over NVLink, so release.gpu
// should deliver the increment to peer-GPU readers (the `wait()` in the send
// path uses ld.relaxed.sys, whose acquire is for ordering not visibility).
// Used in fused_comm_tile_impl to avoid the sys-scope atomic cost on the
// intra-RS → send barrier path.
__device__ inline void gemm_rs_signal_barrier_gpu(int *barrier_slot_ptr, int val) {
    comm::atomic_u32::release_add_gpu(barrier_slot_ptr, val);
}

__device__ inline void wait_ready(const int *ready, int task_id) {
    unsigned int v;
    do {
        v = comm::atomic_u32::acquire_load_gpu(ready + task_id);
        if (v != 1u) __nanosleep(16);
    } while (v != 1u);
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_queue_capacity(
    int row_blocks_per_dev, int num_send_sms, int chunks_per_row) {
    if (num_send_sms <= 0) return 0;
    const int max_rbs_per_sender =
        (row_blocks_per_dev + num_send_sms - 1) / num_send_sms;
    return max_rbs_per_sender * chunks_per_row;
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_words_per_queue(
    int row_blocks_per_dev, int num_send_sms, int chunks_per_row) {
    const int cap = gemm_rs_send_ready_bitmap_queue_capacity(
        row_blocks_per_dev, num_send_sms, chunks_per_row);
    return (cap + 31) >> 5;
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_region_base(
    int row_blocks_per_dev, int chunks_per_row) {
    return row_blocks_per_dev * fused_globals::NUM_DEVICES * chunks_per_row;
}


// ============================================================================
// Host entrypoint
// ============================================================================

// Launch scratch reset descriptor. Each region is zeroed with 32-bit writes;
// all target buffers are 4-byte aligned scratch arrays.
#define GEMM_RS_FUSED_RESET_MAX_REGIONS 16
struct gemm_rs_zero_regions_t {
    void*    ptrs[GEMM_RS_FUSED_RESET_MAX_REGIONS];
    size_t   bytes[GEMM_RS_FUSED_RESET_MAX_REGIONS];
    int      n;
};

// Forward declarations: kernel bodies live in src/gemm_rs.cu. fused_kernel is
// launched via the thin wrapper below (template instantiation needs the body
// in scope, so it must happen in the same TU as the definition); the zero
// kernel is __global__ so a forward decl is enough.
void launch_fused_gemm_rs(const fused_globals& G, unsigned int active_sms);
__global__ void gemm_rs_fused_zero_kernel(gemm_rs_zero_regions_t regs);

static unsigned int *g_counters[intra_globals::NUM_DEVICES] = {nullptr};
static int g_counters_words[intra_globals::NUM_DEVICES] = {0};
// Per-device sender_done flags (one per inter-node tile). The sender CTA
// sets sender_done[tile_id]=1 after reading and staging the tile from
// output_local; the reducer CTA waits for sender_done[tile_id]==1 before
// overwriting output_local in place. This eliminates the output_final
// scratch buffer and the post-kernel cudaMemcpyAsync tail.
static unsigned int *g_sender_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_sender_done_ntiles[intra_globals::NUM_DEVICES] = {0};
static int *g_inter_ready[intra_globals::NUM_DEVICES] = {nullptr};
static int g_inter_ready_ntiles[intra_globals::NUM_DEVICES] = {0};
static fused_globals::runtime_state *g_fused_runtime[intra_globals::NUM_DEVICES] = {nullptr};
// Per-device storage for the multicast ready-chunk descriptor.
static intra_globals::ready_chunk_distributed_tensor *g_ready_chunk_distributed_tensor_dev[intra_globals::NUM_DEVICES] = {nullptr};
// Per-chunk completion tracking (mirrors gemm_ar's intra_chunk_tiles_done).
static uint32_t *g_chunk_tiles_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_alloc_nchunks[intra_globals::NUM_DEVICES] = {0};
// Chunk barrier: per-chunk counter for batching dbuf signals (global row indexing).
static uint32_t *g_comp_chunk_tiles_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_comp_chunk_alloc[intra_globals::NUM_DEVICES] = {0};
// Compute-side chunk-done counter, separate from the owner-side progress counter.
static uint32_t *g_comp_gemm_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_comp_gemm_done_alloc[intra_globals::NUM_DEVICES] = {0};
// Per-chunk "ready to send" flag set by the owner's intra-RS.
static uint32_t *g_chunk_sendable[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_sendable_alloc[intra_globals::NUM_DEVICES] = {0};
// Per-chunk send claim bitmap. 0=unclaimed, 1=claimed.
static uint32_t *g_chunk_send_claimed[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_send_claimed_alloc[intra_globals::NUM_DEVICES] = {0};
// Arrival queue: per-chunk remote_arrived_flag (published by reducer).
static uint32_t *g_remote_arrived_flag[intra_globals::NUM_DEVICES] = {nullptr};
static int g_remote_arrived_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_remote_arrived_peer_mask[intra_globals::NUM_DEVICES] = {nullptr};
static int g_remote_arrived_peer_mask_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_arrival_queue_head[intra_globals::NUM_DEVICES] = {nullptr};
static int g_arrival_queue_head_alloc[intra_globals::NUM_DEVICES] = {0};
// Optional ready-reduce queue experiment. Reducers pop chunks only after a
// progress scan has observed local send completion plus all peer arrivals.
static uint32_t *g_ready_reduce_queue[intra_globals::NUM_DEVICES] = {nullptr};
static int g_ready_reduce_queue_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_peer_accum_queue[intra_globals::NUM_DEVICES] = {nullptr};
static int g_peer_accum_queue_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_peer_accum_done_count[intra_globals::NUM_DEVICES] = {nullptr};
static int g_peer_accum_done_count_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_chunk_reduce_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_reduce_done_alloc[intra_globals::NUM_DEVICES] = {0};
static uint32_t *g_chunk_accum_lock[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_accum_lock_alloc[intra_globals::NUM_DEVICES] = {0};

__device__ inline unsigned long long gemm_rs_globaltimer() {
    return comm::globaltimer();
}

__host__ inline int gemm_rs_effective_num_qps_host(int num_nodes) {
    constexpr int kMaxSessionQPs = 24;
    const int num_peers = std::max(1, num_nodes - 1);
    int num_qps = 4;
    if (num_peers > 1 && std::getenv("MKERNEL_CHANNELIZE_GPU_PEERS") != nullptr) {
        num_qps = std::min(kMaxSessionQPs, num_peers * 8);
    }
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        num_qps = std::atoi(env_num_qps);
    }
    if (num_qps <= 0) num_qps = 1;
    if (num_peers > 1 && std::getenv("MKERNEL_CHANNELIZE_GPU_PEERS") != nullptr) {
        num_qps = std::max(num_qps, num_peers * 8);
    }
    return std::min(kMaxSessionQPs, num_qps);
}

__host__ inline int gemm_rs_logical_queues_per_qp_host() {
    int logical = 1;
    if (const char* env_lq = std::getenv("MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        logical = std::atoi(env_lq);
    } else if (const char* env_lq = std::getenv("GEMM_RS_LOGICAL_QUEUES_PER_QP")) {
        logical = std::atoi(env_lq);
    }
    if (logical <= 0) logical = 1;
    return std::min(16, logical);
}

// Single-launch gemm_rs entrypoint: compute + intra-node RS + inter-node send + inter-node reduce.
void entrypoint_fused(
    const at::Tensor &A,
    const at::Tensor &B,
    dist::ParallelBuffer &workspace,
    dist::ParallelBuffer &output,
    dist::ParallelBuffer &barrier,
    const at::Tensor &ready,
    int64_t recv_buf_ptr,
    int64_t staging_buf_ptr,
    int64_t fifo_triggers,
    int64_t fifo_head,
    int64_t fifo_tail,
    int64_t fifo_tail_cache,
    int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_comp_sms,
    int num_intra_comm,
    int num_send_sms,
    int num_reduce_sms,
    int64_t use_acquire_poll,
    int64_t reduce_poll_sleep_ns,
    // Optional multicast ready tensor. Pass a dummy scalar-sized DistBuffer
    // when not in use.
    dist::ParallelBuffer &ready_chunk,
    // Staging DistBuffer used as the chunk-major intra-RS atomic-add target.
    pybind11::object staging_obj,
    int num_nodes
) {
    const int dev_idx = output.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    const int M = (int)A.size(0);
    const int N = (int)B.size(1);
    const int M_local = (int)output.data_.size(0);
    const int row_blocks = M / intra_globals::ROW_BLOCK;
    const int col_blocks = N / intra_globals::COL_BLOCK;
    const int local_row_blocks = M_local / fused_globals::ROW_BLOCK;
    const int total_inter_tiles = local_row_blocks * col_blocks;

    int num_comp = std::max(1, num_comp_sms);
    int num_intra = std::max(0, num_intra_comm);
    int num_send = std::max(0, num_send_sms);
    int num_reduce = std::max(0, num_reduce_sms);
    const bool intra_only_debug = (num_send == 0 && num_reduce == 0);
    // When compute directly performs the intra-RS peer store_add into staging,
    // keep the total CTA budget unchanged but collapse the scheduler to
    // 3 logical bands: (compute+intra), send, reduce. This avoids leaving a
    // hollow intra CTA partition in the persistent kernel.
    // Compute CTAs already issue the peer store_add into staging (fused
    // intra-RS), so dedicated intra CTAs are vestigial in both production and
    // intra_only_debug. Fold them into compute unconditionally.
    num_comp += num_intra;
    num_intra = 0;
    int total_ctas = num_comp + num_intra + num_send + num_reduce;
    if (!intra_only_debug && total_ctas > config::NUM_BLOCKS) {
        int overflow = total_ctas - config::NUM_BLOCKS;
        if (num_comp > overflow) {
            num_comp -= overflow;
        } else {
            num_comp = std::max(1, config::NUM_BLOCKS - (num_intra + num_send + num_reduce));
            int remaining = config::NUM_BLOCKS - (num_comp + num_intra + num_send + num_reduce);
            if (remaining < 0) {
                num_reduce = std::max(1, num_reduce + remaining);
            }
        }
    } else if (!intra_only_debug && total_ctas < config::NUM_BLOCKS) {
        num_comp += config::NUM_BLOCKS - total_ctas;
    }

    // Compute chunk geometry before allocation and scratch reset setup.
    int chunk_tiles_override = 0;
    if (const char* e = std::getenv("GEMM_RS_RDMA_CHUNK_TILES_RT")) {
        chunk_tiles_override = std::atoi(e);
    }
    const int chunk_tiles_ct = chunk_tiles_override > 0
        ? min(chunk_tiles_override, col_blocks)
        : min((int)fused_globals::CHUNK_TILES, col_blocks);
    const int chunks_per_row = (col_blocks + chunk_tiles_ct - 1) / chunk_tiles_ct;
    const int total_chunks = local_row_blocks * chunks_per_row;
    const int total_global_chunks = row_blocks * chunks_per_row;

    // Reset scratch through one compact kernel launch. The memset branch below
    // is kept as a simple local knob if this changes on another platform.
    const bool use_fused_reset = true;

    // Direct producer writes to staging are disabled in this build. The sender
    // path is responsible for packing output_local into staging before RDMA.
    const bool use_intra_rs_dual_write_rt = false;

    if (g_sender_done_ntiles[dev_idx] < total_inter_tiles) {
        if (g_sender_done[dev_idx] != nullptr) {
            cudaFree(g_sender_done[dev_idx]);
            g_sender_done[dev_idx] = nullptr;
        }
        cudaMalloc(&g_sender_done[dev_idx], total_inter_tiles * sizeof(unsigned int));
        g_sender_done_ntiles[dev_idx] = total_inter_tiles;
    }
    if (g_inter_ready_ntiles[dev_idx] < total_inter_tiles) {
        if (g_inter_ready[dev_idx] != nullptr) {
            cudaFree(g_inter_ready[dev_idx]);
            g_inter_ready[dev_idx] = nullptr;
        }
        cudaMalloc(&g_inter_ready[dev_idx], total_inter_tiles * sizeof(int));
        g_inter_ready_ntiles[dev_idx] = total_inter_tiles;
    }

    // 12 counters: next_compute, next_comm, kernel_done, next_send,
    // next_reduce, chunks_processed, ready_reduce_head/tail/scan,
    // remote_arrived_chunks, peer_accum_head/tail.
    constexpr int kGemmRsCounterWords = 12;
    if (g_counters[dev_idx] == nullptr ||
        g_counters_words[dev_idx] < kGemmRsCounterWords) {
        if (g_counters[dev_idx] != nullptr) cudaFree(g_counters[dev_idx]);
        cudaMalloc(&g_counters[dev_idx], kGemmRsCounterWords * sizeof(unsigned int));
        g_counters_words[dev_idx] = kGemmRsCounterWords;
    }

    // Per-chunk completion tracking.
    if (g_chunk_alloc_nchunks[dev_idx] < total_chunks) {
        if (g_chunk_tiles_done[dev_idx] != nullptr) cudaFree(g_chunk_tiles_done[dev_idx]);
        cudaMalloc(&g_chunk_tiles_done[dev_idx], total_chunks * sizeof(uint32_t));
        g_chunk_alloc_nchunks[dev_idx] = total_chunks;
    }

    // Chunk barrier: comp_chunk_tiles_done (global row indexing, all slices).
    if (g_comp_chunk_alloc[dev_idx] < total_global_chunks) {
        if (g_comp_chunk_tiles_done[dev_idx] != nullptr) cudaFree(g_comp_chunk_tiles_done[dev_idx]);
        cudaMalloc(&g_comp_chunk_tiles_done[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_comp_chunk_alloc[dev_idx] = total_global_chunks;
    }

    if (g_comp_gemm_done_alloc[dev_idx] < total_global_chunks) {
        if (g_comp_gemm_done[dev_idx] != nullptr) cudaFree(g_comp_gemm_done[dev_idx]);
        cudaMalloc(&g_comp_gemm_done[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_comp_gemm_done_alloc[dev_idx] = total_global_chunks;
    }

    if (g_chunk_sendable_alloc[dev_idx] < total_global_chunks) {
        if (g_chunk_sendable[dev_idx] != nullptr) cudaFree(g_chunk_sendable[dev_idx]);
        cudaMalloc(&g_chunk_sendable[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_chunk_sendable_alloc[dev_idx] = total_global_chunks;
    }

    // Per-chunk send claim bitmap.
    if (g_chunk_send_claimed_alloc[dev_idx] < total_global_chunks) {
        if (g_chunk_send_claimed[dev_idx] != nullptr) cudaFree(g_chunk_send_claimed[dev_idx]);
        cudaMalloc(&g_chunk_send_claimed[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_chunk_send_claimed_alloc[dev_idx] = total_global_chunks;
    }

    // Arrival queue: remote_arrived_flag per local chunk.
    if (g_remote_arrived_alloc[dev_idx] < total_chunks) {
        if (g_remote_arrived_flag[dev_idx] != nullptr) cudaFree(g_remote_arrived_flag[dev_idx]);
        cudaMalloc(&g_remote_arrived_flag[dev_idx], total_chunks * sizeof(uint32_t));
        g_remote_arrived_alloc[dev_idx] = total_chunks;
    }
    if (g_remote_arrived_peer_mask_alloc[dev_idx] < total_chunks) {
        if (g_remote_arrived_peer_mask[dev_idx] != nullptr) cudaFree(g_remote_arrived_peer_mask[dev_idx]);
        cudaMalloc(&g_remote_arrived_peer_mask[dev_idx], total_chunks * sizeof(uint32_t));
        g_remote_arrived_peer_mask_alloc[dev_idx] = total_chunks;
    }

    if (g_ready_reduce_queue_alloc[dev_idx] < total_chunks) {
        if (g_ready_reduce_queue[dev_idx] != nullptr) cudaFree(g_ready_reduce_queue[dev_idx]);
        cudaMalloc(&g_ready_reduce_queue[dev_idx], total_chunks * sizeof(uint32_t));
        g_ready_reduce_queue_alloc[dev_idx] = total_chunks;
    }

    const int n_peers_rt = std::max(1, num_nodes - 1);
    const int peer_accum_queue_entries = total_chunks * n_peers_rt;
    if (g_peer_accum_queue_alloc[dev_idx] < peer_accum_queue_entries) {
        if (g_peer_accum_queue[dev_idx] != nullptr) cudaFree(g_peer_accum_queue[dev_idx]);
        cudaMalloc(&g_peer_accum_queue[dev_idx],
                   (size_t)peer_accum_queue_entries * sizeof(uint32_t));
        g_peer_accum_queue_alloc[dev_idx] = peer_accum_queue_entries;
    }
    if (g_peer_accum_done_count_alloc[dev_idx] < total_chunks) {
        if (g_peer_accum_done_count[dev_idx] != nullptr) cudaFree(g_peer_accum_done_count[dev_idx]);
        cudaMalloc(&g_peer_accum_done_count[dev_idx], total_chunks * sizeof(uint32_t));
        g_peer_accum_done_count_alloc[dev_idx] = total_chunks;
    }
    if (g_chunk_reduce_done_alloc[dev_idx] < total_chunks) {
        if (g_chunk_reduce_done[dev_idx] != nullptr) cudaFree(g_chunk_reduce_done[dev_idx]);
        cudaMalloc(&g_chunk_reduce_done[dev_idx], total_chunks * sizeof(uint32_t));
        g_chunk_reduce_done_alloc[dev_idx] = total_chunks;
    }
    if (g_chunk_accum_lock_alloc[dev_idx] < total_chunks) {
        if (g_chunk_accum_lock[dev_idx] != nullptr) cudaFree(g_chunk_accum_lock[dev_idx]);
        cudaMalloc(&g_chunk_accum_lock[dev_idx], total_chunks * sizeof(uint32_t));
        g_chunk_accum_lock_alloc[dev_idx] = total_chunks;
    }

    auto env_flag = [](const char* name, bool default_value) -> bool {
        const char* e = std::getenv(name);
        if (e == nullptr) return default_value;
        return e[0] == '1';
    };
    const bool use_receiver_owner_rs_rt =
        env_flag("GEMM_RS_RECEIVER_OWNER_RS", false);
    const bool use_incremental_peer_reduce_rt =
        env_flag("GEMM_RS_INCREMENTAL_PEER_REDUCE", use_receiver_owner_rs_rt);
    const bool use_transport_arrival_queue_rt =
        env_flag("GEMM_RS_TRANSPORT_ARRIVAL_QUEUE", use_incremental_peer_reduce_rt);
    const bool use_ready_reduce_queue_rt =
        use_incremental_peer_reduce_rt || use_transport_arrival_queue_rt ||
        (std::getenv("GEMM_RS_READY_REDUCE_QUEUE") != nullptr &&
         std::getenv("GEMM_RS_READY_REDUCE_QUEUE")[0] == '1');
    const int num_remote_queues_rt =
        gemm_rs_effective_num_qps_host(num_nodes) * gemm_rs_logical_queues_per_qp_host();
    const int session_arrival_slots_rt = std::max(1, num_nodes - 1) * total_inter_tiles;
    const int remote_queue_stride_rt =
        std::max(1, (session_arrival_slots_rt + num_remote_queues_rt - 1) / num_remote_queues_rt);
    int num_recv_progress_sms_rt = use_transport_arrival_queue_rt ? std::min(2, num_reduce) : 0;
    if (const char* e = std::getenv("GEMM_RS_RECV_PROGRESS_SMS")) {
        num_recv_progress_sms_rt = std::max(0, std::atoi(e));
    }
    num_recv_progress_sms_rt = std::min(num_recv_progress_sms_rt, num_reduce);
    if (g_arrival_queue_head_alloc[dev_idx] < num_remote_queues_rt) {
        if (g_arrival_queue_head[dev_idx] != nullptr) cudaFree(g_arrival_queue_head[dev_idx]);
        cudaMalloc(&g_arrival_queue_head[dev_idx],
                   (size_t)num_remote_queues_rt * sizeof(uint32_t));
        g_arrival_queue_head_alloc[dev_idx] = num_remote_queues_rt;
    }
    const int owner_chunks_total_rt = use_receiver_owner_rs_rt
        ? (node_idx < total_chunks
            ? 1 + (total_chunks - 1 - node_idx) / std::max(1, num_nodes)
            : 0)
        : total_chunks;


    if (use_fused_reset) {
        gemm_rs_zero_regions_t regs{};
        regs.n = 0;
        auto add = [&](void* p, size_t b) {
            if (regs.n < GEMM_RS_FUSED_RESET_MAX_REGIONS && b > 0) {
                regs.ptrs[regs.n] = p;
                regs.bytes[regs.n] = b;
                ++regs.n;
            }
        };
        add(g_sender_done[dev_idx], (size_t)total_inter_tiles * sizeof(unsigned int));
        add(g_inter_ready[dev_idx], (size_t)total_inter_tiles * sizeof(int));
        add(g_counters[dev_idx], kGemmRsCounterWords * sizeof(unsigned int));
        add(g_chunk_tiles_done[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_comp_chunk_tiles_done[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_comp_gemm_done[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_chunk_sendable[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_chunk_send_claimed[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_remote_arrived_flag[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_remote_arrived_peer_mask[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_arrival_queue_head[dev_idx], (size_t)num_remote_queues_rt * sizeof(uint32_t));
        add(g_ready_reduce_queue[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_peer_accum_queue[dev_idx], (size_t)peer_accum_queue_entries * sizeof(uint32_t));
        add(g_peer_accum_done_count[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_chunk_reduce_done[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_chunk_accum_lock[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        // Grid = one block per region (up to 16); 128 threads/block, strided.
        if (regs.n > 0) {
            gemm_rs_fused_zero_kernel<<<dim3((unsigned)regs.n, 1, 1),
                                   dim3(128, 1, 1), 0, stream>>>(regs);
        }
    } else {
        // Memset fallback over the same regions as the fused-zero path.
        cudaMemsetAsync(g_sender_done[dev_idx], 0,
            (size_t)total_inter_tiles * sizeof(unsigned int), stream);
        cudaMemsetAsync(g_inter_ready[dev_idx], 0,
            (size_t)total_inter_tiles * sizeof(int), stream);
        cudaMemsetAsync(g_counters[dev_idx], 0,
            kGemmRsCounterWords * sizeof(unsigned int), stream);
        cudaMemsetAsync(g_chunk_tiles_done[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_comp_chunk_tiles_done[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_comp_gemm_done[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_sendable[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_send_claimed[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_remote_arrived_flag[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_remote_arrived_peer_mask[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_arrival_queue_head[dev_idx], 0,
            (size_t)num_remote_queues_rt * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_ready_reduce_queue[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_peer_accum_queue[dev_idx], 0,
            (size_t)peer_accum_queue_entries * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_peer_accum_done_count[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_reduce_done[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_accum_lock[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
    }

    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity, 4);

    // Stage the ready_chunk descriptor to device memory so runtime_state can
    // hold a stable pointer for the kernel.
    intra_globals::ready_chunk_distributed_tensor *ready_chunk_dev_ptr = nullptr;
    if (!intra_only_debug) {
        if (g_ready_chunk_distributed_tensor_dev[dev_idx] == nullptr) {
            cudaMalloc(&g_ready_chunk_distributed_tensor_dev[dev_idx],
                       sizeof(intra_globals::ready_chunk_distributed_tensor));
        }
        auto rc_distributed_tensor = ::dist::distributed_tensor_from_buffer<intra_globals::ready_chunk_distributed_tensor>(
            ready_chunk, 1, 1, 1, (int)ready_chunk.data_.numel());
        cudaMemcpyAsync(g_ready_chunk_distributed_tensor_dev[dev_idx], &rc_distributed_tensor,
                        sizeof(rc_distributed_tensor), cudaMemcpyHostToDevice, stream);
        ready_chunk_dev_ptr = g_ready_chunk_distributed_tensor_dev[dev_idx];
    }

    fused_globals::runtime_state *rt_ptr = nullptr;
    if (!intra_only_debug) {
        if (g_fused_runtime[dev_idx] == nullptr) {
            cudaMalloc(&g_fused_runtime[dev_idx], sizeof(fused_globals::runtime_state));
        }
        fused_globals::runtime_state rt{
            .inter_ready = g_inter_ready[dev_idx],
            .output_local = reinterpret_cast<bf16*>(output.data_.data_ptr()),
            .recv_buf = reinterpret_cast<bf16*>(recv_buf_ptr),
            .staging_buf = reinterpret_cast<bf16*>(staging_buf_ptr),
            .M_local = M_local,
            .N = N,
            .node_idx = node_idx,
            .num_nodes = num_nodes,
            .sender_done = g_sender_done[dev_idx],
            .d2h_fifos = fifo_bundle,
            .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
            .epoch = (uint32_t)epoch,
            .total_tiles = total_inter_tiles,
            .next_send = &g_counters[dev_idx][3],
            .next_reduce = &g_counters[dev_idx][4],
            .chunks_processed = &g_counters[dev_idx][5],
            .remote_arrived_chunks = &g_counters[dev_idx][9],
            .chunk_tiles_done = g_chunk_tiles_done[dev_idx],
            .comp_chunk_tiles_done = g_comp_chunk_tiles_done[dev_idx],
            .comp_gemm_done = g_comp_gemm_done[dev_idx],
            .chunk_sendable = g_chunk_sendable[dev_idx],
            .chunk_send_claimed = g_chunk_send_claimed[dev_idx],
            .ready_chunk = ready_chunk_dev_ptr,
            .remote_arrived_flag = g_remote_arrived_flag[dev_idx],
            .remote_arrived_peer_mask = g_remote_arrived_peer_mask[dev_idx],
            .arrival_queue_head = g_arrival_queue_head[dev_idx],
            .ready_reduce_queue = g_ready_reduce_queue[dev_idx],
            .ready_reduce_head = &g_counters[dev_idx][6],
            .ready_reduce_tail = &g_counters[dev_idx][7],
            .ready_reduce_scan = &g_counters[dev_idx][8],
            .peer_accum_queue = g_peer_accum_queue[dev_idx],
            .peer_accum_head = &g_counters[dev_idx][10],
            .peer_accum_tail = &g_counters[dev_idx][11],
            .peer_accum_done_count = g_peer_accum_done_count[dev_idx],
            .chunk_reduce_done = g_chunk_reduce_done[dev_idx],
            .chunk_accum_lock = g_chunk_accum_lock[dev_idx],
            .chunks_per_row = chunks_per_row,
            .chunk_tiles_val = chunk_tiles_ct,
            .col_blocks_val = col_blocks,
            .row_blocks_per_slice = local_row_blocks,
            .num_remote_queues = num_remote_queues_rt,
            .remote_queue_stride = remote_queue_stride_rt,
            .num_recv_progress_sms = num_recv_progress_sms_rt,
            .owner_chunks_total = owner_chunks_total_rt,
            .use_acquire_poll = (uint8_t)(use_acquire_poll != 0 ? 1u : 0u),
            .use_intra_rs_dual_write = (uint8_t)(use_intra_rs_dual_write_rt ? 1u : 0u),
            .use_ready_reduce_queue = (uint8_t)(use_ready_reduce_queue_rt ? 1u : 0u),
            .use_transport_arrival_queue = (uint8_t)(use_transport_arrival_queue_rt ? 1u : 0u),
            .use_incremental_peer_reduce = (uint8_t)(use_incremental_peer_reduce_rt ? 1u : 0u),
            .use_receiver_owner_rs = (uint8_t)(use_receiver_owner_rs_rt ? 1u : 0u),
            ._pad1 = {0, 0, 0},
            .reduce_poll_sleep_ns = (uint32_t)(reduce_poll_sleep_ns > 0 ? reduce_poll_sleep_ns : 100),
        };
        cudaMemcpyAsync(g_fused_runtime[dev_idx], &rt, sizeof(rt),
                        cudaMemcpyHostToDevice, stream);
        rt_ptr = g_fused_runtime[dev_idx];
    }

    // Build the chunk-major view of staging. Python allocates a
    // DistBuffer with shape (m_local, n); we view the bytes as
    // (total_inter_tiles*128, 256) so tma::store_add_async with C_tile
    // (64, 256) and tile coord (2*global_tile_idx+i, 0) lands at chunk-major
    // offset (global_tile_idx*128 + i*64)*256 elements from the base.
    TORCH_CHECK(!staging_obj.is_none(),
        "GEMM_RS_INTRA_RS_DIRECT_STAGING=1 requires a staging DistBuffer; "
        "Python caller passed None");
    auto &staging = staging_obj.cast<dist::ParallelBuffer &>();
    const int staging_rows = total_inter_tiles * 128;
    auto staging_distributed_tensor_built =
        ::dist::distributed_tensor_from_buffer<intra_globals::output_distributed_tensor>(
            staging, 1, 1, staging_rows, intra_globals::COL_BLOCK);

    intra_globals intra{
        .A = ::dist::local_tensor_from_tensor<intra_globals::A_local_tensor>(A),
        .B = ::dist::local_tensor_from_tensor<intra_globals::B_local_tensor>(B),
        .workspace = ::dist::distributed_tensor_from_buffer<intra_globals::workspace_distributed_tensor>(workspace),
        .output = ::dist::distributed_tensor_from_buffer<intra_globals::output_distributed_tensor>(output),
        .staging = staging_distributed_tensor_built,
        .barrier = ::dist::distributed_tensor_from_buffer<intra_globals::barrier_distributed_tensor>(barrier),
        .ready = ready.data_ptr<int>(),
        .dev_idx = dev_idx,
        .num_comm_sms = num_intra,
        .num_comp_sms = num_comp,
        .next_compute = &g_counters[dev_idx][0],
        .next_comm = &g_counters[dev_idx][1],
        .kernel_done = &g_counters[dev_idx][2],
    };

    fused_globals G{
        .intra = intra,
        .num_send_sms = num_send,
        .num_reduce_sms = num_reduce,
        .rt = rt_ptr,
    };

    (void)stream;
    if (intra_only_debug) {
        launch_fused_gemm_rs(G, (unsigned int)num_comp);
    } else {
        launch_fused_gemm_rs(G, 0);
    }
}

}  // namespace gemm_rs_multinode

