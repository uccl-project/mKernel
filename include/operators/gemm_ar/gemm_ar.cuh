#pragma once

/**
 * @file include/operators/gemm_ar/gemm_ar.cuh
 * @brief Multi-node GEMM + All-Reduce — host setup + device-side prelude.
 *
 * Contains:
 *   - config struct (CTA-role layout, register splits)
 *   - fused_globals struct (compile-time tile shapes + runtime SM split)
 *   - role-shared device helpers: fifo selection, send-cmd ABI, arrival
 *     queue / chunk indexing, cross-node barrier primitives, deadlock-debug
 *     hooks, multicast publication
 *   - host-side scratch/role-split layout + host entrypoint
 *
 * The 4 CTA roles' implementations (compute / intra-AR / inter-send /
 * inter-reduce-and-publish) live in `src/gemm_ar.cu` along with the fused
 * kernel dispatcher; pybind glue lives in `session.cuh`.
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"
#include "comm/atomic_u32.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/ready_queue.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace gemm_ar_multinode {

// Forward declarations: kernel bodies (fused_kernel + fused_epilogue_kernel)
// live in src/gemm_ar.cu. They're launched via these thin wrappers (template
// instantiation needs the body in scope, so the launch must happen in the
// same TU as the definition). The host entrypoint below calls these.
struct fused_globals;
void launch_fused_gemm_ar(const fused_globals& G);
void launch_fused_gemm_ar_epilogue(const fused_globals& G);

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
// Helpers: tile indexing within each device-owned slice
// ============================================================================
__device__ inline void slice_row_major_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int& row_idx, int& col_idx
) {
    const int tiles_per_slice = row_blocks_per_slice * col_blocks;
    const int slice_idx = task_id / tiles_per_slice;
    const int local_idx = task_id - slice_idx * tiles_per_slice;
    const int rb_in_slice = local_idx / col_blocks;
    col_idx = local_idx - rb_in_slice * col_blocks;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Round-robin across GPU slices by row-block index, row-major within each
// (slice, rb_in_slice) tile. One "round" walks every slice once for the same
// rb_in_slice, visiting all column tiles of that row-block before advancing
// rb_in_slice.
//
// Compared to slice_row_major_decode (all tiles of slice 0, then slice 1,
// ...), this aligns barrier / intra-AR progress across all M GPUs so no rank
// sits in the final multicast poll for ~one GEMM duration waiting for the
// last slice to start reducing.
//
// RDMA / send coalescing: each GPU still completes row-blocks in ascending
// rb_in_slice (0..row_blocks_per_slice-1). Row-level coalescing only needs
// all column chunks of a row-block to reach CHUNK_LOCAL_AR_DONE; their
// completion order within the row is already non-deterministic (CTA stride).
__device__ inline void slice_interleaved_slices_decode(
    int task_id,
    int row_blocks_per_slice,
    int col_blocks,
    int num_slices,
    int& row_idx,
    int& col_idx
) {
    const int tiles_per_round = num_slices * col_blocks;
    const int round_idx = task_id / tiles_per_round;
    const int within_round = task_id - round_idx * tiles_per_round;
    const int slice_idx = within_round / col_blocks;
    col_idx = within_round - slice_idx * col_blocks;
    const int rb_in_slice = round_idx;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Device-relative interleaving: same as slice_interleaved_slices_decode but
// each GPU starts its round with its OWN slice. GPU d visits slices in order
// d, d+1, ..., d+7 (mod num_slices) within each round. This front-loads
// each GPU's RDMA-relevant tiles so inter-node sends start earlier, while
// preserving cross-slice balance (all tiles still produced within one round).
__device__ inline void slice_interleaved_devrel_decode(
    int task_id,
    int row_blocks_per_slice,
    int col_blocks,
    int num_slices,
    int dev_idx,
    int& row_idx,
    int& col_idx
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

__device__ inline void slice_super_m_decode(
    int task_id,
    int row_blocks_per_slice,
    int col_blocks,
    int super_m,
    int& row_idx,
    int& col_idx
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

__host__ __device__ inline int gemm_ar_chunk_tiles(int col_blocks) {
    // Keep RDMA chunks at 256 KiB. The previous col_blocks>=128 path used
    // 32 tiles (2 MiB) per WR; on 4-node L20x this can leave the receiver
    // waiting forever at M=32768+.
    int chunk_tiles = 4;
#ifndef __CUDA_ARCH__
    if (const char* env = std::getenv("GEMM_AR_CHUNK_TILES")) {
        const int override_tiles = std::atoi(env);
        if (override_tiles > 0) chunk_tiles = override_tiles;
    }
#endif
    return (chunk_tiles < col_blocks) ? chunk_tiles : col_blocks;
}

__host__ __device__ inline int gemm_ar_chunks_per_row(int col_blocks) {
    const int chunk_tiles = gemm_ar_chunk_tiles(col_blocks);
    return (col_blocks + chunk_tiles - 1) / chunk_tiles;
}

__host__ __device__ inline int gemm_ar_units_for_queue(int total_units, int queue_id, int num_queues) {
    if (num_queues <= 0 || queue_id < 0 || queue_id >= num_queues || total_units <= queue_id) {
        return 0;
    }
    return 1 + (total_units - 1 - queue_id) / num_queues;
}
// Shared fused-kernel prelude for multinode GEMM+AR.
// Defined inside the gemm_ar_multinode namespace; instantiated in src/gemm_ar.cu.

static constexpr int TILE_BYTES = 128 * 256 * 2;

// ============================================================================
// Fused globals — holds state for all 4 CTA roles
// ============================================================================

struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int TILE_ELEMS = ROW_BLOCK * COL_BLOCK;  // 32768

    // Tile types for GEMM pipeline
    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    // TMA descriptors
    using A_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, B_tile>;
    using C_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;
    using final_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<NUM_DEVICES>;

    // GEMM inputs
    A_local_tensor A;
    B_local_tensor B;
    C_distributed_tensor C;                  // IPC multicast buffer for intra-node all-reduce
    final_distributed_tensor C_final;        // multicast-backed final output replicated intra-node
    barrier_distributed_tensor barrier;      // per-tile barrier for compute→intra-AR handoff

    // Inter-node RDMA
    bf16* C_local;            // raw pointer to C.data_ (intra-reduced result)
    bf16* C_recv;             // RDMA recv buffer (peer node's slice)
    bf16* staging_buf;        // RDMA-registered staging (my slice only)
    bf16* C_remote_accum;     // optional local accumulator for early remote reduction

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t* arrival_flags;  // row/chunk/tile arrival flags, depending on transport mode
    volatile uint32_t* arrival_tails;  // optional sender-published per-queue tails
    uint32_t epoch;

    volatile uint32_t* cross_node_barrier;  // GPU-readable, RDMA-writable barrier flag
                                            // (host-pinned mapped from stage_barrier)
    uint32_t* xnode_ready_device;           // Device-memory (HBM) mirror set by the
                                            // single PCIe-polling thread once the
                                            // cross-node barrier clears. All other
                                            // CTAs spin on this fast HBM flag.


    // Backing scratch allocation provided by the benchmark. The chunk-centric
    // path splits it into explicit per-chunk/state regions so correctness no
    // longer depends on a global CTA retirement count.
    int* ar_done;
    int* chunk_remote_arrived;         // per-chunk remote-arrival bit
    uint32_t* arrival_queue_head;      // per-queue recv cursors
    uint32_t* local_ar_done_chunks;    // chunks that reached LOCAL_AR_DONE
    uint32_t* send_issued_chunks;      // chunks whose RDMA was issued
    uint32_t* remote_arrived_chunks;   // chunks whose remote arrival was observed
    uint32_t* published_chunks;        // chunks reduced + published to C_final
    uint32_t* final_publish_done;      // slice-complete signal for local CTAs
    uint32_t* debug_flags;             // runtime-enable deadlock instrumentation
    uint32_t* debug_dump_counter;      // throttles periodic chunk dumps
    uint32_t* queue_expected_arrivals; // exact arrivals expected per logical queue
    uint32_t* queue_observed_arrivals; // arrivals actually decoded per queue
    uint32_t* row_send_count;          // AQ coalescing: completed chunks per row-block

    // Producer-published gemm_ready_queue: compute CTAs hand off chunks to
    // intra-AR CTAs as soon as all tiles in the chunk are ready.
    uint32_t* gemm_tiles_ready_count;  // per-chunk tile barrier counter
    uint32_t* gemm_tile_scanned;       // per-tile barrier-detected flag
    int* gemm_rq_entries;              // ring buffer of chunk IDs (+1 encoded)
    uint32_t* gemm_rq_head;            // consumer CAS to dequeue
    uint32_t* gemm_rq_tail;            // producer reserve/publish cursor
    uint32_t* gemm_scan_cursor;        // idle intra-AR CTAs claim chunks to test
    int gemm_rq_capacity;              // power-of-2

    // Producer-published local_ar_done_queue: intra-AR CTAs hand off chunks
    // to inter-send CTAs once the local all-reduce step is done.
    int* local_ar_rq_entries;          // ring buffer of chunk IDs (+1 encoded)
    uint32_t* local_ar_rq_head;        // consumer CAS to dequeue
    uint32_t* local_ar_rq_tail;        // producer reserve/publish cursor
    int local_ar_rq_capacity;          // power-of-2

    // Producer-published remote_arrived_queue: recv-progress CTAs hand off
    // chunks to inter-reduce CTAs once the remote partial has landed.
    int* remote_rq_entries;            // ring buffer of chunk IDs (+1 encoded)
    uint32_t* remote_rq_head;          // consumer CAS to dequeue
    uint32_t* remote_rq_tail;          // producer reserve/publish cursor
    int remote_rq_capacity;            // power-of-2
    int recv_ready_queue_enabled;       // recv-progress CTAs enqueue ready chunks for reducers
    int* owner_pending_entries;        // per-owner MPSC ring of chunk IDs (+1 encoded)
    uint32_t* owner_pending_head;      // single-consumer head per owner queue
    uint32_t* owner_pending_tail;      // producer reserve/publish cursor per owner queue
    uint32_t* owner_pending_max_depth; // per-owner maximum observed queue depth
    uint32_t* owner_pending_push_count;// per-owner enqueued chunks
    uint32_t* owner_pending_pop_count; // per-owner successfully claimed chunks
    int owner_pending_capacity;        // power-of-2 per owner queue

    // Static-ownership flag arrays for the inter-reduce stage.
    // Written with st.release.gpu.global by the unique owner CTA for each chunk.
    // Read with ld.acquire.gpu.global by downstream role CTAs (static ownership).
    // Sized `total_chunks` each. Allocated from the ar_done scratch buffer.
    uint32_t* local_done_flag;         // set to 1 by intra_ar owner after local AR done
    uint32_t* remote_arrived_flag;     // set to 1 by recv_progress owner after RDMA arrival
    uint32_t* remote_arrived_peer_mask; // per-chunk bitmask of arrived remote peer slots
    uint32_t* intra_started_flag;      // GEMM_AR_STAGGER_INTRA: up to 16 entries (stride/2);
                                       // primary CTA[i] sets 1 after passing its first tile barrier;
                                       // secondary CTA[i+half] polls this (L2-cached) before its own

    // Cursor-based work-stealing reduce.
    // All CTAs (including recycled compute/intra/send) join the reduce phase.
    // reduce_cursor: single global counter, atomicAdd'd to get candidate chunk.
    // chunk_claimed_flag: per-chunk claim bit (0=unclaimed, 1=claimed via CAS).
    uint32_t* reduce_cursor;
    uint32_t* chunk_claimed_flag;

    // Dimensions
    int N;                    // output columns
    int dev_idx;              // local rank within node (0..INTRA_NUM_DEVICES-1)
    int node_idx;             // 0..num_nodes-1
    int num_nodes;            // total node count (>= 2)
    int slice_rows;           // M / NUM_DEVICES
    int row_blocks_per_slice; // slice_rows / ROW_BLOCK
    int col_blocks;           // N / COL_BLOCK

    // CTA role boundaries
    int num_comp_sms;
    int num_intra_ar_sms;
    int num_inter_send_sms;
    int num_inter_reduce_publish_sms;
    int num_inter_recv_progress_sms;
    int num_inter_reduce_store_sms;
    int num_qps;
    int num_remote_queues;
    int remote_queue_stride;
    int defer_final_multicast_finish;
    int work_steal_enabled;
    int intra_ready_multimem;
    // When true, intra-AR xdev barrier waits use acquire loads instead of
    // relaxed loads. The branch is outside the spin body.
    bool use_acquire_poll = false;
    uint32_t r8_warp_spec = 0;
    int early_remote_accum = 0;
    int total_chunks;
    int total_tiles_per_device;       // row_blocks_per_slice * col_blocks (tiles this device reduces)
    int chunk_tiles;
    int chunks_per_row;
    uint32_t* comp_chunk_tiles_done; // Per-chunk tile completion counter (local GPU).
                                     // Each GPU atomicAdd's per tile; last tile fires
                                     // the NVSwitch barrier signal. Size = row_blocks * chunks_per_row.
    uint32_t* intra_chunk_tiles_done; // Per-chunk tile completion counter for intra-AR.
                                      // Each intra-AR CTA atomicAdd's per tile; last tile
                                      // sets local_done_flag. Size = total_chunks.
    // remaining CTAs = inter-reduce-and-publish
    unsigned int* epilogue_blocks_done;


    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};


__device__ __forceinline__ internode::D2HFifoDevice gemm_ar_send_fifo_for_lane(
    const fused_globals& G,
    uint32_t lane_id
) {
    return internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, lane_id);
}

// Per-CTA TX path for gemm_ar. `send_id` selects this CTA's logical sender,
// `target_rx_id` selects the remote logical queue.
//
// FIFO routing splits by topology:
//   N == 2: lane = target_rx_id (work-keyed; better load balancing across
//           QPs when 8 devices share K logical queues).
//   N >  2: lane = cmd.reserved0 = peer_slot * NUM_DEVICES + dev_idx
//           (peer-coherent so the proxy batches never mix dst_ranks).
// A/B tested in 5/2026 — unifying on reserved0 cost 2x at M=2048; on
// target_rx_id the proxy can't batch at all at N>2.
__device__ __forceinline__ void gemm_ar_post_send_cmd(
    const fused_globals& G,
    int send_id,
    int target_rx_id,
    const internode::TransferCmd& cmd
) {
    (void)send_id;
    const uint32_t fifo_lane =
        (G.num_nodes > 2 && cmd.cmd_type == internode::CmdType::WRITE)
            ? (uint32_t)cmd.reserved0
            : (uint32_t)target_rx_id;
    gemm_ar_send_fifo_for_lane(G, fifo_lane).push(cmd);
}

// Batched-flush variant: enqueue WQE only (no fence, no DB). Pair with
// gemm_ar_flush_send_qp() after all enqueues for the same TX QP. Mirrors
// reference's flush_batch (gda_endpoint.cuh:163-195) which amortizes one
// __threadfence_system across multiple WQEs. The proxy backend falls back to
// gemm_ar_post_send_cmd.
__device__ __forceinline__ void gemm_ar_enqueue_send_cmd(
    const fused_globals& G,
    int send_id,
    int target_rx_id,
    const internode::TransferCmd& cmd
) {
    // Proxy path posts immediately on each call — no batched flush.
    gemm_ar_post_send_cmd(G, send_id, target_rx_id, cmd);
}

// Flush previously-enqueued WQEs on this CTA's TX QP: one __threadfence_system
// followed by one doorbell ring at the current pc. No-op under the proxy
// backend, which posts per call. Matches reference's pattern of one fence +
// N doorbells per flush_batch.
__device__ __forceinline__ void gemm_ar_flush_send_qp(
    const fused_globals& G,
    int send_id
) {
    (void)G; (void)send_id;
}


// Cooperative iter-end reset of arrival flags + tails. Must be called BEFORE
// gemm_ar_cross_node_barrier_push_from on iter boundary: the cross-node barrier
// gates peer's next-iter kernel launch, so peer's next-iter RDMA writes to
// our flags can only arrive AFTER we pushed barrier — which is AFTER this
// reset. Replaces the host-side reset_arrival_flags call in commit_epoch
// which races with in-flight RDMA from a peer that has already advanced.
__device__ inline void gemm_ar_iter_end_reset_arrival_flags(
    const fused_globals& G, int participating_block_start, int participating_block_count
) {
    const int local_bid = blockIdx.x - participating_block_start;
    if (local_bid < 0 || local_bid >= participating_block_count) return;
    const int total_flag_words = G.num_remote_queues * G.remote_queue_stride;
    const int total_tail_words = G.num_remote_queues;
    const int stride = participating_block_count * blockDim.x;
    const int offset = local_bid * blockDim.x + threadIdx.x;
    volatile uint32_t* flag_ptr = G.arrival_flags;
    for (int i = offset; i < total_flag_words; i += stride) {
        flag_ptr[i] = 0u;
    }
    if (G.arrival_tails != nullptr) {
        volatile uint32_t* tail_ptr = G.arrival_tails;
        for (int i = offset; i < total_tail_words; i += stride) {
            tail_ptr[i] = 0u;
        }
    }
    __threadfence_system();
    __syncthreads();
}

// GPU-side cross-node barrier: the caller-designated CTA (`push_block_id`)
// pushes BARRIER_NOTIFY to the proxy, which posts an RDMA write of epoch to
// the remote node's stage_barrier slot. All participating CTAs spin-poll the
// local barrier flag until the remote has signaled. Used both at kernel start
// (push_block=0, all 132 CTAs poll) and at iter-end in the epilogue
// (push_block=reduce_base, only epilogue CTAs poll).
__device__ inline void gemm_ar_cross_node_barrier_push_from(
    const fused_globals& G, int push_block_id
) {
    if (G.cross_node_barrier == nullptr) return;

    if (blockIdx.x == push_block_id && threadIdx.x == 0) {
        internode::TransferCmd cmd{};
        cmd.cmd_type = internode::CmdType::BARRIER_NOTIFY;
        cmd.enqueue_device_ns = comm::globaltimer();
        gemm_ar_send_fifo_for_lane(G, 0).push(cmd);
    }

    // cross_node_barrier lives in cudaHostAllocMapped memory — every GPU read
    // traverses PCIe. With ~132 blocks * 12 warps spinning (~1600 readers) the
    // PCIe gets saturated and adds milliseconds. Hierarchical spin: ONE thread
    // (push_block_id, tid 0) polls the PCIe flag; when it clears, it publishes
    // to a device-memory mirror (xnode_ready_device) in HBM. All other CTAs
    // spin on that HBM mirror — same monotonic epoch semantics, but PCIe
    // traffic stays at a single reader.
    //
    // Monotonic compare (>=): commit_epoch resets the flag to 0 each iter, so
    // in steady-state drift a remote RDMA write of epoch N could land after
    // local's reset for iter N+1 — `!=` would then spin on a wiped flag.
    // Using `>=` is idempotent: flag resets to 0 < epoch, and any write at or
    // newer than our current epoch satisfies the barrier.
    if (blockIdx.x == push_block_id) {
        if (threadIdx.x == 0) {
            while (*G.cross_node_barrier < G.epoch) {
            }
            if (G.xnode_ready_device != nullptr) {
                comm::atomic_u32::release_store_gpu(G.xnode_ready_device, G.epoch);
            }
        }
    } else if (G.xnode_ready_device != nullptr) {
        if (threadIdx.x == 0) {
            uint32_t v;
            do {
                v = comm::atomic_u32::acquire_load_gpu(G.xnode_ready_device);
            } while (v < G.epoch);
        }
    } else {
        // Fallback (no HBM mirror wired): all blocks spin on PCIe flag.
        if (threadIdx.x == 0) {
            while (*G.cross_node_barrier < G.epoch) {
            }
        }
    }
    __syncthreads();
}

__device__ inline void gemm_ar_cross_node_barrier(const fused_globals& G) {
    gemm_ar_cross_node_barrier_push_from(G, 0);
}

// Pairwise iter-end cross-node barrier (monotonic-epoch compare, no clear).
//
// On each rank, one driver thread does:
//   1. intra-node M-way barrier (plane-1 slot {0,0}) — syncs all M local GPUs.
//   2. push BARRIER_NOTIFY → proxy RDMA-writes its current cfg_.epoch into
//      pair-peer's stage_barrier slot 0; spin on our own local slot until
//      slot >= G.epoch. Never cleared by the kernel.
//
// Why monotonic (not clear-to-0): clearing to 0 after observing the arrival
// races with the peer's next-iter RDMA write landing on the same slot —
// if peer's iter N+1 write lands before we clear, our clear wipes it and we
// hang. Since set_epoch() is called per-iter with a monotonic value and the
// proxy RDMA-writes cfg_.epoch, the receiver can simply wait for slot >=
// current epoch; stale-larger values (peer ahead) satisfy immediately, which
// is correct.
//
// Why pairwise (not leader-only): each rank has its own Session + pinned
// stage_barrier slot + pair-peer (rank X ↔ rank X across nodes). All M
// RDMAs per direction fly concurrently → better NIC utilization, symmetric
// code, no single-leader serialization.
__device__ inline void gemm_ar_hierarchical_xnode_barrier(
    const fused_globals& G, int driver_block_id
) {
    if (G.cross_node_barrier == nullptr) return;
    if (blockIdx.x != driver_block_id || threadIdx.x != 0) return;


    // Step 1: M-way intra-node arrival via per-epoch slot to rule out any
    // cross-iter residue on slot {1,0,0}. epoch & 1023 gives 1024 distinct
    // slots cycling, far longer than any plausible in-flight op horizon.
    //
    // GEMM_AR_SKIP_XBAR_INTRA_BARRIER: when the caller just returned from
    // gemm_ar_finish_multicast_publication on every reduce CTA, the M-way intra-node
    // sync has already been achieved per-chunk (signal_all + wait_mc(==1) on
    // each reduce CTA's slot). Re-doing barrier_all here is redundant and
    // costs an extra NVSwitch round-trip. Gated so we can ablate.

    // Step 2: pairwise cross-node handshake. Push NOTIFY (proxy drains inflight
    // data WRs, then RDMA-writes cfg_.epoch into pair-peer's slot 0), then
    // spin on our local slot until pair-peer's proxy has published an epoch
    // >= ours.
    internode::TransferCmd cmd{};
    cmd.cmd_type = internode::CmdType::BARRIER_NOTIFY;
    cmd.enqueue_device_ns = comm::globaltimer();
    gemm_ar_send_fifo_for_lane(G, 0).push(cmd);


    // ld.acquire.sys: system-scope acquire so writes from the NIC (populated
    // via peer's RDMA) are observed with proper ordering.
    uint32_t v;
    do {
        v = comm::atomic_u32::acquire_load_sys(const_cast<uint32_t*>(G.cross_node_barrier));
    } while (v < G.epoch);

}

// Split xbar into push-only and wait-only phases so the ~RTT can overlap
// with finish_multicast_publication. Correctness argument:
//   - The NOTIFY push must come after gemm_ar_iter_end_reset_arrival_flags so the
//     peer's next-iter RDMA writes (gated on receiving our NOTIFY) can't
//     collide with our flag-clear.
//   - The NOTIFY push does NOT need to come after finish_multicast_publication
//     because finish_multicast guards only local-to-node multicast slots,
//     which the peer never reads or writes. Thus the push can fire before
//     finish_multicast, and we spin on arrival afterwards.
__device__ inline void gemm_ar_xbar_push_only(
    const fused_globals& G, int driver_block_id
) {
    if (G.cross_node_barrier == nullptr) return;
    if (blockIdx.x != driver_block_id || threadIdx.x != 0) return;
    internode::TransferCmd cmd{};
    cmd.cmd_type = internode::CmdType::BARRIER_NOTIFY;
    cmd.enqueue_device_ns = comm::globaltimer();
    gemm_ar_send_fifo_for_lane(G, 0).push(cmd);
}

__device__ inline void gemm_ar_xbar_wait_only(
    const fused_globals& G, int driver_block_id
) {
    if (G.cross_node_barrier == nullptr) return;
    if (blockIdx.x != driver_block_id || threadIdx.x != 0) return;
    uint32_t v;
    do {
        v = comm::atomic_u32::acquire_load_sys(const_cast<uint32_t*>(G.cross_node_barrier));
    } while (v < G.epoch);
}

__host__ __device__ inline int gemm_ar_tiles_for_queue(
    int row_blocks_per_slice, int col_blocks, int queue_id, int num_queues
) {
    if (num_queues <= 0 || queue_id < 0 || queue_id >= num_queues) return 0;
    const int chunk_tiles = gemm_ar_chunk_tiles(col_blocks);
    const int chunks_per_row = gemm_ar_chunks_per_row(col_blocks);
    const int total_chunks = row_blocks_per_slice * chunks_per_row;
    int total = 0;
    for (int chunk_id = queue_id; chunk_id < total_chunks; chunk_id += num_queues) {
        const int chunk_idx = chunk_id % chunks_per_row;
        total += min(chunk_tiles, col_blocks - chunk_idx * chunk_tiles);
    }
    return total;
}

__host__ __device__ inline int gemm_ar_queue_miss_budget() {
    return 8;
}

__device__ inline int gemm_ar_arrival_idx(int rb, int col_idx, int col_blocks) {
    (void)col_idx;
    (void)col_blocks;
    return rb;
}

__device__ __host__ inline void gemm_ar_chunk_decode(
    int chunk_id,
    int row_blocks_per_slice,
    int col_blocks,
    int chunk_tiles,
    int dev_idx,
    int& rb_in_slice,
    int& row_idx,
    int& col_start,
    int& tiles_this_chunk
) {
    const int chunks_per_row = gemm_ar_chunks_per_row(col_blocks);
    rb_in_slice = chunk_id / chunks_per_row;
    const int chunk_col = chunk_id - rb_in_slice * chunks_per_row;
    row_idx = dev_idx * row_blocks_per_slice + rb_in_slice;
    col_start = chunk_col * chunk_tiles;
    tiles_this_chunk = min(chunk_tiles, col_blocks - col_start);
}

__device__ __host__ inline int gemm_ar_chunk_first_tile(
    int chunk_id, int row_blocks_per_slice, int col_blocks, int chunk_tiles
) {
    int rb_in_slice, row_idx, col_start, tiles_this_chunk;
    gemm_ar_chunk_decode(
        chunk_id, row_blocks_per_slice, col_blocks, chunk_tiles, 0,
        rb_in_slice, row_idx, col_start, tiles_this_chunk);
    (void)row_idx;
    (void)tiles_this_chunk;
    return rb_in_slice * col_blocks + col_start;
}

// GPU-scope acquire load for same-GPU progress counters. Sees all writes
// published by other CTAs on this device with at-least acquire ordering.
// Inline PTX with "memory" clobber so the compiler cannot CSE or cache
// the result across calls (unlike a plain `const uint32_t*` helper, which
// `comm::atomic_u32::acquire_load_gpu` / `release_store_gpu` come from
// include/comm/atomic_u32.cuh — single-instruction PTX wrappers shared with
// gemm_rs (and any future kernel that needs the same primitive). Aliased
// here so existing call sites keep working.
template <typename PtrT>
__device__ inline uint32_t gemm_ar_acquire_load_u32(PtrT* ptr) {
    return comm::atomic_u32::acquire_load_gpu(ptr);
}
template <typename PtrT>
__device__ inline void gemm_ar_release_store_u32(PtrT* ptr, uint32_t val) {
    comm::atomic_u32::release_store_gpu(ptr, val);
}

__device__ inline int gemm_ar_load_chunk_remote_arrived(const fused_globals& G, int chunk_id) {
    return (int)gemm_ar_acquire_load_u32(reinterpret_cast<unsigned int*>(G.chunk_remote_arrived + chunk_id));
}

__device__ inline uint32_t gemm_ar_load_arrival_word(volatile uint32_t* ptr) {
    return comm::atomic_u32::volatile_load(ptr);
}

__device__ inline uint32_t gemm_ar_load_arrival_queue_tail(const fused_globals& G, int q) {
    if (G.arrival_tails == nullptr) return G.arrival_queue_head[q];
    return gemm_ar_load_arrival_word(G.arrival_tails + q);
}

// Drain any new arrivals on a single queue and publish
// the per-chunk remote_arrived_flag (release-store) so static-ownership
// reducers can wait on it. Caller must be a single thread (uses
// non-atomic head advance). Returns true if any arrival was processed.
__device__ inline bool gemm_ar_drain_arrival_queue_publish_flags(
    const fused_globals& G, int q
) {
    uint32_t q_head = G.arrival_queue_head[q];
    // NOTE: arrival_tails is intentionally NOT used as the loop upper bound.
    // EFA SRD does not preserve WR ordering between independent RDMA writes
    // on the same QP, so a sender that posts (slot=0,tail=1) then
    // (slot=1,tail=2) can have those two tail-publish writes land in either
    // order at the receiver. Whichever lands second wins — if tail=1 lands
    // last, the receiver's cap stays at 1 forever and slot 1 is never
    // drained even though its flag word is correctly delivered. The proxy's
    // staging-cursor fix in proxy_efa.h ensures every flag word lands with
    // the correct payload, so a non-zero per-slot flag is now a fully
    // self-contained signal that the slot is ready. We keep arrival_tails
    // around only as a coarse "more-arrivals-pending" hint upstream.
    bool any = false;
    while ((int)q_head < G.remote_queue_stride) {
        const uint32_t flag_val = gemm_ar_load_arrival_word(
            (volatile uint32_t*)&G.arrival_flags[q * G.remote_queue_stride + q_head]);
        if (flag_val == 0u) break;
        G.arrival_queue_head[q] = q_head + 1u;
        q_head += 1u;
        (void)atomicAdd(G.queue_observed_arrivals + q, 1u);

        int first_tile = (int)internode::unpack_arrival_first_tile(flag_val);
        if (first_tile < 0) first_tile = 0;
        const int peer_slot = first_tile / G.total_tiles_per_device;
        const int local_first_tile = first_tile - peer_slot * G.total_tiles_per_device;
        const int rb = local_first_tile / G.col_blocks;
        int col_start = local_first_tile - rb * G.col_blocks;
        int work_tiles = (int)internode::unpack_arrival_num_tiles(flag_val);
        if (work_tiles < 1) work_tiles = 1;
        work_tiles = min(work_tiles, G.col_blocks - col_start);
        // Make remote payload visible before publishing the per-chunk flag.
        __threadfence_system();
        while (work_tiles > 0 && col_start < G.col_blocks) {
            const int chunk_id = rb * G.chunks_per_row + col_start / G.chunk_tiles;
            const int tiles_this_chunk = min(G.chunk_tiles, G.col_blocks - col_start);
            const int consumed_tiles = min(work_tiles, tiles_this_chunk);
            if (consumed_tiles <= 0) break;
            const uint32_t peer_bit =
                (peer_slot >= 0 && peer_slot < 31) ? (1u << peer_slot) : 0u;
            const uint32_t old_mask = atomicOr(
                reinterpret_cast<unsigned int*>(G.remote_arrived_peer_mask + chunk_id),
                peer_bit);
            const uint32_t new_mask = old_mask | peer_bit;
            const int needed_peers = G.num_nodes - 1;
            if (__popc(old_mask) < needed_peers && __popc(new_mask) >= needed_peers) {
                atomicAdd(G.remote_arrived_chunks, 1u);
                gemm_ar_release_store_u32(G.remote_arrived_flag + chunk_id, 1u);
            }
            col_start += consumed_tiles;
            work_tiles -= consumed_tiles;
        }
        any = true;
    }
    return any;
}

__device__ inline int gemm_ar_chunk_owner_queue(const fused_globals& G, int chunk_id) {
    int rb_in_slice, row_idx, col_start, tiles_this_chunk;
    gemm_ar_chunk_decode(
        chunk_id, G.row_blocks_per_slice, G.col_blocks, G.chunk_tiles, G.dev_idx,
        rb_in_slice, row_idx, col_start, tiles_this_chunk);
    (void)row_idx;
    (void)tiles_this_chunk;
    return gemm_ar_arrival_idx(rb_in_slice, col_start, G.col_blocks) % G.num_remote_queues;
}

__device__ inline int gemm_ar_send_logical_queue(
    const fused_globals& G, int rb_in_slice, int col_start
) {
    return gemm_ar_arrival_idx(rb_in_slice, col_start, G.col_blocks) % G.num_remote_queues;
}

__device__ inline bool gemm_ar_deadlock_debug_enabled(const fused_globals& G) {
    (void)G;
    return false;
}

__device__ inline unsigned long long gemm_ar_globaltimer() {
    return comm::globaltimer();
}


__device__ inline void gemm_ar_debug_log_transition(
    const fused_globals& G, const char* stage, int chunk_id, uint32_t counter
) {
    (void)G;
    (void)stage;
    (void)chunk_id;
    (void)counter;
}

__device__ inline void gemm_ar_debug_log_queue_overrun(
    const fused_globals& G, int queue_id, uint32_t observed, uint32_t expected, uint32_t head
) {
    (void)G;
    (void)queue_id;
    (void)observed;
    (void)expected;
    (void)head;
}

__device__ inline void gemm_ar_debug_maybe_dump_stuck(
    const fused_globals& G, const char* stage, int cta_id, int scan_cursor = -1, int scan_stride = -1
) {
    (void)G;
    (void)stage;
    (void)cta_id;
    (void)scan_cursor;
    (void)scan_stride;
}

__device__ inline dist::coord gemm_ar_final_slice_slot(int dev_idx, uint32_t epoch) {
    // Per-epoch row so back-to-back iters under GEMM_AR_STEADY_STATE_BENCH do not
    // reuse the same slot (otherwise, with no kernel-side reset, iter N leaves
    // value=1 and iter N+1's signal_all becomes 2 → wait_mc(==1) hangs).
    // epoch & 1023 gives 1024 distinct rows, matching the xbar per-epoch slot.
    return {1, (int)(epoch & 1023u), 32 + dev_idx};
}

__device__ inline void gemm_ar_publish_final_vec8(
    const fused_globals& G,
    int global_row,
    int global_col,
    const uint4& packed
) {
    // Compute all 4 multicast store addresses up front.
    bf16_2* ptr0 = reinterpret_cast<bf16_2*>(G.C_final.mc_ptr_at({0, 0, global_row, global_col + 0}));
    bf16_2* ptr1 = reinterpret_cast<bf16_2*>(G.C_final.mc_ptr_at({0, 0, global_row, global_col + 2}));
    bf16_2* ptr2 = reinterpret_cast<bf16_2*>(G.C_final.mc_ptr_at({0, 0, global_row, global_col + 4}));
    bf16_2* ptr3 = reinterpret_cast<bf16_2*>(G.C_final.mc_ptr_at({0, 0, global_row, global_col + 6}));

    // Issue all 4 multicast stores without "memory" clobber between them,
    // allowing the SM to pipeline NVSwitch round-trips (~600ns each).
    // Stores are to independent adjacent addresses — no ordering needed.
    // Cross-chunk ordering is handled by __threadfence() + published_chunks.
    // Same pattern as the pipelined intra-AR tile (gemm_ar_pipelined_ar_tile).
    comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(ptr0, packed.x);
    comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(ptr1, packed.y);
    comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(ptr2, packed.z);
    comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(ptr3, packed.w);
}

__device__ inline uint4 gemm_ar_sum_local_and_remote_vec8(
    const fused_globals& G,
    int tile_id,
    int tile_row,
    int col_elem,
    const uint4& local_vec
) {
    __nv_bfloat162 acc0 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.x);
    __nv_bfloat162 acc1 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.y);
    __nv_bfloat162 acc2 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.z);
    __nv_bfloat162 acc3 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.w);
    const int n_peers = G.num_nodes - 1;
    for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
        const bf16* recv = G.C_recv
            + ((long)peer_slot * G.total_tiles_per_device + tile_id)
                * fused_globals::TILE_ELEMS;
        const uint4 rv = *reinterpret_cast<const uint4*>(
            recv + tile_row * fused_globals::COL_BLOCK + col_elem);
        acc0 = __hadd2(acc0, *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
        acc1 = __hadd2(acc1, *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
        acc2 = __hadd2(acc2, *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
        acc3 = __hadd2(acc3, *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
    }
    uint4 out;
    out.x = *reinterpret_cast<unsigned int*>(&acc0);
    out.y = *reinterpret_cast<unsigned int*>(&acc1);
    out.z = *reinterpret_cast<unsigned int*>(&acc2);
    out.w = *reinterpret_cast<unsigned int*>(&acc3);
    return out;
}

__device__ inline uint4 gemm_ar_sum_local_and_accum_vec8(
    const fused_globals& G,
    int tile_id,
    int tile_row,
    int col_elem,
    const uint4& local_vec
) {
    __nv_bfloat162 acc0 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.x);
    __nv_bfloat162 acc1 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.y);
    __nv_bfloat162 acc2 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.z);
    __nv_bfloat162 acc3 = *reinterpret_cast<const __nv_bfloat162*>(&local_vec.w);
    const bf16* remote_accum = G.C_remote_accum + (long)tile_id * fused_globals::TILE_ELEMS;
    const uint4 rv = *reinterpret_cast<const uint4*>(
        remote_accum + tile_row * fused_globals::COL_BLOCK + col_elem);
    acc0 = __hadd2(acc0, *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
    acc1 = __hadd2(acc1, *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
    acc2 = __hadd2(acc2, *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
    acc3 = __hadd2(acc3, *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
    uint4 out;
    out.x = *reinterpret_cast<unsigned int*>(&acc0);
    out.y = *reinterpret_cast<unsigned int*>(&acc1);
    out.z = *reinterpret_cast<unsigned int*>(&acc2);
    out.w = *reinterpret_cast<unsigned int*>(&acc3);
    return out;
}

__device__ inline void gemm_ar_accumulate_remote_peer_vec8(
    const fused_globals& G,
    int peer_slot,
    int tile_id,
    int tile_row,
    int col_elem,
    bool first_peer_for_chunk
) {
    bf16* accum = G.C_remote_accum + (long)tile_id * fused_globals::TILE_ELEMS;
    const bf16* recv = G.C_recv
        + ((long)peer_slot * G.total_tiles_per_device + tile_id)
            * fused_globals::TILE_ELEMS;
    const uint4 rv = *reinterpret_cast<const uint4*>(
        recv + tile_row * fused_globals::COL_BLOCK + col_elem);
    uint4 out = rv;
    if (!first_peer_for_chunk) {
        const uint4 av = *reinterpret_cast<const uint4*>(
            accum + tile_row * fused_globals::COL_BLOCK + col_elem);
        __nv_bfloat162 acc0 = *reinterpret_cast<const __nv_bfloat162*>(&av.x);
        __nv_bfloat162 acc1 = *reinterpret_cast<const __nv_bfloat162*>(&av.y);
        __nv_bfloat162 acc2 = *reinterpret_cast<const __nv_bfloat162*>(&av.z);
        __nv_bfloat162 acc3 = *reinterpret_cast<const __nv_bfloat162*>(&av.w);
        acc0 = __hadd2(acc0, *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
        acc1 = __hadd2(acc1, *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
        acc2 = __hadd2(acc2, *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
        acc3 = __hadd2(acc3, *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
        out.x = *reinterpret_cast<unsigned int*>(&acc0);
        out.y = *reinterpret_cast<unsigned int*>(&acc1);
        out.z = *reinterpret_cast<unsigned int*>(&acc2);
        out.w = *reinterpret_cast<unsigned int*>(&acc3);
    }
    *reinterpret_cast<uint4*>(accum + tile_row * fused_globals::COL_BLOCK + col_elem) = out;
}


__device__ inline void gemm_ar_finish_multicast_publication(const fused_globals& G, int red_id) {
    if (threadIdx.x == 0) {
        if (red_id == 0) {
            while ((int)gemm_ar_acquire_load_u32(G.published_chunks) < G.total_chunks) {
                gemm_ar_debug_maybe_dump_stuck(G, "final_publish_wait", red_id);
                __nanosleep(50);
            }
            // Multicast-published stores must become visible before the
            // slice-complete signal advertises that this replica is ready.
            __threadfence_system();
            signal_all(G.barrier, gemm_ar_final_slice_slot(G.dev_idx, G.epoch), 1);
            for (int peer = 0; peer < fused_globals::NUM_DEVICES; ++peer) {
                wait_mc(G.barrier, gemm_ar_final_slice_slot(peer, G.epoch), 1);
            }
            atomicExch(G.final_publish_done, 1u);
        } else {
            while (gemm_ar_acquire_load_u32(G.final_publish_done) == 0u) {
                __nanosleep(50);
            }
        }
    }
    __syncthreads();
}

__device__ inline void gemm_ar_decode_comp_task(
    int task_id, int row_blocks_per_slice, int col_blocks, int dev_idx, int& row_idx, int& col_idx
) {
    slice_interleaved_slices_decode(
        task_id, row_blocks_per_slice, col_blocks, fused_globals::NUM_DEVICES, row_idx, col_idx);
}

// ============================================================================
// Role 1: GEMM compute SM — produce C tiles, signal per-tile barrier
// ============================================================================

__device__ inline int gemm_ar_inter_tail_base(const fused_globals& G) {
    return G.num_comp_sms + G.num_intra_ar_sms + G.num_inter_send_sms;
}

__device__ inline int gemm_ar_inter_reduce_store_base(const fused_globals& G) {
    return gemm_ar_inter_tail_base(G) + G.num_inter_recv_progress_sms;
}

// ============================================================================
// Role 4: Inter-node reduce + multicast publication SM
// Claim remote-arrived chunks, publish them, then announce slice completion.
// ============================================================================

// Shared host-side layout/debug helpers for multinode GEMM+AR.
// Defined inside the gemm_ar_multinode namespace; instantiated in src/gemm_ar.cu.

__host__ inline bool gemm_ar_host_debug_enabled() {
    return false;
}

__host__ inline void gemm_ar_host_debug_log(int node_idx, int dev_idx, const char* msg) {
    (void)node_idx;
    (void)dev_idx;
    (void)msg;
}

struct gemm_ar_role_split {
    int num_comp_sms;
    int num_intra_ar_sms;
    int num_inter_send_sms;
    int num_inter_reduce_publish_sms;
    int num_inter_recv_progress_sms;
    int num_inter_reduce_store_sms;
};

struct gemm_ar_scratch_layout {
    int slice_rows;
    int row_blocks_per_slice;
    int col_blocks;
    int total_chunks;
    int total_tiles;
    int chunk_tiles;
    int chunks_per_row;
    int chunk_remote_arrived_offset;
    int arrival_queue_head_offset;
    int local_ar_done_offset;
    int send_issued_offset;
    int remote_arrived_offset;
    int published_offset;
    int final_publish_done_offset;
    int debug_flags_offset;
    int debug_dump_counter_offset;
    int queue_expected_offset;
    int queue_observed_offset;
    int row_send_count_offset;
    // gemm_ready_queue scratch
    int gemm_tiles_ready_count_offset;
    int gemm_tile_scanned_offset;
    int gemm_rq_entries_offset;
    int gemm_rq_head_offset;
    int gemm_rq_tail_offset;
    int gemm_scan_cursor_offset;
    int gemm_rq_capacity;
    int local_ar_rq_entries_offset;
    int local_ar_rq_head_offset;
    int local_ar_rq_tail_offset;
    int local_ar_rq_capacity;
    int remote_rq_entries_offset;
    int remote_rq_head_offset;
    int remote_rq_tail_offset;
    int remote_rq_capacity;
    int owner_pending_entries_offset;
    int owner_pending_head_offset;
    int owner_pending_tail_offset;
    int owner_pending_max_depth_offset;
    int owner_pending_push_count_offset;
    int owner_pending_pop_count_offset;
    int owner_pending_capacity;
    int slice_tiles;
    // Static-ownership flag arrays (u32, one entry per chunk)
    int local_done_flag_offset;
    int remote_arrived_flag_offset;
    int remote_arrived_peer_mask_offset;
    // GEMM_AR_STAGGER_INTRA: per-primary-CTA started signal (max 16 entries).
    // Primary intra CTAs (offset < num_intra_ar_sms/2) set these after passing
    // their first tile barrier; secondary CTAs poll these (L2-cached, no NVSwitch).
    int intra_started_flag_offset;
    // GEMM_AR_INTER_REDUCE_WORKSTEAL: cursor-based work-stealing reduce.
    int reduce_cursor_offset;        // 1 u32
    int chunk_claimed_flag_offset;   // total_chunks u32 entries
    // Per-chunk tile-completion counters (always allocated even without
    // GEMM_AR_CHUNK_BARRIER — intra_chunk_tiles_done is read unconditionally by
    // the intra_ar role at fused_intra_ar_sm's last-tile commit).
    int comp_chunk_tiles_done_offset;  // row_blocks * chunks_per_row u32 entries
    int intra_chunk_tiles_done_offset; // total_chunks u32 entries
    int xnode_ready_offset;            // 1 u32 (GEMM_AR_EPILOGUE_CROSS_NODE_BARRIER)
    int scratch_ints_needed;
    int remote_queue_stride;
};


__host__ inline gemm_ar_role_split gemm_ar_compute_role_split(
    int num_intra_comm_sms, int num_inter_comm_sms, int row_blocks_per_slice = 0
) {
    gemm_ar_role_split split{};
    split.num_intra_ar_sms = num_intra_comm_sms;
    int requested_send_sms = 0;
    const char* send_env = std::getenv("GEMM_AR_INTER_SEND_SMS");
    if (send_env != nullptr && send_env[0] != '\0') {
        requested_send_sms = std::atoi(send_env);
    }
#ifdef GEMM_AR_INTER_SEND_SMS
    if (requested_send_sms <= 0) requested_send_sms = (int)GEMM_AR_INTER_SEND_SMS;
#else
    if (requested_send_sms <= 0) requested_send_sms = num_inter_comm_sms / 2;
#endif
    requested_send_sms = max(1, std::min(num_inter_comm_sms - 1, requested_send_sms));
    if (row_blocks_per_slice > 0) {
        requested_send_sms = std::min(requested_send_sms, row_blocks_per_slice);
    }
    split.num_inter_send_sms = requested_send_sms;
    int requested_recv_progress_sms = 0;
    const char* recv_env = std::getenv("GEMM_AR_RECV_PROGRESS_SMS");
    if (recv_env != nullptr && recv_env[0] != '\0') {
        requested_recv_progress_sms = std::atoi(recv_env);
    }
    const int remaining_after_send = num_inter_comm_sms - split.num_inter_send_sms;
    if (requested_recv_progress_sms > 0) {
        requested_recv_progress_sms =
            std::min(requested_recv_progress_sms, std::max(0, config::NUM_BLOCKS
                - num_intra_comm_sms - num_inter_comm_sms));
    } else {
        requested_recv_progress_sms = 0;
    }
    split.num_inter_recv_progress_sms = requested_recv_progress_sms;
    split.num_inter_reduce_store_sms = remaining_after_send;
    split.num_inter_reduce_publish_sms = split.num_inter_reduce_store_sms;
    split.num_comp_sms = config::NUM_BLOCKS - num_intra_comm_sms
                       - split.num_inter_send_sms - split.num_inter_recv_progress_sms
                       - split.num_inter_reduce_publish_sms;
    return split;
}

__host__ inline gemm_ar_scratch_layout gemm_ar_compute_scratch_layout(
    int M, int N, int num_remote_queues, int num_allocated_remote_queues,
    int num_nodes
) {
    gemm_ar_scratch_layout scratch{};
    scratch.slice_rows = M / fused_globals::NUM_DEVICES;
    scratch.row_blocks_per_slice = scratch.slice_rows / fused_globals::ROW_BLOCK;
    scratch.col_blocks = N / fused_globals::COL_BLOCK;
    scratch.total_chunks = scratch.row_blocks_per_slice * gemm_ar_chunks_per_row(scratch.col_blocks);
    scratch.total_tiles = (M / fused_globals::ROW_BLOCK) * scratch.col_blocks;
    scratch.chunk_tiles = gemm_ar_chunk_tiles(scratch.col_blocks);
    scratch.chunks_per_row = gemm_ar_chunks_per_row(scratch.col_blocks);
    scratch.chunk_remote_arrived_offset = 0;
    scratch.arrival_queue_head_offset = scratch.chunk_remote_arrived_offset + scratch.total_chunks;
    scratch.local_ar_done_offset = scratch.arrival_queue_head_offset + num_remote_queues;
    scratch.send_issued_offset = scratch.local_ar_done_offset + 1;
    scratch.remote_arrived_offset = scratch.send_issued_offset + 1;
    scratch.published_offset = scratch.remote_arrived_offset + 1;
    scratch.final_publish_done_offset = scratch.published_offset + 1;
    scratch.debug_flags_offset = scratch.final_publish_done_offset + 1;
    scratch.debug_dump_counter_offset = scratch.debug_flags_offset + 1;
    scratch.queue_expected_offset = scratch.debug_dump_counter_offset + 1;
    scratch.queue_observed_offset = scratch.queue_expected_offset + num_remote_queues;
    scratch.row_send_count_offset = scratch.queue_observed_offset + num_remote_queues;
    scratch.slice_tiles = scratch.row_blocks_per_slice * scratch.col_blocks;
    scratch.gemm_tiles_ready_count_offset =
        scratch.row_send_count_offset + scratch.row_blocks_per_slice;
    scratch.gemm_tile_scanned_offset = scratch.gemm_tiles_ready_count_offset + scratch.total_chunks;
    scratch.gemm_rq_entries_offset = scratch.gemm_tile_scanned_offset + scratch.slice_tiles;
    {
        int cap = std::max(16, scratch.total_chunks);
        int v = cap - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        scratch.gemm_rq_capacity = v + 1;
    }
    scratch.gemm_rq_head_offset = scratch.gemm_rq_entries_offset + scratch.gemm_rq_capacity;
    scratch.gemm_rq_tail_offset = scratch.gemm_rq_head_offset + 1;
    scratch.gemm_scan_cursor_offset = scratch.gemm_rq_tail_offset + 1;
    scratch.local_ar_rq_capacity = scratch.gemm_rq_capacity;
    scratch.local_ar_rq_entries_offset = scratch.gemm_scan_cursor_offset + 1;
    scratch.local_ar_rq_head_offset =
        scratch.local_ar_rq_entries_offset + scratch.local_ar_rq_capacity;
    scratch.local_ar_rq_tail_offset = scratch.local_ar_rq_head_offset + 1;
    scratch.remote_rq_capacity = scratch.gemm_rq_capacity;
    scratch.remote_rq_entries_offset = scratch.local_ar_rq_tail_offset + 1;
    scratch.remote_rq_head_offset =
        scratch.remote_rq_entries_offset + scratch.remote_rq_capacity;
    scratch.remote_rq_tail_offset = scratch.remote_rq_head_offset + 1;
    {
        int cap = std::max(8, scratch.chunks_per_row * 2);
        int v = cap - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        scratch.owner_pending_capacity = v + 1;
    }
    scratch.owner_pending_entries_offset = scratch.remote_rq_tail_offset + 1;
    scratch.owner_pending_head_offset =
        scratch.owner_pending_entries_offset + num_remote_queues * scratch.owner_pending_capacity;
    scratch.owner_pending_tail_offset =
        scratch.owner_pending_head_offset + num_remote_queues;
    scratch.owner_pending_max_depth_offset =
        scratch.owner_pending_tail_offset + num_remote_queues;
    scratch.owner_pending_push_count_offset =
        scratch.owner_pending_max_depth_offset + num_remote_queues;
    scratch.owner_pending_pop_count_offset =
        scratch.owner_pending_push_count_offset + num_remote_queues;
    // Static-ownership flag arrays: one u32 per chunk each.
    scratch.local_done_flag_offset =
        scratch.owner_pending_pop_count_offset + num_remote_queues;
    scratch.remote_arrived_flag_offset =
        scratch.local_done_flag_offset + scratch.total_chunks;
    scratch.remote_arrived_peer_mask_offset =
        scratch.remote_arrived_flag_offset + scratch.total_chunks;
    // GEMM_AR_STAGGER_INTRA: up to 16 u32 entries (one per primary intra CTA,
    // max stride/2=16 — see fused_prelude.cuh:111). Allocate 16 to match the
    // Python-side _compute_scratch_ints_needed layout exactly.
    scratch.intra_started_flag_offset =
        scratch.remote_arrived_peer_mask_offset + scratch.total_chunks;
    // GEMM_AR_INTER_REDUCE_WORKSTEAL: cursor (1 u32) + per-chunk claimed flags.
    // Always allocated regardless of compile flag for stable scratch sizing.
    scratch.reduce_cursor_offset =
        scratch.intra_started_flag_offset + 16;
    scratch.chunk_claimed_flag_offset =
        scratch.reduce_cursor_offset + 1;
    // GEMM_AR_CHUNK_BARRIER / steady-state: per-chunk tile completion counters.
    // comp_chunk_tiles_done is sized across ALL row_blocks (not just this
    // device's slice) because compute produces tiles for every slice.
    // intra_chunk_tiles_done is accessed unconditionally by the intra_ar role
    // even when GEMM_AR_CHUNK_BARRIER is not defined at compile time — do not
    // gate on any macro here.
    const int row_blocks = M / fused_globals::ROW_BLOCK;
    const int total_all_chunks = row_blocks * scratch.chunks_per_row;
    scratch.comp_chunk_tiles_done_offset =
        scratch.chunk_claimed_flag_offset + scratch.total_chunks;
    scratch.intra_chunk_tiles_done_offset =
        scratch.comp_chunk_tiles_done_offset + total_all_chunks;
    scratch.xnode_ready_offset =
        scratch.intra_chunk_tiles_done_offset + scratch.total_chunks;
    scratch.scratch_ints_needed =
        scratch.xnode_ready_offset + 1;
    const int n_peers = std::max(1, num_nodes - 1);
    scratch.remote_queue_stride =
        std::max(1, (scratch.total_tiles * n_peers + num_allocated_remote_queues - 1)
                     / num_allocated_remote_queues);
    return scratch;
}

__host__ inline bool gemm_ar_deadlock_debug_enabled_host() {
    const char* env = std::getenv("MKERNEL_GEMM_AR_DEADLOCK_DEBUG");
    return env != nullptr && env[0] == '1';
}

__host__ inline void gemm_ar_init_debug_scratch(
    int64_t ar_done_ptr, const gemm_ar_scratch_layout& scratch, int num_remote_queues
) {
    std::vector<uint32_t> host_tail(
        scratch.scratch_ints_needed - scratch.remote_arrived_offset, 0u);
    host_tail[scratch.debug_flags_offset - scratch.remote_arrived_offset] =
        gemm_ar_deadlock_debug_enabled_host() ? 1u : 0u;
    for (int q = 0; q < num_remote_queues; ++q) {
        host_tail[scratch.queue_expected_offset - scratch.remote_arrived_offset + q] =
            (uint32_t)gemm_ar_units_for_queue(scratch.total_chunks, q, num_remote_queues);
    }
    cudaMemcpy(
        reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_arrived_offset,
        host_tail.data(),
        host_tail.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);
}
// Host-side globals construction and fused entrypoint.
// Defined inside the gemm_ar_multinode namespace; instantiated in src/gemm_ar.cu.

__host__ inline fused_globals gemm_ar_make_globals(
    const at::Tensor& A, const at::Tensor& B,
    dist::ParallelBuffer& C,
    dist::ParallelBuffer& barrier,
    dist::ParallelBuffer& C_final,
    int64_t staging_buf_ptr, int64_t recv_buf_ptr,
    const internode::D2HFifoDeviceBundle& fifo_bundle,
    int64_t remote_accum_ptr,
    int64_t arrival_flags_ptr, int64_t arrival_tails_ptr, int epoch, int node_idx,
    int64_t ar_done_ptr,
    int num_qps, int num_remote_queues,
    const gemm_ar_role_split& split,
    const gemm_ar_scratch_layout& scratch,
    int64_t cross_node_barrier_ptr,
    int num_nodes
) {
    fused_globals G{
        .A = ::dist::local_tensor_from_tensor<fused_globals::A_local_tensor>(A),
        .B = ::dist::local_tensor_from_tensor<fused_globals::B_local_tensor>(B),
        .C = ::dist::distributed_tensor_from_buffer<fused_globals::C_distributed_tensor>(C),
        .C_final = ::dist::distributed_tensor_from_buffer<fused_globals::final_distributed_tensor>(C_final),
        .barrier = ::dist::distributed_tensor_from_buffer<fused_globals::barrier_distributed_tensor>(barrier),
        .C_local = reinterpret_cast<bf16*>(C.data_.data_ptr()),
        .C_recv = reinterpret_cast<bf16*>(recv_buf_ptr),
        .staging_buf = reinterpret_cast<bf16*>(staging_buf_ptr),
        .C_remote_accum = reinterpret_cast<bf16*>(remote_accum_ptr),
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .arrival_tails = reinterpret_cast<volatile uint32_t*>(arrival_tails_ptr),
        .epoch = (uint32_t)epoch,
        .cross_node_barrier = cross_node_barrier_ptr
            ? reinterpret_cast<volatile uint32_t*>(cross_node_barrier_ptr) : nullptr,
        .ar_done = reinterpret_cast<int*>(ar_done_ptr),
        .chunk_remote_arrived =
            reinterpret_cast<int*>(ar_done_ptr) + scratch.chunk_remote_arrived_offset,
        .arrival_queue_head =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.arrival_queue_head_offset,
        .local_ar_done_chunks =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.local_ar_done_offset,
        .send_issued_chunks =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.send_issued_offset,
        .remote_arrived_chunks =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_arrived_offset,
        .published_chunks =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.published_offset,
        .final_publish_done =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.final_publish_done_offset,
        .debug_flags =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.debug_flags_offset,
        .debug_dump_counter =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.debug_dump_counter_offset,
        .queue_expected_arrivals =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.queue_expected_offset,
        .queue_observed_arrivals =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.queue_observed_offset,
        .row_send_count =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.row_send_count_offset,
        .gemm_tiles_ready_count =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.gemm_tiles_ready_count_offset,
        .gemm_tile_scanned =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.gemm_tile_scanned_offset,
        .gemm_rq_entries =
            reinterpret_cast<int*>(ar_done_ptr) + scratch.gemm_rq_entries_offset,
        .gemm_rq_head =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.gemm_rq_head_offset,
        .gemm_rq_tail =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.gemm_rq_tail_offset,
        .gemm_scan_cursor =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.gemm_scan_cursor_offset,
        .gemm_rq_capacity = scratch.gemm_rq_capacity,
        .local_ar_rq_entries =
            reinterpret_cast<int*>(ar_done_ptr) + scratch.local_ar_rq_entries_offset,
        .local_ar_rq_head =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.local_ar_rq_head_offset,
        .local_ar_rq_tail =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.local_ar_rq_tail_offset,
        .local_ar_rq_capacity = scratch.local_ar_rq_capacity,
        .remote_rq_entries =
            reinterpret_cast<int*>(ar_done_ptr) + scratch.remote_rq_entries_offset,
        .remote_rq_head =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_rq_head_offset,
        .remote_rq_tail =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_rq_tail_offset,
        .remote_rq_capacity = scratch.remote_rq_capacity,
        .recv_ready_queue_enabled = 0,
        .owner_pending_entries =
            reinterpret_cast<int*>(ar_done_ptr) + scratch.owner_pending_entries_offset,
        .owner_pending_head =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.owner_pending_head_offset,
        .owner_pending_tail =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.owner_pending_tail_offset,
        .owner_pending_max_depth =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.owner_pending_max_depth_offset,
        .owner_pending_push_count =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.owner_pending_push_count_offset,
        .owner_pending_pop_count =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.owner_pending_pop_count_offset,
        .owner_pending_capacity = scratch.owner_pending_capacity,
        .local_done_flag =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.local_done_flag_offset,
        .remote_arrived_flag =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_arrived_flag_offset,
        .remote_arrived_peer_mask =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.remote_arrived_peer_mask_offset,
        .intra_started_flag =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.intra_started_flag_offset,
        .reduce_cursor =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.reduce_cursor_offset,
        .chunk_claimed_flag =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.chunk_claimed_flag_offset,
        .N = static_cast<int>(B.size(1)),
        .dev_idx = C.local_rank_,
        .node_idx = node_idx,
        .num_nodes = num_nodes,
        .slice_rows = scratch.slice_rows,
        .row_blocks_per_slice = scratch.row_blocks_per_slice,
        .col_blocks = scratch.col_blocks,
        .num_comp_sms = split.num_comp_sms,
        .num_intra_ar_sms = split.num_intra_ar_sms,
        .num_inter_send_sms = split.num_inter_send_sms,
        .num_inter_reduce_publish_sms = split.num_inter_reduce_publish_sms,
        .num_inter_recv_progress_sms = split.num_inter_recv_progress_sms,
        .num_inter_reduce_store_sms = split.num_inter_reduce_store_sms,
        .num_qps = num_qps,
        .num_remote_queues = num_remote_queues,
        .remote_queue_stride = scratch.remote_queue_stride,
        .defer_final_multicast_finish = 0,
        .work_steal_enabled = 0,
        .intra_ready_multimem = 0,
        .total_chunks = scratch.total_chunks,
        .total_tiles_per_device = scratch.slice_tiles,
        .chunk_tiles = scratch.chunk_tiles,
        .chunks_per_row = scratch.chunks_per_row,
        // intra_chunk_tiles_done is read unconditionally by the intra_ar role
        // (fused_intra_ar_sm: atomicAdd(G.intra_chunk_tiles_done + chunk_id, 1u)),
        // so it must be assigned here even when GEMM_AR_CHUNK_BARRIER is not defined.
        .comp_chunk_tiles_done =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.comp_chunk_tiles_done_offset,
        .intra_chunk_tiles_done =
            reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.intra_chunk_tiles_done_offset,
        .epilogue_blocks_done = nullptr,
    };
    G.xnode_ready_device =
        reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.xnode_ready_offset;
    G.total_tiles_per_device = scratch.slice_tiles;
    G.comp_chunk_tiles_done =
        reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.comp_chunk_tiles_done_offset;
    G.intra_chunk_tiles_done =
        reinterpret_cast<uint32_t*>(ar_done_ptr) + scratch.intra_chunk_tiles_done_offset;
    if (const char* env_multimem = std::getenv("GEMM_AR_INTRA_READY_MULTIMEM")) {
        G.intra_ready_multimem =
            (env_multimem[0] != '\0' && env_multimem[0] != '0') ? 1 : 0;
    }
    return G;
}

// ============================================================================
// Host entrypoint
// ============================================================================

void entrypoint(
    const at::Tensor& A, const at::Tensor& B,
    dist::ParallelBuffer& C,
    dist::ParallelBuffer& barrier,
    dist::ParallelBuffer& C_final,
    int64_t staging_buf_ptr, int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr, int epoch, int node_idx,
    int num_intra_comm_sms, int num_inter_comm_sms,
    int64_t ar_done_ptr,
    int num_nodes,
    int64_t arrival_tails_ptr = 0,
    int scratch_ints = 0,
    int num_qps = 1,
    int num_remote_queues = 1,
    int num_allocated_remote_queues = 1,
    // Push model (GEMM_AR_PUSH_MODEL): ready queue handles. Ignored when not compiled
    // with -DGEMM_AR_PUSH_MODEL. All default to 0 for backward compatibility.
    int64_t rq_entries_ptr = 0, int64_t rq_tail_ptr = 0,
    int64_t rq_head_ptr = 0, int rq_capacity = 0, int rq_total = 0,
    int64_t cross_node_barrier_ptr = 0,
    int trace_slot = -1,
    // Select acquire vs relaxed loads for the intra-AR xdev barrier wait.
    bool use_acquire_poll = false,
    int64_t remote_accum_ptr = 0
) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    if (num_qps <= 0) num_qps = 1;
    if (num_remote_queues <= 0) num_remote_queues = 1;
    if (num_allocated_remote_queues <= 0) num_allocated_remote_queues = num_remote_queues;
    if (num_allocated_remote_queues < num_remote_queues) num_allocated_remote_queues = num_remote_queues;

    const gemm_ar_scratch_layout scratch =
        gemm_ar_compute_scratch_layout(
            M, N, num_remote_queues, num_allocated_remote_queues, num_nodes);
    const gemm_ar_role_split split =
        gemm_ar_compute_role_split(
            num_intra_comm_sms, num_inter_comm_sms, scratch.row_blocks_per_slice);
    const int num_inter_reduce_publish = split.num_inter_reduce_publish_sms;
    const int num_inter_reduce_store = split.num_inter_reduce_store_sms;
    const int num_comp_sms = split.num_comp_sms;
    TORCH_CHECK(num_comp_sms > 0, "num_comp_sms must be > 0");
    TORCH_CHECK(num_inter_reduce_publish > 0, "num_inter_reduce_publish must be > 0");
    TORCH_CHECK(num_inter_reduce_store > 0, "num_inter_reduce_store must be > 0");
    // scratch_ints == 0 means caller did not pass an explicit size; fall back
    // to total_tiles. Otherwise use the actual provided buffer size.
    const int ar_done_buf_size = (scratch_ints > 0) ? scratch_ints : scratch.total_tiles;
    TORCH_CHECK(
        scratch.scratch_ints_needed <= ar_done_buf_size,
        "ar_done scratch too small for chunk state layout: need ",
        scratch.scratch_ints_needed, " ints but got ", ar_done_buf_size);
    gemm_ar_init_debug_scratch(ar_done_ptr, scratch, num_remote_queues);

    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity,
        num_qps, 1, nullptr, "multi-proxy FIFO bundle pointer is null");

    fused_globals G = gemm_ar_make_globals(
        A, B, C, barrier, C_final,
        staging_buf_ptr, recv_buf_ptr, fifo_bundle, remote_accum_ptr,
        arrival_flags_ptr, arrival_tails_ptr, epoch, node_idx, ar_done_ptr,
        num_qps, num_remote_queues, split, scratch,
        cross_node_barrier_ptr, num_nodes);
    // Small shapes: finish multicast in the main kernel to avoid the extra
    // epilogue launch overhead. For larger shapes, keep the epilogue path.
    const bool need_epilogue = (scratch.total_chunks > 16);
    G.defer_final_multicast_finish = need_epilogue ? 1 : 0;
    {
        const char* ws_env = std::getenv("GEMM_AR_WORK_STEAL");
        G.work_steal_enabled = (ws_env != nullptr && ws_env[0] == '1') ? 1 : 0;
    }
    {
        const char* r8_env = std::getenv("GEMM_AR_R8_WARP_SPEC");
        G.r8_warp_spec = (r8_env != nullptr && r8_env[0] == '1') ? 1u : 0u;
    }
    G.use_acquire_poll = use_acquire_poll;
    {
        const char* accum_env = std::getenv("GEMM_AR_EARLY_REMOTE_ACCUM");
        G.early_remote_accum =
            (remote_accum_ptr != 0 && accum_env != nullptr && accum_env[0] == '1') ? 1 : 0;
    }
    gemm_ar_host_debug_log(node_idx, dev_idx, "entrypoint before launch_kernel");
    launch_fused_gemm_ar(G);
    gemm_ar_host_debug_log(node_idx, dev_idx, "entrypoint after launch_kernel");
    if (need_epilogue) {
        gemm_ar_host_debug_log(node_idx, dev_idx, "entrypoint before launch_epilogue_kernel");
        launch_fused_gemm_ar_epilogue(G);
        gemm_ar_host_debug_log(node_idx, dev_idx, "entrypoint after launch_epilogue_kernel");
    }
}

}  // namespace gemm_ar_multinode
