#pragma once

/**
 * @file ag_gemm_multinode.cu
 * @brief Multi-node All-Gather + GEMM — truly fused single-kernel (multi-node).
 *
 * Single kernel launch. Two CTA groups run concurrently:
 *
 *   Intra-comm CTAs [0, num_intra_comm):
 *     IPC multicast gather of A shards within the node.
 *     Also post zero-copy RDMA WRs for this rank's M_local slice at kernel
 *     entry (src_view=1 → DMA-BUF MR aliasing A.data_), so the NIC pipelines
 *     the inter-node send concurrently with intra-gather + compute.
 *     Signals barrier[row_block] when each row block is fully gathered.
 *
 *   Comp CTAs [num_intra_comm, 132):
 *     GEMM on all M rows. Atomically claim tiles:
 *       - Local-half tile: wait on intra-node barrier, TMA load from A_distributed_tensor
 *       - Remote-half tile: wait on arrival_flags, TMA load from A_recv_local_tensor
 *
 * The distributed A buffer is DMA-BUF-registered for RDMA (no staging copy).
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "common/cuda_checks.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "comm/comm.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace ag_gemm_multinode {

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

// 64 KB chunks — tuned via EFA sweep. Larger than DeepEP's per-token
// strategy because EFA SRD per-WR cost is high enough that larger chunks
// amortize it well (see also dispatch_gemm.cu's 512 KB choice).
static constexpr int CHUNK_BYTES = 64 * 1024;

struct comp_task {
    int rb;
    int col_idx;
    bool is_remote;
};


// ============================================================================
// Fused globals
// ============================================================================

struct globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    // Three stages keep the producer/consumer pipeline deep enough while
    // leaving more shared memory headroom than a four-stage pipeline.
    static constexpr int PIPELINE_STAGES = 3;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using A_comm_tile = st_bf<ROW_BLOCK * 2, RED_BLOCK * 2>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    static constexpr int NUM_COMM_CHUNKS = config::DYNAMIC_SHARED_MEMORY / sizeof(A_comm_tile);

    // Intra-node IPC
    using A_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, 0, 1, A_comm_tile>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<NUM_DEVICES>;

    A_distributed_tensor A;
    barrier_distributed_tensor barrier;

    // Inter-node RDMA + compute. A_local_tensor carries both A_tile (compute loads) and
    // A_comm_tile (phase-2 receive fan-out under #8) so TMA descriptors for both
    // are available on recv_buf / A_local without creating separate GL types.
    using A_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>;
    using B_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>;

    A_local_tensor A_local;
    A_local_tensor A_recv_local_tensor;      // per-rank unicast view of recv_buf (RDMA landing zone)
    // Plan federated-weaving-ocean #8: multicast-backed A_recv buffer. Each
    // rank r on node N publishes the 1/NUM_DEVICES slice of peer A_half it
    // received via RDMA into this dbuf, so all ranks on node N see the full
    // peer A_half after phase-2. Compute remote-tile loads read from this.
    A_distributed_tensor A_recv;
    B_local_tensor B;
    C_local_tensor C;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t*       arrival_flags;
    uint32_t                 epoch;
    int                      total_chunks;
    uint64_t                 a_half_bytes;

    const int dev_idx;
    const int node_idx;
    // Total node count (>= 2). For N > 2, recv/A_recv are laid out as
    // (num_nodes - 1) peer slots in local peer-slot order.
    const int num_nodes;
    // 0 = direct fanout, 1 = next-hop ring all-gather.
    const int collective_mode;
    // Ring mode stages each hop in a separate recv/arrival bank so a forward
    // cannot RDMA-overwrite a slot another GPU is still TMA-reading (intra-node
    // sync does not order remote sends). Banks = (num_nodes-1); direct uses 1.
    const int ring_recv_banks;
    const int debug_skip_remote_compute;
    const int debug_skip_phase1;
    const int debug_skip_phase1_gate;
    const int debug_skip_phase2;
    const int debug_skip_compute;
    const int debug_skip_reset;
    const int ring_proxy_forward;
    const int remote_ready_per_col;
    const int tiled_direct;
    const int num_intra_comm;  // CTAs for intra-node IPC gather + RDMA push
    const int num_comp_sms;    // CTAs for GEMM compute

    enum activity_kind : int {
        ACTIVITY_LOCAL_GATHER = 0,
        ACTIVITY_REMOTE_PUBLISH = 1,
        ACTIVITY_COMPUTE_LOCAL = 2,
        ACTIVITY_COMPUTE_REMOTE = 3,
    };
    struct activity_event {
        unsigned long long start_ns;
        unsigned long long end_ns;
        int work_id;
        int kind;
    };
    activity_event* activity_buf = nullptr;
    uint32_t* activity_counts = nullptr;
    unsigned long long* kernel_start_ns = nullptr;
    unsigned long long* kernel_end_ns = nullptr;
    int activity_max_events = 0;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

__device__ inline unsigned long long ag_gemm_globaltimer() {
    return comm::globaltimer();
}

__device__ inline void ag_gemm_record_activity_event(
    const globals& G, int kind, int work_id,
    unsigned long long start_ns, unsigned long long end_ns
) {
    if (G.activity_buf == nullptr || G.activity_counts == nullptr) return;
    uint32_t idx = atomicAdd(&G.activity_counts[blockIdx.x], 1u);
    if (idx < (uint32_t)G.activity_max_events) {
        auto& ev = G.activity_buf[
            (size_t)blockIdx.x * (size_t)G.activity_max_events + (size_t)idx];
        ev.start_ns = start_ns;
        ev.end_ns = end_ns;
        ev.work_id = work_id;
        ev.kind = kind;
    }
}

// Wait for arrival_flags[chunk_id] == G.epoch. Single helper used at every
// consumer-CTA poll site so PTX scope and inline RX-CQ poll integration stay
// in one place.
//
// Under proxy: just spin on a volatile load (host-mapped memory or HBM
// remote-NIC writes). Under inline-poll backends: also poll the local RX
// CQ each iteration; on RX-IMM CQE the helper publishes the imm into
// arrival_flags via st.release.gpu.global. Reader uses .gpu scope (writer
// is a local CTA on same GPU L2).
__device__ __forceinline__ void ag_gemm_wait_arrival_one(const globals& G, int slot_ck) {
    uint32_t v;
    do {
        v = comm::atomic_u32::volatile_load(&G.arrival_flags[slot_ck]);
        if (v == G.epoch) break;
        __nanosleep(100);
    } while (true);
}

__device__ __forceinline__ void ag_gemm_wait_arrival_slot(
    const globals& G, int peer_slot, int ck
) {
    ag_gemm_wait_arrival_one(G, peer_slot * G.total_chunks + ck);
}

__device__ __forceinline__ int ag_gemm_ring_origin_for_step(
    int node_idx, int num_nodes, int step
) {
    int origin = node_idx - 1 - step;
    while (origin < 0) origin += num_nodes;
    return origin;
}


// ============================================================================
// Intra-comm SM: IPC multicast gather + signal intra_done per row block
// ============================================================================

// Split large row-block sends into roughly 2 MB sub-WRs, capped by `ceiling`.
__device__ __forceinline__ int ag1_compute_wr_split(int rb_bytes, int ceiling) {
    // Default knee at 2 MB. AG1_WR_SPLIT_KNEE_BYTES could override at compile
    // time if tuning is needed.
    constexpr int KNEE_BYTES = 2 * 1024 * 1024;
    int s = rb_bytes / KNEE_BYTES;
    if (s < 1) s = 1;
    if (s > ceiling) s = ceiling;
    // chunks_per_rb must be divisible by s for clean flag splitting; caller
    // guarantees chunks_per_rb is power-of-2 at k16 shapes so s in {1,2,4}
    // always divides. If a weird shape breaks this, caller falls back to 1.
    return s;
}

// Merge: post both inter rbs' DMA-BUF WRs for a single intra row that this
// rank owns (intra 256-row unit = 2 inter 128-row rbs).
//
// Under AG1_MERGE_WR_SPLIT (ceiling N>1), each inter-rb is split into
// effective_split sub-WRs sized rb_bytes/split, striped across QPs by
// (rb*split + sw) key. effective_split computed from rb_bytes — only kicks
// in at large shapes where sub-WR stays >= 2 MB.
__device__ inline void post_merge_wrs_for_intra_row(
    const globals& G, int global_row_idx, int chunks_per_rb) {
    constexpr int WR_SPLIT_CEILING = 1;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int rb = 2 * global_row_idx + sub;
        int first_chunk = rb * chunks_per_rb;
        int last_chunk = min(first_chunk + chunks_per_rb, G.total_chunks);
        uint32_t base_offset = (uint32_t)(first_chunk * CHUNK_BYTES);
        uint32_t end_offset = (last_chunk * CHUNK_BYTES > G.a_half_bytes)
            ? (uint32_t)G.a_half_bytes
            : (uint32_t)(last_chunk * CHUNK_BYTES);
        uint32_t rb_bytes = end_offset - base_offset;

        // Decide runtime split: knee-based + divisibility check on chunks_per_rb.
        int split = ag1_compute_wr_split((int)rb_bytes, WR_SPLIT_CEILING);
        if (chunks_per_rb < split || (chunks_per_rb % split) != 0) split = 1;
        int chunks_per_sub = chunks_per_rb / split;
        uint32_t bytes_per_sub = (uint32_t)(chunks_per_sub * CHUNK_BYTES);

        const int n_peers = G.num_nodes - 1;
        // Per-peer recv_buf / arrival-flag layout: peer p's data lands at
        //   recv_buf  + slot_at_peer * G.a_half_bytes
        //   arrival   + slot_at_peer * G.total_chunks
        // For N == 2, slot_at_peer is always 0, so this is bit-identical to
        // the legacy path. For N > 2, every peer receives a private slot.
        if (split == 1) {
            // Fast-path unchanged behavior: single WR per rb, keyed by rb.
            const int send_peer_count = (G.collective_mode == 1) ? 1 : n_peers;
            for (int peer_slot = 0; peer_slot < send_peer_count; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint16_t)(sap * G.total_chunks + first_chunk);
                cmd.bytes = rb_bytes;
                cmd.local_offset = base_offset;
                cmd.remote_offset = (uint32_t)sap * (uint32_t)G.a_half_bytes + base_offset;
                cmd.src_view = 1;
                cmd.lane_id = (uint16_t)rb;
                cmd.reserved0 = (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(
                        G.d2h_fifos, (uint32_t)cmd.lane_id);
                fifo.push(cmd);
            }
        } else {
            for (int sw = 0; sw < split; ++sw) {
                int sub_first_chunk = first_chunk + sw * chunks_per_sub;
                uint32_t sub_base = (uint32_t)(sub_first_chunk * CHUNK_BYTES);
                uint32_t sub_end = sub_base + bytes_per_sub;
                if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
                uint32_t sub_bytes = sub_end - sub_base;
                const int send_peer_count = (G.collective_mode == 1) ? 1 : n_peers;
                for (int peer_slot = 0; peer_slot < send_peer_count; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint16_t)(sap * G.total_chunks + sub_first_chunk);
                    cmd.bytes = sub_bytes;
                    cmd.local_offset = sub_base;
                    cmd.remote_offset = (uint32_t)sap * (uint32_t)G.a_half_bytes + sub_base;
                    cmd.src_view = 1;
                    cmd.lane_id = (uint16_t)(rb * split + sw);
                    cmd.reserved0 = (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                    internode::D2HFifoDevice fifo =
                        internode::gemm_ar_select_fifo_for_lane(
                            G.d2h_fifos, (uint32_t)cmd.lane_id);
                    fifo.push(cmd);
                }
            }
        }
    }
}

// Direct-send layout for the two 128x128 K-slice halves consumed by one
// phase-2 remote_publish tile. The proxy gathers strided rows from A and writes
// adjacent packed half-tiles into recv_buf, so two arrival flags unlock one
// 256x128 multicast publish tile.
__device__ inline void post_tiled_direct_wrs_for_intra_row(
    const globals& G, int global_row_idx) {
    const int col_blocks = G.A_local.cols() / (globals::RED_BLOCK * 2);
    const int n_peers = G.num_nodes - 1;
    constexpr int half_tile_rows = globals::ROW_BLOCK;
    constexpr int tile_cols = globals::RED_BLOCK * 2;
    constexpr int half_tile_bytes =
        half_tile_rows * tile_cols * (int)sizeof(bf16);
    for (int col_idx = 0; col_idx < col_blocks; ++col_idx) {
        for (int half = 0; half < 2; ++half) {
            const int rb = 2 * global_row_idx + half;
            const uint32_t local_offset =
                (uint32_t)(((uint64_t)rb * half_tile_rows *
                            (uint64_t)G.A_local.cols() +
                            (uint64_t)col_idx * tile_cols) *
                           (uint64_t)sizeof(bf16));
            const uint32_t tile_idx =
                (uint32_t)((global_row_idx * col_blocks + col_idx) * 2 + half);
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(
                    G.node_idx, peer_rank, G.num_nodes);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint16_t)(sap * G.total_chunks + tile_idx);
                cmd.bytes = half_tile_bytes;
                cmd.local_offset = local_offset;
                cmd.remote_offset =
                    (uint32_t)(sap * G.total_chunks + tile_idx) *
                    (uint32_t)half_tile_bytes;
                cmd.src_view = 2;
                cmd.lane_id = (uint16_t)tile_idx;
                cmd.reserved0 =
                    (uint8_t)(peer_slot * globals::NUM_DEVICES + G.dev_idx);
                cmd.row_span = tile_cols * (uint16_t)sizeof(bf16);
                cmd.row_count = half_tile_rows;
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(
                        G.d2h_fifos, (uint32_t)cmd.lane_id);
                fifo.push(cmd);
            }
        }
    }
}

// Forward a fully published A row-block from A_recv (registered as local_data_mr
// in ring mode) to the next node. `source_slot` is the physical peer slot in
// this node's A_recv multicast buffer that contains `origin_rank`'s shard.
// `dst_bank` selects the destination bank on the next rank (1..n_peers-1).
__device__ inline void post_ring_forward_wrs_for_intra_row(
    const globals& G, int source_slot, int origin_rank,
    int global_row_idx, int chunks_per_rb, int dst_bank) {
    constexpr int WR_SPLIT_CEILING = 1;
    const int next_rank = internode::peer_rank_for_slot(G.node_idx, G.num_nodes, 0);
    const int dst_slot = internode::slot_at_peer(origin_rank, next_rank, G.num_nodes);
    const int n_peers = G.num_nodes - 1;
    const int dst_virtual = dst_slot + n_peers * dst_bank;
    const uint32_t src_slot_base =
        (uint32_t)source_slot * (uint32_t)G.a_half_bytes;
    const uint32_t dst_slot_base =
        (uint32_t)dst_virtual * (uint32_t)G.a_half_bytes;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int rb = 2 * global_row_idx + sub;
        int first_chunk = rb * chunks_per_rb;
        int last_chunk = min(first_chunk + chunks_per_rb, G.total_chunks);
        uint32_t base_offset = (uint32_t)(first_chunk * CHUNK_BYTES);
        uint32_t end_offset = (last_chunk * CHUNK_BYTES > G.a_half_bytes)
            ? (uint32_t)G.a_half_bytes
            : (uint32_t)(last_chunk * CHUNK_BYTES);
        uint32_t rb_bytes = end_offset - base_offset;

        int split = ag1_compute_wr_split((int)rb_bytes, WR_SPLIT_CEILING);
        if (chunks_per_rb < split || (chunks_per_rb % split) != 0) split = 1;
        int chunks_per_sub = chunks_per_rb / split;
        uint32_t bytes_per_sub = (uint32_t)(chunks_per_sub * CHUNK_BYTES);

        for (int sw = 0; sw < split; ++sw) {
            int sub_first_chunk = first_chunk + sw * chunks_per_sub;
            uint32_t sub_base = (uint32_t)(sub_first_chunk * CHUNK_BYTES);
            uint32_t sub_end = sub_base + bytes_per_sub;
            if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
            uint32_t sub_bytes = sub_end - sub_base;

            internode::TransferCmd cmd{};
            cmd.cmd_type = internode::CmdType::WRITE;
            cmd.dst_rank = (uint8_t)next_rank;
            cmd.tile_id = (uint16_t)(dst_virtual * G.total_chunks + sub_first_chunk);
            cmd.bytes = sub_bytes;
            cmd.local_offset = src_slot_base + sub_base;
            cmd.remote_offset = dst_slot_base + sub_base;
            cmd.src_view = 0;
            cmd.lane_id = (uint16_t)(rb * split + sw);
            // Logical peer slot 0 == next hop (same encoding as ring merge WRs).
            cmd.reserved0 = (uint8_t)(0 * globals::NUM_DEVICES + G.dev_idx);
            internode::D2HFifoDevice fifo =
                internode::gemm_ar_select_fifo_for_lane(
                    G.d2h_fifos, (uint32_t)cmd.lane_id);
            fifo.push(cmd);
        }
    }
}

// ============================================================================
// Host entrypoint
// ============================================================================

// Forward declaration: kernel body and raw CUDA launch live in src/ag_gemm.cu.
void launch_fused_ag_gemm(const globals& G, unsigned int active_sms);

static globals::activity_event* g_ag_gemm_trace_buf[globals::NUM_DEVICES] = {};
static uint32_t* g_ag_gemm_trace_counts[globals::NUM_DEVICES] = {};
static unsigned long long* g_ag_gemm_trace_start[globals::NUM_DEVICES] = {};
static unsigned long long* g_ag_gemm_trace_end[globals::NUM_DEVICES] = {};
static size_t g_ag_gemm_trace_buf_cap[globals::NUM_DEVICES] = {};

__host__ inline const char* ag_gemm_activity_kind_name(int kind) {
    switch (kind) {
        case globals::ACTIVITY_LOCAL_GATHER: return "local_gather";
        case globals::ACTIVITY_REMOTE_PUBLISH: return "remote_publish";
        case globals::ACTIVITY_COMPUTE_LOCAL: return "compute_local";
        case globals::ACTIVITY_COMPUTE_REMOTE: return "compute_remote";
        default: return "unknown";
    }
}

__host__ inline const char* ag_gemm_block_role_name(int block_idx, const globals& G) {
    return block_idx < G.num_intra_comm ? "intra_comm" : "compute";
}

__host__ inline bool ag_gemm_trace_dump_enabled(int node_idx, int dev_idx) {
    const char* all_ranks = std::getenv("AG_GEMM_ACTIVITY_TRACE_ALL_RANKS");
    if (all_ranks != nullptr && all_ranks[0] == '1') return true;
    const char* all_local = std::getenv("AG_GEMM_ACTIVITY_TRACE_ALL_LOCAL_RANKS");
    if (all_local != nullptr && all_local[0] == '1') return node_idx == 0;
    const char* all_nodes = std::getenv("AG_GEMM_ACTIVITY_TRACE_RANK0_ALL_NODES");
    if (all_nodes != nullptr && all_nodes[0] == '1') return dev_idx == 0;
    return node_idx == 0 && dev_idx == 0;
}

__host__ inline void ag_gemm_trace_dump_path(
    int node_idx, int dev_idx, const char* base_path,
    char* out_path, size_t out_path_size
) {
    const char* all_ranks = std::getenv("AG_GEMM_ACTIVITY_TRACE_ALL_RANKS");
    const char* all_local = std::getenv("AG_GEMM_ACTIVITY_TRACE_ALL_LOCAL_RANKS");
    const char* all_nodes = std::getenv("AG_GEMM_ACTIVITY_TRACE_RANK0_ALL_NODES");
    if ((all_ranks != nullptr && all_ranks[0] == '1') ||
        (all_local != nullptr && all_local[0] == '1') ||
        (all_nodes != nullptr && all_nodes[0] == '1')) {
        std::snprintf(out_path, out_path_size, "%s.node%d_rank%d.json",
                      base_path, node_idx, dev_idx);
        return;
    }
    std::snprintf(out_path, out_path_size, "%s", base_path);
}

__host__ inline void ag_gemm_alloc_activity_trace(
    globals& G, int M, int K, int N
) {
    const char* out_path = std::getenv("AG_GEMM_ACTIVITY_TRACE_OUT");
    if (out_path == nullptr || out_path[0] == '\0') return;
    const int dev = G.dev_idx;
    const int global_row_blocks = (M / G.num_nodes) / (globals::ROW_BLOCK * 2);
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int intra_col_blocks = K / (globals::RED_BLOCK * 2);
    const int node_row_blocks = (M / G.num_nodes) / globals::ROW_BLOCK;
    const int gemm_col_blocks = N / globals::COL_BLOCK;
    const int num_local_blocks = local_row_blocks * intra_col_blocks;
    const int num_compute_blocks = node_row_blocks * gemm_col_blocks * G.num_nodes;
    const int intra_events = (G.num_intra_comm > 0)
        ? (2 * (G.num_nodes - 1) + 1)
          * ((num_local_blocks + G.num_intra_comm - 1) / G.num_intra_comm)
        : 0;
    const int comp_events = (G.num_comp_sms > 0)
        ? (num_compute_blocks + G.num_comp_sms - 1) / G.num_comp_sms
        : 0;
    G.activity_max_events = std::max(64, 2 * std::max(intra_events, comp_events) + 64);

    const size_t event_bytes =
        (size_t)config::NUM_BLOCKS * (size_t)G.activity_max_events
        * sizeof(globals::activity_event);
    const size_t count_bytes = (size_t)config::NUM_BLOCKS * sizeof(uint32_t);
    if (g_ag_gemm_trace_buf_cap[dev] < event_bytes) {
        if (g_ag_gemm_trace_buf[dev] != nullptr) cudaFree(g_ag_gemm_trace_buf[dev]);
        cudaMalloc(&g_ag_gemm_trace_buf[dev], event_bytes);
        g_ag_gemm_trace_buf_cap[dev] = event_bytes;
    }
    if (g_ag_gemm_trace_counts[dev] == nullptr) cudaMalloc(&g_ag_gemm_trace_counts[dev], count_bytes);
    if (g_ag_gemm_trace_start[dev] == nullptr) cudaMalloc(&g_ag_gemm_trace_start[dev], sizeof(unsigned long long));
    if (g_ag_gemm_trace_end[dev] == nullptr) cudaMalloc(&g_ag_gemm_trace_end[dev], sizeof(unsigned long long));
    cudaMemset(g_ag_gemm_trace_buf[dev], 0, event_bytes);
    cudaMemset(g_ag_gemm_trace_counts[dev], 0, count_bytes);
    cudaMemset(g_ag_gemm_trace_start[dev], 0, sizeof(unsigned long long));
    cudaMemset(g_ag_gemm_trace_end[dev], 0, sizeof(unsigned long long));
    G.activity_buf = g_ag_gemm_trace_buf[dev];
    G.activity_counts = g_ag_gemm_trace_counts[dev];
    G.kernel_start_ns = g_ag_gemm_trace_start[dev];
    G.kernel_end_ns = g_ag_gemm_trace_end[dev];
}

__host__ inline void ag_gemm_dump_activity_trace(
    globals& G, int M, int N, int node_idx, int dev_idx
) {
    if (G.activity_buf == nullptr || G.activity_counts == nullptr) return;
    std::vector<globals::activity_event> host_events(
        (size_t)config::NUM_BLOCKS * (size_t)G.activity_max_events);
    std::vector<uint32_t> host_counts(config::NUM_BLOCKS, 0);
    unsigned long long kernel_start_ns = 0;
    unsigned long long kernel_end_ns = 0;
    cudaDeviceSynchronize();
    cudaMemcpy(host_events.data(), G.activity_buf,
               host_events.size() * sizeof(globals::activity_event),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_counts.data(), G.activity_counts,
               host_counts.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&kernel_start_ns, G.kernel_start_ns,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&kernel_end_ns, G.kernel_end_ns,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long min_event_start = ~0ull;
    unsigned long long max_event_end = 0;
    for (int b = 0; b < config::NUM_BLOCKS; ++b) {
        const uint32_t count = std::min(host_counts[b], (uint32_t)G.activity_max_events);
        for (uint32_t i = 0; i < count; ++i) {
            const auto& ev = host_events[(size_t)b * (size_t)G.activity_max_events + i];
            if (ev.start_ns != 0 && ev.start_ns < min_event_start) min_event_start = ev.start_ns;
            if (ev.end_ns > max_event_end) max_event_end = ev.end_ns;
        }
    }
    if (min_event_start != ~0ull && (kernel_start_ns == 0 || kernel_start_ns > min_event_start)) {
        kernel_start_ns = min_event_start;
    }
    if (kernel_end_ns < max_event_end) kernel_end_ns = max_event_end;

    G.activity_buf = nullptr;
    G.activity_counts = nullptr;
    G.kernel_start_ns = nullptr;
    G.kernel_end_ns = nullptr;

    const char* base_out_path = std::getenv("AG_GEMM_ACTIVITY_TRACE_OUT");
    if (base_out_path == nullptr || base_out_path[0] == '\0') return;
    if (!ag_gemm_trace_dump_enabled(node_idx, dev_idx)) return;
    char out_path[4096];
    ag_gemm_trace_dump_path(node_idx, dev_idx, base_out_path, out_path, sizeof(out_path));
    FILE* f = std::fopen(out_path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "[AG_GEMM_ACTIVITY_TRACE] failed to open %s\n", out_path);
        return;
    }
    const int total_gemm_tiles =
        ((M / G.num_nodes) / globals::ROW_BLOCK) * (N / globals::COL_BLOCK) * G.num_nodes;
    std::fprintf(f,
        "{\n"
        "  \"kernel\": \"ag_gemm\",\n"
        "  \"node_idx\": %d,\n"
        "  \"dev_idx\": %d,\n"
        "  \"M\": %d,\n"
        "  \"N\": %d,\n"
        "  \"num_blocks\": %d,\n"
        "  \"num_intra_comm_sms\": %d,\n"
        "  \"num_comp_sms\": %d,\n"
        "  \"num_nodes\": %d,\n"
        "  \"total_chunks\": %d,\n"
        "  \"total_gemm_tiles\": %d,\n"
        "  \"kernel_start_ns\": %llu,\n"
        "  \"kernel_end_ns\": %llu,\n"
        "  \"activity_max_events\": %d,\n"
        "  \"blocks\": [\n",
        node_idx, dev_idx, M, N, config::NUM_BLOCKS,
        G.num_intra_comm, G.num_comp_sms, G.num_nodes, G.total_chunks,
        total_gemm_tiles, kernel_start_ns, kernel_end_ns, G.activity_max_events);
    for (int b = 0; b < config::NUM_BLOCKS; ++b) {
        const uint32_t count = std::min(host_counts[b], (uint32_t)G.activity_max_events);
        std::fprintf(f,
            "    {\n"
            "      \"block\": %d,\n"
            "      \"role\": \"%s\",\n"
            "      \"events\": [",
            b, ag_gemm_block_role_name(b, G));
        for (uint32_t i = 0; i < count; ++i) {
            const auto& ev = host_events[(size_t)b * (size_t)G.activity_max_events + i];
            if (i != 0) std::fprintf(f, ",");
            std::fprintf(f,
                "\n        {\"kind\":\"%s\",\"work_id\":%d,\"start_ns\":%llu,\"end_ns\":%llu}",
                ag_gemm_activity_kind_name(ev.kind), ev.work_id, ev.start_ns, ev.end_ns);
        }
        if (count != 0) std::fprintf(f, "\n");
        std::fprintf(f, "      ]\n    }%s\n", (b + 1 == config::NUM_BLOCKS) ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("[AG_GEMM_ACTIVITY_TRACE rank=%d node=%d M=%d N=%d file=%s]\n",
                dev_idx, node_idx, M, N, out_path);
}

void entrypoint(
    dist::ParallelBuffer& A,
    const at::Tensor& B,
    at::Tensor& C,
    dist::ParallelBuffer& barrier,
    int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_comm_sms,
    int64_t a_half_bytes,
    dist::ParallelBuffer& A_recv,  // #8: multicast-backed peer A_half
    const int active_sms = config::NUM_BLOCKS,
    int num_intra_comm_override = 0,
    int num_nodes = 2  // total node count (>= 2).
) {
    TORCH_CHECK(B.is_cuda() && B.is_contiguous(), "B must be contiguous CUDA");
    TORCH_CHECK(C.is_cuda() && C.is_contiguous(), "C must be contiguous CUDA");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");

    const int dev_idx = A.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M_node = A.data_.size(0);
    const int K = A.data_.size(1);
    const int N = B.size(1);
    const int M = M_node * num_nodes;

    TORCH_CHECK(M % globals::ROW_BLOCK == 0);
    TORCH_CHECK(K % globals::RED_BLOCK == 0);
    TORCH_CHECK(N % globals::COL_BLOCK == 0);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    // Intra-gather geometry needs the per-node shard to be divisible by
    // NUM_DEVICES 256-row multicast tiles. For 8 GPUs: M_node % 2048 == 0.
    TORCH_CHECK(M_node >= globals::NUM_DEVICES * globals::ROW_BLOCK * 2,
                "M_node must be >= ", globals::NUM_DEVICES * globals::ROW_BLOCK * 2,
                " (got M_node=", M_node, ")");
    TORCH_CHECK(M_node % (globals::NUM_DEVICES * globals::ROW_BLOCK * 2) == 0,
                "M_node must be a multiple of ",
                globals::NUM_DEVICES * globals::ROW_BLOCK * 2,
                " (got M_node=", M_node, ")");

    uint64_t a_half_bytes_u64 = (uint64_t)a_half_bytes;
    int total_chunks = (int)((a_half_bytes_u64 + CHUNK_BYTES - 1) / CHUNK_BYTES);
    const bool tiled_direct_host =
        std::getenv("AG_GEMM_TILED_DIRECT") != nullptr &&
        std::getenv("AG_GEMM_TILED_DIRECT")[0] == '1';
    if (tiled_direct_host) {
        total_chunks *= 2;
    }

    // Split CTAs between intra-comm and compute. Intra-gather CTAs also post
    // zero-copy inter-node RDMA WRs, so there is no separate inter-comm pool.
    int adaptive_comm_sms = num_comm_sms;
    const char* perf_preset = std::getenv("AG_GEMM_PERF_PRESET");
    const bool perf_legacy =
        perf_preset != nullptr && std::strcmp(perf_preset, "legacy") == 0;
    int adaptive_cap_large_m = perf_legacy ? 8 : 16;
    if (const char* e = std::getenv("AG_GEMM_ADAPTIVE_CAP_LARGE_M")) {
        adaptive_cap_large_m = std::max(1, std::atoi(e));
    }
    if (std::getenv("AG1_ADAPTIVE_COMM_SMS") == nullptr ||
        std::atoi(std::getenv("AG1_ADAPTIVE_COMM_SMS")) != 0) {
        if (M >= 32768) adaptive_comm_sms = std::min(num_comm_sms, adaptive_cap_large_m);
        else if (M >= 16384) adaptive_comm_sms = std::min(num_comm_sms, 32);
        // Small-M: at M<=4K intra-gather has only local_row_blocks=2 rows per
        // rank with col_blocks=2 (K=256/128), so 4 total intra tasks. Under
        // MERGE+EARLY_SEND extra intra CTAs sit idle but steal from compute.
        else if (M <= 4096) adaptive_comm_sms = std::min(num_comm_sms, 8);
        // At M<=8K, keep enough intra CTAs to cover the small number of
        // gather tasks while returning idle CTAs to compute.
        else if (M <= 8192) adaptive_comm_sms = std::min(num_comm_sms, 32);
    }
    const int num_intra_comm = (num_intra_comm_override > 0)
        ? num_intra_comm_override
        : std::max(4, adaptive_comm_sms / 2);
    int num_comp_sms = active_sms - num_intra_comm;
    const bool host_skip_compute =
        std::getenv("AG_GEMM_SKIP_COMPUTE") != nullptr &&
        std::getenv("AG_GEMM_SKIP_COMPUTE")[0] == '1';
    TORCH_CHECK(num_comp_sms > 0 || host_skip_compute,
                "num_comp_sms must be > 0, got ", num_comp_sms,
                " (active_sms=", active_sms, " num_intra_comm=", num_intra_comm, ")");

    auto A_local = ::dist::make_local_tensor<globals::A_local_tensor>(
        (uint64_t)A.data_.data_ptr(), 1, 1, M_node, K);

    int logical_lq = 1;
    if (const char* e = std::getenv("MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        logical_lq = std::max(1, std::atoi(e));
    }
    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity,
        16, logical_lq);
    // Internode AG pattern: ring when num_nodes > 2 unless overridden.
    //   unset / "auto"  → ring if num_nodes > 2 else direct
    //   "ring" / "direct" → force that mode
    int collective_mode = 0;
    const char* ic = std::getenv("AG_GEMM_INTERNODE_COLLECTIVE");
    if (ic != nullptr && ic[0] != '\0') {
        if (std::strcmp(ic, "ring") == 0) {
            collective_mode = 1;
        } else if (std::strcmp(ic, "direct") == 0) {
            collective_mode = 0;
        } else if (std::strcmp(ic, "auto") == 0) {
            collective_mode = (num_nodes > 2) ? 1 : 0;
        } else {
            collective_mode = (num_nodes > 2) ? 1 : 0;
        }
    } else {
        collective_mode = (num_nodes > 2) ? 1 : 0;
    }
    if (perf_legacy) {
        collective_mode = 0;
    }
    int ring_recv_banks = 1;
    if (collective_mode == 1) {
        ring_recv_banks = std::max(1, num_nodes - 1);
    }
    const bool tiled_direct =
        collective_mode == 0 &&
        tiled_direct_host;
    const int recv_tensor_rows = tiled_direct
        ? (num_nodes - 1) * total_chunks * globals::ROW_BLOCK
        : M_node * (num_nodes - 1) * ring_recv_banks;
    const int recv_tensor_cols = tiled_direct
        ? (globals::RED_BLOCK * 2)
        : K;
    auto A_recv_local_tensor = ::dist::make_local_tensor<globals::A_local_tensor>(
        (uint64_t)recv_buf_ptr, 1, 1,
        recv_tensor_rows, recv_tensor_cols);

    globals G{
        .A = ::dist::distributed_tensor_from_buffer<globals::A_distributed_tensor>(A),
        .barrier = ::dist::distributed_tensor_from_buffer<globals::barrier_distributed_tensor>(barrier),
        .A_local = A_local,
        .A_recv_local_tensor = A_recv_local_tensor,
        .A_recv = ::dist::distributed_tensor_from_buffer<globals::A_distributed_tensor>(
            A_recv, 1, 1, M_node * (num_nodes - 1), K),
        .B = ::dist::local_tensor_from_tensor<globals::B_local_tensor>(B),
        .C = ::dist::local_tensor_from_tensor<globals::C_local_tensor>(C),
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
        .total_chunks = total_chunks,
        .a_half_bytes = a_half_bytes_u64,
        .dev_idx = dev_idx,
        .node_idx = node_idx,
        .num_nodes = num_nodes,
        .collective_mode = collective_mode,
        .ring_recv_banks = ring_recv_banks,
        .debug_skip_remote_compute =
            (std::getenv("AG_GEMM_SKIP_REMOTE_COMPUTE") != nullptr &&
             std::getenv("AG_GEMM_SKIP_REMOTE_COMPUTE")[0] == '1') ? 1 : 0,
        .debug_skip_phase1 =
            (std::getenv("AG_GEMM_SKIP_PHASE1") != nullptr &&
             std::getenv("AG_GEMM_SKIP_PHASE1")[0] == '1') ? 1 : 0,
        .debug_skip_phase1_gate =
            (std::getenv("AG_GEMM_SKIP_PHASE1_GATE") != nullptr &&
             std::getenv("AG_GEMM_SKIP_PHASE1_GATE")[0] == '1') ? 1 : 0,
        .debug_skip_phase2 =
            (std::getenv("AG_GEMM_SKIP_PHASE2") != nullptr &&
             std::getenv("AG_GEMM_SKIP_PHASE2")[0] == '1') ? 1 : 0,
        .debug_skip_compute =
            (std::getenv("AG_GEMM_SKIP_COMPUTE") != nullptr &&
             std::getenv("AG_GEMM_SKIP_COMPUTE")[0] == '1') ? 1 : 0,
        .debug_skip_reset =
            (std::getenv("AG_GEMM_SKIP_RESET") != nullptr &&
             std::getenv("AG_GEMM_SKIP_RESET")[0] == '1') ? 1 : 0,
        .ring_proxy_forward =
            (std::getenv("AG_GEMM_RING_PROXY_FORWARD") != nullptr &&
             std::getenv("AG_GEMM_RING_PROXY_FORWARD")[0] == '1') ? 1 : 0,
        .remote_ready_per_col =
            (std::getenv("AG_GEMM_REMOTE_READY_PER_COL") != nullptr &&
             std::getenv("AG_GEMM_REMOTE_READY_PER_COL")[0] == '1') ? 1 : 0,
        .tiled_direct = tiled_direct ? 1 : 0,
        .num_intra_comm = num_intra_comm,
        .num_comp_sms = num_comp_sms,
    };

    ag_gemm_alloc_activity_trace(G, M, K, N);

    // Fast-path #1: barrier_reset is inlined into fused_kernel's exit (see
    // fused_kernel in src/ag_gemm.cu). Saves one cudaLaunchKernel +
    // persistent-CTA-startup per iter — measurable at small M where total
    // time is sub-millisecond.
    launch_fused_ag_gemm(G, (unsigned int)active_sms);
    ag_gemm_dump_activity_trace(G, M, N, node_idx, dev_idx);

}

}  // namespace ag_gemm_multinode
