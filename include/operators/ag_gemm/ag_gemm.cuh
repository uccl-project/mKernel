#pragma once

/**
 * @file ag_gemm.cuh
 * @brief Multi-node All-Gather + GEMM — truly fused single-kernel (multi-node).
 *
 * Single kernel launch. Two CTA groups run concurrently:
 *
 *   Intra-comm CTAs [0, num_intra_comm):
 *     IPC multicast gather of A shards within the node, then phase-2
 *     republish of RDMA-received peer shards into A_recv. Zero-copy RDMA WRs
 *     for this rank's M_local slice (src_view=1 → DMA-BUF MR aliasing A.data_)
 *     are posted by a separate prologue kernel so the NIC pipelines the
 *     inter-node send concurrently with intra-gather + compute.
 *     Signals barrier[row_block] when each row block is fully gathered.
 *
 *   Comp CTAs [num_intra_comm, 132):
 *     GEMM on all M rows. CTAs claim tiles by CTA-stride:
 *       - Local-half tile: wait on intra-node barrier (plane 0), TMA load from A_local
 *       - Remote-half tile: wait on intra-node barrier (plane 2), TMA load from A_recv
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
    static constexpr int NUM_NODES = 2;
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
    // Multicast-backed A_recv buffer. Each rank r on node N publishes the
    // 1/NUM_DEVICES slice of peer A_half it received via RDMA into this
    // dbuf, so all ranks on node N see the full peer A_half after phase-2.
    // Compute remote-tile loads read from this.
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
    // Each hop in the ring uses a separate recv/arrival bank so a forward
    // cannot RDMA-overwrite a slot another GPU is still TMA-reading (intra-
    // node sync does not order remote sends). Banks = num_nodes - 1.
    const int ring_recv_banks;
    const int debug_skip_remote_compute;
    const int debug_skip_phase1;
    const int debug_skip_phase1_gate;
    const int debug_skip_phase2;
    const int debug_skip_compute;
    const int debug_skip_reset;
    const int ring_proxy_forward;
    const int remote_ready_per_col;
    const int num_intra_comm;  // CTAs for intra-node IPC gather + RDMA push
    const int num_comp_sms;    // CTAs for GEMM compute

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

__device__ inline unsigned long long ag_gemm_globaltimer() {
    return comm::globaltimer();
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
// When WR_SPLIT_CEILING > 1, each inter-rb is split into effective_split
// sub-WRs sized rb_bytes/split, striped across QPs by (rb*split + sw) key.
// effective_split computed from rb_bytes — only kicks in at large shapes
// where sub-WR stays >= 2 MB.
__device__ inline void post_merge_wrs_for_intra_row(
    const globals& G, int global_row_idx, int chunks_per_rb) {
    constexpr int WR_SPLIT_CEILING = 1;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int rb = 2 * global_row_idx + sub;
        int first_chunk = rb * chunks_per_rb;
        int last_chunk = min(first_chunk + chunks_per_rb, G.total_chunks);
        uint64_t base_offset = (uint64_t)first_chunk * CHUNK_BYTES;
        uint64_t last_chunk_bytes = (uint64_t)last_chunk * CHUNK_BYTES;
        uint64_t end_offset = (last_chunk_bytes > (uint64_t)G.a_half_bytes)
            ? (uint64_t)G.a_half_bytes
            : last_chunk_bytes;
        uint32_t rb_bytes = (uint32_t)(end_offset - base_offset);

        // Decide runtime split: knee-based + divisibility check on chunks_per_rb.
        int split = ag1_compute_wr_split((int)rb_bytes, WR_SPLIT_CEILING);
        if (chunks_per_rb < split || (chunks_per_rb % split) != 0) split = 1;
        int chunks_per_sub = chunks_per_rb / split;
        uint32_t bytes_per_sub = (uint32_t)(chunks_per_sub * CHUNK_BYTES);

        // Per-peer recv_buf / arrival-flag layout: peer p's data lands at
        //   recv_buf  + slot_at_peer * G.a_half_bytes
        //   arrival   + slot_at_peer * G.total_chunks
        if (split == 1) {
            // Fast-path: single WR per rb, keyed by rb. Ring sends to one
            // next-hop peer per step (n_peers steps over the kernel's life).
            for (int peer_slot = 0; peer_slot < 1; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint32_t)(sap * G.total_chunks + first_chunk);
                cmd.bytes = rb_bytes;
                cmd.local_offset = base_offset;
                cmd.remote_offset = (uint64_t)sap * (uint64_t)G.a_half_bytes + base_offset;
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
                uint64_t sub_base = (uint64_t)sub_first_chunk * CHUNK_BYTES;
                uint64_t sub_end = sub_base + bytes_per_sub;
                if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
                uint32_t sub_bytes = (uint32_t)(sub_end - sub_base);
                for (int peer_slot = 0; peer_slot < 1; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank, G.num_nodes);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint32_t)(sap * G.total_chunks + sub_first_chunk);
                    cmd.bytes = sub_bytes;
                    cmd.local_offset = sub_base;
                    cmd.remote_offset = (uint64_t)sap * (uint64_t)G.a_half_bytes + sub_base;
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
    const uint64_t src_slot_base =
        (uint64_t)source_slot * (uint64_t)G.a_half_bytes;
    const uint64_t dst_slot_base =
        (uint64_t)dst_virtual * (uint64_t)G.a_half_bytes;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int rb = 2 * global_row_idx + sub;
        int first_chunk = rb * chunks_per_rb;
        int last_chunk = min(first_chunk + chunks_per_rb, G.total_chunks);
        uint64_t base_offset = (uint64_t)first_chunk * CHUNK_BYTES;
        uint64_t last_chunk_bytes = (uint64_t)last_chunk * CHUNK_BYTES;
        uint64_t end_offset = (last_chunk_bytes > (uint64_t)G.a_half_bytes)
            ? (uint64_t)G.a_half_bytes
            : last_chunk_bytes;
        uint32_t rb_bytes = (uint32_t)(end_offset - base_offset);

        int split = ag1_compute_wr_split((int)rb_bytes, WR_SPLIT_CEILING);
        if (chunks_per_rb < split || (chunks_per_rb % split) != 0) split = 1;
        int chunks_per_sub = chunks_per_rb / split;
        uint32_t bytes_per_sub = (uint32_t)(chunks_per_sub * CHUNK_BYTES);

        for (int sw = 0; sw < split; ++sw) {
            int sub_first_chunk = first_chunk + sw * chunks_per_sub;
            uint64_t sub_base = (uint64_t)sub_first_chunk * CHUNK_BYTES;
            uint64_t sub_end = sub_base + bytes_per_sub;
            if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
            uint32_t sub_bytes = (uint32_t)(sub_end - sub_base);

            internode::TransferCmd cmd{};
            cmd.cmd_type = internode::CmdType::WRITE;
            cmd.dst_rank = (uint8_t)next_rank;
            cmd.tile_id = (uint32_t)(dst_virtual * G.total_chunks + sub_first_chunk);
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
    dist::ParallelBuffer& A_recv,  // multicast-backed peer A_half
    const int active_sms,
    int num_intra_comm_override,
    int num_nodes
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

    // Split CTAs between intra-comm and compute. Intra-gather CTAs also post
    // zero-copy inter-node RDMA WRs, so there is no separate inter-comm pool.
    int adaptive_comm_sms = num_comm_sms;
    int adaptive_cap_large_m = 16;
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
    // Ring all-gather: n_peers hops, each into its own recv bank so a
    // forward never overwrites a slot another GPU is still TMA-reading.
    const int ring_recv_banks = std::max(1, num_nodes - 1);
    const int recv_tensor_rows = M_node * (num_nodes - 1) * ring_recv_banks;
    const int recv_tensor_cols = K;
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
        .num_intra_comm = num_intra_comm,
        .num_comp_sms = num_comp_sms,
    };

    // barrier_reset is inlined into fused_kernel's exit (see fused_kernel
    // in src/ag_gemm.cu). Saves one cudaLaunchKernel + persistent-CTA-
    // startup per iter — measurable at small M where total time is
    // sub-millisecond.
    launch_fused_ag_gemm(G, (unsigned int)active_sms);
}

}  // namespace ag_gemm_multinode
