#pragma once

/**
 * @file ring_attention.cuh
 * @brief Multi-node Ring Attention — infrastructure + host entrypoint.
 *
 * Kernel content (role functions, __global__ stubs)
 * lives in src/ring_attention.cu.
 * Python/session glue + pybind module live in
 *   include/operators/ring_attention/session.cuh
 *
 * Algorithm.
 *
 *   N nodes x M GPUs/node (M = INTRA_NUM_DEVICES). Total ring length is
 *   N * M stages. The ring is decomposed into N rounds of M intra-node
 *   stages each:
 *
 *     Round 0 (local):
 *       Stage 0:           attention(Q, K0[dev_idx], V0[dev_idx]) -> O, L
 *       Stages 1..M-1:     intra-node ring rotation across local GPUs.
 *
 *     Rounds 1..N-1 (each remote peer):
 *       Before round:      wait for peer's K/V to arrive in recv_buf;
 *                          D2D copy peer_slot's K/V into K0/V0[dev_idx].
 *       Stages r*M..r*M+M-1:
 *                          intra-node ring rotation on the freshly copied
 *                          peer KV, merging into the running (O, L).
 *
 *     Final reduction:     online softmax merge of per-stage partials.
 *
 *   RDMA: kv_send_kernel posts WRs from every rank to every peer at the
 *   start, so the (N-1) peer copies overlap with the round-0 compute.
 *   kv_copy_kernel(peer_slot) is launched once per remote round and
 *   reads from recv_buf + peer_slot * single_peer_bytes.
 *
 *   The intra-node ring_attn primitive (attn_partial / attn_comm /
 *   attn_reduction) is invoked once per stage, ping-ponging between K0/V0
 *   and K1/V1 on stage parity so each rank sees every shard exactly once
 *   per round.
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace ring_attn_multinode {

static constexpr int CHUNK_BYTES = 256 * 1024;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - STATIC_SHARED_MEMORY;
    // CONSUMER_WARPGROUPS=3 (4 wg total → 16 warps).
    static constexpr int CONSUMER_WARPGROUPS = 3;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    // Register split: PRODUCER_REGISTERS + CONSUMER_WARPGROUPS *
    // CONSUMER_REGISTERS must stay ≤ 512 (per-CTA reg file budget).
    // Over-allocation is silently accepted by the compiler but hangs at
    // runtime when setmaxnreg.inc can't be satisfied.
    static constexpr int PRODUCER_REGISTERS = 24;
    static constexpr int CONSUMER_REGISTERS = 160;
};

// ============================================================================
// Globals
// ============================================================================

struct globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int D = 128;
    static constexpr int QO_BLOCK = 64;
    static constexpr int KV_BLOCK = 128;
    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile = st_bf<QO_BLOCK, D>;
    using K_tile = st_bf<KV_BLOCK, D>;
    using V_tile = st_bf<KV_BLOCK, D>;
    using L_vec = col_vec<st_fl<QO_BLOCK, D>>;
    using O_tile = st_bf<QO_BLOCK, D>;
    using L_vec_2x = col_vec<st_fl<2 * QO_BLOCK, D>>;
    using O_tile_2x = st_bf<2 * QO_BLOCK, D>;

    using Q_local_tensor = dist::local_tensor<bf16, -1, -1, -1, D, Q_tile>;
    using K_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, -1, -1, -1, D, K_tile>, NUM_DEVICES, false>;
    using V_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, -1, -1, -1, D, V_tile>, NUM_DEVICES, false>;
    using L_local_tensor = dist::local_tensor<float, 1, -1, -1, -1, L_vec, L_vec_2x>;
    using O_local_tensor = dist::local_tensor<bf16, -1, -1, -1, D, O_tile, O_tile_2x>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<NUM_DEVICES>;

    Q_local_tensor Q;
    K_distributed_tensor K0;
    K_distributed_tensor K1;
    V_distributed_tensor V0;
    V_distributed_tensor V1;
    L_local_tensor L_block;
    L_local_tensor L;
    O_local_tensor O_block;
    O_local_tensor O;
    barrier_distributed_tensor barrier;

    int ring_stage;
    const int dev_idx;
    const int num_comm_sms;

    __host__ inline int num_partial_blocks() const {
        return Q.batch() * Q.depth() * Q.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK);
    }
    __host__ inline int num_reduction_blocks() const {
        return O.batch() * O.depth() * O.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK * 2);
    }

};

// ============================================================================
// Inter-node KV exchange globals
// ============================================================================
//
// In the entrypoint prologue kv_send_kernel posts RDMA WRs from every GPU to
// its same-index counterpart on every remote node. The peers' K/V land in this
// rank's recv_buf, partitioned into (num_nodes - 1) [K | V] slots.
//
// Between rounds, kv_copy_kernel(peer_slot) does:
//   1. Wait for that peer slot's K/V to fully arrive (arrival_flags).
//   2. D2D copy recv_buf[peer_slot * single_peer_bytes ..] into the local
//      slot of K0/V0 (overwriting whatever the previous round left there).
//
// After each kv_copy, the next M intra-node ring stages rotate the freshly
// copied peer KV around the local M-GPU ring just like the round-0 local KV.

struct kv_exchange_globals {
    bf16 *send_buf;          // legacy staging buffer pointer, unused under
                             // zero-copy send (K0/V0 are RDMA-registered
                             // directly via DMA-BUF; src_view selects between
                             // them at the proxy).
    // Base pointer to the receive buffer. At N peers the buffer is laid out
    // as (N-1) consecutive [K | V] slots; the kv_copy kernel takes a peer
    // slot argument and reads from `recv_buf + peer_slot * single_peer_bytes`.
    bf16 *recv_buf_base;
    bf16 *K0_local;          // local slot of K0 dbuf, D2D copy destination
    bf16 *V0_local;          // local slot of V0 dbuf
    int   K_bytes;           // K tensor byte size
    int   V_bytes;           // V tensor byte size
    int   total_chunks_K;
    int   total_chunks_V;
    int   node_idx;
    int   dev_idx;
    int   num_nodes;  // total node count (>= 2).
    int   num_send_sms;
    int   num_copy_sms;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t *arrival_flags;
    uint32_t epoch;
};

// ============================================================================
// Cross-GPU barrier-only kernel config (declared here so entrypoint can launch
// it; the kernel itself lives in the .cu).
// ============================================================================
struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 32;
    static constexpr int DYNAMIC_SHARED_MEMORY = 1024;
};

// ============================================================================
// Forward declarations for __global__ kernels defined in src/ring_attention.cu.
// The host entrypoint launches these via plain <<<...>>>, so a forward decl is
// sufficient (no launch_kernel<> template instantiation that would require a
// launch wrapper).
// ============================================================================
__global__ void attn_comm_partial_stage_kernel(
    const __grid_constant__ globals G, const int stage);
__global__ void attn_reduction_stage_kernel(
    const __grid_constant__ globals G);
__global__ void kv_send_kernel(
    const __grid_constant__ kv_exchange_globals KE);
__global__ void kv_copy_kernel(
    const __grid_constant__ kv_exchange_globals KE,
    const int peer_slot);
__global__ void barrier_only_kernel(
    const __grid_constant__ globals::barrier_distributed_tensor barrier, const int dev_idx);

// ============================================================================
// Host orchestration entrypoint
// ============================================================================
//
// Sequence (per call). Let M = INTRA_NUM_DEVICES (GPUs per node).
//   1. Launch kv_send_kernel (RDMA prologue — fans WRs to every remote
//      peer; returns immediately so EFA transfers overlap with the round-0
//      compute).
//   2. Round 0 — local KV. Loop ring_stage = 0..M-1:
//        - attn_comm_partial_stage_kernel(ring_stage)
//        - if ring_stage > 0: attn_reduction_stage_kernel
//        - barrier_only_kernel
//   3. For each remote peer slot p in 0..num_nodes-2:
//        - kv_copy_kernel(peer_slot=p) (wait for peer p's KV, D2D into K0/V0)
//        - barrier_only_kernel
//        - Loop ring_stage = (p+1)*M .. (p+2)*M - 1 with the same pattern as
//          step 2 but reduction always runs.
//
// The intra-node ring ping-pongs K0↔K1 / V0↔V1 every stage; kv_copy_kernel
// re-initializes K0/V0 to the current peer's KV before each remote round.

inline void entrypoint(
    const at::Tensor &Q,
    dist::ParallelBuffer &K0,
    dist::ParallelBuffer &K1,
    dist::ParallelBuffer &V0,
    dist::ParallelBuffer &V1,
    at::Tensor &L,
    at::Tensor &L_block,
    at::Tensor &O,
    at::Tensor &O_block,
    dist::ParallelBuffer &barrier,
    int64_t send_buf_ptr,
    int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_comm_sms,
    int num_send_sms,
    int num_copy_sms,
    int num_nodes
) {
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    // Construct globals used for all attn_partial / attn_comm calls
    auto make_G = [&](int ring_stage) {
        return globals{
            .Q = ::dist::local_tensor_from_tensor<typename globals::Q_local_tensor>(Q),
            .K0 = ::dist::distributed_tensor_from_buffer<typename globals::K_distributed_tensor>(K0),
            .K1 = ::dist::distributed_tensor_from_buffer<typename globals::K_distributed_tensor>(K1),
            .V0 = ::dist::distributed_tensor_from_buffer<typename globals::V_distributed_tensor>(V0),
            .V1 = ::dist::distributed_tensor_from_buffer<typename globals::V_distributed_tensor>(V1),
            .L_block = ::dist::local_tensor_from_tensor<typename globals::L_local_tensor>(L_block),
            .L = ::dist::local_tensor_from_tensor<typename globals::L_local_tensor>(L),
            .O_block = ::dist::local_tensor_from_tensor<typename globals::O_local_tensor>(O_block),
            .O = ::dist::local_tensor_from_tensor<typename globals::O_local_tensor>(O),
            .barrier = ::dist::distributed_tensor_from_buffer<typename globals::barrier_distributed_tensor>(barrier),
            .ring_stage = ring_stage,
            .dev_idx = dev_idx,
            .num_comm_sms = num_comm_sms,
        };
    };

    int K_bytes = (int)(K0.data_.numel() * 2);
    int V_bytes = (int)(V0.data_.numel() * 2);
    int total_chunks_K = (K_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;
    int total_chunks_V = (V_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;

    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity, 4);

    int n_send = std::max(1, num_send_sms);
    int n_copy = std::max(1, num_copy_sms);
    if (n_send + n_copy > 132) { n_send = 66; n_copy = 66; }

    kv_exchange_globals KE{
        .send_buf = reinterpret_cast<bf16*>(send_buf_ptr),
        .recv_buf_base = reinterpret_cast<bf16*>(recv_buf_ptr),
        .K0_local = reinterpret_cast<bf16*>(K0.data_.data_ptr()),
        .V0_local = reinterpret_cast<bf16*>(V0.data_.data_ptr()),
        .K_bytes = K_bytes,
        .V_bytes = V_bytes,
        .total_chunks_K = total_chunks_K,
        .total_chunks_V = total_chunks_V,
        .node_idx = node_idx,
        .dev_idx = dev_idx,
        .num_nodes = num_nodes,
        .num_send_sms = n_send,
        .num_copy_sms = n_copy,
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
    };
    {
        globals G = make_G(0);
        int n_partial = G.num_partial_blocks();
        int n_reduction = G.num_reduction_blocks();

        // Per-stage launch path. Each launch isolates its register frame so
        // attn_partial / attn_comm / attn_reduction get STACK:0. A fused
        // persistent variant forces ~150 bytes of silent spills in a single
        // kernel that no amount of __noinline__ refactoring or FA3 layout
        // can eliminate without halving wgmma throughput.
        cudaFuncSetAttribute(attn_comm_partial_stage_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY);
        cudaFuncSetAttribute(attn_reduction_stage_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY);

        int comm_partial_grid = num_comm_sms + n_partial;
        int reduction_grid    = n_reduction;
        int kv_send_grid      = n_send;
        int kv_copy_grid      = n_send + n_copy;


        // Prologue: post RDMA WRs so peer-KV transfer overlaps with round-0 stages.
        // kv_send/kv_copy don't use dynamic smem — launch with smem=0.
        kv_send_kernel<<<kv_send_grid, config::NUM_THREADS, 0, stream>>>(KE);
        // Match persistent kernel: barrier_all after send, before stages start.
        {
            globals Gb = make_G(0);
            barrier_only_kernel<<<1, 32, 0, stream>>>(Gb.barrier, dev_idx);
        }

        // Total ring is `num_nodes` rounds of `M` intra-node stages each,
        // where M = INTRA_NUM_DEVICES (the per-node ring length). Round 0
        // processes the local node's KV; each subsequent round first fetches
        // a peer node's KV into K0/V0, then runs the intra-node ring on it.
        constexpr int M = globals::NUM_DEVICES;
        for (int round = 0; round < num_nodes; ++round) {
            if (round > 0) {
                // Wait for the (round-1)-th peer's KV to land and D2D-copy
                // it into this rank's K0[dev_idx] / V0[dev_idx]. Subsequent
                // intra-node ring stages will rotate it.
                kv_copy_kernel<<<kv_copy_grid, config::NUM_THREADS, 0, stream>>>(
                    KE, /*peer_slot=*/round - 1);
                globals Gb = make_G(round * M);
                barrier_only_kernel<<<1, 32, 0, stream>>>(Gb.barrier, dev_idx);
            }
            for (int local_stage = 0; local_stage < M; ++local_stage) {
                const int stage = round * M + local_stage;
                globals Gs = make_G(stage);
                attn_comm_partial_stage_kernel<<<comm_partial_grid,
                    config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY,
                    stream>>>(Gs, stage);
                // Reduction merges the per-stage partial into the running
                // accumulator. Skip only on the very first stage of the
                // very first round where there is nothing to merge yet.
                if (stage > 0) {
                    attn_reduction_stage_kernel<<<reduction_grid,
                        config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY,
                        stream>>>(Gs);
                }
                barrier_only_kernel<<<1, 32, 0, stream>>>(Gs.barrier, dev_idx);
            }
        }
        (void)num_comm_sms;
    }
}

}  // namespace ring_attn_multinode
