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
 * Original design notes (from src/ring_attention.cu pre-split):
 *
 *  Multi-node Ring Attention.
 *
 *  Algorithm (16 logical ring stages = 8 local + 1 inter-node hop + 8 local):
 *    Stage 0:        Each GPU computes attention(Q, local KV in K0/V0) -> O, L
 *    Stages 1..7:    Intra-node ring rotation around the 8 local GPUs.
 *    Stage 7.5:      Inter-node RDMA exchange — each GPU sends K_local/V_local
 *                    to peer node's same-index GPU; peer receives into
 *                    K_recv/V_recv; D2D-copy into K0/V0.
 *    Stages 8..15:   Intra-node ring rotation again on peer KV.
 *    Final reduction: Merge per-stage partials via online softmax.
 *
 *  The intra-node ring_attn primitive (attn_partial / attn_comm / attn_reduction)
 *  is invoked repeatedly with an RDMA hop at the midpoint.
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
    // Register split: P + N·C ≤ 512 (per-CTA reg file).
    //   N=3 consumers: 24 + 3×160 = 504 — fits the budget. An earlier attempt
    //   at 24/168 silently exceeded it (24+3×168 = 528 > 512) and deadlocked
    //   ROLE_SPLIT — the 3rd consumer's setmaxnreg.inc could never be
    //   satisfied because the per-CTA register file (65536) was fully claimed
    //   by producer + first 2 consumers. Compiler accepted the over-
    //   allocation; the runtime hung. Fixed 2026-04-28.
    static constexpr int PRODUCER_REGISTERS = 24;
    static constexpr int CONSUMER_REGISTERS = 160;
};

// ============================================================================
// Globals — same shape as intranode ring_attention globals
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

// Tiny launch config for the cross-GPU barrier-only kernel (1 CTA, 32 threads).
struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

// ============================================================================
// Inter-node KV exchange globals
// ============================================================================
//
// Each GPU does:
//   1. Push RDMA WR to send K_local + V_local to peer node's same-index GPU
//   2. Wait for peer's K + V to arrive in K_recv/V_recv
//   3. D2D copy K_recv → K0[dev_idx] (DistBuffer slot for this GPU)
//      D2D copy V_recv → V0[dev_idx]
//
// After this kernel, K0/V0 contain peer node's KV (originally for this GPU's
// counterpart on the other node). The next ring round of stages 8..15 will
// rotate this KV around the local 8-GPU ring just like stages 0..7.

struct kv_exchange_globals {
    bf16 *send_buf;          // registered [K | V] staging buffer
    // The send buffer (registered with RDMA) holds [K | V] contiguously.
    // The host stages K0_local→send_buf[0:K_bytes], V0_local→send_buf[K_bytes:]
    // before launching this kernel. The kernel pushes FIFO commands referencing
    // local_offset within that send buffer.
    bf16 *K_recv;            // points to recv_buf + 0           (peer K)
    bf16 *V_recv;            // points to recv_buf + K_bytes      (peer V)
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
struct zm_barrier_config {
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
__global__ void zm_attn_comm_partial_stage_kernel(
    const __grid_constant__ globals G, const int stage);
__global__ void zm_attn_reduction_stage_kernel(
    const __grid_constant__ globals G);
__global__ void zm_kv_send_kernel(
    const __grid_constant__ kv_exchange_globals KE);
__global__ void zm_kv_copy_kernel(
    const __grid_constant__ kv_exchange_globals KE);
__global__ void zm_barrier_only_kernel(
    const __grid_constant__ globals::barrier_distributed_tensor barrier, const int dev_idx);

// ============================================================================
// Host orchestration entrypoint
// ============================================================================
//
// Sequence (per call):
//   1. Launch zm_kv_send_kernel (RDMA prologue — fires WRs, returns immediately
//      so EFA transfers overlap with stages 0..7)
//   2. Loop ring_stage = 0..7:
//        - Launch zm_attn_comm_partial_stage_kernel(ring_stage) [intranode ring]
//        - If ring_stage > 0: launch zm_attn_reduction_stage_kernel
//        - Launch zm_barrier_only_kernel (cross-GPU barrier_all)
//   3. Launch zm_kv_copy_kernel (wait for peer KV arrival + D2D copy → K0/V0)
//   4. Loop ring_stage = 8..15: same as step 2 but on peer-sourced KV.
//
// The intra-node ring naturally rotates K0↔K1 / V0↔V1 every stage; we re-use
// the same K0/K1 buffers for both rounds. zm_kv_copy_kernel re-initializes
// K0/V0 to peer's KV before the second round.

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
    int num_nodes = 2  // total node count (>= 2).
) {
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    // Construct intranode globals (used for all attn_partial / attn_comm calls)
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
        .K_recv = reinterpret_cast<bf16*>(recv_buf_ptr),
        .V_recv = reinterpret_cast<bf16*>(recv_buf_ptr + K_bytes),
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

        // Per-stage launch path: matches intranode design. Each launch isolates
        // its register frame so attn_partial / attn_comm / attn_reduction get
        // STACK:0. A fused persistent variant was tried and removed — the role
        // union forced ~150 bytes of silent spills in a single kernel that no
        // amount of __noinline__ refactoring or FA3 layout could eliminate
        // without halving wgmma throughput.
        cudaFuncSetAttribute(zm_attn_comm_partial_stage_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY);
        cudaFuncSetAttribute(zm_attn_reduction_stage_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY);

        int comm_partial_grid = num_comm_sms + n_partial;
        int reduction_grid    = n_reduction;
        int kv_send_grid      = n_send;
        int kv_copy_grid      = n_send + n_copy;


        // Prologue: post RDMA WRs so peer-KV transfer overlaps with stages 0-7.
        // kv_send/kv_copy don't use dynamic smem — launch with smem=0.
        zm_kv_send_kernel<<<kv_send_grid, config::NUM_THREADS, 0, stream>>>(KE);
        // Match persistent kernel: barrier_all after send, before stages start.
        {
            globals Gb = make_G(0);
            zm_barrier_only_kernel<<<1, 32, 0, stream>>>(Gb.barrier, dev_idx);
        }

        // Round A: ring stages 0..7 on local KV.
        for (int stage = 0; stage < 8; ++stage) {
            globals Gs = make_G(stage);
            zm_attn_comm_partial_stage_kernel<<<comm_partial_grid, config::NUM_THREADS,
                config::DYNAMIC_SHARED_MEMORY, stream>>>(Gs, stage);
            if (stage > 0) {
                zm_attn_reduction_stage_kernel<<<reduction_grid, config::NUM_THREADS,
                    config::DYNAMIC_SHARED_MEMORY, stream>>>(Gs);
            }
            zm_barrier_only_kernel<<<1, 32, 0, stream>>>(Gs.barrier, dev_idx);
        }

        // Stage 8 prep: wait for peer KV to land, D2D-copy into K0/V0.
        zm_kv_copy_kernel<<<kv_copy_grid, config::NUM_THREADS, 0, stream>>>(KE);
        {
            globals Gb = make_G(0);
            zm_barrier_only_kernel<<<1, 32, 0, stream>>>(Gb.barrier, dev_idx);
        }

        // Round B: ring stages 8..15 on peer KV.
        for (int stage = 8; stage < 16; ++stage) {
            globals Gs = make_G(stage);
            zm_attn_comm_partial_stage_kernel<<<comm_partial_grid, config::NUM_THREADS,
                config::DYNAMIC_SHARED_MEMORY, stream>>>(Gs, stage);
            // Reduction always runs in round B (stage > 0 condition is always true).
            zm_attn_reduction_stage_kernel<<<reduction_grid, config::NUM_THREADS,
                config::DYNAMIC_SHARED_MEMORY, stream>>>(Gs);
            zm_barrier_only_kernel<<<1, 32, 0, stream>>>(Gs.barrier, dev_idx);
        }
        (void)num_comm_sms;
    }
}

}  // namespace ring_attn_multinode
