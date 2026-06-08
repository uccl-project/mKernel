#pragma once

/**
 * @file dispatch_gemm_glu_combine.cuh
 * @brief Multi-node fused MoE FFN: dispatch -> gemm1+SwiGLU -> gemm2 -> combine,
 *        in a single persistent kernel. Config, globals, host setup, entrypoint.
 *
 * Builds on the intra-node dispatch pattern (pre_tokens_distributed_tensor +
 * pull-based dispatch + per-row-block barrier counter + per-expert GEMM with
 * NUM_EXPERTS_PER_DEV experts per GPU) and adds the full MoE FFN plus the
 * all-to-all combine back to the token owners. Let M = INTRA_NUM_DEVICES,
 * N = num_nodes: N*M GPUs total, M per node, NUM_EXPERTS_PER_DEV = NUM_EXPERTS/(N*M).
 *
 * The hot path is one launch (`fused()`); the kernel body (dispatch_gemm_glu_combine.cu)
 * runs as:
 *   Dispatch: RDMA-exchange pre_tokens with every remote node (same-index GPU),
 *     land peers via IPC-shared peer_tokens, and pull each expert's rows into
 *     post_tokens (src_node/src_dev/src_token from pull_dispatch_indices). CTA
 *     roles (send / copy / dispatch / compute) overlap this with the GEMMs.
 *   gemm1+SwiGLU (fused): post_tokens @ w1 -> silu(gate)*up -> act, with SwiGLU
 *     applied in the gemm1 epilogue (no h1[M,2I] HBM round-trip).
 *   gemm2: act @ w2 -> y_expert.
 *   Combine: scatter weighted expert outputs to each token owner over IPC, gather
 *     -reduce into y_out, and RDMA the cross-node contributions back; an in-kernel
 *     cross-node handshake makes the combine self-contained (done before exit).
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
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif


namespace moe_dispatch_gemm_glu_combine_multinode {

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int NUM_EPILOGUE_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;
// RDMA transfer chunk size for the inter-node phases.
static constexpr int CHUNK_BYTES = 16 * 1024 * 1024;

#ifndef TK_MOE_NUM_NODES
#define TK_MOE_NUM_NODES 2
#endif

// ============================================================================
// Globals
// ============================================================================

struct globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;       // GPUs per node
    static constexpr int NUM_NODES = TK_MOE_NUM_NODES;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int TOP_K = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS = TK_MOE_NUM_EXPERTS;
    // Each GPU owns NUM_EXPERTS / (NUM_DEVICES * NUM_NODES) experts
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / (NUM_DEVICES * NUM_NODES);

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = 16;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    // Two dbufs: local-node tokens, peer-node tokens (filled via RDMA + IPC copy)
    using pre_tokens_distributed_tensor  = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using peer_tokens_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_local_tensor = dist::local_tensor<bf16, 1, 1, -1, H, token_vec, A_tile>;
    using weights_local_tensor = dist::local_tensor<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_local_tensor = dist::local_tensor<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_local_tensor = dist::local_tensor<int, 1, 1, 1, NUM_EXPERTS>;
    // pull_dispatch_indices now has 3 columns: src_node, src_dev, src_token
    using pull_dispatch_indices_local_tensor = dist::local_tensor<int, 1, 1, -1, 3>;
    using barrier_distributed_tensor = dist::distributed_tensor<dist::local_tensor<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

    pre_tokens_distributed_tensor  pre_tokens;
    peer_tokens_distributed_tensor peer_tokens;       // peer node's pre_tokens, IPC-shared after K1
    post_tokens_local_tensor  post_tokens;
    weights_local_tensor      weights;
    outputs_local_tensor      outputs;
    padded_tokens_per_expert_local_tensor padded_tokens_per_expert;
    pull_dispatch_indices_local_tensor    pull_dispatch_indices;
    barrier_distributed_tensor     barrier;

    const int dev_idx;
    const int node_idx;
    const int num_local_tokens;        // tokens in this GPU's pre_tokens
    const int num_padded_local_tokens; // padded total this GPU dispatches to its experts
    const int num_comm_sms;
    const int num_comp_sms;
    unsigned int *kernel_done;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int NUM_NODES = TK_MOE_NUM_NODES;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int TOP_K = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS = TK_MOE_NUM_EXPERTS;
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / (NUM_DEVICES * NUM_NODES);

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int TOKENS_PER_BLOCK = 16;

    static constexpr int I2 = 2 * I;   // gemm1 output width (gate | up)

    // gemm1 SwiGLU-fusion: gemm1 uses a HALF-width col tile (COL_BLOCK_1=128) so
    // that one task can hold BOTH a gate (col c) and an up (col c+I/COL_BLOCK_1)
    // accumulator simultaneously in registers (2x rt_fl<16,128> = 128 fp32
    // regs/thread, fits the 232 budget) and fuse SwiGLU at store time, writing
    // `act` directly and skipping the h1[M,2I] HBM round-trip + the separate
    // swiglu_phase + bar(2). gemm2 keeps COL_BLOCK=256 unchanged.
    static constexpr int COL_BLOCK_1 = 128;

    using token_vec = sv_bf<H>;
    using act_vec   = sv_bf<I>;
    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;
    // gemm1-fusion tiles (128-wide): B1 = w1 col tile, ACT1 = fused act store tile.
    using B1_tile  = st_bf<RED_BLOCK, COL_BLOCK_1>;
    using ACT1_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK_1>;

    using pre_tokens_distributed_tensor  = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using peer_tokens_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using copy_ready_distributed_tensor  = dist::distributed_tensor<dist::local_tensor<int, -1, -1, -1, -1>, NUM_DEVICES, false>;
    // Tensor-core grouped GEMM tensors (warpgroup-MMA pipeline):
    // gemm1+SwiGLU: post_tokens[M,H] @ w1[e][H,2I] -> silu(gate)*up -> act[M,I]
    // gemm2:        act[M,I]         @ w2[e][I,H]  -> y_expert[M,H]
    using post_tokens_local_tensor = dist::local_tensor<bf16, 1, 1, -1, H, token_vec, A_tile>;
    // w1 gets the 128-wide B1_tile descriptor (gemm1 fusion) in addition to B_tile.
    using w1_local_tensor   = dist::local_tensor<bf16, 1, NUM_EXPERTS_PER_DEV, H, I2, B_tile, B1_tile>;
    // act gets the 128-wide ACT1_tile store descriptor (gemm1 fusion) plus its
    // A_tile gemm2-load descriptor.
    using act_local_tensor  = dist::local_tensor<bf16, 1, 1, -1, I, act_vec, A_tile, ACT1_tile>;
    using w2_local_tensor   = dist::local_tensor<bf16, 1, NUM_EXPERTS_PER_DEV, I, H, B_tile>;
    using y_expert_local_tensor = dist::local_tensor<bf16, 1, 1, -1, H, token_vec, C_tile>;
    using padded_tokens_per_expert_local_tensor = dist::local_tensor<int, 1, 1, 1, NUM_EXPERTS>;
    using pull_dispatch_indices_local_tensor = dist::local_tensor<int, 1, 1, -1, 3>;
    // Per-GPU count of pure-local row_blocks at the head of each of this
    // rank's NUM_EXPERTS_PER_DEV owned experts. Under DISPATCH_LOCAL_FIRST routing
    // these are ready at t=0 (no RDMA dependency); DISPATCH_GEMM_LOCAL_FIRST lets
    // group_gemm_fused drain them in a first pass during RDMA.
    using local_rb_per_expert_local_tensor = dist::local_tensor<int, 1, 1, 1, NUM_EXPERTS_PER_DEV>;
    using barrier_distributed_tensor = dist::distributed_tensor<dist::local_tensor<int, -1, -1, -1, -1>, NUM_DEVICES, false>;
    using sync_barrier_distributed_tensor = dist::barrier_distributed_tensor<INTRA_NUM_DEVICES>;
    // Combine output: per-GPU [num_local_tokens, H] fp32, IPC-shared across the
    // M local GPUs so a compute CTA can atomic-add a weighted expert result into
    // the token's owner GPU (src_dev) via IPC. Zeroed by the caller each epoch.
    using y_out_distributed_tensor = dist::distributed_tensor<dist::local_tensor<float, 1, 1, -1, H>, NUM_DEVICES, false>;
    // Flat (non-vec) view of pre_tokens [2*num_local_tokens, H] for element-wise
    // combine staging via IPC atomicAdd. Lower half = dispatch input (untouched
    // here); upper half [num_local_tokens, 2*num_local_tokens) = combine staging.
    using pre_tokens_flat_dt = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    // Atomic-free gather combine (intra): owner-indexed contribution buffer,
    // [comb_slots, H] bf16, IPC-shared. Producer GPUs store pre-weighted rows into
    // the owner GPU's slot; owner gather-reduces its slots into y_out.
    using comb_buf_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;


    pre_tokens_distributed_tensor pre_tokens;
    peer_tokens_distributed_tensor peer_tokens;
    copy_ready_distributed_tensor copy_ready;
    post_tokens_local_tensor post_tokens;   // gemm1 A  [num_padded, H]
    w1_local_tensor          w1;            // gemm1 B  [E, H, 2I]
    act_local_tensor         act;           // gemm1+SwiGLU out / gemm2 A [num_padded, I]
    w2_local_tensor          w2;            // gemm2 B  [E, I, H]
    y_expert_local_tensor    y_expert;      // gemm2 C  [num_padded, H]
    const __nv_bfloat16* y_expert_ptr;      // [num_padded, H]  (combine read)
    const int*   row_expert_ptr;            // [num_padded] local expert id, -1 = padding
    const float* topk_weights_ptr;          // [num_padded] combine weight per dispatched row
    padded_tokens_per_expert_local_tensor padded_tokens_per_expert;
    pull_dispatch_indices_local_tensor pull_dispatch_indices;
    local_rb_per_expert_local_tensor local_rb_per_expert;
    barrier_distributed_tensor barrier;
    sync_barrier_distributed_tensor sync_barrier;
    y_out_distributed_tensor y_out;
    pre_tokens_flat_dt pre_tokens_flat;   // same buffer as pre_tokens, flat view
    // GATHER combine (intra). Inert when use_gather == 0.
    comb_buf_distributed_tensor comb_buf;
    const int* owner_offset_ptr;  // [num_local_tokens+1] CSR row ptr into comb_buf
    const int* row_to_slot_ptr;   // [num_padded] absolute comb_buf slot per intra row
    const int* row_to_owner_ptr;  // [num_padded] owner local dev (intra) or -1
    int use_gather;               // 1 = atomic-free gather path, 0 = scatter

    // GATHER combine (INTER). Sender-side pre-sum: each producer non-atomic
    // stores w*y into the owner-staging-dev's inter_comb_buf[slot] over IPC, then
    // the owner gather-reduces its CSR range in fp32 (HBM-local) and writes the
    // dense pre_tokens UPPER send row. combine_send/reduce UNCHANGED; wire = 1x.
    // Inert when use_inter_gather == 0 (atomic scatter else-branch stays live).
    comb_buf_distributed_tensor inter_comb_buf;   // [inter_comb_slots, H] bf16
    const int* inter_owner_offset_ptr;  // [num_local_tokens+1] CSR for this gpu as inter-owner
    const int* inter_row_to_slot_ptr;   // [num_padded] abs inter_comb_buf slot per inter row
    const int* inter_row_to_owner_ptr;  // [num_padded] owner-staging dev (inter) or -1
    int use_inter_gather;               // 1 = atomic-free inter gather, 0 = atomic scatter

    bf16 *recv_buf;
    bf16 *peer_tokens_local;
    int pre_tokens_bytes;
    int total_chunks;
    int node_idx;
    int num_nodes;  // total node count (>= 2).
    int dev_idx;
    int num_local_tokens;
    int num_padded_local_tokens;
    int num_send_sms;
    int num_copy_sms;
    int num_dispatch_sms;
    int num_comp_sms;
    // Super-M L2-rasterization tile-decode factor (DGC_SUPER_M env). 1 = byte-identical
    // row-major rollback; >1 = rasterize that many row-blocks per col for weight L2 reuse.
    int super_m;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t *arrival_flags;
    uint32_t epoch;
    // Cross-node combine completion handshake (see combine_xnode_barrier).
    // cross_node_barrier: host-pinned, RDMA-writable stage_barrier slot the peer
    //   writes its epoch into on BARRIER_NOTIFY.
    // xnode_ready_device: HBM mirror so only one CTA polls the PCIe flag.
    // Both null on single-node / when unwired -> handshake is a no-op.
    volatile uint32_t *cross_node_barrier = nullptr;
    uint32_t *xnode_ready_device = nullptr;
    unsigned int *copy_phase_done;
    // In-kernel cleanup counter. The last CTA zeroes per-row-block barriers.
    unsigned int *cleanup_done;
    // Inter-node combine phase: two in-kernel global barriers (count + ready
    // gate per barrier; index [0]=after-stage-zero, [1]=after-stage).
    unsigned int *combine_bar_count;   // [2]
    unsigned int *combine_bar_ready;   // [2]

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
    // gemm1 SwiGLU-fusion pipeline: A[2] (two row-halves, shared by gate+up) +
    // Bg (gate col tile) + Bu (up col tile), all 128-wide. Same 49152 B/stage as
    // pipeline_inputs (A[2]=16384 + Bg=16384 + Bu=16384). Outputs hold the fused
    // act tiles (st_bf<64,128> x2 = 32768, half of pipeline_outputs).
    struct pipeline_inputs1  { A_tile A[2]; B1_tile Bg; B1_tile Bu; };
    struct pipeline_outputs1 { ACT1_tile C[2]; };

};

struct fused_cleanup_globals {
    using barrier_distributed_tensor = dist::distributed_tensor<dist::local_tensor<int, -1, -1, -1, -1>, INTRA_NUM_DEVICES, false>;
    barrier_distributed_tensor barrier;
    int dev_idx;
    int num_row_blocks;
};


// ============================================================================
// Host entrypoint
// ============================================================================

// Forward declaration: kernel body (template<bool Overlap>) lives in
// src/dispatch_gemm.cu. Launched via this thin wrapper so the .cuh entrypoint
// doesn't need the template body in scope.
void launch_fused_dispatch_gemm(const fused_globals& G, cudaStream_t stream);

static unsigned int *g_fused_copy_phase_done[globals::NUM_DEVICES] = {nullptr};
static unsigned int *g_fused_cleanup_done[globals::NUM_DEVICES] = {nullptr};
// Inter-node combine in-kernel barriers: 2 (count, ready) ints per device.
static unsigned int *g_combine_bar_count[globals::NUM_DEVICES] = {nullptr};
static unsigned int *g_combine_bar_ready[globals::NUM_DEVICES] = {nullptr};
// HBM mirror for the cross-node combine handshake (xnode_ready_device). One u32
// per device; written monotonically with the current epoch, never needs reset
// (monotonic >= compare). Allocated once, persists across launches.
static uint32_t *g_combine_xnode_ready[globals::NUM_DEVICES] = {nullptr};

void fused(
    dist::ParallelBuffer &pre_tokens,
    dist::ParallelBuffer &peer_tokens,
    dist::ParallelBuffer &copy_ready,
    at::Tensor &post_tokens,
    at::Tensor &w1,
    at::Tensor &w2,
    at::Tensor &act,
    at::Tensor &y_expert,
    at::Tensor &row_expert,
    at::Tensor &topk_weights,
    at::Tensor &padded_tokens_per_expert,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &local_rb_per_expert,
    dist::ParallelBuffer &barrier,
    dist::ParallelBuffer &sync_barrier,
    dist::ParallelBuffer &y_out,
    dist::ParallelBuffer &comb_buf,
    at::Tensor &owner_offset,
    at::Tensor &row_to_slot,
    at::Tensor &row_to_owner,
    dist::ParallelBuffer &inter_comb_buf,
    at::Tensor &inter_owner_offset,
    at::Tensor &inter_row_to_slot,
    at::Tensor &inter_row_to_owner,
    int use_inter_gather,
    int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_local_tokens,
    int num_padded_local_tokens,
    int num_send_sms,
    int num_copy_sms,
    int num_comm_sms_intra,
    int use_gather,
    int num_nodes,
    int64_t cross_node_barrier_ptr = 0
) {
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    const int pre_tokens_bytes = num_local_tokens * globals::H * 2;
    const int total_chunks = (pre_tokens_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;
    int n_send = std::max(1, num_send_sms);
    int n_copy = std::max(1, num_copy_sms);
    int n_dispatch_req = num_comm_sms_intra;
    // Shape-aware CTA split. Smaller token counts need fewer dispatch CTAs;
    // freed CTAs are assigned to compute. Environment variables below keep
    // these split points tunable without recompilation.
    if (num_local_tokens >= 8192) {
        int nc_131k = 28;
        if (const char* e = std::getenv("MKERNEL_Q3_NC_131K")) {
            int v = std::atoi(e);
            if (v > 0) nc_131k = v;
        }
        if (const char* e = std::getenv("DGC_NC_131K")) {
            int v = std::atoi(e);
            if (v > 0) nc_131k = v;
        }
        n_dispatch_req = nc_131k;
        // gen3-H2: the copy CTAs only poll arrival flags + set copy_ready (zero-copy
        // peer_tokens path = no data move) and the send CTAs push the combine chunks
        // which are RDMA-overlapped here; both are over-provisioned at this
        // compute-bound shape. Trim 4->2 each, freeing 4 SMs to GEMM.
        int nsc_131k = 2;
        if (const char* e = std::getenv("DGC_NSC_131K")) { int v = std::atoi(e); if (v > 0) nsc_131k = v; }
        n_send = nsc_131k; n_copy = nsc_131k;
    } else if (num_local_tokens == 2048) {
        // 32K global-token case.
        int nc_32k = 28;
        if (const char* e = std::getenv("MKERNEL_Q3_NC_32K")) {
            int v = std::atoi(e);
            if (v > 0) nc_32k = v;
        }
        if (const char* e = std::getenv("DGC_NC_32K")) {
            int v = std::atoi(e);
            if (v > 0) nc_32k = v;
        }
        n_dispatch_req = nc_32k;
    } else if (num_local_tokens == 4096) {
        // 64K global-token case.
        int nc_64k = 28;
        if (const char* e = std::getenv("MKERNEL_Q3_NC_64K")) {
            int v = std::atoi(e);
            if (v > 0) nc_64k = v;
        }
        if (const char* e = std::getenv("DGC_NC_64K")) {
            int v = std::atoi(e);
            if (v > 0) nc_64k = v;
        }
        n_dispatch_req = nc_64k;
        // gen3-H2: trim send/copy 4->2 (see 131K branch). Measured -3% at 65536.
        int nsc_64k = 2;
        if (const char* e = std::getenv("DGC_NSC_64K")) { int v = std::atoi(e); if (v > 0) nsc_64k = v; }
        n_send = nsc_64k; n_copy = nsc_64k;
    } else if (num_local_tokens == 512) {
        // 8K global-token overlap case.
        int nc_8k = 32;
        if (const char* e = std::getenv("MKERNEL_Q3_NC_8K")) {
            int v = std::atoi(e);
            if (v > 0) nc_8k = v;
        }
        n_dispatch_req = nc_8k;
        int nsend_8k = 4;
        if (const char* e = std::getenv("MKERNEL_Q3_NSEND_8K")) {
            int v = std::atoi(e);
            if (v > 0) nsend_8k = v;
        }
        n_send = std::max(1, nsend_8k);
    } else if (num_local_tokens == 1024) {
        // 16K global-token overlap case.
        int nc_16k = 28;
        if (const char* e = std::getenv("MKERNEL_Q3_NC_16K")) {
            int v = std::atoi(e);
            if (v > 0) nc_16k = v;
        }
        if (const char* e = std::getenv("DGC_NC_16K")) {
            int v = std::atoi(e);
            if (v > 0) nc_16k = v;
        }
        n_dispatch_req = nc_16k;
        int ncopy_16k = 12;
        if (const char* e = std::getenv("MKERNEL_Q3_NCOPY_16K")) {
            int v = std::atoi(e);
            if (v > 0) ncopy_16k = v;
        }
        n_copy = std::max(1, ncopy_16k);
    }
    int n_dispatch = std::max(1, n_dispatch_req);
    if (n_send + n_copy + n_dispatch >= SM_COUNT)
        n_dispatch = std::max(1, SM_COUNT - n_send - n_copy - 1);
    int n_comp = std::max(1, SM_COUNT - n_send - n_copy - n_dispatch);

    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity, 4);

    if (g_fused_copy_phase_done[dev_idx] == nullptr) {
        cudaMalloc(&g_fused_copy_phase_done[dev_idx], sizeof(unsigned int));
    }
    cudaMemsetAsync(g_fused_copy_phase_done[dev_idx], 0, sizeof(unsigned int), stream);
    if (g_fused_cleanup_done[dev_idx] == nullptr) {
        cudaMalloc(&g_fused_cleanup_done[dev_idx], sizeof(unsigned int));
        cudaMemsetAsync(g_fused_cleanup_done[dev_idx], 0, sizeof(unsigned int), stream);
    }
    // No per-iter memset needed: the kernel resets the counter to 0 on the
    // last-CTA path before exit (atomicExch). Saves a small launch.

    // Inter-node combine in-kernel barriers: 2 (count, ready) ints each. Reset
    // every launch (the kernel resets count, but ready must start at 0).
    if (g_combine_bar_count[dev_idx] == nullptr)
        cudaMalloc(&g_combine_bar_count[dev_idx], 6 * sizeof(unsigned int));
    if (g_combine_bar_ready[dev_idx] == nullptr)
        cudaMalloc(&g_combine_bar_ready[dev_idx], 6 * sizeof(unsigned int));
    cudaMemsetAsync(g_combine_bar_count[dev_idx], 0, 6 * sizeof(unsigned int), stream);
    cudaMemsetAsync(g_combine_bar_ready[dev_idx], 0, 6 * sizeof(unsigned int), stream);

    // Cross-node combine handshake HBM mirror. Monotonic (>= epoch) — no per-iter
    // reset needed; allocate-and-zero once.
    if (g_combine_xnode_ready[dev_idx] == nullptr) {
        cudaMalloc(&g_combine_xnode_ready[dev_idx], sizeof(uint32_t));
        cudaMemsetAsync(g_combine_xnode_ready[dev_idx], 0, sizeof(uint32_t), stream);
    }

    // Super-M L2-rasterization decode factor. 8 gives the best weight-reuse
    // locality here; bit-identical to plain row-major decode for any value.
    // Env DGC_SUPER_M overrides (set =1 for plain row-major decode).
    int super_m_val = 8;
    if (const char* e = std::getenv("DGC_SUPER_M")) {
        int v = std::atoi(e);
        if (v >= 1) super_m_val = v;
    }

    fused_globals GF{
        .pre_tokens = ::dist::distributed_tensor_from_buffer<fused_globals::pre_tokens_distributed_tensor>(pre_tokens),
        .peer_tokens = ::dist::distributed_tensor_from_buffer<fused_globals::peer_tokens_distributed_tensor>(peer_tokens),
        .copy_ready = ::dist::distributed_tensor_from_buffer<fused_globals::copy_ready_distributed_tensor>(copy_ready),
        .post_tokens = ::dist::local_tensor_from_tensor<fused_globals::post_tokens_local_tensor>(post_tokens),
        .w1 = ::dist::local_tensor_from_tensor<fused_globals::w1_local_tensor>(w1),
        .act = ::dist::local_tensor_from_tensor<fused_globals::act_local_tensor>(act),
        .w2 = ::dist::local_tensor_from_tensor<fused_globals::w2_local_tensor>(w2),
        .y_expert = ::dist::local_tensor_from_tensor<fused_globals::y_expert_local_tensor>(y_expert),
        .y_expert_ptr = reinterpret_cast<const __nv_bfloat16*>(y_expert.data_ptr()),
        .row_expert_ptr = row_expert.data_ptr<int>(),
        .topk_weights_ptr = topk_weights.data_ptr<float>(),
        .padded_tokens_per_expert =
            ::dist::local_tensor_from_tensor<fused_globals::padded_tokens_per_expert_local_tensor>(padded_tokens_per_expert),
        .pull_dispatch_indices =
            ::dist::local_tensor_from_tensor<fused_globals::pull_dispatch_indices_local_tensor>(pull_dispatch_indices),
        .local_rb_per_expert =
            ::dist::local_tensor_from_tensor<fused_globals::local_rb_per_expert_local_tensor>(local_rb_per_expert),
        .barrier = ::dist::distributed_tensor_from_buffer<fused_globals::barrier_distributed_tensor>(barrier),
        .sync_barrier =
            ::dist::distributed_tensor_from_buffer<fused_globals::sync_barrier_distributed_tensor>(sync_barrier),
        .y_out = ::dist::distributed_tensor_from_buffer<fused_globals::y_out_distributed_tensor>(y_out),
        .pre_tokens_flat = ::dist::distributed_tensor_from_buffer<fused_globals::pre_tokens_flat_dt>(pre_tokens),
        .comb_buf = ::dist::distributed_tensor_from_buffer<fused_globals::comb_buf_distributed_tensor>(comb_buf),
        .owner_offset_ptr = owner_offset.data_ptr<int>(),
        .row_to_slot_ptr = row_to_slot.data_ptr<int>(),
        .row_to_owner_ptr = row_to_owner.data_ptr<int>(),
        .use_gather = use_gather,
        .inter_comb_buf = ::dist::distributed_tensor_from_buffer<fused_globals::comb_buf_distributed_tensor>(inter_comb_buf),
        .inter_owner_offset_ptr = inter_owner_offset.data_ptr<int>(),
        .inter_row_to_slot_ptr = inter_row_to_slot.data_ptr<int>(),
        .inter_row_to_owner_ptr = inter_row_to_owner.data_ptr<int>(),
        .use_inter_gather = use_inter_gather,
        .recv_buf = reinterpret_cast<bf16*>(recv_buf_ptr),
        .peer_tokens_local = reinterpret_cast<bf16*>(peer_tokens.data_.data_ptr()),
        .pre_tokens_bytes = pre_tokens_bytes,
        .total_chunks = total_chunks,
        .node_idx = node_idx,
        .num_nodes = num_nodes,
        .dev_idx = dev_idx,
        .num_local_tokens = num_local_tokens,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_send_sms = n_send,
        .num_copy_sms = n_copy,
        .num_dispatch_sms = n_dispatch,
        .num_comp_sms = n_comp,
        .super_m = super_m_val,
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
        .cross_node_barrier = (num_nodes > 1 && cross_node_barrier_ptr)
            ? reinterpret_cast<volatile uint32_t*>(cross_node_barrier_ptr) : nullptr,
        .xnode_ready_device = (num_nodes > 1 && cross_node_barrier_ptr)
            ? g_combine_xnode_ready[dev_idx] : nullptr,
        .copy_phase_done = g_fused_copy_phase_done[dev_idx],
        .cleanup_done = g_fused_cleanup_done[dev_idx],
        .combine_bar_count = g_combine_bar_count[dev_idx],
        .combine_bar_ready = g_combine_bar_ready[dev_idx],
    };

    // Dispatch-side local-first walk: dispatch fills local row_blocks during
    // Phase 1's RDMA wait, then peer row_blocks. Compute Pass 0 streams
    // through them while Inter-Recv is still ferrying chunks. Per-chunk
    // copy_ready polling inside dispatch_fused gates each peer token.
    launch_fused_dispatch_gemm(GF, stream);
}

}  // namespace moe_dispatch_gemm_glu_combine_multinode
