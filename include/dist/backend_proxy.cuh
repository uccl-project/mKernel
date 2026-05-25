/**
 * @file
 * @brief Proxy-backend method bodies for dist::dbuf inter-node API.
 *
 * Wraps the existing `internode::D2HFifoDevice` (CPU-proxy WQE post path).
 * Compute CTAs hand tiles to channel comm CTAs via a single-producer
 * single-consumer ring; the comm CTA
 * forwards each job to the proxy as a WRITE_WITH_IMM transfer command.
 * The CPU proxy issues the actual RDMA WRITE_WITH_IMM with `src_view=1`,
 * so the data path is DMA-BUF zero-copy: NIC reads directly from the
 * dbuf's local data buffer (registered as MR at bind time).
 *
 * Receive side: the peer's NIC writes WRITE_WITH_IMM into our remote
 * `arrived[]` flag array (host-pinned, MR-registered for the peer). GPU
 * compute CTAs spin on `arrived[t].v >= expected_iter`. The CPU proxy
 * also reposts recv WQEs in batches as it drains its CQ.
 *
 * `drain_step` is therefore a no-op on the GPU side under this backend.
 */

#pragma once

#include "distributed_buffer.cuh"
#include "../comm/atomic_u32.cuh"
#include "../comm/internode/d2h_fifo.cuh"
#include "../comm/internode/types.h"
#if defined(INTERNODE_BACKEND_EFA) || defined(INTERNODE_BACKEND_IBVERBS)
#include "../comm/internode/session_select.h"
#else
// Keep editor tooling usable when compile-time backend defines are absent.
#include "../comm/internode/session_efa.h"
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

namespace dist {

/* ----------   Device-side method bodies  ---------- */

template<typename GL, int LOCAL_SIZE, bool MULTICAST,
         int NUM_CHANNELS, int NUM_NODES, typename... TMA_Types>
__device__ inline void
distributed_tensor<GL, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>
::enqueue_send(int channel, uint32_t tile_id, uint16_t dst_node,
               uint64_t off, uint32_t bytes, uint32_t imm) const {
    const auto& ch = channels[channel];
    const uint32_t mask = ch.ring_capacity - 1u;

    // Single-producer head: own write, no atomic needed.
    uint32_t h = ch.head->v;

    // Backpressure: spin until consumer (comm CTA) has caught up.
    uint32_t t;
    while (true) {
        t = comm::atomic_u32::acquire_load_gpu(&ch.tail->v);
        if (h - t < ch.ring_capacity) break;
        __nanosleep(64);
    }

    SendJob* slot = &ch.ring[h & mask];
    slot->tile_id  = tile_id;
    slot->bytes    = bytes;
    slot->off      = off;
    slot->imm      = imm;
    slot->dst_node = dst_node;

    // Release-store the new head so the comm CTA's acquire-load sees the
    // fully-written job before the index advance.
    comm::atomic_u32::release_store_gpu(&ch.head->v, h + 1u);
}

template<typename GL, int LOCAL_SIZE, bool MULTICAST,
         int NUM_CHANNELS, int NUM_NODES, typename... TMA_Types>
__device__ inline bool
distributed_tensor<GL, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>
::try_dequeue_send(int channel, SendJob* out) const {
    const auto& ch = channels[channel];
    const uint32_t mask = ch.ring_capacity - 1u;

    // Single-consumer tail.
    uint32_t t = ch.tail->v;
    uint32_t h;
    h = comm::atomic_u32::acquire_load_gpu(&ch.head->v);
    if (t == h) return false;

    *out = ch.ring[t & mask];

    comm::atomic_u32::release_store_gpu(&ch.tail->v, t + 1u);
    return true;
}

template<typename GL, int LOCAL_SIZE, bool MULTICAST,
         int NUM_CHANNELS, int NUM_NODES, typename... TMA_Types>
__device__ inline void
distributed_tensor<GL, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>
::put_inter(int channel, int dst_node,
            uint64_t local_off, uint64_t remote_off,
            uint32_t bytes, uint32_t imm, uint32_t lane_id) const {
    const auto& ch = channels[channel];

    // Build a TransferCmd. src_view=1 tells the proxy to post a single-SGE
    // WQE pointing at the local DMA-BUF MR — no pack, no staging, true
    // zero-copy from the kernel's own data buffer.
    internode::TransferCmd cmd{};
    cmd.cmd_type      = internode::CmdType::WRITE;
    cmd.dst_rank      = (uint8_t)dst_node;
    cmd.tile_id = (uint32_t)imm;
    cmd.bytes         = bytes;
    cmd.local_offset = (uint64_t)local_off;
    cmd.remote_offset = (uint64_t)remote_off;
    cmd.lane_id       = (uint16_t)lane_id;
    cmd.src_view      = (uint8_t)ch.mode;

    // Pick the right FIFO for this lane: the proxy that owns the destination QP.
    const auto& bundle = *static_cast<const internode::D2HFifoDeviceBundle*>(ch.fifo_bundle);
    internode::D2HFifoDevice fifo =
        internode::q2_select_fifo_for_lane(bundle, lane_id);
    fifo.push(cmd);
}

template<typename GL, int LOCAL_SIZE, bool MULTICAST,
         int NUM_CHANNELS, int NUM_NODES, typename... TMA_Types>
__device__ inline void
distributed_tensor<GL, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>
::flush_inter(int /*channel*/) const {
    // CPU-proxy backend: no doorbell to ring on the GPU side. The CPU proxy
    // already amortizes WQE posts internally as it drains the FIFO.
}

template<typename GL, int LOCAL_SIZE, bool MULTICAST,
         int NUM_CHANNELS, int NUM_NODES, typename... TMA_Types>
__device__ inline void
distributed_tensor<GL, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>
::drain_step(int /*channel*/) const {
    // CPU-proxy backend: the CPU thread drains the receive CQ and writes
    // arrived[].v on each WRITE_WITH_IMM completion. Nothing to do here.
}

/* ----------   Host-side binding (zero-copy by default)  ---------- */

/**
 * @brief Wire a dbuf's channels[] onto an already-created session's resources.
 *
 * Use this when the session was created externally (e.g. by an existing
 * `create_session_py` Python entry point). The `mode` argument selects the
 * send strategy:
 *
 *   - `SendMode::DmabufDirect` (zero-copy, default): asserts that every rail
 *     has a non-null `clocal_data_mr` — i.e., the session was created with
 *     `direct_dmabuf_enabled=true` and `local_gpu_buf` set to the dbuf's
 *     data. `put_inter` will produce `cmd.src_view=1` (single-SGE direct
 *     read from the local DMA-BUF MR).
 *   - `SendMode::Staging`: no MR check; proxy reads from a caller-provided
 *     staging buffer. The kernel must pack data into staging before
 *     `put_inter`. Compatible with kernels that don't have working DMA-BUF.
 *   - `SendMode::StridedGather`: zero-copy multi-SGE gather (advanced).
 *
 * `bind_inter_proxy` internally creates the session and then calls this.
 */
template<typename DBUF>
__host__ inline void attach_channels_proxy(
    DBUF& d,
    const internode::Session& session,
    TileFlag* arrived_flags,
    int tiles_per_channel,
    int ring_capacity,
    SendMode mode = SendMode::DmabufDirect
) {
    using ChannelT = typename std::remove_reference_t<decltype(d.channels[0])>;
    using IndexT   = typename ChannelT::Index;

    if ((ring_capacity & (ring_capacity - 1)) != 0)
        throw std::runtime_error("dist::attach_channels_proxy: ring_capacity must be a power of two");
    if (session.fifo_bundle.num_fifos <= 0)
        throw std::runtime_error("dist::attach_channels_proxy: empty D2HFifoDeviceBundle");

    // Zero-copy modes require DMA-BUF MR on every rail. Staging mode skips
    // this check — its proxy reads from a separately-registered buffer.
    if (mode == SendMode::DmabufDirect || mode == SendMode::StridedGather) {
        for (int r = 0; r < session.num_rails; ++r) {
            if (!session.rails[r].clocal_data_mr) {
                throw std::runtime_error(
                    "dist::attach_channels_proxy: rail " + std::to_string(r) +
                    " has no clocal_data_mr — zero-copy mode requires session "
                    "created with direct_dmabuf_enabled=true and local_gpu_buf "
                    "set to the dbuf's data. Pass SendMode::Staging if your "
                    "kernel packs into a staging buffer instead.");
            }
        }
    }

    for (int c = 0; c < DBUF::num_channels; ++c) {
        auto& ch = d.channels[c];

        // Each channel carries a pointer to the shared bundle; put_inter picks
        // the right FIFO per send based on lane_id.
        // This preserves the proxy's lane_id -> QP routing semantics.
        ch.fifo_bundle = (void*)&session.fifo_bundle;

        // Send-ring storage (single-producer single-consumer).
        cudaMalloc(&ch.ring, (size_t)ring_capacity * sizeof(SendJob));
        cudaMemset(ch.ring,  0, (size_t)ring_capacity * sizeof(SendJob));
        cudaMalloc(&ch.head, sizeof(IndexT));
        cudaMalloc(&ch.tail, sizeof(IndexT));
        cudaMemset(ch.head, 0, sizeof(IndexT));
        cudaMemset(ch.tail, 0, sizeof(IndexT));
        ch.ring_capacity = (uint32_t)ring_capacity;

        // Arrival flag slot range for this channel within the shared array.
        ch.arrived            = arrived_flags + (size_t)c * tiles_per_channel;
        ch.tiles_per_channel  = (uint32_t)tiles_per_channel;

        // Send mode picks the proxy's WQE construction strategy at put_inter
        // time. All channels of a dbuf use the same mode by default.
        ch.mode               = mode;
    }
}

/**
 * @brief Create an internode session for zero-copy dbuf sends.
 *
 * Requires DMA-BUF registration for the dbuf data buffer. `dbuf.put_inter()`
 * sends with `src_view = 1`, so the proxy reads directly from the local MR.
 */
template<typename DBUF>
__host__ inline internode::Session* bind_inter_proxy(
    DBUF& d,
    internode::SessionConfig cfg,
    void*  dbuf_data_ptr,
    size_t dbuf_data_bytes,
    TileFlag* arrived_flags,
    int    tiles_per_channel,
    int    ring_capacity,
    SendMode mode = SendMode::DmabufDirect
) {
    static_assert(DBUF::num_channels > 0,
                  "bind_inter_proxy requires NUM_CHANNELS > 0");

    // Zero-copy modes (DmabufDirect, StridedGather) force the session to
    // register the dbuf's data buffer as a DMA-BUF MR on every rail. Staging
    // mode leaves direct_dmabuf_enabled at the caller's setting (typically
    // false) — the proxy will read from a separately-managed staging buffer.
    if (mode == SendMode::DmabufDirect || mode == SendMode::StridedGather) {
        cfg.direct_dmabuf_enabled = true;
        cfg.local_gpu_buf         = dbuf_data_ptr;
        cfg.local_gpu_buf_size    = dbuf_data_bytes;
    }

    internode::Session* session = internode::create_session(cfg);
    if (!session)
        throw std::runtime_error("bind_inter_proxy: create_session failed");

    if (mode == SendMode::DmabufDirect || mode == SendMode::StridedGather) {
        for (int r = 0; r < session->num_rails; ++r) {
            if (!session->rails[r].clocal_data_mr) {
                internode::destroy_session(session);
                throw std::runtime_error(
                    "bind_inter_proxy: DMA-BUF MR registration failed on rail "
                    + std::to_string(r) + " — zero-copy contract violated");
            }
        }
    }

    attach_channels_proxy(d, *session, arrived_flags,
                          tiles_per_channel, ring_capacity, mode);
    return session;
}

template<typename DBUF>
__host__ inline void unbind_inter_proxy(DBUF& d) {
    for (int c = 0; c < DBUF::num_channels; ++c) {
        auto& ch = d.channels[c];
        if (ch.ring) { cudaFree(ch.ring); ch.ring = nullptr; }
        if (ch.head) { cudaFree(ch.head); ch.head = nullptr; }
        if (ch.tail) { cudaFree(ch.tail); ch.tail = nullptr; }
    }
}

} // namespace dist
