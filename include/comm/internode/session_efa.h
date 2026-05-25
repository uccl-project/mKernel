/**
 * @file session_efa.h
 * @brief EFA session lifecycle for inter-node RDMA communication.
 *
 * Same API as session.h but uses EFA SRD transport instead of RC.
 *
 * This backend uses direct verbs (`efadv`) for EFA while keeping the same
 * GPU-resident buffer model as the other internode backends. It mirrors the
 * CX7 session's multi-knob layout:
 *   - num_qps            : multiple SRD QPs pointing at the same peer AH
 *   - num_proxy_threads  : N host proxy threads, each owning a QP slice
 *   - logical_queues_per_qp : software queue multiplexing per QP
 *
 * The only differences from session.h are the transport-level details (SRD
 * address handles, QP state transitions, device selection) and that the
 * arrival flags live in host-mapped memory (the SRD receive path uses
 * RDMA_WRITE_WITH_IMM CQEs to publish flags from the host proxy).
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "types.h"
#include "rdma_transport_efa.h"
#include "rdma_gpu_mr.cuh"
#include "d2h_fifo.cuh"
#include "arrival.cuh"
#include "ready_queue.cuh"
#include "proxy_efa.h"
#include "proxy_diagnostics.h"


namespace internode {

// Match CX7: two stage-barrier slots per session.
static constexpr int kEfaStageBarrierSlots = 2;
// Cap total QPs per session (mirrors CX7's kMaxQPs/kMaxExchangeQPs).
static constexpr int kEfaMaxQPs = kMaxExchangeQPs;
// Operator session.cuh code (shared across CX7 + EFA backends) references
// internode::kMaxQPs when sizing num_qps; EFA aliases it to kEfaMaxQPs so
// the same code compiles against either backend.
static constexpr int kMaxQPs = kEfaMaxQPs;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// kMaxPeers lives in types.h so proxy_efa.h can use it independently.

struct SessionConfig {
    int         rank;
    // peer_ips[i] / peer_tcp_ports[i] describe peer slot i (i in
    // [0, num_peers)). Peer slots are local to this session — they do NOT
    // correspond to global ranks directly. The proxy maps cmd.dst_rank →
    // peer slot via peer_slot_of_rank.
    int                num_peers       = 0;
    const char* const* peer_ips        = nullptr;
    const int*         peer_tcp_ports  = nullptr;
    // Optional: peer_ranks[i] is the global node rank corresponding to peer
    // slot i. When null, slot i is assumed to map to global rank i for ranks
    // < this->rank, and rank i+1 for ranks >= this->rank (the natural "skip
    // self" ordering). Provided explicitly by callers that use a non-trivial
    // peer ordering.
    const int*         peer_ranks      = nullptr;

    void*       local_gpu_buf;
    size_t      local_gpu_buf_size;

    size_t      recv_buf_size;

    // Zero-copy receive: when non-null, create_session does NOT allocate its
    // own recv_buf via cudaMalloc. Instead it registers this externally-owned
    // GPU buffer (e.g. an IPC-shared peer_tokens slot) on every rail's PD and
    // exchanges its rkey + remote address over TCP. RDMA writes from peers
    // land directly into this buffer — receivers can read it via IPC without
    // an intermediate D2D copy. Caller owns the lifetime of the buffer.
    void*       external_recv_buf       = nullptr;
    int         num_tiles;
    int         fifo_capacity;
    int         device_id;

    int         max_inflight;

    // Optional second source buffer (e.g. `output_local`) registered as a
    // DMA-BUF MR on every rail's PD. When populated + direct_dmabuf_enabled,
    // the GPU kernel can set cmd.src_view = 1 and compute cmd.local_offset
    // into this buffer; the proxy sends directly from HBM without an
    // intermediate pack into the staging buffer.
    void*       clocal_gpu_buf          = nullptr;
    size_t      clocal_gpu_buf_size     = 0;
    bool        direct_dmabuf_enabled   = false;

    // Kept for config parity with session.h's CX7 backend so callers that
    // set row_stride_bytes (for src_view=2 strided sends) compile cleanly
    // even though EFA SRD (max_sge=2) doesn't implement strided gather.
    size_t      row_stride_bytes        = 0;

    // Performance knobs — same semantics as session.h.
    int         num_qps                = 1;
    int         logical_queues_per_qp  = 1;
    int         num_proxy_threads      = 1;

    // Unused on SRD but kept for config parity with the ibverbs backend.
    bool        use_write_imm          = false;
    int         ready_queue_cap        = 0;
    bool        use_arrival_queue      = false;

    // Multi-rail: number of EFA NICs to use per GPU session. Each rail is a
    // separate ibv_context with its own PD, CQs, QPs, and MR registrations.
    // QPs are distributed round-robin across rails. Each proxy thread handles
    // QPs from exactly one rail. Default 2 uses all 16 NICs on p5 (2 per GPU).
    int         num_rails              = 2;

    // Whether to pin each proxy thread to the GPU's NUMA node (default true).
    bool        pin_proxy              = true;

    // Accepted for source compatibility with CX7's SessionConfig. EFA's proxy
    // does not route by TransferCmd::reserved0, so the only effect on EFA is
    // that operator session.cuh code bumps num_qps when the flag is true
    // (capped to kEfaMaxQPs in create_session()).
    bool        channelize_gpu_peers   = false;

    // Accepted for source compatibility with CX7's SessionConfig. EFA's proxy
    // does not implement the >2-node forward-notify hop, so the field is
    // parsed but ignored on this backend.
    bool        enable_forward_notify  = false;
};

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

// Per-rail RDMA resources. Each rail is a separate ibv_context (NIC device)
// with its own PD, CQs, QPs, and MR registrations.
struct RailResources {
    ibv_context* ctx  = nullptr;
    ibv_pd*      pd   = nullptr;
    ibv_mr*      local_data_mr  = nullptr;  // same GPU buf, registered on this rail's PD
    ibv_mr*      clocal_data_mr = nullptr;  // direct-DMA-BUF MR for output_local (optional)
    ibv_mr*      recv_buf_mr    = nullptr;  // same recv buf, registered on this rail's PD
    ibv_mr*      arrival_mr     = nullptr;  // arrival flags, registered on this rail's PD
    ibv_mr*      staging_mrs[kMaxProxyThreads] = {};  // per-proxy staging MRs
    ibv_mr*      barrier_mr     = nullptr;  // stage-barrier, registered on this rail's PD
    ibv_ah*      dst_ah         = nullptr;  // slot-0 alias of dst_ah_per_peer[0] (CX7 ProxyConfig compat)
    ibv_ah*      dst_ah_per_peer[kMaxPeers] = {};  // AH per peer slot, on this rail's PD
};

struct Session {
    // Multi-rail: each rail is a separate NIC device with its own RDMA resources.
    // Rail 0 is the "primary" rail; its ctx/pd/cq are aliased by the legacy
    // single-rail fields below for minimal churn in non-session code.
    int           num_rails;
    RailResources rails[kMaxRails];

    // Legacy aliases — point to rail 0's resources for backwards compatibility.
    ibv_context*  ctx;   // = rails[0].ctx
    ibv_pd*       pd;    // = rails[0].pd

    // Primary (index 0) QP / CQ. Extra QPs live in `extra_qps`; per-proxy CQs
    // live in `proxy_cqs` — proxy_cqs[0] aliases cq.
    ibv_cq*       cq;
    ibv_qp*       qp;
    ibv_qp*       extra_qps[kEfaMaxQPs - 1];
    ibv_cq*       proxy_cqs[kMaxProxyThreads];
    int           qp_rail[kEfaMaxQPs];       // which rail each QP belongs to

    // Per-QP destination QPN. Slot-0 alias of dst_qpns_per_peer[0][:];
    // populated for ProxyConfig source-compat with the CX7 backend, which
    // reads this scalar on its non-channelized path. The EFA proxy hot
    // path reads dst_qpns_per_peer[peer_slot] directly.
    uint32_t      dst_qpns[kEfaMaxQPs];

    // Multi-peer destination QPN table, indexed [peer_slot][qp_idx]. The
    // EFA proxy hot path resolves peer_slot from cmd.dst_rank and reads
    // dst_qpns_per_peer[peer_slot][qp_idx].
    int           num_peers;
    uint32_t      dst_qpns_per_peer[kMaxPeers][kEfaMaxQPs];
    // Per-peer remote_info, indexed by peer slot.
    ConnectionInfo remote_infos[kMaxPeers];
    // Map from this->rank to peer slot. Indexed by global node rank, returns
    // peer slot in [0, num_peers) or -1 if rank == self. Populated by
    // create_session() from cfg.peer_ranks (or the implicit "skip self"
    // ordering when peer_ranks is null).
    int           peer_slot_by_rank[kMaxPeers + 1];

    // Translate a cmd.dst_rank (global node rank) to a peer slot for
    // dst_qpns_per_peer / remote_infos lookup. peer_slot_by_rank is
    // populated for every session (slot 0 holds the single peer at N=2).
    inline int peer_slot_of_rank(int dst_rank) const {
        if (dst_rank < 0 || dst_rank > kMaxPeers) return 0;
        int s = peer_slot_by_rank[dst_rank];
        return s < 0 ? 0 : s;
    }

    // Session-wide counters used by create_session + diagnostics.
    int           num_qps;
    int           num_proxy_threads;
    int           logical_queues_per_qp;

    // Receive buffer (allocated on GPU, RDMA-registered for remote writes).
    // Registered on every rail's PD; the per-rail MRs live in rails[r].recv_buf_mr.
    gpu_mr::GpuRdmaBuffer recv_buf;

    // D2H command FIFOs (one per proxy thread / FIFO channel).
    D2HFifoPair         fifos[kMaxProxyThreads];
    D2HFifoDeviceBundle fifo_bundle;

    // Arrival flags + per-proxy metadata staging.
    ArrivalFlags        arrival;
    FlagStaging         flag_stagings[kMaxProxyThreads];
    StageBarrierFlags   stage_barrier;

    // CPU proxy threads.
    Proxy*        proxies[kMaxProxyThreads];
    // Per-proxy storage for the multi-peer table that the proxy hot path
    // reads when num_peers > 1. Each proxy thread owns QPs on a single
    // rail; per_proxy_peer[t][p] holds the AH/dst_qpns/remote_* for peer
    // slot p as seen from proxy t's rail. Lifetime is the Session's.
    PerPeerProxyData per_proxy_peer[kMaxProxyThreads][kMaxPeers];
    // Per-proxy fifo pointer storage (used when num_fifos > num_proxy_threads).
    // Each proxy[t] gets a contiguous slice of fifos via per_proxy_fifo_ptrs[t].
    D2HFifoHost*  per_proxy_fifo_ptrs[kMaxProxyThreads][kMaxProxyThreads];


    // Remote connection info (populated after TCP exchange).
    ConnectionInfo remote_info;

    int rank;
    int sq_depth;
    uint32_t epoch;
};

// ---------------------------------------------------------------------------
// Create / Destroy
// ---------------------------------------------------------------------------

inline Session* create_session(const SessionConfig& cfg) {
    Session* s = new Session{};
    s->rank = cfg.rank;
    s->epoch = 1;

    // Clamp knobs and derive session-wide counters.
    int num_rails = cfg.num_rails <= 0 ? 1 : cfg.num_rails;
    if (num_rails > kMaxRails) num_rails = kMaxRails;
    // Allow MKERNEL_EFA_NUM_RAILS env override.
    {
        const char* env = std::getenv("MKERNEL_EFA_NUM_RAILS");
        if (env && env[0]) {
            int v = std::atoi(env);
            if (v >= 1 && v <= kMaxRails) num_rails = v;
        }
    }
    s->num_rails = num_rails;

    int num_qps = cfg.num_qps <= 0 ? 1 : cfg.num_qps;
    if (num_qps > kEfaMaxQPs) num_qps = kEfaMaxQPs;
    // Can't have more rails than QPs (each rail needs at least one QP).
    if (num_rails > num_qps) num_rails = num_qps;
    s->num_rails = num_rails;
    int num_proxy_threads = cfg.num_proxy_threads <= 0 ? 1 : cfg.num_proxy_threads;
    // With multi-rail, need at least one proxy per rail (each proxy's QPs must
    // share a single CQ, which is tied to one ibv_context / NIC device).
    if (num_proxy_threads < num_rails) num_proxy_threads = num_rails;
    if (num_proxy_threads > kMaxProxyThreads) num_proxy_threads = kMaxProxyThreads;
    if (num_proxy_threads > num_qps) num_proxy_threads = num_qps;
    num_proxy_threads = std::max(1, num_proxy_threads);
    int logical_queues_per_qp = cfg.logical_queues_per_qp <= 0 ? 1 : cfg.logical_queues_per_qp;
    if (logical_queues_per_qp > 16) logical_queues_per_qp = 16;

    s->num_qps = num_qps;
    s->num_proxy_threads = num_proxy_threads;
    s->logical_queues_per_qp = logical_queues_per_qp;

    const int total_logical_queues = num_qps * logical_queues_per_qp;
    const int logical_queue_stride =
        std::max(1, (cfg.num_tiles + total_logical_queues - 1) / total_logical_queues);
    const int total_arrival_slots = logical_queue_stride * total_logical_queues;
    const int qps_per_proxy =
        std::max(1, (num_qps + num_proxy_threads - 1) / num_proxy_threads);

    // Zero-init fixed arrays for safe partial teardown.
    memset(s->extra_qps, 0, sizeof(s->extra_qps));
    memset(s->proxy_cqs, 0, sizeof(s->proxy_cqs));
    memset(s->proxies, 0, sizeof(s->proxies));
    memset(s->flag_stagings, 0, sizeof(s->flag_stagings));
    memset(s->dst_qpns, 0, sizeof(s->dst_qpns));
    memset(s->qp_rail, 0, sizeof(s->qp_rail));
    memset(s->rails, 0, sizeof(s->rails));

    // Open EFA devices (PCIe-root-aware, multi-rail).
    auto opened_devs = rdma::open_efa_devices(cfg.device_id, num_rails);
    num_rails = (int)opened_devs.size();
    s->num_rails = num_rails;

    // Initialize per-rail resources: context, PD.
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].ctx = opened_devs[r].ctx;
        s->rails[r].pd  = rdma::alloc_pd(s->rails[r].ctx);
    }
    // Legacy aliases for rail 0.
    s->ctx = s->rails[0].ctx;
    s->pd  = s->rails[0].pd;

    // Block-assign QPs to rails.
    // QP i → rail = i * num_rails / num_qps. This groups the first
    // ceil(num_qps/num_rails) QPs onto rail 0, the next block onto rail 1, etc.
    // Block assignment ensures each proxy's contiguous QP slice lands entirely
    // on one rail, which is required because all QPs in a proxy share a single
    // CQ tied to one ibv_context.
    for (int i = 0; i < num_qps; i++) {
        s->qp_rail[i] = i * num_rails / num_qps;
    }

    // Verify alignment: every proxy's contiguous QP slice must be single-rail.
    //   qp_base = first global QP index owned by proxy t
    //   qp_end  = one past the last (half-open range [qp_base, qp_end))
    // Each proxy thread owns one CQ tied to one rail (ibv_context), so every
    // QP in its slice must be on that same rail or its CQEs land on a CQ
    // that can't receive them.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        if (qp_base >= num_qps) break;
        const int first_rail = s->qp_rail[qp_base];
        const int qp_end = std::min(qp_base + qps_per_proxy, num_qps);
        for (int i = qp_base; i < qp_end; i++) {
            if (s->qp_rail[i] != first_rail) {
                fprintf(stderr,
                    "session_efa: QP %d (rail %d) in proxy %d (rail %d) — "
                    "misaligned! Adjust num_qps=%d or num_proxy_threads=%d so "
                    "contiguous QP slices don't span rail boundaries.\n",
                    i, s->qp_rail[i], t, first_rail, num_qps, num_proxy_threads);
                // Auto-fix: clamp to 1 rail to avoid crash.
                for (int j = 0; j < num_qps; j++) s->qp_rail[j] = 0;
                num_rails = 1;
                s->num_rails = 1;
                break;
            }
        }
        if (num_rails == 1) break;
    }

    // Per-proxy CQs. Proxy t's rail is determined by its first QP's rail.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int rail = (qp_base < num_qps) ? s->qp_rail[qp_base] : 0;
        s->proxy_cqs[t] = rdma::create_cq(s->rails[rail].ctx, 4096);
    }
    s->cq = s->proxy_cqs[0];

    // Create SRD QPs on their assigned rail's ctx/pd, sharing the
    // owning proxy's CQ.
    s->sq_depth = 512;
    for (int i = 0; i < num_qps; i++) {
        const int rail = s->qp_rail[i];
        const int owner_thread = std::min(num_proxy_threads - 1, i / qps_per_proxy);

        int sq_depth_out = s->sq_depth;
        ibv_qp* new_qp = rdma::create_srd_qp(
            s->rails[rail].pd, s->proxy_cqs[owner_thread],
            s->rails[rail].ctx, 512, &sq_depth_out);
        rdma::modify_srd_qp_init(new_qp);
        rdma::modify_srd_qp_rtr(new_qp);
        rdma::modify_srd_qp_rts(new_qp);

        if (i == 0) {
            s->qp = new_qp;
        } else {
            s->extra_qps[i - 1] = new_qp;
        }
    }

    // Register the caller-owned GPU send buffer on every rail's PD.
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].local_data_mr = gpu_mr::register_gpu_buffer(
            s->rails[r].pd, cfg.local_gpu_buf, cfg.local_gpu_buf_size);
    }

    // Optional: register the direct-send GPU source buffer (e.g.
    // output_local) as a DMA-BUF MR on every rail's PD. Skipping the staging
    // pack + gather on the send critical path. Hard-fails if DMA-BUF export
    // is unavailable — callers opt in explicitly.
    if (cfg.direct_dmabuf_enabled && cfg.clocal_gpu_buf != nullptr
        && cfg.clocal_gpu_buf_size > 0) {
        for (int r = 0; r < num_rails; r++) {
            const char* path = nullptr;
            s->rails[r].clocal_data_mr = gpu_mr::register_gpu_buffer_dmabuf_only(
                s->rails[r].pd, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size,
                gpu_mr::kDefaultAccess, &path);
            if (s->rails[r].clocal_data_mr == nullptr) {
                fprintf(stderr,
                    "session_efa: direct-DMA-BUF registration failed on rail %d "
                    "(ptr=%p bytes=%zu). EFA driver may not support DMA-BUF or "
                    "the buffer is not exportable. Disable the direct path.\n",
                    r, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size);
                exit(EXIT_FAILURE);
            }
        }
    }

    // GPU receive buffer.
    // Two modes:
    //   (a) Default — allocate via cudaMalloc and register on every rail's PD.
    //   (b) external_recv_buf set — register the caller-owned buffer instead.
    //       Used by dispatch_gemm zero-copy inter-recv where peer_tokens is the RDMA
    //       target, eliminating the recv_buf -> peer_tokens D2D copy. Caller
    //       owns the buffer and its lifetime.
    if (cfg.external_recv_buf != nullptr) {
        s->recv_buf.gpu_ptr = cfg.external_recv_buf;
        s->recv_buf.size = cfg.recv_buf_size;
        s->recv_buf.mr = nullptr;  // not owned: free_buffer becomes a no-op
                                   // for the gpu_ptr; per-rail MRs still freed.
        s->rails[0].recv_buf_mr = gpu_mr::register_gpu_buffer(
            s->rails[0].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        for (int r = 1; r < num_rails; r++) {
            s->rails[r].recv_buf_mr = gpu_mr::register_gpu_buffer(
                s->rails[r].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        }
    } else {
        // Use rail 0 for the allocation + primary MR.
        s->recv_buf = gpu_mr::alloc_and_register(
            s->rails[0].pd, cfg.recv_buf_size, cfg.device_id);
        s->rails[0].recv_buf_mr = s->recv_buf.mr;
        for (int r = 1; r < num_rails; r++) {
            s->rails[r].recv_buf_mr = gpu_mr::register_gpu_buffer(
                s->rails[r].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        }
    }

    // D2H FIFOs.
    // Default: one FIFO per proxy thread (num_fifos = num_proxy_threads).
    // MKERNEL_FIFO_PER_QP=1 enables one-fifo-per-QP mode (num_fifos = num_qps),
    // which decouples FIFO count from thread count and reduces GPU-side
    // atomicAdd contention on the FIFO head pointer (each QP gets its own
    // head). Each proxy thread then round-robins through its slice of fifos.
    bool fifo_per_qp = false;
    if (const char* e = std::getenv("MKERNEL_FIFO_PER_QP")) {
        fifo_per_qp = (std::atoi(e) != 0);
    }
    int num_fifos = fifo_per_qp ? num_qps : num_proxy_threads;
    if (num_fifos > kMaxProxyThreads) num_fifos = kMaxProxyThreads;  // clamp to bundle.fifos[] size
    const int fifos_per_proxy = num_fifos / num_proxy_threads;
    const int fifo_cap = cfg.fifo_capacity > 0 ? cfg.fifo_capacity : 1024;
    s->fifo_bundle = D2HFifoDeviceBundle{};
    s->fifo_bundle.num_fifos = num_fifos;
    s->fifo_bundle.global_num_qps = num_qps;
    s->fifo_bundle.logical_queues_per_qp = logical_queues_per_qp;
    // qps_per_fifo follows num_fifos: when fifo-per-qp, each fifo owns 1 QP.
    s->fifo_bundle.qps_per_fifo = fifo_per_qp ? 1 : qps_per_proxy;
    for (int t = 0; t < num_fifos; t++) {
        s->fifos[t] = create_d2h_fifo(fifo_cap);
        s->fifo_bundle.fifos[t] = s->fifos[t].device;
    }

    // Arrival flags. Register on every rail's PD so the peer can target
    // any of our NICs and the rkey will be valid on the receiving device.
    const size_t arrival_bytes =
        ((size_t)total_arrival_slots + (size_t)total_logical_queues) * sizeof(uint32_t);
    s->arrival = create_mapped_arrival_flags(total_arrival_slots, total_logical_queues);
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].arrival_mr = rdma::reg_mr(
            s->rails[r].pd, (void*)s->arrival.host_ptr, arrival_bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    s->arrival.mr = s->rails[0].arrival_mr;

    // Per-proxy flag staging. Register each staging buffer on its proxy's
    // rail PD (only the local NIC needs lkey access for the SGE source).
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int rail = (qp_base < num_qps) ? s->qp_rail[qp_base] : 0;
        s->flag_stagings[t] = create_flag_staging(8192);
        s->flag_stagings[t].mr = rdma::reg_mr(
            s->rails[rail].pd, s->flag_stagings[t].host_ptr,
            s->flag_stagings[t].count * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE);
        s->rails[rail].staging_mrs[t] = s->flag_stagings[t].mr;
    }

    // Stage-barrier flags: register on every rail's PD.
    s->stage_barrier = create_stage_barrier_flags(kEfaStageBarrierSlots);
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].barrier_mr = rdma::reg_mr(
            s->rails[r].pd, (void*)s->stage_barrier.host_ptr,
            kEfaStageBarrierSlots * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    s->stage_barrier.mr = s->rails[0].barrier_mr;

    // TCP exchange — pack primary QP + extras + multi-rail info.
    ConnectionInfo local_info{};
    rdma::fill_local_info(local_info, s->qp, s->rails[0].ctx);
    // Rail 0 rkeys (primary fields).
    local_info.data_rkey    = s->rails[0].recv_buf_mr->rkey;
    local_info.data_addr    = (uint64_t)s->recv_buf.gpu_ptr;
    local_info.data_len     = s->recv_buf.size;
    local_info.flags_rkey   = s->rails[0].arrival_mr->rkey;
    local_info.flags_addr   = (uint64_t)s->arrival.host_ptr;
    local_info.tail_rkey    = s->rails[0].arrival_mr->rkey;
    local_info.tail_addr    = (uint64_t)(s->arrival.host_ptr + total_arrival_slots);
    local_info.barrier_rkey = s->rails[0].barrier_mr->rkey;
    local_info.barrier_addr = (uint64_t)s->stage_barrier.host_ptr;
    local_info.num_qps = num_qps;
    for (int i = 1; i < num_qps; i++) {
        local_info.extra_qp_nums[i - 1] = s->extra_qps[i - 1]->qp_num;
        local_info.extra_psns[i - 1] = 0;  // PSN unused by SRD
    }

    // Multi-rail exchange fields: per-QP rail assignment + per-rail GIDs/rkeys.
    local_info.num_rails = num_rails;
    for (int i = 0; i < num_qps; i++) {
        local_info.qp_to_rail[i] = s->qp_rail[i];
    }
    for (int r = 1; r < num_rails; r++) {
        auto& ri = local_info.extra_rails[r - 1];
        // GID for this rail's NIC.
        ibv_gid gid{};
        ibv_query_gid(s->rails[r].ctx, 1, 0, &gid);
        memcpy(ri.gid, &gid, 16);
        ri.data_rkey    = s->rails[r].recv_buf_mr->rkey;
        ri.flags_rkey   = s->rails[r].arrival_mr->rkey;
        ri.tail_rkey    = s->rails[r].arrival_mr->rkey;
        ri.barrier_rkey = s->rails[r].barrier_mr->rkey;
    }

    // Multi-peer config resolution. Normalizes the (legacy peer_ip,
    // multi-peer peer_ips[]) inputs into a single uniform "(peer_slot, ip,
    // port, remote_rank)" tuple list so subsequent code can iterate.
    int            sess_num_peers = cfg.num_peers;
    if (sess_num_peers <= 0 || sess_num_peers > kMaxPeers) {
        fprintf(stderr, "create_session_efa: num_peers=%d out of range [1, %d]\n",
                sess_num_peers, kMaxPeers);
        delete s;
        return nullptr;
    }
    const char*    sess_peer_ips[kMaxPeers];
    int            sess_peer_ports[kMaxPeers];
    int            sess_peer_ranks[kMaxPeers];
    for (int p = 0; p < sess_num_peers; ++p) {
        sess_peer_ips[p]   = cfg.peer_ips[p];
        sess_peer_ports[p] = cfg.peer_tcp_ports[p];
        // Ring order matches Python's get_peer_ips() so slot p here
        // refers to the same peer that supplied peer_ips[p].
        sess_peer_ranks[p] = cfg.peer_ranks
            ? cfg.peer_ranks[p]
            : (cfg.rank + 1 + p) % (sess_num_peers + 1);
    }

    // Populate Session.peer_slot_by_rank: rank → slot, -1 for self / unused.
    s->num_peers = sess_num_peers;
    for (int r = 0; r <= kMaxPeers; ++r) s->peer_slot_by_rank[r] = -1;
    for (int p = 0; p < sess_num_peers; ++p) {
        const int r = sess_peer_ranks[p];
        if (r >= 0 && r <= kMaxPeers) s->peer_slot_by_rank[r] = p;
    }

    // TCP handshake order: iterate global (lo, hi) pairs. Peer-slot order
    // deadlocks at N>2 because slot-0 maps to a different peer per rank,
    // leaving every rank waiting on accept() for a peer that hasn't dialed.
    int inferred_num_nodes = sess_num_peers + 1;
    for (int lo = 0; lo < inferred_num_nodes; ++lo) {
        for (int hi = lo + 1; hi < inferred_num_nodes; ++hi) {
            if (cfg.rank != lo && cfg.rank != hi) continue;
            const int peer_rank = (cfg.rank == lo) ? hi : lo;
            // Look up which slot this peer occupies in sess_peer_ranks.
            int p = -1;
            for (int i = 0; i < sess_num_peers; ++i) {
                if (sess_peer_ranks[i] == peer_rank) { p = i; break; }
            }
            if (p < 0) continue;
            const bool is_server = (cfg.rank < peer_rank);
            ConnectionInfo remote = rdma::exchange_info_tcp(
                local_info, sess_peer_ips[p], sess_peer_ports[p], is_server);
            s->remote_infos[p] = remote;
        }
    }

    // Iterate in peer-slot order so the slot-0 alias of the single-peer
    // ProxyConfig fields stays consistent with per_peer[0].
    for (int p = 0; p < sess_num_peers; ++p) {
        ConnectionInfo remote = s->remote_infos[p];
        if (p == 0) {
            s->remote_info = remote;
            s->dst_qpns[0] = remote.qp_num;
            for (int i = 1; i < num_qps; i++) {
                s->dst_qpns[i] = remote.extra_qp_nums[i - 1];
            }
        }
        s->dst_qpns_per_peer[p][0] = remote.qp_num;
        for (int i = 1; i < num_qps; i++) {
            s->dst_qpns_per_peer[p][i] = remote.extra_qp_nums[i - 1];
        }
        // Per-rail AH for this peer. With multi-rail, local rail r pairs
        // with remote rail r.
        for (int r = 0; r < num_rails; r++) {
            const uint8_t* remote_gid = (r == 0)
                ? remote.gid
                : (remote.num_rails > 1)
                    ? remote.extra_rails[r - 1].gid
                    : remote.gid;
            ibv_ah* ah = rdma::create_ah(s->rails[r].pd, remote_gid);
            s->rails[r].dst_ah_per_peer[p] = ah;
            if (p == 0) {
                s->rails[r].dst_ah = ah;
            }
        }
    }

    // Spawn proxy threads. Each proxy handles QPs on its assigned rail.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int proxy_rail = (qp_base < num_qps) ? s->qp_rail[qp_base] : 0;
        const int local_qps = std::min(qps_per_proxy, num_qps - qp_base);
        if (local_qps <= 0) break;

        // Determine remote rkeys for this proxy's rail. The proxy posts RDMA
        // WRITEs to the remote peer's NIC on the remote rail paired with the
        // local rail. Remote rail = qp_to_rail[qp_base] from the exchange.
        const int remote_rail = (s->remote_info.num_rails > 1)
            ? s->remote_info.qp_to_rail[qp_base]
            : 0;
        const uint32_t remote_data_rkey = (remote_rail == 0)
            ? s->remote_info.data_rkey
            : s->remote_info.extra_rails[remote_rail - 1].data_rkey;
        const uint32_t remote_flags_rkey = (remote_rail == 0)
            ? s->remote_info.flags_rkey
            : s->remote_info.extra_rails[remote_rail - 1].flags_rkey;
        const uint32_t remote_tail_rkey = (remote_rail == 0)
            ? s->remote_info.tail_rkey
            : s->remote_info.extra_rails[remote_rail - 1].tail_rkey;
        const uint32_t remote_barrier_rkey = (remote_rail == 0)
            ? s->remote_info.barrier_rkey
            : s->remote_info.extra_rails[remote_rail - 1].barrier_rkey;

        ProxyConfig pcfg{};
        pcfg.fifo = &s->fifos[t].host;
        // Multi-fifo: hand this proxy a slice of fifos[fifos_per_proxy*t ..
        // fifos_per_proxy*(t+1)). Single-fifo mode: num_fifos==1 and run()
        // uses pcfg.fifo (legacy path). Storage is per-proxy in s->per_proxy_fifo_ptrs.
        if (num_fifos > num_proxy_threads) {
            D2HFifoHost** slot = s->per_proxy_fifo_ptrs[t];
            for (int k = 0; k < fifos_per_proxy; ++k) {
                slot[k] = &s->fifos[t * fifos_per_proxy + k].host;
            }
            pcfg.fifos = slot;
            pcfg.num_fifos = fifos_per_proxy;
            pcfg.fifo = slot[0];
        }
        pcfg.qp   = (qp_base == 0) ? s->qp : s->extra_qps[qp_base - 1];
        pcfg.cq   = s->proxy_cqs[t];

        pcfg.dst_ah = s->rails[proxy_rail].dst_ah;
        for (int i = 0; i < local_qps; i++) {
            pcfg.dst_qpns[i] = s->dst_qpns[qp_base + i];
        }
        memset(pcfg.extra_qps, 0, sizeof(pcfg.extra_qps));
        for (int i = 1; i < local_qps; i++) {
            const int global_qp = qp_base + i;
            pcfg.extra_qps[i - 1] = s->extra_qps[global_qp - 1];
        }
        pcfg.num_qps          = local_qps;
        pcfg.qp_base_idx      = qp_base;
        pcfg.global_num_qps   = num_qps;
        pcfg.logical_queues_per_qp = logical_queues_per_qp;

        // Per-rail local lkey and remote rkeys.
        pcfg.local_data_addr  = (uint64_t)cfg.local_gpu_buf;
        pcfg.local_data_lkey  = s->rails[proxy_rail].local_data_mr->lkey;
        pcfg.remote_data_addr = s->remote_info.data_addr;
        pcfg.remote_data_rkey = remote_data_rkey;

        // Direct-DMA-BUF (src_view=1) path: proxy posts SGEs sourced from the
        // caller-owned output buffer instead of staging. Only populated if the
        // session config opted in AND per-rail registration succeeded.
        if (cfg.direct_dmabuf_enabled && cfg.clocal_gpu_buf != nullptr
            && s->rails[proxy_rail].clocal_data_mr != nullptr) {
            pcfg.clocal_data_addr    = (uint64_t)cfg.clocal_gpu_buf;
            pcfg.clocal_data_lkey    = s->rails[proxy_rail].clocal_data_mr->lkey;
            pcfg.clocal_data_bytes   = cfg.clocal_gpu_buf_size;
            pcfg.direct_dmabuf_enabled = true;
        }

        pcfg.local_arrival_host_ptr = s->arrival.host_ptr;
        pcfg.local_arrival_count    = s->arrival.count;

        pcfg.flag_staging      = &s->flag_stagings[t];
        pcfg.remote_flags_addr = s->remote_info.flags_addr;
        pcfg.remote_flags_rkey = remote_flags_rkey;

        pcfg.use_arrival_queue   = cfg.use_arrival_queue;
        pcfg.remote_queue_stride = (uint32_t)logical_queue_stride;
        pcfg.remote_tail_addr    = s->remote_info.tail_addr;
        pcfg.remote_tail_rkey    = remote_tail_rkey;
        {
            const char* tail_env = std::getenv("Q2_SENDER_PUBLISHED_TAIL");
            pcfg.enable_remote_tail =
                (tail_env != nullptr && tail_env[0] == '1');
        }
        // Optional proxy post/poll controller.
        {
            const char* pipe_env = std::getenv("Q2_PROXY_PIPELINE");
            pcfg.pipeline_enabled =
                (pipe_env != nullptr && pipe_env[0] == '1');
        }

        pcfg.remote_barrier_addr = s->remote_info.barrier_addr;
        pcfg.remote_barrier_rkey = remote_barrier_rkey;

        // Multi-peer endpoint table for the proxy hot path. Each entry
        // covers a single peer slot from the perspective of THIS proxy's
        // rail. The EFA proxy resolves the slot from
        // peer_slot_by_rank[cmd.dst_rank] and reads per_peer[slot].
        for (int p = 0; p < sess_num_peers; ++p) {
            const ConnectionInfo& ri = s->remote_infos[p];
            const int peer_remote_rail = (ri.num_rails > 1)
                ? ri.qp_to_rail[qp_base]
                : 0;
            const RailExchangeInfo* extra_rail =
                (ri.num_rails > 1 && peer_remote_rail >= 1)
                    ? &ri.extra_rails[peer_remote_rail - 1]
                    : nullptr;
            PerPeerProxyData& peer = s->per_proxy_peer[t][p];
            peer.dst_ah = s->rails[proxy_rail].dst_ah_per_peer[p];
            for (int i = 0; i < local_qps; i++) {
                peer.dst_qpns[i] = s->dst_qpns_per_peer[p][qp_base + i];
            }
            peer.remote_data_addr    = ri.data_addr;
            peer.remote_data_rkey    = extra_rail ? extra_rail->data_rkey    : ri.data_rkey;
            peer.remote_flags_addr   = ri.flags_addr;
            peer.remote_flags_rkey   = extra_rail ? extra_rail->flags_rkey   : ri.flags_rkey;
            peer.remote_tail_addr    = ri.tail_addr;
            peer.remote_tail_rkey    = extra_rail ? extra_rail->tail_rkey    : ri.tail_rkey;
            peer.remote_barrier_addr = ri.barrier_addr;
            peer.remote_barrier_rkey = extra_rail ? extra_rail->barrier_rkey : ri.barrier_rkey;
        }
        pcfg.num_peers         = sess_num_peers;
        pcfg.per_peer          = s->per_proxy_peer[t];
        pcfg.peer_slot_by_rank = s->peer_slot_by_rank;

        pcfg.epoch        = s->epoch;
        pcfg.max_inflight = cfg.max_inflight > 0 ? cfg.max_inflight : 512;
        pcfg.sq_depth     = s->sq_depth;
        pcfg.device_id    = cfg.device_id;
        pcfg.pin_proxy    = cfg.pin_proxy;

        s->proxies[t] = new Proxy(pcfg);
        s->proxies[t]->start();
    }

    return s;
}

/**
 * Wait for all proxies to finish processing their FIFO commands and complete
 * all in-flight RDMA writes. Call after cudaDeviceSynchronize() so no new FIFO
 * pushes will arrive.
 */
inline void drain_proxy(Session* s) {
    uint64_t fifo_heads[kMaxProxyThreads]{};
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaError_t err = cudaMemcpy(
            &fifo_heads[t], s->fifos[t].device.head,
            sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "session_efa: cudaMemcpy(head) failed: %s\n",
                    cudaGetErrorString(err));
            std::abort();
        }
    }

    auto last_log = std::chrono::steady_clock::now();
    for (int i = 0; i < 50000; i++) {  // 5s soft budget, then log+continue
        bool drained = true;
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (s->fifos[t].host.cpu_head < fifo_heads[t] ||
                s->proxies[t]->inflight() != 0) {
                drained = false;
                break;
            }
        }
        if (drained) return;

        auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(1)) {
            fprintf(stderr, "session_efa: waiting for proxy drain");
            for (int t = 0; t < s->num_proxy_threads; t++) {
                fprintf(stderr, " [t=%d cpu_head=%lu target=%lu inflight=%d]",
                        t,
                        (unsigned long)s->fifos[t].host.cpu_head,
                        (unsigned long)fifo_heads[t],
                        s->proxies[t]->inflight());
            }
            fprintf(stderr, "\n");
            last_log = now;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

inline void destroy_session(Session* s) {
    if (!s) return;

    // Ensure proxies have processed all FIFO commands before shutdown.
    if (s->proxies[0]) {
        cudaDeviceSynchronize();
        drain_proxy(s);
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (!s->proxies[t]) continue;
            s->proxies[t]->stop();
            delete s->proxies[t];
            s->proxies[t] = nullptr;
        }
    }

    // Deregister per-rail MRs (arrival, barrier, recv_buf, local_data).
    // External recv_buf mode: recv_buf.mr is nullptr (caller-owned buffer),
    // so all per-rail recv_buf MRs must be dereg'd here (free_buffer below
    // is a no-op for external buffers since gpu_ptr is also cleared).
    const bool external_recv_buf = (s->recv_buf.mr == nullptr);
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].arrival_mr) rdma::dereg_mr(s->rails[r].arrival_mr);
        if (s->rails[r].barrier_mr) rdma::dereg_mr(s->rails[r].barrier_mr);
        if (s->rails[r].local_data_mr) rdma::dereg_mr(s->rails[r].local_data_mr);
        if (s->rails[r].clocal_data_mr) rdma::dereg_mr(s->rails[r].clocal_data_mr);
        // owned mode: rail 0's MR is freed by gpu_mr::free_buffer below.
        // external mode: dereg rail 0 too (free_buffer won't touch it).
        const bool dereg_recv = external_recv_buf || (r > 0);
        if (dereg_recv && s->rails[r].recv_buf_mr) rdma::dereg_mr(s->rails[r].recv_buf_mr);
    }
    // Clear the pointers that gpu_mr::free_buffer and arrival/barrier teardown
    // use so they don't double-free the rail 0 MRs.
    s->arrival.mr = nullptr;
    s->stage_barrier.mr = nullptr;

    // Deregister per-proxy staging MRs (already dereg'd above is fine — the MR
    // pointer in flag_stagings[t].mr is the same object as rails[r].staging_mrs[t]).
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->flag_stagings[t].mr) rdma::dereg_mr(s->flag_stagings[t].mr);
        s->flag_stagings[t].mr = nullptr;
    }

    // Free buffers. External recv_buf mode: clear gpu_ptr so free_buffer
    // doesn't cudaFree the caller-owned buffer.
    if (external_recv_buf) {
        s->recv_buf.gpu_ptr = nullptr;
    }
    gpu_mr::free_buffer(s->recv_buf);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_d2h_fifo(s->fifos[t]);
    }
    destroy_arrival_flags(s->arrival);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_flag_staging(s->flag_stagings[t]);
    }
    destroy_stage_barrier_flags(s->stage_barrier);

    // Destroy per-rail AHs.
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].dst_ah) rdma::destroy_ah(s->rails[r].dst_ah);
    }

    // Destroy extra QPs in reverse order.
    for (int i = s->num_qps - 1; i >= 1; i--) {
        if (s->extra_qps[i - 1]) rdma::destroy_qp(s->extra_qps[i - 1]);
    }
    if (s->qp) rdma::destroy_qp(s->qp);

    // Destroy per-proxy CQs.
    for (int t = s->num_proxy_threads - 1; t >= 0; t--) {
        if (s->proxy_cqs[t]) rdma::destroy_cq(s->proxy_cqs[t]);
    }

    // Destroy per-rail PDs and close device contexts.
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].pd)  rdma::dealloc_pd(s->rails[r].pd);
        if (s->rails[r].ctx) rdma::close_device(s->rails[r].ctx);
    }

    delete s;
}

// ---------------------------------------------------------------------------
// Helpers for kernel globals
// ---------------------------------------------------------------------------

inline D2HFifoDeviceBundle get_fifo_device_handle(const Session* s) {
    return s->fifo_bundle;
}

inline uint32_t* get_arrival_device_ptr(const Session* s) {
    return s->arrival.device_ptr;
}

/**
 * Device pointer to the per-queue tail counters colocated with the arrival
 * flags array. The EFA SRD backend does not publish tails yet, so this returns
 * the trailing tail slots the arrival buffer carries — kernels reading the
 * tail will simply observe zeros.
 */
inline uint32_t* get_arrival_tail_device_ptr(const Session* s) {
    return s ? s->arrival.tail_device_ptr : nullptr;
}

inline void* get_recv_buf_ptr(const Session* s) {
    return s->recv_buf.gpu_ptr;
}

inline int get_num_qps(const Session* s) {
    return s ? s->num_qps : 1;
}

/**
 * Device pointer to the stage-barrier flags — the same host-pinned/device-
 * aliased memory the peer writes tokens into. Kernels spin on this directly.
 */
inline uint32_t* get_stage_barrier_device_ptr(const Session* s) {
    return s ? s->stage_barrier.device_ptr : nullptr;
}

/** Device-side ready-queue handle — SRD does not use it. */
inline ReadyQueueDevice get_ready_queue_device(const Session* /*s*/) {
    return ReadyQueueDevice{};
}

/** Set total expected ready-queue entries — no-op on SRD. */
inline void set_ready_queue_total(Session* /*s*/, uint32_t /*total*/) {}

/** Per-proxy-thread diagnostic counters. */
inline std::vector<ProxyDiagnostics> get_proxy_diagnostics(const Session* s) {
    std::vector<ProxyDiagnostics> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        out.push_back(s->proxies[t]->get_diagnostics());
    }
    return out;
}

/** Per-proxy-thread signaled-post / completion timelines. */
inline std::vector<ProxyTimeline> get_proxy_timelines(const Session* s) {
    std::vector<ProxyTimeline> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        out.push_back(s->proxies[t]->get_timeline());
    }
    return out;
}

inline ProxyTimestamps get_proxy_timestamps(const Session* s) {
    return s && s->proxies[0] ? s->proxies[0]->get_timestamps() : ProxyTimestamps{};
}

/**
 * Phase 1 of the epoch transition — mirrors session.h::prepare_epoch.
 * Waits for GPU + all proxies to quiesce, pauses them, drains CQs, resets
 * inflight and timestamps.
 */
inline void prepare_epoch(Session* s) {
    static const bool log_timing = ([]() {
        const char* e = std::getenv("Q2_EPOCH_TIMING");
        return e && e[0] == '1';
    })();
    // Read every call — Python may set this env var AFTER the first set_epoch(1)
    // call (mirrors the MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET pattern at
    // commit_epoch:1029). Three-level fast path:
    //   '1' = skip cudaDeviceSynchronize (Sub-test A)
    //   '2' = also skip pause/drain_cq/reset block (Sub-test B)
    //   '3' = also skip drain_proxy (Sub-test C)
    const char* fast_env = std::getenv("MKERNEL_PREP_EPOCH_FAST");
    const int fast_level = (fast_env && fast_env[0] >= '1' && fast_env[0] <= '3')
                               ? (fast_env[0] - '0')
                               : 0;

    auto t0 = std::chrono::steady_clock::now();
    if (fast_level < 1) cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    if (fast_level < 3) drain_proxy(s);
    auto t2 = std::chrono::steady_clock::now();
    if (fast_level < 2) {
        for (int t = 0; t < s->num_proxy_threads; t++) {
            s->proxies[t]->pause();
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    if (fast_level < 2) {
        for (int t = 0; t < s->num_proxy_threads; t++) {
            s->proxies[t]->drain_cq();
            s->proxies[t]->reset_inflight();
            s->proxies[t]->reset_timestamps();
        }
    }
    auto t4 = std::chrono::steady_clock::now();
    if (log_timing) {
        auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        fprintf(stderr,
                "[EPOCH_TIMING rank=%d] prepare_epoch: cudaSync=%ldus drain_proxy=%ldus "
                "pause=%ldus drain_cq+reset=%ldus total=%ldus (fast=%d)\n",
                s->rank, us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4), us(t0, t4),
                fast_level);
    }
}

/**
 * Phase 2 of the epoch transition — mirrors session.h::commit_epoch.
 * Updates epoch, resets arrivals/stage-barrier/FIFOs, resumes all proxies.
 *
 * When MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET=1, skips arrival/stage-barrier
 * resets and FIFO reinit. This is safe for steady-state benchmarks because:
 *   - arrival flags: the kernel resets them on-GPU before the epilogue barrier,
 *     so they're clean by the time the next iter starts.
 *   - stage_barrier: kernel uses monotonic (>= epoch) checks and never clears,
 *     so host-side zeroing is unnecessary and actually harmful — it races with
 *     incoming BARRIER_NOTIFY RDMA WRITEs from a faster peer that has already
 *     entered its next epoch's kernel.
 *   - FIFOs: poll() clears consumed slots to EMPTY; drain_proxy (in prepare_epoch)
 *     ensures all slots are consumed before we reach commit_epoch.
 */
inline void commit_epoch(Session* s, uint32_t epoch) {
    static const bool log_timing = ([]() {
        const char* e = std::getenv("Q2_EPOCH_TIMING");
        return e && e[0] == '1';
    })();
    auto t0 = std::chrono::steady_clock::now();
    // Read every call — Python sets this env var AFTER the first set_epoch(1)
    // call (line 629 of benchmark_gemm_ar_multinode.py), so a static-const
    // cache would latch false and never pick up the later mutation.
    const char* skip_env = getenv("MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET");
    const bool skip_reset = (skip_env && skip_env[0] == '1');

    s->epoch = epoch;
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->set_epoch(epoch);
    }

    if (!skip_reset) {
        reset_arrival_flags(s->arrival);
        reset_stage_barrier_flags(s->stage_barrier);

        for (int t = 0; t < s->num_proxy_threads; t++) {
            cudaMemset(s->fifos[t].device.head, 0, sizeof(uint64_t));
            cudaMemset(s->fifos[t].device.tail_cache, 0, sizeof(uint64_t));
            *s->fifos[t].host.tail = 0;
            s->fifos[t].host.cpu_head = 0;
            memset(s->fifos[t].host.triggers, 0,
                   s->fifos[t].host.capacity * sizeof(TransferCmd));
        }

        cudaDeviceSynchronize();
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
    auto t1 = std::chrono::steady_clock::now();
    if (log_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        fprintf(stderr,
                "[EPOCH_TIMING rank=%d] commit_epoch: %ldus (skip_reset=%d)\n",
                s->rank, us, (int)skip_reset);
    }
}

/**
 * Host-side stage barrier — send `token` to the peer's barrier slot via SRD,
 * then spin on our own slot until the peer does the same. Mirrors
 * session.h::stage_barrier. Uses proxy 0 / QP 0 on both sides so ordering
 * with the peer is deterministic.
 */
inline void stage_barrier(Session* s, int slot, uint32_t token) {
    if (slot < 0 || slot >= s->stage_barrier.count) {
        fprintf(stderr, "stage_barrier: invalid slot %d\n", slot);
        exit(EXIT_FAILURE);
    }


    cudaDeviceSynchronize();
    drain_proxy(s);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->pause();
        s->proxies[t]->drain_cq();
        s->proxies[t]->reset_inflight();
    }

    s->proxies[0]->post_stage_barrier(slot, token);
    s->proxies[0]->drain_cq();
    s->proxies[0]->reset_inflight();

    for (int i = 0; i < 50000; i++) {
        if (s->stage_barrier.host_ptr[slot] == token) {
            for (int t = 0; t < s->num_proxy_threads; t++) {
                s->proxies[t]->resume();
            }
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
    fprintf(stderr, "stage_barrier timeout: slot=%d token=%u observed=%u\n",
            slot, token, s->stage_barrier.host_ptr[slot]);
    exit(EXIT_FAILURE);
}

inline bool should_prime_first_launch(Session* /*s*/) {
    const char* env = std::getenv("Q5_PRIME_FIRST_LAUNCH");
    if (env && env[0]) {
        return env[0] == '1';
    }
    const char* fci = std::getenv("Q5_FUSE_COMPUTE_INTRA");
    return fci && fci[0] == '1';
}

inline int prime_first_launch_rounds() {
    const char* env = std::getenv("Q5_PRIME_FIRST_LAUNCH_ROUNDS");
    if (env && env[0]) {
        return std::max(1, std::atoi(env));
    }
    // Four host-injected dummy writes are enough to make the first real
    // fused-compute-intra launch behave like the warmed path at medium sizes.
    return 4;
}

inline void host_publish_transfer_cmd(D2HFifoHost* fifo, uint64_t slot, const TransferCmd& cmd) {
    const uint64_t mask = (uint64_t)(fifo->capacity - 1);
    TransferCmd* dst = &fifo->triggers[slot & mask];
    const char* src_bytes = reinterpret_cast<const char*>(&cmd);
    char* dst_bytes = reinterpret_cast<char*>(dst);
    std::memcpy(dst_bytes + 1, src_bytes + 1, sizeof(TransferCmd) - 1);
    __atomic_store_n(
        reinterpret_cast<uint8_t*>(&dst->cmd_type),
        static_cast<uint8_t>(cmd.cmd_type),
        __ATOMIC_RELEASE);
}

inline void prime_first_launch_transport(Session* s) {
    if (!s || !should_prime_first_launch(s) || s->num_qps <= 0 || s->arrival.tail_count <= 0) {
        return;
    }

    const int qps_per_proxy =
        std::max(1, (s->num_qps + s->num_proxy_threads - 1) / s->num_proxy_threads);
    const uint32_t dummy_tile = (uint32_t)s->arrival.count;  // first tail slot
    const uint32_t tile_bytes = 128u * 256u * 2u;
    const uint32_t prime_bytes = (uint32_t)std::max<uint64_t>(
        4u, std::min<uint64_t>(tile_bytes, s->recv_buf.size / (size_t)std::max(1, s->num_qps)));
    if (dummy_tile > 0xFFFFu) {
        fprintf(stderr,
                "session_efa: skipping first-launch prime; dummy tile %u exceeds TransferCmd tile_id\n",
                dummy_tile);
        return;
    }

    const int rounds = prime_first_launch_rounds();
    for (int round = 0; round < rounds; round++) {
        uint64_t target_heads[kMaxProxyThreads]{};
        int injected[kMaxProxyThreads]{};
        for (int t = 0; t < s->num_proxy_threads; t++) {
            target_heads[t] = s->fifos[t].host.cpu_head;
        }

        for (int global_qp = 0; global_qp < s->num_qps; global_qp++) {
            const int proxy_idx = std::min(s->num_proxy_threads - 1, global_qp / qps_per_proxy);
            D2HFifoHost* fifo = &s->fifos[proxy_idx].host;
            TransferCmd cmd{};
            cmd.cmd_type = CmdType::WRITE;
            cmd.dst_rank = (uint8_t)(s->rank == 0 ? 1 : 0);
            cmd.tile_id = (uint32_t)dummy_tile;
            cmd.bytes = prime_bytes;
            cmd.local_offset = (uint64_t)global_qp * prime_bytes;
            cmd.remote_offset = (uint64_t)global_qp * prime_bytes;
            cmd.lane_id = (uint16_t)global_qp;
            cmd.src_view = 0;
            cmd.reserved0 = 0;
            cmd.row_span = 0;
            cmd.row_count = 0;
            cmd.enqueue_device_ns = 0;
            host_publish_transfer_cmd(fifo, target_heads[proxy_idx], cmd);
            target_heads[proxy_idx]++;
            injected[proxy_idx]++;
        }

        auto last_log = std::chrono::steady_clock::now();
        bool done = false;
        for (int i = 0; i < 50000; i++) {
            done = true;
            for (int t = 0; t < s->num_proxy_threads; t++) {
                if (injected[t] == 0) continue;
                if (s->fifos[t].host.cpu_head < target_heads[t] || s->proxies[t]->inflight() != 0) {
                    done = false;
                    break;
                }
            }
            if (done) break;

            auto now = std::chrono::steady_clock::now();
            if (now - last_log >= std::chrono::seconds(1)) {
                fprintf(stderr, "session_efa: waiting for first-launch transport prime round %d/%d",
                        round + 1, rounds);
                for (int t = 0; t < s->num_proxy_threads; t++) {
                    if (injected[t] == 0) continue;
                    fprintf(stderr, " [t=%d cpu_head=%lu target=%lu inflight=%d]",
                            t,
                            (unsigned long)s->fifos[t].host.cpu_head,
                            (unsigned long)target_heads[t],
                            s->proxies[t]->inflight());
                }
                fprintf(stderr, "\n");
                last_log = now;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (!done) {
            fprintf(stderr, "session_efa: first-launch transport prime timed out in round %d/%d\n",
                    round + 1, rounds);
            break;
        }

        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (injected[t] == 0) continue;
            const uint64_t new_head = target_heads[t];
            MKERNEL_CUDACHECK(cudaMemcpy(
                s->fifos[t].device.head, &new_head, sizeof(uint64_t),
                cudaMemcpyHostToDevice));
            MKERNEL_CUDACHECK(cudaMemcpy(
                s->fifos[t].device.tail_cache, &new_head, sizeof(uint64_t),
                cudaMemcpyHostToDevice));
        }
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        s->proxies[t]->reset_timestamps();
    }
}

/** Legacy single-call epoch transition kept for existing callers. */
inline void set_epoch(Session* s, uint32_t epoch) {
    prepare_epoch(s);
    commit_epoch(s, epoch);
    if (epoch == 1u) {
        prime_first_launch_transport(s);
    }
}

} // namespace internode
