/**
 * @file proxy_efa.h
 * @brief CPU proxy thread for EFA-based inter-node RDMA communication.
 *
 * Same role as proxy.h but uses EFA SRD new verbs API:
 *   ibv_wr_start -> ibv_wr_rdma_write[_imm] -> ibv_wr_set_ud_addr
 *   -> ibv_wr_set_sge -> ibv_wr_complete
 *
 * The SRD backend supports two notification modes:
 *   1. write_imm   (default): each data write carries the tile id as immediate
 *      data; the receiver-side proxy updates local mapped arrival flags after
 *      polling RECV_RDMA_WITH_IMM CQEs.
 *   2. remote_flag: posts a chained plain RDMA write for data followed by a
 *      plain RDMA write of the epoch into the peer's mapped arrival flag slot.
 *
 * Multi-QP: each proxy owns a contiguous slice of the session's QPs starting
 * at `qp_base_idx`. Data batches are round-robined across those QPs to spread
 * bandwidth over several SRD queue pairs. The proxy-level CQ is shared across
 * all QPs assigned to this proxy.
 */
#pragma once

#include "types.h"
#include "d2h_fifo.cuh"
#include "arrival.cuh"
#include "rdma_transport_efa.h"
#include "proxy_diagnostics.h"

#include <algorithm>
#include <atomic>
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>

namespace internode {


// Per-peer endpoint info for the proxy hot path. Each proxy thread owns
// QPs on a single local rail; for each peer slot p, peers[p] holds the
// data the hot path needs to post a WR to peer p over THIS rail. Filled in
// by session_efa.h create_session() when handing off ProxyConfig.
struct PerPeerProxyData {
    ibv_ah*  dst_ah                = nullptr;
    uint32_t dst_qpns[kMaxExchangeQPs] = {};
    uint64_t remote_data_addr      = 0;
    uint32_t remote_data_rkey      = 0;
    uint64_t remote_flags_addr     = 0;
    uint32_t remote_flags_rkey     = 0;
    uint64_t remote_tail_addr      = 0;
    uint32_t remote_tail_rkey      = 0;
    uint64_t remote_barrier_addr   = 0;
    uint32_t remote_barrier_rkey   = 0;
};

struct ProxyConfig {
    D2HFifoHost*  fifo;
    // Multi-fifo extension. When num_fifos > 1, run() rotates current_fifo_idx
    // across fifos[] each batch, polling one fifo per outer iteration. Each
    // batch is still single-QP (the existing batching invariant) — multiple
    // fifos just give multiple QP slices the proxy thread cycles through.
    // session_efa.h sets num_fifos = qps_per_proxy (one fifo per QP) when the
    // MKERNEL_FIFO_PER_QP env is set; otherwise num_fifos = 1 and `fifo` is used.
    D2HFifoHost** fifos     = nullptr;
    int           num_fifos = 1;

    // Primary QP (local index 0 within this proxy). QPs are also addressed by
    // global index qp_base_idx .. qp_base_idx + num_qps - 1.
    ibv_qp*       qp;
    ibv_cq*       cq;

    // EFA-specific: single address handle shared by all QPs (same remote GID)
    // and one remote QP number per local QP. dst_qpns[0] pairs with cfg_.qp.
    ibv_ah*       dst_ah;
    uint32_t      dst_qpns[kMaxExchangeQPs];

    // Extra local QPs (indices 1..num_qps-1 within this proxy). Allocated in
    // session_efa.h and re-grouped per proxy.
    ibv_qp*       extra_qps[kMaxExchangeQPs - 1];
    int           num_qps           = 1;  // local QP count for this proxy
    int           qp_base_idx       = 0;  // global QP index of cfg_.qp
    int           global_num_qps    = 1;  // total QPs across all proxies
    int           logical_queues_per_qp = 1;

    // Local data buffer (GPU HBM, RDMA-registered)
    uint64_t      local_data_addr;
    uint32_t      local_data_lkey;

    // Optional second source MR (e.g. output_local). When
    // direct_dmabuf_enabled && cmd.src_view == 1, the data-path SGE posts
    // from (clocal_data_addr + cmd.local_offset) with clocal_data_lkey.
    // Populated by session_efa.h create_session() when cfg.direct_dmabuf_enabled.
    uint64_t      clocal_data_addr       = 0;
    uint32_t      clocal_data_lkey       = 0;
    size_t        clocal_data_bytes      = 0;
    bool          direct_dmabuf_enabled  = false;

    // Remote data buffer (peer's GPU HBM)
    uint64_t      remote_data_addr;
    uint32_t      remote_data_rkey;

    // Local arrival flags (mapped host memory, GPU polls device alias)
    volatile uint32_t* local_arrival_host_ptr;
    int                local_arrival_count;

    // Flag staging (host-pinned, holds epoch value for flag RDMA write)
    FlagStaging*  flag_staging;

    // Remote arrival flags (peer's host-pinned memory)
    uint64_t      remote_flags_addr;
    uint32_t      remote_flags_rkey;

    // Arrival-queue layout selector. When true, the proxy uses the queue
    // layout (each WR claims a unique slot in a per-logical-queue ring; the
    // payload encodes pack_arrival_work(first_tile, num_tiles)). When false,
    // the proxy uses the flat layout (one flag-slot per tile_id, payload =
    // epoch). Wired from SessionConfig.use_arrival_queue. Today only gemm_ar
    // sets this to true.
    bool          use_arrival_queue    = false;

    // Queue-layout fields (only consulted when use_arrival_queue == true).
    // remote_queue_stride is the number of 4-byte slots per logical queue on
    // the peer; each logical queue is laid out contiguously so slot k of
    // queue q lives at
    //   remote_flags_addr + (q * remote_queue_stride + k) * 4
    // remote_tail_addr / remote_tail_rkey point to per-queue tail counters
    // (one uint32 per logical queue) if enable_remote_tail is set.
    uint64_t      remote_tail_addr     = 0;
    uint32_t      remote_tail_rkey     = 0;
    uint32_t      remote_queue_stride  = 0;
    bool          enable_remote_tail   = false;

    // Remote stage-barrier slot (host-pinned, RDMA-writable on the peer).
    // Host-side stage_barrier() uses these to post a 4-byte SRD send into a
    // specific slot on the peer, matching the CX7 backend's barrier path.
    uint64_t      remote_barrier_addr = 0;
    uint32_t      remote_barrier_rkey = 0;

    uint32_t      epoch;
    int           max_inflight;
    int           sq_depth;

    int           device_id = 0;
    bool          pin_proxy = true;

    // Enables the adaptive post/poll controller inside Proxy::run().
    bool          pipeline_enabled = false;

    // Multi-peer routing. When num_peers > 1, the hot path resolves
    // peer_slot = peer_slot_by_rank[cmd.dst_rank] and indexes into per_peer
    // for AH/dst_qpns/remote_*. peer_slot_by_rank is owned by the Session
    // (lifetime > Proxy); per_peer is owned by the Session too.
    //
    // For num_peers == 1, per_peer[0] aliases the scalar dst_ah / dst_qpns /
    // remote_* fields.
    int                  num_peers           = 1;
    const PerPeerProxyData* per_peer         = nullptr;  // length = num_peers
    const int*           peer_slot_by_rank   = nullptr;  // length = kMaxPeers + 1
};

class Proxy {
public:
    static constexpr int BATCH_SIZE = 8;
    static constexpr int ACCUMULATE_SPINS = 128;
    static constexpr int ACCUMULATE_MISS_BUDGET = 8;
    static constexpr int SRD_SQ_DEPTH = 512;

    // Adaptive BATCH_SIZE / ACCUMULATE_SPINS bounds. The fixed constants above
    // remain the static-mode fallback.
    static constexpr uint32_t BATCH_SIZE_MIN     = 2;
    static constexpr uint32_t BATCH_SIZE_MAX     = 8;
    static constexpr uint32_t ACCUM_SPINS_MIN    = 16;
    static constexpr uint32_t ACCUM_SPINS_MAX    = 256;

    enum class NotifyMode {
        WriteImm,
        RemoteFlag,
    };

    explicit Proxy(const ProxyConfig& cfg)
        : cfg_(cfg), running_(false), paused_(false), ack_paused_(false),
          inflight_(0), notify_mode_(detect_notify_mode()) {
        // Mirror the session-level pipeline flag into the proxy.
        use_adaptive_batch_  = cfg_.pipeline_enabled;
        batch_size_current_  = BATCH_SIZE_MAX;
        accum_spins_current_ = ACCUMULATE_SPINS;
        cqe_ia_ewma_ns_      = 20000;
        last_cq_poll_ns_     = 0;
        empty_loop_streak_   = 0;
        if (cfg_.max_inflight <= 0) cfg_.max_inflight = 128;
        if (cfg_.num_qps <= 0) cfg_.num_qps = 1;
        if (cfg_.num_qps > kMaxExchangeQPs) cfg_.num_qps = kMaxExchangeQPs;
        if (cfg_.global_num_qps <= 0) cfg_.global_num_qps = cfg_.num_qps;
        if (cfg_.logical_queues_per_qp <= 0) cfg_.logical_queues_per_qp = 1;
        if (cfg_.logical_queues_per_qp > 16) cfg_.logical_queues_per_qp = 16;

        const int sq_depth = cfg_.sq_depth > 0 ? cfg_.sq_depth : SRD_SQ_DEPTH;
        int per_qp_cap;
        if (notify_mode_ == NotifyMode::RemoteFlag) {
            // remote_flag emits two signaled WRs per logical command. Clamp the
            // logical inflight budget to the actual SRD SQ depth rather than the
            // higher experiment-side logical cap. In practice EFA starts
            // returning ENOMEM well before a nominal 512-entry SQ is full, so
            // keep a larger reserve for in-flight CQ progress and batched posts.
            per_qp_cap = sq_depth / 4;
            if (per_qp_cap > BATCH_SIZE) per_qp_cap -= BATCH_SIZE;
        } else {
            per_qp_cap = cfg_.max_inflight;
        }
        per_qp_cap = std::min(per_qp_cap, cfg_.max_inflight);
        if (per_qp_cap <= 0) per_qp_cap = 1;
        // Effective inflight scales with local QP count — mirrors proxy.h.
        effective_max_inflight_ = per_qp_cap * cfg_.num_qps;

        // Cache ibv_qp_ex pointer per local QP, keyed by local index.
        qpx_[0] = ibv_qp_to_qp_ex(cfg_.qp);
        for (int i = 1; i < cfg_.num_qps; i++) {
            qpx_[i] = ibv_qp_to_qp_ex(cfg_.extra_qps[i - 1]);
        }
    }

    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&Proxy::run, this);
        if (cfg_.pin_proxy) {
            pin_thread_to_gpu_numa(cfg_.device_id);
        }
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    void pause() {
        paused_.store(true, std::memory_order_release);
        while (!ack_paused_.load(std::memory_order_acquire)) {}
    }

    void resume() {
        ack_paused_.store(false, std::memory_order_release);
        paused_.store(false, std::memory_order_release);
    }

    void set_epoch(uint32_t epoch) {
        cfg_.epoch = epoch;
        // Per-queue send sequence counters restart every epoch so arrival-queue
        // senders write into slot 0 of each queue on the first post of the epoch.
        memset(sender_seq_, 0, sizeof(sender_seq_));
        // Reset the staging cursor too. Safe because set_epoch is only called
        // while the proxy is paused with no in-flight WRs, so no SGE pointer
        // into the staging buffer is still pending NIC DMA.
        staging_cursor_ = 0;
    }
    int inflight() const { return inflight_; }
    void reset_inflight() {
        inflight_ = 0;
        // Any stashed BARRIER belongs to the epoch being torn down; forget it.
        has_pending_barrier_ = false;
    }
    ProxyTimestamps get_timestamps() const { return ts_; }
    void reset_timestamps() {
        ts_.reset();
        diag_total_loops_ = 0;
        diag_empty_loops_ = 0;
        diag_inflight_limited_ = 0;
        diag_full_batches_ = 0;
        diag_partial_batches_ = 0;
        diag_post_calls_ = 0;
        diag_total_cmds_ = 0;
        pending_post_head_ = 0;
        pending_post_tail_ = 0;
        pending_post_count_ = 0;
        diag_batch_completion_count_ = 0;
        diag_batch_completion_sum_ns_ = 0;
        diag_batch_completion_max_ns_ = 0;
        signaled_post_ns_.clear();
        completion_ns_.clear();
    }

    /**
     * Drain all outstanding completions. Call ONLY while the proxy is paused,
     * to mirror the CX7 Proxy::drain_cq semantics used by session-level epoch
     * transitions and stage barriers.
     */
    int drain_cq() {
        ibv_wc wc[32];
        int total = 0;
        while (inflight_ > 0) {
            int ne = rdma::poll_cq(cfg_.cq, 32, wc);
            if (ne <= 0) {
                std::this_thread::yield();
                continue;
            }
            for (int i = 0; i < ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "proxy_efa drain_cq: CQ error wr_id=%lu status=%d (%s)\n",
                            wc[i].wr_id, wc[i].status,
                            ibv_wc_status_str(wc[i].status));
                }
                if (notify_mode_ == NotifyMode::WriteImm &&
                    wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    continue;
                }
                inflight_ -= (int)wc[i].wr_id;
                total += (int)wc[i].wr_id;
            }
        }
        return total;
    }

    /**
     * Post a 4-byte SRD send of `token` into the peer's stage-barrier slot.
     * Call while the proxy is paused (host thread owns the QP). Always issued
     * on local QP 0 / remote QP 0 for deterministic ordering with the peer.
     */
    void post_stage_barrier(int slot, uint32_t token) {
        if (cfg_.remote_barrier_addr == 0 || cfg_.flag_staging == nullptr ||
            cfg_.flag_staging->mr == nullptr) {
            fprintf(stderr, "proxy_efa: post_stage_barrier called without remote "
                            "barrier or flag staging configured\n");
            return;
        }
        // Claim our own unique slot in the staging ring — mirrors the WRITE cmd
        // path. Previously we unconditionally wrote to host_ptr[0], which raced
        // with the first WRITE of each epoch (staging_cursor_ resets to 0 in
        // set_epoch, so WRITE #1 also wrote to slot 0). The NIC DMAs SGEs
        // asynchronously, so either side could see the other's bytes: either
        // the barrier delivered a packed arrival payload, or — far more
        // visibly at M=2048 — the WRITE delivered epoch=2 (0x00000002) into the
        // arrival queue, which the kernel drained and interpreted as
        // first_tile=1/num_tiles=1, never publishing real arrivals and hanging
        // in shared_reduce_my_slice_static spin.
        const uint32_t barrier_slot =
            (staging_cursor_++) % (uint32_t)cfg_.flag_staging->count;
        cfg_.flag_staging->host_ptr[barrier_slot] = token;
        ibv_qp_ex* qpx = qpx_[0];
        ibv_wr_start(qpx);
        qpx->wr_id = 1;
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;
        ibv_wr_rdma_write(qpx,
            cfg_.remote_barrier_rkey,
            cfg_.remote_barrier_addr + (uint64_t)slot * sizeof(uint32_t));
        ibv_wr_set_ud_addr(qpx, cfg_.dst_ah, cfg_.dst_qpns[0], rdma::QKEY);
        ibv_wr_set_sge(qpx,
            cfg_.flag_staging->mr->lkey,
            (uint64_t)(cfg_.flag_staging->host_ptr + barrier_slot),
            sizeof(uint32_t));
        int ret = ibv_wr_complete(qpx);
        if (ret != 0) {
            fprintf(stderr, "proxy_efa: post_stage_barrier ibv_wr_complete failed: %s\n",
                    strerror(ret));
            return;
        }
        inflight_ += 1;
    }

    /**
     * Return host-side proxy diagnostics for the current epoch, including
     * batch-completion (post→CQE) latency measured on this proxy.
     */
    ProxyDiagnostics get_diagnostics() const {
        ProxyDiagnostics d;
        d.qp_base_idx = cfg_.qp_base_idx;
        d.num_qps = cfg_.num_qps;
        d.loops = diag_total_loops_;
        d.empty_loops = diag_empty_loops_;
        d.full_batches = diag_full_batches_;
        d.partial_batches = diag_partial_batches_;
        d.inflight_limited_loops = diag_inflight_limited_;
        d.batches_posted = diag_post_calls_;
        d.cmds_posted = diag_total_cmds_;
        d.first_cmd_ns = ts_.first_cmd_ns;
        d.first_post_ns = ts_.first_post_ns;
        d.first_completion_ns = ts_.first_completion_ns;
        d.batch_completion_count = diag_batch_completion_count_;
        d.batch_completion_sum_ns = diag_batch_completion_sum_ns_;
        d.batch_completion_max_ns = diag_batch_completion_max_ns_;
        return d;
    }

    /** Per-completion timelines for post→CQE instrumentation. */
    ProxyTimeline get_timeline() const {
        ProxyTimeline out;
        out.signaled_post_ns = signaled_post_ns_;
        out.completion_ns = completion_ns_;
        return out;
    }

    ~Proxy() { stop(); }

private:
    ProxyConfig cfg_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    std::atomic<bool> ack_paused_;
    std::thread thread_;
    int inflight_;
    ProxyTimestamps ts_;
    NotifyMode notify_mode_;
    int effective_max_inflight_;
    ibv_qp_ex* qpx_[kMaxExchangeQPs] = {};       // cached ibv_qp_ex per local QP

    // Per-logical-queue next-slot counters for the arrival-queue layout. The
    // upper bound matches proxy.h (kMaxExchangeQPs * 16 logical queues) so the
    // same lane→queue mapping fits without any per-session realloc.
    uint32_t sender_seq_[kMaxExchangeQPs * 16] = {};

    // Monotonic cursor into cfg_.flag_staging->host_ptr. Each WR that needs
    // its own 4-byte payload claims the next slot via
    //   uint32_t slot = (staging_cursor_++) % cfg_.flag_staging->count;
    // The staging buffer is sized (in session_efa.h) so that count >>
    // worst-case in-flight WRs per proxy, guaranteeing the NIC has DMA'd a
    // slot's bytes well before the cursor wraps back to it. This is a
    // deliberate replacement for IBV_SEND_INLINE, which EFA's extended-verbs
    // SRD path silently drops — without per-WR slots, every batch's SGE
    // points at the same host_ptr offset and the NIC reads whatever value is
    // there at gather time, delivering duplicate flag payloads to the peer.
    uint32_t staging_cursor_ = 0;

    // When a polled command maps to a different local QP than the one the
    // current batch targets, we stash it here and close the batch. The stashed
    // command becomes the seed of the next batch. Mirrors proxy.h's
    // has_pending_cmd_ / pending_cmd_ pair.
    TransferCmd pending_cmd_{};
    bool has_pending_cmd_ = false;

    // Set when the run loop polls a BARRIER_NOTIFY while a batch is still
    // being accumulated. The in-progress batch MUST post before BARRIER goes
    // on the wire: on EFA SRD there is no cross-QP ordering, so a BARRIER on
    // QP 0 that precedes unposted sibling-QP arrival_flag WRITEs can reach
    // the peer first and the peer's iter-end arrival-flag reset will wipe
    // the slots before our late WRITEs land, clobbering the next iter.
    //
    // Flow: polling BARRIER with count>0 sets this flag and breaks out of
    // the accumulate-spin. The current batch posts via the normal path, then
    // the top of the next outer-loop iteration observes the flag, drains the
    // CQ (so every prior WRITE is PCIe-committed at the peer's HBM — SRD
    // WRITE CQE == reliable delivery), and posts BARRIER_NOTIFY.
    bool has_pending_barrier_ = false;
    uint64_t diag_total_loops_ = 0;
    uint64_t diag_empty_loops_ = 0;
    uint64_t diag_inflight_limited_ = 0;
    uint64_t diag_full_batches_ = 0;
    uint64_t diag_partial_batches_ = 0;
    uint64_t diag_post_calls_ = 0;
    uint64_t diag_total_cmds_ = 0;
    // Post→CQE round-trip latency tracking (mirrors CX7 proxy.h batch_completion).
    // Each post records its timestamp; the next CQ poll that retires it computes
    // the delta. Uses a small ring of pending post timestamps (one per batch).
    // R1 Commit 1: raised 64 → 256 to pre-empt the Risk 4 ring-overflow
    // window once adaptive BATCH_SIZE can shrink to 2 under high inflight
    // (256 inflight / 2 min batch = 128 concurrent batches, doubled for
    // headroom). ~3 KB per proxy, single allocation.
    static constexpr int kMaxPendingBatches = 256;
    uint64_t pending_post_ns_[kMaxPendingBatches] = {};
    int pending_post_head_ = 0;  // next write slot
    int pending_post_tail_ = 0;  // next read slot
    int pending_post_count_ = 0;
    uint64_t diag_batch_completion_count_ = 0;
    uint64_t diag_batch_completion_sum_ns_ = 0;
    uint64_t diag_batch_completion_max_ns_ = 0;

    // R1 Commit 1: adaptive post/poll controller state. All values are
    // plain single-thread scalars touched only by the proxy thread (same
    // memory-model regime as inflight_ / staging_cursor_ / diag_*), so no
    // atomics or fences are required. Commit 1 declares and initializes
    // these; they are NOT read by run() yet. Commits 2/3 wire them in.
    bool     use_adaptive_batch_  = false;     // mirror of cfg_.pipeline_enabled
    uint32_t batch_size_current_  = BATCH_SIZE_MAX;  // current batch target
    uint32_t accum_spins_current_ = ACCUMULATE_SPINS;  // current spin budget
    uint64_t cqe_ia_ewma_ns_      = 20000;     // EWMA CQE inter-arrival, 20us warmup
    uint64_t last_cq_poll_ns_     = 0;         // last non-empty poll timestamp
    uint32_t empty_loop_streak_   = 0;         // consecutive outer iters with no post

    // Per-batch timeline vectors for proxy instrumentation.
    std::vector<uint64_t> signaled_post_ns_;
    std::vector<uint64_t> completion_ns_;

    static uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    static NotifyMode detect_notify_mode() {
        const char* env = std::getenv("MKERNEL_EFA_VERBS_NOTIFY_MODE");
        if (!env || !env[0]) return NotifyMode::WriteImm;
        if (std::strcmp(env, "remote_flag") == 0) return NotifyMode::RemoteFlag;
        return NotifyMode::WriteImm;
    }

    /**
     * Pin this thread to the NUMA node of the given CUDA device. Best-effort:
     * ignores errors silently so benchmarks still run on machines without
     * libnuma-style cpusets configured.
     */
    static void pin_thread_to_gpu_numa(int device_id) {
        char gpu_pci[32] = {};
        if (cudaDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), device_id) != cudaSuccess) {
            return;
        }
        const std::string norm_bdf = rdma::normalize_pci_bdf(gpu_pci);
        const int numa = rdma::read_numa_node(
            std::string("/sys/bus/pci/devices/") + norm_bdf + "/numa_node");
        if (numa < 0) return;
        std::string cpulist_path =
            "/sys/devices/system/node/node" + std::to_string(numa) + "/cpulist";
        std::ifstream f(cpulist_path);
        if (!f) return;
        std::string cpulist;
        std::getline(f, cpulist);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        size_t pos = 0;
        while (pos < cpulist.size()) {
            size_t comma = cpulist.find(',', pos);
            std::string tok = cpulist.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            pos = (comma == std::string::npos) ? cpulist.size() : comma + 1;
            size_t dash = tok.find('-');
            if (dash == std::string::npos) {
                int c = std::atoi(tok.c_str());
                if (c >= 0) CPU_SET(c, &cpuset);
            } else {
                int lo = std::atoi(tok.substr(0, dash).c_str());
                int hi = std::atoi(tok.substr(dash + 1).c_str());
                for (int c = lo; c <= hi; ++c) if (c >= 0) CPU_SET(c, &cpuset);
            }
        }
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    void post_write_imm_batch(ibv_qp_ex* qpx, uint32_t dst_qpn,
                              const PerPeerProxyData& eff,
                              const TransferCmd* batch, int count) {
        for (int i = 0; i < count; i++) {
            const TransferCmd& cmd = batch[i];

            // Data write with immediate: local GPU -> remote GPU recv buffer.
            // The immediate payload carries the tile id for receiver-side
            // CQ processing, which keeps SRD notification separate from the
            // existing RC/libfabric remote-flag-write model.
            qpx->wr_id = 1;
            qpx->comp_mask = 0;
            qpx->wr_flags = IBV_SEND_SIGNALED;
            ibv_wr_rdma_write_imm(qpx,
                eff.remote_data_rkey,
                eff.remote_data_addr + cmd.remote_offset,
                htonl(static_cast<uint32_t>(cmd.tile_id)));
            ibv_wr_set_ud_addr(qpx, eff.dst_ah, dst_qpn, rdma::QKEY);
            // Src MR selection: direct-DMA-BUF (src_view=1) sources from the
            // caller-owned output buffer; default (src_view=0) sources from
            // the staging buffer. Strided (src_view=2) isn't implemented here
            // — only the single-SGE direct path is wired up so far.
            if (cfg_.direct_dmabuf_enabled && cmd.src_view == 1) {
                ibv_wr_set_sge(qpx,
                    cfg_.clocal_data_lkey,
                    cfg_.clocal_data_addr + cmd.local_offset,
                    cmd.bytes);
            } else {
                ibv_wr_set_sge(qpx,
                    cfg_.local_data_lkey,
                    cfg_.local_data_addr + cmd.local_offset,
                    cmd.bytes);
            }
        }
    }

    void post_remote_flag_batch(ibv_qp_ex* qpx, uint32_t dst_qpn,
                                const PerPeerProxyData& eff,
                                const TransferCmd* batch, int count) {
        // Pre-batch setup. The two arrival-flag layouts diverge here:
        //   - Flat: stamp the shared epoch payload into host_ptr[0]; every
        //     flag WR sources its 4 bytes from there.
        //   - Queue: derive total_logical_queues for the runtime modulo
        //     used inside the per-cmd loop.
        constexpr uint32_t kTileBytes = 128u * 256u * 2u;
        uint32_t total_logical_queues = 0;
        if (cfg_.use_arrival_queue) {
            total_logical_queues =
                (uint32_t)(cfg_.global_num_qps * cfg_.logical_queues_per_qp);
        } else {
            *cfg_.flag_staging->host_ptr = cfg_.epoch;
        }
        for (int i = 0; i < count; i++) {
            const TransferCmd& cmd = batch[i];
            const bool is_last = (i == count - 1);

            // Data WR: local GPU -> remote GPU. EFA SRD rejects non-signaled
            // WRs in the new-verbs path, so every WR in the chain must be
            // signaled; we account only on the final WR per command so
            // inflight_ still tracks logical commands rather than raw CQEs.
            qpx->wr_id = 0;
            qpx->comp_mask = 0;
            qpx->wr_flags = IBV_SEND_SIGNALED;
            ibv_wr_rdma_write(qpx,
                eff.remote_data_rkey,
                eff.remote_data_addr + cmd.remote_offset);
            ibv_wr_set_ud_addr(qpx, eff.dst_ah, dst_qpn, rdma::QKEY);
            if (cfg_.direct_dmabuf_enabled && cmd.src_view == 1) {
                ibv_wr_set_sge(qpx,
                    cfg_.clocal_data_lkey,
                    cfg_.clocal_data_addr + cmd.local_offset,
                    cmd.bytes);
            } else {
                ibv_wr_set_sge(qpx,
                    cfg_.local_data_lkey,
                    cfg_.local_data_addr + cmd.local_offset,
                    cmd.bytes);
            }

            if (!cfg_.use_arrival_queue) {
                // Flat layout: write the shared epoch into remote_flags[tile_id].
                // Payload (cfg_.epoch) is constant across the batch and was
                // written into host_ptr[0] at the top of this function, so all
                // in-flight WRs may safely share the same SGE source slot.
                qpx->wr_id = 1;
                qpx->comp_mask = 0;
                qpx->wr_flags = IBV_SEND_SIGNALED;
                ibv_wr_rdma_write(qpx,
                    eff.remote_flags_rkey,
                    eff.remote_flags_addr + cmd.tile_id * sizeof(uint32_t));
                ibv_wr_set_ud_addr(qpx, eff.dst_ah, dst_qpn, rdma::QKEY);
                ibv_wr_set_sge(qpx,
                    cfg_.flag_staging->mr->lkey,
                    (uint64_t)cfg_.flag_staging->host_ptr,
                    sizeof(uint32_t));
                (void)is_last;  // silence unused in this branch
            } else {
                // Queue layout: derive logical queue from lane, grab next slot,
                // stage the packed (tile_id, run_tiles) payload, and write it
                // to remote_flags_addr + (q*stride + slot)*4. Each WR that
                // carries a distinct payload claims its own slot in the
                // staging ring via staging_cursor_, so the proxy never
                // overwrites a slot whose WR is still pending NIC DMA — see
                // the staging_cursor_ comment in the private members.
                // Optionally chain a tail publish so the kernel drainer can
                // learn how many arrivals are in its queue without scanning
                // stale words.
                const uint32_t run_tiles = (cmd.bytes + kTileBytes - 1u) / kTileBytes;
                const uint32_t flag_slot =
                    (staging_cursor_++) % (uint32_t)cfg_.flag_staging->count;
                cfg_.flag_staging->host_ptr[flag_slot] =
                    pack_arrival_work((uint32_t)cmd.tile_id, run_tiles);
                const uint32_t logical_q = (total_logical_queues > 0)
                    ? (cmd.lane_id % total_logical_queues) : 0u;
                const uint32_t q_slot = sender_seq_[logical_q]++;
                const uint64_t flag_remote_addr = eff.remote_flags_addr +
                    (uint64_t)(logical_q * cfg_.remote_queue_stride + q_slot) *
                    sizeof(uint32_t);

                // If a tail chain follows, the flag WR is still signaled (SRD
                // requirement) but carries wr_id=0 so accounting happens on
                // the tail completion instead. Otherwise, the flag WR is the
                // last WR for this command and carries wr_id=1.
                const bool has_tail = cfg_.enable_remote_tail &&
                                      eff.remote_tail_addr != 0;
                qpx->wr_id = has_tail ? 0 : 1;
                qpx->comp_mask = 0;
                qpx->wr_flags = IBV_SEND_SIGNALED;
                ibv_wr_rdma_write(qpx, eff.remote_flags_rkey, flag_remote_addr);
                ibv_wr_set_ud_addr(qpx, eff.dst_ah, dst_qpn, rdma::QKEY);
                ibv_wr_set_sge(qpx,
                    cfg_.flag_staging->mr->lkey,
                    (uint64_t)(cfg_.flag_staging->host_ptr + flag_slot),
                    sizeof(uint32_t));

                if (has_tail) {
                    const uint32_t tail_slot =
                        (staging_cursor_++) % (uint32_t)cfg_.flag_staging->count;
                    cfg_.flag_staging->host_ptr[tail_slot] = q_slot + 1u;
                    qpx->wr_id = 1;
                    qpx->comp_mask = 0;
                    qpx->wr_flags = IBV_SEND_SIGNALED;
                    ibv_wr_rdma_write(qpx,
                        eff.remote_tail_rkey,
                        eff.remote_tail_addr + (uint64_t)logical_q * sizeof(uint32_t));
                    ibv_wr_set_ud_addr(qpx, eff.dst_ah, dst_qpn, rdma::QKEY);
                    ibv_wr_set_sge(qpx,
                        cfg_.flag_staging->mr->lkey,
                        (uint64_t)(cfg_.flag_staging->host_ptr + tail_slot),
                        sizeof(uint32_t));
                }
                (void)is_last;  // kept for parity with the flat branch
            }
        }
    }

    /**
     * Main proxy loop using EFA new verbs API. Each iteration selects one
     * local QP (round-robin), collects up to BATCH_SIZE commands, posts them
     * as a single ibv_wr_complete on that QP, then polls the shared CQ.
     *
     * The notification mode is selected by MKERNEL_EFA_VERBS_NOTIFY_MODE:
     *   - write_imm   : data writes carry tile ids via immediate data
     *   - remote_flag : chain data write + remote flag write
     */
    void run() {
        ibv_wc wc[32];
        TransferCmd pending_batch[BATCH_SIZE];
        int pending_count = 0;
        int pending_batch_qp = 0;
        uint64_t pending_cpu_head = 0;
        // Multi-fifo: round-robin index. fifo() resolves to the active fifo
        // for the current iteration. Single-fifo mode keeps the legacy `fifo`
        // pointer (num_fifos == 1).
        int current_fifo_idx = 0;
        auto fifo = [&]() -> D2HFifoHost* {
            if (cfg_.num_fifos > 1 && cfg_.fifos != nullptr) {
                return cfg_.fifos[current_fifo_idx];
            }
            return cfg_.fifo;
        };

        // Map a command's lane_id to one of THIS proxy's local QPs.
        //
        // Kernels route commands to a proxy via `gemm_ar_select_fifo_for_lane`,
        // and may use either `cmd.lane_id` or `cmd.reserved0` as the routing
        // key — both schemes coexist (gemm_ar / ring_attention use lane_id;
        // dispatch_gemm / gemm_rs / ag_gemm use reserved0 to keep work for the
        // same source GPU on one proxy and reduce cross-proxy FIFO contention).
        //
        // The proxy should therefore accept any command that lands in its
        // FIFO and pick a local QP via lane_id-mod-num_qps, rather than
        // validating against a global slice (which only matched the lane_id
        // routing scheme and dropped commands from the reserved0 scheme,
        // causing deadlocks).
        auto map_to_local_qp = [this](const TransferCmd& cmd) -> int {
            if (cfg_.num_qps <= 0) return 0;
            return (int)(cmd.lane_id % (uint32_t)cfg_.num_qps);
        };

        while (running_.load(std::memory_order_relaxed)) {
            // Pause gate
            if (paused_.load(std::memory_order_acquire)) {
                ack_paused_.store(true, std::memory_order_release);
                while (paused_.load(std::memory_order_acquire)) {}
                ack_paused_.store(false, std::memory_order_release);
                if (!ts_.recorded) {
                    ts_.proxy_start_ns = now_ns();
                }
                continue;
            }

            // Pre-BARRIER drain gate. A BARRIER_NOTIFY stashed by the previous
            // accumulate-spin (see has_pending_barrier_ rationale) runs here
            // once the in-progress batch has posted and no cross-QP WRITE is
            // still waiting to seed the next batch. Draining the CQ until
            // inflight_==0 forces every prior WRITE to be PCIe-committed at
            // the remote before BARRIER leaves the wire, which is the
            // ordering required for steady-state correctness on SRD.
            if (has_pending_barrier_ && pending_count == 0 && !has_pending_cmd_) {
                has_pending_barrier_ = false;
                drain_cq();
                post_stage_barrier(/*slot=*/0, cfg_.epoch);
                continue;
            }

            // Step 1: Collect a batch of commands targeting a single local QP.
            // The target QP is derived from the first command's lane_id, so the
            // same lane always hits the same QP (matches proxy.h semantics).
            TransferCmd batch[BATCH_SIZE];
            int count = 0;
            int batch_qp = 0;
            int pre_inflight = inflight_;
            const bool using_pending_batch = (pending_count > 0);

            if (using_pending_batch) {
                // Retry path for a previous ibv_wr_complete failure. Replay
                // the saved batch on the same QP it was destined for.
                count = pending_count;
                batch_qp = pending_batch_qp;
                for (int i = 0; i < pending_count; ++i) {
                    batch[i] = pending_batch[i];
                }
            } else {
                // Multi-fifo: rotate to next fifo for this new batch.
                // Single-fifo (num_fifos==1) is a no-op.
                if (cfg_.num_fifos > 1) {
                    current_fifo_idx = (current_fifo_idx + 1) % cfg_.num_fifos;
                }
                // Seed the batch with a previously-stashed command if any.
                if (has_pending_cmd_ && pre_inflight < effective_max_inflight_) {
                    const int seed_qp = map_to_local_qp(pending_cmd_);
                    if (seed_qp >= 0) {
                        batch[0] = pending_cmd_;
                        batch_qp = seed_qp;
                        count = 1;
                    }
                    has_pending_cmd_ = false;
                }

                // R1 Commit 2: batch cap is BATCH_SIZE (static-mode fallback)
                // or the adaptive `batch_size_current_` target when pipeline
                // flag is on. Flag=0 + C1-initialized state → identical 8.
                const int batch_cap =
                    use_adaptive_batch_ ? (int)batch_size_current_ : BATCH_SIZE;
                while (count < batch_cap &&
                       pre_inflight + count < effective_max_inflight_) {
                    TransferCmd cmd{};
                    if (!fifo()->poll(&cmd)) break;
                    if (cmd.cmd_type == CmdType::BARRIER_NOTIFY) {
                        // Cross-node barrier on SRD requires draining prior
                        // sibling-QP WRITEs before BARRIER leaves the wire.
                        // If no batch is in progress yet (count==0), drain +
                        // post inline. Otherwise stash and flush the batch
                        // first (top-of-loop gate posts BARRIER next iter).
                        if (count > 0) {
                            has_pending_barrier_ = true;
                            break;
                        }
                        drain_cq();
                        post_stage_barrier(/*slot=*/0, cfg_.epoch);
                        continue;
                    }
                    if (cmd.cmd_type != CmdType::WRITE) continue;
                    const int cmd_qp = map_to_local_qp(cmd);
                    if (cmd_qp < 0) {
                        fprintf(stderr,
                                "proxy_efa: lane %u routed to out-of-range local qp "
                                "(base=%d num=%d global_num=%d)\n",
                                (unsigned)cmd.lane_id, cfg_.qp_base_idx,
                                cfg_.num_qps, cfg_.global_num_qps);
                        continue;
                    }
                    if (count == 0) {
                        batch_qp = cmd_qp;
                        batch[count++] = cmd;
                    } else if (cmd_qp == batch_qp) {
                        batch[count++] = cmd;
                    } else {
                        // Different QP — stash and close this batch so we keep
                        // the ibv_wr_start/_complete block single-QP.
                        pending_cmd_ = cmd;
                        has_pending_cmd_ = true;
                        break;
                    }
                }
                if (count > 0) {
                    pending_cpu_head = fifo()->cpu_head;
                }

                // R1 Commit 2: accumulate-spin cap is ACCUMULATE_SPINS (static
                // fallback) or the adaptive `accum_spins_current_` when the
                // pipeline flag is on. Batch target `batch_cap` (above) is
                // reused here to keep the two collection phases consistent.
                const int spin_cap = use_adaptive_batch_
                                         ? (int)accum_spins_current_
                                         : ACCUMULATE_SPINS;
                if (count > 0 && count < batch_cap && !has_pending_cmd_) {
                    int misses = 0;
                    for (int spin = 0;
                         spin < spin_cap &&
                         misses < ACCUMULATE_MISS_BUDGET &&
                         count < batch_cap &&
                         pre_inflight + count < effective_max_inflight_;
                         ++spin) {
                        TransferCmd cmd{};
                        if (!fifo()->poll(&cmd)) {
                            misses++;
                            continue;
                        }
                        misses = 0;
                        if (cmd.cmd_type == CmdType::BARRIER_NOTIFY) {
                            // Mid-batch BARRIER: stash and break so the in-
                            // progress batch flushes via the existing post
                            // path. Next outer-loop iteration drains the CQ
                            // and posts BARRIER. See has_pending_barrier_
                            // member-doc for the SRD ordering rationale.
                            has_pending_barrier_ = true;
                            break;
                        }
                        if (cmd.cmd_type != CmdType::WRITE) continue;
                        const int cmd_qp = map_to_local_qp(cmd);
                        if (cmd_qp < 0) continue;
                        if (cmd_qp != batch_qp) {
                            pending_cmd_ = cmd;
                            has_pending_cmd_ = true;
                            break;
                        }
                        batch[count++] = cmd;
                    }
                    if (count > 0) {
                        pending_cpu_head = fifo()->cpu_head;
                    }
                }
            }

            diag_total_loops_++;
            if (count == 0) {
                diag_empty_loops_++;
            } else if (count == BATCH_SIZE) {
                diag_full_batches_++;
            } else if (pre_inflight + count >= effective_max_inflight_) {
                diag_inflight_limited_++;
            } else {
                diag_partial_batches_++;
            }

            // Step 2: Post RDMA writes on the QP this batch targets.
            bool posted_batch = false;
            if (count > 0) {
                if (!ts_.recorded) {
                    ts_.first_cmd_ns = now_ns();
                }

                ibv_qp_ex* qpx = qpx_[batch_qp];

                // Resolve the per-peer endpoint for this batch. Batches are
                // filled per QP; callers that mix dst_rank values on one QP
                // should split batches by peer before posting.
                int peer_slot = 0;
                if (cfg_.num_peers > 1 && cfg_.peer_slot_by_rank != nullptr) {
                    const uint8_t r = batch[0].dst_rank;
                    const int s = cfg_.peer_slot_by_rank[r];
                    if (s >= 0) peer_slot = s;
                }
                // Session always populates per_peer.
                const PerPeerProxyData& eff = cfg_.per_peer[peer_slot];
                const uint32_t dst_qpn = eff.dst_qpns[batch_qp];

                ibv_wr_start(qpx);
                if (notify_mode_ == NotifyMode::RemoteFlag) {
                    post_remote_flag_batch(qpx, dst_qpn, eff, batch, count);
                } else {
                    post_write_imm_batch(qpx, dst_qpn, eff, batch, count);
                }

                int ret = ibv_wr_complete(qpx);
                if (ret != 0) {
                    fprintf(stderr, "proxy_efa: ibv_wr_complete failed (qp=%d, %d cmds): %s\n",
                            batch_qp, count, strerror(ret));
                    pending_count = count;
                    pending_batch_qp = batch_qp;
                    for (int i = 0; i < count; ++i) {
                        pending_batch[i] = batch[i];
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                } else {
                    const uint64_t post_ns = now_ns();
                    if (!ts_.recorded) {
                        ts_.first_post_ns = post_ns;
                    }
                    diag_post_calls_++;
                    diag_total_cmds_ += count;
                    inflight_ += count;
                    posted_batch = true;
                    pending_count = 0;
                    // Timeline: record post timestamp for each batch.
                    signaled_post_ns_.push_back(post_ns);
                    // Record post timestamp for batch-completion latency.
                    if (pending_post_count_ < kMaxPendingBatches) {
                        pending_post_ns_[pending_post_head_] = post_ns;
                        pending_post_head_ = (pending_post_head_ + 1) % kMaxPendingBatches;
                        pending_post_count_++;
                    }
                }
            }

            // Step 3: Poll CQ (shared across all QPs owned by this proxy)
            int ne = rdma::poll_cq(cfg_.cq, 32, wc);
            uint64_t cq_poll_ns = 0;
            if (ne > 0) {
                cq_poll_ns = now_ns();
                if (!ts_.recorded) {
                    ts_.first_completion_ns = cq_poll_ns;
                    ts_.recorded = true;
                }
                // Timeline: record each CQ poll that returned completions.
                completion_ns_.push_back(cq_poll_ns);
                // Measure post→CQE latency. Each signaled CQE with wr_id>0
                // retires one batch; pop the oldest post timestamp and compute
                // the round-trip.
                for (int i = 0; i < ne; i++) {
                    if (wc[i].wr_id > 0 && pending_post_count_ > 0) {
                        const uint64_t post_ts = pending_post_ns_[pending_post_tail_];
                        pending_post_tail_ = (pending_post_tail_ + 1) % kMaxPendingBatches;
                        pending_post_count_--;
                        const uint64_t lat_ns = cq_poll_ns - post_ts;
                        diag_batch_completion_count_++;
                        diag_batch_completion_sum_ns_ += lat_ns;
                        if (lat_ns > diag_batch_completion_max_ns_) {
                            diag_batch_completion_max_ns_ = lat_ns;
                        }
                    }
                }
            }
            for (int i = 0; i < ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "proxy_efa: CQ error: wr_id=%lu status=%d (%s)\n",
                            wc[i].wr_id, wc[i].status,
                            ibv_wc_status_str(wc[i].status));
                }
                if (notify_mode_ == NotifyMode::WriteImm &&
                    wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    uint32_t tile_id = ntohl(wc[i].imm_data);
                    if (cfg_.local_arrival_host_ptr && tile_id < (uint32_t)cfg_.local_arrival_count) {
                        cfg_.local_arrival_host_ptr[tile_id] = cfg_.epoch;
                        std::atomic_thread_fence(std::memory_order_release);
                    } else {
                        fprintf(stderr, "proxy_efa: bad tile id from imm=%u (count=%d)\n",
                                tile_id, cfg_.local_arrival_count);
                    }
                    continue;
                }
                inflight_ -= (int)wc[i].wr_id;
            }

            // Step 4: Advance FIFO tail
            if (posted_batch) {
                fifo()->advance_tail(pending_cpu_head);
            }
        }

        // Drain
        while (inflight_ > 0) {
            int ne = rdma::poll_cq(cfg_.cq, 32, wc);
            for (int i = 0; i < ne; i++) {
                if (notify_mode_ != NotifyMode::WriteImm ||
                    wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
                    inflight_ -= (int)wc[i].wr_id;
                }
            }
        }

        const char* diag_env = std::getenv("MKERNEL_EFA_PROXY_DIAG");
        if (diag_total_loops_ > 0 && diag_env != nullptr && diag_env[0] == '1') {
            double avg_batch = diag_post_calls_ > 0
                ? static_cast<double>(diag_total_cmds_) / static_cast<double>(diag_post_calls_)
                : 0.0;
            double avg_completion_us = diag_batch_completion_count_ > 0
                ? (double)diag_batch_completion_sum_ns_ / (double)diag_batch_completion_count_ / 1000.0
                : 0.0;
            double max_completion_us = (double)diag_batch_completion_max_ns_ / 1000.0;
            fprintf(stderr,
                    "proxy_efa diag: mode=%s qp_base=%d qps=%d loops=%lu empty=%lu(%.0f%%) "
                    "full_batch=%lu partial=%lu inflight_limited=%lu avg_batch=%.2f "
                    "max_inflight=%d post→CQE avg=%.1fus max=%.1fus (n=%lu) "
                    "timeline: %zu posts %zu completions\n",
                    notify_mode_ == NotifyMode::RemoteFlag ? "remote_flag" : "write_imm",
                    cfg_.qp_base_idx, cfg_.num_qps,
                    diag_total_loops_, diag_empty_loops_,
                    100.0 * diag_empty_loops_ / diag_total_loops_,
                    diag_full_batches_, diag_partial_batches_, diag_inflight_limited_,
                    avg_batch, effective_max_inflight_,
                    avg_completion_us, max_completion_us, diag_batch_completion_count_,
                    signaled_post_ns_.size(), completion_ns_.size());
        }
    }
};


} // namespace internode
