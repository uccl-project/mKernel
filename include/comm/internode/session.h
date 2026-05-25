/**
 * @file session.h
 * @brief Session lifecycle for inter-node RDMA communication.
 *
 * One-call setup and teardown. Wires together all internode modules:
 * RDMA transport, GPU MR registration, D2H FIFO, arrival flags, proxy thread.
 *
 * Usage from Python extension host code:
 *   auto* s = internode::create_session({rank, peer_ip, port, gpu_buf, ...});
 *   // ... launch kernel with s->fifo.device, s->arrival.device_ptr ...
 *   internode::destroy_session(s);
 */
#pragma once

#include <chrono>
#include <thread>
#include "types.h"
#include "rdma_transport.h"
#include "rdma_gpu_mr.cuh"
#include "d2h_fifo.cuh"
#include "../device_clock.cuh"
#include "arrival.cuh"
#include "proxy.h"
#include "ready_queue.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace internode {

inline uint64_t q2_host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static __global__ void q2_read_globaltimer_kernel(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = comm::globaltimer();
    }
}

inline int64_t calibrate_gpu_to_host_offset_ns(int samples = 32) {
    if (samples <= 0) samples = 1;
    unsigned long long* dev_out = nullptr;
    unsigned long long host_out = 0;
    cudaStream_t stream = nullptr;
    MKERNEL_CUDACHECK(cudaMalloc(&dev_out, sizeof(unsigned long long)));
    MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t best_window_ns = std::numeric_limits<uint64_t>::max();
    int64_t best_offset_ns = 0;
    for (int i = 0; i < samples; ++i) {
        const uint64_t host_before_ns = q2_host_now_ns();
        q2_read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        MKERNEL_CUDACHECK(cudaGetLastError());
        MKERNEL_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        MKERNEL_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = q2_host_now_ns();
        const uint64_t window_ns = host_after_ns - host_before_ns;
        if (window_ns < best_window_ns) {
            best_window_ns = window_ns;
            const uint64_t host_mid_ns = host_before_ns + window_ns / 2ULL;
            best_offset_ns = (int64_t)host_mid_ns - (int64_t)host_out;
        }
    }

    MKERNEL_CUDACHECK(cudaStreamDestroy(stream));
    MKERNEL_CUDACHECK(cudaFree(dev_out));
    return best_offset_ns;
}

// ---------------------------------------------------------------------------
// PCIe-topology-aware NIC selection
// ---------------------------------------------------------------------------
//
// Picks the mlx5 device that shares the same PCIe root complex (NUMA node)
// as the given CUDA device. Falls back to mlx5_{device_id} then mlx5_0 if
// auto-detection fails.
//
// Strategy:
//   1. Read /sys/bus/pci/devices/<gpu_pci>/numa_node for the GPU's NUMA node
//   2. For each /sys/class/infiniband/mlx5_N, read its numa_node
//   3. Match. If multiple match, pick by GPU index modulo number of matches.
inline std::string select_nic_for_device(int device_id) {
    auto read_numa_node = [](const std::string& path) -> int {
        std::ifstream f(path);
        if (!f) return -1;
        int n = -1;
        f >> n;
        return n;
    };

    // 1. Get GPU PCIe BDF from CUDA
    int gpu_domain = 0, gpu_bus = 0, gpu_dev = 0;
    if (cudaDeviceGetAttribute(&gpu_domain, cudaDevAttrPciDomainId, device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&gpu_bus, cudaDevAttrPciBusId, device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&gpu_dev, cudaDevAttrPciDeviceId, device_id) != cudaSuccess) {
        return "";
    }
    char gpu_pci[32];
    snprintf(gpu_pci, sizeof(gpu_pci), "%04x:%02x:%02x.0", gpu_domain, gpu_bus, gpu_dev);
    std::string gpu_numa_path = std::string("/sys/bus/pci/devices/") + gpu_pci + "/numa_node";
    int gpu_numa = read_numa_node(gpu_numa_path);

    // 2. Enumerate mlx5 devices and read their NUMA node
    struct NicInfo {
        std::string name;
        int numa_node;
    };
    std::vector<NicInfo> nics;
    DIR* dir = opendir("/sys/class/infiniband");
    if (dir) {
        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            std::string name = ent->d_name;
            if (name.rfind("mlx5_", 0) != 0) continue;  // skip non-mlx5
            // Check if port 1 is active (state=4)
            std::string state_path = "/sys/class/infiniband/" + name + "/ports/1/state";
            std::ifstream sf(state_path);
            std::string state_line;
            if (sf) std::getline(sf, state_line);
            if (state_line.find("ACTIVE") == std::string::npos) continue;  // skip inactive
            // Read its NUMA node from the underlying PCI device
            std::string numa_path = "/sys/class/infiniband/" + name + "/device/numa_node";
            int numa = read_numa_node(numa_path);
            nics.push_back({name, numa});
        }
        closedir(dir);
    }

    if (nics.empty()) return "";

    // 3. Filter to NICs matching GPU NUMA, then pick by device_id stride
    std::vector<NicInfo> match;
    for (const auto& n : nics) {
        if (n.numa_node == gpu_numa || gpu_numa < 0 || n.numa_node < 0) {
            match.push_back(n);
        }
    }
    if (match.empty()) match = nics;  // fall back to all

    // Sort by name for determinism
    std::sort(match.begin(), match.end(),
              [](const NicInfo& a, const NicInfo& b) { return a.name < b.name; });

    // Round-robin GPUs across same-NUMA NICs
    return match[device_id % match.size()].name;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SessionConfig {
    int         rank;                // 0..num_nodes-1
    // peer_ips[i] / peer_tcp_ports[i] describe peer slot i. peer_ranks[i]
    // optionally overrides the implicit "skip self" mapping from slot →
    // global rank.
    int                num_peers       = 0;
    const char* const* peer_ips        = nullptr;
    const int*         peer_tcp_ports  = nullptr;
    const int*         peer_ranks      = nullptr;

    void*       local_gpu_buf;       // existing GPU buffer to register for RDMA reads
    size_t      local_gpu_buf_size;  // size of local_gpu_buf

    // Optional second buffer to register as a DMA-BUF MR (for direct-from-
    // C_local sends that bypass the staging copy). Required when
    // direct_dmabuf_enabled is true. Must point to a VMM-backed allocation
    // (DistBuffer::data_). Registration uses ibv_reg_dmabuf_mr only —
    // no nvidia_peermem / ibv_reg_mr fallback.
    void*       clocal_gpu_buf = nullptr;
    size_t      clocal_gpu_buf_size = 0;
    bool        direct_dmabuf_enabled = false;
    // Row stride in bytes for strided direct-gather sends (src_view=2).
    // Typically N * sizeof(bf16) for GEMM output. Used by the proxy to
    // compute per-row SGE addresses inside C_local.
    size_t      row_stride_bytes = 0;

    size_t      recv_buf_size;       // size of receive buffer to allocate on GPU
    void*       external_recv_buf = nullptr; // optional caller-owned RDMA target
    int         num_tiles;           // number of arrival flag slots
    int         fifo_capacity;       // D2H FIFO capacity (default 1024, must be power of 2)
    int         device_id;           // CUDA device index

    int         max_inflight;        // proxy max outstanding WRs (default 128)

    int         num_qps;             // number of QPs to create (0 or 1 = single QP, max 24)
    int         logical_queues_per_qp = 1; // software queues mapped onto each QP
    int         num_proxy_threads = 1;     // host proxy threads / FIFO channels
    bool        channelize_gpu_peers = false; // route peer/GPU channels through stable QPs
    bool        use_write_imm = false;
    int         ready_queue_cap = 0;
    bool        use_arrival_queue = false;
    bool        enable_forward_notify = false;

    const char* nic_name;            // IB device name (e.g., "mlx5_4"); NULL = auto-select based on device_id

    bool        pin_proxy = true;    // pin proxy thread to GPU's NUMA node (default true)
};

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/** Maximum number of QPs per session (QP 0 + up to 23 extra). */
static constexpr int kMaxQPs = 24;
static constexpr int kStageBarrierSlots = 2;

struct Session {
    // RDMA resources (QP 0 / CQ 0).
    ibv_context*  ctx;
    ibv_pd*       pd;
    ibv_cq*       cq;       // CQ 0
    ibv_qp*       qp;       // QP 0

    // Extra QPs/CQs for future multi-QP support (indices 1..num_qps-1).
    // QPs are indexed by global QP id. CQs are stored per proxy thread and QPs
    // within a proxy thread share the same CQ.
    ibv_qp*       extra_qps[kMaxQPs - 1];  // extra_qps[0] = QP 1, etc.
    ibv_cq*       proxy_cqs[kMaxProxyThreads]; // proxy_cqs[0] aliases cq
    int           num_qps;                  // total QP count (1..kMaxQPs)

    // Local data buffer MR (registered over caller's existing GPU buffer)
    ibv_mr*       local_data_mr;

    // Optional DMA-BUF MR over the C_local (DistBuffer data_) buffer.
    // Non-null only when SessionConfig::direct_dmabuf_enabled was set and the
    // strict DMA-BUF registration succeeded.
    ibv_mr*       clocal_data_mr = nullptr;

    // Receive buffer (allocated on GPU, RDMA-registered for remote writes)
    gpu_mr::GpuRdmaBuffer recv_buf;
    bool          recv_buf_external = false;

    // D2H command FIFOs — one per proxy thread / FIFO channel.
    D2HFifoPair   fifos[kMaxProxyThreads];
    D2HFifoDeviceBundle fifo_bundle;
    int           num_proxy_threads;

    // Arrival flags + per-proxy metadata staging
    ArrivalFlags  arrival;
    FlagStaging   flag_stagings[kMaxProxyThreads];
    StageBarrierFlags stage_barrier;
    ForwardNotifyTable forward_notify;

    // CPU proxy threads.
    Proxy*        proxies[kMaxProxyThreads];

    // Remote connection info (populated after TCP exchange)
    ConnectionInfo remote_info;
    ConnectionInfo remote_infos[kMaxPeers];
    int            num_remote_peers = 1;

    // Session config (for reference)
    int rank;
    uint32_t epoch;
};

// ---------------------------------------------------------------------------
// Create / Destroy
// ---------------------------------------------------------------------------

/**
 * Create an inter-node RDMA session.
 *
 * Steps:
 *   1. Open IB device, create PD, CQ, RC QP
 *   2. Register local GPU buffer for RDMA (so NIC can read for sends)
 *   3. Allocate + register receive buffer on GPU (remote writes land here)
 *   4. Create D2H FIFO (host-pinned triggers, device-memory head)
 *   5. Create arrival flags (device memory, RDMA-registered)
 *   6. Create flag staging (host-pinned, RDMA-registered)
 *   7. Fill local ConnectionInfo, exchange with peer over TCP
 *   8. Transition QP: INIT → RTR → RTS
 *   9. Spawn CPU proxy thread
 */
inline Session* create_session(const SessionConfig& cfg) {
    Session* s = new Session{};
    s->rank = cfg.rank;
    s->epoch = 1;

    // Determine effective QP count (0 treated as 1, clamped to kMaxQPs)
    int effective_num_qps = (cfg.num_qps <= 0) ? 1 : cfg.num_qps;
    if (cfg.channelize_gpu_peers && cfg.num_peers > 1) {
        effective_num_qps = std::max(effective_num_qps, cfg.num_peers * 8);
    }
    if (effective_num_qps > kMaxQPs) effective_num_qps = kMaxQPs;
    s->num_qps = effective_num_qps;
    int requested_proxy_threads = cfg.num_proxy_threads <= 0 ? 1 : cfg.num_proxy_threads;
    if (requested_proxy_threads > kMaxProxyThreads) requested_proxy_threads = kMaxProxyThreads;
    if (requested_proxy_threads > s->num_qps) requested_proxy_threads = s->num_qps;
    s->num_proxy_threads = std::max(1, requested_proxy_threads);
    int logical_queues_per_qp = cfg.logical_queues_per_qp;
    if (logical_queues_per_qp <= 0) logical_queues_per_qp = 1;
    if (logical_queues_per_qp > 16) logical_queues_per_qp = 16;
    const int total_logical_queues = s->num_qps * logical_queues_per_qp;
    const int logical_queue_stride =
        std::max(1, (cfg.num_tiles + total_logical_queues - 1) / total_logical_queues);
    const int total_arrival_slots = logical_queue_stride * total_logical_queues;
    const int qps_per_proxy =
        std::max(1, (s->num_qps + s->num_proxy_threads - 1) / s->num_proxy_threads);

    // Zero-init extra QP/CQ arrays
    memset(s->extra_qps, 0, sizeof(s->extra_qps));
    memset(s->proxy_cqs, 0, sizeof(s->proxy_cqs));
    memset(s->proxies, 0, sizeof(s->proxies));
    memset(s->flag_stagings, 0, sizeof(s->flag_stagings));

    // RDMA resources (QP 0 / CQ 0).
    // NIC selection priority:
    //   1. Explicit cfg.nic_name (highest priority)
    //   2. PCIe-topology-aware auto-select (matches GPU's NUMA node)
    //   3. mlx5_{device_id} fallback
    //   4. mlx5_0 default
    std::string auto_nic;
    char nic_buf[32];
    const char* nic = cfg.nic_name;
    if (!nic || nic[0] == '\0') {
        auto_nic = select_nic_for_device(cfg.device_id);
        if (!auto_nic.empty()) {
            nic = auto_nic.c_str();
        } else {
            snprintf(nic_buf, sizeof(nic_buf), "mlx5_%d", cfg.device_id);
            nic = nic_buf;
        }
    }
    s->ctx = rdma::open_device(nic);
    s->pd  = rdma::alloc_pd(s->ctx);
    s->cq  = rdma::create_cq(s->ctx, 4096);
    s->proxy_cqs[0] = s->cq;
    for (int t = 1; t < s->num_proxy_threads; t++) {
        s->proxy_cqs[t] = rdma::create_cq(s->ctx, 4096);
    }
    // QP SGE capacity: 2 default; for DMA-BUF strided gather we need up to 32
    // to pack ROW_BLOCK=128 rows into 4 WRs. Clamped to device cap.
    int qp_max_send_sge = 2;
    if (cfg.direct_dmabuf_enabled) {
        ibv_device_attr dev_attr{};
        if (ibv_query_device(s->ctx, &dev_attr) == 0) {
            qp_max_send_sge = std::min(32, dev_attr.max_sge);
            if (qp_max_send_sge < 8) {
                fprintf(stderr,
                        "session: direct_dmabuf_enabled but device max_sge=%d < 8 — "
                        "cannot support strided gather. Exiting.\n", dev_attr.max_sge);
                exit(EXIT_FAILURE);
            }
        } else {
            qp_max_send_sge = 32;  // hopeful default on mlx5
        }
    }
    // SQ depth: proxy batches WRs via ibv_post_send so 2048 is plenty.
    int qp_max_send_wr = 2048;
    s->qp  = rdma::create_rc_qp(s->pd, s->cq, qp_max_send_wr, qp_max_send_sge);
    rdma::modify_qp_init(s->qp);

    // Extra QPs (indices 1..num_qps-1), sharing one CQ per proxy thread.
    for (int i = 1; i < s->num_qps; i++) {
        const int owner_thread = std::min(s->num_proxy_threads - 1, i / qps_per_proxy);
        s->extra_qps[i - 1] = rdma::create_rc_qp(s->pd, s->proxy_cqs[owner_thread], qp_max_send_wr, qp_max_send_sge);
        rdma::modify_qp_init(s->extra_qps[i - 1]);
    }

    // Register local GPU buffer.
    // Try DMA-BUF first for VMM-backed buffers, then fall back to nvidia_peermem
    // for cudaMalloc pointers. Raw ibv_reg_mr does not work for VMM pointers.
    s->local_data_mr = gpu_mr::register_gpu_buffer(
        s->pd, cfg.local_gpu_buf, cfg.local_gpu_buf_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        | IBV_ACCESS_RELAXED_ORDERING);
    if (!s->local_data_mr) {
        fprintf(stderr, "session: register_gpu_buffer(local_gpu_buf) failed: %s\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    // Optional: DMA-BUF-only registration of C_local (direct-send MR).
    if (cfg.direct_dmabuf_enabled) {
        if (cfg.clocal_gpu_buf == nullptr || cfg.clocal_gpu_buf_size == 0) {
            fprintf(stderr,
                    "session: direct_dmabuf_enabled but clocal_gpu_buf not provided\n");
            exit(EXIT_FAILURE);
        }
        s->clocal_data_mr = gpu_mr::register_gpu_buffer_dmabuf_only(
            s->pd, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
            | IBV_ACCESS_RELAXED_ORDERING,
            nullptr);
        if (!s->clocal_data_mr) {
            fprintf(stderr,
                    "session: DMA-BUF-only registration of C_local failed "
                    "(ptr=%p bytes=%zu). No peermem fallback permitted — exiting.\n",
                    cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size);
            exit(EXIT_FAILURE);
        }
    }

    // Allocate + register receive buffer.
    if (cfg.external_recv_buf != nullptr) {
        s->recv_buf.gpu_ptr = cfg.external_recv_buf;
        s->recv_buf.size = cfg.recv_buf_size;
        s->recv_buf.mr = gpu_mr::register_gpu_buffer(
            s->pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        s->recv_buf_external = true;
    } else {
        s->recv_buf = gpu_mr::alloc_and_register(
            s->pd, cfg.recv_buf_size, cfg.device_id);
    }
    if (!s->recv_buf.mr) {
        fprintf(stderr, "session: register receive buffer failed: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // D2H FIFO.
    int fifo_cap = cfg.fifo_capacity > 0 ? cfg.fifo_capacity : 1024;
    s->fifo_bundle = D2HFifoDeviceBundle{};
    s->fifo_bundle.num_fifos = s->num_proxy_threads;
    s->fifo_bundle.global_num_qps = s->num_qps;
    s->fifo_bundle.logical_queues_per_qp = logical_queues_per_qp;
    s->fifo_bundle.qps_per_fifo = qps_per_proxy;
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->fifos[t] = create_d2h_fifo(fifo_cap);
        s->fifo_bundle.fifos[t] = s->fifos[t].device;
    }

    // Arrival flags.
    s->arrival = create_arrival_flags(total_arrival_slots, total_logical_queues);
    // Register arrival flags for RDMA (remote writes 4-byte epoch here)
    s->arrival.mr = rdma::reg_mr(s->pd, (void*)s->arrival.device_ptr,
                                  ((size_t)total_arrival_slots + (size_t)total_logical_queues) * sizeof(uint32_t),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                  | IBV_ACCESS_RELAXED_ORDERING);

    // Per-proxy flag staging.
    for (int t = 0; t < s->num_proxy_threads; t++) {
        // +2 extra slots reserved for strided-direct single-cmd posts
        // (flag and tail scratch, read via IBV_SEND_INLINE).
        s->flag_stagings[t] = create_flag_staging(Proxy::BATCH_SIZE * 2 + 2);
        s->flag_stagings[t].mr = rdma::reg_mr(
            s->pd, s->flag_stagings[t].host_ptr,
            s->flag_stagings[t].count * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE);
    }

    // Stage barrier flags.
    s->stage_barrier = create_stage_barrier_flags(kStageBarrierSlots);
    s->stage_barrier.mr = rdma::reg_mr(s->pd, (void*)s->stage_barrier.host_ptr,
                                       kStageBarrierSlots * sizeof(uint32_t),
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                       | IBV_ACCESS_RELAXED_ORDERING);

    // Optional host-polled forward notifications.
    if (cfg.enable_forward_notify) {
        s->forward_notify = create_forward_notify_table(total_arrival_slots);
        s->forward_notify.mr = rdma::reg_mr(
            s->pd, (void*)s->forward_notify.host_ptr,
            (size_t)s->forward_notify.count * sizeof(ForwardNotify),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_RELAXED_ORDERING);
    }

    // TCP exchange.
    ConnectionInfo local_info{};
    rdma::fill_local_info(local_info, s->qp, s->ctx);

    // Fill buffer info
    local_info.data_rkey = s->recv_buf.mr->rkey;
    local_info.data_addr = (uint64_t)s->recv_buf.gpu_ptr;
    local_info.data_len  = s->recv_buf.size;
    local_info.flags_rkey = s->arrival.mr->rkey;
    local_info.flags_addr = (uint64_t)s->arrival.device_ptr;
    local_info.tail_rkey = s->arrival.mr->rkey;
    local_info.tail_addr = (uint64_t)s->arrival.tail_device_ptr;
    local_info.barrier_rkey = s->stage_barrier.mr->rkey;
    local_info.barrier_addr = (uint64_t)s->stage_barrier.host_ptr;
    if (s->forward_notify.mr != nullptr) {
        local_info.forward_notify_rkey = s->forward_notify.mr->rkey;
        local_info.forward_notify_addr = (uint64_t)s->forward_notify.host_ptr;
        local_info.forward_notify_count = (uint32_t)s->forward_notify.count;
    }

    // Multi-QP: fill extra QP info for TCP exchange
    local_info.num_qps = s->num_qps;
    for (int i = 1; i < s->num_qps; i++) {
        local_info.extra_qp_nums[i - 1] = s->extra_qps[i - 1]->qp_num;
        local_info.extra_psns[i - 1] = 0;  // PSN 0 for extra QPs
    }

    // Multi-peer normalization. Slot order is the same skip-self order used by
    // peer_rank_for_slot(), so device-side reserved0=peer_slot*8+gpu maps to
    // the QP subset connected to that peer.
    const int num_remote_peers = cfg.num_peers;
    if (num_remote_peers <= 0 || num_remote_peers > kMaxPeers) {
        fprintf(stderr, "session.h (CX7): num_peers=%d out of range [1, %d]\n",
                num_remote_peers, kMaxPeers);
        delete s;
        return nullptr;
    }
    s->num_remote_peers = num_remote_peers;
    memset(s->remote_infos, 0, sizeof(s->remote_infos));
    const int inferred_num_nodes = num_remote_peers + 1;
    const int qps_per_peer = cfg.channelize_gpu_peers
        ? std::max(1, s->num_qps / num_remote_peers)
        : s->num_qps;
    auto qp_for_global = [&](int global_qp) -> ibv_qp* {
        return (global_qp == 0) ? s->qp : s->extra_qps[global_qp - 1];
    };
    auto local_psn_for_global = [&](int global_qp) -> uint32_t {
        return (global_qp == 0) ? local_info.psn : local_info.extra_psns[global_qp - 1];
    };

    auto peer_rank_for_local_slot = [&](int peer_slot) -> int {
        return cfg.peer_ranks
            ? cfg.peer_ranks[peer_slot]
            : peer_rank_for_slot(cfg.rank, inferred_num_nodes, peer_slot);
    };
    auto peer_slot_for_rank = [&](int peer_rank) -> int {
        for (int slot = 0; slot < num_remote_peers; ++slot) {
            if (peer_rank_for_local_slot(slot) == peer_rank) return slot;
        }
        return -1;
    };

    // Bring up unordered node pairs in a global order to avoid N>2 bootstrap
    // cycles such as 0->1, 1->2, 2->0 when every node uses local peer order.
    for (int lo = 0; lo < inferred_num_nodes; ++lo) {
        for (int hi = lo + 1; hi < inferred_num_nodes; ++hi) {
            if (cfg.rank != lo && cfg.rank != hi) continue;
            const int peer_rank = (cfg.rank == lo) ? hi : lo;
            const int peer_slot = peer_slot_for_rank(peer_rank);
            if (peer_slot < 0) {
                fprintf(stderr,
                        "session.h: could not map peer_rank=%d for rank=%d "
                        "(num_peers=%d)\n",
                        peer_rank, cfg.rank, num_remote_peers);
                delete s;
                return nullptr;
            }
            const char* sess_peer_ip = cfg.peer_ips[peer_slot];
            const int sess_tcp_port  = cfg.peer_tcp_ports[peer_slot];
            const bool is_server = (cfg.rank == lo);
            s->remote_infos[peer_slot] =
                rdma::exchange_info_tcp(local_info, sess_peer_ip, sess_tcp_port, is_server);
        }
    }
    s->remote_info = s->remote_infos[0];

    // QP transitions.
    for (int peer_slot = 0; peer_slot < num_remote_peers; ++peer_slot) {
        const int peer_rank = cfg.peer_ranks
            ? cfg.peer_ranks[peer_slot]
            : peer_rank_for_slot(cfg.rank, inferred_num_nodes, peer_slot);
        const int remote_slot = (num_remote_peers == 1)
            ? 0
            : slot_at_peer(cfg.rank, peer_rank, inferred_num_nodes);
        const int local_qp_base = cfg.channelize_gpu_peers ? peer_slot * qps_per_peer : 0;
        const int remote_qp_base = cfg.channelize_gpu_peers ? remote_slot * qps_per_peer : 0;
        const int local_qp_end = cfg.channelize_gpu_peers
            ? std::min(s->num_qps, local_qp_base + qps_per_peer)
            : s->num_qps;
        for (int local_qp = local_qp_base, remote_qp = remote_qp_base;
             local_qp < local_qp_end;
             ++local_qp, ++remote_qp) {
            ConnectionInfo remote_qp_info = s->remote_infos[peer_slot];
            if (remote_qp == 0) {
                remote_qp_info.qp_num = s->remote_infos[peer_slot].qp_num;
                remote_qp_info.psn = s->remote_infos[peer_slot].psn;
            } else {
                remote_qp_info.qp_num = s->remote_infos[peer_slot].extra_qp_nums[remote_qp - 1];
                remote_qp_info.psn = s->remote_infos[peer_slot].extra_psns[remote_qp - 1];
            }
            rdma::modify_qp_rtr(qp_for_global(local_qp), remote_qp_info);
            rdma::modify_qp_rts(qp_for_global(local_qp), local_psn_for_global(local_qp));
        }
    }

    // Proxy threads.
    const int64_t gpu_to_host_offset_ns = calibrate_gpu_to_host_offset_ns(64);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int local_qps = std::min(qps_per_proxy, s->num_qps - qp_base);
        if (local_qps <= 0) break;
        ProxyConfig pcfg{};
        pcfg.fifo = &s->fifos[t].host;
        pcfg.qp = (qp_base == 0) ? s->qp : s->extra_qps[qp_base - 1];
        pcfg.cq = s->proxy_cqs[t];
        pcfg.num_qps = local_qps;
        pcfg.qp_base_idx = qp_base;
        pcfg.global_num_qps = s->num_qps;
        memset(pcfg.extra_qps, 0, sizeof(pcfg.extra_qps));
        for (int i = 1; i < local_qps; i++) {
            const int global_qp = qp_base + i;
            pcfg.extra_qps[i - 1] = s->extra_qps[global_qp - 1];
        }
        pcfg.local_data_addr = (uint64_t)cfg.local_gpu_buf;
        pcfg.local_data_lkey = s->local_data_mr->lkey;
        if (s->clocal_data_mr != nullptr) {
            pcfg.clocal_data_addr = (uint64_t)cfg.clocal_gpu_buf;
            pcfg.clocal_data_lkey = s->clocal_data_mr->lkey;
            pcfg.clocal_data_bytes = cfg.clocal_gpu_buf_size;
            pcfg.direct_dmabuf_enabled = true;
            pcfg.row_stride_bytes = cfg.row_stride_bytes;
            pcfg.max_send_sge = (uint32_t)qp_max_send_sge;
        }
        pcfg.remote_data_addr = s->remote_info.data_addr;
        pcfg.remote_data_rkey = s->remote_info.data_rkey;
        pcfg.flag_staging = &s->flag_stagings[t];
        pcfg.remote_flags_addr = s->remote_info.flags_addr;
        pcfg.remote_flags_rkey = s->remote_info.flags_rkey;
        pcfg.remote_tail_addr = s->remote_info.tail_addr;
        pcfg.remote_tail_rkey = s->remote_info.tail_rkey;
        memset(pcfg.remote_data_addr_by_qp, 0, sizeof(pcfg.remote_data_addr_by_qp));
        memset(pcfg.remote_data_rkey_by_qp, 0, sizeof(pcfg.remote_data_rkey_by_qp));
        memset(pcfg.remote_flags_addr_by_qp, 0, sizeof(pcfg.remote_flags_addr_by_qp));
        memset(pcfg.remote_flags_rkey_by_qp, 0, sizeof(pcfg.remote_flags_rkey_by_qp));
        memset(pcfg.remote_tail_addr_by_qp, 0, sizeof(pcfg.remote_tail_addr_by_qp));
        memset(pcfg.remote_tail_rkey_by_qp, 0, sizeof(pcfg.remote_tail_rkey_by_qp));
        memset(pcfg.remote_barrier_addr_by_qp, 0, sizeof(pcfg.remote_barrier_addr_by_qp));
        memset(pcfg.remote_barrier_rkey_by_qp, 0, sizeof(pcfg.remote_barrier_rkey_by_qp));
        memset(pcfg.remote_forward_notify_addr_by_qp, 0, sizeof(pcfg.remote_forward_notify_addr_by_qp));
        memset(pcfg.remote_forward_notify_rkey_by_qp, 0, sizeof(pcfg.remote_forward_notify_rkey_by_qp));
        for (int q = 0; q < s->num_qps && q < kMaxExchangeQPs; ++q) {
            const int peer_slot = cfg.channelize_gpu_peers
                ? std::min(num_remote_peers - 1, q / qps_per_peer)
                : 0;
            const ConnectionInfo& ri = s->remote_infos[peer_slot];
            pcfg.remote_data_addr_by_qp[q] = ri.data_addr;
            pcfg.remote_data_rkey_by_qp[q] = ri.data_rkey;
            pcfg.remote_flags_addr_by_qp[q] = ri.flags_addr;
            pcfg.remote_flags_rkey_by_qp[q] = ri.flags_rkey;
            pcfg.remote_tail_addr_by_qp[q] = ri.tail_addr;
            pcfg.remote_tail_rkey_by_qp[q] = ri.tail_rkey;
            pcfg.remote_barrier_addr_by_qp[q] = ri.barrier_addr;
            pcfg.remote_barrier_rkey_by_qp[q] = ri.barrier_rkey;
            pcfg.remote_forward_notify_addr_by_qp[q] = ri.forward_notify_addr;
            pcfg.remote_forward_notify_rkey_by_qp[q] = ri.forward_notify_rkey;
        }
        pcfg.use_arrival_queue   = cfg.use_arrival_queue;
        pcfg.remote_queue_stride = (uint32_t)logical_queue_stride;
        pcfg.logical_queues_per_qp = logical_queues_per_qp;
        pcfg.channelize_gpu_peers = cfg.channelize_gpu_peers;
        pcfg.enable_remote_tail = (std::getenv("Q2_SENDER_PUBLISHED_TAIL") != nullptr
                                   && std::getenv("Q2_SENDER_PUBLISHED_TAIL")[0] == '1');
        pcfg.remote_barrier_addr = s->remote_info.barrier_addr;
        pcfg.remote_barrier_rkey = s->remote_info.barrier_rkey;
        pcfg.local_forward_notify = s->forward_notify.host_ptr;
        pcfg.forward_notify_slots = s->forward_notify.count;
        pcfg.remote_forward_notify_addr = s->remote_info.forward_notify_addr;
        pcfg.remote_forward_notify_rkey = s->remote_info.forward_notify_rkey;
        pcfg.enable_forward_notify = cfg.enable_forward_notify && !cfg.use_arrival_queue;
        pcfg.rank = cfg.rank;
        pcfg.num_nodes = inferred_num_nodes;
        const int ring_banks = pcfg.enable_forward_notify && num_remote_peers > 1
            ? num_remote_peers : 1;
        pcfg.total_chunks =
            total_arrival_slots / std::max(1, num_remote_peers * ring_banks);
        pcfg.a_half_bytes =
            cfg.local_gpu_buf_size / (size_t)std::max(1, num_remote_peers * ring_banks);
        pcfg.epoch = s->epoch;
        pcfg.max_inflight = cfg.max_inflight > 0 ? cfg.max_inflight : 512;
        pcfg.device_id = cfg.device_id;
        pcfg.pin_proxy = cfg.pin_proxy;
        pcfg.gpu_to_host_offset_ns = gpu_to_host_offset_ns;

        s->proxies[t] = new Proxy(pcfg);
        s->proxies[t]->start();
    }

    return s;
}

/**
 * Wait for the proxy to finish processing all FIFO commands and complete
 * all in-flight RDMA writes. Call after cudaDeviceSynchronize() to ensure
 * no new FIFO pushes will arrive.
 */
inline void drain_proxy(Session* s) {
    uint64_t fifo_heads[kMaxProxyThreads]{};
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaMemcpy(&fifo_heads[t], s->fifos[t].device.head, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Wait until proxy consumed all FIFO commands and all WRs completed.
    for (int i = 0; i < 50000; i++) {  // 5s timeout
        bool drained = true;
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (s->fifos[t].host.cpu_head < fifo_heads[t] || s->proxies[t]->inflight() != 0) {
                drained = false;
                break;
            }
        }
        if (drained)
            break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

/**
 * Destroy an inter-node session. Stops proxy, deregisters MRs, frees everything.
 */
inline void destroy_session(Session* s) {
    if (!s) return;

    // Ensure proxy has processed all FIFO commands and RDMA writes before shutdown.
    // Without this, the remote node may never receive some tile arrivals.
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

    // Deregister MRs
    if (s->arrival.mr)     rdma::dereg_mr(s->arrival.mr);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->flag_stagings[t].mr) rdma::dereg_mr(s->flag_stagings[t].mr);
    }
    if (s->stage_barrier.mr) rdma::dereg_mr(s->stage_barrier.mr);
    if (s->forward_notify.mr) rdma::dereg_mr(s->forward_notify.mr);
    if (s->clocal_data_mr) rdma::dereg_mr(s->clocal_data_mr);
    if (s->local_data_mr)  rdma::dereg_mr(s->local_data_mr);

    // Free buffers
    if (s->recv_buf_external) {
        if (s->recv_buf.mr) rdma::dereg_mr(s->recv_buf.mr);
        s->recv_buf = gpu_mr::GpuRdmaBuffer{};
    } else {
        gpu_mr::free_buffer(s->recv_buf);
    }
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_d2h_fifo(s->fifos[t]);
    }
    destroy_arrival_flags(s->arrival);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_flag_staging(s->flag_stagings[t]);
    }
    destroy_stage_barrier_flags(s->stage_barrier);
    destroy_forward_notify_table(s->forward_notify);

    // Destroy extra QPs (indices 1..num_qps-1, in reverse order)
    for (int i = s->num_qps - 1; i >= 1; i--) {
        rdma::destroy_qp(s->extra_qps[i - 1]);
    }

    // Destroy per-proxy shared CQs.
    for (int t = s->num_proxy_threads - 1; t >= 1; t--) {
        rdma::destroy_cq(s->proxy_cqs[t]);
    }

    // Destroy primary RDMA resources (QP 0 / CQ 0)
    rdma::destroy_qp(s->qp);
    rdma::destroy_cq(s->cq);
    rdma::dealloc_pd(s->pd);
    rdma::close_device(s->ctx);

    delete s;
}

// ---------------------------------------------------------------------------
// Helpers for kernel globals
// ---------------------------------------------------------------------------

/** Get the device-side FIFO handle to pass into kernel globals. */
inline D2HFifoDeviceBundle get_fifo_device_handle(const Session* s) {
    return s->fifo_bundle;
}

/** Get the device-accessible arrival flags pointer for kernel globals. */
inline uint32_t* get_arrival_device_ptr(const Session* s) {
    return s->arrival.device_ptr;
}

inline uint32_t* get_arrival_tail_device_ptr(const Session* s) {
    return s ? s->arrival.tail_device_ptr : nullptr;
}

/** Get the receive buffer device pointer. */
inline void* get_recv_buf_ptr(const Session* s) {
    return s->recv_buf.gpu_ptr;
}

inline int get_num_qps(const Session* s) {
    return s ? s->num_qps : 1;
}

inline uint32_t* get_stage_barrier_device_ptr(const Session* s) {
    return s ? s->stage_barrier.device_ptr : nullptr;
}

/** Get the device-side ready queue handle (stub — returns empty handle). */
inline ReadyQueueDevice get_ready_queue_device(const Session* /*s*/) {
    return ReadyQueueDevice{};
}

/** Set the total expected items for the ready queue (stub — no-op). */
inline void set_ready_queue_total(Session* /*s*/, uint32_t /*total*/) {}

/**
 * Phase 1 of the epoch transition: wait for GPU work and proxy RDMA writes
 * to drain, then pause the proxy so no new FIFO commands are consumed.
 *
 * This allows the caller to perform a cross-rank barrier while both nodes are
 * fully quiesced before either one resets arrival flags or FIFO state.
 */
inline void prepare_epoch(Session* s) {
    // Wait for GPU to finish any pending work — no new FIFO pushes after this.
    cudaDeviceSynchronize();

    // Wait for proxy to consume all FIFO commands and drain all RDMA writes.
    drain_proxy(s);

    // Pause proxy so it stops polling the FIFO during reset.
    // The proxy finishes its current loop iteration before acknowledging.
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->pause();
    }

    // Drain any CQ completions for WRs the proxy posted in its final
    // loop iteration (between the drain check and pause acknowledgment).
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->drain_cq();
    }

    // Reset proxy inflight counter (guaranteed 0 after drain_cq)
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->reset_inflight();
    }

    // Reset latency timestamps for the next epoch while the proxy is paused.
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->reset_timestamps();
    }
}

/**
 * Phase 2 of the epoch transition: update epoch metadata, reset arrival flags
 * and FIFO state, then resume the proxy after the reset is visible.
 *
 * Caller must invoke prepare_epoch() first and typically synchronize ranks
 * between the two phases.
 */
inline void commit_epoch(Session* s, uint32_t epoch) {
    // Update epoch on session and proxy
    s->epoch = epoch;
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->set_epoch(epoch);
    }

    // Reset arrival flags.
    // NOTE: we intentionally do NOT reset stage_barrier here. The slot is
    // monotonically written (proxy's BARRIER_NOTIFY handler writes cfg_.epoch;
    // host-side stage_barrier() writes monotonically-increasing stage tokens).
    // Resetting it races with in-flight RDMA writes from a peer that has already
    // advanced to the new epoch — the reset can clobber an arriving token and
    // cause a lost-signal deadlock in steady-state benchmarking, because the
    // sender only issues one BARRIER_NOTIFY per iter and never retransmits.
    // Kernel waiters use >= comparison against the current epoch, so stale
    // larger values are harmless (they satisfy the condition immediately, which
    // is correct — the peer is ahead).
    //
    // reset_arrival_flags: also removed from the default path for the same
    // clobber reason — by the time this host memset fires, the peer (which
    // finished its iter boundary earlier) may have already RDMA-written
    // next-iter flag values into our host-pinned arrival array; memset here
    // would zero them and deadlock our next-iter kernel which spins on
    // `flag_val != 0`. Callers that need a reset must do it stream-ordered
    // on the GPU at iter-end BEFORE the cross-node barrier, or call reset_arrival_flags()
    // explicitly in a regime where no in-flight peer writes can race.
    if (std::getenv("MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET") == nullptr) {
        reset_arrival_flags(s->arrival);
    }
    reset_forward_notify_table(s->forward_notify);

    // Reset FIFO: zero head (device mem), zero tail (host mem), clear trigger slots
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaMemset(s->fifos[t].device.head, 0, sizeof(uint64_t));
        cudaMemset(s->fifos[t].device.tail_cache, 0, sizeof(uint64_t));
        *s->fifos[t].host.tail = 0;
        s->fifos[t].host.cpu_head = 0;
        memset(s->fifos[t].host.triggers, 0,
               s->fifos[t].host.capacity * sizeof(TransferCmd));
    }

    cudaDeviceSynchronize();

    // Resume proxy — it will start polling the now-clean FIFO
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
}

/**
 * Host-side stage barrier backed by a 4-byte proxy-issued RDMA write.
 *
 * Sequence at launch boundaries:
 * 1. wait for local kernels and RDMA traffic to quiesce
 * 2. pause the proxy so we can safely use the same QP from the host thread
 * 3. send our stage token to the peer's barrier slot
 * 4. wait until the peer sends the same token to our local barrier slot
 * 5. resume the proxy for the next stage
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

/**
 * Legacy single-call epoch transition kept for existing callers.
 */
inline void set_epoch(Session* s, uint32_t epoch) {
    prepare_epoch(s);
    commit_epoch(s, epoch);
}

/** Get proxy first-message latency timestamps for diagnostics. */
inline ProxyTimestamps get_proxy_timestamps(const Session* s) {
    return s->proxies[0]->get_timestamps();
}

inline std::vector<ProxyDiagnostics> get_proxy_diagnostics(const Session* s) {
    std::vector<ProxyDiagnostics> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->proxies[t] == nullptr) continue;
        out.push_back(s->proxies[t]->get_diagnostics());
    }
    return out;
}

inline std::vector<ProxyTimeline> get_proxy_timelines(const Session* s) {
    std::vector<ProxyTimeline> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->proxies[t] == nullptr) continue;
        out.push_back(s->proxies[t]->get_timeline());
    }
    return out;
}

} // namespace internode
