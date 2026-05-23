#pragma once

// Session management + pybind module for ag_gemm_multinode.
// Included from src/ag_gemm.cu after the kernel namespace closes.
// ============================================================================
// Session management
// ============================================================================
#include "comm/internode/session_py.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>

static internode::Session* g_session = nullptr;

// Stable storage for the multi-peer string/int arrays that SessionConfig
// references (it stores raw pointers, not values). Lifetime extends until
// the next create_session_py call replaces the session.
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t clocal_buf_ptr = 0,
                       int64_t clocal_buf_size = 0,
                       std::vector<std::string> peer_ips = {},
                       std::vector<int> peer_tcp_ports = {}) {
    internode::py::destroy_session(g_session);
    internode::SessionConfig cfg = internode::py::make_base_config(
        rank, peer_ip.c_str(), tcp_port,
        send_buf_ptr, send_buf_size, recv_buf_size,
        num_tiles, fifo_capacity, device_id);
    internode::py::apply_peer_ips(
        cfg, peer_ips, peer_tcp_ports, tcp_port,
        g_peer_ips_storage, g_peer_ips_cstr, g_peer_ports_storage);
    const char* perf_preset = std::getenv("AG_GEMM_PERF_PRESET");
    const bool perf_legacy =
        perf_preset != nullptr && std::strcmp(perf_preset, "legacy") == 0;
    cfg.max_inflight = 256;
    if (const char* e = std::getenv("MKERNEL_MAX_INFLIGHT")) {
        int v = std::atoi(e);
        if (v > 0) {
            cfg.max_inflight = v;
        }
    }
    if (clocal_buf_ptr == 0 || clocal_buf_size == 0) {
        fprintf(stderr, "create_session_py: AG1 direct sends require clocal_buf_ptr/size\n");
        std::exit(EXIT_FAILURE);
    }
    cfg.clocal_gpu_buf = reinterpret_cast<void*>(clocal_buf_ptr);
    cfg.clocal_gpu_buf_size = (size_t)clocal_buf_size;
    cfg.direct_dmabuf_enabled = true;
    cfg.row_stride_bytes = 0;
    if (const char* env_stride = std::getenv("AG_GEMM_ROW_STRIDE_BYTES")) {
        long long stride = std::atoll(env_stride);
        if (stride > 0) {
            cfg.row_stride_bytes = (size_t)stride;
        }
    }
    // Ring AG forwards received shards from the same A_recv buffer that is
    // used as the RDMA landing zone. Keeping receive and forward source in
    // one registered MR avoids a session-private recv buffer -> multicast
    // buffer handoff on the ring critical path. Direct fanout still uses the
    // session-owned receive buffer because its send buffer is the local A.
    if (peer_ips.size() > 1 && send_buf_size == recv_buf_size) {
        cfg.external_recv_buf = reinterpret_cast<void*>(send_buf_ptr);
    }
    // Early-send posts per-row WRs in a burst, so use multiple QPs by default.
    // MKERNEL_EFA_NUM_QPS can still override this.
    cfg.num_qps = 16;
    if (cfg.num_peers > 1 && std::getenv("MKERNEL_CHANNELIZE_GPU_PEERS") != nullptr) {
        cfg.num_qps = std::min(internode::kMaxQPs, cfg.num_peers * 8);
        cfg.channelize_gpu_peers = true;
    }
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        cfg.num_qps = cfg.channelize_gpu_peers
            ? std::max(cfg.num_qps, std::atoi(env_num_qps))
            : std::atoi(env_num_qps);
    }
    cfg.logical_queues_per_qp = 1;
    cfg.enable_forward_notify =
        std::getenv("AG_GEMM_RING_PROXY_FORWARD") != nullptr &&
        std::getenv("AG_GEMM_RING_PROXY_FORWARD")[0] == '1';
    if (const char* env_logical = std::getenv("MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::max(1, std::atoi(env_logical));
    } else if (const char* env_logical = std::getenv("AG_GEMM_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::max(1, std::atoi(env_logical));
    } else if (!perf_legacy && cfg.num_peers > 1) {
        // gemm_ar-style software striping on arrivals (see GEMM_AR_LOGICAL_QUEUES_PER_QP);
        // opt out with MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP=1 or legacy preset.
        cfg.logical_queues_per_qp = 2;
    }
    cfg.num_proxy_threads = 1;
    if (const char* env_proxy = std::getenv("MKERNEL_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    } else if (const char* env_proxy = std::getenv("AG_GEMM_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    }
    // Throughput-oriented defaults for true multi-node (N>2 peers): more QPs +
    // more D2H FIFO / proxy threads reduce host-side serialization. Opt out
    // with AG_GEMM_PERF_PRESET=legacy for A/B comparisons.
    if (!perf_legacy && cfg.num_peers > 1) {
        if (std::getenv("MKERNEL_EFA_NUM_QPS") == nullptr &&
            std::getenv("MKERNEL_INTERNODE_NUM_QPS") == nullptr &&
            std::getenv("MKERNEL_IB_NUM_QPS") == nullptr) {
            cfg.num_qps = std::max(cfg.num_qps, std::min(24, internode::kMaxQPs));
        }
        if (std::getenv("MKERNEL_PROXY_THREADS") == nullptr &&
            std::getenv("AG_GEMM_PROXY_THREADS") == nullptr) {
            cfg.num_proxy_threads = std::max(cfg.num_proxy_threads, 4);
        }
        if (cfg.num_proxy_threads > cfg.num_qps) {
            cfg.num_proxy_threads = cfg.num_qps;
        }
    }
    g_session = internode::create_session(cfg);
}

void destroy_session_py() {
    internode::py::destroy_session(g_session);
}

void set_epoch_py(int epoch) {
    internode::py::set_epoch(g_session, epoch);
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int> get_fifo_handles_py() {
    return internode::py::get_fifo_handles(g_session);
}

int64_t get_arrival_flags_ptr_py() {
    return internode::py::get_arrival_flags_ptr(g_session);
}

int64_t get_recv_buf_ptr_py() {
    return internode::py::get_recv_buf_ptr(g_session);
}

#include <torch/csrc/utils/pybind.h>

// Per-proxy diagnostic counters — mirrors the binding in
// include/operators/gemm_ar/session.cuh so the V2 Planner can read
// post→CQE per-WR latency directly from Python.
pybind11::list get_proxy_diagnostics_py() {
    pybind11::list out;
    for (const auto& diag : internode::get_proxy_diagnostics(g_session)) {
        pybind11::dict d;
        d["qp_base_idx"] = diag.qp_base_idx;
        d["num_qps"] = diag.num_qps;
        d["loops"] = diag.loops;
        d["empty_loops"] = diag.empty_loops;
        d["full_batches"] = diag.full_batches;
        d["partial_batches"] = diag.partial_batches;
        d["inflight_limited_loops"] = diag.inflight_limited_loops;
        d["batches_posted"] = diag.batches_posted;
        d["cmds_posted"] = diag.cmds_posted;
        d["cmds_completed"] = diag.cmds_completed;
        d["backlog_samples"] = diag.backlog_samples;
        d["backlog_nonzero_loops"] = diag.backlog_nonzero_loops;
        d["backlog_gt_batch_loops"] = diag.backlog_gt_batch_loops;
        d["backlog_sum"] = diag.backlog_sum;
        d["backlog_max"] = diag.backlog_max;
        d["batch_cmd_to_post_count"] = diag.batch_cmd_to_post_count;
        d["batch_cmd_to_post_sum_ns"] = diag.batch_cmd_to_post_sum_ns;
        d["batch_cmd_to_post_max_ns"] = diag.batch_cmd_to_post_max_ns;
        d["batch_completion_count"] = diag.batch_completion_count;
        d["batch_completion_sum_ns"] = diag.batch_completion_sum_ns;
        d["batch_completion_max_ns"] = diag.batch_completion_max_ns;
        d["enqueue_to_seen_count"] = diag.enqueue_to_seen_count;
        d["enqueue_to_seen_raw_sum_ns"] = diag.enqueue_to_seen_raw_sum_ns;
        d["enqueue_to_seen_raw_max_ns"] = diag.enqueue_to_seen_raw_max_ns;
        d["enqueue_to_seen_raw_min_ns"] = diag.enqueue_to_seen_raw_min_ns;
        d["loop_gap_max_ns"] = diag.loop_gap_max_ns;
        d["loop_gap_over_100us"] = diag.loop_gap_over_100us;
        d["loop_gap_over_1ms"] = diag.loop_gap_over_1ms;
        d["first_cmd_ns"] = diag.first_cmd_ns;
        d["last_cmd_ns"] = diag.last_cmd_ns;
        d["first_post_ns"] = diag.first_post_ns;
        d["last_post_ns"] = diag.last_post_ns;
        d["first_completion_ns"] = diag.first_completion_ns;
        d["last_completion_ns"] = diag.last_completion_ns;
        out.append(d);
    }
    return out;
}

pybind11::list get_proxy_timelines_py() {
    pybind11::list out;
    for (const auto& timeline : internode::get_proxy_timelines(g_session)) {
        pybind11::dict d;
        pybind11::list posted;
        for (uint64_t ts : timeline.signaled_post_ns) posted.append(ts);
        pybind11::list completed;
        for (uint64_t ts : timeline.completion_ns) completed.append(ts);
        d["signaled_post_ns"] = posted;
        d["completion_ns"] = completed;
        out.append(d);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("clocal_buf_ptr") = 0,
          pybind11::arg("clocal_buf_size") = 0,
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("get_proxy_diagnostics", &get_proxy_diagnostics_py);
    m.def("get_proxy_timelines", &get_proxy_timelines_py);
    m.def("ag_gemm_multinode", &ag_gemm_multinode::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("recv_buf_ptr"),
          pybind11::arg("fifo_triggers"),
          pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"),
          pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_comm_sms"),
          pybind11::arg("a_half_bytes"),
          pybind11::arg("A_recv"),
          pybind11::arg("active_sms") = 132,
          pybind11::arg("num_intra_comm_override") = 0,
          pybind11::arg("num_nodes") = 2);
}
