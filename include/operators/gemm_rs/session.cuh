#pragma once

// Session management + pybind module for gemm_rs_multinode.
// Included from src/gemm_rs.cu after the kernel namespace closes.

// ============================================================================
// Session management (same as v1)
// ============================================================================
#include "comm/internode/session_py.cuh"

static internode::Session* g_session = nullptr;
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t output_buf_ptr = 0, int64_t output_buf_size = 0,
                       int output_n = 0,
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

    // GEMM_RS_DIRECT_DMABUF_SEND: register output_local as a DMA-BUF MR on every
    // rail's PD so the send path can skip the pack + gather step. Session
    // creation hard-fails if the buffer can't be DMA-BUF-exported (no peermem
    // fallback — we want the direct path or nothing).
    cfg.max_inflight = 256;
    // Round 26: default to 4 QPs per endpoint to match the gemm_ar round-13 win
    // (51e19e8). Multi-NIC striping is also enabled by default via the
    // round-18 cluster default MKERNEL_EFA_NUM_NICS=2 in session_fi.h.
    cfg.num_qps = 4;
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    // gemm_rs proxy-thread count: default 1 (bumped to num_rails=2 inside session
    // setup). Multi-proxy spreads send-cmd dispatch across host threads —
    // matters when the per-proxy dispatch rate approaches NIC latency floor
    // (~2 µs). Overridable with MKERNEL_PROXY_THREADS (gemm_ar uses the same env).
    cfg.num_proxy_threads = 1;
    if (const char* env_proxy = std::getenv("MKERNEL_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    } else if (const char* env_proxy = std::getenv("GEMM_RS_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    }
    cfg.logical_queues_per_qp = 1;
    if (const char* env_lq = std::getenv("MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_lq);
    } else if (const char* env_lq = std::getenv("GEMM_RS_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_lq);
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
int64_t get_arrival_flags_ptr_py() { return internode::py::get_arrival_flags_ptr(g_session); }
int64_t get_arrival_flags_total_words_py() {
    return internode::py::get_arrival_flags_total_words(g_session);
}
int64_t get_recv_buf_ptr_py() { return internode::py::get_recv_buf_ptr(g_session); }

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"),
          pybind11::arg("peer_ip"),
          pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"),
          pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"),
          pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("device_id"),
          pybind11::arg("output_buf_ptr") = 0,
          pybind11::arg("output_buf_size") = 0,
          pybind11::arg("output_n") = 0,
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_arrival_flags_total_words", &get_arrival_flags_total_words_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("gemm_rs_fused", &gemm_rs_multinode::entrypoint_fused,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("workspace"),
          pybind11::arg("output"),
          pybind11::arg("barrier"),
          pybind11::arg("ready"),
          pybind11::arg("recv_buf_ptr"),
          pybind11::arg("staging_buf_ptr"),
          pybind11::arg("fifo_triggers"),
          pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"),
          pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_comp_sms"),
          pybind11::arg("num_intra_comm"),
          pybind11::arg("num_send_sms"),
          pybind11::arg("num_reduce_sms"),
          pybind11::arg("use_acquire_poll") = (int64_t)0,
          pybind11::arg("reduce_poll_sleep_ns") = (int64_t)100,
          pybind11::arg("ready_chunk"),
          pybind11::arg("staging") = pybind11::none(),
          pybind11::arg("num_nodes") = 2,
          pybind11::arg("arrival_total_words") = 0);
}
