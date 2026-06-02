#pragma once

// Session management + pybind module for ring_attention.
// Included from src/ring_attention.cu after the kernel namespace closes.
// ============================================================================
// Session management + pybind module
// ============================================================================
#include "comm/internode/session_py.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>

static internode::Session* g_session = nullptr;
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t k0_buf_ptr,
                       int64_t k0_buf_size,
                       int64_t v0_buf_ptr,
                       int64_t v0_buf_size,
                       bool staging_source,
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
    // Default zero-copy send: K0 (src_view=0) and V0 (src_view=1) are direct
    // GPU MRs. Peermem-compatible mode keeps K/V as VMM ring buffers and sends
    // from a cudaMalloc staging buffer instead, because the intra-node TMA ring
    // relies on the VMM/IPC peer views.
    if (!staging_source && (k0_buf_ptr == 0 || v0_buf_ptr == 0)) {
        fprintf(stderr, "create_session_py: ring_attention zero-copy send requires k0/v0 ptrs\n");
        std::exit(EXIT_FAILURE);
    }
    if (staging_source) {
        if (send_buf_ptr == 0 || send_buf_size == 0) {
            fprintf(stderr, "create_session_py: ring_attention staging source requires send_buf ptr+size\n");
            std::exit(EXIT_FAILURE);
        }
        cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
        cfg.local_gpu_buf_size = (size_t)send_buf_size;
        cfg.direct_dmabuf_enabled = false;
    } else {
        (void)send_buf_ptr; (void)send_buf_size;
        cfg.local_gpu_buf = reinterpret_cast<void*>(k0_buf_ptr);
        cfg.local_gpu_buf_size = (size_t)k0_buf_size;
        cfg.clocal_gpu_buf = reinterpret_cast<void*>(v0_buf_ptr);
        cfg.clocal_gpu_buf_size = (size_t)v0_buf_size;
        cfg.direct_dmabuf_enabled = true;
    }
    cfg.max_inflight = 32;
    if (const char* env_mi = std::getenv("MKERNEL_MAX_INFLIGHT")) {
        cfg.max_inflight = std::atoi(env_mi);
    }
    // Default to 4 QPs per endpoint; multi-NIC striping is controlled by the
    // internode session configuration.
    cfg.num_qps = 4;
    if (cfg.num_peers > 1 && std::getenv("MKERNEL_CHANNELIZE_GPU_PEERS") != nullptr) {
        cfg.num_qps = std::min(
            internode::kMaxQPs,
            cfg.num_peers * ring_attn_multinode::globals::NUM_DEVICES);
        cfg.channelize_gpu_peers = true;
    }
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    g_session = internode::create_session(cfg);
}
void destroy_session_py() { internode::py::destroy_session(g_session); }
void set_epoch_py(int epoch) { internode::py::set_epoch(g_session, epoch); }
std::tuple<int64_t, int64_t, int64_t, int64_t, int> get_fifo_handles_py() {
    return internode::py::get_fifo_handles(g_session);
}
int64_t get_arrival_flags_ptr_py() { return internode::py::get_arrival_flags_ptr(g_session); }
int64_t get_recv_buf_ptr_py() { return internode::py::get_recv_buf_ptr(g_session); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("k0_buf_ptr"), pybind11::arg("k0_buf_size"),
          pybind11::arg("v0_buf_ptr"), pybind11::arg("v0_buf_size"),
          pybind11::arg("staging_source"),
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("ring_attn_multinode", &ring_attn_multinode::entrypoint,
          pybind11::arg("Q"),
          pybind11::arg("K0"),
          pybind11::arg("K1"),
          pybind11::arg("V0"),
          pybind11::arg("V1"),
          pybind11::arg("L"),
          pybind11::arg("L_block"),
          pybind11::arg("O"),
          pybind11::arg("O_block"),
          pybind11::arg("barrier"),
          pybind11::arg("send_buf_ptr"),
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
          pybind11::arg("num_send_sms"),
          pybind11::arg("num_copy_sms"),
          pybind11::arg("num_nodes"),
          pybind11::arg("staging_source"));
}
