#pragma once

// Session management + pybind module for dispatch_gemm.
// Included from src/dispatch_gemm.cu after the kernel namespace closes.

// ============================================================================
// Session management
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
                       int64_t external_recv_buf_ptr,
                       int64_t pre_tokens_buf_ptr,
                       int64_t pre_tokens_buf_size,
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
    // Zero-copy send: register pre_tokens (DistBuffer, VMM) as the
    // session's only data MR via register_gpu_buffer(), which transparently
    // uses cuMemGetHandleForAddressRange + ibv_reg_dmabuf_mr on VMM ranges.
    // The kernel emits cmd.src_view=0; only the source address changes.
    // send_buf_ptr/size are kept in the signature for compatibility but unused.
    if (pre_tokens_buf_ptr == 0 || pre_tokens_buf_size == 0) {
        fprintf(stderr, "create_session_py: dispatch_gemm zero-copy requires pre_tokens_buf ptr+size\n");
        std::exit(EXIT_FAILURE);
    }
    (void)send_buf_ptr; (void)send_buf_size;
    cfg.local_gpu_buf = reinterpret_cast<void*>(pre_tokens_buf_ptr);
    cfg.local_gpu_buf_size = (size_t)pre_tokens_buf_size;
    cfg.recv_buf_size = (size_t)recv_buf_size;
    // Zero-copy inter-recv: when external_recv_buf_ptr != 0, the session
    // registers caller-owned peer_tokens as the RDMA destination, replacing
    // the staged recv_buf. fused_inter_copy_sm under DISPATCH_ZERO_COPY
    // then drops the D2D copy entirely.
    if (external_recv_buf_ptr != 0) {
        cfg.external_recv_buf = reinterpret_cast<void*>(external_recv_buf_ptr);
    }
    // Tier 1.2 (DeepEP-style sliding window): cap inflight WRs to keep queue
    // pressure low. Sweep on EFA: 32 ≈ 64 < 256 < 1024 at 131k tokens.
    // Benefit is small (~1-2%) on top of CHUNK_BYTES=512 KB but consistent.
    cfg.max_inflight = 32;
    if (const char* env_mi = std::getenv("MKERNEL_MAX_INFLIGHT")) {
        cfg.max_inflight = std::atoi(env_mi);
    }
    // Multiple QPs let the proxy post WRs in parallel.
    cfg.num_qps = 4;
    const char* channelize_env = std::getenv("MKERNEL_CHANNELIZE_GPU_PEERS");
    if (cfg.num_peers > 1 && channelize_env != nullptr && std::atoi(channelize_env) != 0) {
        cfg.num_qps = std::min(internode::kMaxQPs, cfg.num_peers * 8);
        cfg.channelize_gpu_peers = true;
    }
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        cfg.num_qps = cfg.channelize_gpu_peers
            ? std::max(cfg.num_qps, std::atoi(env_num_qps))
            : std::atoi(env_num_qps);
    }
    // Allow override of CPU proxy thread count (default = max(1,num_rails)=2).
    // More threads parallelize ibv_post_send across QP slices.
    if (const char* env_np = std::getenv("MKERNEL_NUM_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_np);
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

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("external_recv_buf_ptr"),
          pybind11::arg("pre_tokens_buf_ptr"),
          pybind11::arg("pre_tokens_buf_size"),
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("moe_dispatch_gemm_glu_combine_fused", &moe_dispatch_gemm_glu_combine_multinode::fused,
          pybind11::arg("pre_tokens"),
          pybind11::arg("peer_tokens"),
          pybind11::arg("copy_ready"),
          pybind11::arg("post_tokens"),
          pybind11::arg("w1"),
          pybind11::arg("w2"),
          pybind11::arg("h1"),
          pybind11::arg("act"),
          pybind11::arg("y_expert"),
          pybind11::arg("row_expert"),
          pybind11::arg("topk_weights"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("local_rb_per_expert"),
          pybind11::arg("barrier"),
          pybind11::arg("sync_barrier"),
          pybind11::arg("y_out"),
          pybind11::arg("comb_buf"),
          pybind11::arg("owner_offset"),
          pybind11::arg("row_to_slot"),
          pybind11::arg("row_to_owner"),
          pybind11::arg("inter_comb_buf"),
          pybind11::arg("inter_owner_offset"),
          pybind11::arg("inter_row_to_slot"),
          pybind11::arg("inter_row_to_owner"),
          pybind11::arg("use_inter_gather"),
          pybind11::arg("recv_buf_ptr"),
          pybind11::arg("fifo_triggers"),
          pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"),
          pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_local_tokens"),
          pybind11::arg("num_padded_local_tokens"),
          pybind11::arg("num_send_sms"),
          pybind11::arg("num_copy_sms"),
          pybind11::arg("num_comm_sms_intra"),
          pybind11::arg("use_gather"),
          pybind11::arg("num_nodes"));
}
