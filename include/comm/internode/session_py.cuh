#pragma once

#include "session_select.h"
#include "../device_clock.cuh"
#include "../../common/cuda_checks.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include <torch/csrc/utils/pybind.h>

namespace internode::py {

using FifoHandleTuple = std::tuple<int64_t, int64_t, int64_t, int64_t, int>;

inline SessionConfig make_base_config(
    int rank,
    const char* peer_ip,
    int tcp_port,
    int64_t send_buf_ptr,
    int64_t send_buf_size,
    int64_t recv_buf_size,
    int num_tiles,
    int fifo_capacity,
    int device_id
) {
    SessionConfig cfg{};
    cfg.rank = rank;
    cfg.peer_ip = peer_ip;
    cfg.tcp_port = tcp_port;
    cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
    cfg.local_gpu_buf_size = static_cast<size_t>(send_buf_size);
    cfg.recv_buf_size = static_cast<size_t>(recv_buf_size);
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;
    return cfg;
}

// Attach a Python-supplied list of peer IPs (and optional matching ports) to
// the SessionConfig's multi-peer fields. The C arrays in cfg point at the
// caller-supplied vector storage; the caller must keep that storage alive
// until the session is destroyed (typically file-scope statics in each
// operator shim). When peer_ips is empty, leaves cfg unchanged so the
// single-peer shorthand (cfg.peer_ip / cfg.tcp_port) is used.
inline void apply_peer_ips(
    SessionConfig& cfg,
    std::vector<std::string>& peer_ips,
    std::vector<int>& peer_tcp_ports,
    int default_tcp_port,
    std::vector<std::string>& storage_ips,
    std::vector<const char*>& storage_cstr,
    std::vector<int>& storage_ports
) {
    if (peer_ips.empty()) return;
    storage_ips   = std::move(peer_ips);
    storage_ports = std::move(peer_tcp_ports);
    if (storage_ports.empty()) {
        storage_ports.assign(storage_ips.size(), default_tcp_port);
    }
    storage_cstr.resize(storage_ips.size());
    for (size_t i = 0; i < storage_ips.size(); ++i) {
        storage_cstr[i] = storage_ips[i].c_str();
    }
    cfg.num_peers      = static_cast<int>(storage_ips.size());
    cfg.peer_ips       = storage_cstr.data();
    cfg.peer_tcp_ports = storage_ports.data();
}

inline void destroy_session(Session*& session) {
    if (session) {
        internode::destroy_session(session);
        session = nullptr;
    }
}

inline void set_epoch(Session* session, int epoch) {
    if (session) internode::set_epoch(session, static_cast<uint32_t>(epoch));
}

inline FifoHandleTuple get_fifo_handles(Session* session) {
    if (!session) {
        return {0, 0, 0, 0, 0};
    }
    // Always pass the session's D2HFifoDeviceBundle by pointer (negative
    // capacity sentinel). Kernels used to reconstruct a bundle from raw FIFO
    // pointers + MKERNEL_EFA_NUM_QPS default (16), which could disagree with the
    // session's actual num_qps / logical_queues when num_fifos==1 (gemm_ar
    // passes explicit num_qps into resolve_fifo_bundle; ag_gemm did not).
    const int nf = session->fifo_bundle.num_fifos > 0 ? session->fifo_bundle.num_fifos : 1;
    return {
        reinterpret_cast<int64_t>(&session->fifo_bundle),
        0,
        0,
        0,
        -nf
    };
}

inline int64_t get_arrival_flags_ptr(Session* session) {
    return reinterpret_cast<int64_t>(internode::get_arrival_device_ptr(session));
}

inline int64_t get_recv_buf_ptr(Session* session) {
    return reinterpret_cast<int64_t>(internode::get_recv_buf_ptr(session));
}

inline uint64_t host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

static __global__ void read_globaltimer_kernel(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = comm::globaltimer();
    }
}

inline pybind11::dict get_globaltimer_calibration(int samples = 64) {
    if (samples <= 0) samples = 1;
    unsigned long long* dev_out = nullptr;
    unsigned long long host_out = 0;
    cudaStream_t stream = nullptr;
    MKERNEL_CUDACHECK(cudaMalloc(&dev_out, sizeof(unsigned long long)));
    MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t best_host_before_ns = 0;
    uint64_t best_host_after_ns = 0;
    uint64_t best_host_mid_ns = 0;
    uint64_t best_gpu_ns = 0;
    uint64_t best_window_ns = std::numeric_limits<uint64_t>::max();

    for (int i = 0; i < samples; ++i) {
        const uint64_t host_before_ns = host_now_ns();
        read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        MKERNEL_CUDACHECK(cudaGetLastError());
        MKERNEL_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        MKERNEL_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = host_now_ns();
        const uint64_t window_ns = host_after_ns - host_before_ns;
        if (window_ns < best_window_ns) {
            best_window_ns = window_ns;
            best_host_before_ns = host_before_ns;
            best_host_after_ns = host_after_ns;
            best_host_mid_ns = host_before_ns + window_ns / 2ULL;
            best_gpu_ns = static_cast<uint64_t>(host_out);
        }
    }

    MKERNEL_CUDACHECK(cudaStreamDestroy(stream));
    MKERNEL_CUDACHECK(cudaFree(dev_out));

    pybind11::dict d;
    d["host_before_ns"] = best_host_before_ns;
    d["host_after_ns"] = best_host_after_ns;
    d["host_mid_ns"] = best_host_mid_ns;
    d["gpu_ns"] = best_gpu_ns;
    d["window_ns"] = best_window_ns;
    d["offset_ns"] = static_cast<int64_t>(best_host_mid_ns) - static_cast<int64_t>(best_gpu_ns);
    return d;
}

} // namespace internode::py
