/**
 * @file rdma_transport.h
 * @brief libibverbs wrappers for inter-node RDMA communication.
 *
 * Pure C++ host code — no CUDA dependency. Link with -libverbs.
 * Provides RC QP setup, memory registration, RDMA write posting, CQ polling,
 * and TCP bootstrap for connection info exchange.
 */
#pragma once

#include "types.h"

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

namespace internode {
namespace rdma {

inline int roce_gid_index() {
    const char* env = std::getenv("MKERNEL_ROCE_GID_INDEX");
    if (!env || !env[0]) return 0;
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env || v < 0 || v > 255) {
        fprintf(stderr, "rdma: ignoring invalid MKERNEL_ROCE_GID_INDEX=%s\n", env);
        return 0;
    }
    return static_cast<int>(v);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define RDMA_CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "RDMA error at %s:%d: %s (errno=%d: %s)\n", \
                __FILE__, __LINE__, msg, errno, strerror(errno)); \
        throw std::runtime_error(std::string("RDMA: ") + msg); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Device & Protection Domain
// ---------------------------------------------------------------------------

/** Open IB device by name (default "mlx5_0"). */
inline ibv_context* open_device(const char* name = "mlx5_0") {
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    RDMA_CHECK(dev_list && num_devices > 0, "no IB devices found");

    ibv_device* target = nullptr;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), name) == 0) {
            target = dev_list[i];
            break;
        }
    }
    RDMA_CHECK(target != nullptr, ("device not found: " + std::string(name)).c_str());

    ibv_context* ctx = ibv_open_device(target);
    ibv_free_device_list(dev_list);
    RDMA_CHECK(ctx != nullptr, "ibv_open_device failed");
    return ctx;
}

inline ibv_pd* alloc_pd(ibv_context* ctx) {
    ibv_pd* pd = ibv_alloc_pd(ctx);
    RDMA_CHECK(pd != nullptr, "ibv_alloc_pd failed");
    return pd;
}

// ---------------------------------------------------------------------------
// Completion Queue
// ---------------------------------------------------------------------------

inline ibv_cq* create_cq(ibv_context* ctx, int cqe = 2048) {
    ibv_cq* cq = ibv_create_cq(ctx, cqe, nullptr, nullptr, 0);
    RDMA_CHECK(cq != nullptr, "ibv_create_cq failed");
    return cq;
}

/** Poll CQ. Returns number of completions (0 if empty, -1 on error). */
inline int poll_cq(ibv_cq* cq, int max_wc, ibv_wc* wc) {
    return ibv_poll_cq(cq, max_wc, wc);
}

// ---------------------------------------------------------------------------
// RC Queue Pair
// ---------------------------------------------------------------------------

inline ibv_qp* create_rc_qp(ibv_pd* pd, ibv_cq* send_cq, int sq_depth = 2048, int max_send_sge = 2) {
    ibv_qp_init_attr init_attr{};
    init_attr.send_cq = send_cq;
    init_attr.recv_cq = send_cq;  // not using recv, but QP requires it
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = sq_depth;
    init_attr.cap.max_recv_wr = 1;  // minimal recv queue
    init_attr.cap.max_send_sge = max_send_sge; // 2 default; raised to 32 for DMA-BUF strided gather
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 64; // inline small WRs (e.g. 4-byte flag)

    ibv_qp* qp = ibv_create_qp(pd, &init_attr);
    RDMA_CHECK(qp != nullptr, "ibv_create_qp(RC) failed");
    return qp;
}

/** Transition QP: RESET → INIT. */
inline void modify_qp_init(ibv_qp* qp, uint8_t port = 1) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ;
    int ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    RDMA_CHECK(ret == 0, "modify_qp_init failed");
}

/** Transition QP: INIT → RTR. */
inline void modify_qp_rtr(ibv_qp* qp, const ConnectionInfo& remote,
                           uint8_t port = 1) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = remote.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    // Address handle
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port;

    // Check if GID is non-zero (needed for RoCE / IB with GRH)
    bool has_gid = false;
    for (int i = 0; i < 16; i++) {
        if (remote.gid[i] != 0) { has_gid = true; break; }
    }
    if (has_gid) {
        attr.ah_attr.is_global = 1;
        memcpy(&attr.ah_attr.grh.dgid, remote.gid, 16);
        attr.ah_attr.grh.sgid_index = roce_gid_index();
        attr.ah_attr.grh.hop_limit = 64;
        attr.ah_attr.grh.traffic_class = 0;
    }

    int ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    RDMA_CHECK(ret == 0, "modify_qp_rtr failed");
}

/** Transition QP: RTR → RTS. */
inline void modify_qp_rts(ibv_qp* qp, uint32_t local_psn) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 1;

    int ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    RDMA_CHECK(ret == 0, "modify_qp_rts failed");
}

// ---------------------------------------------------------------------------
// Memory Registration
// ---------------------------------------------------------------------------

inline ibv_mr* reg_mr(ibv_pd* pd, void* addr, size_t len, int access) {
    ibv_mr* mr = ibv_reg_mr(pd, addr, len, access);
    RDMA_CHECK(mr != nullptr, "ibv_reg_mr failed");
    return mr;
}

inline void dereg_mr(ibv_mr* mr) {
    if (mr) ibv_dereg_mr(mr);
}

// ---------------------------------------------------------------------------
// RDMA Write Operations
// ---------------------------------------------------------------------------

/**
 * Post a single RDMA WRITE.
 * @param signaled If true, generates a CQE on completion.
 * @return 0 on success, errno on failure.
 */
inline int post_write(ibv_qp* qp,
                      uint64_t local_addr, uint32_t lkey,
                      uint64_t remote_addr, uint32_t rkey,
                      uint32_t length, uint64_t wr_id, bool signaled) {
    ibv_sge sge{};
    sge.addr = local_addr;
    sge.length = length;
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    wr.next = nullptr;

    ibv_send_wr* bad = nullptr;
    return ibv_post_send(qp, &wr, &bad);
}

/**
 * Post chained RDMA WRITEs: tile data + 4-byte arrival flag.
 * Two WRs posted atomically via one ibv_post_send. Only the flag WR is signaled.
 * RC ordering guarantees data is visible on remote before flag lands.
 *
 * @return 0 on success, errno on failure.
 */
inline int post_write_with_flag(ibv_qp* qp,
                                uint64_t data_laddr, uint32_t data_lkey,
                                uint64_t data_raddr, uint32_t data_rkey,
                                uint32_t data_len,
                                uint64_t flag_laddr, uint32_t flag_lkey,
                                uint64_t flag_raddr, uint32_t flag_rkey,
                                uint64_t wr_id) {
    // WR[0]: tile data (unsignaled)
    ibv_sge sge_data{};
    sge_data.addr = data_laddr;
    sge_data.length = data_len;
    sge_data.lkey = data_lkey;

    // WR[1]: 4-byte arrival flag (signaled)
    ibv_sge sge_flag{};
    sge_flag.addr = flag_laddr;
    sge_flag.length = sizeof(uint32_t);
    sge_flag.lkey = flag_lkey;

    ibv_send_wr wr_flag{};
    wr_flag.wr_id = wr_id;
    wr_flag.sg_list = &sge_flag;
    wr_flag.num_sge = 1;
    wr_flag.opcode = IBV_WR_RDMA_WRITE;
    wr_flag.send_flags = IBV_SEND_SIGNALED;
    wr_flag.wr.rdma.remote_addr = flag_raddr;
    wr_flag.wr.rdma.rkey = flag_rkey;
    wr_flag.next = nullptr;

    ibv_send_wr wr_data{};
    wr_data.wr_id = wr_id;
    wr_data.sg_list = &sge_data;
    wr_data.num_sge = 1;
    wr_data.opcode = IBV_WR_RDMA_WRITE;
    wr_data.send_flags = 0; // unsignaled
    wr_data.wr.rdma.remote_addr = data_raddr;
    wr_data.wr.rdma.rkey = data_rkey;
    wr_data.next = &wr_flag;  // chain: data first, then flag

    ibv_send_wr* bad = nullptr;
    return ibv_post_send(qp, &wr_data, &bad);
}

// ---------------------------------------------------------------------------
// TCP Bootstrap: exchange ConnectionInfo between peer nodes
// ---------------------------------------------------------------------------

namespace detail {

inline void send_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, p + sent, len - sent, 0);
        RDMA_CHECK(n > 0, "TCP send failed");
        sent += (size_t)n;
    }
}

inline void recv_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t recvd = 0;
    while (recvd < len) {
        ssize_t n = ::recv(fd, p + recvd, len - recvd, MSG_WAITALL);
        RDMA_CHECK(n > 0, "TCP recv failed");
        recvd += (size_t)n;
    }
}

} // namespace detail

/**
 * Exchange ConnectionInfo with peer over TCP.
 * One side is server (listens), the other is client (connects).
 *
 * @param local  Our ConnectionInfo to send.
 * @param peer_ip  Remote node IP address (e.g., "38.123.21.6").
 * @param port  TCP port to use.
 * @param is_server  If true, listen+accept. If false, connect.
 * @return Remote node's ConnectionInfo.
 */
inline ConnectionInfo exchange_info_tcp(const ConnectionInfo& local,
                                         const char* peer_ip, int port,
                                         bool is_server) {
    ConnectionInfo remote{};

    if (is_server) {
        // Server: listen, accept, exchange
        int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        RDMA_CHECK(listen_fd >= 0, "socket() failed");

        int reuse = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;
        RDMA_CHECK(bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) == 0,
                   "bind() failed");
        RDMA_CHECK(listen(listen_fd, 1) == 0, "listen() failed");

        int conn_fd = accept(listen_fd, nullptr, nullptr);
        RDMA_CHECK(conn_fd >= 0, "accept() failed");
        close(listen_fd);

        // Send ours, recv theirs
        detail::send_all(conn_fd, &local, sizeof(local));
        detail::recv_all(conn_fd, &remote, sizeof(remote));
        close(conn_fd);
    } else {
        // Client: connect with retry
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        RDMA_CHECK(fd >= 0, "socket() failed");

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);

        // Extension rebuilds can take well over 30s on one rank while the peer
        // has already finished and started retrying the TCP bootstrap connect.
        // Re-create the socket on each retry since failed connects can mark
        // the fd unusable on some kernels.
        const int max_retries = 1200;  // 1200 × 100ms = 120s
        bool connected = false;
        for (int attempt = 0; attempt < max_retries; attempt++) {
            if (connect(fd, (sockaddr*)&addr, sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            close(fd);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            fd = socket(AF_INET, SOCK_STREAM, 0);
            RDMA_CHECK(fd >= 0, "socket() retry failed");
        }
        if (!connected) {
            close(fd);
            RDMA_CHECK(false, "TCP connect failed after 120s of retries");
        }

        // Recv theirs, send ours (opposite order from server)
        detail::recv_all(fd, &remote, sizeof(remote));
        detail::send_all(fd, &local, sizeof(local));
        close(fd);
    }

    return remote;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/** Fill ConnectionInfo with local QP/port attributes. */
inline void fill_local_info(ConnectionInfo& info, ibv_qp* qp,
                            ibv_context* ctx, uint8_t port = 1) {
    info.qp_num = qp->qp_num;
    info.psn = 0; // deterministic for simplicity

    ibv_port_attr port_attr{};
    ibv_query_port(ctx, port, &port_attr);
    info.lid = port_attr.lid;

    ibv_gid gid{};
    ibv_query_gid(ctx, port, roce_gid_index(), &gid);
    memcpy(info.gid, &gid, 16);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

inline void destroy_qp(ibv_qp* qp) { if (qp) ibv_destroy_qp(qp); }
inline void destroy_cq(ibv_cq* cq) { if (cq) ibv_destroy_cq(cq); }
inline void dealloc_pd(ibv_pd* pd) { if (pd) ibv_dealloc_pd(pd); }
inline void close_device(ibv_context* ctx) { if (ctx) ibv_close_device(ctx); }

} // namespace rdma
} // namespace internode
