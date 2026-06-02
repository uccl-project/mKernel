/**
 * @file
 * @brief dist::distributed_tensor — intra-node + inter-node distributed tensor descriptor.
 *
 * Distributed multi-GPU tensor descriptor. Owns:
 *   - per-device `dist::local_tensor` array (intra-node fan-out)
 *   - optional multicast pointer + multicast TMA descriptor cache
 *   - per-channel inter-node state (channels[], arrived[], single-producer
 *     single-consumer send ring)
 *
 * No external inheritance or duck tag. TMA helpers accept it
 * structurally through `get_tma`, shape/stride accessors, and multicast
 * address access.
 *
 * Design contract:
 *   - One channel = one QP = one comm CTA. No CAS in fast path.
 *   - Inter-node sends are RDMA WRITE_WITH_IMM, imm = tile_id.
 *   - DMA-BUF zero-copy by default.
 *   - Send ring per channel is single-producer (the compute CTA that owns
 *     the tile) / single-consumer (the channel's comm CTA). No CAS on
 *     producer or consumer index — each side owns its own counter.
 *   - 128B-aligned arrival flags — no false sharing.
 *   - Monotonic flag semantics — no inter-iter reset.
 */

#pragma once

#include "local_tensor.cuh"
#include "../comm/atomic_u32.cuh"
#include "../comm/multimem.cuh"

#include <cstdint>
#include <utility>

namespace dist {

/* ----------   Inter-node primitive types  ---------- */

/// 128B-aligned monotonic arrival flag — no false sharing across tiles.
struct alignas(128) TileFlag {
    uint32_t v;
    uint32_t _pad[31];
};

/// Send mode — picks the proxy's WQE construction strategy.
///   DmabufDirect: src_view=1, zero-copy single-SGE from direct GPU MR.
///                 Historical name; MR may be DMA-BUF or peermem-backed.
///                 Requires session.rails[*].clocal_data_mr non-null.
///   Staging     : src_view=0, proxy reads from a caller-provided staging
///                 buffer (the kernel packs into staging before put_inter).
///                 Compatible with kernels that don't have DMA-BUF support.
///   StridedGather: src_view=2, zero-copy multi-SGE gather (advanced; used
///                 when source layout is row-major but destination is tile-major).
enum class SendMode : uint8_t {
    DmabufDirect  = 1,
    Staging       = 0,
    StridedGather = 2,
};

/// Single send-ring job entry: written by the producing compute CTA,
/// read by the channel's consuming comm CTA.
struct SendJob {
    uint32_t tile_id;
    uint32_t bytes;
    uint64_t off;          // byte offset into local DMABUF MR
    uint32_t imm;          // RDMA immediate (typically == tile_id)
    uint16_t dst_node;
    uint16_t _pad;
};

/// Per-channel state. One channel = one QP = one comm CTA.
template<int NUM_NODES>
struct Channel {
    /* ----- RDMA peer info (reserved for direct GPU networking paths; the
     *       current proxy backend relies on the embedded FIFO instead) ----- */
    uint64_t remote_addr[NUM_NODES];
    uint32_t rkey       [NUM_NODES];
    uint32_t qp_handle;
    uint32_t cq_handle;
    uint32_t lkey;

    /* ----- Proxy backend: full FIFO bundle pointer -----
     *   Each channel carries a pointer to the session's D2HFifoDeviceBundle
     *   so put_inter can route by lane_id (the proxy thread that owns the
     *   destination QP differs across lanes). `bind_inter_proxy` /
     *   `attach_channels_proxy` populates this; `put_inter` calls
     *   the backend lane-to-FIFO helper internally.
     *
     *   We store the bundle as `void*` to avoid pulling internode headers
     *   into this header; backend_proxy.cuh casts it back to the real type.
     */
    void*     fifo_bundle;

    /* ----- Arrival flags (host-pinned, RDMA-writable) -----
     *   The receiver's NIC writes here on each WRITE_WITH_IMM completion;
     *   compute CTAs spin on `arrived[t].v >= expected_iter`. Single writer
     *   (the NIC / proxy), so monotonic add is enough — no CAS.
     */
    TileFlag* arrived;
    uint32_t  tiles_per_channel;

    /* ----- Compute -> comm send ring (device memory) -----
     *   Single producer per channel: the compute CTA(s) that own this
     *   channel's tile slice are scheduled so only one writes here at a time
     *   (channel = tile_id % NUM_CHANNELS, and tiles map 1:1 to producer CTAs).
     *   Single consumer: this channel's comm CTA.
     *   Each side owns its own index counter — no CAS on head or tail.
     */
    SendJob*  ring;
    uint32_t  ring_capacity;     // power of two

    struct alignas(128) Index { uint32_t v; uint32_t _pad[31]; };
    Index*    head;              // producer (compute CTA)
    Index*    tail;              // consumer (comm CTA)

    /* ----- Send mode (set at attach/bind time) -----
     * Controls the src_view byte the proxy sees on every put_inter from this
     * channel. Zero-copy modes (DmabufDirect, StridedGather) require the
     * session to have a non-null direct GPU clocal_data_mr per rail; Staging
     * mode tolerates a session without direct GPU source registration.
     */
    SendMode  mode;
};

template<int NUM_CHANNELS, int NUM_NODES>
struct ChannelArray {
    Channel<NUM_NODES> data[NUM_CHANNELS];

    __host__ __device__ inline Channel<NUM_NODES>& operator[](int idx) {
        return data[idx];
    }
    __host__ __device__ inline const Channel<NUM_NODES>& operator[](int idx) const {
        return data[idx];
    }
};

template<int NUM_NODES>
struct ChannelArray<0, NUM_NODES> {
    __host__ __device__ inline Channel<NUM_NODES>& operator[](int) {
#ifdef __CUDA_ARCH__
        __trap();
#else
        __builtin_trap();
#endif
    }
    __host__ __device__ inline const Channel<NUM_NODES>& operator[](int) const {
#ifdef __CUDA_ARCH__
        __trap();
#else
        __builtin_trap();
#endif
    }
};

/* ----------   Multicast TMA descriptor cache (variadic recursion)  ---------- */

namespace detail_mc {

template<typename... Args>
struct tma_dict {
    __host__ tma_dict() {}
    template<typename DT> __host__ tma_dict(DT*, int, int, int, int) {}
    __host__ __device__ tma_dict(const tma_dict&) {}
    template<typename U, int A> __device__ inline const CUtensorMap* get() const { return nullptr; }
};

template<typename ST, typename... Rest>
struct tma_dict<ST, Rest...> {
    using DESC = ::dist::detail::tma_descriptor<ST>;
    using TILE = typename DESC::T;
    static constexpr int AXIS = DESC::axis;

    CUtensorMap         desc;
    tma_dict<Rest...>   rest;

    __host__ tma_dict() {}
    __host__ tma_dict(typename TILE::dtype* data, int b, int d, int r, int c)
        : rest(data, b, d, r, c) {
        ::dist::detail::create_tensor_map<TILE, AXIS>(&desc, data, b, d, r, c);
    }
    __host__ __device__ tma_dict(const tma_dict& other)
        : desc(other.desc), rest(other.rest) {}

    template<typename U, int A> __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<TILE, U> && AXIS == A) { return &desc; }
        else                                                { return rest.template get<U, A>(); }
    }
};

} // namespace detail_mc


/* ----------   distributed_tensor  ---------- */

/**
 * @brief Distributed tensor descriptor.
 *
 * @tparam LocalTensor     `dist::local_tensor<>` — per-device tensor descriptor.
 * @tparam LOCAL_SIZE      number of GPUs in this node.
 * @tparam MULTICAST       bind a CUDA multicast VA + TMA descriptors.
 * @tparam NUM_CHANNELS    inter-node channels (= QPs = comm CTAs); 0 = intra-only.
 * @tparam NUM_NODES       peer node count for inter-node sends; 1 = no inter-node.
 * @tparam TMA_Types       `kittens::st<>` tile metadata for multicast TMA descs.
 */
template<typename LocalTensor,
         int  LOCAL_SIZE   = 8,
         bool MULTICAST    = true,
         int  NUM_CHANNELS = 0,
         int  NUM_NODES    = 1,
         typename... TMA_Types>
struct distributed_tensor {
    using local_tensor_t = LocalTensor;
    using GL_t = LocalTensor;
    using T    = typename LocalTensor::dtype;
    using dtype = T;

    static constexpr int  num_devices  = LOCAL_SIZE;
    static constexpr bool multicast    = MULTICAST;
    static constexpr int  num_channels = NUM_CHANNELS;
    static constexpr int  num_nodes    = NUM_NODES;

    /* ---- Intra-node state ---- */
    T*  mc_ptr;                       // multicast VA (nullptr if !MULTICAST)
    LocalTensor  gls[LOCAL_SIZE];       // per-device local tensor views
    detail_mc::tma_dict<TMA_Types...> tma_descs;   // multicast TMA descs

    /* ---- Inter-node state ---- */
    ChannelArray<NUM_CHANNELS, NUM_NODES> channels;

    /* ---- Distributed-buffer accessors ---- */
    __host__ __device__ inline const LocalTensor& operator[](int idx) const { return gls[idx]; }

    __device__ inline T* mc_ptr_at(const coord& idx) const {
        static_assert(MULTICAST, "Multicast not enabled for this dbuf.");
        const LocalTensor& g = gls[0];
        return &mc_ptr[((idx.b * (uint64_t)g.depth() + idx.d) * g.rows() + idx.r) * g.cols() + idx.c];
    }

    template<typename U, int axis>
    __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U, axis>();
    }

    __host__ __device__ inline auto batch() const { return gls[0].batch(); }
    __host__ __device__ inline auto depth() const { return gls[0].depth(); }
    __host__ __device__ inline auto rows()  const { return gls[0].rows(); }
    __host__ __device__ inline auto cols()  const { return gls[0].cols(); }
    __host__ __device__ inline auto numel() const { return gls[0].numel(); }
    template<int axis> __device__ inline size_t shape()  const { return gls[0].template shape<axis>(); }
    template<int axis> __device__ inline size_t stride() const { return gls[0].template stride<axis>(); }

    /* ---- Constructors ---- */
    template<size_t... I>
    __host__ inline distributed_tensor(std::index_sequence<I...>,
                                       T** data,
                                       detail::arg_t<LocalTensor::__b__> b,
                                       detail::arg_t<LocalTensor::__d__> d,
                                       detail::arg_t<LocalTensor::__r__> r,
                                       detail::arg_t<LocalTensor::__c__> c)
        : mc_ptr(nullptr),
          gls{ LocalTensor(data[I], b, d, r, c)... },
          tma_descs() {
        static_assert(!MULTICAST, "Multicast pointer required.");
    }

    template<size_t... I>
    __host__ inline distributed_tensor(std::index_sequence<I...>,
                                       T*  mc,
                                       T** data,
                                       detail::arg_t<LocalTensor::__b__> b,
                                       detail::arg_t<LocalTensor::__d__> d,
                                       detail::arg_t<LocalTensor::__r__> r,
                                       detail::arg_t<LocalTensor::__c__> c)
        : mc_ptr(mc),
          gls{ LocalTensor(data[I], b, d, r, c)... },
          tma_descs(mc,
                    static_cast<int>(static_cast<size_t>(gls[0].batch_internal)),
                    static_cast<int>(static_cast<size_t>(gls[0].depth_internal)),
                    static_cast<int>(static_cast<size_t>(gls[0].rows_internal)),
                    static_cast<int>(static_cast<size_t>(gls[0].cols_internal))) {
        static_assert(MULTICAST, "Multicast disabled — don't pass mc_ptr.");
    }

    __host__ inline distributed_tensor(T** data,
                                       detail::arg_t<LocalTensor::__b__> b,
                                       detail::arg_t<LocalTensor::__d__> d,
                                       detail::arg_t<LocalTensor::__r__> r,
                                       detail::arg_t<LocalTensor::__c__> c)
        : distributed_tensor(std::make_index_sequence<LOCAL_SIZE>{}, data, b, d, r, c) {}

    __host__ inline distributed_tensor(T* mc, T** data,
                                       detail::arg_t<LocalTensor::__b__> b,
                                       detail::arg_t<LocalTensor::__d__> d,
                                       detail::arg_t<LocalTensor::__r__> r,
                                       detail::arg_t<LocalTensor::__c__> c)
        : distributed_tensor(std::make_index_sequence<LOCAL_SIZE>{}, mc, data, b, d, r, c) {}

    /* ---- Inter-node compute-CTA API ---- */

    __device__ inline void enqueue_send(int channel, uint32_t tile_id,
                                        uint16_t dst_node, uint64_t off,
                                        uint32_t bytes, uint32_t imm) const;

    __device__ inline void wait_inter(int tile_id, uint32_t expected_iter) const {
        const int c = (NUM_CHANNELS > 0) ? (tile_id % NUM_CHANNELS) : 0;
        const int t = (NUM_CHANNELS > 0) ? (tile_id / NUM_CHANNELS) : tile_id;
        const TileFlag* f = &channels[c].arrived[t];
        uint32_t v;
        do {
            v = comm::atomic_u32::acquire_load_gpu(&f->v);
        } while (v < expected_iter);
    }

    /* ---- Inter-node comm-CTA API (one comm CTA per channel) ---- */

    __device__ inline bool try_dequeue_send(int channel, SendJob* out) const;
    /**
     * @brief Post a single RDMA WRITE_WITH_IMM transfer.
     *
     * @param local_off   byte offset within the sender's DMA-BUF MR (the
     *                    kernel's own data buffer). Source for the SGE.
     * @param remote_off  byte offset within the receiver's data MR (typically
     *                    the recv_buf). Destination for the RDMA WRITE.
     *                    Pass equal to local_off when the layouts match.
     * @param lane_id     proxy-side QP-routing hint; pass 0 if you don't care.
     */
    __device__ inline void put_inter(int channel, int dst_node,
                                     uint64_t local_off,
                                     uint64_t remote_off,
                                     uint32_t bytes,
                                     uint32_t imm,
                                     uint32_t lane_id = 0) const;
    __device__ inline void flush_inter(int channel) const;
    __device__ inline void drain_step(int channel) const;
};

/* ----------   Convenience aliases  ---------- */

template<int LOCAL_SIZE>
using barrier_distributed_tensor =
    distributed_tensor<local_tensor<int, -1, -1, -1, -1>, LOCAL_SIZE, true>;

template<int LOCAL_SIZE>
using barrier_dbuf = barrier_distributed_tensor<LOCAL_SIZE>;

/* ----------   Cross-device barrier ops (multicast int dbuf)  ---------- */

template<int LOCAL_SIZE>
__device__ static inline void signal(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx, int dst_dev_idx, int val
) {
    comm::atomic_u32::release_add_sys(&barrier[dst_dev_idx][idx], val);
}

template<int LOCAL_SIZE>
__device__ static inline void signal_all(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx, int val
) {
    comm::multimem<int>::red<comm::reduce_op::ADD>(
        reinterpret_cast<int*>(barrier.mc_ptr_at(idx)), val);
}

template<int LOCAL_SIZE>
__device__ static inline void wait(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx,
    int dev_idx, int expected
) {
    int val;
    do {
        val = comm::atomic_u32::relaxed_load_s32_sys(&barrier[dev_idx][idx]);
    } while (val != expected);
}

template<int LOCAL_SIZE>
__device__ static inline void barrier_all(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx, int dev_idx
) {
    signal_all<LOCAL_SIZE>(barrier, idx, 1);
    wait<LOCAL_SIZE>(barrier, idx, dev_idx, LOCAL_SIZE);
    comm::atomic_u32::release_add_sys(&barrier[dev_idx][idx], -LOCAL_SIZE);
}

template<int LOCAL_SIZE>
__device__ static inline void wait_acquire(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx,
    int dev_idx, int expected
) {
    int val;
    do {
        val = comm::atomic_u32::acquire_load_s32_sys(&barrier[dev_idx][idx]);
    } while (val != expected);
}

template<int LOCAL_SIZE>
__device__ static inline void wait_mc(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx, int expected
) {
    int val;
    do {
        comm::multimem<int>::ld_reduce<comm::reduce_op::MAX, comm::memory_model::STRONG>(
            val, reinterpret_cast<const int*>(barrier.mc_ptr_at(idx)));
    } while (val != expected);
}

template<int LOCAL_SIZE>
__device__ static inline bool is_ready(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx,
    int dev_idx, int expected
) {
    int val;
    val = comm::atomic_u32::relaxed_load_s32_sys(&barrier[dev_idx][idx]);
    return val == expected;
}

template<int LOCAL_SIZE>
__device__ static inline bool is_ready_mc(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx, int expected
) {
    int val;
    comm::multimem<int>::ld_reduce<comm::reduce_op::MAX, comm::memory_model::STRONG>(
        val, reinterpret_cast<const int*>(barrier.mc_ptr_at(idx)));
    return val == expected;
}

template<int LOCAL_SIZE>
__device__ static inline void clear_slot_mc(
    const barrier_distributed_tensor<LOCAL_SIZE>& barrier,
    const coord& idx
) {
    comm::multimem<int>::st<comm::memory_model::STRONG>(
        reinterpret_cast<int*>(barrier.mc_ptr_at(idx)), 0);
}

/* ----------   Construction helpers  ---------- */

template<typename DistributedTensor>
__host__ inline DistributedTensor make_distributed_tensor(uint64_t* data, int b, int d, int r, int c) {
    return DistributedTensor(reinterpret_cast<typename DistributedTensor::T**>(data),
                             detail::make_arg<DistributedTensor::GL_t::__b__>(b),
                             detail::make_arg<DistributedTensor::GL_t::__d__>(d),
                             detail::make_arg<DistributedTensor::GL_t::__r__>(r),
                             detail::make_arg<DistributedTensor::GL_t::__c__>(c));
}

template<typename DistributedTensor>
__host__ inline DistributedTensor make_distributed_tensor(
    uint64_t mc, uint64_t* data, int b, int d, int r, int c
) {
    return DistributedTensor(reinterpret_cast<typename DistributedTensor::T*>(mc),
                             reinterpret_cast<typename DistributedTensor::T**>(data),
                             detail::make_arg<DistributedTensor::GL_t::__b__>(b),
                             detail::make_arg<DistributedTensor::GL_t::__d__>(d),
                             detail::make_arg<DistributedTensor::GL_t::__r__>(r),
                             detail::make_arg<DistributedTensor::GL_t::__c__>(c));
}

template<typename LocalTensor,
         int  LOCAL_SIZE   = 8,
         bool MULTICAST    = true,
         int  NUM_CHANNELS = 0,
         int  NUM_NODES    = 1,
         typename... TMA_Types>
using dbuf = distributed_tensor<LocalTensor, LOCAL_SIZE, MULTICAST, NUM_CHANNELS, NUM_NODES, TMA_Types...>;

template<typename DistributedTensor>
__host__ inline DistributedTensor make_dbuf(uint64_t* data, int b, int d, int r, int c) {
    return make_distributed_tensor<DistributedTensor>(data, b, d, r, c);
}

template<typename DistributedTensor>
__host__ inline DistributedTensor make_dbuf(uint64_t mc, uint64_t* data, int b, int d, int r, int c) {
    return make_distributed_tensor<DistributedTensor>(mc, data, b, d, r, c);
}

} // namespace dist
