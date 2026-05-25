/**
 * @file arrival.cuh
 * @brief Arrival flag helpers for inter-node tile notification.
 *
 * Arrival flags are host-pinned uint32_t arrays, RDMA-registered so the
 * remote node can write a 4-byte epoch value via chained RDMA_WRITE.
 * GPU compute/comm CTAs poll these flags to know when inbound tiles have arrived.
 *
 * Memory layout:
 *   arrival_flags[tile_id] = 0       → tile not yet arrived
 *   arrival_flags[tile_id] = epoch   → tile data is in recv buffer
 *
 * The chained RDMA write (data + flag) with RC ordering guarantees that
 * when the GPU sees arrival_flags[tile_id] == epoch, the tile data in
 * the recv buffer is fully visible.
 */
#pragma once

#include "types.h"
#include "../atomic_u32.cuh"
#include "../../common/cuda_checks.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <cstdio>

// Forward-declare ibv types to avoid requiring verbs.h in CUDA compilation units.
// The actual ibv_mr*/ibv_pd* usage is in host-only code paths.
struct ibv_mr;
struct ibv_pd;

namespace internode {

// ---------------------------------------------------------------------------
// ArrivalFlags: host-pinned array, RDMA-registered, GPU-pollable
// ---------------------------------------------------------------------------

struct ArrivalFlags {
    volatile uint32_t* host_ptr;    // host-pinned memory (CPU can read/write)
    uint32_t*          device_ptr;  // device-accessible alias (GPU polls)
    volatile uint32_t* tail_host_ptr;   // per-queue producer-published tail counters
    uint32_t*          tail_device_ptr; // device alias for tail counters
    ibv_mr*            mr;          // RDMA memory region (remote writes here)
    int                count;       // number of tile slots
    int                tail_count;  // number of queue-tail slots
    bool               host_mapped; // true when host_ptr/device_ptr alias mapped host memory
};

/**
 * Create arrival flags array.
 * Allocates device memory with cudaMalloc; the host_ptr field aliases the
 * same device pointer. RDMA registration is done separately (caller passes
 * pd + registers mr).
 */
inline ArrivalFlags create_arrival_flags(int count, int tail_count = 0) {
    ArrivalFlags flags{};
    flags.count = count;
    flags.tail_count = tail_count > 0 ? tail_count : 0;
    flags.mr = nullptr;
    flags.host_mapped = false;

    // Allocate in device memory (GPU HBM) for fast polling.
    // Remote RDMA writes to this via GDR (nvidia_peermem).
    uint32_t* dev_ptr = nullptr;
    const size_t total_words = (size_t)count + (size_t)flags.tail_count;
    MKERNEL_CUDACHECK(cudaMalloc(&dev_ptr, total_words * sizeof(uint32_t)));
    MKERNEL_CUDACHECK(cudaMemset(dev_ptr, 0, total_words * sizeof(uint32_t)));
    flags.device_ptr = dev_ptr;
    flags.host_ptr = (volatile uint32_t*)dev_ptr;  // same pointer (device mem)
    flags.tail_device_ptr = dev_ptr + count;
    flags.tail_host_ptr = (volatile uint32_t*)(dev_ptr + count);

    return flags;
}

/**
 * Create arrival flags in host-mapped pinned memory.
 * This is used by the EFA direct-verbs backend so the receiver-side proxy can
 * update flags directly after processing RDMA-with-immediate CQEs.
 */
inline ArrivalFlags create_mapped_arrival_flags(int count, int tail_count = 0) {
    ArrivalFlags flags{};
    flags.count = count;
    flags.tail_count = tail_count > 0 ? tail_count : 0;
    flags.mr = nullptr;
    flags.host_mapped = true;
    MKERNEL_CUDACHECK(cudaHostAlloc((void**)&flags.host_ptr,
                                 ((size_t)count + (size_t)flags.tail_count) * sizeof(uint32_t),
                                 cudaHostAllocMapped));
    memset((void*)flags.host_ptr, 0,
           ((size_t)count + (size_t)flags.tail_count) * sizeof(uint32_t));
    MKERNEL_CUDACHECK(cudaHostGetDevicePointer((void**)&flags.device_ptr,
                                            (void*)flags.host_ptr, 0));
    flags.tail_device_ptr = flags.device_ptr + count;
    flags.tail_host_ptr = flags.host_ptr + count;
    return flags;
}

/**
 * Reset all arrival flags to zero (call between kernel launches).
 */
inline void reset_arrival_flags(ArrivalFlags& flags) {
    const size_t total_words = (size_t)flags.count + (size_t)flags.tail_count;
    if (flags.host_mapped) {
        memset((void*)flags.host_ptr, 0, total_words * sizeof(uint32_t));
    } else {
        MKERNEL_CUDACHECK(cudaMemset((void*)flags.device_ptr, 0, total_words * sizeof(uint32_t)));
    }
}

/**
 * Destroy arrival flags. Caller must already have deregistered the MR;
 * this releases device memory (default) or host-pinned memory (mapped flags).
 */
inline void destroy_arrival_flags(ArrivalFlags& flags) {
    // MR deregistration is caller's responsibility (needs ibv_dereg_mr)
    if (flags.host_mapped) {
        if (flags.host_ptr) cudaFreeHost((void*)flags.host_ptr);
    } else {
        if (flags.device_ptr) cudaFree((void*)flags.device_ptr);
    }
    flags = ArrivalFlags{};
}

// ---------------------------------------------------------------------------
// FlagStaging: single uint32_t in host-pinned memory, used as the sge source
// for the 4-byte RDMA write that sets the remote arrival flag.
// ---------------------------------------------------------------------------

struct FlagStaging {
    uint32_t* host_ptr;   // host-pinned (value = current epoch)
    ibv_mr*   mr;         // RDMA memory region (local read for sge)
    int       count;      // number of uint32_t slots allocated
};

/**
 * Create flag staging buffer.
 * RDMA registration is done separately by caller.
 */
inline FlagStaging create_flag_staging(int count = 8) {
    FlagStaging s{};
    s.mr = nullptr;
    s.count = count > 0 ? count : 1;
    MKERNEL_CUDACHECK(cudaHostAlloc(&s.host_ptr, s.count * sizeof(uint32_t),
                                  cudaHostAllocDefault));
    memset(s.host_ptr, 0, s.count * sizeof(uint32_t));
    return s;
}

inline void destroy_flag_staging(FlagStaging& s) {
    if (s.host_ptr) cudaFreeHost(s.host_ptr);
    s = FlagStaging{};
}

// ---------------------------------------------------------------------------
// StageBarrierFlags: small host-pinned array used for host-side stage barriers.
// The proxy writes a token to the peer via RDMA, and the local host polls it.
// ---------------------------------------------------------------------------

struct StageBarrierFlags {
    volatile uint32_t* host_ptr;    // host-pinned memory (CPU polls)
    uint32_t*          device_ptr;  // device alias for future GPU-side use
    ibv_mr*            mr;          // RDMA memory region (remote writes here)
    int                count;       // number of stage slots
};

inline StageBarrierFlags create_stage_barrier_flags(int count) {
    StageBarrierFlags flags{};
    flags.count = count;
    flags.mr = nullptr;
    MKERNEL_CUDACHECK(cudaHostAlloc((void**)&flags.host_ptr,
                                 count * sizeof(uint32_t),
                                 cudaHostAllocMapped));
    memset((void*)flags.host_ptr, 0, count * sizeof(uint32_t));
    MKERNEL_CUDACHECK(cudaHostGetDevicePointer((void**)&flags.device_ptr,
                                            (void*)flags.host_ptr, 0));
    return flags;
}

inline void reset_stage_barrier_flags(StageBarrierFlags& flags) {
    memset((void*)flags.host_ptr, 0, flags.count * sizeof(uint32_t));
}

inline void destroy_stage_barrier_flags(StageBarrierFlags& flags) {
    if (flags.host_ptr) cudaFreeHost((void*)flags.host_ptr);
    flags = StageBarrierFlags{};
}

// ---------------------------------------------------------------------------
// ForwardNotifyTable: host-pinned receiver-side store-and-forward notices.
// ---------------------------------------------------------------------------

struct ForwardNotifyTable {
    volatile ForwardNotify* host_ptr;
    ibv_mr*                 mr;
    int                     count;
};

inline ForwardNotifyTable create_forward_notify_table(int count) {
    ForwardNotifyTable table{};
    table.count = count > 0 ? count : 0;
    table.mr = nullptr;
    if (table.count == 0) return table;
    MKERNEL_CUDACHECK(cudaHostAlloc((void**)&table.host_ptr,
                                    (size_t)table.count * sizeof(ForwardNotify),
                                    cudaHostAllocMapped));
    memset((void*)table.host_ptr, 0, (size_t)table.count * sizeof(ForwardNotify));
    return table;
}

inline void reset_forward_notify_table(ForwardNotifyTable& table) {
    if (table.host_ptr && table.count > 0) {
        memset((void*)table.host_ptr, 0, (size_t)table.count * sizeof(ForwardNotify));
    }
}

inline void destroy_forward_notify_table(ForwardNotifyTable& table) {
    if (table.host_ptr) cudaFreeHost((void*)table.host_ptr);
    table = ForwardNotifyTable{};
}

// ---------------------------------------------------------------------------
// Device-side polling helper
// ---------------------------------------------------------------------------

/**
 * Wait until arrival_flags[tile_id] == expected_epoch.
 * Uses volatile load (__ldcv) to bypass L1 cache.
 * After flag is seen, issues __threadfence_system() to ensure the tile
 * data written via RDMA is visible to this thread.
 */
#ifdef __CUDA_ARCH__
__device__ __forceinline__ void wait_arrival(
    volatile uint32_t* flags, int tile_id, uint32_t expected_epoch)
{
    uint32_t val;
    do {
        val = comm::atomic_u32::volatile_load(&flags[tile_id]);
        if (val == expected_epoch) break;
        __nanosleep(100);
    } while (true);
    __threadfence_system();
}
#else
inline void wait_arrival(volatile uint32_t*, int, uint32_t) {} // host stub
#endif

} // namespace internode
