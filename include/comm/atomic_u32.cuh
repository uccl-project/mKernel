#pragma once

// Single-instruction acquire/release/relaxed 32-bit primitives, shared across
// the multinode kernels. Each helper emits exactly one PTX instruction with
// the named memory-ordering and scope.
//
// Why a header instead of a real library: these are pure inline asm wrappers,
// so any compilation unit that uses them just needs the inline definition in
// scope. Putting them under `comm::atomic_u32::` keeps the call sites
// self-documenting (you can read off the scope/ordering at the call) and lets
// us rip the per-kernel duplicates that previously lived as `gemm_ar_*` /
// `gemm_rs_*` near-identical copies.
//
// All variants take `PtrT*` (template) and reinterpret to `uint32_t*` at the
// asm boundary, so callers can pass `int*`, `uint32_t*`, or any 32-bit
// pointer without a manual cast. The underlying PTX type is `.u32`; sign
// interpretation is the caller's responsibility (PTX emits identical SASS
// for `.u32` and `.s32` 32-bit acquire/release loads).

#include <cstdint>

namespace comm {
namespace atomic_u32 {

// ============================================================================
// Acquire loads — match a `release` store on the same address; downstream
// reads of memory by the loading thread see the producer's writes.
// ============================================================================

// GPU-scope acquire: pairs with .gpu-scope release on the same GPU.
// Use this for cross-CTA, same-GPU handoffs (the common case).
template <typename PtrT>
__device__ inline uint32_t acquire_load_gpu(PtrT* ptr) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(reinterpret_cast<const volatile uint32_t*>(ptr))
                 : "memory");
    return v;
}

template <typename PtrT>
__device__ inline int32_t acquire_load_s32_gpu(PtrT* ptr) {
    return static_cast<int32_t>(acquire_load_gpu(ptr));
}

// System-scope acquire: pairs with `.sys` release OR with NIC writes (e.g. an
// EFA proxy thread on the host CPU). Required when the writer is OFF this
// GPU (peer GPU multimem stores OR NIC RDMA arrivals).
template <typename PtrT>
__device__ inline uint32_t acquire_load_sys(PtrT* ptr) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(reinterpret_cast<const volatile uint32_t*>(ptr))
                 : "memory");
    return v;
}

template <typename PtrT>
__device__ inline int32_t acquire_load_s32_sys(PtrT* ptr) {
    return static_cast<int32_t>(acquire_load_sys(ptr));
}

// ============================================================================
// Release stores — pair with the matching `acquire` load. Stores done before
// this on the storing thread are visible to the loader on success.
// ============================================================================

template <typename PtrT>
__device__ inline void release_store_gpu(PtrT* ptr, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<volatile uint32_t*>(ptr)), "r"(v)
                 : "memory");
}

template <typename PtrT>
__device__ inline void release_store_sys(PtrT* ptr, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<volatile uint32_t*>(ptr)), "r"(v)
                 : "memory");
}

template <typename PtrT>
__device__ inline void release_add_gpu(PtrT* ptr, int32_t v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<volatile int32_t*>(ptr)), "r"(v)
                 : "memory");
}

template <typename PtrT>
__device__ inline void release_add_sys(PtrT* ptr, int32_t v) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<volatile int32_t*>(ptr)), "r"(v)
                 : "memory");
}

// ============================================================================
// Split release — a fence followed by relaxed adds is a release pattern in the
// PTX memory model. One fence can cover several adds, each of which would fence
// on its own as a red.release.
// ============================================================================

__device__ inline void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

template <typename PtrT>
__device__ inline void relaxed_add_gpu(PtrT* ptr, int32_t v) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<volatile int32_t*>(ptr)), "r"(v)
                 : "memory");
}

// ============================================================================
// Relaxed loads — no synchronization, just an opaque 32-bit read. Mostly used
// for poll loops where the reader will re-load with stronger ordering after
// the comparison succeeds.
// ============================================================================

template <typename PtrT>
__device__ inline uint32_t relaxed_load_gpu(PtrT* ptr) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(reinterpret_cast<const volatile uint32_t*>(ptr))
                 : "memory");
    return v;
}

template <typename PtrT>
__device__ inline int32_t relaxed_load_s32_gpu(PtrT* ptr) {
    return static_cast<int32_t>(relaxed_load_gpu(ptr));
}

template <typename PtrT>
__device__ inline uint32_t relaxed_load_sys(PtrT* ptr) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(reinterpret_cast<const volatile uint32_t*>(ptr))
                 : "memory");
    return v;
}

template <typename PtrT>
__device__ inline int32_t relaxed_load_s32_sys(PtrT* ptr) {
    return static_cast<int32_t>(relaxed_load_sys(ptr));
}

template <typename PtrT>
__device__ inline uint32_t volatile_load(PtrT* ptr) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(reinterpret_cast<const volatile uint32_t*>(ptr))
                 : "memory");
    return v;
}

}  // namespace atomic_u32
}  // namespace comm
