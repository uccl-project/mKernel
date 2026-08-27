/**
 * @file
 * @brief Thread-level BF16 tcgen05 primitives for CTA-group 1.
 */
#pragma once

#include "../common/types.cuh"
#include "tk_ops_thread_util_sync.cuh"

namespace kittens {
namespace detail {
namespace tcgen05_bf16 {

template<int M, int N, bool TRANS_B>
__device__ static inline constexpr uint32_t instruction_descriptor() {
    static_assert(M == 64 || M == 128);
    static_assert(N >= 8 && N <= 256 && N % 8 == 0);

    uint32_t desc = 0;
    desc |= 0b01u << 4;             // FP32 accumulator
    desc |= 0b001u << 7;            // BF16 A
    desc |= 0b001u << 10;           // BF16 B
    desc |= static_cast<uint32_t>(TRANS_B) << 16;
    desc |= static_cast<uint32_t>(N >> 3) << 17;
    desc |= static_cast<uint32_t>(M >> 4) << 24;
    return desc;
}

template<bool ACCUMULATE>
__device__ static inline void issue(
    uint32_t d_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.eq.u32 p, 1, %4;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(d_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "n"(ACCUMULATE ? 1 : 0)
        : "memory");
}

template<bool ACCUMULATE, bool TRANS_B,
         ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mma(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    using A_st = typename ducks::st_descriptor::detail::get_st<A>;
    using B_st = typename ducks::st_descriptor::detail::get_st<B>;

    static_assert(std::is_same_v<typename A_st::T, bf16>);
    static_assert(std::is_same_v<typename B_st::T, bf16>);
    static_assert(std::is_same_v<typename D::T, float>);

    constexpr int M = A_st::rows;
    constexpr int N = TRANS_B ? B_st::cols : B_st::rows;
    constexpr int K = A_st::cols;
    static_assert(M == D::rows && N == D::cols);
    static_assert((TRANS_B ? B_st::rows : B_st::cols) == K);
    static_assert(K % 16 == 0);

    constexpr uint32_t idesc = instruction_descriptor<M, N, TRANS_B>();
    st_descriptor<A_st, transpose::N> a_desc(a);
    st_descriptor<B_st, TRANS_B> b_desc(b);

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    issue<ACCUMULATE>(
        d.addr, a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0), idesc);

    #pragma unroll
    for (int chunk = 1; chunk < K / 16; ++chunk) {
        issue<true>(
            d.addr, a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk), idesc);
    }

    // Release this shared-memory stage only after all tcgen05 instructions
    // that reference it have completed.
    tensor_commit<1>(inputs_finished);
}

template<bool ACCUMULATE,
         ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    tcgen05_bf16::mma<ACCUMULATE, false>(d, a, b, inputs_finished);
}

template<bool ACCUMULATE,
         ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mma_AB(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    tcgen05_bf16::mma<ACCUMULATE, true>(d, a, b, inputs_finished);
}

} // namespace tcgen05_bf16
} // namespace detail
} // namespace kittens
