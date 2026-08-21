/**
 * @file
 * @brief Minimal CTA-group 1 BF16 tcgen05 MMA backend.
 *
 * Included inside kittens::group.  Only the SS ABt path used by the first
 * Blackwell dispatch GEMM is exposed: A and B are BF16 K-major shared tiles,
 * D is an FP32 tensor-memory tile.
 */

template<ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    if (laneid() == 0) {
        ::kittens::detail::tcgen05_bf16::mma_ABt<false>(
            d, a, b, inputs_finished);
    }
}

template<ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mm_AB(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    if (laneid() == 0) {
        ::kittens::detail::tcgen05_bf16::mma_AB<false>(
            d, a, b, inputs_finished);
    }
}

template<ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mma_AB(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    if (laneid() == 0) {
        ::kittens::detail::tcgen05_bf16::mma_AB<true>(
            d, a, b, inputs_finished);
    }
}

template<ducks::tt::all D,
         ducks::st_descriptor::input A,
         ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(
    D &d, const A &a, const B &b, semaphore &inputs_finished) {
    if (laneid() == 0) {
        ::kittens::detail::tcgen05_bf16::mma_ABt<true>(
            d, a, b, inputs_finished);
    }
}
