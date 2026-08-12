/**
 * @file
 * @brief Various utilities for single-thead synchronization operations.
 */

#pragma once

namespace kittens {

struct semaphore {
private:
    uint64_t value;
}; // note that this is an opaque type, so the value should not be accessed directly.
template<int num_warps> struct barrier {
    int barrier_id;
    __device__ __forceinline__ barrier(int _id) : barrier_id(_id) {}
    __device__ __forceinline__ barrier operator[](int i) {
        return barrier(barrier_id + i);
    }
};

/**
 * @brief Initializes a synchronization semaphore with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a semaphore that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the semaphore with a thread count semaphore.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    asm volatile (
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& sem) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_ptr)
        : "memory"
    );
}
template<int num_warps> __device__ static inline void arrive(barrier<num_warps> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& sem, uint32_t count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

/**
* @brief Waits for the requested semaphore phase.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static inline bool try_wait(semaphore &sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    uint32_t success;

    asm volatile(
        "{\n"
        ".reg .pred P1; \n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
        "selp.b32 %0, 1, 0, P1; \n"
        "}\n"
        : "=r"(success)
        : "r"(mbar_ptr), "r"(kPhaseBit)
        : "memory"
    );

    return static_cast<bool>(success);
}

__device__ static inline void careful_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"
        "mov.b64                   start_clock, %clock64;\n"
        ".reg .pred                P_CLOCK;\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "mov.b64                   current_clock, %clock64;\n"
        "sub.u64                   current_clock, current_clock, start_clock;\n"
        "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n"
        "@P_CLOCK                  trap;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

/**
* @brief Checks if the requested semaphore phase is ready.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline int test_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    int result;
    asm volatile (
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0,1,0,P1;"
        "}\n"
        : "=r"(result)
        : "r"(mbar_ptr), "r"(kPhaseBit)
    );
    return result;
}

__device__ static inline void arrive_and_wait(semaphore& sem, int kPhaseBit) {
    arrive(sem);
    wait(sem, kPhaseBit);
}
template<int num_warps> __device__ static inline void arrive_and_wait(barrier<num_warps> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

template<int N=0> __device__ static inline void load_async_wait() { // for completing (non-TMA) async loads
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    }
    __syncwarp();
}

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>             inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args> inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

/* ----------   TCGEN05 synchronization  ---------- */

__device__ static inline void tensor_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}

__device__ static inline void tensor_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

__device__ static inline void tensor_load_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
}

__device__ static inline void tensor_store_wait() {
    asm volatile("tcgen05.wait::st.sync.aligned;\n");
}

template<int NCTA>
__device__ static inline void tensor_commit(semaphore &sem) {
    static_assert(NCTA == 1 || NCTA == 2);
    if constexpr (NCTA == 1) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
            :: "l"(__cvta_generic_to_shared(&sem))
            : "memory");
    } else {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem))),
               "h"(static_cast<uint16_t>(0x3))
            : "memory");
    }
}

} // namespace kittens
