#pragma once
/**
 * @file
 * @brief Utilities run by groups.
 */

// tma.cuh and tma_cluster.cuh are included in group.cuh

// CLC scheduler operations live at thread scope in kittens::clc
// (tk_ops_thread_util_util.cuh), guarded by KITTENS_SM10X || KITTENS_SM120.

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)

/**
 * @brief Programmatic Dependent Kernel Launch (PDL) utilities. Available on Hopper and later.
 *
 * PDL allows partial overlap between two consecutive kernels in the same stream.
 *
 * @note The secondary kernel must be launched with
 * `cudaLaunchAttributeProgrammaticStreamSerialization` attribute and
 * `programmaticStreamSerializationAllowed` set to 1.
 */
struct pdl {
    /**
     * @brief Signals that a primary kernel has completed its dependent work, enabling a secondary
     * kernel to launch.
     *
     * @note The secondary kernel will only launch when all threadblocks in the primary kernel have
     * called this function. If a threadblock does not call this, the arrival is implicitly
     * triggered at threadblock exit.
     * @note This does not guarantee memory visibility. For memory visibility, the secondary kernel
     * must call wait().
     */
    __device__ static inline void arrive() {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    /**
     * @brief Blocks until the primary kernel fully completes and flushes memory.
     */
    __device__ static inline void wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
};

#endif