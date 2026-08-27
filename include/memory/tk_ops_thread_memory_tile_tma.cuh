/**
 * @file
 * @brief Minimal tile TMA operations used by the release kernels.
 */

#pragma once

#include "../common/tk_common_common.cuh"
#include "../common/types.cuh"
#include "tk_ops_tma_layout_concepts.cuh"
#include "tk_ops_thread_util_util.cuh"

namespace kittens {
namespace tma {
namespace detail {

template<kittens::ducks::st::all ST, int axis>
__device__ inline int4 tma_coords(const coord<ducks::default_type> &unit_coord) {
    static_assert(ST::swizzle, "tma_coords should only be called for swizzled tiles");
    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(typename ST::dtype);
    if constexpr      (axis == 2) return {unit_coord.r, unit_coord.c / swizzle_elements, unit_coord.d, unit_coord.b};
    else if constexpr (axis == 1) return {unit_coord.d, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.b};
    else if constexpr (axis == 0) return {unit_coord.b, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.d};
}

} // namespace detail

template<int axis, cache_policy policy, ducks::st::all ST, detail::tma_layout_for<ST, axis> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();

    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    store_commit_group();
}

template<ducks::st::all ST, detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

template<int axis, cache_policy policy, ducks::st::all ST, detail::tma_layout_for<ST, axis> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>),
                  "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();

    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    store_commit_group();
}

template<ducks::st::all ST, detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

template<int axis, cache_policy policy, ducks::st::all ST, detail::tma_layout_for<ST, axis> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();

    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, %6}], [%2];"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        } else {
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
}

template<ducks::st::all ST, detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
}

namespace cluster {

/**
 * @brief Multicast tile load: one global fetch lands in every CTA of the
 *        cluster selected by `cluster_mask`, and completes each of their
 *        mbarriers.
 *
 * Vendored from ThunderKittens ops/thread/memory/tile/tma.cuh for the sm_103a
 * port, retyped onto mKernel's `detail::tma_layout_for` descriptor concept
 * (mKernel uses dist::local_tensor rather than TK's gl). Only the plain
 * multicast form is brought over - the cta_group::2 variant that feeds a 2-CTA
 * MMA operand directly is not needed here.
 *
 * Every CTA named in `cluster_mask` must have run expect_bytes for these bytes
 * on its own copy of `bar` before this is issued.
 */
/**
 * @brief Cluster tile load whose completion is reported to `dst_mbar_cta`'s
 *        copy of `bar` rather than the issuing CTA's.
 *
 * This is how a CTA pair funnels both halves of an MMA operand into the leader
 * CTA's barrier: every CTA loads its own slice (cluster_mask = 1 << ctarank)
 * but points the transaction count at CTA 0, so the leader's single wait covers
 * the whole cluster's inputs.
 */
template<int axis, cache_policy policy, ducks::st::all ST,
         detail::tma_layout_for<ST, axis> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx,
                                         semaphore &bar, uint16_t cluster_mask,
                                         int dst_mbar_cta) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();

    uint32_t neighbor_mbar_ptr;
    asm volatile ("mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_ptr) : "r"(mbar_ptr), "r"(dst_mbar_cta));

    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
              "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
              "r"(tma_coords.w), "h"(cluster_mask)
            : "memory"
        );
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
              "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b),
              "h"(cluster_mask)
            : "memory"
        );
    }
}

template<ducks::st::all ST, detail::tma_layout_for<ST, dim::ROW> GL,
         ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx,
                                         semaphore &bar, uint16_t cluster_mask,
                                         int dst_mbar_cta) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar,
                                                              cluster_mask, dst_mbar_cta);
}

template<int axis, cache_policy policy, ducks::st::all ST,
         detail::tma_layout_for<ST, axis> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx,
                                         semaphore &bar, uint16_t cluster_mask) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();

    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
                ".mbarrier::complete_tx::bytes.multicast::cluster"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                  "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                  "r"(tma_coords.w), "h"(cluster_mask)
                : "memory"
            );
        } else {
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
                ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                  "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                  "r"(tma_coords.w), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
              "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b),
              "h"(cluster_mask)
            : "memory"
        );
    }
}

template<ducks::st::all ST, detail::tma_layout_for<ST, dim::ROW> GL,
         ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx,
                                         semaphore &bar, uint16_t cluster_mask) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
}

} // namespace cluster

} // namespace tma
} // namespace kittens
