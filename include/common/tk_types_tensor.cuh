/**
 * @file
 * @brief Minimal Blackwell tensor-memory types used by BF16 tcgen05 GEMM.
 *
 * This deliberately implements CTA-group 1 only.  Cluster allocation and
 * scaled FP8/FP4 tensor-memory layouts can be added when a kernel needs them.
 */
#pragma once

#include "tk_common_common.cuh"

namespace kittens {

static constexpr int MAX_TENSOR_ROWS = 128;
static constexpr int MAX_TENSOR_COLS = 512;

namespace ducks {
namespace tt {
struct identifier {};
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
template<typename T> concept full = all<T> && T::rows == MAX_TENSOR_ROWS;
} // namespace tt
} // namespace ducks

template<typename _T, int _rows, int _cols>
struct tt {
    using identifier = ducks::tt::identifier;
    using T = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;

    static constexpr int rows = _rows;
    static constexpr int cols = _cols;

    static_assert(rows > 0 && rows <= MAX_TENSOR_ROWS && rows % BASE_TILE_DIM == 0);
    static_assert(cols > 0 && cols <= MAX_TENSOR_COLS && cols % BASE_TILE_DIM == 0);

    uint32_t addr;

    __device__ inline tt() : addr(0) {}
    __device__ inline explicit tt(uint32_t value) : addr(value) {}

    template<ducks::tt::all TT>
    __device__ inline TT subtile(int row_offset, int col_offset) const {
        return TT(addr + (row_offset << 16) +
                  col_offset / (4 / static_cast<uint32_t>(sizeof(T))));
    }

    template<ducks::tt::all TT>
    __device__ inline TT subtile(int col_offset) const {
        return TT(addr + col_offset / (4 / static_cast<uint32_t>(sizeof(T))));
    }
};

template<int Width> using full_tt_fl = tt<float, MAX_TENSOR_ROWS, Width>;

template<int _nblocks_per_sm = 1, int _ncta = 1>
struct tensor_allocator {
    static_assert(_nblocks_per_sm == 1 || _nblocks_per_sm == 2);
    static_assert(_ncta == 1 || _ncta == 2);
    static constexpr int cols =
        ((MAX_TENSOR_COLS / _nblocks_per_sm) / 32) * 32;

    uint32_t addr = 0;

    __device__ inline void set_addr(uint32_t value) { addr = value; }

    // Must be called by one complete warp. shared_addr must live in shared
    // memory and is used to publish the allocated TMEM base to the CTA.
    __device__ inline void provision(uint32_t &shared_addr) {
       if constexpr (_ncta == 1) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;\n"
            ::  "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(cols)
            );
        } else {
            asm volatile(
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
                ::  "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(cols)
            );
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
        }
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }

    template<ducks::tt::full TT>
    __device__ inline TT allocate(int col_offset = 0) const {
        return TT(addr + col_offset);
    }

    // Must be called by one complete warp after every TMEM user has finished.
    __device__ inline void deprovision() {
       if constexpr (_ncta == 1) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1;\n"
            ::  "r"(addr), "n"(cols)
            );
        } else {
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1;\n"
            ::  "r"(addr), "n"(cols)
            );
        }
    }
};

} // namespace kittens
