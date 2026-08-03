/**
 * @file
 * @brief Core ThunderKittens data layout/type primitives.
 *
 * This release only exposes the ThunderKittens types used by the five kernels.
 * Keep this header explicit so unused type families do not get pulled into the
 * self-contained release package by broad aggregate headers.
 */
#pragma once

#include <cuda.h>

#include "tk_types_register_rt.cuh"
#include "tk_types_shared_st.cuh"
#include "tk_types_shared_descriptor.cuh"
#include "tk_types_global_util.cuh"
// Two tensor-memory (tcgen05) type sets live in this tree and define the same
// names (kittens::tt, kittens::tensor_allocator). MKERNEL_TCGEN05 (GPU=blackwell,
// gemm_rs/ag_gemm) selects the full ThunderKittens pair; everything else keeps
// the minimal set that came with dispatch_gemm_blackwell.
#ifdef MKERNEL_TCGEN05
#include "tk_types_tensor_tt.cuh"
#include "tk_types_tensor_tensor.cuh"
#else
#include "tk_types_tensor.cuh"
#endif

namespace kittens {

template<typename T>
using row_vec = typename T::row_vec;

template<typename T>
using col_vec = typename T::col_vec;

using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

} // namespace kittens
