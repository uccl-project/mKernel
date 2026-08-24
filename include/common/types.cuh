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
#include "tk_types_tensor.cuh"
#include "tk_types_global_util.cuh"
#include "tk_types_tensor.cuh"

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
