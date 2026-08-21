/**
 * @file
 * @brief TMA forwarding helpers for dist tensor descriptors.
 *
 * The implementation still uses the ThunderKittens TMA overloads because
 * compute-side shared tile/vector types remain TK-owned. Keeping the call-site
 * spelling under `dist::tma` avoids leaking that dependency into operator code.
 */

#pragma once

#include "distributed_buffer.cuh"

namespace dist {
namespace tma {

using ::kittens::tma::store_async;
using ::kittens::tma::store_add_async;
using ::kittens::tma::load_async;
using ::kittens::tma::store_async_read_wait;
using ::kittens::tma::store_async_wait;
using ::kittens::tma::expect_bytes;
using ::kittens::tma::bulk_load_async;
using ::kittens::tma::bulk_store_async;

} // namespace tma
} // namespace dist
