/**
 * @file dynamic_sm_allocation_utils.cuh
 * @brief Shared device utilities for the dynamic SM allocation kernels.
 *
 * Contains decode_task_id (super-tile task-ID to row/col mapping), used by the
 * GEMM+AR and GEMM+RS kernels and their static variants.
 */
#pragma once

/**
 * Decode a linear task_id into (row_idx, col_idx) tile coordinates.
 *
 * Tasks in the "super" region (first super_rows * col_blocks tasks) use a
 * super-tile interleaving pattern of width Globals::SUPER_M.  Remaining
 * "final" tasks use a simple row-major layout over the leftover rows.
 *
 * @tparam Globals  The kernel's globals struct (must expose SUPER_M).
 */
template <typename Globals>
__device__ inline void decode_task_id(
    const int task_id,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks,
    int &row_idx,
    int &col_idx
) {
    if (task_id < super_rows * col_blocks) {
        row_idx = Globals::SUPER_M * (task_id / super_blocks) +
                  task_id % Globals::SUPER_M;
        col_idx = (task_id % super_blocks) / Globals::SUPER_M;
    } else {
        const int remainder_id = task_id - super_rows * col_blocks;
        row_idx = super_rows + remainder_id % final_rows;
        col_idx = remainder_id / final_rows;
    }
}
