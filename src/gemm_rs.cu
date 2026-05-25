/**
 * @file gemm_rs.cu
 * @brief Multi-node GEMM + Reduce-Scatter - single fused kernel.
 *
 * Single kernel launch. CTA groups run with role recycling:
 *
 *   Compute CTAs [0, num_comp_sms):
 *     GEMM A@B into per-GPU output tiles. In the fused-compute-intra path,
 *     compute CTAs also contribute their partial tiles directly into the
 *     owning GPU's reduce-scatter staging buffer and publish chunk readiness.
 *
 *   Send CTAs [..., ... + num_send_sms):
 *     Wait until all local GPUs have contributed the chunks owned by this rank,
 *     then coalesce row-block chunks and push them to the peer node through the
 *     D2H FIFO/RDMA path.
 *
 *   Reduce CTAs [..., 132):
 *     Poll peer-node arrival flags, reduce local staging plus remote payloads,
 *     and publish the final reduce-scattered slice. CTAs that finish their
 *     primary role recycle into the same work-stealing reduce pool.
 *
 * Synchronization is device-side through distributed-tensor barriers, ready
 * bitmaps, atomics, and RDMA arrival flags.
 *
 * Infrastructure (config, intra_globals, fused_globals, helpers, host setup,
 * entrypoint_fused, static state, gemm_rs_zero_regions_t) lives in
 *   include/operators/gemm_rs/gemm_rs.cuh
 * Python/session glue + pybind module live in
 *   include/operators/gemm_rs/session.cuh
 */
#include "operators/gemm_rs/gemm_rs.cuh"

namespace gemm_rs_multinode {

template <typename G>
__device__ inline void compute_tile_impl(
    const G &Gv, int row_idx, int col_idx, int ready_idx,
    typename G::pipeline_inputs (&inputs)[G::PIPELINE_STAGES],
    typename G::pipeline_outputs &outputs,
    semaphore (&inputs_arrived)[G::PIPELINE_STAGES],
    semaphore (&inputs_finished)[G::PIPELINE_STAGES],
    semaphore &outputs_arrived, semaphore &outputs_finished,
    int &stage, uint32_t &phasebits,
    int row_blocks, int col_blocks, int num_iters)
{
    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();

    __builtin_assume(row_idx >= 0 && row_idx < row_blocks);
    __builtin_assume(col_idx >= 0 && col_idx < col_blocks);

    if (wg_id == config::NUM_WARPGROUPS - 1) {
        if (w_id == 0 && l_id == 0) {
            // Do not overwrite the shared output tile until the previous
            // store warp has finished consuming it.
            wait(outputs_finished, get_phasebit<1>(phasebits, G::PIPELINE_STAGES));
            update_phasebit<1>(phasebits, G::PIPELINE_STAGES);
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage], sizeof(typename G::pipeline_inputs));
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::load_async(inputs[stage].A[i], Gv.A,
                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                tma::load_async(inputs[stage].B, Gv.B, {red_idx, col_idx}, inputs_arrived[stage]);
                stage = (stage + 1) % G::PIPELINE_STAGES;
            }
        } else if (w_id == 1 && l_id == 0) {
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                tma::store_async(Gv.workspace[Gv.dev_idx], outputs.C[i], {row_idx * 2 + i, col_idx});
            // Also accumulate directly into the owning GPU's chunk-major
            // staging buffer while the tile is still in shared memory.
            {
                const int row_blocks_per_dev_fuse = row_blocks / G::NUM_DEVICES;
                const int owner_dev_idx_fuse = row_idx / row_blocks_per_dev_fuse;
                const int local_row_idx_at_owner_fuse =
                    row_idx - owner_dev_idx_fuse * row_blocks_per_dev_fuse;
                const int col_blocks_local_fuse =
                    (int)(Gv.B.cols() / G::COL_BLOCK);
                const int global_tile_idx_owner_fuse =
                    local_row_idx_at_owner_fuse * col_blocks_local_fuse + col_idx;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::store_add_async(
                        Gv.staging[owner_dev_idx_fuse], outputs.C[i],
                        {2 * global_tile_idx_owner_fuse + i, 0});
                }
            }
            tma::store_async_read_wait();
            signal_ready(Gv.ready, ready_idx);
            arrive(outputs_finished);
        }
    } else {
        rt_fl<G::ROW_BLOCK / 8, G::COL_BLOCK> C_accum;
        warp::zero(C_accum);
        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            warpgroup::mma_AB(C_accum, inputs[stage].A[wg_id], inputs[stage].B);
            warpgroup::mma_async_wait();
            warp::arrive(inputs_finished[stage]);
            stage = (stage + 1) % G::PIPELINE_STAGES;
        }
        group<8>::sync(3);
        warpgroup::store(outputs.C[wg_id], C_accum);
        warpgroup::sync(wg_id + 1);
        warpgroup::arrive(outputs_arrived);
    }
}


// Core fused_comm_tile: takes (row_idx, col_idx, ready_idx) directly.
// ready_idx is the flat index for the ready[] signal (must match compute_tile).
__device__ inline void fused_comm_tile_impl(
    const fused_globals &G, int row_idx, int col_idx, int ready_idx,
    int row_blocks)
{
    const auto &I = G.intra;
    const int row_blocks_per_dev = row_blocks / fused_globals::NUM_DEVICES;
    const int owner_dev_idx = row_idx / row_blocks_per_dev;
    const int local_row_idx = row_idx % row_blocks_per_dev;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    fused_globals::C_tile (&partials)[2] =
        allocator.allocate<fused_globals::C_tile, 2>();

    __shared__ semaphore partials_arrived[2];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) init_semaphore(partials_arrived[i], 0, 1);
    }
    __syncthreads();

    wait_ready(I.ready, ready_idx);

    const int wid = warp::groupid();

    if (wid == 0 && laneid() == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            tma::expect_bytes(partials_arrived[i], sizeof(fused_globals::C_tile));
            tma::load_async(partials[i], I.workspace[I.dev_idx],
                            {row_idx * 2 + i, col_idx}, partials_arrived[i]);
        }
    } else if (wid == 1 && laneid() == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wait(partials_arrived[i], 0);
            // Accumulate this partial tile into the owning GPU's chunk-major
            // staging buffer. Each C_tile maps to one row-tile in that layout.
            const int col_blocks_local =
                (int)(I.B.cols() / fused_globals::COL_BLOCK);
            const int global_tile_idx_owner =
                local_row_idx * col_blocks_local + col_idx;
            tma::store_add_async(I.staging[owner_dev_idx], partials[i],
                                 {2 * global_tile_idx_owner + i, 0});
        }
        tma::store_async_read_wait();
        // The TMA store-add and the downstream barrier signal target the same
        // peer GPU; store_async_read_wait plus a system fence orders them for
        // both peer GPUs and the NIC read path.
        __threadfence_system();

        if (G.rt != nullptr && G.num_send_sms > 0) {
            // Chunk barrier: batch per-tile signals into per-chunk (matches gemm_ar).
            auto &Rt = *G.rt;
            const int chunk_col = col_idx / Rt.chunk_tiles_val;
            const int flat_chunk = row_idx * Rt.chunks_per_row + chunk_col;
            const int chunk_start = chunk_col * Rt.chunk_tiles_val;
            const int tiles_this_chunk = min(Rt.chunk_tiles_val,
                                             Rt.col_blocks_val - chunk_start);
            uint32_t prev = atomicAdd(Rt.comp_chunk_tiles_done + flat_chunk, 1u);
            if (prev == (uint32_t)(tiles_this_chunk - 1)) {
                const dist::coord slot = {1, 1 + row_idx, chunk_col};
                // Distributed tensor barrier memory is coherent over NVLink.
                gemm_rs_signal_barrier_gpu(&I.barrier[owner_dev_idx][slot], 1);
            }
        }
    }
    __syncthreads();
    (void)ready_idx;
}

// Dedicated CTA role: poll dbuf barriers for locally-owned tiles and set
// Tile size in bytes (ROW_BLOCK × COL_BLOCK × sizeof(bf16)).
static constexpr int TILE_BYTES =
    fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK * 2;

// Pack one chunk from row-major local output into tile-major staging.
template <typename G>
__device__ __forceinline__ void gemm_rs_sender_pack_chunk(
    bf16 *staging_buf, const bf16 *output_local, int N,
    int chunk_first_tile, int cols_this_chunk, int rb, int col_start)
{
    constexpr int ELEMS_PER_VEC = 8;
    constexpr int VECS_PER_ROW = G::COL_BLOCK / ELEMS_PER_VEC;
    constexpr int VECS_PER_TILE = G::ROW_BLOCK * VECS_PER_ROW;
    for (int ti = 0; ti < cols_this_chunk; ++ti) {
        bf16 *dst = staging_buf
            + (long)(chunk_first_tile + ti) * (G::ROW_BLOCK * G::COL_BLOCK);
        const bf16 *src = output_local
            + (long)rb * G::ROW_BLOCK * N
            + (long)(col_start + ti) * G::COL_BLOCK;
        for (int v = threadIdx.x; v < VECS_PER_TILE; v += blockDim.x) {
            const int r = v / VECS_PER_ROW;
            const int c = (v % VECS_PER_ROW) * ELEMS_PER_VEC;
            *reinterpret_cast<uint4*>(dst + r * G::COL_BLOCK + c) =
                *reinterpret_cast<const uint4*>(src + r * N + c);
        }
    }
}

// Send CTAs own row blocks in a static stride and post ready chunks as soon as
// their local contributions have arrived. Chunks are claimed through a bitmap,
// so later columns do not wait behind an earlier column in the same row block.
template <typename G>
__device__ inline void send_tiles_coalesced(const G &Gv) {
    if (Gv.rt == nullptr) return;
    const auto &I = Gv.intra;
    auto &Rt = *Gv.rt;
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;

    const int send_base = I.num_comp_sms + I.num_comm_sms;
    const int send_id = (int)blockIdx.x - send_base;
    const int stride = Gv.num_send_sms;

    // Each send CTA scans the chunks for its owned row blocks and atomically
    // clears a ready bit before posting, so every chunk is sent exactly once.
    const int my_rbs = (send_id < row_blocks_per_dev)
        ? ((row_blocks_per_dev - send_id + stride - 1) / stride) : 0;
    const int my_chunks_total = my_rbs * chunks_per_row;
    int posted = 0;
    __shared__ int shared_found_idx;
    while (posted < my_chunks_total) {
        if (threadIdx.x == 0) {
            shared_found_idx = -1;
            const int queue_words = gemm_rs_send_ready_bitmap_words_per_queue(
                row_blocks_per_dev, stride, chunks_per_row);
            const int queue_base =
                gemm_rs_send_ready_bitmap_region_base(row_blocks_per_dev, chunks_per_row)
                + send_id * queue_words;
            while (shared_found_idx < 0) {
                for (int w = 0; w < queue_words; ++w) {
                    unsigned int *word_ptr = reinterpret_cast<unsigned int*>(
                        &((*Rt.ready_chunk)[I.dev_idx][{0, 0, 0, queue_base + w}]));
                    uint32_t old = gemm_rs_acquire_load_u32(word_ptr);
                    while (old != 0u) {
                        const int bit_idx = __ffs((int)old) - 1;
                        const uint32_t bit = 1u << bit_idx;
                        const uint32_t desired = old & ~bit;
                        const uint32_t prev = atomicCAS(word_ptr, old, desired);
                        if (prev == old) {
                            const int idx = (w << 5) + bit_idx;
                            if (idx < my_chunks_total) {
                                shared_found_idx = idx;
                                break;
                            }
                            old = desired;
                        } else {
                            old = prev;
                        }
                    }
                    if (shared_found_idx >= 0) {
                        break;
                    }
                }
                if (shared_found_idx < 0) {
                    __nanosleep(50);
                }
            }
        }
        __syncthreads();
        const int idx = shared_found_idx;
        const int rb_k = idx / chunks_per_row;
        const int ci = idx - rb_k * chunks_per_row;
        const int rb = send_id + rb_k * stride;
        const int first_tile_rb = rb * col_blocks;
        const int first_chunk_rb = rb * chunks_per_row;
        const int col_start = ci * chunk_tiles;
        const int cols_this_chunk = min(chunk_tiles, col_blocks - col_start);
        const int chunk_first_tile = first_tile_rb + col_start;
        const int global_chunk_id = first_chunk_rb + ci;

        if (threadIdx.x == 0) {
            if (Rt.use_incremental_peer_reduce != 0) {
                __threadfence_system();
            }
            gemm_rs_release_store_u32(Rt.sender_done + global_chunk_id, 1u);

            const uint32_t chunk_bytes = (uint32_t)((long)cols_this_chunk * TILE_BYTES);
            const uint32_t offset = (uint32_t)((long)chunk_first_tile * TILE_BYTES);
            // Per-peer slot offsets: bit-identical at N == 2 (sap == 0).
            // single_peer_bytes / tiles = local scratch sized for one peer.
            const long single_peer_bytes =
                (long)row_blocks_per_dev * (long)col_blocks * TILE_BYTES;
            const int  single_peer_tiles = row_blocks_per_dev * col_blocks;
            const int n_peers = Rt.num_nodes - 1;
            const int queues_per_peer =
                max(1, Rt.num_remote_queues / max(1, n_peers));
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    Rt.node_idx, Rt.num_nodes, peer_slot);
                if (Rt.use_receiver_owner_rs != 0) {
                    const int owner_node = global_chunk_id % Rt.num_nodes;
                    if (owner_node == Rt.node_idx || peer_rank != owner_node) {
                        continue;
                    }
                }
                const int sap = internode::slot_at_peer(Rt.node_idx, peer_rank, Rt.num_nodes);
                const int logical_q = (rb * chunks_per_row + ci) % queues_per_peer;
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint32_t)(sap * single_peer_tiles + chunk_first_tile);
                cmd.bytes    = chunk_bytes;
                cmd.local_offset = offset;
                cmd.remote_offset = (uint64_t)((long)sap * single_peer_bytes) + offset;
                cmd.lane_id  = (uint16_t)(
                    Rt.use_transport_arrival_queue != 0
                        ? (peer_slot * queues_per_peer + logical_q)
                        : (rb * chunks_per_row + ci));
                cmd.reserved0 = (uint8_t)(peer_slot * fused_globals::NUM_DEVICES + I.dev_idx);
                __threadfence();
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(
                        Rt.d2h_fifos, (uint32_t)cmd.lane_id);
                fifo.push(cmd);
            }
        }
        __syncthreads();
        posted++;
    }
}

__device__ __forceinline__ bool gemm_rs_receiver_owns_chunk(
    const fused_globals::runtime_state &Rt, int chunk_id
) {
    return Rt.use_receiver_owner_rs == 0 || (chunk_id % Rt.num_nodes) == Rt.node_idx;
}

template <typename G>
__device__ inline void gemm_rs_enqueue_peer_accum_work(
    typename G::runtime_state &Rt, int peer_slot, int chunk_id
);

template <typename G>
__device__ inline bool gemm_rs_drain_arrival_queue_publish_flags(
    typename G::runtime_state &Rt, int q
) {
    if (q < 0 || q >= Rt.num_remote_queues) return false;
    bool any = false;
    while (true) {
        uint32_t q_head = gemm_rs_acquire_load_u32(Rt.arrival_queue_head + q);
        if ((int)q_head >= Rt.remote_queue_stride) break;
        const uint32_t flag_val = comm::atomic_u32::volatile_load(
            &Rt.arrival_flags[q * Rt.remote_queue_stride + q_head]);
        if (flag_val == 0u) break;
        const uint32_t claimed = atomicCAS(
            reinterpret_cast<unsigned int*>(Rt.arrival_queue_head + q),
            q_head, q_head + 1u);
        if (claimed != q_head) continue;

        int first_tile = (int)internode::unpack_arrival_first_tile(flag_val);
        if (first_tile < 0) first_tile = 0;
        const int single_peer_tiles = Rt.row_blocks_per_slice * Rt.col_blocks_val;
        const int peer_slot = first_tile / single_peer_tiles;
        const int local_first_tile = first_tile - peer_slot * single_peer_tiles;
        const int rb = local_first_tile / Rt.col_blocks_val;
        int col_start = local_first_tile - rb * Rt.col_blocks_val;
        int work_tiles = (int)internode::unpack_arrival_num_tiles(flag_val);
        if (work_tiles < 1) work_tiles = 1;
        work_tiles = min(work_tiles, Rt.col_blocks_val - col_start);

        // Data WR precedes the packed arrival write on the same QP. Fence before
        // publishing per-chunk readiness consumed by reducer CTAs.
        __threadfence_system();
        while (work_tiles > 0 && col_start < Rt.col_blocks_val) {
            const int chunk_col = col_start / Rt.chunk_tiles_val;
            const int chunk_id = rb * Rt.chunks_per_row + chunk_col;
            const int chunk_start = chunk_col * Rt.chunk_tiles_val;
            const int tiles_this_chunk = min(
                Rt.chunk_tiles_val, Rt.col_blocks_val - chunk_start);
            const int consumed_tiles = min(work_tiles, tiles_this_chunk);
            if (consumed_tiles <= 0) break;
            if (!gemm_rs_receiver_owns_chunk(Rt, chunk_id)) {
                col_start += consumed_tiles;
                work_tiles -= consumed_tiles;
                continue;
            }
            const uint32_t peer_bit =
                (peer_slot >= 0 && peer_slot < 31) ? (1u << peer_slot) : 0u;
            const uint32_t old_mask = atomicOr(
                reinterpret_cast<unsigned int*>(Rt.remote_arrived_peer_mask + chunk_id),
                peer_bit);
            const uint32_t new_mask = old_mask | peer_bit;
            const int needed_peers = Rt.num_nodes - 1;
            if (Rt.use_incremental_peer_reduce != 0) {
                if ((old_mask & peer_bit) == 0u) {
                    gemm_rs_enqueue_peer_accum_work<G>(Rt, peer_slot, chunk_id);
                }
            } else if (__popc(old_mask) < needed_peers && __popc(new_mask) >= needed_peers) {
                atomicAdd(Rt.remote_arrived_chunks, 1u);
                gemm_rs_release_store_u32(Rt.remote_arrived_flag + chunk_id, 1u);
            }
            col_start += consumed_tiles;
            work_tiles -= consumed_tiles;
        }
        any = true;
    }
    return any;
}

template <typename G>
__device__ inline void gemm_rs_recv_progress_once(
    typename G::runtime_state &Rt, int progress_id, int progress_stride
) {
    if (Rt.use_transport_arrival_queue == 0) return;
    if (progress_id < 0) return;
    const int stride = max(1, progress_stride);
    for (int q = progress_id; q < Rt.num_remote_queues; q += stride) {
        gemm_rs_drain_arrival_queue_publish_flags<G>(Rt, q);
    }
}

template <typename G>
__device__ inline void gemm_rs_recv_progress_loop(
    const G &Gv, int progress_id, int progress_stride, int total_chunks
) {
    auto &Rt = *Gv.rt;
    if (threadIdx.x == 0) {
        const int done_target = Rt.use_receiver_owner_rs != 0
            ? Rt.owner_chunks_total : total_chunks;
        while ((int)gemm_rs_acquire_load_u32(Rt.chunks_processed) < done_target) {
            gemm_rs_recv_progress_once<G>(Rt, progress_id, progress_stride);
            __nanosleep(50);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
    }
}

__device__ __forceinline__ uint32_t gemm_rs_encode_peer_accum_work(
    int peer_slot, int chunk_id
) {
    return ((uint32_t)(peer_slot & 0x7f) << 24) | (uint32_t)(chunk_id + 1);
}

__device__ __forceinline__ void gemm_rs_decode_peer_accum_work(
    uint32_t encoded, int& peer_slot, int& chunk_id
) {
    peer_slot = (int)((encoded >> 24) & 0x7f);
    chunk_id = (int)(encoded & 0x00ffffffu) - 1;
}

template <typename G>
__device__ inline void gemm_rs_enqueue_peer_accum_work(
    typename G::runtime_state &Rt, int peer_slot, int chunk_id
) {
    const int total_chunks = Rt.row_blocks_per_slice * Rt.chunks_per_row;
    const int n_peers = Rt.num_nodes - 1;
    const uint32_t slot = atomicAdd(Rt.peer_accum_tail, 1u);
    const uint32_t cap = (uint32_t)(total_chunks * max(1, n_peers));
    if (slot < cap) {
        gemm_rs_release_store_u32(
            Rt.peer_accum_queue + slot,
            gemm_rs_encode_peer_accum_work(peer_slot, chunk_id));
    }
}

template <typename G>
__device__ inline bool gemm_rs_process_peer_accum_work(
    const G &Gv
) {
    auto &Rt = *Gv.rt;
    __shared__ uint32_t s_encoded_work;
    if (threadIdx.x == 0) {
        s_encoded_work = 0u;
        const uint32_t head = gemm_rs_acquire_load_u32(Rt.peer_accum_head);
        const uint32_t tail = gemm_rs_acquire_load_u32(Rt.peer_accum_tail);
        if (head < tail) {
            const uint32_t old = atomicCAS(
                reinterpret_cast<unsigned int*>(Rt.peer_accum_head), head, head + 1u);
            if (old == head) {
                uint32_t encoded = 0u;
                while (encoded == 0u) {
                    encoded = gemm_rs_acquire_load_u32(Rt.peer_accum_queue + head);
                    if (encoded == 0u) __nanosleep(32);
                }
                s_encoded_work = encoded;
            }
        }
    }
    __syncthreads();
    const uint32_t encoded = s_encoded_work;
    if (encoded == 0u) return false;

    int peer_slot = 0;
    int chunk_id = -1;
    gemm_rs_decode_peer_accum_work(encoded, peer_slot, chunk_id);
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;
    const int total_chunks = row_blocks_per_dev * chunks_per_row;
    const int n_peers = Rt.num_nodes - 1;
    if (chunk_id < 0 || chunk_id >= total_chunks ||
        peer_slot < 0 || peer_slot >= n_peers) {
        return true;
    }
    if (!gemm_rs_receiver_owns_chunk(Rt, chunk_id)) {
        return true;
    }

    // Arrival queue draining and peer accumulation can run on different CTAs.
    // Re-fence on the consumer side before reading RDMA-written recv_buf data.
    __threadfence_system();

    while (gemm_rs_acquire_load_u32(Rt.sender_done + chunk_id) == 0u) {
        __nanosleep(32);
    }

    const int rb = chunk_id / chunks_per_row;
    const int ci = chunk_id % chunks_per_row;
    const int col_start = ci * chunk_tiles;
    const int cols_this_chunk = min(chunk_tiles, col_blocks - col_start);
    const int first_tile = rb * col_blocks + col_start;
    const int total_elems = G::ROW_BLOCK * G::COL_BLOCK;

    if (threadIdx.x == 0) {
        while (atomicCAS(
            reinterpret_cast<unsigned int*>(Rt.chunk_accum_lock + chunk_id),
            0u, 1u) != 0u) {
            __nanosleep(32);
        }
    }
    __syncthreads();

    constexpr int ELEMS_PER_VEC = 8;
    constexpr int VECS_PER_ROW = G::COL_BLOCK / ELEMS_PER_VEC;
    constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;
    const int vec_lane = threadIdx.x % VECS_PER_ROW;
    const int row_lane = threadIdx.x / VECS_PER_ROW;
    const int col_elem = vec_lane * ELEMS_PER_VEC;
    const int single_peer_tiles = row_blocks_per_dev * col_blocks;
    for (int ti = 0; ti < cols_this_chunk; ++ti) {
        for (int r = row_lane; r < G::ROW_BLOCK; r += ROWS_PER_WAVE) {
            bf16 *accum = Rt.staging_buf
                + (long)(first_tile + ti) * total_elems
                + (long)r * G::COL_BLOCK + col_elem;
            const bf16 *peer_recv = Rt.recv_buf
                + ((long)peer_slot * single_peer_tiles + first_tile + ti)
                    * total_elems
                + (long)r * G::COL_BLOCK + col_elem;
            const uint4 av = *reinterpret_cast<const uint4*>(accum);
            const uint4 rv = *reinterpret_cast<const uint4*>(peer_recv);
            __nv_bfloat162 o0 = __hadd2(
                *reinterpret_cast<const __nv_bfloat162*>(&av.x),
                *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
            __nv_bfloat162 o1 = __hadd2(
                *reinterpret_cast<const __nv_bfloat162*>(&av.y),
                *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
            __nv_bfloat162 o2 = __hadd2(
                *reinterpret_cast<const __nv_bfloat162*>(&av.z),
                *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
            __nv_bfloat162 o3 = __hadd2(
                *reinterpret_cast<const __nv_bfloat162*>(&av.w),
                *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
            uint4 ov;
            ov.x = *reinterpret_cast<unsigned int*>(&o0);
            ov.y = *reinterpret_cast<unsigned int*>(&o1);
            ov.z = *reinterpret_cast<unsigned int*>(&o2);
            ov.w = *reinterpret_cast<unsigned int*>(&o3);
            *reinterpret_cast<uint4*>(accum) = ov;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        gemm_rs_release_store_u32(Rt.chunk_accum_lock + chunk_id, 0u);
        const uint32_t done =
            atomicAdd(Rt.peer_accum_done_count + chunk_id, 1u) + 1u;
        if (done == (uint32_t)n_peers) {
            gemm_rs_release_store_u32(Rt.chunk_reduce_done + chunk_id, 1u);
            const uint32_t slot = atomicAdd(Rt.ready_reduce_tail, 1u);
            if (slot < (uint32_t)total_chunks) {
                gemm_rs_release_store_u32(Rt.ready_reduce_queue + slot,
                                          (uint32_t)(chunk_id + 1));
            }
        }
    }
    __syncthreads();
    return true;
}

template <typename G>
__device__ inline bool gemm_rs_chunk_prereqs_ready(
    typename G::runtime_state &Rt, int chunk_id
) {
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;
    const int rb = chunk_id / chunks_per_row;
    const int ci = chunk_id % chunks_per_row;
    const int col_start = ci * chunk_tiles;
    const int first_tile = rb * col_blocks + col_start;
    if (gemm_rs_acquire_load_u32(Rt.sender_done + chunk_id) == 0u) return false;
    if (Rt.use_transport_arrival_queue != 0) {
        return gemm_rs_acquire_load_u32(Rt.remote_arrived_flag + chunk_id) == 1u;
    }
    const int single_peer_tiles = row_blocks_per_dev * col_blocks;
    const int n_peers = Rt.num_nodes - 1;
    for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
        volatile uint32_t* flag_ptr =
            &Rt.arrival_flags[peer_slot * single_peer_tiles + first_tile];
        if (gemm_rs_poll_arrival_acquire(flag_ptr) != Rt.epoch) return false;
    }
    return true;
}

template <typename G>
__device__ inline void gemm_rs_ready_reduce_queue_progress(
    typename G::runtime_state &Rt, int total_chunks
) {
    constexpr int kScansPerProgress = 8;
    if (Rt.use_incremental_peer_reduce != 0) {
        gemm_rs_recv_progress_once<G>(Rt, 0, 1);
        return;
    }
    for (int i = 0; i < kScansPerProgress; ++i) {
        const uint32_t scan = atomicAdd(Rt.ready_reduce_scan, 1u);
        const int chunk_id = (int)(scan % (uint32_t)total_chunks);
        const uint32_t state = gemm_rs_acquire_load_u32(Rt.remote_arrived_flag + chunk_id);
        if ((Rt.use_transport_arrival_queue != 0 && state == 2u) ||
            (Rt.use_transport_arrival_queue == 0 && state != 0u)) {
            continue;
        }
        gemm_rs_recv_progress_once<G>(Rt, (int)(scan % (uint32_t)max(1, Rt.num_recv_progress_sms)),
                                      max(1, Rt.num_recv_progress_sms));
        if (!gemm_rs_chunk_prereqs_ready<G>(Rt, chunk_id)) continue;
        const uint32_t expected = (Rt.use_transport_arrival_queue != 0) ? 1u : 0u;
        const uint32_t old = atomicCAS(
            reinterpret_cast<unsigned int*>(Rt.remote_arrived_flag + chunk_id),
            expected, 2u);
        if (old != expected) continue;
        const uint32_t slot = atomicAdd(Rt.ready_reduce_tail, 1u);
        if (slot < (uint32_t)total_chunks) {
            gemm_rs_release_store_u32(Rt.ready_reduce_queue + slot,
                                      (uint32_t)(chunk_id + 1));
        }
    }
}

template <typename G>
__device__ inline int gemm_rs_ready_reduce_queue_pop(
    typename G::runtime_state &Rt
) {
    while (true) {
        const uint32_t head = gemm_rs_acquire_load_u32(Rt.ready_reduce_head);
        const uint32_t tail = gemm_rs_acquire_load_u32(Rt.ready_reduce_tail);
        if (head >= tail) return -1;
        const uint32_t old = atomicCAS(
            reinterpret_cast<unsigned int*>(Rt.ready_reduce_head), head, head + 1u);
        if (old != head) continue;
        uint32_t encoded = 0u;
        while (encoded == 0u) {
            encoded = gemm_rs_acquire_load_u32(Rt.ready_reduce_queue + head);
            if (encoded == 0u) __nanosleep(32);
        }
        return (int)encoded - 1;
    }
}

template <typename G>
__device__ inline void reduce_tiles_ws(const G &Gv) {
    if (Gv.rt == nullptr) return;
    auto &Rt = *Gv.rt;
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int total_elems = G::ROW_BLOCK * G::COL_BLOCK;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;
    const int total_chunks = row_blocks_per_dev * chunks_per_row;
    const int done_target = Rt.use_receiver_owner_rs != 0
        ? Rt.owner_chunks_total : total_chunks;

    while (true) {
        __shared__ int _ws_red_chunk;
        if (Rt.use_incremental_peer_reduce != 0) {
            while (true) {
                if (threadIdx.x == 0) {
                    _ws_red_chunk = -1;
                    gemm_rs_recv_progress_once<G>(Rt, 0, 1);
                    const int ready_chunk = gemm_rs_ready_reduce_queue_pop<G>(Rt);
                    if (ready_chunk >= 0) {
                        _ws_red_chunk = ready_chunk;
                    } else if ((int)gemm_rs_acquire_load_u32(Rt.chunks_processed) >= done_target) {
                        _ws_red_chunk = -2;
                    }
                }
                __syncthreads();
                if (_ws_red_chunk >= 0 || _ws_red_chunk == -2) break;
                const bool did_peer_accum = gemm_rs_process_peer_accum_work<G>(Gv);
                if (!did_peer_accum && threadIdx.x == 0) {
                    __nanosleep(Rt.reduce_poll_sleep_ns);
                }
                __syncthreads();
            }
        } else if (threadIdx.x == 0) {
            if (Rt.use_ready_reduce_queue != 0) {
                _ws_red_chunk = -1;
                while (_ws_red_chunk < 0) {
                    gemm_rs_ready_reduce_queue_progress<G>(Rt, total_chunks);
                    _ws_red_chunk = gemm_rs_ready_reduce_queue_pop<G>(Rt);
                    if (_ws_red_chunk >= 0) break;
                    if ((int)gemm_rs_acquire_load_u32(Rt.chunks_processed) >= done_target) {
                        break;
                    }
                    __nanosleep(Rt.reduce_poll_sleep_ns);
                }
            } else {
                _ws_red_chunk = (int)atomicAdd(Rt.next_reduce, 1u);
            }
        }
        __syncthreads();
        const int chunk_id = _ws_red_chunk;
        if (chunk_id >= total_chunks) break;
        if (chunk_id < 0) break;
        if (!gemm_rs_receiver_owns_chunk(Rt, chunk_id)) {
            continue;
        }

        const int rb = chunk_id / chunks_per_row;
        const int ci = chunk_id % chunks_per_row;
        const int col_start = ci * chunk_tiles;
        const int cols_this_chunk = min(chunk_tiles, col_blocks - col_start);
        const int first_tile = rb * col_blocks + col_start;


        // Wait for both prerequisites in parallel: thread 0 tracks the remote
        // RDMA arrival, while warp 1 lane 0 tracks local sender completion.
        // Keeping the polls in separate warps avoids SIMT serialization.
        if (Rt.use_incremental_peer_reduce != 0 && threadIdx.x == 0) {
            while (gemm_rs_acquire_load_u32(Rt.chunk_reduce_done + chunk_id) == 0u) {
                gemm_rs_recv_progress_once<G>(Rt, 0, 1);
                __nanosleep(Rt.reduce_poll_sleep_ns);
            }
        } else if (threadIdx.x == 0) {
            if (gemm_rs_acquire_load_u32(Rt.remote_arrived_flag + chunk_id) == 0u) {
                const uint32_t sleep_ns = Rt.reduce_poll_sleep_ns;
                constexpr int kTightSpin = 64;
                const int single_peer_tiles = row_blocks_per_dev * col_blocks;
                const int n_peers = Rt.num_nodes - 1;
                for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                    volatile uint32_t* flag_ptr =
                        &Rt.arrival_flags[peer_slot * single_peer_tiles + first_tile];
                    int spin = 0;
                    if (Rt.use_acquire_poll) {
                        while (gemm_rs_poll_arrival_acquire(flag_ptr) != Rt.epoch) {
                            if (spin < kTightSpin) { ++spin; }
                            else { __nanosleep(sleep_ns); }
                        }
                    } else {
                        while (gemm_rs_poll_arrival_relaxed(flag_ptr) != Rt.epoch) {
                            if (spin < kTightSpin) { ++spin; }
                            else { __nanosleep(sleep_ns); }
                        }
                    }
                }
                // Publish a same-GPU flag for any other reducer CTA that later
                // claims this chunk.
                __threadfence_system();
                gemm_rs_release_store_u32(Rt.remote_arrived_flag + chunk_id, 1u);
            }
        } else if (Rt.use_incremental_peer_reduce == 0 && threadIdx.x == 32) {
            while (gemm_rs_acquire_load_u32(Rt.sender_done + chunk_id) == 0u) {
                __nanosleep(32);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
        }

        constexpr int ELEMS_PER_VEC = 8;
        constexpr int VECS_PER_ROW = G::COL_BLOCK / ELEMS_PER_VEC;
        constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;
        const int vec_lane = threadIdx.x % VECS_PER_ROW;
        const int row_lane = threadIdx.x / VECS_PER_ROW;
        const int col_elem = vec_lane * ELEMS_PER_VEC;
        for (int ti = 0; ti < cols_this_chunk; ++ti) {
            for (int r = row_lane; r < G::ROW_BLOCK; r += ROWS_PER_WAVE) {
                const int si = (rb * G::ROW_BLOCK + r) * Rt.N
                             + (col_start + ti) * G::COL_BLOCK + col_elem;
                // Local and remote chunks are both tile-major; write the final
                // reduced values back to row-major output.
                const bf16 *local_src = Rt.staging_buf
                    + (long)(first_tile + ti) * total_elems
                    + (long)r * G::COL_BLOCK + col_elem;
                const uint4 lv = *reinterpret_cast<const uint4*>(local_src);
                __nv_bfloat162 o0 = *reinterpret_cast<const __nv_bfloat162*>(&lv.x);
                __nv_bfloat162 o1 = *reinterpret_cast<const __nv_bfloat162*>(&lv.y);
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&lv.z);
                __nv_bfloat162 o3 = *reinterpret_cast<const __nv_bfloat162*>(&lv.w);
                if (Rt.use_incremental_peer_reduce == 0) {
                    const int single_peer_tiles = row_blocks_per_dev * col_blocks;
                    const int n_peers = Rt.num_nodes - 1;
                    for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                        const bf16 *peer_recv = Rt.recv_buf
                            + ((long)peer_slot * single_peer_tiles + first_tile + ti)
                                * total_elems
                            + (long)r * G::COL_BLOCK + col_elem;
                        const uint4 rv = *reinterpret_cast<const uint4*>(peer_recv);
                        o0 = __hadd2(o0, *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
                        o1 = __hadd2(o1, *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
                        o2 = __hadd2(o2, *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
                        o3 = __hadd2(o3, *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
                    }
                }
                uint4 ov;
                ov.x = *reinterpret_cast<unsigned int*>(&o0);
                ov.y = *reinterpret_cast<unsigned int*>(&o1);
                ov.z = *reinterpret_cast<unsigned int*>(&o2);
                ov.w = *reinterpret_cast<unsigned int*>(&o3);
                *reinterpret_cast<uint4*>(Rt.output_local + si) = ov;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (Rt.use_ready_reduce_queue != 0) {
                atomicAdd(Rt.chunks_processed, 1u);
            }
        }
    }

}




__device__ inline void fused_kernel(const fused_globals &G) {
    const auto &I = G.intra;
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    intra_globals::pipeline_inputs (&inputs)[intra_globals::PIPELINE_STAGES] =
        allocator.allocate<intra_globals::pipeline_inputs, intra_globals::PIPELINE_STAGES>();
    intra_globals::pipeline_outputs &outputs =
        *reinterpret_cast<intra_globals::pipeline_outputs *>(
            &inputs[intra_globals::PIPELINE_STAGES - 1]);

    __shared__ semaphore inputs_arrived[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < intra_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();



    const int row_blocks = I.A.rows() / intra_globals::ROW_BLOCK;
    const int col_blocks = I.B.cols() / intra_globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int num_iters = I.A.cols() / intra_globals::RED_BLOCK;

    const int row_blocks_per_slice = row_blocks / intra_globals::NUM_DEVICES;

    // CTA role dispatch with recycling:
    // - Compute: static stride claiming
    // - Intra-RS: work-stealing
    // - Send: static row-block ownership + coalesced RDMA
    // - Reduce: work-stealing with remote_arrived_flag
    // All roles recycle to reduce after primary role completes.
    if ((int)blockIdx.x < I.num_comp_sms) {
        // Phase bits persist across tasks because the pipeline semaphores are
        // shared CTA state.
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        for (int task_id = (int)blockIdx.x; task_id < num_blocks; task_id += I.num_comp_sms) {
            int row_idx, col_idx;
            gemm_rs_decode_comp_task<intra_globals>(task_id, row_blocks_per_slice,
                                               col_blocks, I.dev_idx,
                                               row_idx, col_idx);
            const int ready_idx = row_idx * col_blocks + col_idx;
            compute_tile_impl<intra_globals>(I, row_idx, col_idx, ready_idx,
                                              inputs, outputs, inputs_arrived, inputs_finished,
                                              outputs_arrived, outputs_finished, stage, phasebits,
                                              row_blocks, col_blocks, num_iters);
            // Compute-side chunk-ready signal. When this GPU finishes all
            // tiles in a chunk, emit a "I'm done contributing" signal so the
            // owner's send CTA knows this GPU's partials have landed in
            // staging. Counter is local atomic; the signal itself is either a
            // cross-device barrier signal (default) or a local release-store
            // into a multicast flag (GEMM_RS_READY_VIA_MULTIMEM).
            // With GEMM_RS_FUSE_COMPUTE_INTRA, the compute CTA has also just
            // issued the peer atomic-add into staging_buf — so this signal
            // correctly marks "all tile atomic-adds for this chunk issued."
            if (G.rt != nullptr && G.num_send_sms > 0 && threadIdx.x == 0) {
                auto &Rt = *G.rt;
                const int chunk_col = col_idx / Rt.chunk_tiles_val;
                const int flat_chunk = row_idx * Rt.chunks_per_row + chunk_col;
                const int chunk_start = chunk_col * Rt.chunk_tiles_val;
                const int tiles_this_chunk = min(Rt.chunk_tiles_val,
                                                 Rt.col_blocks_val - chunk_start);
                uint32_t prev = atomicAdd(Rt.comp_gemm_done + flat_chunk, 1u);
                if (prev == (uint32_t)(tiles_this_chunk - 1)) {
                    __threadfence_system();
                    const int owner_dev_idx =
                        row_idx / (row_blocks / intra_globals::NUM_DEVICES);
                    const int local_row_idx_at_owner =
                        row_idx - owner_dev_idx * Rt.row_blocks_per_slice;
                    const int owner_send_id =
                        local_row_idx_at_owner % G.num_send_sms;
                    const int owner_rb_in_queue =
                        local_row_idx_at_owner / G.num_send_sms;
                    const int queue_idx =
                        owner_rb_in_queue * Rt.chunks_per_row + chunk_col;
                    auto *owner_ready_count = reinterpret_cast<unsigned int*>(
                        &((*Rt.ready_chunk)[owner_dev_idx][{0, 0, 0, flat_chunk}]));
                    uint32_t ready_prev = atomicAdd(owner_ready_count, 1u);
                    if (ready_prev == (uint32_t)(intra_globals::NUM_DEVICES - 1)) {
                        const int queue_words = gemm_rs_send_ready_bitmap_words_per_queue(
                            Rt.row_blocks_per_slice, G.num_send_sms, Rt.chunks_per_row);
                        const int queue_base = gemm_rs_send_ready_bitmap_region_base(
                            Rt.row_blocks_per_slice, Rt.chunks_per_row);
                        const int word_idx = queue_idx >> 5;
                        const uint32_t bit = 1u << (queue_idx & 31);
                        auto *owner_ready_word = reinterpret_cast<unsigned int*>(
                            &((*Rt.ready_chunk)[owner_dev_idx][
                                {0, 0, 0, queue_base + owner_send_id * queue_words + word_idx}]));
                        atomicOr(owner_ready_word, bit);
                    }
                }
            }
        }
        // Reuse the CTA for any remaining reduce chunks.
        reduce_tiles_ws<fused_globals>(G);
    } else if ((int)blockIdx.x < I.num_comp_sms + I.num_comm_sms + G.num_send_sms) {
        // Primary: inter-node send (static row-block ownership, coalesced RDMA)
        send_tiles_coalesced<fused_globals>(G);
        // Reuse the CTA for any remaining reduce chunks.
        reduce_tiles_ws<fused_globals>(G);
    } else {
        // Dedicated reduce CTAs: work-stealing from the start
        const int reduce_base = I.num_comp_sms + I.num_comm_sms + G.num_send_sms;
        const int reduce_id = (int)blockIdx.x - reduce_base;
        if (G.rt != nullptr && G.rt->use_transport_arrival_queue != 0 &&
            reduce_id >= 0 && reduce_id < G.rt->num_recv_progress_sms) {
            const int total_chunks = G.rt->row_blocks_per_slice * G.rt->chunks_per_row;
            gemm_rs_recv_progress_loop<fused_globals>(
                G, reduce_id, G.rt->num_recv_progress_sms, total_chunks);
        } else {
            reduce_tiles_ws<fused_globals>(G);
        }
    }

    // Path taken when we intentionally launch only the compute + intra-node
    // CTAs (no inter-node send/reduce), e.g. for debugging or reuse checks.
    if ((int)gridDim.x == I.num_comp_sms + I.num_comm_sms && threadIdx.x == 0) {
        __threadfence();
        unsigned int prev = atomicAdd(I.kernel_done, 1u);
        if (prev + 1 == (unsigned int)gridDim.x) {
            atomicExch(I.kernel_done, 0u);
            barrier_all(I.barrier, {0, 0, 0}, I.dev_idx);
        }
    }
}

__global__ void gemm_rs_fused_zero_kernel(gemm_rs_zero_regions_t regs) {
    const int rid = blockIdx.x;
    if (rid >= regs.n) return;
    unsigned int* p = reinterpret_cast<unsigned int*>(regs.ptrs[rid]);
    const size_t words = regs.bytes[rid] / sizeof(unsigned int);
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    for (size_t i = tid; i < words; i += nthr) {
        p[i] = 0u;
    }
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void gemm_rs_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

// Launch wrapper stays in this TU so the kernel body stays out of the .cuh.
void launch_fused_gemm_rs(const fused_globals& G, unsigned int active_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    const unsigned int grid = (active_sms == 0u)
        ? (unsigned int)config::NUM_BLOCKS
        : active_sms;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        gemm_rs_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    gemm_rs_fused_kernel_stub<<<grid, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}

}  // namespace gemm_rs_multinode

#include "operators/gemm_rs/session.cuh"
