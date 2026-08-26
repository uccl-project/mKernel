/**
 * @file dispatch_fc1_mega_blackwell_profile.cu
 * @brief Instrumented persistent dispatch + BF16 FC1 for Blackwell.
 *
 * This is intentionally a standalone kernel implementation. Keep algorithmic
 * changes synchronized with dispatch_fc1_mega_blackwell.cu.
 */

#include "operators/dispatch_fc1_mega_blackwell/dispatch_fc1_mega_blackwell.cuh"

namespace moe_dispatch_fc1_mega_blackwell {

__device__ __forceinline__ void wait_epoch(int *ptr, uint32_t target) {
    uint32_t value = comm::atomic_u32::acquire_load_gpu(ptr);
    while (value != target) {
        __nanosleep(64);
        value = comm::atomic_u32::acquire_load_gpu(ptr);
    }
}

template <typename DispatchBuffers>
__device__ inline void dispatch_warpgroup(
    const mega_globals &G,
    DispatchBuffers &dispatch_buffers,
    semaphore (&dispatch_arrived)[mega_globals::NUM_DISPATCH_WARPS]
#ifdef PROFILE_TIMINGS
    , uint64_t (&work_done_timestamps)[mega_globals::NUM_DISPATCH_WARPS]
    , uint32_t *timing_head
#endif
) {
    constexpr int NUM_CHUNKS =
        mega_globals::H * static_cast<int>(sizeof(bf16)) /
        mega_globals::PULL_BYTES;
    constexpr int UINT4S_PER_CHUNK =
        mega_globals::PULL_BYTES / static_cast<int>(sizeof(uint4));

    const int warp_idx = warpgroup::warpid();
    const int lane = warp::laneid();
    uint32_t barrier_phase = 0;

    const int dispatch_group_count =
        G.num_sms / mega_globals::DISPATCH_CTAS_PER_BLOCK;
    const int num_dispatch_ctas =
        dispatch_group_count * mega_globals::DISPATCH_CTAS_PER_BLOCK;
    if (static_cast<int>(blockIdx.x) >= num_dispatch_ctas)
        return;
    const int dispatch_cta_rank =
        static_cast<int>(blockIdx.x) %
        mega_globals::DISPATCH_CTAS_PER_BLOCK;
    const int dispatch_warp_rank =
        dispatch_cta_rank * mega_globals::NUM_DISPATCH_WARPS + warp_idx;
    constexpr int NUM_DISPATCH_GROUP_WARPS =
        mega_globals::DISPATCH_CTAS_PER_BLOCK *
        mega_globals::NUM_DISPATCH_WARPS;

    for (int row_block = static_cast<int>(blockIdx.x) /
                             mega_globals::DISPATCH_CTAS_PER_BLOCK;
         row_block < G.num_row_blocks;
         row_block += dispatch_group_count) {
        const int slot = row_block % G.num_ring_blocks;
        const uint32_t generation =
            static_cast<uint32_t>(row_block / G.num_ring_blocks);
#ifdef PROFILE_TIMINGS
        const uint32_t timing_payload =
            (static_cast<uint32_t>(warp_idx) << 28) |
            (static_cast<uint32_t>(row_block) & 0x0FFFFFFFu);
        if (lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, timing_head,
                                EV_DISPATCH_WAIT_BEGIN, timing_payload);
        }
#endif

        // Every dispatch warp waits independently. This avoids a full-CTA
        // barrier with the concurrently running GEMM roles.
        if (lane == 0)
            wait_epoch(&G.ring_empty_epoch[slot], generation);
#ifdef PROFILE_TIMINGS
        if (lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, timing_head,
                                EV_DISPATCH_WAIT_END, timing_payload);
        }
#endif
        __syncwarp();

        for (int row_in_block = dispatch_warp_rank;
             row_in_block < mega_globals::BLOCK_M;
             row_in_block += NUM_DISPATCH_GROUP_WARPS) {
            const int dst_row = row_block * mega_globals::BLOCK_M +
                                row_in_block;
            const int src_gpu = G.pull_dispatch_indices[{dst_row, 0}];
            const int src_token = G.pull_dispatch_indices[{dst_row, 1}];
            const bool valid = src_gpu >= 0 && src_token >= 0;

            #pragma unroll
            for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
                if (valid) {
                    if (lane == 0) {
                        auto &arrived = dispatch_arrived[warp_idx];
                        ::dist::tma::expect_bytes(
                            arrived, mega_globals::PULL_BYTES);
                        const bf16 *src =
                            G.pre_tokens[src_gpu].raw_ptr +
                            static_cast<int64_t>(src_token) * mega_globals::H +
                            chunk * (mega_globals::PULL_BYTES /
                                     static_cast<int>(sizeof(bf16)));
                        ::dist::tma::bulk_load_async(
                            &dispatch_buffers[warp_idx][0], src,
                            mega_globals::PULL_BYTES, arrived);
                        wait(arrived, barrier_phase);
                        barrier_phase ^= 1;
                    }
                    __syncwarp();
                } else {
                    // Padded expert rows are materialized as zeros so FC1 can
                    // safely compute a full BLOCK_M tile.
                    #pragma unroll
                    for (int i = lane; i < UINT4S_PER_CHUNK; i += 32)
                        dispatch_buffers[warp_idx][i] = make_uint4(0, 0, 0, 0);
                    __syncwarp();
                }

                if (lane == 0) {
                    bf16 *dst = G.ring_tokens.raw_ptr +
                        static_cast<int64_t>(slot * mega_globals::BLOCK_M +
                                             row_in_block) * mega_globals::H +
                        chunk * (mega_globals::PULL_BYTES /
                                 static_cast<int>(sizeof(bf16)));
                    ::dist::tma::bulk_store_async(
                        dst, &dispatch_buffers[warp_idx][0],
                        mega_globals::PULL_BYTES);
                    ::dist::tma::store_async_wait();
                }
                __syncwarp();
            }
        }

        // Each CTA contributes once after its four dispatch warps finish.
        // The GEMM producer waits for all cooperating CTA contributions.
#ifdef PROFILE_TIMINGS
        if (lane == 0) {
            work_done_timestamps[warp_idx] =
                ::mkernel::timing::globaltimer_ns();
        }
        uint64_t group_sync_done_ts = 0;
        uint64_t publish_done_ts = 0;
#endif
        warpgroup::sync(4);
        if (warp_idx == 0 && lane == 0) {
#ifdef PROFILE_TIMINGS
            group_sync_done_ts = ::mkernel::timing::globaltimer_ns();
#endif
            comm::atomic_u32::release_add_gpu(
                &G.ring_full_epoch[slot], 1);
#ifdef PROFILE_TIMINGS
            publish_done_ts = ::mkernel::timing::globaltimer_ns();
#endif
        }
        warpgroup::sync(4);
#ifdef PROFILE_TIMINGS
        if (warp_idx == 0 && lane == 0) {
            const uint64_t round_sync_done_ts =
                ::mkernel::timing::globaltimer_ns();
            const uint32_t row_payload = static_cast<uint32_t>(row_block);
            #pragma unroll
            for (int timing_warp = 0;
                 timing_warp < mega_globals::NUM_DISPATCH_WARPS;
                 ++timing_warp) {
                const uint32_t work_payload =
                    (static_cast<uint32_t>(timing_warp) << 28) |
                    row_payload;
                MKERNEL_TIMING_EMIT_AT(
                    G.timings, timing_head, EV_DISPATCH_WORK_DONE,
                    work_payload, work_done_timestamps[timing_warp]);
            }
            MKERNEL_TIMING_EMIT_AT(
                G.timings, timing_head, EV_DISPATCH_GROUP_SYNC_DONE,
                row_payload, group_sync_done_ts);
            MKERNEL_TIMING_EMIT_AT(
                G.timings, timing_head, EV_DISPATCH_PUBLISH_DONE,
                row_payload, publish_done_ts);
            MKERNEL_TIMING_EMIT_AT(
                G.timings, timing_head, EV_DISPATCH_ROUND_SYNC_DONE,
                row_payload, round_sync_done_ts);
        }
#endif
    }
}

__global__ __launch_bounds__(mega_globals::THREADS, 1)
void mega_kernel(const __grid_constant__ mega_globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);

    mega_globals::pipeline_input (&inputs)[mega_globals::NUM_STAGES] =
        allocator.allocate<mega_globals::pipeline_input,
                           mega_globals::NUM_STAGES>();
    mega_globals::C_tile &C_smem =
        allocator.allocate<mega_globals::C_tile>();
    mega_globals::dispatch_chunk
        (&dispatch_buffers)[mega_globals::NUM_DISPATCH_WARPS] =
            allocator.allocate<mega_globals::dispatch_chunk,
                               mega_globals::NUM_DISPATCH_WARPS>();

    __shared__ semaphore inputs_arrived[mega_globals::NUM_STAGES];
    __shared__ semaphore inputs_finished[mega_globals::NUM_STAGES];
    __shared__ semaphore outputs_arrived[mega_globals::NUM_OUTPUT_STAGES];
    __shared__ semaphore outputs_finished[mega_globals::NUM_OUTPUT_STAGES];
    __shared__ semaphore
        dispatch_arrived[mega_globals::NUM_DISPATCH_WARPS];
    __shared__ uint32_t tmem_addr;
#ifdef PROFILE_TIMINGS
    __shared__ uint64_t
        dispatch_work_done_timestamps[mega_globals::NUM_DISPATCH_WARPS];
    __shared__ uint32_t timing_head;
    if (threadIdx.x == 0) {
        timing_head = 0;
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_CTA_START, ::mkernel::timing::smid());
    }
#endif

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < mega_globals::NUM_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < mega_globals::NUM_OUTPUT_STAGES; ++i) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }
    if (threadIdx.x < mega_globals::NUM_DISPATCH_WARPS)
        init_semaphore(dispatch_arrived[threadIdx.x], 0, 1);
    __syncthreads();

    tensor_allocator<1> tm_allocator;
    using C_tmem = tt<float, mega_globals::BLOCK_M,
                      mega_globals::BLOCK_N>;
    const int wg = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane = warp::laneid();

    // One complete warp allocates CTA-local tensor memory before the roles
    // diverge. All roles rendezvous here exactly once.
    if (wg == 2 && warp_in_wg == 0)
        tm_allocator.provision(tmem_addr);
    tensor_before_thread_sync();
    __syncthreads();
    tensor_after_thread_sync();
    tm_allocator.set_addr(tmem_addr);
    C_tmem C_tm[mega_globals::NUM_OUTPUT_STAGES] = {
        tm_allocator.allocate<C_tmem>(0),
        tm_allocator.allocate<C_tmem>(mega_globals::BLOCK_N),
    };

    constexpr int COL_BLOCKS = mega_globals::I / mega_globals::BLOCK_N;
    const int num_tasks = G.num_row_blocks * COL_BLOCKS;

    if (wg == 0) {
        // Communication producer. It progresses independently of all GEMM
        // roles and only synchronizes its own four warps.
#ifdef PROFILE_TIMINGS
        if (warp_in_wg == 0 && lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_DISPATCH_BEGIN, 0);
        }
#endif
        dispatch_warpgroup(G, dispatch_buffers, dispatch_arrived
#ifdef PROFILE_TIMINGS
                           , dispatch_work_done_timestamps, &timing_head
#endif
        );
#ifdef PROFILE_TIMINGS
        if (warp_in_wg == 0 && lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_DISPATCH_END, 0);
        }
#endif
    } else if (wg == 1) {
        // GEMM epilogue warpgroup.
#ifdef PROFILE_TIMINGS
        if (warp_in_wg == 0 && lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_EPILOGUE_BEGIN, 0);
        }
#endif
        int output_stage = 0;
        int output_phase[mega_globals::NUM_OUTPUT_STAGES] = {0, 0};

        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
            const int row_block = task_id / COL_BLOCKS;
            const int col_block = task_id % COL_BLOCKS;
            const int slot = row_block % G.num_ring_blocks;
            const uint32_t generation =
                static_cast<uint32_t>(row_block / G.num_ring_blocks);
#ifdef PROFILE_TIMINGS
            const uint32_t timing_task = static_cast<uint32_t>(task_id);
            uint64_t epilogue_tmem_wait_ns = 0;
            uint64_t epilogue_store_wait_ns = 0;
            if (warp_in_wg == 0 && lane == 0) {
                MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                    EV_EPILOGUE_WAIT_BEGIN, timing_task);
            }
#endif

            wait(outputs_arrived[output_stage], output_phase[output_stage]);
            output_phase[output_stage] ^= 1;
#ifdef PROFILE_TIMINGS
            if (warp_in_wg == 0 && lane == 0) {
                MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                    EV_EPILOGUE_WAIT_END, timing_task);
            }
#endif

            rt_bf<32, 32> C_reg;
            #pragma unroll
            for (int n = 0; n < mega_globals::BLOCK_N / 32; ++n) {
#ifdef PROFILE_TIMINGS
                uint64_t tmem_wait_begin = 0;
                if (warp_in_wg == 0 && lane == 0)
                    tmem_wait_begin = ::mkernel::timing::globaltimer_ns();
#endif
                warpgroup::load_async(
                    C_reg,
                    C_tm[output_stage].template subtile<
                        tt<float, mega_globals::BLOCK_M, 32>>(0, n * 32));
                tensor_load_wait();
#ifdef PROFILE_TIMINGS
                if (warp_in_wg == 0 && lane == 0) {
                    epilogue_tmem_wait_ns +=
                        ::mkernel::timing::globaltimer_ns() - tmem_wait_begin;
                }
#endif
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::store(C_smem, C_reg);
                warpgroup::sync(1);

                if (warp_in_wg == 0 && lane == 0) {
#ifdef PROFILE_TIMINGS
                    const uint64_t store_wait_begin =
                        ::mkernel::timing::globaltimer_ns();
#endif
                    ::dist::tma::store_async(
                        G.outputs, C_smem,
                        {row_block,
                         col_block * (mega_globals::BLOCK_N / 32) + n});
                    ::dist::tma::store_async_wait();
#ifdef PROFILE_TIMINGS
                    epilogue_store_wait_ns +=
                        ::mkernel::timing::globaltimer_ns() - store_wait_begin;
#endif
                }
                warpgroup::sync(1);
            }

            if (warp_in_wg == 0 && lane == 0) {
                arrive(outputs_finished[output_stage]);

                const uint32_t old = atomicAdd(
                    reinterpret_cast<unsigned int*>(
                        &G.ring_done_tiles[slot]), 1u);
                if (old + 1u == static_cast<uint32_t>(COL_BLOCKS)) {
                    // Reset before publishing empty_epoch. The release/acquire
                    // pair makes this reset visible to the next generation.
                    G.ring_done_tiles[slot] = 0;
                    comm::atomic_u32::release_store_gpu(
                        &G.ring_empty_epoch[slot], generation + 1);
                }
#ifdef PROFILE_TIMINGS
                MKERNEL_TIMING_EMIT_AT(
                    G.timings, &timing_head,
                    EV_EPILOGUE_TMEM_WAIT_TOTAL, timing_task,
                    epilogue_tmem_wait_ns);
                MKERNEL_TIMING_EMIT_AT(
                    G.timings, &timing_head,
                    EV_EPILOGUE_STORE_WAIT_TOTAL, timing_task,
                    epilogue_store_wait_ns);
                MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                    EV_EPILOGUE_TASK_END, timing_task);
#endif
            }
            output_stage ^= 1;
        }
#ifdef PROFILE_TIMINGS
        if (warp_in_wg == 0 && lane == 0) {
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_EPILOGUE_END, 0);
        }
#endif
    } else if (wg == 2 && warp_in_wg == 3 && lane == 0) {
        // A/B TMA producer. Tasks are row-major, matching dispatch order.
#ifdef PROFILE_TIMINGS
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_TMA_PRODUCER_BEGIN, 0);
#endif
        int stage = 0;
        int finished_phase[mega_globals::NUM_STAGES];
        #pragma unroll
        for (int i = 0; i < mega_globals::NUM_STAGES; ++i)
            finished_phase[i] = 1;
        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
            const int row_block = task_id / COL_BLOCKS;
            const int col_block = task_id % COL_BLOCKS;
            const int expert = G.row_block_to_expert[row_block];
            const int slot = row_block % G.num_ring_blocks;
            const uint32_t generation =
                static_cast<uint32_t>(row_block / G.num_ring_blocks);
#ifdef PROFILE_TIMINGS
            const uint32_t timing_task = static_cast<uint32_t>(task_id);
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_TMA_RING_WAIT_BEGIN, timing_task);
#endif

            wait_epoch(
                &G.ring_full_epoch[slot],
                (generation + 1) *
                    mega_globals::DISPATCH_CTAS_PER_BLOCK);
#ifdef PROFILE_TIMINGS
            MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                                EV_TMA_RING_WAIT_END, timing_task);
            const uint64_t tma_k_loop_begin =
                ::mkernel::timing::globaltimer_ns();
#endif

            #pragma unroll 1
            for (int k = 0; k < mega_globals::H / mega_globals::BLOCK_K;
                 ++k) {
                wait(inputs_finished[stage], finished_phase[stage]);
                finished_phase[stage] ^= 1;
                ::dist::tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(mega_globals::pipeline_input));
                ::dist::tma::load_async(
                    inputs[stage].A, G.ring_tokens,
                    {slot, k}, inputs_arrived[stage]);
                ::dist::tma::load_async(
                    inputs[stage].B, G.weights,
                    {expert, k, col_block}, inputs_arrived[stage]);
                stage = (stage + 1) % mega_globals::NUM_STAGES;
            }
#ifdef PROFILE_TIMINGS
            const uint64_t tma_k_loop_end =
                ::mkernel::timing::globaltimer_ns();
            MKERNEL_TIMING_EMIT_AT(
                G.timings, &timing_head, EV_TMA_K_LOOP_BEGIN,
                timing_task, tma_k_loop_begin);
            MKERNEL_TIMING_EMIT_AT(
                G.timings, &timing_head, EV_TMA_K_LOOP_END,
                timing_task, tma_k_loop_end);
#endif
        }
#ifdef PROFILE_TIMINGS
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_TMA_PRODUCER_END, 0);
#endif
    } else if (wg == 2 && warp_in_wg == 0 && lane == 0) {
        // tcgen05 issue thread.
#ifdef PROFILE_TIMINGS
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_MMA_ISSUER_BEGIN, 0);
#endif
        int stage = 0;
        int arrived_phase[mega_globals::NUM_STAGES];
        #pragma unroll
        for (int i = 0; i < mega_globals::NUM_STAGES; ++i)
            arrived_phase[i] = 0;
        int output_stage = 0;
        int reuse_phase[mega_globals::NUM_OUTPUT_STAGES] = {1, 1};
        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
#ifdef PROFILE_TIMINGS
            const uint32_t timing_task = static_cast<uint32_t>(task_id);
#endif
            wait(outputs_finished[output_stage], reuse_phase[output_stage]);
            reuse_phase[output_stage] ^= 1;
#ifdef PROFILE_TIMINGS
            const uint64_t mma_k_loop_begin =
                ::mkernel::timing::globaltimer_ns();
#endif

            #pragma unroll 1
            for (int k = 0; k < mega_globals::H / mega_globals::BLOCK_K;
                 ++k) {
                wait(inputs_arrived[stage], arrived_phase[stage]);
                arrived_phase[stage] ^= 1;
                if (k == 0) {
                    warpgroup::mm_AB(C_tm[output_stage], inputs[stage].A,
                                     inputs[stage].B,
                                     inputs_finished[stage]);
                } else {
                    warpgroup::mma_AB(C_tm[output_stage], inputs[stage].A,
                                      inputs[stage].B,
                                      inputs_finished[stage]);
                }
                stage = (stage + 1) % mega_globals::NUM_STAGES;
            }
#ifdef PROFILE_TIMINGS
            const uint64_t mma_k_loop_end =
                ::mkernel::timing::globaltimer_ns();
#endif
            tensor_commit<1>(outputs_arrived[output_stage]);
#ifdef PROFILE_TIMINGS
            MKERNEL_TIMING_EMIT_AT(
                G.timings, &timing_head, EV_MMA_K_LOOP_BEGIN,
                timing_task, mma_k_loop_begin);
            MKERNEL_TIMING_EMIT_AT(
                G.timings, &timing_head, EV_MMA_K_LOOP_END,
                timing_task, mma_k_loop_end);
#endif
            output_stage ^= 1;
        }
#ifdef PROFILE_TIMINGS
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_MMA_ISSUER_END, 0);
#endif
    }

    __syncthreads();
#ifdef PROFILE_TIMINGS
    if (threadIdx.x == 0) {
        MKERNEL_TIMING_EMIT(G.timings, &timing_head,
                            EV_CTA_END, ::mkernel::timing::smid());
    }
#endif
    if (wg == 2 && warp_in_wg == 0)
        tm_allocator.deprovision();
}

void launch_mega(const mega_globals &G, cudaStream_t stream) {
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        mega_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SHARED_MEMORY));
    mega_kernel<<<G.num_sms, mega_globals::THREADS,
                  DYNAMIC_SHARED_MEMORY, stream>>>(G);
    MKERNEL_CUDACHECK(cudaGetLastError());
}

}  // namespace moe_dispatch_fc1_mega_blackwell

#include "operators/dispatch_fc1_mega_blackwell/session.cuh"
