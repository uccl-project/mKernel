/**
 * @file dispatch_gemm_warp_specialization.cu
 * @brief Persistent warp-specialized dispatch + BF16 FC1 for Blackwell.
 */

#include "operators/dispatch_gemm_warp_specialization/dispatch_gemm_warp_specialization.cuh"

namespace moe_dispatch_gemm_warp_specialization {

__device__ __forceinline__ void wait_epoch(int *ptr, uint32_t target) {
    uint32_t value = comm::atomic_u32::acquire_load_gpu(ptr);
    while (value != target) {
        __nanosleep(64);
        value = comm::atomic_u32::acquire_load_gpu(ptr);
    }
}

template <typename DispatchBuffers>
__device__ inline void dispatch_warpgroup(
    const warp_specialization_globals &G,
    DispatchBuffers &dispatch_buffers,
    semaphore (&dispatch_arrived)[warp_specialization_globals::NUM_DISPATCH_WARPS]
) {
    constexpr int NUM_CHUNKS =
        warp_specialization_globals::H * static_cast<int>(sizeof(bf16)) /
        warp_specialization_globals::PULL_BYTES;
    constexpr int UINT4S_PER_CHUNK =
        warp_specialization_globals::PULL_BYTES / static_cast<int>(sizeof(uint4));

    const int warp_idx = warpgroup::warpid();
    const int lane = warp::laneid();
    uint32_t barrier_phase = 0;

    const int dispatch_group_count =
        G.num_sms / warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK;
    const int num_dispatch_ctas =
        dispatch_group_count * warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK;
    if (static_cast<int>(blockIdx.x) >= num_dispatch_ctas)
        return;
    const int dispatch_cta_rank =
        static_cast<int>(blockIdx.x) %
        warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK;
    const int dispatch_warp_rank =
        dispatch_cta_rank * warp_specialization_globals::NUM_DISPATCH_WARPS + warp_idx;
    constexpr int NUM_DISPATCH_GROUP_WARPS =
        warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK *
        warp_specialization_globals::NUM_DISPATCH_WARPS;

    for (int row_block = static_cast<int>(blockIdx.x) /
                             warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK;
         row_block < G.num_row_blocks;
         row_block += dispatch_group_count) {
        const int slot = row_block % G.num_ring_blocks;
        const uint32_t generation =
            static_cast<uint32_t>(row_block / G.num_ring_blocks);

        // Every dispatch warp waits independently. This avoids a full-CTA
        // barrier with the concurrently running GEMM roles.
        if (lane == 0)
            wait_epoch(&G.ring_empty_epoch[slot], generation);
        __syncwarp();

        for (int row_in_block = dispatch_warp_rank;
             row_in_block < warp_specialization_globals::BLOCK_M;
             row_in_block += NUM_DISPATCH_GROUP_WARPS) {
            const int dst_row = row_block * warp_specialization_globals::BLOCK_M +
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
                            arrived, warp_specialization_globals::PULL_BYTES);
                        const bf16 *src =
                            G.pre_tokens[src_gpu].raw_ptr +
                            static_cast<int64_t>(src_token) * warp_specialization_globals::H +
                            chunk * (warp_specialization_globals::PULL_BYTES /
                                     static_cast<int>(sizeof(bf16)));
                        ::dist::tma::bulk_load_async(
                            &dispatch_buffers[warp_idx][0], src,
                            warp_specialization_globals::PULL_BYTES, arrived);
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
                        static_cast<int64_t>(slot * warp_specialization_globals::BLOCK_M +
                                             row_in_block) * warp_specialization_globals::H +
                        chunk * (warp_specialization_globals::PULL_BYTES /
                                 static_cast<int>(sizeof(bf16)));
                    ::dist::tma::bulk_store_async(
                        dst, &dispatch_buffers[warp_idx][0],
                        warp_specialization_globals::PULL_BYTES);
                    ::dist::tma::store_async_wait();
                }
                __syncwarp();
            }
        }

        // Each CTA contributes once after its four dispatch warps finish.
        // The GEMM producer waits for all cooperating CTA contributions.
        warpgroup::sync(4);
        if (warp_idx == 0 && lane == 0) {
            comm::atomic_u32::release_add_gpu(
                &G.ring_full_epoch[slot], 1);
        }
        warpgroup::sync(4);
    }
}

__global__ __launch_bounds__(warp_specialization_globals::THREADS, 1)
void warp_specialized_kernel(const __grid_constant__ warp_specialization_globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);

    warp_specialization_globals::pipeline_input (&inputs)[warp_specialization_globals::NUM_STAGES] =
        allocator.allocate<warp_specialization_globals::pipeline_input,
                           warp_specialization_globals::NUM_STAGES>();
    warp_specialization_globals::C_tile &C_smem =
        allocator.allocate<warp_specialization_globals::C_tile>();
    warp_specialization_globals::dispatch_chunk
        (&dispatch_buffers)[warp_specialization_globals::NUM_DISPATCH_WARPS] =
            allocator.allocate<warp_specialization_globals::dispatch_chunk,
                               warp_specialization_globals::NUM_DISPATCH_WARPS>();

    __shared__ semaphore inputs_arrived[warp_specialization_globals::NUM_STAGES];
    __shared__ semaphore inputs_finished[warp_specialization_globals::NUM_STAGES];
    __shared__ semaphore outputs_arrived[warp_specialization_globals::NUM_OUTPUT_STAGES];
    __shared__ semaphore outputs_finished[warp_specialization_globals::NUM_OUTPUT_STAGES];
    __shared__ semaphore
        dispatch_arrived[warp_specialization_globals::NUM_DISPATCH_WARPS];
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < warp_specialization_globals::NUM_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < warp_specialization_globals::NUM_OUTPUT_STAGES; ++i) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }
    if (threadIdx.x < warp_specialization_globals::NUM_DISPATCH_WARPS)
        init_semaphore(dispatch_arrived[threadIdx.x], 0, 1);
    __syncthreads();

    tensor_allocator<1> tm_allocator;
    using C_tmem = tt<float, warp_specialization_globals::BLOCK_M,
                      warp_specialization_globals::BLOCK_N>;
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
    C_tmem C_tm[warp_specialization_globals::NUM_OUTPUT_STAGES] = {
        tm_allocator.allocate<C_tmem>(0),
        tm_allocator.allocate<C_tmem>(warp_specialization_globals::BLOCK_N),
    };

    constexpr int COL_BLOCKS = warp_specialization_globals::I / warp_specialization_globals::BLOCK_N;
    const int num_tasks = G.num_row_blocks * COL_BLOCKS;

    if (wg == 0) {
        // Communication producer. It progresses independently of all GEMM
        // roles and only synchronizes its own four warps.
        dispatch_warpgroup(G, dispatch_buffers, dispatch_arrived
        );
    } else if (wg == 1) {
        // GEMM epilogue warpgroup.
        int output_stage = 0;
        int output_phase[warp_specialization_globals::NUM_OUTPUT_STAGES] = {0, 0};

        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
            const int row_block = task_id / COL_BLOCKS;
            const int col_block = task_id % COL_BLOCKS;
            const int slot = row_block % G.num_ring_blocks;
            const uint32_t generation =
                static_cast<uint32_t>(row_block / G.num_ring_blocks);

            wait(outputs_arrived[output_stage], output_phase[output_stage]);
            output_phase[output_stage] ^= 1;

            rt_bf<32, 32> C_reg;
            #pragma unroll
            for (int n = 0; n < warp_specialization_globals::BLOCK_N / 32; ++n) {
                warpgroup::load_async(
                    C_reg,
                    C_tm[output_stage].template subtile<
                        tt<float, warp_specialization_globals::BLOCK_M, 32>>(0, n * 32));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::store(C_smem, C_reg);
                warpgroup::sync(1);

                if (warp_in_wg == 0 && lane == 0) {
                    ::dist::tma::store_async(
                        G.outputs, C_smem,
                        {row_block,
                         col_block * (warp_specialization_globals::BLOCK_N / 32) + n});
                    ::dist::tma::store_async_wait();
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
            }
            output_stage ^= 1;
        }
    } else if (wg == 2 && warp_in_wg == 3 && lane == 0) {
        // A/B TMA producer. Tasks are row-major, matching dispatch order.
        int stage = 0;
        int finished_phase[warp_specialization_globals::NUM_STAGES];
        #pragma unroll
        for (int i = 0; i < warp_specialization_globals::NUM_STAGES; ++i)
            finished_phase[i] = 1;

        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
            const int row_block = task_id / COL_BLOCKS;
            const int col_block = task_id % COL_BLOCKS;
            const int expert = G.row_block_to_expert[row_block];
            const int slot = row_block % G.num_ring_blocks;
            const uint32_t generation =
                static_cast<uint32_t>(row_block / G.num_ring_blocks);

            wait_epoch(
                &G.ring_full_epoch[slot],
                (generation + 1) *
                    warp_specialization_globals::DISPATCH_CTAS_PER_BLOCK);

            #pragma unroll 1
            for (int k = 0; k < warp_specialization_globals::H / warp_specialization_globals::BLOCK_K;
                 ++k) {
                wait(inputs_finished[stage], finished_phase[stage]);
                finished_phase[stage] ^= 1;
                ::dist::tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(warp_specialization_globals::pipeline_input));
                ::dist::tma::load_async(
                    inputs[stage].A, G.ring_tokens,
                    {slot, k}, inputs_arrived[stage]);
                ::dist::tma::load_async(
                    inputs[stage].B, G.weights,
                    {expert, k, col_block}, inputs_arrived[stage]);
                stage = (stage + 1) % warp_specialization_globals::NUM_STAGES;
            }
        }
    } else if (wg == 2 && warp_in_wg == 0 && lane == 0) {
        // tcgen05 issue thread.
        int stage = 0;
        int arrived_phase[warp_specialization_globals::NUM_STAGES];
        #pragma unroll
        for (int i = 0; i < warp_specialization_globals::NUM_STAGES; ++i)
            arrived_phase[i] = 0;
        int output_stage = 0;
        int reuse_phase[warp_specialization_globals::NUM_OUTPUT_STAGES] = {1, 1};

        for (int task_id = static_cast<int>(blockIdx.x);
             task_id < num_tasks; task_id += G.num_sms) {
            wait(outputs_finished[output_stage], reuse_phase[output_stage]);
            reuse_phase[output_stage] ^= 1;

            #pragma unroll 1
            for (int k = 0; k < warp_specialization_globals::H / warp_specialization_globals::BLOCK_K;
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
                stage = (stage + 1) % warp_specialization_globals::NUM_STAGES;
            }
            tensor_commit<1>(outputs_arrived[output_stage]);
            output_stage ^= 1;
        }
    }

    __syncthreads();
    if (wg == 2 && warp_in_wg == 0)
        tm_allocator.deprovision();
}

void launch_warp_specialization(const warp_specialization_globals &G, cudaStream_t stream) {
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        warp_specialized_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SHARED_MEMORY));
    warp_specialized_kernel<<<G.num_sms, warp_specialization_globals::THREADS,
                  DYNAMIC_SHARED_MEMORY, stream>>>(G);
    MKERNEL_CUDACHECK(cudaGetLastError());
}

}  // namespace moe_dispatch_gemm_warp_specialization

#include "operators/dispatch_gemm_warp_specialization/session.cuh"
