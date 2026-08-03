/**
 * @file dispatch_gemm_blackwell.cu
 * @brief Single NVLink-domain MoE token gather using peer TMA/LSA.
 *
 * It replaces the RDMA send/copy/arrival protocol with direct loads from the
 * owning GPU, then consumes the gathered rows with a persistent tcgen05 GEMM.
 */
#include "operators/dispatch_gemm_blackwell/dispatch_gemm_blackwell.cuh"

namespace moe_dispatch_gemm_blackwell {

__device__ inline void dispatch_slice(const fused_globals &G, int slice) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);
    fused_globals::token_vec (&tokens)[fused_globals::TOKENS_PER_BLOCK] =
        allocator.allocate<fused_globals::token_vec,
                           fused_globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore arrived;
    __shared__ int src_gpus[fused_globals::TOKENS_PER_BLOCK];
    __shared__ int src_tokens[fused_globals::TOKENS_PER_BLOCK];

    const int lane = threadIdx.x;
    if (lane < 32) {
        const int dst_token =
            slice * fused_globals::TOKENS_PER_BLOCK + lane;
        const bool valid_dst =
            lane < fused_globals::TOKENS_PER_BLOCK &&
            dst_token < G.num_output_tokens;
        int src_gpu = -1;
        int src_token = -1;
        if (valid_dst) {
            src_gpu = G.pull_dispatch_indices[{dst_token, 0}];
            src_token = G.pull_dispatch_indices[{dst_token, 1}];
        }
        if (lane < fused_globals::TOKENS_PER_BLOCK) {
            src_gpus[lane] = src_gpu;
            src_tokens[lane] = src_token;
        }
        __syncwarp();

        constexpr uint32_t TOKEN_BYTES = sizeof(fused_globals::token_vec);
        static_assert(TOKEN_BYTES <= 16 * 1024 && TOKEN_BYTES % 16 == 0);
        if (lane == 0) {
            int source_rows = 0;
            #pragma unroll
            for (int i = 0; i < fused_globals::TOKENS_PER_BLOCK; ++i) {
                source_rows += src_gpus[i] >= 0 && src_tokens[i] >= 0;
            }
            if (source_rows != 0) {
                init_semaphore(arrived, 0, 1);
                ::dist::tma::expect_bytes(
                    arrived, source_rows * sizeof(fused_globals::token_vec));
                #pragma unroll
                for (int i = 0; i < fused_globals::TOKENS_PER_BLOCK; ++i) {
                    if (src_gpus[i] >= 0 && src_tokens[i] >= 0) {
                        const bf16 *src =
                            G.pre_tokens[src_gpus[i]].raw_ptr +
                            static_cast<int64_t>(src_tokens[i]) *
                                fused_globals::H;
                        ::dist::tma::bulk_load_async(
                            &tokens[i], src, TOKEN_BYTES, arrived);
                    }
                }
                wait(arrived, 0);
            }
        }
        __syncwarp();
        if (valid_dst && src_gpu >= 0 && src_token >= 0) {
            bf16 *dst = G.post_tokens.raw_ptr +
                static_cast<int64_t>(dst_token) * fused_globals::H;
            ::dist::tma::bulk_store_async(dst, &tokens[lane], TOKEN_BYTES);
            ::dist::tma::store_async_wait();
        }

        const unsigned valid_mask = __ballot_sync(0xffffffffu, valid_dst);
        __syncwarp();
        if (lane == 0 && valid_mask != 0u) {
            constexpr int SLICES_PER_ROW_BLOCK =
                fused_globals::ROW_BLOCK / fused_globals::TOKENS_PER_BLOCK;
            const int row_block = slice / SLICES_PER_ROW_BLOCK;
            comm::atomic_u32::release_add_gpu(
                &G.row_ready[{row_block}], __popc(valid_mask));
        }
    }
}

// GEMM SM check this, if it hits output row block, then starts working
__device__ inline void wait_row_ready(const fused_globals &G,
                                      int row_block,
                                      int expected_rows) {
    int ready = comm::atomic_u32::acquire_load_s32_gpu(
        &G.row_ready[{row_block}]);
    while (ready != expected_rows) {
        __nanosleep(64);
        ready = comm::atomic_u32::acquire_load_s32_gpu(
            &G.row_ready[{row_block}]);
    }
}


// Task iterator, for each CTA worker, uses this to loop over all 
struct gemm_task {
    int expert;
    int row_block;
    int col_block;
};

__device__ inline void super_m_decode(int task_id,
                                      int row_blocks,
                                      int &row,
                                      int &col) {
    constexpr int COL_BLOCKS = fused_globals::I / fused_globals::COL_BLOCK;
    constexpr int SUPER_M = fused_globals::SUPER_M;
    const int super_rows = (row_blocks / SUPER_M) * SUPER_M;
    const int super_tasks = super_rows * COL_BLOCKS;

    if (task_id < super_tasks) {
        const int tasks_per_group = SUPER_M * COL_BLOCKS;
        const int task_in_group = task_id % tasks_per_group;
        row = (task_id / tasks_per_group) * SUPER_M + task_in_group % SUPER_M;
        col = task_in_group / SUPER_M;
    } else {
        const int final_rows = row_blocks - super_rows;
        const int remainder = task_id - super_tasks;
        row = super_rows + remainder % final_rows;
        col = remainder / final_rows;
    }
}

struct gemm_task_iterator {
    const fused_globals &G;
    int stride;
    int expert = 0;
    int cumulative_rows = 0;
    int local_task;

    __device__ gemm_task_iterator(const fused_globals &globals,
                                  int worker,
                                  int worker_count)
        : G(globals), stride(worker_count), local_task(worker) {}

    __device__ bool next(gemm_task &task) {
        constexpr int COL_BLOCKS = fused_globals::I / fused_globals::COL_BLOCK;
        while (expert < G.num_local_experts) {
            const int padded = G.padded_tokens_per_expert[{expert}];
            const int row_begin = cumulative_rows / fused_globals::ROW_BLOCK;
            const int next_cumulative = cumulative_rows + padded;
            const int row_end =
                (next_cumulative + fused_globals::ROW_BLOCK - 1) /
                fused_globals::ROW_BLOCK;
            const int row_blocks = row_end - row_begin;
            const int num_tasks = row_blocks * COL_BLOCKS;

            if (local_task < num_tasks) {
                int row_in_expert;
                super_m_decode(local_task, row_blocks,
                               row_in_expert, task.col_block);
                task.expert = expert;
                task.row_block = row_begin + row_in_expert;
                local_task += stride;
                return true;
            }

            local_task -= num_tasks;
            cumulative_rows = next_cumulative;
            ++expert;
        }
        return false;
    }
};

__device__ inline void gemm_sm(const fused_globals &G, int worker) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);
    fused_globals::pipeline_input (&inputs)[fused_globals::PIPELINE_STAGES] =
        allocator.allocate<fused_globals::pipeline_input,
                           fused_globals::PIPELINE_STAGES>();
    fused_globals::C_tile &C_smem =
        allocator.allocate<fused_globals::C_tile>();

    __shared__ semaphore inputs_arrived[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived[fused_globals::OUTPUT_STAGES];
    __shared__ semaphore outputs_finished[fused_globals::OUTPUT_STAGES];
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < fused_globals::OUTPUT_STAGES; ++i) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }
    __syncthreads();

    tensor_allocator<1> tm_allocator;
    using C_tmem = tt<float, fused_globals::ROW_BLOCK,
                      fused_globals::COL_BLOCK>;
    const int wg = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane = warp::laneid();

    if (wg == 0 && warp_in_wg == 0) {
        tm_allocator.provision(tmem_addr);
    }
    tensor_before_thread_sync();
    __syncthreads();
    tensor_after_thread_sync();
    tm_allocator.set_addr(tmem_addr);
    C_tmem C_tm[fused_globals::OUTPUT_STAGES] = {
        tm_allocator.allocate<C_tmem>(0),
        tm_allocator.allocate<C_tmem>(fused_globals::COL_BLOCK),
    };

    if (wg == 0) {
        int output_stage = 0;
        int output_phase[fused_globals::OUTPUT_STAGES] = {0, 0};
        gemm_task_iterator tasks(G, worker, G.num_gemm_sms);
        gemm_task task;
        while (tasks.next(task)) {
            wait(outputs_arrived[output_stage], output_phase[output_stage]);
            output_phase[output_stage] ^= 1;

            rt_bf<32, 32> C_reg;
            #pragma unroll
            for (int n = 0; n < fused_globals::COL_BLOCK / 32; ++n) {
                warpgroup::load_async(
                    C_reg,
                    C_tm[output_stage].template subtile<tt<float,
                        fused_globals::ROW_BLOCK, 32>>(0, n * 32));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::store(C_smem, C_reg);
                warpgroup::sync(1);

                if (warp_in_wg == 0 && lane == 0) {
                    ::dist::tma::store_async(
                        G.outputs, C_smem,
                        {task.row_block,
                         task.col_block * (fused_globals::COL_BLOCK / 32) + n});
                    ::dist::tma::store_async_wait();
                }
                warpgroup::sync(1);
            }

            if (warp_in_wg == 0 && lane == 0) {
                arrive(outputs_finished[output_stage]);
            }
            output_stage ^= 1;
        }
    } else if (warp_in_wg == 3 && lane == 0) {
        int stage = 0;
        int finished_phase[fused_globals::PIPELINE_STAGES];
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            finished_phase[i] = 1;
        }
        gemm_task_iterator tasks(G, worker, G.num_gemm_sms);
        gemm_task task;
        while (tasks.next(task)) {
            const int expected_rows =
                min(fused_globals::ROW_BLOCK,
                    G.num_output_tokens - task.row_block * fused_globals::ROW_BLOCK);
            wait_row_ready(G, task.row_block, expected_rows);

            #pragma unroll 1
            for (int k = 0; k < fused_globals::H / fused_globals::RED_BLOCK; ++k) {
                wait(inputs_finished[stage], finished_phase[stage]);
                finished_phase[stage] ^= 1;
                ::dist::tma::expect_bytes(
                    inputs_arrived[stage], sizeof(fused_globals::pipeline_input));
                ::dist::tma::load_async(
                    inputs[stage].A, G.post_tokens,
                    {task.row_block, k}, inputs_arrived[stage]);
                ::dist::tma::load_async(
                    inputs[stage].B, G.weights,
                    {task.expert, k, task.col_block}, inputs_arrived[stage]);
                stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
            }
        }
    } else if (warp_in_wg == 0 && lane == 0) {
        int stage = 0;
        int arrived_phase[fused_globals::PIPELINE_STAGES];
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            arrived_phase[i] = 0;
        }
        int output_stage = 0;
        int reuse_phase[fused_globals::OUTPUT_STAGES] = {1, 1};
        gemm_task_iterator tasks(G, worker, G.num_gemm_sms);
        gemm_task task;
        while (tasks.next(task)) {
            wait(outputs_finished[output_stage], reuse_phase[output_stage]);
            reuse_phase[output_stage] ^= 1;

            #pragma unroll 1
            for (int k = 0; k < fused_globals::H / fused_globals::RED_BLOCK; ++k) {
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
                stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
            }
            tensor_commit<1>(outputs_arrived[output_stage]);
            output_stage ^= 1;
        }
    }

    __syncthreads();
    if (wg == 0 && warp_in_wg == 0) {
        tm_allocator.deprovision();
    }
}

__global__ __launch_bounds__(256, 1)
void fused_kernel(const __grid_constant__ fused_globals G) {
    if (blockIdx.x < G.num_dispatch_sms) {
        const int total_slices =
            (G.num_output_tokens + fused_globals::TOKENS_PER_BLOCK - 1) /
            fused_globals::TOKENS_PER_BLOCK;
        for (int slice = blockIdx.x; slice < total_slices;
             slice += G.num_dispatch_sms) {
            dispatch_slice(G, slice);
        }
    } else {
        gemm_sm(G, blockIdx.x - G.num_dispatch_sms);
    }
}

void launch_fused(const fused_globals& G, cudaStream_t stream) {
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SHARED_MEMORY));
    fused_kernel<<<G.num_dispatch_sms + G.num_gemm_sms, 256,
                   DYNAMIC_SHARED_MEMORY, stream>>>(G);
    MKERNEL_CUDACHECK(cudaGetLastError());
}

}  // namespace moe_dispatch_gemm_blackwell

#include "operators/dispatch_gemm_blackwell/session.cuh"
