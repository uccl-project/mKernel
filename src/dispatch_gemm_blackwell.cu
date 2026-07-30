/**
 * @file dispatch_gemm_blackwell.cu
 * @brief Single NVLink-domain MoE token gather using peer TMA/LSA.
 *
 * This is stage 1 of the Blackwell dispatch+GEMM port.  It replaces the full
 * RDMA send/copy/arrival protocol with direct loads from the owning GPU.  The
 * tcgen05 consumer will be added after this producer is validated separately.
 */
#include "operators/dispatch_gemm_blackwell/dispatch_gemm_blackwell.cuh"

namespace moe_dispatch_gemm_blackwell {

template<typename Globals>
__device__ inline void dispatch_slice(const Globals &G, int slice) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);
    typename Globals::token_vec (&tokens)[Globals::TOKENS_PER_BLOCK] =
        allocator.allocate<typename Globals::token_vec,
                           Globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore arrived[Globals::TOKENS_PER_BLOCK];

    const int lane = threadIdx.x;
    const int dst_token = slice * Globals::TOKENS_PER_BLOCK + lane;
    const bool valid_dst =
        lane < Globals::TOKENS_PER_BLOCK &&
        dst_token < G.num_output_tokens;

    if (valid_dst) {
        const int src_gpu = G.pull_dispatch_indices[{dst_token, 0}];
        const int src_token = G.pull_dispatch_indices[{dst_token, 1}];

        // Negative source coordinates represent a padded row. post_tokens is
        // initialized to zero by the caller, but the row still counts toward
        // readiness because GEMM consumes the padded row block as a whole.
        if (src_gpu >= 0 && src_token >= 0) {
            init_semaphore(arrived[lane], 0, 1);
            ::dist::tma::expect_bytes(arrived[lane],
                                      sizeof(typename Globals::token_vec));

            // [TODO:yihan] split into 2 pass for further opt, local weights can get to gemm while overlapping with loading from peer rank
            ::dist::tma::load_async(tokens[lane], G.pre_tokens[src_gpu],
                                    {src_token, 0}, arrived[lane]);
            wait(arrived[lane], 0);

            ::dist::tma::store_async(G.post_tokens, tokens[lane],
                                     {dst_token, 0});
            ::dist::tma::store_async_wait();
        }
    }

    // All lanes, including inactive lanes 16..31, reach this point. Therefore
    // lane 0 publishes readiness only after every valid lane has completed its
    // post_tokens TMA store.
    const unsigned valid_mask = __ballot_sync(0xffffffffu, valid_dst);
    __syncwarp();
    if (lane == 0 && valid_mask != 0u) {
        constexpr int SLICES_PER_ROW_BLOCK =
            Globals::ROW_BLOCK / Globals::TOKENS_PER_BLOCK;
        const int row_block = slice / SLICES_PER_ROW_BLOCK;
        comm::atomic_u32::release_add_gpu(
            &G.row_ready[{row_block}], __popc(valid_mask));
    }
}

// Consumer-side half of the dispatch/GEMM handoff. The future tcgen05 loader
// calls this before TMA-loading an A tile from the corresponding row block.
template<typename Globals>
__device__ inline void wait_row_ready(const Globals &G,
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
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1> tm_allocator;
    using C_tmem = tt<float, fused_globals::ROW_BLOCK,
                      fused_globals::COL_BLOCK>;
    const int wg = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane = warp::laneid();

    if (wg == 0) {
        if (warp_in_wg == 0) {
            tm_allocator.provision(tmem_addr);
            if (lane == 0)
                arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        C_tmem C_tm = tm_allocator.allocate<C_tmem>(0);

        int output_phase = 0;
        gemm_task_iterator tasks(G, worker, G.num_gemm_sms);
        gemm_task task;
        while (tasks.next(task)) {
            wait(outputs_arrived, output_phase);
            output_phase ^= 1;

            rt_bf<32, 32> C_reg;
            #pragma unroll
            for (int n = 0; n < fused_globals::COL_BLOCK / 32; ++n) {
                warpgroup::load_async(
                    C_reg,
                    C_tm.template subtile<tt<float,
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

            if (warp_in_wg == 0 && lane == 0)
                arrive(outputs_finished);
        }

        warpgroup::sync(1);
        if (warp_in_wg == 0)
            tm_allocator.deprovision();
    } else if (warp_in_wg == 3 && lane == 0) {
        int stage = 0;
        int finished_phase[fused_globals::PIPELINE_STAGES] = {1, 1, 1};
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
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        C_tmem C_tm = tm_allocator.allocate<C_tmem>(0);

        int stage = 0;
        int arrived_phase[fused_globals::PIPELINE_STAGES] = {0, 0, 0};
        int output_reuse_phase = 1;
        gemm_task_iterator tasks(G, worker, G.num_gemm_sms);
        gemm_task task;
        while (tasks.next(task)) {
            wait(outputs_finished, output_reuse_phase);
            output_reuse_phase ^= 1;

            #pragma unroll 1
            for (int k = 0; k < fused_globals::H / fused_globals::RED_BLOCK; ++k) {
                wait(inputs_arrived[stage], arrived_phase[stage]);
                arrived_phase[stage] ^= 1;
                if (k == 0)
                    warpgroup::mm_AB(C_tm, inputs[stage].A, inputs[stage].B,
                                     inputs_finished[stage]);
                else
                    warpgroup::mma_AB(C_tm, inputs[stage].A, inputs[stage].B,
                                      inputs_finished[stage]);
                stage = (stage + 1) % fused_globals::PIPELINE_STAGES;
            }
            tensor_commit<1>(outputs_arrived);
        }
    }
}

__global__ __launch_bounds__(256, 1)
void fused_kernel(const __grid_constant__ fused_globals G) {
    if (blockIdx.x < G.num_dispatch_sms) {
        const int total_slices =
            (G.num_output_tokens + fused_globals::TOKENS_PER_BLOCK - 1) /
            fused_globals::TOKENS_PER_BLOCK;
        for (int slice = blockIdx.x; slice < total_slices;
             slice += G.num_dispatch_sms)
            dispatch_slice(G, slice);
    } else {
        gemm_sm(G, blockIdx.x - G.num_dispatch_sms);
    }
}

void launch_fused(const fused_globals& G, cudaStream_t stream) {
    cudaFuncSetAttribute(fused_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    fused_kernel<<<G.num_dispatch_sms + G.num_gemm_sms, 256,
                   DYNAMIC_SHARED_MEMORY, stream>>>(G);
}

__global__ __launch_bounds__(NUM_DISPATCH_THREADS, 1)
void dispatch_lsa_kernel(const __grid_constant__ dispatch_globals G) {
    const int total_slices =
        (G.num_output_tokens + dispatch_globals::TOKENS_PER_BLOCK - 1) /
        dispatch_globals::TOKENS_PER_BLOCK;
    for (int slice = blockIdx.x; slice < total_slices; slice += gridDim.x)
        dispatch_slice(G, slice);
}

void launch_dispatch_lsa(const dispatch_globals& G, cudaStream_t stream) {
    cudaFuncSetAttribute(dispatch_lsa_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    dispatch_lsa_kernel<<<G.num_dispatch_sms, NUM_DISPATCH_THREADS,
                          DYNAMIC_SHARED_MEMORY, stream>>>(G);
}

__global__ void dummy_weight_load_kernel(
    const __grid_constant__ weight_load_globals G) {
    __shared__ weight_load_globals::weight_tile tile;
    __shared__ semaphore arrived;

    if (threadIdx.x == 0) {
        init_semaphore(arrived, 0, 1);
        ::dist::tma::expect_bytes(arrived,
                                  sizeof(weight_load_globals::weight_tile));
        ::dist::tma::load_async(tile, G.weights,
                                {G.row_tile, G.col_tile}, arrived);
        wait(arrived, 0);
        ::dist::tma::store_async(G.output, tile, {0, 0});
        ::dist::tma::store_async_wait();
    }
}

void launch_dummy_weight_load(const weight_load_globals& G,
                              cudaStream_t stream) {
    dummy_weight_load_kernel<<<1, 32, 0, stream>>>(G);
}

__global__ __launch_bounds__(256, 1)
void tcgen05_micro_kernel(
    const __grid_constant__ tcgen05_micro_globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);
    tcgen05_micro_globals::A_tile &A_smem =
        allocator.allocate<tcgen05_micro_globals::A_tile>();
    tcgen05_micro_globals::B_tile &B_smem =
        allocator.allocate<tcgen05_micro_globals::B_tile>();
    tcgen05_micro_globals::C_tile &C_smem =
        allocator.allocate<tcgen05_micro_globals::C_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore inputs_finished;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(inputs_finished, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1> tm_allocator;
    using C_tmem = tt<float, 128, 256>;

    const int wg = warpgroup::groupid();
    const int warp_in_wg = warpgroup::warpid();
    const int lane = warp::laneid();

    if (wg == 0) {
        // TMEM allocation is a warp-wide instruction.
        if (warp_in_wg == 0) {
            tm_allocator.provision(tmem_addr);
            if (lane == 0)
                arrive(tmem_provisioned);
        }

        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        C_tmem C_tm = tm_allocator.allocate<C_tmem>(0);

        wait(outputs_arrived, 0);
        rt_bf<32, 32> C_reg;

        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            warpgroup::load_async(
                C_reg, C_tm.template subtile<tt<float, 128, 32>>(0, n * 32));
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            warpgroup::store(C_smem, C_reg);
            warpgroup::sync(1);
            if (warp_in_wg == 0 && lane == 0) {
                ::dist::tma::store_async(G.C, C_smem, {0, n});
                ::dist::tma::store_async_wait();
            }
            warpgroup::sync(1);
        }

        if (warp_in_wg == 0)
            tm_allocator.deprovision();
    } else {
        if (warp_in_wg == 3 && lane == 0) {
            ::dist::tma::expect_bytes(
                inputs_arrived,
                sizeof(tcgen05_micro_globals::A_tile) +
                sizeof(tcgen05_micro_globals::B_tile));
            ::dist::tma::load_async(A_smem, G.A, {0, 0}, inputs_arrived);
            ::dist::tma::load_async(B_smem, G.B, {0, 0}, inputs_arrived);
        } else if (warp_in_wg == 0 && lane == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            C_tmem C_tm = tm_allocator.allocate<C_tmem>(0);
            wait(inputs_arrived, 0);
            warpgroup::mm_AB(C_tm, A_smem, B_smem, inputs_finished);
            tensor_commit<1>(outputs_arrived);
        }
    }
}

void launch_tcgen05_micro(const tcgen05_micro_globals& G,
                          cudaStream_t stream) {
    constexpr int SHMEM_BYTES =
        sizeof(tcgen05_micro_globals::A_tile) +
        sizeof(tcgen05_micro_globals::B_tile) +
        sizeof(tcgen05_micro_globals::C_tile) + 1024;
    cudaFuncSetAttribute(tcgen05_micro_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SHMEM_BYTES);
    tcgen05_micro_kernel<<<1, 256, SHMEM_BYTES, stream>>>(G);
}

}  // namespace moe_dispatch_gemm_blackwell

#include "operators/dispatch_gemm_blackwell/session.cuh"


// [TODO:yihan] replace with tcgen05
