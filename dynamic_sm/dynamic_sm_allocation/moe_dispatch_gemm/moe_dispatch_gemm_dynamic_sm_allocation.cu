#include "pyutils/torchutils.cuh"

#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

// Limited to 2 GPUs: this variant exercises the dispatch/GEMM balancing policy
// without generalising the routing geometry.
#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 2
#endif

#if TK_NUM_DEVICES != 2
#error "moe_dispatch_gemm_dynamic_sm_allocation.cu currently supports exactly 2 GPUs"
#endif

#ifndef TK_MOE_H
#define TK_MOE_H 7168
#endif

#ifndef TK_MOE_I
#define TK_MOE_I 2048
#endif

#ifndef TK_MOE_TOP_K
#define TK_MOE_TOP_K 8
#endif

#ifndef TK_MOE_NUM_EXPERTS
#define TK_MOE_NUM_EXPERTS 256
#endif

namespace dynamic_sm_allocation_moe {

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int NUM_EPILOGUE_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

struct globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int TOP_K = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS = TK_MOE_NUM_EXPERTS;
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / NUM_DEVICES;

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int TOKENS_PER_BLOCK = 16;

    using token_vec = sv_bf<H>;
    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using pre_tokens_pgl = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl = gl<bf16, 1, 1, -1, H, token_vec, A_tile>;
    using weights_gl = gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_gl = gl<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_gl = gl<int, 1, 1, 1, NUM_EXPERTS>;
    using pull_dispatch_indices_gl = gl<int, 1, 1, -1, 2>;
    using barrier_pgl = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

    pre_tokens_pgl pre_tokens;
    post_tokens_gl post_tokens;
    weights_gl weights;
    outputs_gl outputs;
    padded_tokens_per_expert_gl padded_tokens_per_expert;
    pull_dispatch_indices_gl pull_dispatch_indices;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_comm_sms;
    const int num_comp_sms;
    const unsigned long long assist_wait_cycles;
    const int default_assist_blocks;
    const int max_assist_blocks;
    unsigned int* next_dispatch;

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

__device__ inline int load_barrier_value(const globals& G, const int row_idx) {
    int bar_val;
    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                 : "=r"(bar_val)
                 : "l"(&G.barrier[G.dev_idx][{row_idx}])
                 : "memory");
    return bar_val;
}

__device__ inline int choose_assist_blocks(
    const globals& G,
    const unsigned long long waited_cycles,
    const int bar_val) {
    if (waited_cycles < G.assist_wait_cycles) {
        return 0;
    }
    int deficit_tokens = globals::ROW_BLOCK - bar_val;
    if (deficit_tokens <= 0) {
        return 0;
    }
    int deficit_blocks =
        (deficit_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    int assist_blocks = G.default_assist_blocks > 0 ? G.default_assist_blocks : 1;
    if (waited_cycles >= 8ULL * G.assist_wait_cycles) {
        assist_blocks = assist_blocks < 8 ? 8 : assist_blocks;
    } else if (waited_cycles >= 4ULL * G.assist_wait_cycles) {
        assist_blocks = assist_blocks < 4 ? 4 : assist_blocks;
    } else if (waited_cycles >= 2ULL * G.assist_wait_cycles) {
        assist_blocks = assist_blocks < 2 ? 2 : assist_blocks;
    }
    if (assist_blocks > G.max_assist_blocks) {
        assist_blocks = G.max_assist_blocks;
    }
    if (assist_blocks > deficit_blocks) {
        assist_blocks = deficit_blocks;
    }
    return assist_blocks;
}

__device__ inline void dispatch_block(const globals& G, const int dispatch_block_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];

    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int token_idx = dispatch_block_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (token_idx < G.num_padded_local_tokens) {
            const int src_dev_idx = G.pull_dispatch_indices[{token_idx, 0}];
            const int src_token_idx = G.pull_dispatch_indices[{token_idx, 1}];
            if (src_dev_idx >= 0 && src_token_idx >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(
                    token[lane_id],
                    G.pre_tokens[src_dev_idx],
                    {src_token_idx, 0},
                    token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.post_tokens, token[lane_id], {token_idx, 0});
                tma::store_async_wait();
            }
            asm volatile(
                "{red.release.gpu.global.add.s32 [%0], %1;}"
                :
                : "l"(&G.barrier[G.dev_idx][{token_idx / globals::ROW_BLOCK}]), "r"(1)
                : "memory");
        }
    }

    __syncthreads();
}

__device__ inline void run_compute_cta(const globals& G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs& outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(
            &inputs[globals::PIPELINE_STAGES - 1]);

    __shared__ int padded_tokens_per_expert[globals::NUM_EXPERTS_PER_DEV];
    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ unsigned int my_dispatch_task;
    __shared__ int assist_blocks;
    __shared__ int current_row_idx;
    __shared__ int current_col_idx;
    __shared__ int current_expert_id;
    __shared__ bool row_ready;

    if (threadIdx.x < globals::NUM_EXPERTS_PER_DEV) {
        padded_tokens_per_expert[threadIdx.x] = G.padded_tokens_per_expert[
            {G.dev_idx * globals::NUM_EXPERTS_PER_DEV + (int)threadIdx.x}];
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    constexpr int num_iters = globals::H / globals::RED_BLOCK;
    constexpr int col_blocks = globals::I / globals::COL_BLOCK;
    const int num_dispatch_blocks =
        (G.num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;

    for (int task_id = sm_idx, num_tokens_per_expert_cum = 0, expert_id = 0;
         expert_id < globals::NUM_EXPERTS_PER_DEV;
         expert_id++) {
        const int row_block_start = num_tokens_per_expert_cum / globals::ROW_BLOCK;
        num_tokens_per_expert_cum += padded_tokens_per_expert[expert_id];
        const int row_block_end =
            (num_tokens_per_expert_cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
        const int row_blocks = row_block_end - row_block_start;
        const int num_blocks = row_blocks * col_blocks;

        for (; task_id < num_blocks; task_id += G.num_comp_sms) {
            if (threadIdx.x == 0) {
                current_row_idx = task_id / col_blocks + row_block_start;
                current_col_idx = task_id % col_blocks;
                current_expert_id = expert_id;
            }
            __syncthreads();

            const unsigned long long wait_start = clock64();
            while (true) {
                if (threadIdx.x == 0) {
                    const int bar_val = load_barrier_value(G, current_row_idx);
                    row_ready = (bar_val == globals::ROW_BLOCK);
                    assist_blocks =
                        row_ready ? 0 : choose_assist_blocks(G, clock64() - wait_start, bar_val);
                }
                __syncthreads();

                if (row_ready) {
                    break;
                }
                if (assist_blocks == 0) {
                    if (threadIdx.x == 0) {
                        __nanosleep(32);
                    }
                    __syncthreads();
                    continue;
                }

                for (int assist_idx = 0; assist_idx < assist_blocks; ++assist_idx) {
                    if (threadIdx.x == 0) {
                        my_dispatch_task = atomicAdd(G.next_dispatch, 1u);
                    }
                    __syncthreads();
                    if ((int)my_dispatch_task >= num_dispatch_blocks) {
                        break;
                    }
                    dispatch_block(G, (int)my_dispatch_task);
                }
                __syncthreads();
            }

            if (warpgroup_id == 2) {
                warpgroup::decrease_registers<40>();

                if (warp_id == 0 && lane_id == 0) {
                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect_bytes(inputs_arrived[stage],
                                          sizeof(globals::pipeline_inputs));
                        if (red_idx == globals::PIPELINE_STAGES - 1) {
                            wait(outputs_finished,
                                 get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                            update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                        }
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            tma::load_async(inputs[stage].A[i], G.post_tokens,
                                            {current_row_idx * 2 + i, red_idx},
                                            inputs_arrived[stage]);
                        }
                        tma::load_async(inputs[stage].B, G.weights,
                                        {current_expert_id, red_idx, current_col_idx},
                                        inputs_arrived[stage]);
                        stage = (stage + 1) % globals::PIPELINE_STAGES;
                    }
                } else if (warp_id == 1 && lane_id == 0) {
                    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                    update_phasebit<0>(phasebits, 0);
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        tma::store_async(G.outputs, outputs.C[i],
                                         {current_row_idx * 2 + i, current_col_idx});
                    }
                    tma::store_async_read_wait();
                    arrive(outputs_finished);
                }
            } else {
                warpgroup::increase_registers<232>();
                rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
                warp::zero(C_accum);

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id],
                                      inputs[stage].B);
                    warpgroup::mma_async_wait();
                    warp::arrive(inputs_finished[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }

                group<8>::sync(3);
                warpgroup::store(outputs.C[warpgroup_id], C_accum);
                warpgroup::sync(warpgroup_id + 1);
                warpgroup::arrive(outputs_arrived);
            }
        }
        task_id -= num_blocks;
    }
}

__global__ __launch_bounds__(NUM_MAIN_THREADS, 1)
void adaptive_dispatch_group_gemm_kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms) {
        run_compute_cta(G, blockIdx.x);
        return;
    }

    while (true) {
        __shared__ unsigned int my_dispatch_task;
        const int num_dispatch_blocks =
            (G.num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) /
            globals::TOKENS_PER_BLOCK;
        if (threadIdx.x == 0) {
            my_dispatch_task = atomicAdd(G.next_dispatch, 1u);
        }
        __syncthreads();
        if ((int)my_dispatch_task >= num_dispatch_blocks) {
            break;
        }
        dispatch_block(G, (int)my_dispatch_task);
    }
}

__global__ __launch_bounds__(NUM_EPILOGUE_THREADS)
void epilogue_kernel(const __grid_constant__ globals G) {
    const int num_blocks =
        (G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride) {
        G.barrier[G.dev_idx][{i}] = 0;
    }
}

static unsigned int* g_dispatch_counter[globals::NUM_DEVICES] = {nullptr};

}  // namespace dynamic_sm_allocation_moe

void entrypoint_dynamic_sm_allocation_moe(
    kittens::py::TKParallelTensor& pre_tokens,
    at::Tensor& post_tokens,
    at::Tensor& weights,
    at::Tensor& outputs,
    at::Tensor& padded_tokens_per_expert,
    at::Tensor& pull_dispatch_indices,
    kittens::py::TKParallelTensor& barrier,
    const int num_padded_local_tokens,
    const int num_comm_sms,
    const unsigned long long assist_wait_cycles,
    const int default_assist_blocks,
    const int max_assist_blocks) {
    using namespace dynamic_sm_allocation_moe;

    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_dispatch_counter[dev_idx] == nullptr) {
        cudaMalloc(&g_dispatch_counter[dev_idx], sizeof(unsigned int));
    }
    cudaMemsetAsync(g_dispatch_counter[dev_idx], 0, sizeof(unsigned int), stream);

    int clamped_num_comm_sms = num_comm_sms;
    if (clamped_num_comm_sms < 1) {
        clamped_num_comm_sms = 1;
    }
    if (clamped_num_comm_sms >= SM_COUNT) {
        clamped_num_comm_sms = SM_COUNT - 1;
    }

    globals G {
        .pre_tokens =
            kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .post_tokens = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert =
            kittens::py::tensor_to_gl<globals::padded_tokens_per_expert_gl>(
                padded_tokens_per_expert),
        .pull_dispatch_indices =
            kittens::py::tensor_to_gl<globals::pull_dispatch_indices_gl>(
                pull_dispatch_indices),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_comm_sms = clamped_num_comm_sms,
        .num_comp_sms = SM_COUNT - clamped_num_comm_sms,
        .assist_wait_cycles = assist_wait_cycles,
        .default_assist_blocks = default_assist_blocks,
        .max_assist_blocks = max_assist_blocks,
        .next_dispatch = g_dispatch_counter[dev_idx],
    };

    cudaFuncSetAttribute(adaptive_dispatch_group_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYNAMIC_SHARED_MEMORY);
    adaptive_dispatch_group_gemm_kernel<<<SM_COUNT, NUM_MAIN_THREADS,
                                          DYNAMIC_SHARED_MEMORY, stream>>>(G);

    const int epilogue_blocks =
        ((G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK +
         NUM_EPILOGUE_THREADS - 1) /
        NUM_EPILOGUE_THREADS;
    epilogue_kernel<<<epilogue_blocks, NUM_EPILOGUE_THREADS, 0, stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("moe_dispatch_gemm_dynamic_sm_allocation", &entrypoint_dynamic_sm_allocation_moe,
          pybind11::arg("pre_tokens"),
          pybind11::arg("post_tokens"),
          pybind11::arg("weights"),
          pybind11::arg("outputs"),
          pybind11::arg("padded_tokens_per_expert"),
          pybind11::arg("pull_dispatch_indices"),
          pybind11::arg("barrier"),
          pybind11::arg("num_padded_local_tokens"),
          pybind11::arg("num_comm_sms"),
          pybind11::arg("assist_wait_cycles"),
          pybind11::arg("default_assist_blocks"),
          pybind11::arg("max_assist_blocks"));
}
