#include "pyutils/torchutils.cuh"
#include "../common/dynamic_sm_allocation_utils.cuh"

#include <c10/cuda/CUDAGuard.h>

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace static_gemm_rs {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
#ifdef GEMM_RS_NUM_BLOCKS
    static constexpr int NUM_BLOCKS = GEMM_RS_NUM_BLOCKS;
#else
    static constexpr int NUM_BLOCKS = 132;  // H100 default: one CTA per SM
#endif

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

struct globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using workspace_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using output_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;

    A_gl A;
    B_gl B;
    workspace_gl workspace;
    output_pgl output;
    int *ready;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
    unsigned int *next_compute;
    unsigned int *next_comm;

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

__device__ inline int map_task_id(
    const int logical_task_id,
    const int num_blocks,
    const int dev_idx
) {
    const int dev_task_offset =
        ((dev_idx + 1) * (num_blocks / globals::NUM_DEVICES)) % num_blocks;
    return (logical_task_id + dev_task_offset) % num_blocks;
}

__device__ inline void signal_ready(int *ready, const int task_id) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" :: "l"(ready + task_id), "r"(1u) : "memory");
}

__device__ inline void wait_ready(const int *ready, const int task_id) {
    unsigned int val;
    do {
        asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}" : "=r"(val) : "l"(ready + task_id) : "memory");
        if (val != 1u) {
            __nanosleep(16);
        }
    } while (val != 1u);
}

__device__ inline void compute_tile(
    const globals &G,
    const int task_id,
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES],
    globals::pipeline_outputs &outputs,
    semaphore (&inputs_arrived)[globals::PIPELINE_STAGES],
    semaphore (&inputs_finished)[globals::PIPELINE_STAGES],
    semaphore &outputs_arrived,
    semaphore &outputs_finished,
    int &stage,
    uint32_t &phasebits,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks,
    const int num_iters
) {
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();

    int row_idx;
    int col_idx;
    decode_task_id<globals>(task_id, row_blocks, col_blocks, super_rows, final_rows, super_blocks, row_idx, col_idx);

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        if (warp_id == 0 && lane_id == 0) {
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                if (red_idx == globals::PIPELINE_STAGES - 1) {
                    wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                    update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                }
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::load_async(inputs[stage].A[i], G.A, {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                }
                tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) {
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                tma::store_async(G.workspace, outputs.C[i], {row_idx * 2 + i, col_idx});
            }
            tma::store_async_read_wait();
            signal_ready(G.ready, task_id);
            arrive(outputs_finished);
        }
    } else {
        rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
        warp::zero(C_accum);

        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
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

__device__ inline void comm_tile(
    const globals &G,
    const int task_id,
    const int row_blocks,
    const int col_blocks,
    const int super_rows,
    const int final_rows,
    const int super_blocks
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    globals::C_tile (&partials)[2] = allocator.allocate<globals::C_tile, 2>();

    __shared__ semaphore partials_arrived[2];
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            init_semaphore(partials_arrived[i], 0, 1);
        }
    }
    __syncthreads();

    int row_idx;
    int col_idx;
    decode_task_id<globals>(task_id, row_blocks, col_blocks, super_rows, final_rows, super_blocks, row_idx, col_idx);
    wait_ready(G.ready, task_id);

    const int row_blocks_per_dev = row_blocks / globals::NUM_DEVICES;
    const int owner_dev_idx = row_idx / row_blocks_per_dev;
    const int local_row_idx = row_idx % row_blocks_per_dev;
    const int warp_id = warp::groupid();

    if (warp_id == 0 && laneid() == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            tma::expect_bytes(partials_arrived[i], sizeof(globals::C_tile));
            tma::load_async(partials[i], G.workspace, {row_idx * 2 + i, col_idx}, partials_arrived[i]);
        }
    } else if (warp_id == 1 && laneid() == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            wait(partials_arrived[i], 0);
            tma::store_add_async(G.output[owner_dev_idx], partials[i], {local_row_idx * 2 + i, col_idx});
        }
        tma::store_async_read_wait();
    }

    __syncthreads();
}

__device__ inline void main_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);

    static_assert(
        sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) +
                sizeof(globals::pipeline_outputs) <=
            config::DYNAMIC_SHARED_MEMORY
    );
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs =
        *reinterpret_cast<globals::pipeline_outputs *>(
            &inputs[globals::PIPELINE_STAGES - 1]
        );

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;

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

    const int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int super_rows = (row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;
    const int num_blocks = row_blocks * col_blocks;
    const int num_iters = G.A.cols() / globals::RED_BLOCK;

    if ((int)blockIdx.x < G.num_comp_sms) {
        while (true) {
            __shared__ int claimed_compute_task;
            if (threadIdx.x == 0) {
                claimed_compute_task =
                    (int)atomicAdd((unsigned int *)G.next_compute, 1u);
            }
            __syncthreads();
            if (claimed_compute_task >= num_blocks) {
                break;
            }
            int stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            compute_tile(
                G,
                map_task_id(claimed_compute_task, num_blocks, G.dev_idx),
                inputs,
                outputs,
                inputs_arrived,
                inputs_finished,
                outputs_arrived,
                outputs_finished,
                stage,
                phasebits,
                row_blocks,
                col_blocks,
                super_rows,
                final_rows,
                super_blocks,
                num_iters
            );
        }
    } else {
        while (true) {
            __shared__ int claimed_comm_task;
            if (threadIdx.x == 0) {
                claimed_comm_task = (int)atomicAdd((unsigned int *)G.next_comm, 1u);
            }
            __syncthreads();
            if (claimed_comm_task >= num_blocks) {
                break;
            }
            comm_tile(
                G,
                map_task_id(claimed_comm_task, num_blocks, G.dev_idx),
                row_blocks,
                col_blocks,
                super_rows,
                final_rows,
                super_blocks
            );
        }
    }
}

static unsigned int *g_counters[globals::NUM_DEVICES] = {nullptr};

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &workspace,
    kittens::py::TKParallelTensor &output,
    const at::Tensor &ready,
    int num_comm_sms
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(workspace.is_cuda() && ready.is_cuda(), "workspace/ready must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && workspace.is_contiguous() && ready.is_contiguous(),
                "A/B/workspace/ready must be contiguous");
    TORCH_CHECK(A.dtype() == at::ScalarType::BFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");
    TORCH_CHECK(workspace.dtype() == at::ScalarType::BFloat16, "workspace must be bf16");
    TORCH_CHECK(ready.dtype() == at::ScalarType::Int, "ready tensor dtype must be int32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be rank-2");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul shape mismatch: A[M,K], B[K,N]");

    kittens::py::parallel_tensor_check(output);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));
    const int row_blocks = M / globals::ROW_BLOCK;
    const int col_blocks = N / globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;

    TORCH_CHECK(M % globals::ROW_BLOCK == 0, "M must be divisible by ROW_BLOCK (", globals::ROW_BLOCK, ")");
    TORCH_CHECK(K % globals::RED_BLOCK == 0, "K must be divisible by RED_BLOCK (", globals::RED_BLOCK, ")");
    TORCH_CHECK(N % globals::COL_BLOCK == 0, "N must be divisible by COL_BLOCK (", globals::COL_BLOCK, ")");
    TORCH_CHECK(workspace.size(0) == M && workspace.size(1) == N, "workspace shape must be [M, N]");
    TORCH_CHECK(ready.dim() == 1 && ready.size(0) == num_blocks,
                "ready shape must be [M/ROW_BLOCK * N/COL_BLOCK]");
    TORCH_CHECK(output.data_.dtype() == at::ScalarType::BFloat16, "output must be bf16");

    const int max_comm_sms = config::NUM_BLOCKS - 1;
    num_comm_sms = std::max(1, std::min(max_comm_sms, num_comm_sms));

    const int dev_idx = output.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    if (g_counters[dev_idx] == nullptr) {
        cudaMalloc(&g_counters[dev_idx], 2 * sizeof(unsigned int));
    }
    cudaMemsetAsync(g_counters[dev_idx], 0, 2 * sizeof(unsigned int), stream);

    globals G {
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .workspace = kittens::py::tensor_to_gl<globals::workspace_gl>(workspace),
        .output = kittens::py::parallel_tensor_to_pgl<globals::output_pgl>(output),
        .ready = ready.data_ptr<int>(),
        .dev_idx = dev_idx,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = config::NUM_BLOCKS - num_comm_sms,
        .next_compute = &g_counters[dev_idx][0],
        .next_comm = &g_counters[dev_idx][1],
    };

    kittens::py::launch_kernel<config, globals, main_kernel>(G);
}

} // namespace static_gemm_rs

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("matmul_reduce_scatter",
          &static_gemm_rs::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("workspace"),
          pybind11::arg("output"),
          pybind11::arg("ready"),
          pybind11::arg("num_comm_sms"));
}
