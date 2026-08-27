#include "pyutils/torchutils.cuh"

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace gemm_ar {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
#ifdef GEMM_AR_NUM_BLOCKS
    static constexpr int NUM_BLOCKS = GEMM_AR_NUM_BLOCKS;
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
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;
    using barrier_pgl = barrier_t<NUM_DEVICES>;

    A_gl A;
    B_gl B;
    C_pgl C;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

__device__ inline void comm_sm(const globals &G) {
    int comm_sm_id = blockIdx.x - G.num_comp_sms;
    int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / globals::COL_BLOCK;
    int super_rows = (row_blocks / globals::SUPER_M) * globals::SUPER_M;
    int final_rows = row_blocks - super_rows;
    int super_blocks = globals::SUPER_M * col_blocks;
    int num_blocks = row_blocks * col_blocks;

    for (int task_id = globals::NUM_DEVICES * comm_sm_id + G.dev_idx;
         task_id < num_blocks;
         task_id += globals::NUM_DEVICES * G.num_comm_sms) {
        int real_task_id = task_id;
        int row_idx, col_idx;
        if (real_task_id < super_rows * col_blocks) {
            row_idx = globals::SUPER_M * (real_task_id / super_blocks) + real_task_id % globals::SUPER_M;
            col_idx = (real_task_id % super_blocks) / globals::SUPER_M;
        } else {
            int remainder_id = real_task_id - super_rows * col_blocks;
            row_idx = super_rows + remainder_id % final_rows;
            col_idx = remainder_id / final_rows;
        }

        if (threadIdx.x == 0) {
            wait(G.barrier, {row_idx, col_idx}, G.dev_idx, globals::NUM_DEVICES);
        }
        __syncthreads();

        group<config::NUM_WARPS>::all_reduce<globals::ROW_BLOCK, globals::COL_BLOCK, reduce_op::ADD>(G.C, {row_idx, col_idx});
    }
}

__device__ inline void comp_sm(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    static_assert(
        sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) + sizeof(globals::pipeline_outputs)
            <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = *reinterpret_cast<globals::pipeline_outputs *>(&inputs[globals::PIPELINE_STAGES - 1]);

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

    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / globals::COL_BLOCK;
    int super_rows = (row_blocks / globals::SUPER_M) * globals::SUPER_M;
    int final_rows = row_blocks - super_rows;
    int super_blocks = globals::SUPER_M * col_blocks;
    int num_blocks = row_blocks * col_blocks;
    int num_iters = G.A.cols() / globals::RED_BLOCK;

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
                int real_task_id = task_id;
                int row_idx, col_idx;
                if (real_task_id < super_rows * col_blocks) {
                    row_idx = globals::SUPER_M * (real_task_id / super_blocks) + real_task_id % globals::SUPER_M;
                    col_idx = (real_task_id % super_blocks) / globals::SUPER_M;
                } else {
                    int remainder_id = real_task_id - super_rows * col_blocks;
                    row_idx = super_rows + remainder_id % final_rows;
                    col_idx = remainder_id / final_rows;
                }

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
            }
        } else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
                int real_task_id = task_id;
                int row_idx, col_idx;
                if (real_task_id < super_rows * col_blocks) {
                    row_idx = globals::SUPER_M * (real_task_id / super_blocks) + real_task_id % globals::SUPER_M;
                    col_idx = (real_task_id % super_blocks) / globals::SUPER_M;
                } else {
                    int remainder_id = real_task_id - super_rows * col_blocks;
                    row_idx = super_rows + remainder_id % final_rows;
                    col_idx = remainder_id / final_rows;
                }

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::store_async(G.C[G.dev_idx], outputs.C[i], {row_idx * 2 + i, col_idx});
                }
                tma::store_async_read_wait();
                arrive(outputs_finished);

                int signal_dev_idx = task_id % globals::NUM_DEVICES;
                signal(G.barrier, {row_idx, col_idx}, signal_dev_idx, 1);
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < num_blocks; task_id += G.num_comp_sms) {
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
}

__device__ inline void main_kernel(const globals &G) {
    if (blockIdx.x < G.num_comp_sms) {
        comp_sm(G);
    } else {
        comm_sm(G);
    }
}

__device__ inline void epilogue_kernel(const globals &G) {
    const int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride) {
        G.barrier[G.dev_idx][{i / col_blocks, i % col_blocks}] = 0;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        barrier_all(G.barrier, {1, 0, 0}, G.dev_idx);
    }
}

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    int num_comm_sms
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");
    TORCH_CHECK(A.dtype() == at::ScalarType::BFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be rank-2");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul shape mismatch: A[M,K], B[K,N]");

    kittens::py::parallel_tensor_check(C, barrier);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    TORCH_CHECK(M % globals::ROW_BLOCK == 0, "M must be divisible by ROW_BLOCK (", globals::ROW_BLOCK, ")");
    TORCH_CHECK(K % globals::RED_BLOCK == 0, "K must be divisible by RED_BLOCK (", globals::RED_BLOCK, ")");
    TORCH_CHECK(N % globals::COL_BLOCK == 0, "N must be divisible by COL_BLOCK (", globals::COL_BLOCK, ")");
    TORCH_CHECK(C.data_.dtype() == at::ScalarType::BFloat16, "C must be bf16");
    TORCH_CHECK(C.data_.size(0) == M && C.data_.size(1) == N, "C shape must be [M, N]");

    TORCH_CHECK(barrier.data_.dtype() == at::ScalarType::Int, "barrier tensor dtype must be int32");

    const int max_comm_sms = config::NUM_BLOCKS - 1;
    num_comm_sms = std::max(1, std::min(max_comm_sms, num_comm_sms));

    globals G {
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<globals::C_pgl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = config::NUM_BLOCKS - num_comm_sms,
    };

    kittens::py::launch_kernel<config, globals, main_kernel>(G);
    kittens::py::launch_kernel<config, globals, epilogue_kernel>(G);
}

}  // namespace gemm_ar

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef TK_PARALLEL_TENSOR_BINDINGS_EXTERNAL
    BIND_TK_PARALLEL_TENSOR(m);
#endif
    m.def("matmul_all_reduce", &gemm_ar::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("num_comm_sms"));
}
