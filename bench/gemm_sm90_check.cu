/**
 * @file gemm_sm90_check.cu
 * @brief Standalone single-GPU correctness + microbench harness for the Gen-15
 *        modular grouped GEMM (gemm_sm90::run). No dispatch/combine/RDMA.
 *
 * Exposes gemm1(A, B, C, padded_tokens_per_expert, local_rb_per_expert,
 * super_m, mode) where mode 0 = legacy union epilogue (byte-identical to the
 * fused kernel's grouped_gemm), mode 1 = decoupled register epilogue (perf
 * gamble). A=[M,H] post-tokens, B=[E,H,2I] weights, C=[M,2I] out, grouped per
 * expert by padded_tokens_per_expert (a NullGate => no dispatch barrier).
 */
#include "operators/dispatch_gemm_glu_combine/gemm.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

using namespace kittens;

namespace gemm_check {

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

// gemm1 dims: H=7168 (K), I2=4096 (N), ROW/COL/RED = 128/256/64.
static constexpr int H = 7168, I2 = 4096;
static constexpr int ROW_BLOCK = 128, COL_BLOCK = 256, RED_BLOCK = 64;
static constexpr int NUM_EXPERTS_PER_DEV = 16;

using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

using A_t_t = dist::local_tensor<bf16, 1, 1, -1, H, sv_bf<H>, A_tile>;
using B_t_t = dist::local_tensor<bf16, 1, NUM_EXPERTS_PER_DEV, H, I2, B_tile>;
using C_t_t = dist::local_tensor<bf16, 1, 1, -1, I2, C_tile>;

using CFG_BASE = gemm_sm90::Config<
    ROW_BLOCK, COL_BLOCK, RED_BLOCK, 4, 4, 1, /*REG*/false, DYNAMIC_SHARED_MEMORY>;
#ifndef SA
#define SA 4
#endif
#ifndef SB
#define SB 3
#endif
#ifndef HINFLIGHT
#define HINFLIGHT 1
#endif
using CFG_PERF = gemm_sm90::Config<
    ROW_BLOCK, COL_BLOCK, RED_BLOCK, SB, SA, HINFLIGHT, /*REG*/true, DYNAMIC_SHARED_MEMORY>;

struct KArgs {
    A_t_t A; B_t_t B; C_t_t C;
    const int* ptpe;   // [E] padded tokens per expert (global mem)
    const int* lrb;    // [E] local rb per expert
    int super_m;
};

__device__ inline void load_layout(const KArgs& a, int* ptpe, int* lrb) {
    if (threadIdx.x < NUM_EXPERTS_PER_DEV) {
        ptpe[threadIdx.x] = a.ptpe[threadIdx.x];
        lrb[threadIdx.x]  = a.lrb[threadIdx.x];
    }
}

template <class Cfg>
__global__ __launch_bounds__(NUM_MAIN_THREADS, 1)
void gemm_only_kernel(const __grid_constant__ KArgs a) {
    extern __shared__ int __shm[];
    __shared__ int ptpe[NUM_EXPERTS_PER_DEV], lrb[NUM_EXPERTS_PER_DEV];
    load_layout(a, ptpe, lrb);
    __syncthreads();
    gemm_sm90::GroupLayout L{ ptpe, lrb, NUM_EXPERTS_PER_DEV,
                              I2 / COL_BLOCK, H / RED_BLOCK, a.super_m };
    // NB: pass the views BY REFERENCE from the grid_constant struct. Copying a
    // local_tensor into a local var would move its CUtensorMap into local
    // memory and TMA from a local-mem descriptor faults.
    gemm_sm90::run<Cfg>(&__shm[0], (int)blockIdx.x, (int)gridDim.x,
                        a.A, a.B, a.C, L, gemm_sm90::NullGate{});
}

void run_gemm(at::Tensor A, at::Tensor B, at::Tensor C,
              at::Tensor ptpe, at::Tensor lrb, int super_m, int mode) {
    const int dev = A.get_device();
    c10::cuda::CUDAGuard g(dev);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev).stream();

    KArgs a{
        dist::local_tensor_from_tensor<A_t_t>(A),
        dist::local_tensor_from_tensor<B_t_t>(B),
        dist::local_tensor_from_tensor<C_t_t>(C),
        ptpe.data_ptr<int>(),
        lrb.data_ptr<int>(),
        super_m,
    };

    if (mode == 0) {
        cudaFuncSetAttribute(gemm_only_kernel<CFG_BASE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);
        gemm_only_kernel<CFG_BASE><<<SM_COUNT, NUM_MAIN_THREADS, DYNAMIC_SHARED_MEMORY, stream>>>(a);
    } else {
        cudaFuncSetAttribute(gemm_only_kernel<CFG_PERF>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);
        gemm_only_kernel<CFG_PERF><<<SM_COUNT, NUM_MAIN_THREADS, DYNAMIC_SHARED_MEMORY, stream>>>(a);
    }
}

}  // namespace gemm_check

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_gemm", &gemm_check::run_gemm,
          "grouped gemm1 (mode 0=union base, 1=reg perf)");
}
