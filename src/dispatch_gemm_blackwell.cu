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

__device__ inline void dispatch_slice(const dispatch_globals &G, int slice) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(&__shm[0]);
    dispatch_globals::token_vec (&tokens)[dispatch_globals::TOKENS_PER_BLOCK] =
        allocator.allocate<dispatch_globals::token_vec,
                           dispatch_globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore arrived[dispatch_globals::TOKENS_PER_BLOCK];

    const int lane = threadIdx.x;
    const int dst_token = slice * dispatch_globals::TOKENS_PER_BLOCK + lane;
    const bool valid_dst =
        lane < dispatch_globals::TOKENS_PER_BLOCK &&
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
                                      sizeof(dispatch_globals::token_vec));
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
            dispatch_globals::ROW_BLOCK / dispatch_globals::TOKENS_PER_BLOCK;
        const int row_block = slice / SLICES_PER_ROW_BLOCK;
        comm::atomic_u32::release_add_gpu(
            &G.row_ready[{row_block}], __popc(valid_mask));
    }
}

// Consumer-side half of the dispatch/GEMM handoff. The future tcgen05 loader
// calls this before TMA-loading an A tile from the corresponding row block.
__device__ inline void wait_row_ready(const dispatch_globals &G,
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

}  // namespace moe_dispatch_gemm_blackwell

#include "operators/dispatch_gemm_blackwell/session.cuh"


// [TODO:yihan] replace with tcgen05
