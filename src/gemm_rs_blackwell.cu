/**
 * @file gemm_rs_blackwell.cu
 * @brief Single NVLink-domain GEMM + Reduce-Scatter on Blackwell (tcgen05).
 *
 * The Blackwell counterpart to gemm_rs.cu, split out so neither file carries
 * the other's MMA path. Intra-node only, like dispatch_gemm_blackwell.cu and
 * gemm_ar_blackwell.cu: compute CTAs issue the reduce-scatter store_add into
 * the owner's staging buffer directly over NVLink, so there are no send or
 * reduce CTAs and no RDMA arrival protocol. The multi-node path stays in
 * gemm_rs.cu.
 *
 * Two CTA groups, both persistent:
 *   Compute CTAs [0, num_comp_sms):  2-CTA tcgen05 clusters. Each cluster task
 *     covers ROW_BLOCKS_PER_CLUSTER row blocks of one column block, and the
 *     epilogue store_adds the partials into the owning GPU's staging buffer.
 *
 * Python/session glue + pybind module live in
 *   include/operators/gemm_rs/gemm_rs_blackwell_session.cuh
 */
#include "operators/gemm_rs/gemm_rs_blackwell.cuh"

namespace gemm_rs_intranode_blackwell {

// holds exactly two. Spend them on two row blocks of the same task rather than
// on double-buffering one: sharing a B tile across both halves doubles
// arithmetic intensity, which measurement showed matters more than overlapping
// the epilogue (see README_B300 s3.4).
static constexpr int ROW_BLOCKS_PER_TASK = intra_globals::ROW_BLOCKS_PER_TASK;

template <typename G>
__device__ inline void compute_tile_impl(
    const G &Gv, int row_idx, int col_idx, int ready_idx,
    typename G::pipeline_inputs (&inputs)[G::PIPELINE_STAGES],
    typename G::pipeline_outputs &outputs,
    semaphore (&inputs_arrived)[G::PIPELINE_STAGES],
    semaphore (&inputs_finished)[G::PIPELINE_STAGES],
    semaphore &outputs_arrived, semaphore &outputs_finished,
    int &stage, uint32_t &phasebits,
    int row_blocks, int col_blocks, int num_iters
    , tt<float, G::ROW_BLOCK, G::COL_BLOCK * G::ROW_BLOCKS_PER_TASK> &d_tt_pool,
    semaphore &mma_done, semaphore (&tmem_free)[G::ROW_BLOCKS_PER_TASK],
    semaphore &outputs_free, int ctarank, bool write_workspace
    )
{
    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();

    __builtin_assume(row_idx >= 0 && row_idx < row_blocks);
    __builtin_assume(col_idx >= 0 && col_idx < col_blocks);

    if (wg_id == config::NUM_WARPGROUPS - 1) {
        if (w_id == 0 && l_id == 0) {
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                // Every CTA loads its own slice (mask = 1 << ctarank) but points
                // the transaction count at CTA 0, so the leader's single wait
                // covers the whole cluster's inputs. The expect() that balances
                // these bytes is issued by the two MMA warps below.
                //
                // A: this CTA's two row blocks. B: this CTA's half of N -- the
                // pair together covers COL_BLOCK, read once per cluster.
                #pragma unroll
                for (int h = 0; h < G::ROW_BLOCKS_PER_TASK; h++)
                    tma::cluster::load_async(inputs[stage].A[h], Gv.A,
                                             {row_idx + 2 * ctarank + h, red_idx},
                                             inputs_arrived[stage],
                                             (uint16_t)(1 << ctarank), 0);
                tma::cluster::load_async(inputs[stage].B, Gv.B,
                                         {2 * col_idx + ctarank, red_idx}, inputs_arrived[stage],
                                         (uint16_t)(1 << ctarank), 0);
                stage = (stage + 1) % G::PIPELINE_STAGES;
            }
        }
        else if ((w_id == 2 || w_id == 3) && l_id == 0 && ctarank == 0) {
            // Blackwell MMA issuers. mm2_ABt spans the CTA pair: A is split by
            // rows (each CTA keeps its own 128 accumulator rows) and B by
            // columns, so the leader issues one MMA per accumulator index and
            // both CTAs' operands feed it. Only ctarank 0 issues.
            const int acc = w_id - 2;                    // accumulator 0 or 1
            using acc_t = tt<float, G::ROW_BLOCK, G::COL_BLOCK>;
            auto d = d_tt_pool.template subtile<acc_t>(0, G::COL_BLOCK * acc);

            // This accumulator must be drained cluster-wide before the
            // accumulate=0 MMA overwrites it. Per-accumulator, so the warp
            // owning acc 0 is released as soon as acc 0's registers are read,
            // without waiting on the rest of the epilogue.
            tma::cluster::wait(tmem_free[acc], get_phasebit<1>(phasebits, acc));
            update_phasebit<1>(phasebits, acc);

            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                // One expect per MMA warp; together they account for the six
                // tile loads (three per CTA) that the loaders redirected here.
                tma::cluster::expect(inputs_arrived[stage],
                                     inputs[0].A[0], inputs[0].A[1], inputs[0].B);
                tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                // commit<2> multicasts, so each CTA's inputs_finished sees both
                // MMAs and its loader is released in step with the leader.
                if (red_idx == 0)
                    warp::mm2_ABt (d, inputs[stage].A[acc], inputs[stage].B, inputs_finished[stage]);
                else
                    warp::mma2_ABt(d, inputs[stage].A[acc], inputs[stage].B, inputs_finished[stage]);
                stage = (stage + 1) % G::PIPELINE_STAGES;
            }
            kittens::tensor_commit<2>(mma_done);
        }
        else if (w_id == 1 && l_id == 0) {
            // One staging tile for four 64-row halves (two row blocks x two
            // halves), drained in sequence. outputs_free hands the tile back to
            // the consumers between sub-rounds; outputs_finished closes the
            // task after the last one.
            int sub = 0;
            #pragma unroll
            for (int h = 0; h < G::ROW_BLOCKS_PER_TASK; h++) {
                const int rb = row_idx + 2 * ctarank + h;
                const int row_blocks_per_dev_fuse = row_blocks / G::NUM_DEVICES;
                const int owner_dev_idx_fuse = rb / row_blocks_per_dev_fuse;
                const int local_row_idx_at_owner_fuse =
                    rb - owner_dev_idx_fuse * row_blocks_per_dev_fuse;
                const int col_blocks_local_fuse = (int)(Gv.B.rows() / G::COL_BLOCK);
                const int global_tile_idx_owner_fuse =
                    local_row_idx_at_owner_fuse * col_blocks_local_fuse + col_idx;
                #pragma unroll
                for (int hs = 0; hs < 2; hs++) {
                    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                    update_phasebit<0>(phasebits, 0);
                    // workspace is only ever read back by the inter-node
                    // reduce path (fused_comm_tile_impl). With no session that
                    // path never launches, so the copy is dead work - half the
                    // epilogue's TMA stores and half its global write traffic.
                    if (write_workspace)
                        tma::store_async(Gv.workspace[Gv.dev_idx], outputs.C,
                                         {rb * 2 + hs, col_idx});
                    tma::store_add_async(Gv.staging[owner_dev_idx_fuse], outputs.C,
                                         {2 * global_tile_idx_owner_fuse + hs, 0});
                    tma::store_async_read_wait();
                    if (++sub == 2 * G::ROW_BLOCKS_PER_TASK) arrive(outputs_finished);
                    else                                      arrive(outputs_free);
                }
                signal_ready(Gv.ready, ready_idx + (2 * ctarank + h) * col_blocks);
            }
        }
    } else {
        // Blackwell consumers do no MMA. They wait for the tensor-memory
        // accumulator, pull their 16-row slice into registers, release tmem,
        // then stage the tile in shared for the epilogue TMA warp.
        //
        // Row mapping follows ThunderKittens' group<8> tmem layout: the 8
        // consumer warps (2 warpgroups x 4 warps) each own 16 of the 128 rows,
        // at `32*(w%4) + 16*(w/4)`. Two warps share each 32-lane tmem
        // sub-partition, which is the access granularity tcgen05.ld requires.
        const int cw   = wg_id * 4 + w_id;                  // consumer warp 0..7
        const int trow = 32 * (cw % 4) + 16 * (cw / 4);     // its tmem row
        rt_fl<G::ROW_BLOCK / 8, G::COL_BLOCK> C_accum;

        // One signal covers both accumulators: tcgen05.commit fires only after
        // every MMA issued before it has completed.
        wait(mma_done, get_phasebit<0>(phasebits, 0));
        update_phasebit<0>(phasebits, 0);

        #pragma unroll
        for (int h = 0; h < G::ROW_BLOCKS_PER_TASK; h++) {
            warp::load_async(C_accum,
                             d_tt_pool.template subtile<tt<float, G::ROW_BLOCK / 8, G::COL_BLOCK>>(
                                 trow, G::COL_BLOCK * h));
            kittens::tensor_load_wait();
            group<8>::sync(3);
            // Registers hold this accumulator now, so tensor memory is free
            // even though the stores are still ahead.
            if (w_id == 0 && l_id == 0) tma::cluster::arrive(tmem_free[h], 0, 1);

            // The single staging tile takes one 64-row half at a time. Under
            // the group<8> tmem layout warps {0,1,4,5} own rows < 64 and
            // {2,3,6,7} own the rest, so each sub-round only four of the eight
            // warps write -- the others just ride the barrier.
            #pragma unroll
            for (int hs = 0; hs < 2; hs++) {
                if (h == 0 && hs == 0) {
                    // First sub-round of the task: wait on the previous task.
                    // The <1> half starts signalled, so task 0 passes through.
                    wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                    update_phasebit<1>(phasebits, 0);
                } else {
                    wait(outputs_free, get_phasebit<0>(phasebits, 1));
                    update_phasebit<0>(phasebits, 1);
                }
                if (trow / 64 == hs) {
                    auto dst = outputs.C.template subtile<G::ROW_BLOCK / 8, G::COL_BLOCK>(
                                   {(trow % 64) / (G::ROW_BLOCK / 8), 0});
                    warp::store(dst, C_accum);
                }
                group<8>::sync(4);
                if (wg_id == 0 && w_id == 0 && l_id == 0) arrive(outputs_arrived);
            }
        }
    }
}

// Core fused_comm_tile: takes (row_idx, col_idx, ready_idx) directly.
// ready_idx is the flat index for the ready[] signal (must match compute_tile).

__device__ inline void fused_kernel(const fused_globals &G) {
    const auto &I = G.intra;
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    intra_globals::pipeline_inputs (&inputs)[intra_globals::PIPELINE_STAGES] =
        allocator.allocate<intra_globals::pipeline_inputs, intra_globals::PIPELINE_STAGES>();
    // Own allocation, so the loader never waits on the epilogue.
    intra_globals::pipeline_outputs &outputs =
        allocator.allocate<intra_globals::pipeline_outputs>();

    __shared__ semaphore inputs_arrived[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    // mma_done : MMA warp -> consumers, "tensor-memory accumulator is complete"
    // tmem_free: consumers -> MMA warp, "tensor memory drained, safe to reuse"
    // outputs_free: store warp -> consumers, "shared C drained, next row
    // block of the pair may be staged".
    __shared__ semaphore mma_done;
    __shared__ semaphore tmem_free[intra_globals::ROW_BLOCKS_PER_TASK];
    __shared__ semaphore outputs_free;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < intra_globals::PIPELINE_STAGES; ++i) {
            // One expect() per MMA warp on the leader CTA.
            init_semaphore(inputs_arrived[i], 0, 2);
            // Blackwell: the single tcgen05 MMA releases the stage (one commit),
            // instead of 8 consumer warps each arriving after their wgmma.
            // Both MMA warps commit, and commit<2> multicasts to both CTAs.
            init_semaphore(inputs_finished[i], 0, 2);
        }
        init_semaphore(outputs_arrived, 0, 1);   // one signal per epilogue sub-round
        init_semaphore(outputs_finished, 0, 1);
        init_semaphore(mma_done,     0, 2);   // one commit per accumulator
        #pragma unroll
        for (int i = 0; i < intra_globals::ROW_BLOCKS_PER_TASK; ++i)
            init_semaphore(tmem_free[i], 0, 4);   // 2 warpgroups x 2 CTAs
        init_semaphore(outputs_free, 0, 1);   // one per store-warp round
    }
    __syncthreads();

    // Tensor-memory allocation is CTA-wide (the ctor runs tcgen05.alloc on warp
    // 0 then bar.sync 0), so it must be reached by every thread of the CTA and
    // must sit outside the role dispatch below.
    // Cluster-wide sync: every CTA's semaphores must exist before any peer
    // redirects a transaction or arrival at them.
    everyone::tma::cluster::sync();
    tensor_allocator<1, 2> tm_alloc{};
    // All 512 columns: one 128x256 fp32 accumulator per row block of the task.
    auto d_tt_pool = tm_alloc.allocate<tt<float, intra_globals::ROW_BLOCK,
                                                intra_globals::COL_BLOCK *
                                                intra_globals::ROW_BLOCKS_PER_TASK>>(0);

    const int row_blocks = I.A.rows() / intra_globals::ROW_BLOCK;
    const int col_blocks = I.B.rows() / intra_globals::COL_BLOCK;   // B is (N, K)
    const int num_blocks = row_blocks * col_blocks;
    const int num_iters = I.A.cols() / intra_globals::RED_BLOCK;

    const int row_blocks_per_slice = row_blocks / intra_globals::NUM_DEVICES;

    // CTA role dispatch:
    // - Compute: static stride claiming
    // - Send: static row-block ownership + coalesced RDMA
    // - Reduce: dedicated CTAs, work-stealing claim with remote_arrived_flag
    if ((int)blockIdx.x < I.num_comp_sms) {
        // Phase bits persist across tasks because the pipeline semaphores are
        // shared CTA state.
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        // Each task covers ROW_BLOCKS_PER_TASK adjacent row blocks that share a
        // B tile. The decode lays tasks out as (round = row block within slice,
        // slice, column) with column varying fastest, so the partner of task_id
        // is exactly task_id + tiles_per_round: same slice, same column, next
        // row block. Same slice also means the same owner device, so the
        // staging math below is unchanged.
        const int tiles_per_round = intra_globals::NUM_DEVICES * col_blocks;
        const int num_cluster_tasks = num_blocks / intra_globals::ROW_BLOCKS_PER_CLUSTER;
        const int ctarank    = cluster_ctarank();
        const int cluster_id = (int)blockIdx.x / config::CLUSTER_SIZE;
        const int num_clusters = I.num_comp_sms / config::CLUSTER_SIZE;
        // Banded ordering pays only once the row space is deep enough to hold
        // several bands; below that the column-fastest order already wraps into
        // the next row and covers a compact block by itself. Measured, cluster
        // rows (= bands x SUPER_M) against the supertile's delta:
        //   M=4096   8 rows / 1 band  +2%      M=8192  16 /  2  ~0%
        //   M=16384 32 rows / 4 bands +6%      M=24576 48 /  6  -6%
        //   M=32768 64 rows / 8 bands -8%
        // The crossover sits between 4 and 6 bands; it was not localised
        // further, so the threshold is the measurement, not a derivation.
        const int cluster_rows_total =
            (row_blocks_per_slice * intra_globals::NUM_DEVICES)
            / intra_globals::ROW_BLOCKS_PER_CLUSTER;
        const bool use_supertile = GEMM_RS_SUPERTILE_ENABLED &&
                                   (cluster_rows_total >= 6 * GEMM_RS_SUPER_M);
        for (int pair_id = cluster_id; pair_id < num_cluster_tasks; pair_id += num_clusters) {
            int row_idx, col_idx;
            gemm_rs_decode_cluster<intra_globals>(pair_id, row_blocks_per_slice, col_blocks,
                                                  I.dev_idx, tiles_per_round, use_supertile,
                                                  row_idx, col_idx);
            const int ready_idx = row_idx * col_blocks + col_idx;
            const int row_idx_base = row_idx;
            compute_tile_impl<intra_globals>(I, row_idx, col_idx, ready_idx,
                                              inputs, outputs, inputs_arrived, inputs_finished,
                                              outputs_arrived, outputs_finished, stage, phasebits,
                                              row_blocks, col_blocks, num_iters
                                              , d_tt_pool, mma_done, tmem_free, outputs_free,
                                              ctarank, /*write_workspace=*/G.rt != nullptr
                                              );
        }
    }

    // Every CTA is a compute CTA here, so this is the only exit path:
    // the last one out resets the counter and closes the epoch barrier.
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
__cluster_dims__(config::CLUSTER_SIZE, 1, 1)
void gemm_rs_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

// Launch wrapper stays in this TU so the kernel body stays out of the .cuh.
void launch_fused_gemm_rs(const fused_globals& G, unsigned int active_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    unsigned int grid = (active_sms == 0u)
        ? (unsigned int)config::NUM_BLOCKS
        : active_sms;
    // Clusters are launched whole.
    grid -= grid % (unsigned int)config::CLUSTER_SIZE;
    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        gemm_rs_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    gemm_rs_fused_kernel_stub<<<grid, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}

}  // namespace gemm_rs_intranode_blackwell

#include "operators/gemm_rs/gemm_rs_blackwell_session.cuh"
