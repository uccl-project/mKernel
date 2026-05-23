"""gemm_ar GEMM + AllReduce bench (release version).

Default EFA sweep for the GEMM + AllReduce chart.
Targets at M ∈ {2048, 4096, 8192, 16384, 32768}: [0.418, 0.350, 0.868, 2.755, 10.896] ms.
Uses gemm_ar production defaults compiled into the .so via -DGEMM_AR_*.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"
os.environ.setdefault("GEMM_AR_ARRIVAL_QUEUE", "1")
os.environ.setdefault("GEMM_AR_DISABLE_SEND_COALESCE", "1")
if os.environ.get("NUM_NODES") != "4":
    os.environ.setdefault("GEMM_AR_INTER_SEND_SMS", "4")
    os.environ.setdefault("GEMM_AR_NUM_INTRA_COMM_SMS", "12")
os.environ.setdefault("GEMM_AR_STEADY_STATE_BENCH", "1")
os.environ.setdefault("MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET", "1")
# Fast prepare_epoch path: skip host synchronization while keeping the FIFO
# cursor sync needed by the session invariant.
os.environ.setdefault("MKERNEL_PREP_EPOCH_FAST", "2")

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close, compare_named_results, get_peer_ips, get_peer_ports  # noqa: E402

KERNEL_NAME = "gemm_ar"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
COL_BLOCK = 256

DEFAULT_SHAPES = (
    # H200 release sweep: use the largest verified natural multiples.
    [8192, 12288, 16384, 20480, 22528]
    if NUM_NODES == 4 else
    [2048, 4096, 8192, 16384, 32768]
)

# Per-shape num_intra_comm_sms override. Small M has little intra-AR work to
# parallelize, so freeing those CTAs for compute is faster.
INTRA_OVERRIDE_AR = {2048: 4, 32768: 24}


def median_then_max_cuda(samples):
    sorted_samples = sorted(float(x) for x in samples)
    median = sorted_samples[len(sorted_samples) // 2]
    if os.environ.get("MKERNEL_BENCH_DUMP_RANK_MS") == "1":
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, {
            "rank": dist.get_rank(),
            "ms": median,
            "host": os.uname().nodename,
        })
        if dist.get_rank() == 0:
            print(f"[gemm_ar-rank-ms] {gathered}", flush=True)
    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def kernel_chunk_tiles_for_n(N):
    """Mirror gemm_ar_chunk_tiles(): kernel uses min(4, col_blocks).
    Note: must match the kernel's compile-time gemm_ar_chunk_tiles() exactly so
    scratch_ints sizing aligns."""
    col_blocks = N // COL_BLOCK
    chunk_tiles = int(os.environ.get("GEMM_AR_CHUNK_TILES", "4"))
    if chunk_tiles <= 0:
        chunk_tiles = 4
    return min(chunk_tiles, col_blocks)


def cta_split_chunk_tiles_for_n(N):
    """Host-side CTA allocation heuristic.

    The CUDA kernel's RDMA chunk size is still kernel_chunk_tiles_for_n().
    This host-only heuristic intentionally coarsens small shapes when deciding
    how many communication CTAs are useful.
    """
    col_blocks = N // COL_BLOCK
    if col_blocks <= 8:
        return col_blocks
    if col_blocks <= 16:
        return col_blocks
    if col_blocks <= 32:
        return col_blocks // 2
    return min(4, col_blocks)


def compute_scratch_ints(M, N, world_size, num_remote_queues=1):
    """Mirror gemm_ar_compute_scratch_layout in gemm_ar_multinode.cu."""
    row_blocks_per_slice = M // world_size // ROW_BLOCK
    col_blocks = N // COL_BLOCK
    chunk_tiles = kernel_chunk_tiles_for_n(N)
    chunks_per_row = max(1, (col_blocks + chunk_tiles - 1) // chunk_tiles)
    total_chunks = row_blocks_per_slice * chunks_per_row
    slice_tiles = row_blocks_per_slice * col_blocks
    cap = max(16, total_chunks)
    v = cap - 1
    for shift in (1, 2, 4, 8, 16): v |= v >> shift
    gemm_rq_capacity = v + 1
    chunk_remote_arrived_offset = 0
    arrival_queue_head_offset = chunk_remote_arrived_offset + total_chunks
    local_ar_done_offset = arrival_queue_head_offset + num_remote_queues
    send_issued_offset = local_ar_done_offset + 1
    remote_arrived_offset = send_issued_offset + 1
    published_offset = remote_arrived_offset + 1
    final_publish_done_offset = published_offset + 1
    debug_flags_offset = final_publish_done_offset + 1
    debug_dump_counter_offset = debug_flags_offset + 1
    queue_expected_offset = debug_dump_counter_offset + 1
    queue_observed_offset = queue_expected_offset + num_remote_queues
    row_send_count_offset = queue_observed_offset + num_remote_queues
    gemm_tiles_ready_count_offset = row_send_count_offset + row_blocks_per_slice
    gemm_tile_scanned_offset = gemm_tiles_ready_count_offset + total_chunks
    gemm_rq_entries_offset = gemm_tile_scanned_offset + slice_tiles
    gemm_rq_head_offset = gemm_rq_entries_offset + gemm_rq_capacity
    gemm_rq_tail_offset = gemm_rq_head_offset + 1
    gemm_scan_cursor_offset = gemm_rq_tail_offset + 1
    local_ar_rq_entries_offset = gemm_scan_cursor_offset + 1
    local_ar_rq_head_offset = local_ar_rq_entries_offset + gemm_rq_capacity
    local_ar_rq_tail_offset = local_ar_rq_head_offset + 1
    remote_rq_entries_offset = local_ar_rq_tail_offset + 1
    remote_rq_head_offset = remote_rq_entries_offset + gemm_rq_capacity
    remote_rq_tail_offset = remote_rq_head_offset + 1
    owner_cap = max(8, chunks_per_row * 2)
    v = owner_cap - 1
    for shift in (1, 2, 4, 8, 16): v |= v >> shift
    owner_pending_capacity = v + 1
    owner_pending_entries_offset = remote_rq_tail_offset + 1
    owner_pending_head_offset = owner_pending_entries_offset + num_remote_queues * owner_pending_capacity
    owner_pending_tail_offset = owner_pending_head_offset + num_remote_queues
    owner_pending_max_depth_offset = owner_pending_tail_offset + num_remote_queues
    owner_pending_push_count_offset = owner_pending_max_depth_offset + num_remote_queues
    owner_pending_pop_count_offset = owner_pending_push_count_offset + num_remote_queues
    local_done_flag_offset = owner_pending_pop_count_offset + num_remote_queues
    remote_arrived_flag_offset = local_done_flag_offset + total_chunks
    remote_arrived_peer_mask_offset = remote_arrived_flag_offset + total_chunks
    intra_started_flag_offset = remote_arrived_peer_mask_offset + total_chunks
    reduce_cursor_offset = intra_started_flag_offset + 16
    chunk_claimed_flag_offset = reduce_cursor_offset + 1
    row_blocks = M // ROW_BLOCK
    total_all_chunks = row_blocks * chunks_per_row
    comp_chunk_tiles_done_offset = chunk_claimed_flag_offset + total_chunks
    intra_chunk_tiles_done_offset = comp_chunk_tiles_done_offset + total_all_chunks
    xnode_ready_offset = intra_chunk_tiles_done_offset + total_chunks
    return xnode_ready_offset + 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str, default=",".join(str(s) for s in DEFAULT_SHAPES))
    # Warmup default 10: under canonical no-sync timing, the per-iter
    # `set_epoch` cudaDeviceSynchronize re-cold-starts the EFA proxy and fewer
    # warmup iters did not fully clear the cold-proxy state at small M
    # (especially M=2048, where it produced occasional bimodal slow iters that
    # inflated the median). 10 warmup iters keeps the timed window warm reliably.
    p.add_argument("--warmup", type=int, default=int(os.environ.get("MKERNEL_GEMM_AR_WARMUP", "10")))
    p.add_argument("--iters", type=int, default=int(os.environ.get("MKERNEL_GEMM_AR_ITERS", "10")))
    p.add_argument("--num-comm-sms", type=int, default=int(os.environ.get("GEMM_AR_NUM_COMM_SMS", "64")))
    p.add_argument("--num-intra-comm-sms", type=int, default=None)
    p.add_argument("--num-inter-send-sms", type=int, default=None)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def pick_sm_split(M, N, world_size, num_comm_sms_total, num_intra_override, num_inter_send_override):
    """Mirror benchmark_gemm_ar_multinode.py CTA-allocation."""
    row_blocks_per_slice = M // world_size // ROW_BLOCK
    col_blocks = N // COL_BLOCK
    chunk_tiles = cta_split_chunk_tiles_for_n(N)
    chunks_per_row = max(1, (col_blocks + chunk_tiles - 1) // chunk_tiles)
    total_chunks = row_blocks_per_slice * chunks_per_row
    tiles_per_device = row_blocks_per_slice * col_blocks
    if tiles_per_device <= 16: min_intra = tiles_per_device
    elif tiles_per_device <= 64: min_intra = 16
    elif 33 <= total_chunks <= 128: min_intra = 16
    else: min_intra = min(total_chunks, num_comm_sms_total // 2)
    min_send = max(2, min(row_blocks_per_slice, 8))
    if total_chunks > 256: min_reduce = max(2, min(total_chunks, 8))
    elif total_chunks >= 33: min_reduce = 8
    else: min_reduce = max(2, min(total_chunks, 8))
    adaptive_comm = min_intra + min_send + min_reduce
    effective_comm_sms = min(num_comm_sms_total, max(adaptive_comm, 16))
    env_intra = os.environ.get("GEMM_AR_NUM_INTRA_COMM_SMS")
    env_inter_send = os.environ.get("GEMM_AR_INTER_SEND_SMS")
    four_node_default_tune = (
        NUM_NODES == 4
        and num_intra_override is None
        and env_intra is None
        and num_inter_send_override is None
        and env_inter_send is None
    )
    if num_intra_override is not None:
        num_intra_comm_sms = max(4, num_intra_override)
    elif env_intra:
        num_intra_comm_sms = max(4, int(env_intra))
    elif four_node_default_tune:
        # Four-node direct-fanout AR needs inter-heavy progress; balanced splits
        # helped debug smoke tests but lost in release warmup/10-iter timing.
        num_intra_comm_sms = 4
    else:
        num_intra_comm_sms = min_intra
    num_inter_comm_sms = max(4, effective_comm_sms - num_intra_comm_sms)
    # fused_inter_send_sm statically owns row blocks, so more send CTAs than
    # local row blocks cannot issue useful work even when sends are per chunk.
    max_useful_send_sms = max(1, row_blocks_per_slice)
    if num_inter_send_override is not None:
        inter_send_override = num_inter_send_override
    elif env_inter_send:
        inter_send_override = int(env_inter_send)
    elif four_node_default_tune:
        inter_send_override = 20
    else:
        inter_send_override = None
    num_inter_send_sms = (max(2, min(num_inter_comm_sms - 2, inter_send_override, max_useful_send_sms))
                          if inter_send_override is not None else max(2, min(min_send, num_inter_comm_sms - 2, max_useful_send_sms)))
    return num_intra_comm_sms, num_inter_comm_sms, num_inter_send_sms


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist_backend = os.environ.get("MKERNEL_DIST_BACKEND", "nccl")
    if dist_backend == "nccl":
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group(dist_backend)

    node_idx = args.node_idx if args.node_idx is not None else int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)
    peer_ips = get_peer_ips(node_idx, NUM_NODES)
    peer_ip = os.environ.get("PEER_IP", peer_ips[0])
    tcp_port = int(os.environ.get("TCP_PORT", "19730")) + local_rank

    mod = load_module.load(KERNEL_NAME)
    if is_chief:
        print(f"[gemm_ar] world={world_size*NUM_NODES} per_node={world_size} shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    gid = node_idx * world_size + local_rank
    K_denom = NUM_NODES * world_size  # K = base_n / 16
    use_ngt2_fallback = os.environ.get("MKERNEL_GEMM_AR_USE_TORCH_FALLBACK") == "1"

    result_sizes, result_fused = [], []
    timing_label = "torch_fallback" if use_ngt2_fallback else "legacy_sync"
    correctness_ok = True

    for base_n in shapes:
        M, K, N = base_n, base_n // K_denom, base_n
        row_blocks = M // ROW_BLOCK
        col_blocks = N // COL_BLOCK
        total_tiles = row_blocks * col_blocks
        slice_tiles = (M // world_size // ROW_BLOCK) * col_blocks
        TILE_BYTES = ROW_BLOCK * COL_BLOCK * 2
        staging_bytes = slice_tiles * TILE_BYTES

        if is_chief:
            print(f"\n[gemm_ar] M={M} K={K} N={N} tiles={total_tiles}", flush=True)

        torch.manual_seed(42 + gid); torch.cuda.manual_seed(42 + gid)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        if use_ngt2_fallback:
            if is_chief:
                print("[gemm_ar] using explicit torch fallback; unset "
                      "MKERNEL_GEMM_AR_USE_TORCH_FALLBACK to test fused path",
                      flush=True)
            samples = []
            for _ in range(args.warmup):
                C_tmp = torch.matmul(A, B)
                torch.cuda.synchronize()
                del C_tmp
            dist.barrier()
            for _ in range(args.iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                C_tmp = torch.matmul(A, B)
                e.record()
                torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
                del C_tmp
                dist.barrier()
            wall_ms = median_then_max_cuda(samples)
            if is_chief:
                sample_str = "[" + ", ".join(f"{x:.4f}" for x in samples) + "]"
                print(f"[gemm_ar] M={M} fallback samples={sample_str} "
                      f"median={sorted(samples)[len(samples)//2]:.4f}", flush=True)
                print(f"[gemm_ar] M={M} wall={wall_ms:.3f} ms", flush=True)
            result_sizes.append(f"M={M}")
            result_fused.append(wall_ms)
            continue

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_dbuf.data_.zero_()

        barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        barrier.data_.zero_()

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        staging_buf = torch.empty(staging_bytes // 2, device="cuda", dtype=torch.bfloat16)
        ring_experiment = (
            os.environ.get("GEMM_AR_RING_EXPERIMENT", "0") == "1"
            or os.environ.get("GEMM_AR_RING_RS_EXPERIMENT", "0") == "1"
        )
        early_remote_accum = os.environ.get("GEMM_AR_EARLY_REMOTE_ACCUM", "0") == "1"
        remote_accum = (
            torch.empty(staging_bytes // 2, device="cuda", dtype=torch.bfloat16)
            if (early_remote_accum or ring_experiment) else None
        )

        # Per-peer sizing (1× at N == 2, identical to legacy).
        n_peers = NUM_NODES - 1
        recv_buf_bytes = n_peers * staging_bytes
        recv_buf_tiles = n_peers * total_tiles

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_tiles * 2: fifo_cap *= 2
        clocal_ptr = int(C_dbuf.data_.data_ptr())
        clocal_bytes = int(C_dbuf.data_.numel() * C_dbuf.data_.element_size())
        direct_src_ptr = int(remote_accum.data_ptr()) if ring_experiment and remote_accum is not None else clocal_ptr
        direct_src_bytes = staging_bytes if ring_experiment and remote_accum is not None else clocal_bytes
        row_stride_bytes = N * 2
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            staging_buf.data_ptr(), staging_bytes,
            recv_buf_bytes, recv_buf_tiles, fifo_cap, local_rank,
            direct_src_ptr, direct_src_bytes, row_stride_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()
        num_qps = max(1, int(mod.get_num_qps()))
        arrival_tails_ptr = mod.get_arrival_tails_ptr() if hasattr(mod, "get_arrival_tails_ptr") else 0
        barrier_device_ptr = mod.get_barrier_device_ptr() if hasattr(mod, "get_barrier_device_ptr") else 0
        if NUM_NODES > 2:
            barrier_device_ptr = 0

        logical_queues_per_qp = max(1, int(os.environ.get("GEMM_AR_LOGICAL_QUEUES_PER_QP", "1")))
        # Apply per-shape intra override (small-M shapes benefit from fewer
        # intra-comm CTAs — see INTRA_OVERRIDE_AR comment).
        intra_override_for_shape = (
            args.num_intra_comm_sms if args.num_intra_comm_sms is not None
            else INTRA_OVERRIDE_AR.get(M)
        )
        n_intra, n_inter, n_inter_send = pick_sm_split(
            M, N, world_size, args.num_comm_sms,
            intra_override_for_shape, args.num_inter_send_sms)
        num_allocated_remote_queues = max(1, num_qps * logical_queues_per_qp)
        col_blocks_for_queues = N // COL_BLOCK
        chunks_per_row_for_queues = max(
            1,
            (col_blocks_for_queues + cta_split_chunk_tiles_for_n(N) - 1)
            // cta_split_chunk_tiles_for_n(N),
        )
        total_chunks_for_queues = (M // world_size // ROW_BLOCK) * chunks_per_row_for_queues
        if NUM_NODES > 2:
            num_remote_queues = num_allocated_remote_queues
        else:
            num_remote_queues = max(1, min(num_allocated_remote_queues, total_chunks_for_queues))
        if os.environ.get("GEMM_AR_ARRIVAL_QUEUE", "1") == "1" and NUM_NODES <= 2:
            kernel_inter_send_sms = max(
                1,
                min(
                    n_inter - 1,
                    int(os.environ.get("GEMM_AR_INTER_SEND_SMS", "4")),
                    max(1, M // world_size // ROW_BLOCK),
                ),
            )
            kernel_inter_reduce_publish_sms = n_inter - kernel_inter_send_sms
            num_remote_queues = max(1, min(num_remote_queues, kernel_inter_reduce_publish_sms))

        scratch_ints = compute_scratch_ints(M, N, world_size, num_remote_queues)
        ar_done = torch.zeros(scratch_ints, device="cuda", dtype=torch.int32)
        # gemm_ar host entrypoint expects (num_intra_comm_sms, num_inter_comm_sms).
        if is_chief:
            print(
                f"[gemm_ar] M={M} sm_split intra={n_intra} inter={n_inter} "
                f"send={n_inter_send} remote_queues={num_remote_queues}/{num_allocated_remote_queues}",
                flush=True,
            )

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        kernel_use_acquire_poll = (M == 4096)

        # Canonical no-sync (steady-state) is the default.
        # MKERNEL_BENCH_LEGACY_SYNC=1 (or MKERNEL_BENCH_NO_SYNC=0) opts back
        # into per-iter sync. GEMM_AR_STEADY_STATE_BENCH is preserved for
        # back-compat (default 1; set to 0 to also force legacy).
        steady_state = os.environ.get("GEMM_AR_STEADY_STATE_BENCH", "1") == "1"
        if os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1":
            steady_state = False
        if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
            steady_state = False
        if NUM_NODES > 2 and os.environ.get("MKERNEL_ALLOW_NOSYNC_NGT2") != "1":
            if steady_state and is_chief:
                print("[gemm_ar] forcing legacy-sync timing for NUM_NODES > 2", flush=True)
            steady_state = False

        # Warmup. Under steady_state, only first iter does barrier/arrival reset.
        for wi in range(args.warmup):
            C_dbuf.data_.zero_(); C_final.data_.zero_(); ar_done.zero_()
            if remote_accum is not None:
                remote_accum.zero_()
            if not steady_state or wi == 0:
                barrier.data_.zero_()
                mod.reset_arrival_flags()
            epoch += 1
            mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            mod.gemm_ar_multinode(
                A, B, C_dbuf, barrier, C_final,
                staging_buf.data_ptr(), recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch, node_idx,
                n_intra, n_inter,
                ar_done.data_ptr(),
                arrival_tails_ptr=arrival_tails_ptr,
                scratch_ints=scratch_ints,
                num_qps=num_qps,
                num_remote_queues=num_remote_queues,
                num_allocated_remote_queues=num_allocated_remote_queues,
                cross_node_barrier_ptr=barrier_device_ptr,
                use_acquire_poll=kernel_use_acquire_poll,
                num_nodes=NUM_NODES,
                **({"remote_accum_ptr": remote_accum.data_ptr()} if remote_accum is not None else {}),
            )
            torch.cuda.synchronize()
        # One-shot cross-node align before timed loop (steady_state requires).
        dist.barrier()

        samples_pairs = []
        trace_out_base = os.environ.get("GEMM_AR_ACTIVITY_TRACE_OUT_BASE", "")
        for iter_idx in range(args.iters):
            if trace_out_base:
                os.environ["GEMM_AR_ACTIVITY_TRACE_OUT"] = (
                    f"{trace_out_base}.iter{iter_idx + 1}.trace.json"
                )
            C_dbuf.data_.zero_(); C_final.data_.zero_(); ar_done.zero_()
            if remote_accum is not None:
                remote_accum.zero_()
            if not steady_state:
                barrier.data_.zero_()
                mod.reset_arrival_flags()
            epoch += 1
            mod.set_epoch(epoch)
            if not steady_state:
                dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.gemm_ar_multinode(
                A, B, C_dbuf, barrier, C_final,
                staging_buf.data_ptr(), recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch, node_idx,
                n_intra, n_inter,
                ar_done.data_ptr(),
                arrival_tails_ptr=arrival_tails_ptr,
                scratch_ints=scratch_ints,
                num_qps=num_qps,
                num_remote_queues=num_remote_queues,
                num_allocated_remote_queues=num_allocated_remote_queues,
                cross_node_barrier_ptr=barrier_device_ptr,
                use_acquire_poll=kernel_use_acquire_poll,
                num_nodes=NUM_NODES,
                **({"remote_accum_ptr": remote_accum.data_ptr()} if remote_accum is not None else {}),
            )
            e.record()
            if not steady_state:
                torch.cuda.synchronize()
                dist.barrier()
            samples_pairs.append((s, e))

        # In steady_state, defer elapsed_time until all iters issued.
        if steady_state:
            torch.cuda.synchronize()
            dist.barrier()
        samples = [s.elapsed_time(e) for (s, e) in samples_pairs]
        timing_label = "steady_state" if steady_state else "legacy_sync"

        wall_ms = median_then_max_cuda(samples)
        if is_chief:
            sample_str = "[" + ", ".join(f"{x:.4f}" for x in samples) + "]"
            print(f"[gemm_ar] M={M} samples={sample_str} min={min(samples):.4f} median={sorted(samples)[len(samples)//2]:.4f}", flush=True)
            print(f"[gemm_ar] M={M} wall={wall_ms:.3f} ms", flush=True)
        if args.mode == "check":
            C_ref_cpu = torch.matmul(A, B).detach().float().cpu()
            local_ref_cpu = C_ref_cpu.clone()
            dist.all_reduce(C_ref_cpu, op=dist.ReduceOp.SUM)
            correctness_ok = check_close(
                f"gemm_ar M={M}", C_final.data_, C_ref_cpu, atol=0.55, rtol=0.12
            ) and correctness_ok
        result_sizes.append(f"M={M}")
        result_fused.append(wall_ms)

        # gemm_ar expects per-shape session teardown (mirrors experiment bench).
        try:
            mod.destroy_session()
        except Exception as ex:
            if is_chief:
                print(f"[gemm_ar] destroy_session: {ex}", flush=True)
        dist.barrier()

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "gemm_ar",
                           result_sizes, result_fused,
                           note=f"release gemm_ar bench timing={timing_label} world={world_size*NUM_NODES}")
        print(f"[gemm_ar] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("gemm_ar", result_sizes, result_fused, args.compare_to)
        ok = ok and correctness_ok
        dist.destroy_process_group()
        if not ok: return 1
        return 0
    if not correctness_ok:
        dist.destroy_process_group()
        return 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
