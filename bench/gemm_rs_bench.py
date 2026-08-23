"""gemm_rs GEMM + Reduce-Scatter bench (release version).

Default EFA sweep: M∈{2048,4096,8192,16384,32768}, K=M/16.
Bakes the on-by-default gemm_rs env vars into Python:
  GEMM_RS_FUSE_COMPUTE_INTRA=1, GEMM_RS_INTRA_RS_DIRECT_STAGING=1, GEMM_RS_SEND_READY_BITMAP=1.
Per-shape SM split tuned to match the chunk512 numbers (see SM_SPLIT below).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

# Default binding behavior; callers can override for compatibility checks.
os.environ.setdefault("MKERNEL_BIND_RETAINED_HANDLE", "1")

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import (  # noqa: E402
    check_close,
    compare_named_results,
    get_peer_ips,
    get_peer_ports,
    make_dist_buffer,
    rdma_backing,
    rdma_policy_label,
)

KERNEL_NAME = "gemm_rs"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
COL_BLOCK = 256

# Sweep shapes keep K=M/(NUM_NODES*8) on aligned values.
DEFAULT_SHAPES = (
    [3072, 6144, 12288, 24576, 49152]
    if NUM_NODES == 3 else
    [4096, 8192, 16384, 32768, 65536, 98304]
    if NUM_NODES == 4 else
    [2048, 4096, 8192, 16384, 32768]
)

# Tuned role split for the fused compute/intra/send/reduce path:
# (n_comp, n_intra, n_send, n_reduce, chunk_tiles).
SM_SPLIT = {
    2048:  (112, 0, 10, 10, 2),
    4096:  (114, 0, 10,  8,  2),
    8192:  (118, 0, 6, 8, 4),
    16384: (120, 0, 4,  8,  4),
    32768: (120, 0, 4,  8,  8),
    65536: (124, 0, 2,  6,  32),
    98304: (118, 0, 6,  8,  4),
    # Multi-node shapes use smaller chunks for M=24576 and conservative chunks
    # at the extremes.
    6144:  (116, 0, 8,  8,  4),
    12288: (118, 0, 6,  8,  4),
    24576: (120, 0, 4,  8,  2),
    49152: (120, 0, 4,  8,  4),
}


def split_for_shape(m: int) -> tuple[int, int, int, int, int]:
    split = SM_SPLIT.get(m, (118, 0, 6, 8, 4))
    if NUM_NODES == 4 and m == 4096 and "GEMM_RS_SPLIT" not in os.environ:
        split = (116, 0, 8, 8, 2)
    override = os.environ.get("GEMM_RS_SPLIT")
    if override:
        parts = [int(x) for x in override.replace(":", ",").split(",") if x.strip()]
        if len(parts) != 5:
            raise ValueError(
                "GEMM_RS_SPLIT must be comp,intra,send,reduce,chunk_tiles"
            )
        split = tuple(parts)

    n_comp, n_intra, n_send, n_reduce, chunk_tiles = split
    n_comp = int(os.environ.get("GEMM_RS_NUM_COMP_SMS", n_comp))
    n_intra = int(os.environ.get("GEMM_RS_NUM_INTRA_COMM_SMS", n_intra))
    n_send = int(os.environ.get("GEMM_RS_NUM_SEND_SMS", n_send))
    n_reduce = int(os.environ.get("GEMM_RS_NUM_REDUCE_SMS", n_reduce))
    chunk_tiles = int(os.environ.get("GEMM_RS_CHUNK_TILES", chunk_tiles))
    return n_comp, n_intra, n_send, n_reduce, chunk_tiles


def poll_tuning(m: int) -> tuple[int, int]:
    if NUM_NODES > 2:
        return (1, 100)
    # Relaxed polling is the default timing path; acquire polling remains
    # available through the kernel option when needed.
    return (0, 100)


def median_then_max_cuda(samples):
    median = sorted(float(x) for x in samples)[len(samples) // 2]
    if os.environ.get("MKERNEL_BENCH_DUMP_RANK_MS") == "1":
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, {
            "rank": dist.get_rank(),
            "ms": median,
            "host": os.uname().nodename,
        })
        if dist.get_rank() == 0:
            print(f"[gemm_rs-rank-ms] {gathered}", flush=True)
    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str,
                   default=",".join(str(s) for s in DEFAULT_SHAPES))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


# The Blackwell (tcgen05) build takes B as (N, K); the Hopper build takes (K, N).
TCGEN05 = os.environ.get("MKERNEL_TCGEN05", "1") == "1"


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
    node_groups = [
        dist.new_group([node * world_size + lr for lr in range(world_size)])
        for node in range(NUM_NODES)
    ]
    same_local_rank_groups = [
        dist.new_group([node * world_size + lr for node in range(NUM_NODES)])
        for lr in range(world_size)
    ]

    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        if NUM_NODES == 1:
            # Single-node (intra-only): no RDMA session is opened, so no peer.
            peer_ip = "127.0.0.1"
        else:
            peer_node = 1 if node_idx == 0 else 0
            peer_ip = os.environ.get(f"NODE{peer_node}_IP")
            if not peer_ip:
                raise RuntimeError(f"NODE{peer_node}_IP must be set, or set PEER_IP explicitly")
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank

    mod = load_module.load(KERNEL_NAME)

    if is_chief:
        print(f"[gemm_rs] world={world_size*NUM_NODES} per_node={world_size} "
              f"shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_gpu_idx = node_idx * world_size + local_rank

    result_sizes, result_fused = [], []
    correctness_ok = True

    for m in shapes:
        n = m
        k = m // (NUM_NODES * world_size)
        m_local = m // world_size  # each rank owns M/8 rows after intra-RS

        n_comp, n_intra, n_send, n_reduce, chunk_tiles = split_for_shape(m)
        # Per-shape override: MKERNEL_GEMM_RS_SPLIT_<M>="comp,intra,send,reduce,ct".
        # Layered on top of split_for_shape so per-shape env wins over the
        # global GEMM_RS_SPLIT / GEMM_RS_NUM_*_SMS knobs handled there.
        _ovr = os.environ.get(f"MKERNEL_GEMM_RS_SPLIT_{m}")
        if _ovr:
            _vals = [int(x) for x in _ovr.split(",")]
            assert len(_vals) == 5, f"MKERNEL_GEMM_RS_SPLIT_{m} must be 5 ints"
            n_comp, n_intra, n_send, n_reduce, chunk_tiles = _vals

        if is_chief:
            print(f"\n[gemm_rs] M={m} K={k} N={n} m_local={m_local} "
                  f"split=({n_comp},{n_intra},{n_send},{n_reduce}) ct={chunk_tiles}",
                  flush=True)

        torch.manual_seed(42 + global_gpu_idx)
        torch.cuda.manual_seed(42 + global_gpu_idx)
        A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / (k ** 0.25)
        # The Blackwell path issues the MMA as ABt, so it takes B N-major.
        # Keep one logical B and hand the kernel whichever layout it wants.
        B_ref = torch.randn((k, n), device="cuda", dtype=torch.bfloat16) / (k ** 0.25)
        B = B_ref.t().contiguous() if TCGEN05 else B_ref

        workspace = mod.DistBuffer(
            (m, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        workspace.data_.zero_()

        output = mod.DistBuffer(
            (m_local, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        output.data_.zero_()

        barrier = mod.DistBuffer(
            (2, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        barrier.data_.zero_()

        total_compute_tiles = (m // ROW_BLOCK) * (n // COL_BLOCK)
        ready = torch.zeros(total_compute_tiles, device="cuda", dtype=torch.int32)

        def ready_entries(m_):
            ct_ = SM_SPLIT.get(m_, (0,0,0,0,4))[4]
            return (m_ // ROW_BLOCK) * (m_ // COL_BLOCK // max(ct_, 1))
        ready_max = max(ready_entries(m_) for m_ in shapes)
        ready_cap = 4096
        while ready_cap < ready_max * 2:
            ready_cap *= 2
        ready_chunk = mod.DistBuffer(
            (ready_cap,), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        ready_chunk.data_.zero_()

        # The staging buffer is the RDMA source for GEMM_RS.
        staging_backing = rdma_backing()
        staging_dbuf = make_dist_buffer(
            mod,
            (m_local, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size,
            multicast=False, backing=staging_backing,
        )
        staging_dbuf.data_.zero_()
        staging_buf = staging_dbuf.data_
        if is_chief:
            print(rdma_policy_label(KERNEL_NAME, source=staging_backing), flush=True)

        os.environ["GEMM_RS_RDMA_CHUNK_TILES_RT"] = str(chunk_tiles)

        staging_bytes = m_local * n * 2  # bf16
        # Per-peer sizing (1× at N == 2, identical to legacy).
        n_peers = NUM_NODES - 1
        recv_bytes = n_peers * staging_bytes
        per_peer_inter_tiles = (m_local // ROW_BLOCK) * (n // COL_BLOCK)
        total_inter_tiles = n_peers * per_peer_inter_tiles
        fifo_cap = 2048
        while fifo_cap < total_inter_tiles * 2:
            fifo_cap *= 2

        dist.barrier()
        # Single-node (intra-only) runs never touch the network: with
        # n_send == n_reduce == 0 the kernel takes its intra_only_debug path,
        # leaves runtime_state as nullptr, and ignores every session-derived
        # pointer below. Opening an RDMA session would just fail on a host with
        # no usable HCA, so skip it entirely.
        intra_only = (NUM_NODES == 1)
        if intra_only:
            fifo = (0, 0, 0, 0, 0)
            arrival_ptr = 0
            recv_ptr = 0
        else:
            peer_ips = get_peer_ips(node_idx, NUM_NODES)
            mod.create_session(
                node_idx, peer_ip, tcp_port,
                staging_buf.data_ptr(), staging_bytes,
                recv_bytes, total_inter_tiles, fifo_cap, local_rank,
                peer_ips=peer_ips,
                peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
            )
            fifo = mod.get_fifo_handles()
            arrival_ptr = mod.get_arrival_flags_ptr()
            recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        if not intra_only:
            mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        use_acquire_poll, reduce_poll_sleep_ns = poll_tuning(m)

        use_prepare_epoch = hasattr(mod, "prepare_epoch")

        def reset_state():
            workspace.data_.zero_(); output.data_.zero_(); ready.zero_()
            barrier.data_.zero_(); ready_chunk.data_.zero_()
            staging_dbuf.data_.zero_()
            if not intra_only and hasattr(mod, "zero_recv_buf"):
                mod.zero_recv_buf()

        def advance_epoch(next_epoch: int):
            # Queue-mode arrivals carry packed work, not an epoch value. Keep
            # all nodes quiesced before any rank clears arrival slots.
            if intra_only:
                return
            if use_prepare_epoch:
                mod.prepare_epoch()
                dist.barrier()
                mod.commit_epoch(next_epoch)
            else:
                mod.set_epoch(next_epoch)

        def run_once():
            mod.gemm_rs_fused(
                A, B, workspace, output, barrier, ready,
                recv_ptr, staging_buf.data_ptr(),
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch, node_idx,
                n_comp, n_intra, n_send, n_reduce,
                use_acquire_poll, reduce_poll_sleep_ns,
                ready_chunk,
                staging_dbuf,
                num_nodes=NUM_NODES,
            )

        def start_iter():
            nonlocal epoch
            if use_prepare_epoch:
                epoch += 1
                advance_epoch(epoch)
                reset_state()
            else:
                # Clear local state before publishing the next epoch to the session.
                reset_state()
                epoch += 1
                advance_epoch(epoch)

        for wi in range(args.warmup):
            start_iter()
            dist.barrier(); time.sleep(0.1)
            run_once(); torch.cuda.synchronize()
            dist.barrier()

        # GEMM_RS_TRACE_OUT=<dir>: capture one traced iteration (needs a build
        # with GEMM_RS_TRACE=1) and dump the raw records per rank.
        trace_out = os.environ.get("GEMM_RS_TRACE_OUT")
        if trace_out and getattr(mod, "trace_enabled", lambda: False)():
            import numpy as np
            os.makedirs(trace_out, exist_ok=True)
            start_iter()
            dist.barrier()
            # Steady state, not a cold start: without these the traced launch
            # begins on an idle GPU at 120 MHz and measures the clock ramp.
            for _ in range(8):
                run_once()
            torch.cuda.synchronize()
            mod.trace_reset()
            run_once(); torch.cuda.synchronize()
            recs = mod.trace_read().numpy()
            np.save(os.path.join(trace_out, f"gemm_rs_trace_M{m}_rank{dist.get_rank()}.npy"), recs)
            if is_chief:
                print(f"[gemm_rs-trace] M={m} records/rank={recs.shape[0]} -> {trace_out}",
                      flush=True)
            dist.barrier()

        samples = []
        # Per-iter inter-node sync is the default: gemm_rs advances a fresh epoch
        # each iter, which deadlocks the proxy without a per-iter barrier+settle.
        # MKERNEL_BENCH_NO_SYNC=1 (or MKERNEL_BENCH_LEGACY_SYNC=0) forces the
        # NCCL-style back-to-back path.
        # MKERNEL_BENCH_TIMING selects the methodology, and means the same thing
        # in ag_gemm_bench.py. Default "batch" matches ThunderKittens' harness
        # (common.py:benchmark_no_l2_clear): warm up, then N back-to-back kernel
        # launches inside one event pair, divided by N -- no per-iteration host
        # work, no barrier, no sync. "periter" brackets each launch separately
        # and resets between, which leaves idle gaps that let the GPU boost and
        # is therefore NOT comparable to TK.
        #
        # In batch mode the timed loop deliberately produces wrong values: this
        # kernel's epilogue store_adds into staging, so without the per-iteration
        # zero the results accumulate. Correctness is verified separately below,
        # on one clean iteration. Keeping the two coupled is what made these
        # numbers incomparable in the first place.
        timing_mode = os.environ.get("MKERNEL_BENCH_TIMING", "batch")
        if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "1": timing_mode = "batch"
        if os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1": timing_mode = "periter"
        legacy_sync = (timing_mode != "batch")
        if not legacy_sync:
            n_iters = max(args.iters, 32)
            start_iter()
            dist.barrier(); time.sleep(0.05)
            for _ in range(3):          # warm up, as TK does
                run_once()
            torch.cuda.synchronize(); dist.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_iters):
                run_once()
            e.record()
            torch.cuda.synchronize()
            avg_ms = s.elapsed_time(e) / n_iters
            samples = [avg_ms] * args.iters
            if is_chief:
                print(f"[gemm_rs-batch] M={m} N={n_iters} avg={avg_ms:.4f} ms", flush=True)
            dist.barrier()
            # One clean iteration so the correctness check below sees valid state.
            start_iter(); dist.barrier()
            run_once(); torch.cuda.synchronize(); dist.barrier()
        else:
            for _ in range(args.iters):
                start_iter()
                dist.barrier(); time.sleep(0.05)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); run_once(); e.record(); torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
                dist.barrier()

        wall_ms = median_then_max_cuda(samples)
        if is_chief:
            print(f"[gemm_rs] M={m} wall={wall_ms:.3f} ms", flush=True)
            if samples:
                _ss = sorted(float(x) for x in samples)
                _n = len(_ss)
                _med = _ss[_n // 2]
                _p99 = _ss[min(_n - 1, int(0.99 * _n))]
                print(f"[gemm_rs-dist] M={m} n={_n} min={_ss[0]:.3f} "
                      f"med={_med:.3f} p99={_p99:.3f} max={_ss[-1]:.3f}", flush=True)
        if intra_only:
            # Intra-only runs stop after the NVLink reduce-scatter: the result
            # is the cross-GPU store_add accumulation in `staging`, and the
            # reduce CTAs that would normally publish `output` never launch.
            # Undo the chunk-major staging view so the checks below can run
            # unchanged. Host entrypoint lays tile (rb, cb) out at
            # (rb*CB + cb)*128 rows of 256 columns.
            RB = m_local // ROW_BLOCK
            CB = n // COL_BLOCK
            output.data_.copy_(
                staging_buf.reshape(RB, CB, ROW_BLOCK, COL_BLOCK)
                           .permute(0, 2, 1, 3)
                           .reshape(m_local, n)
            )

        C_ref = None
        for target_lr in range(world_size):
            row_lo = target_lr * m_local
            row_hi = row_lo + m_local
            # Mirror the kernel topology: first reduce this row slice across
            # all 8 local GPUs in the node, then reduce the owning local-rank
            # slice across nodes.
            # Keep on GPU so the NCCL backend can all_reduce/all_gather it.
            ref_slice = torch.matmul(A[row_lo:row_hi], B_ref)
            dist.all_reduce(
                ref_slice, op=dist.ReduceOp.SUM, group=node_groups[node_idx]
            )
            if local_rank == target_lr:
                node_refs = [torch.empty_like(ref_slice) for _ in range(NUM_NODES)]
                dist.all_gather(
                    node_refs, ref_slice,
                    group=same_local_rank_groups[target_lr],
                )
                C_ref = node_refs[0]
                for node_ref in node_refs[1:]:
                    C_ref = C_ref + node_ref
                if os.environ.get("GEMM_RS_RECEIVER_OWNER_RS", "0") == "1":
                    # Receiver-owner RS: each tile chunk is owned by exactly
                    # one node, so this rank's output buffer contains only
                    # chunks where chunk_id % NUM_NODES == node_idx.
                    sparse_ref = torch.zeros_like(C_ref)
                    chunks_per_row = (n // COL_BLOCK + chunk_tiles - 1) // chunk_tiles
                    for rb in range(m_local // ROW_BLOCK):
                        row_lo2 = rb * ROW_BLOCK
                        row_hi2 = row_lo2 + ROW_BLOCK
                        for ci in range(chunks_per_row):
                            chunk_id = rb * chunks_per_row + ci
                            if chunk_id % NUM_NODES != node_idx:
                                continue
                            col_lo = ci * chunk_tiles * COL_BLOCK
                            col_hi = min(n, col_lo + chunk_tiles * COL_BLOCK)
                            sparse_ref[row_lo2:row_hi2, col_lo:col_hi] = \
                                C_ref[row_lo2:row_hi2, col_lo:col_hi]
                    C_ref = sparse_ref
        # Only the destination local-rank owns this reduce-scatter shard.
        # Non-owners still participate in the distributed check_close()
        # collective, but compare against themselves so only owner ranks
        # determine the global correctness result.
        if C_ref is None:
            C_ref = output.data_
        correctness_ok = check_close(
            f"gemm_rs M={m}", output.data_, C_ref, atol=0.75, rtol=0.15
        ) and correctness_ok
        result_sizes.append(f"M={m}")
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "gemm_rs",
                           result_sizes, result_fused,
                           note=f"release gemm_rs bench timing={'legacy_sync' if legacy_sync else 'steady_state'} world={world_size*NUM_NODES}")
        print(f"[gemm_rs] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("gemm_rs", result_sizes, result_fused, args.compare_to)
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
    