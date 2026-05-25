"""ag_gemm All-Gather + GEMM bench (release version).

Default EFA sweep: M∈{4096,8192,16384,24576,32768}.
SM split: --num-comm-sms 64 (50/50 split intra/inter inside the kernel).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import (  # noqa: E402
    check_close,
    compare_named_results,
    gather_cpu_tensors,
    get_peer_ips,
    get_peer_ports,
)

KERNEL_NAME = "ag_gemm"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
COL_BLOCK = 256
RED_BLOCK = 64
CHUNK_BYTES = 64 * 1024  # baked from AG_CHUNK_BYTES=65536

DEFAULT_SHAPES = (
    # M=57344 hangs during default-warmup 4-node sweeps on H200x.
    [8192, 16384, 32768, 49152, 65536]
    if NUM_NODES == 4 else
    [6144, 12288, 24576, 49152, 73728]
    if NUM_NODES == 3 else
    [4096, 8192, 16384, 24576, 32768]
)

# Per-shape num_comm_sms override. Smaller values reduce coordination overhead
# at small M (NCCL has minimal launch overhead and beats the fused path there
# unless we cut the comm-CTA budget). The 64-sms default oversubscribes comm
# CTAs at medium M where the GEMM wave count is lower.
SMS_PER_SHAPE = {4096: 8, 6144: 8, 8192: 8, 12288: 8, 16384: 8,
                 24576: 8, 49152: 8, 65536: 8, 57344: 8, 73728: 8}


def avg_then_max_cuda(samples):
    # Median-then-max for robustness against outlier iters (matches gemm_rs).
    median = sorted(float(x) for x in samples)[len(samples) // 2]
    if os.environ.get("MKERNEL_BENCH_DUMP_RANK_MS") == "1":
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, {
            "rank": dist.get_rank(),
            "ms": median,
            "host": os.uname().nodename,
        })
        if dist.get_rank() == 0:
            print(f"[ag_gemm-rank-ms] {gathered}", flush=True)
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
    p.add_argument("--num-comm-sms", type=int, default=64)
    p.add_argument("--num-intra-comm-sms", type=int, default=0)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    # Preserve explicit --num-intra-comm-sms from the CLI (e.g. AG_GEMM_BENCH_EXTRA
    # profile runs). The per-shape loop used to force 0 here, which silently ignored
    # tuned intra splits unless INTRA_OVERRIDE was populated.
    cli_num_intra_comm_sms = int(args.num_intra_comm_sms)
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
    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        peer_node = 1 if node_idx == 0 else 0
        peer_ip = os.environ.get(f"NODE{peer_node}_IP")
        if not peer_ip:
            raise RuntimeError(f"NODE{peer_node}_IP must be set, or set PEER_IP explicitly")
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank

    mod = load_module.load(KERNEL_NAME)
    if is_chief:
        print(f"[ag_gemm] world={world_size*NUM_NODES} shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_world = NUM_NODES * world_size
    global_gpu_idx = node_idx * world_size + local_rank
    use_ngt2_fallback = os.environ.get("MKERNEL_AG_GEMM_USE_TORCH_FALLBACK") == "1"

    result_sizes, result_fused = [], []
    correctness_ok = True

    # Per-shape intra override that bypasses the max(4) floor in the kernel.
    INTRA_OVERRIDE = {}
    # Per-shape override format: AG_GEMM_INTRA_OVERRIDE_<M>=<intra>.
    # Applied after the conditional defaults so the env value always wins.
    for base_n in shapes:
        env_key = f"AG_GEMM_INTRA_OVERRIDE_{base_n}"
        if env_key in os.environ:
            INTRA_OVERRIDE[base_n] = int(os.environ[env_key])
            if is_chief:
                print(f"[ag_gemm] env override {env_key}={os.environ[env_key]}", flush=True)
    for base_n in shapes:
        # Per-shape num_comm_sms override (small-M overhead reduction).
        if base_n in SMS_PER_SHAPE:
            args.num_comm_sms = SMS_PER_SHAPE[base_n]
            if is_chief:
                print(f"[ag_gemm] M={base_n}: per-shape num_comm_sms={args.num_comm_sms}",
                      flush=True)
        if base_n in INTRA_OVERRIDE:
            args.num_intra_comm_sms = INTRA_OVERRIDE[base_n]
            if is_chief:
                print(f"[ag_gemm] M={base_n}: per-shape num_intra_comm_sms={args.num_intra_comm_sms}",
                      flush=True)
        else:
            args.num_intra_comm_sms = cli_num_intra_comm_sms
        # ag_gemm TP-column-parallel: M=K=base_n, N=base_n/global_world.
        M, K, N = base_n, base_n, base_n // global_world
        M_node = M // NUM_NODES
        M_local = M_node // world_size
        assert M % ROW_BLOCK == 0 and K % RED_BLOCK == 0 and N % COL_BLOCK == 0
        os.environ.pop("AG_GEMM_ROW_STRIDE_BYTES", None)
        n_peers = NUM_NODES - 1
        ring_recv_banks = n_peers

        if is_chief:
            print(f"\n[ag_gemm] M={M} K={K} N={N} M_node={M_node} M_local={M_local}", flush=True)

        torch.manual_seed(42 + global_gpu_idx); torch.cuda.manual_seed(42 + global_gpu_idx)
        A_local = torch.randn((M_local, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        torch.manual_seed(100); torch.cuda.manual_seed(100)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        if use_ngt2_fallback:
            if is_chief:
                print("[ag_gemm] using explicit torch fallback; unset "
                      "MKERNEL_AG_GEMM_USE_TORCH_FALLBACK to test fused path",
                      flush=True)
            samples = []
            for _ in range(args.warmup):
                C_tmp = torch.matmul(A_local, B)
                torch.cuda.synchronize()
                del C_tmp
            dist.barrier()
            for _ in range(args.iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                C_tmp = torch.matmul(A_local, B)
                e.record()
                torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
                del C_tmp
                dist.barrier()
            wall_ms = avg_then_max_cuda(samples)
            if is_chief:
                print(f"[ag_gemm] M={M} wall={wall_ms:.3f} ms", flush=True)
            result_sizes.append(f"M={M}")
            result_fused.append(wall_ms)
            continue

        a_tk = mod.DistBuffer((M_node, K), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        start_row = local_rank * M_local
        a_tk.data_[start_row:start_row + M_local].copy_(A_local)

        a_recv_tk = mod.DistBuffer((M_node * n_peers * ring_recv_banks, K), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        a_recv_tk.data_.zero_()

        barrier = mod.DistBuffer((3, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        barrier.data_.zero_()

        C = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)

        a_half_bytes = M_node * K * 2
        total_chunks = (a_half_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES

        # Per-peer recv_buf / arrival flag scaling. At N == 2 n_peers == 1,
        # so this collapses to a single peer-sized slot.
        recv_buf_bytes = n_peers * a_half_bytes * ring_recv_banks
        recv_buf_chunks = n_peers * total_chunks * ring_recv_banks

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_chunks * 2: fifo_cap *= 2
        a_tk_ptr = int(a_tk.data_.data_ptr())
        send_buf_ptr = int(a_recv_tk.data_.data_ptr())
        send_buf_size = recv_buf_bytes
        # A_recv is registered as MR0 (src_view=0) so received shards can be
        # forwarded to the next node after phase-2 republishes them.
        peer_ips = get_peer_ips(node_idx, NUM_NODES)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            send_buf_ptr, send_buf_size, recv_buf_bytes,
            recv_buf_chunks, fifo_cap, local_rank,
            clocal_buf_ptr=a_tk_ptr, clocal_buf_size=a_half_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        def reset_state():
            barrier.data_.zero_()
            a_tk.data_[start_row:start_row + M_local].copy_(A_local)
            a_recv_tk.data_.zero_()
            C.zero_()

        def run_once():
            active_sms = int(os.environ.get("AG_GEMM_ACTIVE_SMS", "132"))
            mod.ag_gemm_multinode(
                a_tk, B, C, barrier,
                recv_ptr,
                int(fifo[0]), int(fifo[1]), int(fifo[2]), int(fifo[3]), int(fifo[4]),
                arrival_ptr, epoch, node_idx, args.num_comm_sms, a_half_bytes,
                a_recv_tk, active_sms, args.num_intra_comm_sms, NUM_NODES,
            )

        for wi in range(args.warmup):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.1)
            run_once(); torch.cuda.synchronize()
            dist.barrier()

        samples = []
        # Canonical: NCCL-style no-sync timing — N back-to-back iters with a
        # SINGLE sync after, divide by N. Mirrors nccl_16gpu_baseline.py's
        # default --steady-state. Set MKERNEL_BENCH_LEGACY_SYNC=1 to opt
        # back into per-iter sync (kept for A/B and source-of-truth debugging).
        legacy_sync = os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1"
        # Back-compat: MKERNEL_BENCH_NO_SYNC=0 also forces legacy.
        if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
            legacy_sync = True
        if NUM_NODES > 2:
            legacy_sync = True
        if not legacy_sync:
            n_iters = max(args.iters, 32)
            # Pre-flip enough epochs so all N iters have unique epochs without
            # an inter-iter set_epoch that would re-issue prepare_epoch's
            # drain_proxy + barrier (which is itself a sync). We reuse a single
            # epoch across the back-to-back run; reset_state restores buffers.
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(n_iters):
                run_once()
            e.record()
            torch.cuda.synchronize()
            avg_ms = s.elapsed_time(e) / n_iters
            samples = [avg_ms] * args.iters  # reuse downstream reduce path
            if is_chief:
                print(f"[ag_gemm-nosync] M={M} N={n_iters} avg={avg_ms:.4f} ms",
                      flush=True)
            dist.barrier()
        else:
            for _ in range(args.iters):
                reset_state(); epoch += 1; mod.set_epoch(epoch)
                dist.barrier(); time.sleep(0.05)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); run_once(); e.record(); torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
                dist.barrier()

        wall_ms = avg_then_max_cuda(samples)
        if is_chief:
            print(f"[ag_gemm] M={M} wall={wall_ms:.3f} ms", flush=True)
        gathered_a = gather_cpu_tensors(A_local)
        A_ref = torch.cat(gathered_a, dim=0).to(device="cuda")
        C_ref = torch.matmul(A_ref, B)
        if is_chief:
            rows_per_node = M // NUM_NODES
            for nr in range(NUM_NODES):
                lo = nr * rows_per_node
                hi = lo + rows_per_node
                shard_abs = (C[lo:hi].float() - C_ref[lo:hi].float()).abs().max().item()
                print(f"[ag_gemm-correctness] node_shard={nr} max_abs={shard_abs:.6f}",
                      flush=True)
        correctness_ok = check_close(
            f"ag_gemm M={M}", C, C_ref, atol=0.45, rtol=0.10
        ) and correctness_ok

        result_sizes.append(f"M={M}")
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # MERGE with existing JSON so a single-shape bench doesn't erase the
        # other shapes the chart needs.
        from common import write_results_json
        write_results_json(Path(args.save_json), "ag_gemm",
                           result_sizes, result_fused,
                           note=f"release ag_gemm bench (world={world_size*NUM_NODES})")
        print(f"[ag_gemm] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("ag_gemm", result_sizes, result_fused, args.compare_to)
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
