"""dispatch_gemm MoE Dispatch + Group GEMM bench (release version).

Runs the prebuilt release/build/libdispatch_gemm.so (no JIT) with the default
dispatch_gemm configuration:

    DISPATCH_LOCAL_FIRST=1, DISPATCH_ZERO_COPY=1, DISPATCH_DISPATCH_PIPELINE=1, fused exec mode.
    CHUNK_BYTES=512KB (baked in src/dispatch_gemm.cu).
    SM split: send=4, copy=4, comm=64 at large shapes (131k).

Representative EFA timings:
    8k: 0.631 ms, 16k: 1.057 ms, 32k: 1.538 ms, 65k: 2.669 ms, 131k: 5.16 ms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Required for multicast bind on this hardware/setup. Must be set BEFORE
# importing the prebuilt module since the bind logic reads getenv at C++ time.
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

KERNEL_NAME = "dispatch_gemm"

# Constants matching the build (-DTK_MOE_*).
H = 7168
I = 2048
NUM_EXPERTS = 256
TOP_K = 8
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
CHUNK_BYTES = 512 * 1024  # baked in source

# Default sweep matches the bar chart x-axis. For 3 nodes, token counts must
# divide evenly across 24 GPUs.
DEFAULT_SHAPES = (
    [12288, 24576, 49152, 98304, 196608]
    if NUM_NODES == 3 else
    [8192, 16384, 32768, 65536, 131072]
)


def avg_then_max_cuda(samples):
    avg = sum(float(x) for x in samples) / len(samples)
    t = torch.tensor([avg], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str,
                   default=",".join(str(s) for s in DEFAULT_SHAPES))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    # SM split, with per-shape tuning available through CLI overrides.
    p.add_argument("--num-comm-sms", type=int, default=None)
    p.add_argument("--num-send-sms", type=int, default=None)
    p.add_argument("--num-copy-sms", type=int, default=None)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def _build_routing_uniform(num_tokens_global):
    """Deterministic uniform routing: expert_id = (t*TOP_K + k) % NUM_EXPERTS.

    Each expert gets exactly num_tokens_global * TOP_K / NUM_EXPERTS tokens —
    perfectly balanced (favors any MoE kernel; the absolute best case).
    """
    chosen = torch.empty((num_tokens_global, TOP_K), dtype=torch.int32)
    flat = (torch.arange(num_tokens_global * TOP_K, dtype=torch.int64) %
            NUM_EXPERTS).to(torch.int32)
    chosen.copy_(flat.view(num_tokens_global, TOP_K))
    return chosen


def _build_routing_multinomial(num_tokens_global, rank, world_size, seed=0):
    """Random multinomial routing: bit-equivalent to NCCL/TritonDist baselines.

    Rank 0 samples from a single Categorical(routing_weights) and broadcasts
    the choices to all ranks, matching the NCCL baseline setup.
    Uses a fixed seed for reproducibility (NCCL itself does not seed; we
    seed so multi-run averages are deterministic).
    """
    if rank == 0:
        g = torch.Generator(device="cuda").manual_seed(seed)
        routing_weights = torch.rand(NUM_EXPERTS, device="cuda",
                                     dtype=torch.float32, generator=g)
        chosen = torch.multinomial(
            routing_weights.repeat(num_tokens_global, 1), TOP_K,
            replacement=False, generator=g
        ).to(torch.int32)
    else:
        chosen = torch.empty((num_tokens_global, TOP_K),
                              device="cuda", dtype=torch.int32)
    if dist.is_initialized() and world_size > 1:
        dist.broadcast(chosen, 0)
    return chosen.cpu()


def build_routing(num_tokens_global, world_size_per_node, mode="uniform",
                  rank=0, world_size=1, seed=0):
    """Build per-expert token lists from a routing mode.

    mode:
      "uniform"     — deterministic (t*TOP_K+k) % NUM_EXPERTS (legacy default)
      "multinomial" — random routing matching the NCCL/TritonDist baselines

    Returns (padded_list, expert_to_tokens) where:
      padded_list[e]      = padded token count for expert e (padded to
                            ROW_BLOCK)
      expert_to_tokens[e] = list of (src_node, src_dev, local_tok) tuples
                            of every token routed to expert e
    """
    total_gpus = NUM_NODES * world_size_per_node
    num_local_tokens = num_tokens_global // total_gpus
    assert num_tokens_global % total_gpus == 0

    if mode == "multinomial":
        chosen = _build_routing_multinomial(num_tokens_global, rank,
                                             world_size, seed=seed)
    else:
        chosen = _build_routing_uniform(num_tokens_global)
    chosen_np = chosen.numpy()

    expert_to_tokens = [[] for _ in range(NUM_EXPERTS)]
    for t in range(num_tokens_global):
        global_gpu = t // num_local_tokens
        src_node = global_gpu // world_size_per_node
        src_dev = global_gpu % world_size_per_node
        local_tok = t % num_local_tokens
        for k in range(TOP_K):
            expert_id = int(chosen_np[t, k])
            expert_to_tokens[expert_id].append((src_node, src_dev, local_tok))
    padded_list = []
    for e in range(NUM_EXPERTS):
        actual = len(expert_to_tokens[e])
        padded = ((actual + ROW_BLOCK - 1) // ROW_BLOCK) * ROW_BLOCK
        padded_list.append(padded)
    return padded_list, expert_to_tokens


def build_pull_indices(global_gpu_idx, num_experts_per_dev, padded_list,
                       expert_to_tokens, world_size_per_node):
    """LOCAL_FIRST=1: emit local-source tokens before peer-source within each expert."""
    expert_start = global_gpu_idx * num_experts_per_dev
    expert_end = expert_start + num_experts_per_dev
    this_node = global_gpu_idx // world_size_per_node
    rows = []
    for e in range(expert_start, expert_end):
        padded = padded_list[e]
        toks = expert_to_tokens[e]
        locals_ = [t for t in toks if t[0] == this_node]
        peers_ = [t for t in toks if t[0] != this_node]
        ordered = locals_ + peers_
        for i in range(padded):
            if i < len(ordered):
                rows.append(ordered[i])
            else:
                rows.append((-1, -1, -1))
    arr = torch.tensor(rows, dtype=torch.int32, device="cuda")
    return arr, arr.shape[0]


def per_shape_sm_split(num_local_tokens):
    """Return (n_send, n_copy, n_comm) tuned per local-token count.

    Small global token counts bias toward more send/copy CTAs; larger counts
    leave more CTAs for GEMM.
    """
    # Iter-4 tweak: M=16384 (1024 local) and Iter-5 attempt: M=8192 (512 local)
    # also into the compute-heavy bucket. Freed 8 SMs go to GEMM.
    if num_local_tokens < 512:
        return 8, 8, 44
    return 4, 4, 64


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
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank

    mod = load_module.load(KERNEL_NAME)

    if is_chief:
        print(f"[dispatch_gemm] world={world_size*NUM_NODES} per_node={world_size} "
              f"shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    total_gpus = NUM_NODES * world_size
    num_experts_per_dev = NUM_EXPERTS // total_gpus
    global_gpu_idx = node_idx * world_size + local_rank

    result_sizes = []
    result_fused = []
    correctness_ok = True

    for num_tokens_global in shapes:
        num_local_tokens = num_tokens_global // total_gpus
        assert num_tokens_global % total_gpus == 0

        n_send, n_copy, n_comm = per_shape_sm_split(num_local_tokens)
        # CLI override (lets the launcher tune without rebuilding).
        if args.num_send_sms is not None: n_send = args.num_send_sms
        if args.num_copy_sms is not None: n_copy = args.num_copy_sms
        if args.num_comm_sms is not None: n_comm = args.num_comm_sms

        if is_chief:
            print(f"\n[dispatch_gemm] tokens={num_tokens_global} "
                  f"local_tokens={num_local_tokens} "
                  f"sm(send,copy,comm)=({n_send},{n_copy},{n_comm})", flush=True)

        # Routing mode: multinomial (default) matches NCCL/TritonDist
        # baselines (`nccl_16gpu_baseline.py:167-170`); uniform is the
        # legacy deterministic balanced-MoE workload. Override via
        # MKERNEL_DISPATCH_GEMM_ROUTING={uniform,multinomial}.
        routing_mode = os.environ.get("MKERNEL_DISPATCH_GEMM_ROUTING",
                                       "multinomial").lower()
        # Per-shape seed so different token counts get different draws
        # but each shape is reproducible across runs.
        routing_seed = int(os.environ.get("MKERNEL_DISPATCH_GEMM_ROUTING_SEED",
                                           "0")) + num_tokens_global
        padded_list, expert_to_tokens = build_routing(
            num_tokens_global, world_size, mode=routing_mode,
            rank=rank, world_size=world_size * NUM_NODES, seed=routing_seed)
        if is_chief:
            print(f"[dispatch_gemm] routing={routing_mode} "
                  f"max_expert_tokens={max(len(toks) for toks in expert_to_tokens)} "
                  f"min_expert_tokens={min(len(toks) for toks in expert_to_tokens)}",
                  flush=True)
        padded_ppe = torch.tensor(padded_list, dtype=torch.int32, device="cuda")
        pull_idx, num_padded_local = build_pull_indices(
            global_gpu_idx, num_experts_per_dev, padded_list,
            expert_to_tokens, world_size)

        # Per-expert pure-local row_block count (LOCAL_FIRST baked on).
        this_node = global_gpu_idx // world_size
        expert_start = global_gpu_idx * num_experts_per_dev
        local_rb_list = []
        for e in range(expert_start, expert_start + num_experts_per_dev):
            local_count = sum(1 for t in expert_to_tokens[e] if t[0] == this_node)
            local_rb_list.append(local_count // ROW_BLOCK)
        local_rb_per_expert = torch.tensor(local_rb_list, dtype=torch.int32, device="cuda")

        # Per-rank deterministic data.
        torch.manual_seed(42 + global_gpu_idx)
        torch.cuda.manual_seed(42 + global_gpu_idx)
        pre_tokens_data = (
            torch.randn((num_local_tokens, H), device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25)
        )
        torch.manual_seed(100 + global_gpu_idx)
        torch.cuda.manual_seed(100 + global_gpu_idx)
        weights = (
            torch.randn((num_experts_per_dev, H, I), device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25)
        )

        pre_tokens = mod.DistBuffer(
            (num_local_tokens, H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        pre_tokens.data_.copy_(pre_tokens_data)

        n_peers = NUM_NODES - 1
        peer_tokens = mod.DistBuffer(
            (n_peers * num_local_tokens, H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        peer_tokens.data_.zero_()

        num_row_blocks = max(1, (num_padded_local + ROW_BLOCK - 1) // ROW_BLOCK)
        barrier = mod.DistBuffer(
            (world_size, num_row_blocks, 1), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        barrier.data_.zero_()

        sync_barrier = mod.DistBuffer(
            (1, 1, 2), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        sync_barrier.data_.zero_()

        post_tokens = torch.zeros((num_padded_local, H),
                                   device="cuda", dtype=torch.bfloat16)
        outputs = torch.zeros((num_padded_local, I),
                               device="cuda", dtype=torch.bfloat16)

        pre_tokens_bytes = num_local_tokens * H * 2
        total_chunks = (pre_tokens_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
        # Per-peer sizing. At N == 2 the multiplier is 1; larger runs allocate
        # one receive/arrival slot per remote peer.
        recv_buf_chunks = n_peers * total_chunks
        copy_ready = mod.DistBuffer(
            (world_size, recv_buf_chunks, 1), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        copy_ready.data_.zero_()
        send_buf = torch.empty((num_local_tokens, H),
                                device="cuda", dtype=torch.bfloat16)

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_chunks * 2:
            fifo_cap *= 2

        # ZERO_COPY baked on: peer_tokens IS the RDMA destination.
        external_recv_buf_ptr = int(peer_tokens.data_.data_ptr())
        # Zero-copy send: register pre_tokens as the proxy's data MR. Kernel
        # reads straight from pre_tokens — no send_buf pack required.
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            int(pre_tokens.data_.data_ptr()), pre_tokens_bytes,
            n_peers * pre_tokens_bytes, recv_buf_chunks, fifo_cap, local_rank,
            external_recv_buf_ptr,
            int(pre_tokens.data_.data_ptr()),
            pre_tokens_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        # send_buf no longer used by the proxy (zero-copy reads pre_tokens
        # directly via DMA-BUF MR), but the allocation is kept for the
        # session-config arg until the signature is cleaned up.

        def run_once():
            mod.moe_dispatch_gemm_fused(
                pre_tokens, peer_tokens, copy_ready,
                post_tokens, weights, outputs, padded_ppe, pull_idx,
                local_rb_per_expert, barrier, sync_barrier,
                recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch,
                node_idx, num_local_tokens, num_padded_local,
                n_send, n_copy, n_comm,
                num_nodes=NUM_NODES,
            )

        def reset_state():
            barrier.data_.zero_()
            sync_barrier.data_.zero_()
            copy_ready.data_.zero_()

        # Prime only for the single-peer path. N-node fanout starts from a
        # fresh epoch/reset before the measured iteration.
        if NUM_NODES <= 2:
            reset_state(); dist.barrier(); time.sleep(0.05)
            run_once(); torch.cuda.synchronize(); dist.barrier()

        epoch += 1; mod.set_epoch(epoch); reset_state()
        dist.barrier(); time.sleep(0.05)

        # Warmup
        for _ in range(args.warmup):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            run_once(); torch.cuda.synchronize()
            dist.barrier()

        # Timed iters
        samples = []
        # Canonical: NCCL-style no-sync timing — N back-to-back iters, single
        # sync at end, divide by N. Set MKERNEL_BENCH_LEGACY_SYNC=1 (or
        # MKERNEL_BENCH_NO_SYNC=0) to opt back. Per-shape N: keep total
        # measurement >=~100 ms but cap by tokens since 131k-token iter is
        # ~5 ms; smaller shapes use bigger N for stability.
        legacy_sync = os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1"
        if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
            legacy_sync = True
        if NUM_NODES > 2:
            legacy_sync = True
        if not legacy_sync:
            # Tier N by shape so total measurement ~>= 100 ms.
            if num_tokens_global <= 16384:
                n_iters = max(args.iters, 64)
            elif num_tokens_global <= 65536:
                n_iters = max(args.iters, 32)
            else:
                n_iters = max(args.iters, 24)
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
            samples = [avg_ms] * args.iters
            if is_chief:
                print(f"[dispatch_gemm-nosync] tokens={num_tokens_global} "
                      f"N={n_iters} avg={avg_ms:.4f} ms", flush=True)
            dist.barrier()
        else:
            for _ in range(args.iters):
                reset_state(); epoch += 1; mod.set_epoch(epoch)
                dist.barrier(); time.sleep(0.05)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); run_once(); e.record(); torch.cuda.synchronize()
                dist.barrier()
                samples.append(s.elapsed_time(e))

        wall_ms = avg_then_max_cuda(samples)
        if is_chief:
            print(f"[dispatch_gemm] tokens={num_tokens_global} wall={wall_ms:.3f} ms", flush=True)
        if args.mode == "check":
            gathered_tokens = gather_cpu_tensors(pre_tokens_data)
            pull_cpu = pull_idx.detach().cpu()
            post_ref_cpu = torch.zeros((num_padded_local, H), dtype=torch.bfloat16)
            for row in range(num_padded_local):
                src_node, src_dev, local_tok = [int(x) for x in pull_cpu[row].tolist()]
                if src_node >= 0:
                    src_rank = src_node * world_size + src_dev
                    post_ref_cpu[row].copy_(gathered_tokens[src_rank][local_tok])
            out_ref = torch.zeros_like(outputs)
            row_off = 0
            expert_start = global_gpu_idx * num_experts_per_dev
            for local_e, expert_id in enumerate(
                range(expert_start, expert_start + num_experts_per_dev)
            ):
                rows = padded_list[expert_id]
                if rows > 0:
                    out_ref[row_off:row_off + rows].copy_(
                        torch.matmul(post_ref_cpu[row_off:row_off + rows].to("cuda"),
                                     weights[local_e])
                    )
                row_off += rows
            correctness_ok = check_close(
                f"dispatch_gemm tokens={num_tokens_global}",
                outputs, out_ref, atol=0.55, rtol=0.12
            ) and correctness_ok
        result_sizes.append(f"tokens={num_tokens_global}")
        result_fused.append(wall_ms)
        # Don't call destroy_session — re-creating per shape is fine and
        # destroy_session has caused state issues. Process exit cleans up.

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "dispatch_gemm",
                           result_sizes, result_fused,
                           note=f"release dispatch_gemm bench (world={world_size*NUM_NODES})")
        print(f"[dispatch_gemm] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("dispatch_gemm", result_sizes, result_fused,
                                   args.compare_to)
        ok = ok and correctness_ok
        dist.destroy_process_group()
        if not ok:
            return 1
        return 0
    if not correctness_ok:
        dist.destroy_process_group()
        return 1

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
