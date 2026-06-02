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
# Optional debug: DGC_DEBUG=1 makes CUDA errors synchronous at the faulting launch.
if os.environ.get("DGC_DEBUG", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import (  # noqa: E402
    check_close,
    check_deterministic_rerun,
    compare_named_results,
    gather_cpu_tensors,
    get_peer_ips,
    get_peer_ports,
)

KERNEL_NAME = "dispatch_gemm_glu_combine"

# Constants matching the build (-DTK_MOE_* / DGC_*). The naive FFN compute is
# slow at full size, so the kernel is typically built small (e.g. DGC_H=256
# DGC_I=128); set TK_MOE_H/TK_MOE_I here to the SAME values used at build time.
H = int(os.environ.get("TK_MOE_H", 7168))
I = int(os.environ.get("TK_MOE_I", 2048))
NUM_EXPERTS = int(os.environ.get("TK_MOE_NUM_EXPERTS", 256))
TOP_K = int(os.environ.get("TK_MOE_TOP_K", 8))
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
CHUNK_BYTES = 16 * 1024 * 1024  # baked in src (dispatch_gemm_glu_combine.cuh)

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


def build_pull_rows_cpu(global_gpu_idx, num_experts_per_dev, padded_list,
                        expert_to_tokens, world_size_per_node):
    """Same ordered (src_node, src_dev, src_token) rows as build_pull_indices,
    but as a plain python list (CPU), used to replay every producer's row order
    when building the GATHER inverse index. Padding rows are (-1,-1,-1)."""
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
            rows.append(ordered[i] if i < len(ordered) else (-1, -1, -1))
    return rows


def build_gather_indices(this_global_gpu, total_gpus, num_experts_per_dev,
                         padded_list, expert_to_tokens, world_size_per_node,
                         num_local_tokens, node_idx):
    """Build the atomic-free GATHER inverse index for the INTRA-node path.

    Returns (owner_offset[num_local_tokens+1] int32, row_to_slot[npad] int32,
             row_to_owner[npad] int32, max_contrib int).

    Slot assignment is GLOBALLY consistent: for each owner GPU g and owner token
    t, contributing intra rows are enumerated in (producer_gpu, producer_row)
    order; the k-th contribution gets slot owner_offset_g[t]+k in g's comb_buf.
    Each producer rank replays the SAME enumeration so its row_to_slot matches
    the owner's owner_offset/gather range. INTER (remote-source) rows and padding
    get row_to_owner=-1 (handled by the unchanged stage->send->reduce path)."""
    # Replay every producer rank's ordered rows once (cheap, off timed path).
    all_rows = [
        build_pull_rows_cpu(g, num_experts_per_dev, padded_list,
                            expert_to_tokens, world_size_per_node)
        for g in range(total_gpus)
    ]

    # Owners are the world_size_per_node GPUs on THIS node. Build a global CSR
    # offset for every owner GPU on this node so each producer can compute the
    # ABSOLUTE slot it writes into the owner's comb_buf.
    node_base = node_idx * world_size_per_node
    # First pass: count intra contributions per (owner_gpu_on_node, owner_token).
    owner_count = [
        [0] * num_local_tokens for _ in range(world_size_per_node)
    ]
    for g in range(total_gpus):
        for (sn, sd, st) in all_rows[g]:
            if sn < 0 or sn != node_idx:
                continue
            owner_count[sd][st] += 1
    # CSR prefix-sum offset per owner GPU on this node.
    owner_offset_per_dev = []
    for d in range(world_size_per_node):
        off = [0] * (num_local_tokens + 1)
        for t in range(num_local_tokens):
            off[t + 1] = off[t] + owner_count[d][t]
        owner_offset_per_dev.append(off)

    # Second pass: assign ABSOLUTE slots in the SAME global (producer_gpu, row)
    # order. running[d][t] = next k for owner GPU d, token t.
    running = [[0] * num_local_tokens for _ in range(world_size_per_node)]
    npad = len(all_rows[this_global_gpu])
    row_to_slot = [0] * npad
    row_to_owner = [-1] * npad
    for g in range(total_gpus):
        is_me = (g == this_global_gpu)
        for r, (sn, sd, st) in enumerate(all_rows[g]):
            if sn < 0 or sn != node_idx:
                continue  # padding or INTER (unchanged stage path)
            k = running[sd][st]
            running[sd][st] = k + 1
            if is_me:
                row_to_owner[r] = sd
                row_to_slot[r] = owner_offset_per_dev[sd][st] + k  # ABSOLUTE

    # owner_offset (CSR) for THIS gpu as the owner of its num_local_tokens.
    this_dev = this_global_gpu - node_base
    owner_offset = owner_offset_per_dev[this_dev]
    max_contrib = max([1] + [max(c) for c in owner_count])

    owner_offset_t = torch.tensor(owner_offset, dtype=torch.int32, device="cuda")
    row_to_slot_t = torch.tensor(row_to_slot, dtype=torch.int32, device="cuda")
    row_to_owner_t = torch.tensor(row_to_owner, dtype=torch.int32, device="cuda")
    # comb_buf row capacity must be node-uniform (DistBuffer shape is the same on
    # every local GPU). Size to the MAX over devices of that device's owner-slot
    # total (= #intra rows targeting that device). +1 guards the empty case.
    comb_slots = max(1, max(off[-1] for off in owner_offset_per_dev))
    return owner_offset_t, row_to_slot_t, row_to_owner_t, max_contrib, comb_slots


def build_inter_gather_indices(this_global_gpu, total_gpus, num_experts_per_dev,
                               padded_list, expert_to_tokens, world_size_per_node,
                               num_local_tokens, node_idx):
    """INTER analogue of build_gather_indices (sender-side pre-sum).

    Identical to build_gather_indices but the filter is flipped to sn != node_idx
    (inter rows) and the owner key is the REMOTE-node owner-dev sd (which, by the
    per-owner-dev staging convention, equals the LOCAL staging-GPU index that the
    atomic scatter writes into and combine_send ships to remote GPU sd). The owner
    gather runs on local GPU d's own inter_comb_buf and writes local pre_tokens[d]
    UPPER, shipped to remote GPU d.

    Returns (inter_owner_offset[n_peers*(L+1)] int32 (flat per-peer CSR),
             inter_row_to_slot[npad] int32, inter_row_to_owner[npad] int32,
             inter_comb_slots int).
    """
    all_rows = [
        build_pull_rows_cpu(g, num_experts_per_dev, padded_list,
                            expert_to_tokens, world_size_per_node)
        for g in range(total_gpus)
    ]
    n_peers = NUM_NODES - 1
    node_base = node_idx * world_size_per_node
    L = num_local_tokens

    # Destination peer slot for an owner on remote node `sn`, matching the kernel's
    # peer_rank_for_slot ring (combine_send ships region p to (node_idx+1+p)%NUM_NODES).
    def peer_slot_of(sn):
        return (sn - node_idx - 1) % NUM_NODES

    # First pass: count INTER contributions per (owner-staging-dev sd, dest peer, owner token).
    inter_count = [[[0] * L for _ in range(n_peers)]
                   for _ in range(world_size_per_node)]
    for g in range(total_gpus):
        for (sn, sd, st) in all_rows[g]:
            if sn < 0 or sn == node_idx:
                continue  # padding or INTRA (handled by intra gather)
            inter_count[sd][peer_slot_of(sn)][st] += 1

    # Per owner-staging-dev: one flat [n_peers*(L+1)] CSR with ABSOLUTE slots into
    # that dev's inter_comb_buf. Peer p's slots follow all of peer (p-1)'s, so each
    # (peer, token) gather range addresses a contiguous block. At N==2 (n_peers==1)
    # this is a single [L+1] CSR — identical to the legacy single-peer layout.
    inter_off_per_dev = []
    dev_totals = []
    for d in range(world_size_per_node):
        off = [0] * (n_peers * (L + 1))
        cum = 0
        for p in range(n_peers):
            b = p * (L + 1)
            off[b] = cum
            for t in range(L):
                off[b + t + 1] = off[b + t] + inter_count[d][p][t]
            cum = off[b + L]
        inter_off_per_dev.append(off)
        dev_totals.append(cum)

    # Second pass: assign ABSOLUTE slots in the SAME global (producer_gpu, row)
    # order so each producer's slot matches the owner's CSR gather range.
    running = [[[0] * L for _ in range(n_peers)]
               for _ in range(world_size_per_node)]
    npad = len(all_rows[this_global_gpu])
    inter_row_to_slot = [0] * npad
    inter_row_to_owner = [-1] * npad
    for g in range(total_gpus):
        is_me = (g == this_global_gpu)
        for r, (sn, sd, st) in enumerate(all_rows[g]):
            if sn < 0 or sn == node_idx:
                continue
            p = peer_slot_of(sn)
            k = running[sd][p][st]
            running[sd][p][st] = k + 1
            if is_me:
                inter_row_to_owner[r] = sd
                inter_row_to_slot[r] = inter_off_per_dev[sd][p * (L + 1) + st] + k  # ABSOLUTE

    this_dev = this_global_gpu - node_base
    inter_owner_offset = inter_off_per_dev[this_dev]   # flat [n_peers*(L+1)]
    inter_comb_slots = max(1, max(dev_totals))

    inter_owner_offset_t = torch.tensor(inter_owner_offset, dtype=torch.int32, device="cuda")
    inter_row_to_slot_t = torch.tensor(inter_row_to_slot, dtype=torch.int32, device="cuda")
    inter_row_to_owner_t = torch.tensor(inter_row_to_owner, dtype=torch.int32, device="cuda")
    return (inter_owner_offset_t, inter_row_to_slot_t, inter_row_to_owner_t,
            inter_comb_slots)


def per_shape_sm_split(num_local_tokens):
    """Return (n_send, n_copy, n_comm) tuned per local-token count.

    Small global token counts bias toward more send/copy CTAs; larger counts
    leave more CTAs for GEMM.
    """
    # Iter-4 tweak: M=16384 (1024 local) and Iter-5 attempt: M=8192 (512 local)
    # also into the compute-heavy bucket. Freed 8 SMs go to GEMM.
    ov = os.environ.get("DGC_SM_SPLIT")
    if ov:
        a, b, c = (int(x) for x in ov.split(","))
        return a, b, c
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

    # Peer IP / TCP port for session bootstrap (matches experiment harness).
    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        peer_node = 1 if node_idx == 0 else 0
        peer_ip = os.environ.get(f"NODE{peer_node}_IP")
        if not peer_ip:
            raise RuntimeError(f"NODE{peer_node}_IP must be set, or set PEER_IP explicitly")
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

        # atomic-free GATHER combine inverse index (INTRA path). Built off
        # the timed path from the global routing table. Default ON; opt out of the
        # atomic-free intra gather with DGC_COMBINE_GATHER=0.
        use_gather = os.environ.get("DGC_COMBINE_GATHER", "1") == "1"
        (owner_offset_t, row_to_slot_t, row_to_owner_t,
         max_contrib, comb_slots) = build_gather_indices(
            global_gpu_idx, total_gpus, num_experts_per_dev, padded_list,
            expert_to_tokens, world_size, num_local_tokens, node_idx)
        if is_chief:
            print(f"[dispatch_gemm] gather={use_gather} max_contrib={max_contrib} "
                  f"comb_slots={comb_slots} (cap {num_local_tokens} tokens)", flush=True)

        # atomic-free INTER gather index (sender-side pre-sum). Built off the
        # timed path. Default OFF; opt in with DGC_INTER_GATHER=1. Requires the
        # intra gather (use_gather) to be on (shares the bar(5) producer/gather phase).
        use_inter_gather = (os.environ.get("DGC_INTER_GATHER", "1") == "1") and use_gather
        (inter_owner_offset_t, inter_row_to_slot_t, inter_row_to_owner_t,
         inter_comb_slots) = build_inter_gather_indices(
            global_gpu_idx, total_gpus, num_experts_per_dev, padded_list,
            expert_to_tokens, world_size, num_local_tokens, node_idx)
        # inter_comb_slots is node-uniform by construction (both nodes derive it
        # from the SAME global routing replay); assert as a cheap guard.
        _ics = torch.tensor([inter_comb_slots], dtype=torch.int64, device="cuda")
        dist.all_reduce(_ics, op=dist.ReduceOp.MAX)
        assert int(_ics.item()) == inter_comb_slots, (
            f"inter_comb_slots not node-uniform: {inter_comb_slots} vs {int(_ics.item())}")
        if is_chief:
            _hbm = inter_comb_slots * H * 2 / 1e9
            print(f"[dispatch_gemm] inter_gather={use_inter_gather} "
                  f"inter_comb_slots={inter_comb_slots} ratio={inter_comb_slots/num_local_tokens:.3f} "
                  f"HBM={_hbm:.3f}GB", flush=True)
            if _hbm > 2.0 or inter_comb_slots / num_local_tokens > 22.0:
                print(f"[dispatch_gemm] WARNING inter_comb_buf large: "
                      f"HBM={_hbm:.3f}GB ratio={inter_comb_slots/num_local_tokens:.3f}", flush=True)

        # Per-expert pure-local row_block count (LOCAL_FIRST baked on).
        this_node = global_gpu_idx // world_size
        expert_start = global_gpu_idx * num_experts_per_dev
        local_rb_list = []
        for e in range(expert_start, expert_start + num_experts_per_dev):
            local_count = sum(1 for t in expert_to_tokens[e] if t[0] == this_node)
            local_rb_list.append(local_count // ROW_BLOCK)
        local_rb_per_expert = torch.tensor(local_rb_list, dtype=torch.int32, device="cuda")

        # Per dispatched row -> local expert id (each expert occupies a padded
        # contiguous row range). The naive FFN reads this to pick W1[e]/W2[e].
        row_expert_list = []
        for local_e in range(num_experts_per_dev):
            row_expert_list.extend([local_e] * padded_list[expert_start + local_e])
        row_expert = torch.tensor(row_expert_list, dtype=torch.int32, device="cuda")
        assert row_expert.numel() == num_padded_local

        # Per-dispatched-row combine weight. Arbitrary values are fine for the
        # correctness check: the reference uses the SAME (gathered) weights, so
        # this validates the combine summation, not the weight source.
        torch.manual_seed(7 + global_gpu_idx)
        torch.cuda.manual_seed(7 + global_gpu_idx)
        topk_weights = torch.rand((num_padded_local,), device="cuda", dtype=torch.float32)

        # Per-rank deterministic data.
        torch.manual_seed(42 + global_gpu_idx)
        torch.cuda.manual_seed(42 + global_gpu_idx)
        pre_tokens_data = (
            torch.randn((num_local_tokens, H), device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25)
        )
        # FFN weights: W1 = [E_per_dev, 2I, H] (gate | up), W2 = [E_per_dev, H, I].
        torch.manual_seed(100 + global_gpu_idx)
        torch.cuda.manual_seed(100 + global_gpu_idx)
        # Tensor-core GEMM layout: gemm1 = post[M,H] @ w1[e][H,2I]; gemm2 =
        # act[M,I] @ w2[e][I,H]  (weights stored [K, N], no transpose in-kernel).
        w1 = torch.randn((num_experts_per_dev, H, 2 * I),
                         device="cuda", dtype=torch.bfloat16) * (H ** -0.5)
        torch.manual_seed(200 + global_gpu_idx)
        torch.cuda.manual_seed(200 + global_gpu_idx)
        w2 = torch.randn((num_experts_per_dev, I, H),
                         device="cuda", dtype=torch.bfloat16) * (I ** -0.5)

        n_peers = NUM_NODES - 1
        # (1 + n_peers)x buffer in ONE registered RDMA MR: the first L rows are the
        # dispatch region (broadcast to all peers), the next n_peers*L rows are the
        # inter-node combine region (one dense [L,H] send buffer per destination
        # peer). Combine never touches the dispatch region, so it needs no cross-node
        # barrier. At N==2 this is the legacy 2x buffer. Combine runs in the kernel.
        pre_tokens = mod.DistBuffer(
            ((1 + n_peers) * num_local_tokens, H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        pre_tokens.data_[:num_local_tokens].copy_(pre_tokens_data)  # dispatch input
        pre_tokens.data_[num_local_tokens:].zero_()                 # combine staging

        peer_tokens = mod.DistBuffer(
            (2 * n_peers * num_local_tokens, H), dtype=torch.bfloat16,
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
        y_expert = torch.zeros((num_padded_local, H),
                               device="cuda", dtype=torch.bfloat16)
        # GEMM intermediates (HBM): gemm1 output / gemm2 input.
        h1 = torch.zeros((num_padded_local, 2 * I), device="cuda", dtype=torch.bfloat16)
        act = torch.zeros((num_padded_local, I), device="cuda", dtype=torch.bfloat16)

        # Combine output: per-GPU [num_local_tokens, H] fp32, IPC-shared across
        # the node's GPUs so combine can atomic-add into the token owner's slot.
        y_out = mod.DistBuffer(
            (num_local_tokens, H), dtype=torch.float32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        y_out.data_.zero_()

        # owner-indexed contribution buffer (intra), IPC-shared. Producer
        # GPUs store pre-weighted rows into the owner GPU's comb_buf slots
        # (non-atomic, one writer per slot); the owner gather-reduces locally.
        # Row count is node-uniform = comb_slots (max over devices). No zero-fill
        # needed: the owner reads only written slots [owner_offset[t],off[t+1]).
        comb_buf = mod.DistBuffer(
            (max(1, comb_slots), H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )

        # owner-staging-dev-indexed INTER contribution buffer, IPC-shared.
        # Producer GPUs store pre-weighted inter rows into the staging GPU's slots
        # (non-atomic, one writer per slot); the owner gather-reduces locally and
        # writes the dense pre_tokens UPPER send row. Node-uniform = inter_comb_slots.
        inter_comb_buf = mod.DistBuffer(
            (max(1, inter_comb_slots), H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )

        pre_tokens_bytes = num_local_tokens * H * 2
        total_chunks = (pre_tokens_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
        # Per-peer sizing. At N == 2 the multiplier is 1 — same buffer /
        # arrival-flag sizing as the legacy single-peer setup.
        recv_buf_chunks = n_peers * total_chunks          # dispatch flag/chunk count
        flag_tiles = 2 * recv_buf_chunks                  # dispatch (lower) + combine (upper)
        copy_ready = mod.DistBuffer(
            (world_size, recv_buf_chunks, 1), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        copy_ready.data_.zero_()
        send_buf = torch.empty((num_local_tokens, H),
                                device="cuda", dtype=torch.bfloat16)

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < flag_tiles * 2:
            fifo_cap *= 2

        # ZERO_COPY baked on: peer_tokens IS the RDMA destination. Register the FULL
        # local pre_tokens MR (dispatch lower + n_peers combine-upper regions) and the
        # remote peer_tokens MR (n_peers dispatch-recv + n_peers combine-recv), and
        # reserve 2x arrival flags so dispatch and combine use disjoint flag slots.
        local_pre_tokens_bytes = (1 + n_peers) * pre_tokens_bytes
        external_recv_buf_ptr = int(peer_tokens.data_.data_ptr())
        peer_ips = get_peer_ips(node_idx, NUM_NODES)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            int(pre_tokens.data_.data_ptr()), local_pre_tokens_bytes,
            2 * n_peers * pre_tokens_bytes, flag_tiles, fifo_cap, local_rank,
            external_recv_buf_ptr,
            int(pre_tokens.data_.data_ptr()),
            local_pre_tokens_bytes,
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
            mod.moe_dispatch_gemm_glu_combine_fused(
                pre_tokens, peer_tokens, copy_ready,
                post_tokens, w1, w2, h1, act, y_expert, row_expert, topk_weights,
                padded_ppe, pull_idx,
                local_rb_per_expert, barrier, sync_barrier, y_out,
                comb_buf, owner_offset_t, row_to_slot_t, row_to_owner_t,
                inter_comb_buf, inter_owner_offset_t, inter_row_to_slot_t,
                inter_row_to_owner_t, int(use_inter_gather),
                recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch,
                node_idx, num_local_tokens, num_padded_local,
                n_send, n_copy, n_comm,
                int(use_gather),
                num_nodes=NUM_NODES,
            )

        def reset_state():
            barrier.data_.zero_()
            sync_barrier.data_.zero_()
            copy_ready.data_.zero_()
            y_out.data_.zero_()   # combine accumulates with atomicAdd

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
        gathered_tokens = gather_cpu_tensors(pre_tokens_data)
        pull_cpu = pull_idx.detach().cpu()
        post_ref_cpu = torch.zeros((num_padded_local, H), dtype=torch.bfloat16)
        for row in range(num_padded_local):
            src_node, src_dev, local_tok = [int(x) for x in pull_cpu[row].tolist()]
            if src_node >= 0:
                src_rank = src_node * world_size + src_dev
                post_ref_cpu[row].copy_(gathered_tokens[src_rank][local_tok])
        # FFN reference per dispatched row: gemm1(2I) -> SwiGLU -> gemm2(H).
        y_expert_ref = torch.zeros_like(y_expert)
        row_off = 0
        expert_start = global_gpu_idx * num_experts_per_dev
        for local_e, expert_id in enumerate(
            range(expert_start, expert_start + num_experts_per_dev)
        ):
            rows = padded_list[expert_id]
            if rows > 0:
                post_g = post_ref_cpu[row_off:row_off + rows].to("cuda").float()
                h1r = post_g @ w1[local_e].float()             # [rows, 2I]  (w1[e]=[H,2I])
                gate, up = h1r[:, :I], h1r[:, I:]
                actr = torch.nn.functional.silu(gate) * up     # [rows, I]
                ye = actr @ w2[local_e].float()                # [rows, H]   (w2[e]=[I,H])
                y_expert_ref[row_off:row_off + rows].copy_(ye.to(torch.bfloat16))
            row_off += rows
        correctness_ok = check_close(
            f"dispatch_gemm_glu_combine tokens={num_tokens_global}",
            y_expert, y_expert_ref, atol=0.1, rtol=0.05
        ) and correctness_ok

        # ---- Perf: naive torch (cuBLAS) FFN, same per-expert grouped GEMM the
        # kernel does naively. Times compute only (no dispatch/combine comm), as a
        # lower-bound reference for the kernel's naive-GEMM compute. ----
        post_g_all = post_ref_cpu.to("cuda").float()
        w1f = [w1[e].float().contiguous() for e in range(num_experts_per_dev)]
        w2f = [w2[e].float().contiguous() for e in range(num_experts_per_dev)]
        torch.cuda.synchronize(); dist.barrier()
        _t0 = time.perf_counter()
        for _ in range(args.iters):
            row_off2 = 0
            for local_e in range(num_experts_per_dev):
                rows = padded_list[expert_start + local_e]
                if rows > 0:
                    # NB: distinct names so we don't shadow the kernel's h1/act buffers.
                    _h1 = post_g_all[row_off2:row_off2 + rows] @ w1f[local_e]
                    _act = torch.nn.functional.silu(_h1[:, :I]) * _h1[:, I:]
                    _ = _act @ w2f[local_e]
                row_off2 += rows
        torch.cuda.synchronize()
        naive_ffn_ms = avg_then_max_cuda(
            [(time.perf_counter() - _t0) / max(1, args.iters) * 1e3])
        if is_chief:
            print(f"[PERF] tokens={num_tokens_global} kernel_wall={wall_ms:.3f}ms "
                  f"naive_torch_ffn={naive_ffn_ms:.3f}ms "
                  f"(kernel is naive-GEMM; ref is cuBLAS, compute-only)", flush=True)

        # ---- Combine check ----
        # y_out accumulates via atomicAdd, so the post-timing value reflects all
        # N timed iters. Do one clean run (y_out zeroed) for the combine check.
        reset_state(); epoch += 1; mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.05)
        run_once(); torch.cuda.synchronize(); dist.barrier()
        # y_out[t] on this GPU must equal the weighted sum over ALL dispatched
        # rows r (across every rank) whose dispatch source is (this node, this
        # dev, t): sum_r topk_weights[r] * y_expert[r]. Gathered via all_gather
        # (variable per-rank row counts) so it covers intra- and inter-node rows.
        def _all_gather_var(t):
            objs = [None] * dist.get_world_size()
            dist.all_gather_object(objs, t.detach().cpu())
            return objs
        g_ye = _all_gather_var(y_expert)
        g_pull = _all_gather_var(pull_idx)
        g_w = _all_gather_var(topk_weights)
        # Full combine (intra + inter-node) is now in the fused kernel. Set
        # DGC_COMBINE_INTRA_ONLY=1 only to debug the intra-node path in isolation.
        intra_only = os.environ.get("DGC_COMBINE_INTRA_ONLY", "0") == "1"
        y_out_ref = torch.zeros((num_local_tokens, H), dtype=torch.float32)
        for sr in range(len(g_ye)):
            if intra_only and (sr // world_size) != node_idx:
                continue
            pull = g_pull[sr]
            ye = g_ye[sr].float()
            wt = g_w[sr].float()
            m = (pull[:, 0] == node_idx) & (pull[:, 1] == local_rank)
            for r in m.nonzero(as_tuple=True)[0].tolist():
                t = int(pull[r, 2])
                if t >= 0:
                    y_out_ref[t] += wt[r].item() * ye[r]
        correctness_ok = check_close(
            f"dispatch_gemm_glu_combine COMBINE tokens={num_tokens_global}",
            y_out.data_, y_out_ref.to("cuda"), atol=0.1, rtol=0.05
        ) and correctness_ok

        if os.environ.get("MKERNEL_INVARIANT_DETERMINISTIC", "0") == "1":
            torch.cuda.synchronize(); dist.barrier(); time.sleep(0.1)
            epoch += 1; mod.set_epoch(epoch); reset_state()
            dist.barrier(); time.sleep(0.05)
            run_once(); torch.cuda.synchronize()
            det_out_a = y_expert.detach().clone()
            epoch += 1; mod.set_epoch(epoch); reset_state()
            dist.barrier(); time.sleep(0.05)
            run_once(); torch.cuda.synchronize()
            det_out_b = y_expert.detach().clone()
            correctness_ok = check_deterministic_rerun(
                f"dispatch_gemm tokens={num_tokens_global}",
                det_out_a, det_out_b, is_chief
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
