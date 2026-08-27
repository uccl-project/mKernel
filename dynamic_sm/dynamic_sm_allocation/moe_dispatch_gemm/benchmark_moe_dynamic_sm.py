#!/usr/bin/env python3
"""Sweep static num_comm_sms and fully dynamic scheduling for MoE Dispatch+GEMM.

The MoE kernel fuses cross-device token dispatch (all-to-all via TMA peer
writes) with expert GEMM.  The static version assigns a fixed number of SMs
to dispatch vs GEMM.  The dynamic version lets the middle CTAs interleave
based on task queue depth.

Run (4 GPUs, GPUs 4-7):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29610 \\
    dynamic_sm_allocation/moe_dispatch_gemm/benchmark_moe_dynamic_sm.py \\
    --arch sm_90a
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
STATIC_SRC  = THIS_DIR / "moe_dispatch_gemm.cu"
DYNAMIC_SRC = THIS_DIR / "dynamic_moe_dispatch_gemm.cu"
ADAPTIVE_SRC = THIS_DIR / "moe_dispatch_gemm_dynamic_sm_allocation.cu"
INCLUDE     = REPO_ROOT / "include"
DEFAULT_OUTPUT = THIS_DIR / "results" / "moe_dynamic_sm.csv"
DEFAULT_SUMMARY_OUTPUT = THIS_DIR / "results" / "moe_dynamic_sm_summary.csv"
DEFAULT_ADAPTIVE_OUTPUT = THIS_DIR / "results" / "moe_dynamic_sm_allocation.csv"
DEFAULT_ADAPTIVE_SUMMARY_OUTPUT = THIS_DIR / "results" / "moe_dynamic_sm_allocation_summary.csv"

# ---------------------------------------------------------------------------
# MoE shape parameters (match TK_MOE_* defaults)
# ---------------------------------------------------------------------------
H           = 7168   # hidden dim
I           = 2048   # expert intermediate dim
NUM_EXPERTS = 256
TOP_K       = 8


def _num_devices(explicit=None):
    if explicit is not None:
        return int(explicit)
    return int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "8")))


def _module_name(nd):           return f"osgc_at_moe_static_d{nd}"
def _dynamic_module_name(nd):
    suffixes: list[str] = []
    if os.environ.get("MOE_PRINT_SCHED_STATS") == "1":
        suffixes.append("sched")
    forced_target = os.environ.get("FIXED_DYNAMIC_TARGET")
    if forced_target:
        suffixes.append(f"force{forced_target}")
    debug_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    return f"osgc_at_moe_dynamic_d{nd}{debug_suffix}"
def _adaptive_module_name():    return "osgc_at_moe_dynamic_sm_allocation_d2"


def _cuda_flags(arch, nd, extra=None):
    flags = [
        "-O3", "-std=c++20", "--use_fast_math",
        "-DKITTENS_HOPPER", "--extended-lambda", "--expt-relaxed-constexpr",
        f"-DTK_NUM_DEVICES={nd}",
        f"-DTK_MOE_H={H}", f"-DTK_MOE_I={I}",
        f"-DTK_MOE_TOP_K={TOP_K}", f"-DTK_MOE_NUM_EXPERTS={NUM_EXPERTS}",
    ]
    if os.environ.get("MOE_PRINT_SCHED_STATS") == "1":
        flags.append("-DMOE_PRINT_SCHED_STATS")
    forced_target = os.environ.get("FIXED_DYNAMIC_TARGET")
    if forced_target:
        flags.append(f"-DFIXED_DYNAMIC_TARGET={int(forced_target)}")
    if arch and arch.lower() == "sm_90a":
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
    elif arch:
        flags.append(f"-arch={arch}")
    if extra:
        flags += extra
    return flags


def _remove_stale_locks(build_dir: Path) -> None:
    """Remove stale NFS lock files left by killed build processes."""
    import fcntl
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return
    try:
        with open(lock_path, "r+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def load_static(arch, nd):
    name = _module_name(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    bd.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(bd)
    return load(name=name, sources=[str(STATIC_SRC)],
                extra_include_paths=[str(INCLUDE)],
                extra_cuda_cflags=_cuda_flags(arch, nd),
                extra_ldflags=["-lcuda"],
                build_directory=str(bd),
                verbose=(int(os.environ.get("LOCAL_RANK", "0")) == 0))


def import_static(nd):
    name = _module_name(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    so   = next(bd.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dynamic(arch, nd):
    name = _dynamic_module_name(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    bd.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(bd)
    return load(name=name, sources=[str(DYNAMIC_SRC)],
                extra_include_paths=[str(INCLUDE)],
                extra_cuda_cflags=_cuda_flags(arch, nd, ["-DTK_PARALLEL_TENSOR_BINDINGS_EXTERNAL"]),
                extra_ldflags=["-lcuda"],
                build_directory=str(bd),
                verbose=(int(os.environ.get("LOCAL_RANK", "0")) == 0))


def import_dynamic(nd):
    name = _dynamic_module_name(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    so   = next(bd.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_adaptive(arch):
    name = _adaptive_module_name()
    bd   = THIS_DIR / ".torch_ext_build" / name
    bd.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(bd)
    return load(name=name, sources=[str(ADAPTIVE_SRC)],
                extra_include_paths=[str(INCLUDE)],
                extra_cuda_cflags=_cuda_flags(arch, 2),
                extra_ldflags=["-lcuda"],
                build_directory=str(bd),
                verbose=(int(os.environ.get("LOCAL_RANK", "0")) == 0))


def import_adaptive():
    name = _adaptive_module_name()
    bd   = THIS_DIR / ".torch_ext_build" / name
    so   = next(bd.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class TokenConfig:
    seq_total: int
    num_tokens: int           # total local tokens before dispatch
    tokens_per_expert: int    # expected average tokens per expert (for reference only in multinomial mode)


@dataclass
class Routing:
    """Pre-built routing tables for one (shape, rank) combination."""
    padded_ppe: torch.Tensor         # (NUM_EXPERTS,) int32  — padded token count per expert
    pull_idx: torch.Tensor           # (num_padded_local_tokens, 2) int32
    local_expert_padded_counts: list[int]
    num_padded_local_tokens: int


def build_routing_uniform(tc: TokenConfig, local_rank: int, world_size: int, device: str) -> Routing:
    """Synthetic uniform routing: every expert gets exactly tokens_per_expert tokens."""
    num_experts_per_dev = NUM_EXPERTS // world_size
    num_padded_local_tokens = tc.tokens_per_expert * num_experts_per_dev

    padded_ppe = torch.full((NUM_EXPERTS,), tc.tokens_per_expert, device=device, dtype=torch.int32)

    pull_idx = torch.stack([
        torch.arange(num_padded_local_tokens, dtype=torch.int32, device=device) % world_size,
        (torch.arange(num_padded_local_tokens, dtype=torch.int32, device=device) // world_size) % tc.num_tokens,
    ], dim=1).contiguous()

    local_expert_padded_counts = [tc.tokens_per_expert] * num_experts_per_dev

    return Routing(
        padded_ppe=padded_ppe,
        pull_idx=pull_idx,
        local_expert_padded_counts=local_expert_padded_counts,
        num_padded_local_tokens=num_padded_local_tokens,
    )


def build_routing_multinomial(tc: TokenConfig, local_rank: int, world_size: int, device: str) -> Routing:
    """Realistic multinomial routing: tokens are randomly assigned to experts via
    top-K sampling, with counts broadcast from rank 0 so all ranks agree."""
    num_experts_per_dev = NUM_EXPERTS // world_size
    num_init_tokens = tc.num_tokens   # local tokens on this device
    expert_start = num_experts_per_dev * local_rank
    expert_end   = num_experts_per_dev * (local_rank + 1)

    if local_rank == 0:
        routing_weights = torch.rand(NUM_EXPERTS, device=device, dtype=torch.float32)
        chosen_experts = torch.multinomial(
            routing_weights.repeat(tc.seq_total, 1), TOP_K, replacement=False
        ).to(torch.int32)
        tokens_per_expert = torch.bincount(
            chosen_experts.view(-1), minlength=NUM_EXPERTS
        ).to(torch.int32)
    else:
        chosen_experts = torch.empty(tc.seq_total, TOP_K, device=device, dtype=torch.int32)
        tokens_per_expert = torch.empty(NUM_EXPERTS, device=device, dtype=torch.int32)
    dist.broadcast(chosen_experts, 0)
    dist.broadcast(tokens_per_expert, 0)

    padded_ppe = ((tokens_per_expert + 127) // 128) * 128

    num_padded_local_tokens = int(padded_ppe[expert_start:expert_end].sum().item())

    token_index_per_expert = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(padded_ppe[expert_start:expert_end - 1], dim=0, dtype=torch.int32),
    ], dim=0)

    pull_idx = torch.full((num_padded_local_tokens, 2), -1, dtype=torch.int32, device=device)
    chosen_experts_cpu = chosen_experts.cpu()
    for i in range(world_size):
        src_dev = (i + local_rank) % world_size
        for src_tok in range(num_init_tokens):
            for exp_idx in chosen_experts_cpu[src_dev * num_init_tokens + src_tok]:
                exp_idx_int = int(exp_idx.item())
                if expert_start <= exp_idx_int < expert_end:
                    local_exp = exp_idx_int % num_experts_per_dev
                    slot = int(token_index_per_expert[local_exp].item())
                    if slot < num_padded_local_tokens:
                        pull_idx[slot, 0] = src_dev
                        pull_idx[slot, 1] = src_tok
                        token_index_per_expert[local_exp] += 1

    local_expert_padded_counts = [
        int(padded_ppe[expert_start + e].item())
        for e in range(num_experts_per_dev)
    ]

    return Routing(
        padded_ppe=padded_ppe,
        pull_idx=pull_idx,
        local_expert_padded_counts=local_expert_padded_counts,
        num_padded_local_tokens=num_padded_local_tokens,
    )


def benchmark_ms(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize(); dist.barrier()
    return s.elapsed_time(e) / iters


def gather_stats(local_ms):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, float(local_ms))
    vals = [float(v) for v in gathered]
    return min(vals), max(vals), sum(vals) / len(vals)


def make_tensors(mod, tc: TokenConfig, routing: Routing, local_rank, world_size, nd):
    """Create all tensors needed for one MoE forward pass."""
    num_experts_per_dev = NUM_EXPERTS // nd
    num_padded_local_tokens = routing.num_padded_local_tokens

    # pre_tokens: (1, 1, num_tokens, H) per device — shape required by the kernel PGL layout
    pre_tokens = mod.TKParallelTensor(
        (1, 1, tc.num_tokens, H),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=world_size,
        multicast=False,
    )
    pre_tokens.data_.normal_()

    post_tokens = torch.zeros(num_padded_local_tokens, H, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(num_experts_per_dev, H, I, device="cuda", dtype=torch.bfloat16)
    outputs = torch.zeros(num_padded_local_tokens, I, device="cuda", dtype=torch.bfloat16)

    # barrier: per-row-block dispatch progress tracker
    num_row_blocks = (num_padded_local_tokens + 127) // 128
    barrier = mod.TKParallelTensor(
        (nd, num_row_blocks, 1),
        dtype=torch.int32,
        local_rank=local_rank,
        local_world_size=world_size,
        multicast=False,
    )
    barrier.data_.zero_()

    return pre_tokens, post_tokens, weights, outputs, routing.padded_ppe, routing.pull_idx, barrier, num_padded_local_tokens


@torch.no_grad()
def torch_dispatch_moe_gemm(
    inputs_local: torch.Tensor,
    inputs_full: torch.Tensor,
    post_tokens: torch.Tensor,
    weights: torch.Tensor,
    outputs: torch.Tensor,
    dispatch_src_linear: torch.Tensor,
    dispatch_dst_linear: torch.Tensor,
    local_expert_padded_counts: list[int],
) -> None:
    dist.all_gather_into_tensor(inputs_full, inputs_local)
    post_tokens.zero_()
    post_tokens.index_copy_(
        0,
        dispatch_dst_linear,
        inputs_full.view(-1, inputs_local.shape[-1]).index_select(0, dispatch_src_linear),
    )

    tokens_start = 0
    for local_expert_idx, token_count in enumerate(local_expert_padded_counts):
        tokens_end = tokens_start + token_count
        if token_count > 0:
            torch.matmul(
                post_tokens[tokens_start:tokens_end],
                weights[local_expert_idx],
                out=outputs[tokens_start:tokens_end],
            )
        tokens_start = tokens_end


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape-family",
        choices=["thunderkittens", "explicit"],
        default="thunderkittens",
        help="Use the upstream MoE sequence sweep or explicit local-token settings.",
    )
    p.add_argument("--batch",               type=int, default=1)
    p.add_argument("--seq-totals",          type=str, default="8192,16384,32768,65536,131072")
    p.add_argument("--num-tokens",          type=str, default="256,512,1024")
    p.add_argument("--tokens-per-expert",   type=str, default="4,8")
    p.add_argument("--num-comm-sms",        type=str, default="28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53")
    p.add_argument("--adaptive-num-comm-sms", type=int, default=16)
    p.add_argument("--adaptive-wait-cycles",  type=int, default=4000)
    p.add_argument("--adaptive-default-assist-blocks", type=int, default=1)
    p.add_argument("--adaptive-max-assist-blocks", type=int, default=8)
    p.add_argument("--routing",
                   choices=["multinomial", "uniform"],
                   default="multinomial",
                   help="Token routing: 'multinomial' (realistic random, default) or 'uniform' (synthetic round-robin).")
    p.add_argument("--variants",            type=str, default="static,dynamic")
    p.add_argument("--arch",                type=str, default="sm_90a")
    p.add_argument("--warmup",              type=int, default=3)
    p.add_argument("--iters",               type=int, default=10)
    p.add_argument("--output",              type=str,
                   default=str(DEFAULT_OUTPUT))
    p.add_argument("--summary-output",      type=str,
                   default=str(DEFAULT_SUMMARY_OUTPUT))
    return p.parse_args()


def parse_csv_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_policies(spec):
    out = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok: continue
        a, b = [int(x) for x in tok.split(",")]
        out.append((a, b))
    return out


def derive_thunderkittens_token_configs(seq_totals: str, batch: int, world_size: int) -> list[TokenConfig]:
    configs: list[TokenConfig] = []
    for seq_total in parse_csv_ints(seq_totals):
        total_tokens = batch * seq_total
        if total_tokens % world_size != 0:
            raise ValueError(
                f"batch*seq_total must be divisible by world_size: "
                f"batch={batch}, seq_total={seq_total}, world_size={world_size}"
            )
        dispatched_tokens = total_tokens * TOP_K
        if dispatched_tokens % NUM_EXPERTS != 0:
            raise ValueError(
                f"batch*seq_total*TOP_K must be divisible by NUM_EXPERTS: "
                f"batch={batch}, seq_total={seq_total}, top_k={TOP_K}, num_experts={NUM_EXPERTS}"
            )
        configs.append(
            TokenConfig(
                seq_total=seq_total,
                num_tokens=total_tokens // world_size,
                tokens_per_expert=dispatched_tokens // NUM_EXPERTS,
            )
        )
    return configs


def resolve_output_paths(args) -> tuple[Path, Path]:
    variants = {t.strip() for t in args.variants.split(",") if t.strip()}
    output = Path(args.output)
    summary = Path(args.summary_output)
    if (
        output == DEFAULT_OUTPUT
        and summary == DEFAULT_SUMMARY_OUTPUT
        and "dynamic_sm_allocation" in variants
        and "dynamic" not in variants
        and "adaptive" not in variants
    ):
        return DEFAULT_ADAPTIVE_OUTPUT, DEFAULT_ADAPTIVE_SUMMARY_OUTPUT
    return output, summary


def main():
    args = parse_args()
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    variants  = {t.strip() for t in args.variants.split(",") if t.strip()}
    valid_variants = {"static", "dynamic", "adaptive", "dynamic_sm_allocation"}
    if not variants or not variants.issubset(valid_variants):
        raise ValueError(f"invalid --variants: {args.variants}")
    if "dynamic_sm_allocation" in variants and world_size != 2:
        raise ValueError("dynamic_sm_allocation currently requires torchrun with exactly 2 GPUs")
    output_path, summary_path = resolve_output_paths(args)
    if args.shape_family == "thunderkittens":
        token_configs = derive_thunderkittens_token_configs(args.seq_totals, args.batch, world_size)
    else:
        num_tokens_list = parse_csv_ints(args.num_tokens)
        ppe_list = parse_csv_ints(args.tokens_per_expert)
        token_configs = [
            TokenConfig(
                seq_total=num_tokens * world_size,
                num_tokens=num_tokens,
                tokens_per_expert=tokens_per_expert,
            )
            for num_tokens in num_tokens_list
            for tokens_per_expert in ppe_list
        ]
    num_comm_sms_vals = [int(x) for x in args.num_comm_sms.split(",") if x.strip()]
    try:
        if rank == 0 and "static" in variants:
            t0 = time.perf_counter()
            load_static(args.arch, world_size)
            print(f"[rank0] static moe built in {time.perf_counter()-t0:.1f}s", flush=True)
        if rank == 0 and (("dynamic" in variants) or ("adaptive" in variants)):
            t0 = time.perf_counter()
            load_dynamic(args.arch, world_size)
            print(f"[rank0] dynamic moe built in {time.perf_counter()-t0:.1f}s", flush=True)
        if rank == 0 and "dynamic_sm_allocation" in variants:
            t0 = time.perf_counter()
            load_adaptive(args.arch)
            print(f"[rank0] adaptive-tile moe built in {time.perf_counter()-t0:.1f}s", flush=True)
        dist.barrier()

        static_mod  = import_static(world_size) if "static"  in variants else None
        need_dynamic = ("dynamic" in variants) or ("adaptive" in variants)
        dynamic_mod = import_dynamic(world_size) if need_dynamic else None
        adaptive_mod = import_adaptive() if "dynamic_sm_allocation" in variants else None

        # The dynamic extension is built with external TKParallelTensor bindings,
        # so we still need a module that exports those bindings for tensor setup.
        tensor_mod = static_mod
        if tensor_mod is None and adaptive_mod is not None:
            tensor_mod = adaptive_mod
        if tensor_mod is None and (dynamic_mod is not None or "adaptive" in variants):
            if rank == 0:
                print("[rank0] loading static bindings for dynamic-only run", flush=True)
                t0 = time.perf_counter()
                load_static(args.arch, world_size)
                print(f"[rank0] static bindings built in {time.perf_counter()-t0:.1f}s", flush=True)
            dist.barrier()
            tensor_mod = import_static(world_size)

        rows = []

        for tc in token_configs:
                device = f"cuda:{local_rank}"
                if args.routing == "multinomial":
                    if rank == 0:
                        print(f"[routing] building multinomial routing for seq_total={tc.seq_total}", flush=True)
                    routing = build_routing_multinomial(tc, local_rank, world_size, device)
                else:
                    routing = build_routing_uniform(tc, local_rank, world_size, device)

                if rank == 0:
                    avg_ppe = float(routing.padded_ppe.float().mean().item())
                    print(
                        f"[routing] seq_total={tc.seq_total} routing={args.routing} "
                        f"num_padded_local_tokens={routing.num_padded_local_tokens} "
                        f"avg_tokens_per_expert={avg_ppe:.1f}",
                        flush=True,
                    )

                pre_tokens_ref, post_tokens_ref, weights_ref, outputs_ref, padded_ppe_ref, pull_idx_ref, _, _ = \
                    make_tensors(tensor_mod, tc, routing, local_rank, world_size, world_size)
                inputs_local_ref = pre_tokens_ref.data_.reshape(tc.num_tokens, H)
                inputs_full_ref = torch.empty(
                    (world_size, tc.num_tokens, H),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                local_expert_padded_counts = routing.local_expert_padded_counts
                valid_dispatch_mask = (
                    (pull_idx_ref[:, 0] >= 0) & (pull_idx_ref[:, 1] >= 0)
                )
                dispatch_dst_linear = torch.nonzero(valid_dispatch_mask, as_tuple=False).flatten()
                dispatch_src_linear = (
                    pull_idx_ref[valid_dispatch_mask, 0].to(torch.int64) * tc.num_tokens
                    + pull_idx_ref[valid_dispatch_mask, 1].to(torch.int64)
                )

                def _run_torch():
                    def fn():
                        torch_dispatch_moe_gemm(
                            inputs_local_ref,
                            inputs_full_ref,
                            post_tokens_ref,
                            weights_ref,
                            outputs_ref,
                            dispatch_src_linear,
                            dispatch_dst_linear,
                            local_expert_padded_counts,
                        )
                    return fn

                torch_fn = _run_torch()
                torch_ms = benchmark_ms(torch_fn, args.warmup, args.iters)
                torch_mn, torch_mx, torch_avg = gather_stats(torch_ms)
                if rank == 0:
                    print(
                        {
                            "variant": "torch",
                            "seq_total": tc.seq_total,
                            "num_tokens": tc.num_tokens,
                            "tokens_per_expert": tc.tokens_per_expert,
                            "ms_min": torch_mn,
                            "ms_max": torch_mx,
                            "ms_avg": torch_avg,
                        },
                        flush=True,
                    )

                def _run_static(mod, num_comm_sms):
                    pre_tokens, post_tokens, weights, outputs, padded_ppe, pull_idx, barrier, nplt = \
                        make_tensors(mod, tc, routing, local_rank, world_size, world_size)
                    barrier.data_.zero_()
                    def fn():
                        post_tokens.zero_()
                        outputs.zero_()
                        barrier.data_.zero_()
                        mod.moe_dispatch_gemm(
                            pre_tokens, post_tokens, weights, outputs,
                            padded_ppe, pull_idx, barrier, num_comm_sms, nplt)
                    return fn, pre_tokens, barrier

                def _run_dynamic(mod):
                    pre_tokens, post_tokens, weights, outputs, padded_ppe, pull_idx, barrier, nplt = \
                        make_tensors(tensor_mod, tc, routing, local_rank, world_size, world_size)
                    barrier.data_.zero_()
                    def fn():
                        post_tokens.zero_()
                        outputs.zero_()
                        barrier.data_.zero_()
                        mod.moe_dispatch_gemm_dynamic(
                            pre_tokens, post_tokens, weights, outputs,
                            padded_ppe, pull_idx, barrier, nplt)
                    return fn, pre_tokens, barrier

                def _run_adaptive_dynamic(mod):
                    pre_tokens, post_tokens, weights, outputs, padded_ppe, pull_idx, barrier, nplt = \
                        make_tensors(tensor_mod, tc, routing, local_rank, world_size, world_size)
                    barrier.data_.zero_()
                    def fn():
                        post_tokens.zero_()
                        outputs.zero_()
                        barrier.data_.zero_()
                        mod.moe_dispatch_gemm_adaptive(
                            pre_tokens, post_tokens, weights, outputs,
                            padded_ppe, pull_idx, barrier, nplt)
                    return fn, pre_tokens, barrier

                def _run_adaptive(mod):
                    pre_tokens, post_tokens, weights, outputs, padded_ppe, pull_idx, barrier, nplt = \
                        make_tensors(mod, tc, routing, local_rank, world_size, 2)
                    barrier.data_.zero_()
                    def fn():
                        post_tokens.zero_()
                        outputs.zero_()
                        barrier.data_.zero_()
                        mod.moe_dispatch_gemm_dynamic_sm_allocation(
                            pre_tokens, post_tokens, weights, outputs,
                            padded_ppe, pull_idx, barrier, nplt,
                            args.adaptive_num_comm_sms,
                            args.adaptive_wait_cycles,
                            args.adaptive_default_assist_blocks,
                            args.adaptive_max_assist_blocks)
                    return fn, pre_tokens, barrier

                _base_row = dict(
                    seq_total=tc.seq_total,
                    num_tokens=tc.num_tokens,
                    tokens_per_expert=tc.tokens_per_expert,
                    num_padded_local_tokens=routing.num_padded_local_tokens,
                    routing=args.routing,
                    torch_ms_min=torch_mn, torch_ms_max=torch_mx, torch_ms_avg=torch_avg,
                )

                if "static" in variants and static_mod is not None:
                    for ncs in num_comm_sms_vals:
                        fn, *_ = _run_static(static_mod, ncs)
                        ms = benchmark_ms(fn, args.warmup, args.iters)
                        mn, mx, avg = gather_stats(ms)
                        row = dict(**_base_row, variant="static",
                                   num_comm_sms=ncs, comp_only_ctas="", reserved_comm_ctas="",
                                   ms_min=mn, ms_max=mx, ms_avg=avg)
                        rows.append(row)
                        if rank == 0: print(row, flush=True)

                if "dynamic" in variants and dynamic_mod is not None:
                    fn, *_ = _run_dynamic(dynamic_mod)
                    ms = benchmark_ms(fn, args.warmup, args.iters)
                    mn, mx, avg = gather_stats(ms)
                    row = dict(**_base_row, variant="dynamic",
                               num_comm_sms="", comp_only_ctas="auto", reserved_comm_ctas="auto",
                               ms_min=mn, ms_max=mx, ms_avg=avg)
                    rows.append(row)
                    if rank == 0: print(row, flush=True)

                if "adaptive" in variants and dynamic_mod is not None:
                    fn, *_ = _run_adaptive_dynamic(dynamic_mod)
                    ms = benchmark_ms(fn, args.warmup, args.iters)
                    mn, mx, avg = gather_stats(ms)
                    row = dict(**_base_row, variant="adaptive",
                               num_comm_sms="", comp_only_ctas="auto", reserved_comm_ctas="auto",
                               adaptive_wait_cycles="", adaptive_default_assist_blocks="",
                               adaptive_max_assist_blocks="", ms_min=mn, ms_max=mx, ms_avg=avg)
                    rows.append(row)
                    if rank == 0: print(row, flush=True)

                if "dynamic_sm_allocation" in variants and adaptive_mod is not None:
                    fn, *_ = _run_adaptive(adaptive_mod)
                    ms = benchmark_ms(fn, args.warmup, args.iters)
                    mn, mx, avg = gather_stats(ms)
                    row = dict(**_base_row, variant="dynamic_sm_allocation",
                               num_comm_sms=args.adaptive_num_comm_sms,
                               comp_only_ctas="", reserved_comm_ctas="",
                               adaptive_wait_cycles=args.adaptive_wait_cycles,
                               adaptive_default_assist_blocks=args.adaptive_default_assist_blocks,
                               adaptive_max_assist_blocks=args.adaptive_max_assist_blocks,
                               ms_min=mn, ms_max=mx, ms_avg=avg)
                    rows.append(row)
                    if rank == 0: print(row, flush=True)

        if rank == 0:
            out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            fields = ["variant","seq_total","num_tokens","tokens_per_expert",
                      "num_padded_local_tokens","routing","num_comm_sms",
                      "comp_only_ctas","reserved_comm_ctas",
                      "adaptive_wait_cycles","adaptive_default_assist_blocks",
                      "adaptive_max_assist_blocks","ms_min","ms_max","ms_avg",
                      "torch_ms_min","torch_ms_max","torch_ms_avg"]
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)

            # Summary: best static vs best dynamic/adaptive per shape.
            from collections import defaultdict
            by_shape = defaultdict(list)
            for r in rows:
                by_shape[(r["seq_total"], r["num_tokens"], r["tokens_per_expert"], r["routing"])].append(r)
            summary_rows = []
            for key, shape_rows in by_shape.items():
                static_rs  = [r for r in shape_rows if r["variant"] == "static"]
                dynamic_rs = [r for r in shape_rows if r["variant"] == "dynamic"]
                adaptive_rs = [r for r in shape_rows if r["variant"] in {"adaptive", "dynamic_sm_allocation"}]
                if not static_rs:
                    continue
                best_s = min(static_rs, key=lambda r: float(r["ms_max"]))
                best_d = min(dynamic_rs, key=lambda r: float(r["ms_max"])) if dynamic_rs else None
                best_a = min(adaptive_rs, key=lambda r: float(r["ms_max"])) if adaptive_rs else None
                summary = {
                    "seq_total":          key[0],
                    "num_tokens":         key[1],
                    "tokens_per_expert":  key[2],
                    "routing":            key[3],
                    "best_static_num_comm_sms":    best_s["num_comm_sms"],
                    "best_static_ms":              float(best_s["ms_max"]),
                    "torch_ms":                    float(best_s["torch_ms_max"]),
                    "best_dynamic_comp_only":      "",
                    "best_dynamic_reserved_comm":  "",
                    "best_dynamic_ms":             "",
                    "dynamic_over_static":         "",
                    "best_adaptive_num_comm_sms":  "",
                    "best_adaptive_wait_cycles":   "",
                    "best_adaptive_default_assist_blocks": "",
                    "best_adaptive_max_assist_blocks": "",
                    "best_adaptive_ms":            "",
                    "adaptive_over_static":        "",
                }
                if best_d is not None:
                    summary["best_dynamic_comp_only"] = best_d["comp_only_ctas"]
                    summary["best_dynamic_reserved_comm"] = best_d["reserved_comm_ctas"]
                    summary["best_dynamic_ms"] = float(best_d["ms_max"])
                    summary["dynamic_over_static"] = float(best_d["ms_max"]) / float(best_s["ms_max"])
                if best_a is not None:
                    summary["best_adaptive_num_comm_sms"] = best_a["num_comm_sms"]
                    summary["best_adaptive_wait_cycles"] = best_a["adaptive_wait_cycles"]
                    summary["best_adaptive_default_assist_blocks"] = best_a["adaptive_default_assist_blocks"]
                    summary["best_adaptive_max_assist_blocks"] = best_a["adaptive_max_assist_blocks"]
                    summary["best_adaptive_ms"] = float(best_a["ms_max"])
                    summary["adaptive_over_static"] = float(best_a["ms_max"]) / float(best_s["ms_max"])
                summary_rows.append(summary)
                print({"summary": summary_rows[-1]}, flush=True)

            summ = summary_path
            summ.parent.mkdir(parents=True, exist_ok=True)
            sfields = ["seq_total","num_tokens","tokens_per_expert","routing","best_static_num_comm_sms",
                       "best_static_ms","best_dynamic_comp_only","best_dynamic_reserved_comm",
                       "best_dynamic_ms","best_adaptive_num_comm_sms","best_adaptive_wait_cycles",
                       "best_adaptive_default_assist_blocks","best_adaptive_max_assist_blocks",
                       "best_adaptive_ms","torch_ms","dynamic_over_static","adaptive_over_static"]
            with summ.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sfields)
                w.writeheader(); w.writerows(summary_rows)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
