#!/usr/bin/env python3
"""Sweep static num_comm_sms and dynamic comp_only/reserved_comm for Ring Attention.

Ring attention passes KV tiles to the next device in the ring while computing
local attention.  The static version fixes how many SMs do the KV transfer;
the dynamic version uses global task queues so interleaving CTAs can do either.

Run (4 GPUs, GPUs 4-7):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29611 \\
    dynamic_sm_allocation/ring_attn/benchmark_ring_attn_dynamic_sm.py \\
    --arch sm_90a
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
STATIC_SRC  = THIS_DIR / "ring_attn.cu"
DYNAMIC_SRC = THIS_DIR / "dynamic_ring_attn.cu"
INCLUDE     = REPO_ROOT / "include"


def _nd(explicit=None):
    if explicit is not None: return int(explicit)
    return int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "8")))


def _sname(nd):  return f"osgc_at_ring_static_d{nd}"
def _dname(nd):
    suffixes: list[str] = []
    if os.environ.get("RING_ATTN_PRINT_SCHED_STATS") == "1":
        suffixes.append("sched")
    forced_target = os.environ.get("FIXED_DYNAMIC_TARGET")
    if forced_target:
        suffixes.append(f"force{forced_target}")
    debug_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    return f"osgc_at_ring_dynamic_d{nd}{debug_suffix}"


def _flags(arch, nd, extra=None):
    flags = ["-O3", "-std=c++20", "--use_fast_math",
             "-DKITTENS_HOPPER", "--extended-lambda", "--expt-relaxed-constexpr",
             f"-DTK_NUM_DEVICES={nd}"]
    if os.environ.get("RING_ATTN_PRINT_SCHED_STATS") == "1":
        flags.append("-DRING_ATTN_PRINT_SCHED_STATS")
    forced_target = os.environ.get("FIXED_DYNAMIC_TARGET")
    if forced_target:
        flags.append(f"-DFIXED_DYNAMIC_TARGET={int(forced_target)}")
    if arch and arch.lower() == "sm_90a":
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
    elif arch:
        flags.append(f"-arch={arch}")
    if extra: flags += extra
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
    name = _sname(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    bd.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(bd)
    return load(name=name, sources=[str(STATIC_SRC)],
                extra_include_paths=[str(INCLUDE)],
                extra_cuda_cflags=_flags(arch, nd),
                extra_ldflags=["-lcuda"], build_directory=str(bd),
                verbose=(int(os.environ.get("LOCAL_RANK","0"))==0))


def import_static(nd):
    name = _sname(nd); bd = THIS_DIR / ".torch_ext_build" / name
    so = next(bd.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_dynamic(arch, nd):
    name = _dname(nd)
    bd   = THIS_DIR / ".torch_ext_build" / name
    bd.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(bd)
    return load(name=name, sources=[str(DYNAMIC_SRC)],
                extra_include_paths=[str(INCLUDE)],
                extra_cuda_cflags=_flags(arch, nd, ["-DTK_PARALLEL_TENSOR_BINDINGS_EXTERNAL"]),
                extra_ldflags=["-lcuda"], build_directory=str(bd),
                verbose=(int(os.environ.get("LOCAL_RANK","0"))==0))


def import_dynamic(nd):
    name = _dname(nd); bd = THIS_DIR / ".torch_ext_build" / name
    so = next(bd.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@dataclass(frozen=True)
class Shape:
    batch: int; heads: int; seq_per_dev: int; D: int = 128


def benchmark_ms(fn, warmup, iters):
    # Align ranks before the first launch so barrier-tensor zeroing cannot race
    # with a peer that has already entered the device-side ring barrier.
    torch.cuda.synchronize(); dist.barrier()
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


def make_attn_tensors(mod, sh: Shape, local_rank, world_size):
    """Allocate all tensors needed for one ring-attn stage."""
    B, H, S, D = sh.batch, sh.heads, sh.seq_per_dev, sh.D
    # K, V: two double-buffered KV stores (K0/K1, V0/V1)
    def pgl(shape): return mod.TKParallelTensor(
        shape, dtype=torch.bfloat16,
        local_rank=local_rank, local_world_size=world_size, multicast=False)

    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    K0 = pgl((B, H, S, D)); K1 = pgl((B, H, S, D))
    V0 = pgl((B, H, S, D)); V1 = pgl((B, H, S, D))
    # L, O: per-block partial and final
    L_block = torch.zeros(B, H, S, device="cuda", dtype=torch.float32)
    L       = torch.zeros(B, H, S, device="cuda", dtype=torch.float32)
    O_block = torch.zeros(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    O       = torch.zeros(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    barrier = mod.TKParallelTensor(
        (2, 1024, 1024), dtype=torch.int32,
        local_rank=local_rank, local_world_size=world_size, multicast=True)

    K0.data_.normal_(); V0.data_.normal_()
    K1.data_.zero_();   V1.data_.zero_()
    barrier.data_.zero_()
    return Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape-family",
        choices=["thunderkittens", "explicit"],
        default="thunderkittens",
        help="Use the upstream Ring Attention sequence sweep or explicit shapes.",
    )
    p.add_argument("--seq-totals",       type=str, default="6144,12288,24576,49152,98304,196608")
    p.add_argument("--shapes",           type=str,
                   default="16,16,1536,128;16,16,3072,128")
    p.add_argument("--ring-stages",      type=int, default=0,
                   help="Number of ring stages to simulate per timing run; 0 means world_size")
    p.add_argument("--num-comm-sms",     type=str, default="2,4,8,16,32,64")
    p.add_argument("--variants",         type=str, default="static,dynamic")
    p.add_argument("--arch",             type=str, default="sm_90a")
    p.add_argument("--warmup",           type=int, default=3)
    p.add_argument("--iters",            type=int, default=10)
    p.add_argument(
        "--debug-skip-device-barrier",
        action="store_true",
        help="Debug option: use the static kernel entrypoint that skips the final cross-device barrier kernel.",
    )
    p.add_argument("--output",           type=str,
                   default=str(THIS_DIR / "results" / "ring_attn_dynamic_sm.csv"))
    p.add_argument("--summary-output",   type=str,
                   default=str(THIS_DIR / "results" / "ring_attn_dynamic_sm_summary.csv"))
    p.add_argument(
        "--sequential-variants",
        action="store_true",
        help="Run static and dynamic as separate torchrun jobs, then merge outputs.",
    )
    p.add_argument(
        "--sequential-nproc-per-node",
        type=int,
        default=0,
        help="torchrun --nproc_per_node used by --sequential-variants; 0 infers from CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--sequential-master-port",
        type=int,
        default=29730,
        help="Base master port used by --sequential-variants subprocess runs.",
    )
    return p.parse_args()


def parse_shapes(spec):
    shapes = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok: continue
        parts = [int(x) for x in tok.split(",")]
        shapes.append(Shape(*parts))
    return shapes


def parse_csv_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def derive_thunderkittens_shapes(seq_totals: str, world_size: int) -> list[Shape]:
    shapes = []
    for seq_total in parse_csv_ints(seq_totals):
        if seq_total % world_size != 0:
            raise ValueError(
                f"seq_total must be divisible by world_size: seq_total={seq_total}, world_size={world_size}"
            )
        shapes.append(Shape(batch=16, heads=16, seq_per_dev=seq_total // world_size, D=128))
    return shapes


def parse_policies(spec):
    out = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok: continue
        a, b = [int(x) for x in tok.split(",")]
        out.append((a, b))
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_outputs(rows: list[dict[str, str]], output_path: Path, summary_output_path: Path, rank: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant","batch","heads","seq","seq_total","head_dim","num_comm_sms",
              "comp_only","reserved_comm","ms_min","ms_max","ms_avg"]
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    from collections import defaultdict
    by_shape = defaultdict(list)
    for r in rows:
        by_shape[(r["batch"], r["heads"], r["seq"], r["seq_total"], r["head_dim"])].append(r)
    summary_rows = []
    for key, sr in by_shape.items():
        static_rs = [r for r in sr if r["variant"] == "static"]
        dynamic_rs = [r for r in sr if r["variant"] == "dynamic"]
        if not static_rs or not dynamic_rs:
            continue
        bs = min(static_rs, key=lambda r: float(r["ms_max"]))
        bd = min(dynamic_rs, key=lambda r: float(r["ms_max"]))
        summary_rows.append({
            "batch": key[0], "heads": key[1], "seq": key[2], "seq_total": key[3], "head_dim": key[4],
            "best_static_num_comm_sms": bs["num_comm_sms"],
            "best_static_ms": float(bs["ms_max"]),
            "best_dynamic_comp_only": bd["comp_only"],
            "best_dynamic_reserved": bd["reserved_comm"],
            "best_dynamic_ms": float(bd["ms_max"]),
            "dynamic_over_static": float(bd["ms_max"]) / float(bs["ms_max"]),
        })
        if rank in (None, 0):
            print({"summary": summary_rows[-1]}, flush=True)

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    sfields = ["batch","heads","seq","seq_total","head_dim","best_static_num_comm_sms","best_static_ms",
               "best_dynamic_comp_only","best_dynamic_reserved","best_dynamic_ms",
               "dynamic_over_static"]
    with summary_output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)


def _infer_nproc_per_node(explicit: int) -> int:
    if explicit > 0:
        return explicit
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        return len([x for x in cuda_visible.split(",") if x.strip()])
    return torch.cuda.device_count()


def maybe_run_sequential_variants(args: argparse.Namespace) -> bool:
    if not args.sequential_variants or "RANK" in os.environ:
        return False

    variants = [t.strip() for t in args.variants.split(",") if t.strip()]
    if len(variants) <= 1:
        return False
    if set(variants) != {"static", "dynamic"}:
        raise ValueError("--sequential-variants currently supports exactly 'static,dynamic'")

    nproc_per_node = _infer_nproc_per_node(args.sequential_nproc_per_node)
    if nproc_per_node <= 0:
        raise ValueError("failed to infer nproc_per_node for --sequential-variants")

    base_argv: list[str] = []
    skip_next = False
    for idx, tok in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if tok == "--sequential-variants":
            continue
        if tok in {"--variants", "--output", "--summary-output", "--sequential-nproc-per-node", "--sequential-master-port"}:
            skip_next = True
            continue
        if tok.startswith("--variants=") or tok.startswith("--output=") or tok.startswith("--summary-output="):
            continue
        if tok.startswith("--sequential-nproc-per-node=") or tok.startswith("--sequential-master-port="):
            continue
        base_argv.append(tok)

    with tempfile.TemporaryDirectory(prefix="q4_seq_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        variant_rows: list[dict[str, str]] = []
        for offset, variant in enumerate(("static", "dynamic")):
            raw_out = tmpdir_path / f"{variant}.csv"
            summary_out = tmpdir_path / f"{variant}_summary.csv"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(nproc_per_node),
                "--master_port",
                str(args.sequential_master_port + offset),
                str(Path(__file__).resolve()),
                *base_argv,
                "--variants",
                variant,
                "--output",
                str(raw_out),
                "--summary-output",
                str(summary_out),
            ]
            print(f"[sequential] running variant={variant}", flush=True)
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
            if proc.returncode != 0:
                raise SystemExit(f"sequential variant failed: {variant} (code={proc.returncode})")
            variant_rows.extend(read_csv_rows(raw_out))

        write_outputs(variant_rows, Path(args.output), Path(args.summary_output), rank=None)
        print(f"[sequential] wrote merged outputs to {args.output}", flush=True)
    return True


def main():
    args = parse_args()
    if maybe_run_sequential_variants(args):
        return
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    variants        = {t.strip() for t in args.variants.split(",") if t.strip()}
    if args.shape_family == "thunderkittens":
        shapes = derive_thunderkittens_shapes(args.seq_totals, world_size)
    else:
        shapes = parse_shapes(args.shapes)
    ncs_vals        = [int(x) for x in args.num_comm_sms.split(",") if x.strip()]
    ring_stages = args.ring_stages if args.ring_stages > 0 else world_size

    # num_comm_sms must be even for ring_attn (K and V pairs)
    ncs_vals = [x if x % 2 == 0 else x + 1 for x in ncs_vals]

    try:
        if rank == 0 and "static" in variants:
            t0 = time.perf_counter()
            load_static(args.arch, world_size)
            print(f"[rank0] static ring built in {time.perf_counter()-t0:.1f}s", flush=True)
        if rank == 0 and "dynamic" in variants:
            t0 = time.perf_counter()
            load_dynamic(args.arch, world_size)
            print(f"[rank0] dynamic ring built in {time.perf_counter()-t0:.1f}s", flush=True)
        dist.barrier()

        static_mod  = import_static(world_size)  if "static"  in variants else None
        dynamic_mod = import_dynamic(world_size) if "dynamic" in variants else None
        if args.debug_skip_device_barrier and "dynamic" in variants:
            raise ValueError("--debug-skip-device-barrier currently supports static-only runs")

        # The dynamic extension uses external TKParallelTensor bindings, so for
        # dynamic-only runs we still need the static module available to allocate
        # the parallel tensors consumed by the kernel entrypoint.
        tensor_mod = static_mod
        if tensor_mod is None and dynamic_mod is not None:
            if rank == 0:
                print("[rank0] loading static bindings for dynamic-only run", flush=True)
                t0 = time.perf_counter()
                load_static(args.arch, world_size)
                print(f"[rank0] static bindings built in {time.perf_counter()-t0:.1f}s", flush=True)
            dist.barrier()
            tensor_mod = import_static(world_size)

        rows = []

        for sh in shapes:
            if "static" in variants and static_mod is not None:
                tensors = make_attn_tensors(static_mod, sh, local_rank, world_size)
                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier = tensors
                static_kernel = (
                    static_mod.tk_mha_fwd_d128_no_barrier
                    if args.debug_skip_device_barrier
                    else static_mod.tk_mha_fwd_d128
                )

                for ncs in ncs_vals:
                    def run():
                        K1.data_.zero_(); V1.data_.zero_()
                        L.zero_(); L_block.zero_(); O.zero_(); O_block.zero_()
                        barrier.data_.zero_()
                        for stage in range(ring_stages):
                            static_kernel(
                                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                                stage, ncs)
                    ms = benchmark_ms(run, args.warmup, args.iters)
                    mn, mx, avg = gather_stats(ms)
                    row = dict(variant="static",
                               batch=sh.batch, heads=sh.heads, seq=sh.seq_per_dev,
                               seq_total=sh.seq_per_dev * world_size, head_dim=sh.D,
                               num_comm_sms=ncs, comp_only="", reserved_comm="",
                               ms_min=mn, ms_max=mx, ms_avg=avg)
                    rows.append(row)
                    if rank == 0: print(row, flush=True)

            if "dynamic" in variants and dynamic_mod is not None:
                tensors = make_attn_tensors(tensor_mod, sh, local_rank, world_size)
                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier = tensors

                def run():
                    K1.data_.zero_(); V1.data_.zero_()
                    L.zero_(); L_block.zero_(); O.zero_(); O_block.zero_()
                    barrier.data_.zero_()
                    for stage in range(ring_stages):
                        dynamic_mod.tk_mha_fwd_d128_dynamic(
                            Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                            stage)
                ms = benchmark_ms(run, args.warmup, args.iters)
                mn, mx, avg = gather_stats(ms)
                row = dict(variant="dynamic",
                           batch=sh.batch, heads=sh.heads, seq=sh.seq_per_dev,
                           seq_total=sh.seq_per_dev * world_size, head_dim=sh.D,
                           num_comm_sms="", comp_only="auto",
                           reserved_comm="auto",
                           ms_min=mn, ms_max=mx, ms_avg=avg)
                rows.append(row)
                if rank == 0: print(row, flush=True)

        if rank == 0:
            write_outputs(rows, Path(args.output), Path(args.summary_output), rank=rank)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
