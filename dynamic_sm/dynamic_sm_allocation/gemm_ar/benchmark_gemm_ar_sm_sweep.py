#!/usr/bin/env python3
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


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC = THIS_DIR / "gemm_ar.cu"
DYNAMIC_SRC = THIS_DIR / "dynamic_gemm_ar.cu"
INCLUDE = REPO_ROOT / "include"


def _num_devices(explicit_num_devices: int | None = None) -> int:
    if explicit_num_devices is not None:
        return int(explicit_num_devices)
    return int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "8")))


def _module_name(num_devices: int, num_blocks: int | None = None) -> str:
    suffix = f"_b{num_blocks}" if num_blocks is not None else ""
    return f"osgc_dynamic_sm_allocation_gemm_ar_d{num_devices}{suffix}"


def _dynamic_module_name(num_devices: int, num_blocks: int | None = None) -> str:
    suffixes: list[str] = []
    if num_blocks is not None:
        suffixes.append(f"b{num_blocks}")
    if os.environ.get("FIXED_DYNAMIC_TARGET"):
        suffixes.append(f"fixed{os.environ['FIXED_DYNAMIC_TARGET']}")
    if os.environ.get("GEMM_AR_PRINT_SCHED_STATS") == "1":
        suffixes.append("sched")
    debug_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    return f"osgc_dynamic_sm_allocation_dynamic_gemm_ar_d{num_devices}{debug_suffix}"

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


def load_gemm_ar_module(arch: str | None, num_devices: int | None = None, num_blocks: int | None = None):
    """Build the static GEMM+AR module.

    Args:
        arch: CUDA architecture (e.g. "sm_90a").
        num_devices: Number of devices (defaults to WORLD_SIZE).
        num_blocks: Total CTA count. Defaults to 132 (H100 SM count).
            Pass a smaller value to leave SMs for co-running kernels.
    """
    python_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(":")
    if python_bin not in path_entries:
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"

    tk_num_devices = _num_devices(num_devices)
    extra_cuda_cflags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-DKITTENS_HOPPER",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        f"-DTK_NUM_DEVICES={tk_num_devices}",
    ]
    if num_blocks is not None:
        extra_cuda_cflags.append(f"-DGEMM_AR_NUM_BLOCKS={num_blocks}")
    if arch and arch.lower() == "sm_90a":
        extra_cuda_cflags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    elif arch:
        extra_cuda_cflags.append(f"-arch={arch}")

    name = _module_name(tk_num_devices, num_blocks)
    build_dir = THIS_DIR / ".torch_ext_build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(build_dir)
    return load(
        name=name,
        sources=[str(SRC)],
        extra_include_paths=[str(INCLUDE), str(THIS_DIR)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=(int(os.environ.get("LOCAL_RANK", "0")) == 0),
    )


def import_gemm_ar_module(num_devices: int | None = None, num_blocks: int | None = None):
    tk_num_devices = _num_devices(num_devices)
    name = _module_name(tk_num_devices, num_blocks)
    build_dir = THIS_DIR / ".torch_ext_build" / name
    so_path = next(build_dir.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dynamic_gemm_ar_module(arch: str | None, num_devices: int | None = None, num_blocks: int | None = None):
    """Build the dynamic GEMM+AR module.

    Args:
        arch: CUDA architecture (e.g. "sm_90a").
        num_devices: Number of devices (defaults to WORLD_SIZE).
        num_blocks: Total CTA count. Defaults to 132 (H100 SM count).
            Pass a smaller value to leave SMs for co-running kernels.
    """
    python_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(":")
    if python_bin not in path_entries:
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"

    tk_num_devices = _num_devices(num_devices)
    extra_cuda_cflags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-DKITTENS_HOPPER",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        f"-DTK_NUM_DEVICES={tk_num_devices}",
        "-DTK_PARALLEL_TENSOR_BINDINGS_EXTERNAL",
    ]
    if num_blocks is not None:
        extra_cuda_cflags.append(f"-DGEMM_AR_NUM_BLOCKS={num_blocks}")
    if os.environ.get("FIXED_DYNAMIC_TARGET"):
        extra_cuda_cflags.append(f"-DFIXED_DYNAMIC_TARGET={os.environ['FIXED_DYNAMIC_TARGET']}")
    if os.environ.get("GEMM_AR_PRINT_SCHED_STATS") == "1":
        extra_cuda_cflags.append("-DGEMM_AR_PRINT_SCHED_STATS")
    if arch and arch.lower() == "sm_90a":
        extra_cuda_cflags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    elif arch:
        extra_cuda_cflags.append(f"-arch={arch}")

    name = _dynamic_module_name(tk_num_devices, num_blocks)
    build_dir = THIS_DIR / ".torch_ext_build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(build_dir)
    return load(
        name=name,
        sources=[str(DYNAMIC_SRC)],
        extra_include_paths=[str(INCLUDE), str(THIS_DIR)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=(int(os.environ.get("LOCAL_RANK", "0")) == 0),
    )


def import_dynamic_gemm_ar_module(num_devices: int | None = None, num_blocks: int | None = None):
    tk_num_devices = _num_devices(num_devices)
    name = _dynamic_module_name(tk_num_devices, num_blocks)
    build_dir = THIS_DIR / ".torch_ext_build" / name
    so_path = next(build_dir.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int


def derive_n_sweep_shapes(base_ns: str, world_size: int) -> list[Shape]:
    shapes: list[Shape] = []
    for base_n in parse_csv_ints(base_ns):
        k = base_n // world_size
        if base_n % 128 != 0:
            raise ValueError(f"derived M/N must be divisible by 128/256-friendly base: N={base_n}")
        if k % 64 != 0:
            raise ValueError(
                f"derived K must be divisible by 64: K={k} from N={base_n}, "
                f"world_size={world_size}"
            )
        if base_n % 256 != 0:
            raise ValueError(f"derived N must be divisible by 256: N={base_n}")
        shapes.append(Shape(m=base_n, k=k, n=base_n))
    return shapes


def parse_shapes(spec: str) -> list[Shape]:
    shapes: list[Shape] = []
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        m, k, n = [int(x.strip()) for x in token.split(",")]
        shapes.append(Shape(m=m, k=k, n=n))
    if not shapes:
        raise ValueError(f"no shapes parsed from {spec}")
    return shapes


def parse_csv_ints(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError(f"empty integer list: {spec}")
    return vals


def parse_dynamic_policies(spec: str) -> list[tuple[int, int]]:
    policies: list[tuple[int, int]] = []
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        comp_only, reserved = [int(x.strip()) for x in token.split(",")]
        policies.append((comp_only, reserved))
    if not policies:
        raise ValueError(f"no dynamic policies parsed from {spec}")
    return policies


def benchmark_ms(fn, warmup: int, iters: int, reset_fn=None) -> float:
    for _ in range(warmup):
        if reset_fn is not None:
            reset_fn()
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    for _ in range(iters):
        if reset_fn is not None:
            reset_fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        dist.barrier()
        samples.append(float(start.elapsed_time(end)))
    return sum(samples) / float(len(samples))


def gather_stats(local_ms: float) -> tuple[float, float, float]:
    world_size = dist.get_world_size()
    gathered: list[float | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, float(local_ms))
    vals = [float(v) for v in gathered]
    return min(vals), max(vals), sum(vals) / len(vals)


def tflops(m: int, k: int, n: int, ms: float) -> float:
    return (2.0 * m * k * n) / (ms * 1.0e-3) / 1.0e12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep static num_comm_sms for fused gemm_ar."
    )
    parser.add_argument(
        "--shape-family",
        choices=["explicit", "n_sweep_world_size_k"],
        default="n_sweep_world_size_k",
        help="Shape generation mode. 'n_sweep_world_size_k' derives M=N, K=N/world_size, N=N.",
    )
    parser.add_argument(
        "--base-ns",
        type=str,
        default="2048,4096,8192,16384,32768",
        help="Base N values used when --shape-family=n_sweep_world_size_k.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="1024,512,4096;2048,512,4096;4096,512,4096;8192,512,4096",
    )
    parser.add_argument("--num-comm-sms", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument(
        "--variants",
        type=str,
        default="static,dynamic",
        help="Comma-separated variants to run: static,dynamic",
    )
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        default=str(THIS_DIR / "results" / "gemm_ar_sm_sweep.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(THIS_DIR / "results" / "gemm_ar_sm_sweep_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    try:
        if args.shape_family == "n_sweep_world_size_k":
            shapes = derive_n_sweep_shapes(args.base_ns, world_size)
        else:
            shapes = parse_shapes(args.shapes)
        num_comm_sms_values = parse_csv_ints(args.num_comm_sms)
        variants = {tok.strip() for tok in args.variants.split(",") if tok.strip()}
        if not variants or not variants.issubset({"static", "dynamic"}):
            raise ValueError(f"invalid --variants: {args.variants}")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path = Path(args.summary_output)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        need_static_bindings = bool({"static", "dynamic"} & variants)

        if rank == 0 and need_static_bindings:
            t0 = time.perf_counter()
            load_gemm_ar_module(args.arch, num_devices=world_size)
            t1 = time.perf_counter()
            print(f"[rank0] built gemm_ar extension in {t1 - t0:.2f}s", flush=True)
        if rank == 0 and "dynamic" in variants:
            t0 = time.perf_counter()
            load_dynamic_gemm_ar_module(args.arch, num_devices=world_size)
            t1 = time.perf_counter()
            print(
                f"[rank0] built dynamic_gemm_ar extension in {t1 - t0:.2f}s",
                flush=True,
            )
        dist.barrier()
        static_mod = (
            import_gemm_ar_module(num_devices=world_size) if need_static_bindings else None
        )
        dynamic_mod = (
            import_dynamic_gemm_ar_module(num_devices=world_size)
            if "dynamic" in variants
            else None
        )
        tensor_mod = static_mod if static_mod is not None else dynamic_mod
        if tensor_mod is None or not hasattr(tensor_mod, "TKParallelTensor"):
            raise RuntimeError("failed to load TKParallelTensor bindings")

        rows: list[dict[str, object]] = []
        for shape in shapes:
            if shape.m % 128 != 0 or shape.k % 64 != 0 or shape.n % 256 != 0:
                raise RuntimeError(
                    f"invalid shape {shape}: requires M%128==0, K%64==0, N%256==0"
                )

            a = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16) / (
                shape.k**0.25
            )
            b = torch.randn((shape.k, shape.n), device="cuda", dtype=torch.bfloat16) / (
                shape.k**0.25
            )
            if "static" in variants and static_mod is not None:
                c = static_mod.TKParallelTensor(
                    (shape.m, shape.n),
                    dtype=torch.bfloat16,
                    local_rank=local_rank,
                    local_world_size=world_size,
                    multicast=True,
                )
                barrier = static_mod.TKParallelTensor(
                    (2, 1024, 1024),
                    dtype=torch.int,
                    local_rank=local_rank,
                    local_world_size=world_size,
                    multicast=True,
                )

                for num_comm_sms in num_comm_sms_values:
                    c.data_.zero_()
                    barrier.data_.zero_()

                    def reset_static() -> None:
                        c.data_.zero_()
                        barrier.data_.zero_()
                        dist.barrier()

                    def run_kernel() -> None:
                        static_mod.matmul_all_reduce(a, b, c, barrier, int(num_comm_sms))

                    ms = benchmark_ms(
                        run_kernel, args.warmup, args.iters, reset_fn=reset_static
                    )
                    ms_min, ms_max, ms_avg = gather_stats(ms)
                    row = {
                        "variant": "static",
                        "m": shape.m,
                        "k": shape.k,
                        "n": shape.n,
                        "num_comm_sms": num_comm_sms,
                        "comp_only_ctas": "",
                        "reserved_comm_ctas": "",
                        "ms_min": ms_min,
                        "ms_max": ms_max,
                        "ms_avg": ms_avg,
                        "tflops_avg": tflops(shape.m, shape.k, shape.n, ms_avg),
                    }
                    rows.append(row)
                    if rank == 0:
                        print(row, flush=True)

            if "dynamic" in variants and dynamic_mod is not None:
                c = tensor_mod.TKParallelTensor(
                    (shape.m, shape.n),
                    dtype=torch.bfloat16,
                    local_rank=local_rank,
                    local_world_size=world_size,
                    multicast=True,
                )
                barrier = tensor_mod.TKParallelTensor(
                    (2, 1024, 1024),
                    dtype=torch.int,
                    local_rank=local_rank,
                    local_world_size=world_size,
                    multicast=True,
                )

                c.data_.zero_()
                barrier.data_.zero_()

                def reset_dynamic() -> None:
                    c.data_.zero_()
                    barrier.data_.zero_()
                    dist.barrier()

                def run_kernel() -> None:
                    dynamic_mod.matmul_all_reduce_dynamic(
                        a,
                        b,
                        c,
                        barrier,
                    )

                ms = benchmark_ms(
                    run_kernel, args.warmup, args.iters, reset_fn=reset_dynamic
                )
                ms_min, ms_max, ms_avg = gather_stats(ms)
                row = {
                    "variant": "dynamic",
                    "m": shape.m,
                    "k": shape.k,
                    "n": shape.n,
                    "num_comm_sms": "",
                    "comp_only_ctas": "auto",
                    "reserved_comm_ctas": "auto",
                    "ms_min": ms_min,
                    "ms_max": ms_max,
                    "ms_avg": ms_avg,
                    "tflops_avg": tflops(shape.m, shape.k, shape.n, ms_avg),
                }
                rows.append(row)
                if rank == 0:
                    print(row, flush=True)

        if rank == 0:
            with output_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant",
                        "m",
                        "k",
                        "n",
                        "num_comm_sms",
                        "comp_only_ctas",
                        "reserved_comm_ctas",
                        "ms_min",
                        "ms_max",
                        "ms_avg",
                        "tflops_avg",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            by_shape: dict[tuple[int, int, int], list[dict[str, object]]] = {}
            for row in rows:
                key = (int(row["m"]), int(row["k"]), int(row["n"]))
                by_shape.setdefault(key, []).append(row)

            summary_rows: list[dict[str, object]] = []
            for (m, k, n), shape_rows in by_shape.items():
                static_rows = [row for row in shape_rows if row["variant"] == "static"]
                dynamic_rows = [row for row in shape_rows if row["variant"] == "dynamic"]
                if not static_rows or not dynamic_rows:
                    continue

                best_static = min(static_rows, key=lambda row: float(row["ms_max"]))
                best_dynamic = min(dynamic_rows, key=lambda row: float(row["ms_max"]))
                static_ms = float(best_static["ms_max"])
                dynamic_ms = float(best_dynamic["ms_max"])
                summary = {
                    "m": m,
                    "k": k,
                    "n": n,
                    "best_static_num_comm_sms": int(best_static["num_comm_sms"]),
                    "best_static_ms": static_ms,
                    "best_dynamic_comp_only_ctas": str(best_dynamic["comp_only_ctas"]),
                    "best_dynamic_reserved_comm_ctas": str(best_dynamic["reserved_comm_ctas"]),
                    "best_dynamic_ms": dynamic_ms,
                    "dynamic_over_static_ratio": dynamic_ms / static_ms,
                    "dynamic_minus_static_ms": dynamic_ms - static_ms,
                }
                summary_rows.append(summary)
                print({"summary": summary}, flush=True)

            with summary_output_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "m",
                        "k",
                        "n",
                        "best_static_num_comm_sms",
                        "best_static_ms",
                        "best_dynamic_comp_only_ctas",
                        "best_dynamic_reserved_comm_ctas",
                        "best_dynamic_ms",
                        "dynamic_over_static_ratio",
                        "dynamic_minus_static_ms",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_rows)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
