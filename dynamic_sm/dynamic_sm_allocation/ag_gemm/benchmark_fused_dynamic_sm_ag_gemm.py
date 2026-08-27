#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from .build_utils import import_ag_gemm_module, load_ag_gemm_module
except ImportError:
    from build_utils import import_ag_gemm_module, load_ag_gemm_module


THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
ROW_BLOCK = 128
DEFAULT_OUTPUT = RESULTS_DIR / "fused_dynamic_sm_ag_gemm_results.csv"


@dataclass(frozen=True)
class Shape:
    rows_per_rank: int
    k: int
    n: int

    @property
    def global_m(self) -> int:
        return self.rows_per_rank * ROW_BLOCK * dist.get_world_size()


def parse_csv_ints(spec: str) -> list[int]:
    values = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not values:
        raise ValueError(f"empty integer list: {spec}")
    return values


def parse_csv_tokens(spec: str) -> list[str]:
    values = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not values:
        raise ValueError(f"empty token list: {spec}")
    return values


def derive_n_sweep_shapes(base_ns: str, world_size: int) -> list[Shape]:
    out: list[Shape] = []
    for base_n in parse_csv_ints(base_ns):
        if base_n % (world_size * ROW_BLOCK) != 0:
            raise ValueError(
                f"base N must be divisible by world_size*{ROW_BLOCK}: "
                f"N={base_n}, world_size={world_size}"
            )
        k = base_n
        if k % 128 != 0:
            raise ValueError(
                f"derived K must be divisible by 128: K={k} from N={base_n}, "
                f"world_size={world_size}"
            )
        output_cols = base_n // world_size
        if base_n % world_size != 0:
            raise ValueError(
                f"base N must be divisible by world_size: N={base_n}, world_size={world_size}"
            )
        if output_cols % 256 != 0:
            raise ValueError(
                f"derived output cols must be divisible by 256: N={output_cols} from base N={base_n}"
            )
        rows_per_rank = base_n // (world_size * ROW_BLOCK)
        out.append(Shape(rows_per_rank=rows_per_rank, k=k, n=output_cols))
    return out


def parse_shapes(rows_per_rank: str, ks: str, ns: str) -> list[Shape]:
    out: list[Shape] = []
    for r in parse_csv_ints(rows_per_rank):
        for k in parse_csv_ints(ks):
            for n in parse_csv_ints(ns):
                out.append(Shape(rows_per_rank=r, k=k, n=n))
    return out


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
    vals = [float(x) for x in gathered]
    return min(vals), max(vals), sum(vals) / len(vals)


def tflops(m: int, k: int, n: int, ms: float) -> float:
    return (2.0 * m * k * n) / (ms * 1.0e-3) / 1.0e12


def make_inputs(mod, shape: Shape, rank: int, world_size: int):
    global_m = shape.global_m
    local_m = global_m // world_size
    a_local = (
        torch.randn((local_m, shape.k), device="cuda", dtype=torch.bfloat16)
        / (shape.k**0.25)
    )
    a_tk = mod.TKParallelTensor(
        (global_m, shape.k),
        dtype=torch.bfloat16,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    start = rank * local_m
    a_tk.data_[start : start + local_m].copy_(a_local)
    b = torch.randn((shape.k, shape.n), device="cuda", dtype=torch.bfloat16) / (
        shape.k**0.25
    )
    c = torch.zeros((global_m, shape.n), device="cuda", dtype=torch.bfloat16)
    barrier = mod.TKParallelTensor(
        (2, 1024, 1024),
        dtype=torch.int,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    barrier.data_.zero_()
    return a_local, a_tk, b, c, barrier


def choose_adaptive_init_comm_chunk(shape: Shape, max_chunk: int) -> int:
    comm_unit = max(1.0, float(shape.k) / 128.0)
    comp_unit = max(1.0, (float(shape.k) / 64.0) * max(1.0, float(shape.n) / 256.0))
    chunk = int(round(comp_unit / comm_unit))
    return max(1, min(max_chunk, chunk))


def resolve_output_paths(
    output_arg: str, summary_arg: str | None, variants: set[str]
) -> tuple[Path, Path]:
    output_path = Path(output_arg)
    if summary_arg:
        summary_path = Path(summary_arg)
    else:
        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    return output_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AG-GEMM static and dynamic CTA scheduling."
    )
    parser.add_argument(
        "--shape-family",
        choices=["cartesian", "n_sweep_world_size_k"],
        default="n_sweep_world_size_k",
        help="Shape generation mode. 'n_sweep_world_size_k' derives M=N, K=N, output_cols=N/world_size.",
    )
    parser.add_argument(
        "--base-ns",
        type=str,
        default="2048,4096,8192,16384,32768",
        help="Base N values used when --shape-family=n_sweep_world_size_k.",
    )
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--rows-per-rank", type=str, default="1,2,4,8,16")
    parser.add_argument("--ks", type=str, default="512,1024,2048")
    parser.add_argument("--ns", type=str, default="4096,8192,16384")
    parser.add_argument("--static-comm-sms", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--variants",
        type=str,
        default="static,dynamic",
        help="Comma-separated variants to run: static,dynamic",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional summary CSV path. Defaults beside --output.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip compilation; use existing .so (fails if not built yet). Use after first successful run.",
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
            shapes = parse_shapes(args.rows_per_rank, args.ks, args.ns)
        static_sms_values = parse_csv_ints(args.static_comm_sms)
        variants = {tok.strip() for tok in args.variants.split(",") if tok.strip()}
        if not variants or not variants.issubset({"static", "dynamic"}):
            raise ValueError(f"invalid --variants: {args.variants}")
        output_path, summary_path = resolve_output_paths(
            args.output, args.summary_output or None, variants
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        plot_summary_rows: list[dict[str, object]] = []

        if not args.skip_build:
            if rank == 0:
                t0 = time.perf_counter()
                load_ag_gemm_module(args.arch, num_devices=world_size)
                t1 = time.perf_counter()
                print(f"[rank0] built AG-GEMM module in {t1 - t0:.2f}s", flush=True)
            dist.barrier()
        else:
            if rank == 0:
                print("[rank0] --skip-build: using existing .so", flush=True)
            dist.barrier()
        mod = import_ag_gemm_module(num_devices=world_size)
        print(f"[rank{rank}] imported module", flush=True)

        for shape in shapes:
                best_static: tuple[float, int] | None = None
                worst_static: tuple[float, int] | None = None
                best_dynamic: tuple[float, str] | None = None

                if "static" in variants:
                    for static_sms in static_sms_values:
                        _, a_tk, b, c, barrier = make_inputs(mod, shape, rank, world_size)

                        def reset_static() -> None:
                            barrier.data_.zero_()
                            c.zero_()
                            dist.barrier()

                        def run_static() -> None:
                            mod.all_gather_matmul(a_tk, b, c, barrier, int(static_sms))

                        ms = benchmark_ms(
                            run_static, args.warmup, args.iters, reset_fn=reset_static
                        )
                        ms_min, ms_max, ms_avg = gather_stats(ms)
                        row = {
                            "variant": "static",
                            "rows_per_rank": shape.rows_per_rank,
                            "m": shape.global_m,
                            "k": shape.k,
                            "n": shape.n,
                            "static_comm_sms": static_sms,
                            "comp_only_ctas": "",
                            "reserved_comm_ctas": "",
                            "adaptive_init_comm_chunk": "",
                            "adaptive_max_comm_chunk": "",
                            "adaptive_wait_event_threshold": "",
                            "ms_min": ms_min,
                            "ms_max": ms_max,
                            "ms_avg": ms_avg,
                            "tflops_avg": tflops(shape.global_m, shape.k, shape.n, ms_avg),
                        }
                        all_rows.append(row)
                        if best_static is None or ms_max < best_static[0]:
                            best_static = (ms_max, static_sms)
                        if worst_static is None or ms_max > worst_static[0]:
                            worst_static = (ms_max, static_sms)

                if "dynamic" in variants:
                    _, a_tk, b, c, barrier = make_inputs(mod, shape, rank, world_size)

                    def reset_dynamic() -> None:
                        barrier.data_.zero_()
                        c.zero_()
                        dist.barrier()

                    def run_dynamic() -> None:
                        mod.all_gather_matmul_dynamic(a_tk, b, c, barrier)

                    ms = benchmark_ms(
                        run_dynamic,
                        args.warmup,
                        args.iters,
                        reset_fn=reset_dynamic,
                    )
                    ms_min, ms_max, ms_avg = gather_stats(ms)
                    row = {
                        "variant": "dynamic",
                        "rows_per_rank": shape.rows_per_rank,
                        "m": shape.global_m,
                        "k": shape.k,
                        "n": shape.n,
                        "static_comm_sms": "",
                        "comp_only_ctas": "auto",
                        "reserved_comm_ctas": "auto",
                        "adaptive_init_comm_chunk": "",
                        "adaptive_max_comm_chunk": "",
                        "adaptive_wait_event_threshold": "",
                        "ms_min": ms_min,
                        "ms_max": ms_max,
                        "ms_avg": ms_avg,
                        "tflops_avg": tflops(shape.global_m, shape.k, shape.n, ms_avg),
                    }
                    all_rows.append(row)
                    best_dynamic = (ms_max, ("auto", "auto"))

                torch_ms_val: float | None = None
                if "static" in variants or "dynamic" in variants:
                    _, a_tk_torch, b_torch, c_torch, barrier_torch = make_inputs(
                        mod, shape, rank, world_size
                    )
                    a_full_torch = torch.empty(
                        (shape.global_m, shape.k), device="cuda", dtype=torch.bfloat16
                    )

                    def run_torch_baseline() -> None:
                        dist.all_gather_into_tensor(
                            a_full_torch, a_tk_torch.data_[
                                rank * (shape.global_m // world_size):
                                (rank + 1) * (shape.global_m // world_size)
                            ]
                        )
                        torch.matmul(a_full_torch, b_torch, out=c_torch)

                    torch_local = benchmark_ms(run_torch_baseline, args.warmup, args.iters)
                    _, torch_ms_val, _ = gather_stats(torch_local)

                if best_static is not None:
                    shape_n = shape.global_m
                    summary = {
                        "rows_per_rank": shape.rows_per_rank,
                        "m": shape.global_m,
                        "k": shape.k,
                        "n": shape.n,
                        "best_static_ms": best_static[0],
                        "best_static_comm_sms": best_static[1],
                        "best_dynamic_ms": "",
                        "best_dynamic_comp_only_ctas": "",
                        "best_dynamic_reserved_comm_ctas": "",
                        "dynamic_speedup_over_best_static": "",
                    }
                    if best_dynamic is not None:
                        summary["best_dynamic_ms"] = best_dynamic[0]
                        summary["best_dynamic_comp_only_ctas"] = best_dynamic[1][0]
                        summary["best_dynamic_reserved_comm_ctas"] = best_dynamic[1][1]
                        summary["dynamic_speedup_over_best_static"] = (
                            best_static[0] / best_dynamic[0]
                        )
                    summary_rows.append(summary)
                    plot_summary_rows.append({
                        "shape_n": shape_n,
                        "m": shape.global_m,
                        "k": shape.k,
                        "n": shape.n,
                        "best_static_num_comm_sms": best_static[1],
                        "best_static_ms": best_static[0],
                        "worst_static_num_comm_sms": worst_static[1] if worst_static else "",
                        "worst_static_ms": worst_static[0] if worst_static else "",
                        "best_dynamic_comp_only_ctas": best_dynamic[1][0] if best_dynamic else "",
                        "best_dynamic_reserved_comm_ctas": best_dynamic[1][1] if best_dynamic else "",
                        "best_dynamic_ms": best_dynamic[0] if best_dynamic else "",
                        "torch_ms": torch_ms_val if torch_ms_val is not None else "",
                    })
                    if rank == 0:
                        print(summary, flush=True)

        if rank == 0:
            with output_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant",
                        "rows_per_rank",
                        "m",
                        "k",
                        "n",
                        "static_comm_sms",
                        "comp_only_ctas",
                        "reserved_comm_ctas",
                        "adaptive_init_comm_chunk",
                        "adaptive_max_comm_chunk",
                        "adaptive_wait_event_threshold",
                        "ms_min",
                        "ms_max",
                        "ms_avg",
                        "tflops_avg",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_rows)

            with summary_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "rows_per_rank",
                        "m",
                        "k",
                        "n",
                        "best_static_ms",
                        "best_static_comm_sms",
                        "best_dynamic_ms",
                        "best_dynamic_comp_only_ctas",
                        "best_dynamic_reserved_comm_ctas",
                        "dynamic_speedup_over_best_static",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_rows)

            plot_summary_path = RESULTS_DIR / f"q1_{world_size}gpu_plot_summary.csv"
            with plot_summary_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "shape_n",
                        "m",
                        "k",
                        "n",
                        "best_static_num_comm_sms",
                        "best_static_ms",
                        "worst_static_num_comm_sms",
                        "worst_static_ms",
                        "best_dynamic_comp_only_ctas",
                        "best_dynamic_reserved_comm_ctas",
                        "best_dynamic_ms",
                        "torch_ms",
                    ],
                )
                writer.writeheader()
                writer.writerows(plot_summary_rows)
            print(f"[rank0] wrote plot summary to {plot_summary_path}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
