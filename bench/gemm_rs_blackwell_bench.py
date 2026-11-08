"""Single-node BF16 GEMM + reduce-scatter: mKernel, cuBLAS/NCCL, CUTLASS, and TK.

Build with make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell.
Run with torchrun --standalone --nproc-per-node=8 and set CUTLASS_PATH to
the pinned checkout linked below. K is the per-rank shard of global K=12288.
For TK, set THUNDERKITTENS_PATH to a checkout with gemm_rs's B200 _C built.
"""

import argparse
import itertools
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Protocol
import time

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))

from common import check_close
import load_module
import load_thirdparty


class BlackwellBenchConfig:
    arch = "blackwell"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_chief = local_rank == 0
    num_nodes = 1

    shapes_to_test = [2048, 4096, 8192, 16384, 32768]
    output_dim = 7168
    split_inner_dim = 12288 // world_size

    # Tile columns per rasterization supergroup; must match the instantiated
    # values in gemm_rs_blackwell::launch.
    mkernel_swizzle_sizes = [1, 2]
    # Ready-counter publications per 256x256 tile and RS strip heights; must match
    # the instantiated values in gemm_rs_blackwell::launch.
    mkernel_signals_per_tile = [1, 2, 4]
    mkernel_rs_strip_rows = [8, 16, 32]
    # MMA warps per CTA, keyed by M: 1 halves the cluster tile to 256 rows. It only
    # supports 1 or 2 signals per tile.
    mkernel_mma_warps = {2048: [1, 2], 4096: [1, 2]}
    mkernel_default_mma_warps = [2]

    def mkernel_candidates(self, M: int):
        return [
            (swizzle, signals, rs_rows, mma_warps)
            for swizzle, signals, rs_rows, mma_warps in itertools.product(
                self.mkernel_swizzle_sizes, self.mkernel_signals_per_tile,
                self.mkernel_rs_strip_rows,
                self.mkernel_mma_warps.get(M, self.mkernel_default_mma_warps),
            )
            if mma_warps == 2 or signals <= 2
        ]

    # Keys match the pinned CUTLASS example's CLI / run() keyword arguments:
    # https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/distributed_gemm_reduce_scatter_blackwell.py#L3009
    cutlass_fixed_config = {
        "use_cuda_graph": True,
        "use_2cta_instrs": True,
        "benchmark_or_test": "benchmark",
        # 2-CTA instructions require an even cluster M; the CLI default is 1.
        "cluster_shape_mn": (2, 1),
    }

    # (raster_order, swizzle_size, mma_tiler_mn), keyed by M. Picked from an
    # autotune sweep over raster {m,n} x swizzle {1,2,4,8} x tile {128,256}^2
    # at N=7168, K_local=1536, 8 ranks; raster=m swizzle=1 is within 0.5% of
    # the best everywhere, so it is also the fallback for untuned shapes.
    cutlass_tuned_configs = {
        2048: ("m", 1, (128, 128)),
        4096: ("m", 1, (256, 256)),
        8192: ("m", 1, (256, 256)),
        16384: ("m", 2, (256, 256)),
        32768: ("m", 1, (256, 256)),
    }
    cutlass_default_config = ("m", 1, (256, 256))


class DistBufferLike(Protocol):
    """Common tensor access for dynamically loaded distributed buffers."""
    data_: torch.Tensor


@dataclass
class BlackwellBenchVars:
    baseline_a: torch.Tensor | None = None
    baseline_b: torch.Tensor | None = None
    baseline_c: torch.Tensor | None = None
    baseline_c_rs: torch.Tensor | None = None
    baseline_graph: torch.cuda.CUDAGraph | None = None

    a_tk: torch.Tensor | None = None
    b_tk: torch.Tensor | None = None
    c_tk_dist: DistBufferLike | None = None
    c_tk_barrier: DistBufferLike | None = None
    tk_graph: torch.cuda.CUDAGraph | None = None

    c_mkernel: DistBufferLike | None = None
    sync_mkernel: DistBufferLike | None = None
    mkernel_graph: torch.cuda.CUDAGraph | None = None


def useful_tflops(M: int, N: int, K: int, elapsed_time_ms: float) -> float:
    """Per-rank GEMM throughput, with elapsed time including reduce-scatter."""
    return 2 * M * N * K / (elapsed_time_ms * 1.0e9)


def cutlass_config_label(candidate) -> str:
    raster, swizzle, tile = candidate
    return f"raster={raster} swizzle={swizzle} tile={tile[0]}x{tile[1]}"


def cutlass_config_kwargs(config: BlackwellBenchConfig, candidate) -> dict:
    raster, swizzle, tile = candidate
    return {
        **config.cutlass_fixed_config,
        "raster_order": raster,
        "swizzle_size": swizzle,
        "mma_tiler_mn": tile,
    }


def cutlass_available(config: BlackwellBenchConfig) -> bool:
    ok, why = load_thirdparty.cutlass_availability(
        "distributed_gemm_reduce_scatter_blackwell.py"
    )
    vote = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    if not vote.item() and config.is_chief:
        print(
            f"[skip] CUTLASS GEMM reduce-scatter: {why or 'unavailable on a peer'}",
            flush=True,
        )
    return bool(vote.item())


def tk_available(config: BlackwellBenchConfig, M: int, N: int, K: int) -> bool:
    tk_ok, tk_why = load_thirdparty.tk_availability(config.world_size, "gemm_rs")
    # B200 has 512x256 cluster tiles and distributes 128-row tiles over 8 GPUs.
    if tk_ok and (M % 1024 or N % 256 or K % 64):
        tk_ok = False
        tk_why = f"M={M}, N={N}, K={K} must be divisible by 1024, 256, 64"
    tk_vote = torch.tensor([1 if tk_ok else 0], device="cuda")
    dist.all_reduce(tk_vote, op=dist.ReduceOp.MIN)
    tk_ok = bool(tk_vote.item())
    if not tk_ok:
        if config.is_chief:
            print(
                f"[skip] ThunderKittens: "
                f"{tk_why or 'unavailable on a peer'}",
                flush=True,
            )

    return tk_ok


def make_blackwell_buffers(
    config: BlackwellBenchConfig, M: int, N: int, K: int, mkernel_module, tk_module=None,
) -> BlackwellBenchVars:
    if M % config.world_size:
        raise ValueError(f"M={M} must be divisible by world_size={config.world_size}")
    torch.manual_seed(42 + config.local_rank)
    buffers = BlackwellBenchVars(
        baseline_a=torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / K**0.25,
        baseline_b=torch.randn((N, K), device="cuda", dtype=torch.bfloat16) / K**0.25,
        baseline_c=torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        baseline_c_rs=torch.empty(
            (M // config.world_size, N), device="cuda", dtype=torch.bfloat16,
        ),
    )
    buffers.c_mkernel = mkernel_module.DistBuffer(
        (M, N), dtype=torch.bfloat16, local_rank=config.local_rank,
        local_world_size=config.world_size, multicast=True,
    )
    # Completion epoch header followed by the ready counters, one per rank.
    buffers.sync_mkernel = mkernel_module.DistBuffer(
        mkernel_module.gemm_rs_blackwell_sync_shape, dtype=torch.int32,
        local_rank=config.local_rank, local_world_size=config.world_size, multicast=False,
    )
    mkernel_module.gemm_rs_blackwell_prepare(
        buffers.baseline_a, buffers.baseline_b, buffers.c_mkernel, buffers.sync_mkernel,
    )
    # Every rank must finish its local reset before any peer publishes a tile.
    torch.cuda.synchronize()
    dist.barrier()
    if tk_module is not None:
        buffers.a_tk = buffers.baseline_a.clone()
        buffers.b_tk = buffers.baseline_b.clone()
        buffers.c_tk_dist = tk_module.TKParallelTensor(
            (M // config.world_size, N), dtype=torch.bfloat16,
            local_rank=config.local_rank, local_world_size=config.world_size,
            multicast=False,
        )
        buffers.c_tk_barrier = tk_module.TKParallelTensor(
            (1, 1), dtype=torch.int,
            local_rank=config.local_rank, local_world_size=config.world_size,
            multicast=True,
        )
        buffers.c_tk_dist.data_.zero_()
        buffers.c_tk_barrier.data_.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    return buffers


def run_baseline(buffers: BlackwellBenchVars):
    torch.mm(buffers.baseline_a, buffers.baseline_b.T, out=buffers.baseline_c)
    dist.reduce_scatter_tensor(buffers.baseline_c_rs, buffers.baseline_c)


def run_tk(tk_module, buffers: BlackwellBenchVars):
    tk_module.matmul_reduce_scatter(
        buffers.a_tk, buffers.b_tk, buffers.c_tk_dist, buffers.c_tk_barrier,
    )


def mkernel_config_label(candidate) -> str:
    swizzle, signals, rs_rows, mma_warps = candidate
    return f"swizzle={swizzle} signals={signals} rs_rows={rs_rows} mma_warps={mma_warps}"


def run_mkernel(mkernel_module, buffers: BlackwellBenchVars, candidate=None):
    """Launch one (swizzle, signals_per_tile, rs_strip_rows, mma_warps); None uses the
    defaults."""
    kwargs = {}
    if candidate is not None:
        swizzle, signals, rs_rows, mma_warps = candidate
        kwargs = dict(
            SCHEDULER_SWIZZLE_SIZE=swizzle, signals_per_tile=signals, rs_strip_rows=rs_rows,
            mma_warps=mma_warps,
        )
    mkernel_module.gemm_rs_blackwell_launch(
        buffers.baseline_a, buffers.baseline_b, buffers.c_mkernel,
        buffers.sync_mkernel, **kwargs,
    )


def mkernel_output(config: BlackwellBenchConfig, buffers: BlackwellBenchVars):
    rows = buffers.baseline_c_rs.shape[0]
    return buffers.c_mkernel.data_.narrow(0, config.local_rank * rows, rows)


def capture_cuda_graph(run_once) -> torch.cuda.CUDAGraph:
    """Warm up and capture one launch; return the graph for replay or checking."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_once()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_once()
    return graph


def check_correctness_gemm_rs_blackwell(config: BlackwellBenchConfig, mkernel_module):
    all_correct = True
    for M in config.shapes_to_test:
        N, K = config.output_dim, config.split_inner_dim

        tk_module = (
            load_thirdparty.load_tk_extension("gemm_rs")
            if tk_available(config, M, N, K) else None
        )
        buffers = make_blackwell_buffers(config, M, N, K, mkernel_module, tk_module)
        run_baseline(buffers)
        for candidate in config.mkernel_candidates(M):
            buffers.c_mkernel.data_.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            run_mkernel(mkernel_module, buffers, candidate)
            torch.cuda.synchronize()
            all_correct = check_close(
                f"gemm_rs mKernel eager {mkernel_config_label(candidate)} M={M}",
                mkernel_output(config, buffers), buffers.baseline_c_rs,
            ) and all_correct

        if tk_module is not None:
            run_tk(tk_module, buffers)
            torch.cuda.synchronize()

            is_correct_tk = check_close(
                f"gemm_rs TK M={M}", buffers.c_tk_dist.data_, buffers.baseline_c_rs,
            )
            all_correct = is_correct_tk and all_correct
            status = "passed :)" if is_correct_tk else "FAILED :("
            if config.is_chief:
                print(
                    f"ThunderKittens M={M} N={N} K={K}: {status}",
                    flush=True,
                )
        for candidate in config.mkernel_candidates(M):
            buffers.mkernel_graph = capture_cuda_graph(
                lambda: run_mkernel(mkernel_module, buffers, candidate)
            )
            # Change inputs on each replay to expose stale counters/output. Prepare
            # is deliberately never repeated: the graph must advance its own epoch.
            for replay in range(3):
                buffers.baseline_a.neg_()
                run_baseline(buffers)
                buffers.mkernel_graph.replay()
                torch.cuda.synchronize()
                all_correct = check_close(
                    f"gemm_rs mKernel graph {mkernel_config_label(candidate)} "
                    f"replay={replay + 1} M={M}",
                    mkernel_output(config, buffers), buffers.baseline_c_rs,
                ) and all_correct
            buffers.mkernel_graph = None
        del buffers

    # All ranks must make the same decision before entering the timed collectives.
    vote = torch.tensor([int(all_correct)], device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    all_correct = bool(vote.item())
    if not all_correct:
        raise RuntimeError("GEMM+RS correctness checks failed; skipping benchmarks.")
    return True


def gemm_rs_blackwell_prepare(
    config: BlackwellBenchConfig, M: int, N: int, K: int, warmup: int, iters: int,
    mkernel_module,
):
    """Return (fn, name, should_wrap); CUTLASS returns its own timing in ms."""
    tk_module = (
        load_thirdparty.load_tk_extension("gemm_rs")
        if tk_available(config, M, N, K) else None
    )
    buffers = make_blackwell_buffers(config, M, N, K, mkernel_module, tk_module)

    def run_baseline_for_graph_capture():
        run_baseline(buffers)

    buffers.baseline_graph = capture_cuda_graph(run_baseline_for_graph_capture)

    def bench_baseline():
        buffers.baseline_graph.replay()

    fns = [(bench_baseline, "baseline", True)]

    # Autotune on graph replays; benchmark_cuda takes the max over ranks, so
    # every rank selects the same configuration.
    mkernel_timings = []
    for candidate in config.mkernel_candidates(M):
        buffers.mkernel_graph = capture_cuda_graph(
            lambda: run_mkernel(mkernel_module, buffers, candidate)
        )
        ms = benchmark_cuda(buffers.mkernel_graph.replay, warmup, iters)
        mkernel_timings.append((candidate, ms))
        buffers.mkernel_graph = None
        time.sleep(5)
    
    best_candidate, _ = min(mkernel_timings, key=lambda item: item[1])
    if config.is_chief:
        for candidate, ms in sorted(mkernel_timings, key=lambda item: item[1]):
            mark = " <- best" if candidate == best_candidate else ""
            print(
                f"  [autotune] mKernel {mkernel_config_label(candidate)}: "
                f"{ms:8.3f} ms  {useful_tflops(M, N, K, ms):8.2f} TFLOP/s{mark}",
                flush=True,
            )

    def bench_mkernel_eager():
        run_mkernel(mkernel_module, buffers, best_candidate)

    buffers.mkernel_graph = capture_cuda_graph(bench_mkernel_eager)

    def bench_mkernel_graph():
        buffers.mkernel_graph.replay()

    fns.append((bench_mkernel_eager, "mKernel eager", True))
    fns.append((bench_mkernel_graph, "mKernel graph", True))
    if cutlass_available(config):
        candidate = config.cutlass_tuned_configs.get(M, config.cutlass_default_config)
        if config.is_chief:
            print(f"  [cutlass] {cutlass_config_label(candidate)}", flush=True)

        def bench_cutlass():
            return load_thirdparty.run_cutlass_reduce_scatter_once(
                m=M, n=N, k=K, config=cutlass_config_kwargs(config, candidate),
                warmup=warmup, iterations=iters,
            )

        fns.append((bench_cutlass, "cutlass", False))

    if tk_module is not None:
        def bench_tk_eager():
            run_tk(tk_module, buffers)

        buffers.tk_graph = capture_cuda_graph(bench_tk_eager)

        def bench_tk_graph():
            buffers.tk_graph.replay()

        fns.append((bench_tk_eager, "ThunderKittens eager", True))
        fns.append((bench_tk_graph, "ThunderKittens graph", True))

    return fns


def benchmark_cuda(run_once, warmup: int, iters: int) -> float:
    """Time the supplied launch or graph replay, taking the slowest rank."""
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    end.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end) / iters], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench"), default="bench")
    parser.add_argument("--shapes", help="Comma-separated M values")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    config = BlackwellBenchConfig()
    if args.shapes is not None:
        config.shapes_to_test = [int(value) for value in args.shapes.split(",")]
    if not config.shapes_to_test or any(
        m <= 0 or m % config.world_size for m in config.shapes_to_test
    ):
        parser.error("M values must be positive and divisible by WORLD_SIZE")
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if config.world_size not in (2, 4, 8):
        parser.error("Launch with torchrun using 2, 4, or 8 GPUs on one node")
    if int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size)) != config.world_size:
        parser.error("This benchmark requires a single node")
    if any(m % (config.world_size * 256) or m % 512 for m in config.shapes_to_test):
        parser.error("mKernel requires M divisible by both WORLD_SIZE * 256 and 512")

    mkernel_module = load_module.load("gemm_rs_blackwell")
    if mkernel_module.gemm_rs_blackwell_world_size != config.world_size:
        parser.error(
            f"Rebuild mKernel with INTRA_NUM_DEVICES={config.world_size} to match WORLD_SIZE"
        )

    torch.cuda.set_device(config.local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This benchmark requires Blackwell GPUs")
    dist.init_process_group("nccl", device_id=torch.device("cuda", config.local_rank))
    fns = []
    fn = None
    try:
        check_correctness_gemm_rs_blackwell(config, mkernel_module)
        if args.mode == "check":
            return
        for M in config.shapes_to_test:
            N, K = config.output_dim, config.split_inner_dim
            # Release the previous shape's graphs before their backing buffers.
            fn = None
            fns.clear()
            fns = gemm_rs_blackwell_prepare(
                config, M, N, K, args.warmup, args.iters, mkernel_module,
            )
            results = {}
            for fn, name, should_wrap in fns:
                ms = (
                    benchmark_cuda(fn, args.warmup, args.iters)
                    if should_wrap else fn()
                )
                if ms is not None:
                    results[name] = ms
                    if config.is_chief:
                        print(
                            f"M={M} N={N} K_local={K} {name}: "
                            f"{ms:.3f} ms  {useful_tflops(M, N, K, ms):.2f} TFLOP/s",
                            flush=True,
                        )
                time.sleep(5)
            if config.is_chief and "cutlass" in results:
                print(
                    f"  CUTLASS speedup vs cuBLAS+NCCL: "
                    f"{results['baseline'] / results['cutlass']:.3f}x",
                    flush=True,
                )
            if config.is_chief:
                for name in ("mKernel eager", "mKernel graph", "ThunderKittens eager", "ThunderKittens graph"):
                    if name in results:
                        print(
                            f"  {name} speedup vs cuBLAS+NCCL: "
                            f"{results['baseline'] / results[name]:.3f}x",
                            flush=True,
                        )
                if "ThunderKittens eager" in results and "ThunderKittens graph" in results:
                    print(
                        f"  TK graph speedup vs eager: "
                        f"{results['ThunderKittens eager'] / results['ThunderKittens graph']:.3f}x",
                        flush=True,
                    )
                # Like-for-like launch modes against TK; CUTLASS times itself.
                for mkernel_name, other_name in (
                    ("mKernel eager", "ThunderKittens eager"),
                    ("mKernel graph", "ThunderKittens graph"),
                    ("mKernel graph", "cutlass"),
                ):
                    if mkernel_name in results and other_name in results:
                        print(
                            f"  {mkernel_name} speedup vs {other_name}: "
                            f"{results[other_name] / results[mkernel_name]:.3f}x",
                            flush=True,
                        )

    finally:
        # all cudagraphs referencing the NCCL communicator have to be released first,
        # else there would be a teardown deadlock
        fn = None
        fns.clear()
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
