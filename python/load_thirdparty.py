"""
Helper to kernels from third parties for blackwell intranode tests, based on path provided by environment variables

For TK, the importer assumes that the kernel has already been compiled

Supported env paths and imported kernels:
1. CUTLASS_PATH: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed
2. THUNDERKITTENS_PATH: https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/parallel
"""

import contextlib
import importlib.util
import io
import os
import sysconfig
from pathlib import Path

import torch
import torch.distributed as dist

# cutlass import helpers

_CUTLASS_ENV_ROOT = "CUTLASS_PATH"
_CUTLASS_MODULE = None

def _cutlass_example_path(kernel_name: str) -> Path | None:
    root = os.environ.get(_CUTLASS_ENV_ROOT)
    if root:
        return (
            Path(root).expanduser()
            / f"examples/python/CuTeDSL/cute/blackwell/kernel/distributed/{kernel_name}"
        )
    return None


def _load_cutlass_example(kernel_name: str):
    global _CUTLASS_MODULE
    if _CUTLASS_MODULE is not None:
        return _CUTLASS_MODULE

    path = _cutlass_example_path(kernel_name)
    if path is None:
        raise RuntimeError(f"set {_CUTLASS_ENV_ROOT} to a CUTLASS checkout")
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")

    spec = importlib.util.spec_from_file_location(
        f"cutlass_{Path(kernel_name).stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"{path} does not define run()")
    _CUTLASS_MODULE = module
    return module


def cutlass_availability(kernel_name: str) -> tuple[bool, str]:
    try:
        _load_cutlass_example(kernel_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


CutlassConfig = tuple[tuple[int, int], tuple[int, int], bool]


def run_cutlass_once(
    kernel_name: str,
    *,
    m: int,
    n: int,
    k: int,
    config: CutlassConfig,
    warmup: int,
    iterations: int,
) -> float:
    """Run one upstream CUTLASS configuration and return its max-rank time in ms.

    The pinned upstream helper owns its benchmark loop and returns microseconds.
    It also destroys the caller's process group before returning, which is right
    for its standalone CLI but not for an embedded benchmark. Temporarily make
    that teardown a no-op so all candidates and mKernel share one process group.
    """
    example = _load_cutlass_example(kernel_name)
    import cutlass

    # Upstream's standalone __main__ defines this module global after parsing
    # torchrun's rank. Imported run() still references it when constructing
    # streams and walking the ring, but __main__ is not executed by our loader.
    example.local_rank = int(os.environ["LOCAL_RANK"])
    mma_tiler_mn, cluster_shape_mn, use_2cta_instrs = config

    destroy_process_group = dist.destroy_process_group
    can_implement = example.PersistentDenseGemmKernel.can_implement

    def quiet_can_implement(*args, **kwargs):
        # Upstream prints MNKL unconditionally from every rank. It is not a
        # tuning result and obscures the per-candidate timing emitted by callers.
        with contextlib.redirect_stdout(io.StringIO()):
            return can_implement(*args, **kwargs)

    dist.destroy_process_group = lambda *args, **kwargs: None
    example.PersistentDenseGemmKernel.can_implement = quiet_can_implement
    try:
        time_us = example.run(
            mnkl=(m, n, k, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="n",
            c_major="n",
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            use_tma_store=True,
            warmup_iterations=warmup,
            iterations=iterations,
            skip_ref_check=True,
            use_cold_l2=False,
        )
    finally:
        example.PersistentDenseGemmKernel.can_implement = can_implement
        dist.destroy_process_group = destroy_process_group
    torch.cuda.synchronize()
    return float(time_us) / 1000.0


def run_cutlass_reduce_scatter_once(
    *,
    m: int,
    n: int,
    k: int,
    config: dict,
    warmup: int,
    iterations: int,
    check: bool = False,
) -> float | None:
    """Run the pinned RS example in the caller's existing process group.

    Return max-rank milliseconds, or None if any rank rejects the candidate.
    In check mode the upstream GEMM, RS, and barrier assertions run instead.
    Call the distributed helpers directly: run() discards their timing result
    and destroys the process group, so it cannot be used for repeated tuning.
    Set CUTLASS_VERBOSE=1 to show upstream benchmark stdout.
    """
    import cutlass

    example = _load_cutlass_example("distributed_gemm_reduce_scatter_blackwell.py")
    kwargs = dict(
        mnkl=(m, n, k, 1),
        ab_dtype=cutlass.BFloat16,
        c_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",  # B is [N, K], matching the cuBLAS baseline.
        c_major="n",
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        swizzle_size=1,
        raster_order="m",
        use_2cta_instrs=True,
        use_tma_store=True,
        reduce_scatter="LDMC",
    )
    settings = dict(config)
    mode = settings.pop("benchmark_or_test", "benchmark")
    if mode != "benchmark":
        raise ValueError("RS tuning requires benchmark_or_test='benchmark'")
    use_cuda_graph = settings.pop("use_cuda_graph", True)
    csv_report = settings.pop("csv_report", "/tmp/mkernel_gemm_rs_cutlass.csv")
    kwargs.update(settings)
    kwargs.update(
        global_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    # Vote before allocating symmetric memory or launching any collective GPU
    # work. A post-launch vote cannot recover a peer stuck inside a kernel.
    try:
        kernel = example.Sm100PersistentDenseGemmReduceScatterLDMCKernel(
            kwargs["acc_dtype"], kwargs["c_dtype"], kwargs["use_2cta_instrs"],
            kwargs["mma_tiler_mn"], kwargs["cluster_shape_mn"], kwargs["use_tma_store"],
            rank_id=kwargs["global_rank"], num_ranks=kwargs["world_size"],
            reduce_scatter=kwargs["reduce_scatter"],
            swizzle_size=kwargs["swizzle_size"], raster_order=kwargs["raster_order"],
        )
        supported = kernel.can_implement(
            kwargs["mnkl"], kwargs["ab_dtype"], kwargs["c_dtype"],
            kwargs["a_major"], kwargs["b_major"], kwargs["c_major"],
        )
    except example.testing.CantImplementError as exc:
        if kwargs["global_rank"] == 0:
            print(f"[skip] CUTLASS GEMM reduce-scatter: {exc}", flush=True)
        supported = False
    vote = torch.tensor([int(supported)], device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    if not vote.item():
        return None

    if check:
        example.run_distributed_test(**kwargs)
        torch.cuda.synchronize()
        return 0.0

    # Upstream prints a timing from every rank, even for a single candidate.
    stdout_context = (
        contextlib.nullcontext() if os.environ.get("CUTLASS_VERBOSE", "0") == "1"
        else contextlib.redirect_stdout(io.StringIO())
    )
    with stdout_context:
        time_us = example.run_distributed_benchmark(
            **kwargs,
            warmup_iterations=warmup,
            iterations=iterations,
            use_cuda_graph=use_cuda_graph,
            benchmark_all=False,
            csv_report=csv_report,
        )
    torch.cuda.synchronize()
    elapsed = torch.tensor([float(time_us) / 1000.0], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


# TK import helper

_TK_ENV_ROOT = "THUNDERKITTENS_PATH"
_TK_MODULE = None

def _tk_extension_path(kernel_folder_path: str) -> Path | None:
    root = os.environ.get(_TK_ENV_ROOT)
    if not root:
        return None
    root_path = Path(root).expanduser()
    nested = root_path / f"kernels/parallel/{kernel_folder_path}"
    search_dir = nested if nested.is_dir() else root_path

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    preferred = search_dir / f"_C{extension_suffix}"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        search_dir.glob("_C*.so"), key=lambda path: path.stat().st_mtime_ns
    )
    return candidates[-1] if candidates else preferred


def _load_tk_extension(kernel_folder_path: str):
    global _TK_MODULE
    if _TK_MODULE is not None:
        return _TK_MODULE

    path = _tk_extension_path(kernel_folder_path)
    if path is None:
        raise RuntimeError(
            f"set {_TK_ENV_ROOT} to a ThunderKittens checkout"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; build the GPU-compatible ThunderKittens "
            "*_C extension first"
        )

    # ThunderKittens names the pybind module `_C`, so the import spec must keep
    # that final component even when the shared object has an ABI-tagged name.
    spec = importlib.util.spec_from_file_location("_C", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel_symbol = (
        "matmul_reduce_scatter" if kernel_folder_path == "gemm_rs"
        else "all_gather_matmul"
    )
    for symbol in ("TKParallelTensor", kernel_symbol):
        if not hasattr(module, symbol):
            raise AttributeError(f"{path} does not define {symbol}")
    _TK_MODULE = module
    return module


def load_tk_extension(kernel_folder_path: str):
    return _load_tk_extension(kernel_folder_path)


def tk_availability(world_size: int, kernel_folder_path: str) -> tuple[bool, str]:
    if world_size != 8:
        return False, "upstream TK kernels require 8 GPUs on one node"
    try:
        _load_tk_extension(kernel_folder_path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""
