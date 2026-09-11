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
        "cutlass_distributed_all_gather_gemm_blackwell", path
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
    for symbol in ("TKParallelTensor", "all_gather_matmul"):
        if not hasattr(module, symbol):
            raise AttributeError(f"{path} does not define {symbol}")
    _TK_MODULE = module
    return module


def load_tk_extension(kernel_folder_path: str):
    return _load_tk_extension(kernel_folder_path)


def tk_availability(world_size: int, kernel_folder_path: str) -> tuple[bool, str]:
    if world_size != 8:
        return False, "upstream TK kernels only support intranode communication"
    try:
        _load_tk_extension(kernel_folder_path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""