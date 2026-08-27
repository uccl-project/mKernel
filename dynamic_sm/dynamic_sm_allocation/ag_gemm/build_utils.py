#!/usr/bin/env python3
"""Build and load the AG-GEMM PyTorch extension.

A cold build takes 3-10 minutes: two large CUDA files, heavy templates
(ThunderKittens, scheduler policies), C++20 and --expt-relaxed-constexpr. Set
MAX_JOBS to compile them in parallel. Any env var that feeds into the module
name forces a full rebuild.

With AG_GEMM_SEPARATE_EXTENSIONS=1 the static and dynamic kernels build into two
.so files, so editing ag_gemm_h100_dynamic.cu only recompiles the dynamic one.
The API is unchanged: load_ag_gemm_module returns a facade exposing
.all_gather_matmul and .all_gather_matmul_dynamic.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent


def _append_cuda_arch_flags(extra_cuda_cflags: list[str], arch: Optional[str]) -> None:
    if arch and arch.lower() == "sm_90a":
        extra_cuda_cflags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    elif arch:
        extra_cuda_cflags.append(f"-arch={arch}")


def _num_devices(explicit_num_devices: int | None = None) -> int:
    if explicit_num_devices is not None:
        return int(explicit_num_devices)
    return int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "8")))


def _module_name(num_devices: int) -> str:
    suffixes: list[str] = []
    if os.environ.get("AG_GEMM_BARRIER_READBACK_DEBUG") == "1":
        suffixes.append("barrierdbg")
    if os.environ.get("AG_GEMM_DYNAMIC_DEBUG") == "1":
        suffixes.append("dyndbg")
    if os.environ.get("USE_GEOMETRY_SELECT_POLICY"):
        suffixes.append("gs")
    forced_target = os.environ.get("AG_GEMM_FORCE_COMM_TARGET")
    if forced_target:
        suffixes.append(f"force{forced_target}")
    if os.environ.get("AG_GEMM_SKIP_HELPER_REBALANCE") == "1":
        suffixes.append("norebalance")
    if os.environ.get("AG_GEMM_STATIC_COMPUTE_ASSIGNMENT") == "1":
        suffixes.append("staticcomp")
    if os.environ.get("AG_GEMM_STATIC_COMM_ASSIGNMENT") == "1":
        suffixes.append("staticcomm")
    if os.environ.get("AG_GEMM_FIXED_ROLE_SPLIT") == "1":
        suffixes.append("fixedsplit")
    if os.environ.get("AG_GEMM_DUMMY_ROLE_SPLIT") == "1":
        suffixes.append("dummyrole")
    if os.environ.get("AG_GEMM_STATIC_COMM_STRIPED") == "1":
        suffixes.append("stripedcomm")
    if os.environ.get("AG_GEMM_STATICIZED_KERNEL") == "1":
        suffixes.append("staticized")
    if os.environ.get("AG_GEMM_STATICIZE_COMM_ROLE") == "1":
        suffixes.append("staticizecommrole")
    if os.environ.get("AG_GEMM_STATICIZE_COMP_ROLE") == "1":
        suffixes.append("staticizecomprole")
    if os.environ.get("AG_GEMM_BLOCKING_REMOTE_WAIT") == "1":
        suffixes.append("blockingwait")
    if os.environ.get("AG_GEMM_DISABLE_COMPUTE_DRAIN") == "1":
        suffixes.append("nodrain")
    if os.environ.get("AG_GEMM_STATICIZED_DYNAMIC_BARRIER") == "1":
        suffixes.append("staticizeddynbar")
    if os.environ.get("AG_GEMM_STATICIZED_EXTRA_SYNC") == "1":
        suffixes.append("staticizedsync")
    if os.environ.get("AG_GEMM_DISABLE_COMPUTE_FASTPATH") == "1":
        suffixes.append("nofastpath")
    if os.environ.get("AG_GEMM_DIRECT_COMPUTE_WAIT") == "1":
        suffixes.append("directwait")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_LOOP") == "1":
        suffixes.append("simpleloop")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_INNER_LOOP") == "1":
        suffixes.append("simpleinner")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_STATIC_MAP") == "1":
        suffixes.append("simplemap")
    if os.environ.get("AG_GEMM_POLICY_FIXED_SCHEDULE", "1") == "1":
        suffixes.append("fixedsched")
    if os.environ.get("AG_GEMM_STATICIZED_COMPUTE_ATOMIC") == "1":
        suffixes.append("staticizedcompatomic")
    if os.environ.get("AG_GEMM_STATICIZED_COMM_TILE") == "1":
        suffixes.append("staticizedcommtile")
    if os.environ.get("AG_GEMM_MUTE_PROGRESS_COUNTERS") == "1":
        suffixes.append("mutecounters")
    if os.environ.get("AG_GEMM_PRINT_SCHED_STATS") == "1":
        suffixes.append("sched")
    if os.environ.get("AG_GEMM_FORCE_COMPUTE_ONLY") == "1":
        suffixes.append("componly")
    if os.environ.get("AG_GEMM_SCHEDULER_ONLY_CTA0") == "1":
        suffixes.append("schedonly")
    debug_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    return f"osgc_adaptive_ag_gemm_d{num_devices}{debug_suffix}"


def _module_name_static(num_devices: int) -> str:
    """Module name for the static-only .so (no env suffixes)."""
    return f"osgc_ag_gemm_static_d{num_devices}"


def _build_dir(name: str) -> Path:
    # Use local disk instead of NFS for faster compilation when available.
    local_base = Path("/data/ziming/torch_ext_build")
    if not local_base.exists():
        local_base = THIS_DIR / ".torch_ext_build"
    return local_base / name


def _get_extra_cuda_cflags(
    tk_num_devices: int,
    arch: Optional[str],
    *,
    combined_build: bool = True,
    dynamic_only: bool = False,
) -> list[str]:
    """Build the list of extra CUDA flags. combined_build adds AG_GEMM_COMBINED_BUILD;
    dynamic_only adds AG_GEMM_DYNAMIC_ONLY (for separate dynamic .so)."""
    flags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-DKITTENS_HOPPER",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        f"-DTK_NUM_DEVICES={tk_num_devices}",
    ]
    if combined_build:
        flags.append("-DAG_GEMM_COMBINED_BUILD")
    if dynamic_only:
        flags.append("-DAG_GEMM_DYNAMIC_ONLY")
    if os.environ.get("AG_GEMM_BARRIER_READBACK_DEBUG") == "1":
        flags.append("-DAG_GEMM_BARRIER_READBACK_DEBUG")
    if os.environ.get("AG_GEMM_DYNAMIC_DEBUG") == "1":
        flags.append("-DAG_GEMM_DYNAMIC_DEBUG")
    if os.environ.get("USE_GEOMETRY_SELECT_POLICY"):
        flags.append("-DUSE_GEOMETRY_SELECT_POLICY")
    forced_target = os.environ.get("AG_GEMM_FORCE_COMM_TARGET")
    if forced_target:
        flags.append(f"-DAG_GEMM_FORCE_COMM_TARGET={int(forced_target)}")
    if os.environ.get("AG_GEMM_SKIP_HELPER_REBALANCE") == "1":
        flags.append("-DAG_GEMM_SKIP_HELPER_REBALANCE")
    if os.environ.get("AG_GEMM_STATIC_COMPUTE_ASSIGNMENT") == "1":
        flags.append("-DAG_GEMM_STATIC_COMPUTE_ASSIGNMENT")
    if os.environ.get("AG_GEMM_STATIC_COMM_ASSIGNMENT") == "1":
        flags.append("-DAG_GEMM_STATIC_COMM_ASSIGNMENT")
    if os.environ.get("AG_GEMM_FIXED_ROLE_SPLIT") == "1":
        flags.append("-DAG_GEMM_FIXED_ROLE_SPLIT")
    if os.environ.get("AG_GEMM_DUMMY_ROLE_SPLIT") == "1":
        flags.append("-DAG_GEMM_DUMMY_ROLE_SPLIT")
    if os.environ.get("AG_GEMM_STATIC_COMM_STRIPED") == "1":
        flags.append("-DAG_GEMM_STATIC_COMM_STRIPED")
    if os.environ.get("AG_GEMM_STATICIZED_KERNEL") == "1":
        flags.append("-DAG_GEMM_STATICIZED_KERNEL")
    if os.environ.get("AG_GEMM_STATICIZE_COMM_ROLE") == "1":
        flags.append("-DAG_GEMM_STATICIZE_COMM_ROLE")
    if os.environ.get("AG_GEMM_STATICIZE_COMP_ROLE") == "1":
        flags.append("-DAG_GEMM_STATICIZE_COMP_ROLE")
    if os.environ.get("AG_GEMM_BLOCKING_REMOTE_WAIT") == "1":
        flags.append("-DAG_GEMM_BLOCKING_REMOTE_WAIT")
    if os.environ.get("AG_GEMM_DISABLE_COMPUTE_DRAIN") == "1":
        flags.append("-DAG_GEMM_DISABLE_COMPUTE_DRAIN")
    if os.environ.get("AG_GEMM_STATICIZED_DYNAMIC_BARRIER") == "1":
        flags.append("-DAG_GEMM_STATICIZED_DYNAMIC_BARRIER")
    if os.environ.get("AG_GEMM_STATICIZED_EXTRA_SYNC") == "1":
        flags.append("-DAG_GEMM_STATICIZED_EXTRA_SYNC")
    if os.environ.get("AG_GEMM_DISABLE_COMPUTE_FASTPATH") == "1":
        flags.append("-DAG_GEMM_DISABLE_COMPUTE_FASTPATH")
    if os.environ.get("AG_GEMM_DIRECT_COMPUTE_WAIT") == "1":
        flags.append("-DAG_GEMM_DIRECT_COMPUTE_WAIT")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_LOOP") == "1":
        flags.append("-DAG_GEMM_SIMPLE_COMPUTE_LOOP")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_INNER_LOOP") == "1":
        flags.append("-DAG_GEMM_SIMPLE_COMPUTE_INNER_LOOP")
    if os.environ.get("AG_GEMM_SIMPLE_COMPUTE_STATIC_MAP") == "1":
        flags.append("-DAG_GEMM_SIMPLE_COMPUTE_STATIC_MAP")
    if os.environ.get("AG_GEMM_POLICY_FIXED_SCHEDULE", "1") == "1":
        flags.append("-DAG_GEMM_POLICY_FIXED_SCHEDULE")
    if os.environ.get("AG_GEMM_STATICIZED_COMPUTE_ATOMIC") == "1":
        flags.append("-DAG_GEMM_STATICIZED_COMPUTE_ATOMIC")
    if os.environ.get("AG_GEMM_STATICIZED_COMM_TILE") == "1":
        flags.append("-DAG_GEMM_STATICIZED_COMM_TILE")
    if os.environ.get("AG_GEMM_MUTE_PROGRESS_COUNTERS") == "1":
        flags.append("-DAG_GEMM_MUTE_PROGRESS_COUNTERS")
    if os.environ.get("AG_GEMM_PRINT_SCHED_STATS") == "1":
        flags.append("-DAG_GEMM_PRINT_SCHED_STATS")
    if os.environ.get("AG_GEMM_FORCE_COMPUTE_ONLY") == "1":
        flags.append("-DAG_GEMM_FORCE_COMPUTE_ONLY")
    if os.environ.get("AG_GEMM_SCHEDULER_ONLY_CTA0") == "1":
        flags.append("-DAG_GEMM_SCHEDULER_ONLY_CTA0")
    _append_cuda_arch_flags(flags, arch)
    return flags


def _ag_gemm_facade(mod_static, mod_dynamic):
    """Single object exposing both static and dynamic entrypoints and TKParallelTensor.

    TKParallelTensor comes from mod_static: PyBind11 uses a global type registry,
    so the dynamic .so is built with -DTK_PARALLEL_TENSOR_BINDINGS_EXTERNAL to skip
    re-registration (which would raise "type already registered").
    """
    class _Facade:
        pass
    f = _Facade()
    f.all_gather_matmul = mod_static.all_gather_matmul
    f.all_gather_matmul_dynamic = mod_dynamic.all_gather_matmul_dynamic
    f.TKParallelTensor = mod_static.TKParallelTensor
    return f


def _remove_stale_locks(build_dir: Path) -> None:
    """Remove stale lock files left by killed build processes.

    torch.utils.cpp_extension.load uses a 'lock' file to prevent concurrent
    builds across distributed workers.  If a build process is killed, the lock
    is never released and all subsequent workers spin-wait indefinitely at 100%
    CPU.  We check whether the lock is truly held by a live process; if not,
    we remove it so the next worker can proceed.
    """
    import fcntl
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return
    try:
        with open(lock_path, "r+") as f:
            try:
                # Non-blocking exclusive lock: succeeds only if NO other process holds it.
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # We got the lock → it was stale.  Remove it.
                fcntl.flock(f, fcntl.LOCK_UN)
                lock_path.unlink(missing_ok=True)
            except OSError:
                # Another live process holds the lock → leave it alone.
                pass
    except OSError:
        pass


def load_ag_gemm_module(
    arch: Optional[str],
    num_devices: int | None = None,
):
    python_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(":")
    if python_bin not in path_entries:
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"

    tk_num_devices = _num_devices(num_devices)
    verbose = int(os.environ.get("LOCAL_RANK", "0")) == 0

    if os.environ.get("AG_GEMM_SEPARATE_EXTENSIONS") == "1":
        # Two .so files: only the one whose source changed will recompile.
        flags_static = _get_extra_cuda_cflags(
            tk_num_devices, arch, combined_build=False, dynamic_only=False
        )
        name_static = _module_name_static(tk_num_devices)
        build_dir_static = _build_dir(name_static)
        build_dir_static.mkdir(parents=True, exist_ok=True)
        _remove_stale_locks(build_dir_static)
        mod_static = load(
            name=name_static,
            sources=[str(THIS_DIR / "ag_gemm_h100_static.cu")],
            extra_include_paths=[str(REPO_ROOT / "include"), str(THIS_DIR)],
            extra_cuda_cflags=flags_static,
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir_static),
            verbose=verbose,
        )
        flags_dynamic = _get_extra_cuda_cflags(
            tk_num_devices, arch, combined_build=False, dynamic_only=True
        )
        # When loading two .so files in the same process, PyBind11 uses a
        # global type registry.  The static .so already registered TKParallelTensor;
        # the dynamic .so must skip registration to avoid "type already registered".
        flags_dynamic.append("-DTK_PARALLEL_TENSOR_BINDINGS_EXTERNAL")
        name_dynamic = _module_name(tk_num_devices)
        build_dir_dynamic = _build_dir(name_dynamic)
        build_dir_dynamic.mkdir(parents=True, exist_ok=True)
        _remove_stale_locks(build_dir_dynamic)
        mod_dynamic = load(
            name=name_dynamic,
            sources=[str(THIS_DIR / "ag_gemm_h100_dynamic.cu")],
            extra_include_paths=[str(REPO_ROOT / "include"), str(THIS_DIR)],
            extra_cuda_cflags=flags_dynamic,
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir_dynamic),
            verbose=verbose,
        )
        return _ag_gemm_facade(mod_static, mod_dynamic)

    # Single combined .so (default).
    extra_cuda_cflags = _get_extra_cuda_cflags(
        tk_num_devices, arch, combined_build=True, dynamic_only=False
    )
    name = _module_name(tk_num_devices)
    build_dir = _build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_locks(build_dir)
    return load(
        name=name,
        sources=[
            str(THIS_DIR / "ag_gemm_h100_static.cu"),
            str(THIS_DIR / "ag_gemm_h100_dynamic.cu"),
        ],
        extra_include_paths=[str(REPO_ROOT / "include"), str(THIS_DIR)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=verbose,
    )


def import_ag_gemm_module(num_devices: int | None = None):
    tk_num_devices = _num_devices(num_devices)

    if os.environ.get("AG_GEMM_SEPARATE_EXTENSIONS") == "1":
        name_static = _module_name_static(tk_num_devices)
        name_dynamic = _module_name(tk_num_devices)
        build_dir_static = _build_dir(name_static)
        build_dir_dynamic = _build_dir(name_dynamic)
        so_static = list(build_dir_static.glob(f"{name_static}*.so"))
        so_dynamic = list(build_dir_dynamic.glob(f"{name_dynamic}*.so"))
        if not so_static:
            raise RuntimeError(
                f"No static .so found in {build_dir_static}. Run without --skip-build first."
            )
        if not so_dynamic:
            raise RuntimeError(
                f"No dynamic .so found in {build_dir_dynamic}. Run without --skip-build first."
            )
        spec_s = importlib.util.spec_from_file_location(name_static, so_static[0])
        if spec_s is None or spec_s.loader is None:
            raise RuntimeError(f"failed to import static from {so_static[0]}")
        mod_static = importlib.util.module_from_spec(spec_s)
        spec_s.loader.exec_module(mod_static)
        spec_d = importlib.util.spec_from_file_location(name_dynamic, so_dynamic[0])
        if spec_d is None or spec_d.loader is None:
            raise RuntimeError(f"failed to import dynamic from {so_dynamic[0]}")
        mod_dynamic = importlib.util.module_from_spec(spec_d)
        spec_d.loader.exec_module(mod_dynamic)
        return _ag_gemm_facade(mod_static, mod_dynamic)

    name = _module_name(tk_num_devices)
    build_dir = _build_dir(name)
    so_files = list(build_dir.glob(f"{name}*.so"))
    if not so_files:
        raise RuntimeError(
            f"No built .so found in {build_dir}. Run the benchmark without --skip-build first."
        )
    so_path = so_files[0]
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
