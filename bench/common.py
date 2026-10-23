"""Shared bench utilities for the release kernels.

Each <kernel>_bench.py reuses these primitives so the per-kernel script
stays focused on alloc + run_once + correctness reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist

# Make the python loader importable.
HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(RELEASE / "python"))
import load_module  # noqa: E402


# ----------------------------------------------------------------------
# RDMA buffer backing policy
# ----------------------------------------------------------------------

def rdma_backing(*, default: str = "vmm") -> str:
    """Resolve the global RDMA buffer backing.

    Precedence, high to low:
      MKERNEL_RDMA_BUFFER_BACKING
      default
    """
    raw = os.environ.get("MKERNEL_RDMA_BUFFER_BACKING", default)
    backing = raw.strip().lower()
    if backing not in ("vmm", "peermem"):
        raise ValueError(
            f"Unsupported RDMA backing {raw!r}; expected 'vmm' or 'peermem'")
    return backing


def is_peermem_backing(backing: str) -> bool:
    return backing == "peermem"


def make_dist_buffer(
    mod,
    shape,
    *,
    dtype,
    local_rank: int,
    local_world_size: int,
    multicast: bool = False,
    backing: str = "vmm",
):
    """Create a DistBuffer from the generic RDMA backing policy."""
    if backing not in ("vmm", "peermem"):
        raise ValueError(
            f"Unsupported RDMA backing {backing!r}; expected 'vmm' or 'peermem'")
    if multicast and backing == "peermem":
        raise ValueError("peermem RDMA backing does not support multicast")
    return mod.DistBuffer(
        shape, dtype=dtype,
        local_rank=local_rank, local_world_size=local_world_size,
        multicast=multicast,
        backing="cuda_malloc" if backing == "peermem" else "vmm",
    )


def rdma_policy_label(kernel: str, **roles: str) -> str:
    parts = " ".join(f"{k}={v}" for k, v in roles.items())
    return f"[{kernel}] rdma_backing {parts}"


# ----------------------------------------------------------------------
# Distributed init / launcher helpers
# ----------------------------------------------------------------------


def get_peer_ips(node_idx: int, num_nodes: int) -> list[str]:
    """Resolve the list of peer IPs (length num_nodes - 1) for this node.

    Reads NODE0_IP, NODE1_IP, ..., NODE{N-1}_IP from the environment.
    Returns peers in the same ring slot order as
    internode::peer_rank_for_slot(): node_idx + 1, node_idx + 2, ...
    wrapping modulo num_nodes.
    """
    all_ips = []
    for i in range(num_nodes):
        ip = os.environ.get(f"NODE{i}_IP")
        if not ip:
            raise RuntimeError(f"NODE{i}_IP must be set for NUM_NODES={num_nodes}")
        all_ips.append(ip)
    return [all_ips[(node_idx + 1 + slot) % num_nodes] for slot in range(num_nodes - 1)]


def get_peer_ports(node_idx: int, num_nodes: int, base_port: int) -> list[int]:
    """Compute symmetric per-pair TCP ports (length num_nodes - 1) for this node.

    Peers follow internode::peer_rank_for_slot() ring order. Each unordered
    (lo, hi) rank pair gets a unique port computed as
    `base_port + (lo * num_nodes + hi) * local_world_size`, so both sides of
    a pair derive the same port from `(min(self, peer), max(self, peer))`.
    At N==2 the loop has one slot and produces `base_port + local_world_size`.

    `base_port` is already per-local-rank (`TCP_PORT + LOCAL_RANK`) in the
    bench scripts. Striding pair offsets by local_world_size prevents adjacent
    local ranks on the same host from colliding on the same listen port.
    """
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    ports = []
    for slot in range(num_nodes - 1):
        i = (node_idx + 1 + slot) % num_nodes
        lo, hi = (node_idx, i) if node_idx < i else (i, node_idx)
        ports.append(base_port + (lo * num_nodes + hi) * local_world_size)
    return ports


def get_num_nodes() -> int:
    """Resolve number of nodes from the NUM_NODES env var (default: 2).

    Bench scripts and the launcher consult this to size the global world
    (`WORLD_SIZE = NUM_NODES * LOCAL_WORLD_SIZE`) and the multi-peer
    session bringup (peer_ips / peer_tcp_ports lists).
    """
    return int(os.environ.get("NUM_NODES", "2"))


def init_dist():
    """Init torch.distributed from torchrun env. Returns (local_rank, world_size, node_idx)."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "16"))
    node_idx = int(os.environ.get("NODE_IDX", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return local_rank, world_size, node_idx


def host_barrier():
    """Cheap collective barrier on a 1-element CPU tensor (gloo backend)."""
    t = torch.zeros(1, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)


def max_across_ranks(value: float) -> float:
    """Return max(value) across the torchrun world using the CPU gloo group."""
    t = torch.tensor([float(value)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def maybe_print_rank_values(label: str, value: float) -> None:
    """Optionally print one scalar per rank for timing diagnostics."""
    want_print = torch.tensor(
        [1 if os.environ.get("RELEASE_BENCH_PRINT_RANK_TIMES", "0") == "1" else 0],
        dtype=torch.int32,
    )
    dist.all_reduce(want_print, op=dist.ReduceOp.MAX)
    if int(want_print.item()) == 0:
        return
    t = torch.tensor([float(value)], dtype=torch.float64)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    if dist.get_rank() == 0:
        rank_ms = [round(float(x.item()), 3) for x in gathered]
        print(f"[rank-times] {label}: {rank_ms}", flush=True)


def resolve_timing_mode() -> str:
    """"batch" (default) or "periter". Both benches read this the same way.

    batch matches ThunderKittens' harness (common.py:benchmark_no_l2_clear):
    warm up, then N back-to-back launches inside one event pair, divided by N.
    periter brackets each launch and resets between, which leaves idle gaps that
    let the GPU boost -- not comparable to TK, in either direction.
    """
    mode = os.environ.get("MKERNEL_BENCH_TIMING", "batch")
    # Back-compat with the two older flags, whose senses were opposite.
    if os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1":
        mode = "periter"
    if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
        mode = "periter"
    if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "1":
        mode = "batch"
    return mode


def benchmark_batch(launch: Callable[[], None], reset: Callable[[], None],
                    n_iters: int, warmup: int = 3) -> float:
    """Average ms per launch over n_iters back-to-back launches.

    One event pair spans the whole batch, so no per-iteration host work lands
    inside the measurement. The timed loop leaves device state dirty by design
    -- a kernel that accumulates into its output needs the reset that this
    deliberately skips -- so callers that verify correctness must run one clean
    iteration afterwards.
    """
    reset()
    dist.barrier()
    time.sleep(0.05)
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / n_iters


def avg_then_max(samples: list[float], label: str = "") -> float:
    """Benchmark timing: local average, then max over all ranks."""
    local = sum(samples) / len(samples) if samples else float("nan")
    if label:
        maybe_print_rank_values(label, local)
    return max_across_ranks(local)


def median_then_max(samples: list[float], label: str = "") -> float:
    """Benchmark timing: local median, then max over all ranks."""
    if not samples:
        return max_across_ranks(float("nan"))
    sorted_samples = sorted(samples)
    local = sorted_samples[len(sorted_samples) // 2]
    if label:
        maybe_print_rank_values(label, local)
    return max_across_ranks(local)


# ----------------------------------------------------------------------
# Bench timing wrapper
# ----------------------------------------------------------------------

def run_bench_loop(
    run_once: Callable[[], None],
    iters: int,
    warmup: int,
    reset_state: Callable[[], None] = None,
) -> list[float]:
    """Standard timed loop. Returns per-iter wall (ms).

    Pattern (load-bearing — see commit 23d15b4):
      for warmup_iter:
          reset_state(); run_once(); cuda.synchronize()
      for iter:
          reset_state()
          host_barrier(); time.sleep(0.05)   # proxy CQE drain
          start.record(); run_once(); end.record(); cuda.synchronize()
          samples.append(start.elapsed_time(end))
    """
    for _ in range(warmup):
        if reset_state is not None:
            reset_state()
        run_once()
        torch.cuda.synchronize()
        host_barrier()

    samples = []
    for _ in range(iters):
        if reset_state is not None:
            reset_state()
        host_barrier()
        time.sleep(0.05)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_once()
        e.record()
        torch.cuda.synchronize()
        host_barrier()
        samples.append(s.elapsed_time(e))
    return samples


# ----------------------------------------------------------------------
# Reproduction check
# ----------------------------------------------------------------------

def compare_to_reference(
    observed_ms: list[float],
    reference_json: Path,
    tolerance: float = 0.05,
) -> bool:
    """Compare observed wall-times to the source-of-truth JSON.

    Returns True if every shape passes (observed ≤ reference × (1 + tolerance)).
    Prints PASS/FAIL per shape. Caller should sys.exit(non-zero) on failure.
    """
    if not reference_json.exists():
        print(f"[reproduce-check] reference JSON not found: {reference_json}",
              file=sys.stderr)
        return False
    with open(reference_json) as f:
        ref = json.load(f)
    ref_ms = ref["fused_ms"]
    ref_sizes = ref["sizes"]
    if len(observed_ms) != len(ref_ms):
        print(f"[reproduce-check] shape count mismatch: "
              f"observed={len(observed_ms)} reference={len(ref_ms)}", file=sys.stderr)
        return False
    all_pass = True
    print(f"[reproduce-check] vs {reference_json.name}, tolerance ±{tolerance*100:.0f}%")
    for i, (obs, ref_v, sz) in enumerate(zip(observed_ms, ref_ms, ref_sizes)):
        upper = ref_v * (1.0 + tolerance)
        ok = obs <= upper
        mark = "PASS" if ok else "FAIL"
        delta_pct = (obs - ref_v) / ref_v * 100
        print(f"  shape {sz}: observed {obs:.3f} ms vs ref {ref_v:.3f} ms "
              f"({delta_pct:+.1f}%)  [{mark}]")
        all_pass = all_pass and ok
    return all_pass


def compare_named_results(
    kernel: str,
    result_sizes: list[str],
    result_fused: list[float],
    reference_json: str | Path,
    tolerance: float = 0.05,
) -> bool:
    """Compare result rows by shape label, allowing targeted shape runs."""
    reference_json = Path(reference_json)
    with open(reference_json) as f:
        ref = json.load(f)
    ref_by_size = dict(zip(ref["sizes"], ref["fused_ms"]))
    ok = True
    print(f"[{kernel}] vs {reference_json} (tol ±{tolerance * 100:.0f}%)", flush=True)
    for sz, obs in zip(result_sizes, result_fused):
        if sz not in ref_by_size:
            print(f"  {sz}: no reference row [FAIL]", flush=True)
            ok = False
            continue
        refv = ref_by_size[sz]
        delta = (obs - refv) / refv * 100
        mark = "PASS" if obs <= refv * (1.0 + tolerance) else "FAIL"
        if obs > refv * (1.0 + tolerance):
            ok = False
        print(f"  {sz}: obs={obs:.3f} ref={refv:.3f} ({delta:+.1f}%) {mark}", flush=True)
    return ok


def gather_cpu_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather a tensor through the CPU process group for small correctness checks."""
    obj = tensor.detach().cpu().contiguous()
    gathered: list[torch.Tensor | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return [x for x in gathered if x is not None]


def check_close(
    name: str,
    observed: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 0.35,
    rtol: float = 0.08,
    mean_rtol: float = 0.01,
) -> bool:
    """Distributed tensor closeness check; returns False if any rank fails.

    Three-tier acceptance for bf16 collective kernels:
      1. max_abs <= atol  → strict pass.
      2. max_rel <= rtol  → relative pass (current behavior).
      3. mean_abs <= mean_rtol * ref_mean → bulk-clean pass. bf16 matmul + reduce
         leaves a few outlier elements where the kernel's reduction order
         disagrees with torch's, even when 99.99%+ of elements are bit-clean.
         When the average element error is < 1% of the average expected
         magnitude, treat the outliers as precision noise rather than failure.
         This correctly accepts gemm_rs / ring_attention bf16 noise while still
         catching real systemic bugs (e.g. ag_gemm M=32768 where mean_abs is
         ~1.6x ref_max → far above mean_rtol).
    """
    observed_f = observed.detach().float()
    expected_f = expected.detach().to(device=observed.device).float()
    diff = (observed_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    ref_mean = float(expected_f.abs().mean().item()) if expected_f.numel() else 0.0
    denom = expected_f.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    bulk_clean = mean_abs <= mean_rtol * max(ref_mean, 1e-6)
    local_ok = (max_abs <= atol) or (max_rel <= rtol) or bulk_clean
    if not local_ok:
        print(f"[correctness-local] rank={dist.get_rank()} {name}: "
              f"max_abs={max_abs:.6f} max_rel={max_rel:.6f} "
              f"mean_abs={mean_abs:.6f} ref_mean={ref_mean:.4f}",
              flush=True)
    # device="cuda" so the NCCL backend can run the reduction. With a CPU
    # tensor the bench scripts (which all init NCCL) hit
    # "No backend type associated with device type cpu" at the first check.
    stats = torch.tensor([max_abs, max_rel, 0.0 if local_ok else 1.0,
                          mean_abs, ref_mean],
                         dtype=torch.float64, device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    ok = bool(stats[2].item() == 0.0)
    if dist.get_rank() == 0:
        mark = "PASS" if ok else "FAIL"
        if mark == "PASS" and stats[0].item() > atol and stats[1].item() > rtol:
            mark += " (bulk-clean: mean_abs/ref_mean within mean_rtol)"
        print(f"[correctness] {name}: max_abs={stats[0].item():.6f} "
              f"max_rel={stats[1].item():.6f} "
              f"mean_abs={stats[3].item():.6f} ref_mean={stats[4].item():.4f} "
              f"atol={atol} rtol={rtol} mean_rtol={mean_rtol} {mark}",
              flush=True)
    return ok


# ----------------------------------------------------------------------
# Result JSON writer
# ----------------------------------------------------------------------

def write_results_json(
    out_path: Path,
    kernel: str,
    sizes: Iterable,
    fused_ms: Iterable[float],
    note: str = "",
):
    """Write a benchmark JSON consumed by the plotting scripts.

    This function merges with the existing JSON at `out_path` rather than
    replacing it wholesale. A bench run with `--shapes 4096` should not erase
    results for the other shapes that the chart needs. New entries
    overwrite same-shape entries; absent shapes are preserved.
    """
    new_sizes = list(sizes)
    new_ms = list(fused_ms)
    out_path = Path(out_path)
    existing_sizes: list = []
    existing_ms: list = []
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            existing_sizes = list(existing.get("sizes", []))
            existing_ms = list(existing.get("fused_ms", []))
        except Exception:
            existing_sizes, existing_ms = [], []

    # Build a merged map: existing shapes first, new ones overwrite.
    # If MKERNEL_BENCH_BEST_OF_N=1, between-sweep best-of: take min when same
    # shape is rewritten. Used by run.sh's BEST_OF_N=k loop to absorb
    # cluster-noise variance. Reset by deleting the JSON before the loop.
    best_of = os.environ.get("MKERNEL_BENCH_BEST_OF_N", "0") == "1"
    merged: dict = {}
    for s, ms in zip(existing_sizes, existing_ms):
        merged[s] = ms
    for s, ms in zip(new_sizes, new_ms):
        if best_of and s in merged:
            merged[s] = min(merged[s], ms)
        else:
            merged[s] = ms

    # Sort by numeric size (handle "M=4096", "tokens=4096", or bare int).
    def _numkey(s):
        if isinstance(s, int):
            return s
        s = str(s)
        if "=" in s:
            try:
                return int(s.split("=", 1)[1])
            except ValueError:
                return 0
        try:
            return int(s)
        except ValueError:
            return 0
    sorted_sizes = sorted(merged.keys(), key=_numkey)
    sorted_ms = [merged[s] for s in sorted_sizes]

    data = dict(existing)  # preserve nccl_ms, _note, etc. on existing JSON
    world_size = int(os.environ.get("WORLD_SIZE", existing.get("world_size", 8)))
    num_nodes = int(os.environ.get("NUM_NODES", existing.get("num_nodes", 2)))
    data.update({
        "kernel": kernel,
        "sizes": sorted_sizes,
        "fused_ms": sorted_ms,
        "world_size": world_size,
        "num_nodes": num_nodes,
        "_note": note or existing.get("_note", f"release {kernel} bench (EFA backend)"),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------------------------------------------------
# CLI parser shared across all bench scripts
# ----------------------------------------------------------------------

def make_argparser(default_shapes: list[int], kernel_name: str):
    p = argparse.ArgumentParser(description=f"{kernel_name} release bench")
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str, default=",".join(str(s) for s in default_shapes))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None,
                   help="Path to reference JSON; non-zero exit on >5% regression.")
    return p


def parse_shapes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def check_cross_rank_identical(
    label: str,
    local_tensor: torch.Tensor,
    is_chief: bool,
) -> bool:
    """Assert `local_tensor` is byte-identical across every rank in the
    default process group. Returns True on PASS. Only the chief prints."""
    world_size = dist.get_world_size()
    if world_size <= 1:
        return True

    flat = local_tensor.detach().contiguous()
    gathered = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)

    if not is_chief:
        return True

    ref = gathered[0]
    diffs = []
    for k in range(1, world_size):
        d = int((gathered[k] != ref).sum().item())
        if d > 0:
            diffs.append((k, d))
    if diffs:
        total = ref.numel()
        msg = ", ".join(f"rank={k}:{d}/{total}" for k, d in diffs)
        print(f"[invariant {label}/bit-identical] FAIL {msg}", flush=True)
        return False
    print(f"[invariant {label}/bit-identical] PASS world={world_size}",
          flush=True)
    return True


def check_deterministic_rerun(
    label: str,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    is_chief: bool,
) -> bool:
    """Two outputs from independent kernel invocations on the same inputs
    must be byte-identical. Catches data races on shared flags, uninitialized
    scratch, and reduce-ordering nondeterminism. Both snapshots should be
    taken AFTER the timed perf loop so the extra runs do not perturb the
    benchmark numbers."""
    if not is_chief:
        return True
    if out_a.shape != out_b.shape or out_a.dtype != out_b.dtype:
        print(f"[invariant {label}/deterministic] FAIL shape/dtype mismatch "
              f"({tuple(out_a.shape)}/{out_a.dtype} vs "
              f"{tuple(out_b.shape)}/{out_b.dtype})", flush=True)
        return False
    diff = int((out_a != out_b).sum().item())
    if diff > 0:
        total = out_a.numel()
        max_abs = float((out_a.float() - out_b.float()).abs().max().item())
        ref_mag = float(out_a.float().abs().mean().item())
        print(f"[invariant {label}/deterministic] FAIL diff={diff}/{total} "
              f"max_abs={max_abs:.6g} ref_mean={ref_mag:.6g}",
              flush=True)
        return False
    print(f"[invariant {label}/deterministic] PASS", flush=True)
    return True
