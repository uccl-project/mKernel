import importlib.util
import os
import sys
import torch
import torch.distributed as dist
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close

SHAPES = [2048, 4096, 8192, 16384, 32768]

WARMUP = 30
BENCH_ITER = 60


def make_barrier(mod, local_rank, world_size):
    """Allocate a zeroed comp->comm barrier buffer.

    It is never cleared again once a shape's runs begin, so every launch against
    it has to carry a strictly increasing epoch -- see the counters in main().
    """
    barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int,
        local_rank=local_rank, local_world_size=world_size, multicast=True)
    barrier.data_.zero_()
    return barrier


def williams_orders(items):
    """
    Generate an interleaved schedule for the different kernels to run

    Balanced Latin square (Williams design) over `items`.

    Returns len(items) orderings in which every condition occupies every
    position exactly once and -- for an even count -- every ordered pair is
    adjacent exactly once.

    Construction: first row alternates from the ends (0, 1, n-1, 2, n-2, ...);
    every later row shifts it by one modulo n. Because all rows are the same
    row shifted, the order is identical on every rank, which it has to be --
    every condition is a collective.
    """
    n = len(items)
    first, lo, hi = [], 0, n - 1
    while lo <= hi:
        first.append(lo)
        if lo != hi:
            first.append(hi)
        lo, hi = lo + 1, hi - 1
    return [tuple(items[(v + r) % n] for v in first) for r in range(n)]


def elapsed_ms(samples):
    """Drain (start, end) cuda event pairs into per-iter wall times (ms)."""
    return [s.elapsed_time(e) for s, e in samples]


def sync_ranks():
    """Drain the local stream, then line every rank up on the host.

    Both timed loops call this before recording, so each iteration starts from
    an idle stream on every rank. Without it the conditions are not comparable:
    a back-to-back loop hides launch overhead behind the queue and lets ranks
    self-synchronize, while a loop that resets state between iters does not.
    """
    torch.cuda.synchronize()
    dist.barrier()


def median_then_max_cuda(samples, label=""):
    ordered = sorted(float(x) for x in samples)
    median = ordered[len(ordered) // 2]

    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    per_rank = [g.item() for g in gathered]

    if label and dist.get_rank() == 0:
        joined = " ".join(f"r{i}={v:.3f}" for i, v in enumerate(per_rank))
        print(f"  [rank-ms] {label}: {joined}", flush=True)

    return max(per_rank)


########## CUTLASS compatability layer ##########
_ENV_ROOT = "CUTLASS_PATH"
_ENV_AUTOTUNE = "CUTLASS_AUTOTUNE"
_REL_DIR = "examples/python/CuTeDSL/cute/blackwell/kernel/distributed"

# Tile geometry is pinned to match our kernel so the comparison is like for
# like: 256x256 MMA tile, 2-CTA cluster, TMA store.
MMA_TILER_MN = (256, 256)
CLUSTER_SHAPE_MN = (2, 1)
USE_2CTA_INSTRS = True
USE_TMA_STORE = True

# Autotune grid. Keep cutlass to same tile size
AUTOTUNE_SWIZZLES = (1, 2, 4, 8)
AUTOTUNE_RASTERS = ("m", "n")

# Only the NVLS path is wired up: multimem.ld_reduce + multimem.st,
# reduce-scatter shaped, the same primitives as our kernel, so the ratio
# isolates our implementation rather than our choice of algorithm. _SPEC stays
# keyed by variant so another upstream example can be dropped in beside it.
VARIANTS = ("ldmc",)
DEFAULT_VARIANT = "ldmc"

_SPEC = {
    "ldmc": dict(
        env="CUTLASS_DGEMM_AR",
        filename="distributed_gemm_all_reduce_blackwell.py",
        cls="Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel",
        all_reduce="LDMCxSTMC",
        # One workspace: the LDMC path writes and reduces in place, so there is
        # no slot to rotate.
        num_workspace=1,
    )
}

_module = {}
_tensor_cache = {}


def _check(variant):
    if variant not in _SPEC:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return _SPEC[variant]


def _locate(variant):
    spec = _check(variant)
    p = os.environ.get(spec["env"])
    if p:
        return Path(p)
    root = os.environ.get(_ENV_ROOT)
    if root:
        return Path(root) / _REL_DIR / spec["filename"]
    return None


def _load(variant=DEFAULT_VARIANT):
    if variant not in _module:
        spec = _check(variant)
        path = _locate(variant)
        if path is None:
            raise RuntimeError(
                f"set {spec['env']} to {spec['filename']}, "
                f"or {_ENV_ROOT} to a cutlass checkout")
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        loader = importlib.util.spec_from_file_location(
            f"cutlass_dgemm_ar_example_{variant}", path)
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        _module[variant] = mod
    return _module[variant]


def cutlass_availability(variant=DEFAULT_VARIANT):
    """(ok, reason). Cheap enough to call before every shape."""
    try:
        _load(variant)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def matched_swizzle(M):
    """Mirror of our SUPERGROUP_WIDTH heuristic in gemm_ar_blackwell.cuh.

    CUTLASS's swizzle_size and our SUPERGROUP_WIDTH are the same quantity in the
    same units -- how many cluster-columns are grouped before the walk advances
    in M. CUTLASS defaults it to 1 (no grouping), so leaving it alone compares
    our L2-aware tile walk against an unswizzled one.
    """
    return 4 if M <= 4096 else 8


def _tensors(ex, variant, M, N, K, rank, world_size, device):
    """Allocate once per (variant, shape) and reuse across autotune candidates.

    A/B/C do not depend on the schedule, and upstream sizes the flag buffer for
    the worst case precisely so it can be shared -- so reallocating per
    candidate would just burn several GB at M=N=32768 for nothing.
    """
    import inspect
    import cutlass
    key = (variant, M, N, K)
    if key not in _tensor_cache:
        want = dict(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            # Our A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major.
            a_major="k",
            b_major="n",
            c_major="n",
            num_workspace=_SPEC[variant]["num_workspace"],
            device=device,
            # "test" fills the workspace with the random data upstream's own
            # verifier expects; "benchmark" arms it the way a timed run needs.
            slot_init_mode="benchmark",
            global_rank=rank,
            local_rank=device,
            world_size=world_size,
        )
        # Upstream is free to add or drop allocate_tensors keywords between
        # tags. Pass the intersection rather than a hardcoded list that rots
        # silently, and fail loudly if a required one is left unfilled.
        accepted = inspect.signature(ex.allocate_tensors).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
            want = {k: v for k, v in want.items() if k in accepted}
        missing = [name for name, p in accepted.items()
                   if p.default is p.empty
                   and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                   and name not in want]
        if missing:
            raise RuntimeError(
                f"cutlass[{variant}] allocate_tensors needs arguments this "
                f"adapter does not supply: {missing}")
        _tensor_cache[key] = ex.allocate_tensors(**want)
    return _tensor_cache[key]


def cutlass_release():
    """Drop every cached allocation.

    The bench walks shapes largest-last and holds its own A/B/C alongside,
    while the cached A/B/C plus the comm in/out buffers run to GBs at
    M=N=32768. Nothing here survives a shape, so hand it back.
    """
    _tensor_cache.clear()


def _ldmc_kwargs(tensors, stream):
    return dict(
        a=tensors["cute_tensor_a_list"][0],
        b=tensors["cute_tensor_b_list"][0],
        c=tensors["cute_tensor_c"],
        comm_in_multicast_tensor=tensors["cute_tensor_comm_in_mc"],
        comm_out_multicast_tensor=tensors["cute_tensor_comm_out_mc"],
        barrier_flag_unicast=tensors["cute_tensor_flag_unicast"],
        barrier_flag_multicast=tensors["cute_tensor_flag_multicast"],
        stream=stream,
    )


def _compile_one(ex, tensors, *, variant, M, N, K, rank, world_size,
                 swizzle_size, raster_order):
    """Compile one candidate. Returns a launcher, or None if unsupported."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.testing as testing
    import cutlass.utils as utils

    spec = _SPEC[variant]
    kernel = getattr(ex, spec["cls"])(
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.BFloat16,
        use_2cta_instrs=USE_2CTA_INSTRS,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_tma_store=USE_TMA_STORE,
        rank_id=rank,
        num_ranks=world_size,
        all_reduce=spec["all_reduce"],
        swizzle_size=swizzle_size,
        raster_order=raster_order,
    )

    # Upstream raises CantImplementError for combinations the kernel rejects --
    # notably num_clusters_n % swizzle_size != 0 under raster_order="m".
    try:
        kernel.can_implement(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            a_major="k",
            b_major="n",
            c_major="n",
        )
    except testing.CantImplementError:
        return None

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        CLUSTER_SHAPE_MN[0] * CLUSTER_SHAPE_MN[1])

    kwargs = _ldmc_kwargs(tensors, stream)
    compiled = cute.compile(kernel, **kwargs,
                            max_active_clusters=max_active_clusters)

    def launch():
        compiled(**kwargs)

    # The compiled kernel holds raw pointers into these, so the launcher has to
    # outlive nothing else -- but the tensors have to outlive the launcher.
    launch._keepalive = tensors
    launch.config = (swizzle_size, raster_order)
    return launch


def _time(launch, iters=5):
    """Median launch time in ms, agreed across ranks (max, as for a collective)."""
    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        samples.append((s, e))
    torch.cuda.synchronize()

    ms = sorted(s.elapsed_time(e) for s, e in samples)[len(samples) // 2]
    t = torch.tensor([ms], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def cutlass_build(*, M, N, K, rank, world_size, device, variant=DEFAULT_VARIANT,
          swizzle_size=None, raster_order="m", autotune=None):
    """Compile one variant for one shape. Returns a zero-arg launcher.

    Autotunes by default: every (swizzle, raster) pair is compiled and timed
    and the fastest is kept, so CUTLASS is benched at its best schedule for the
    shape rather than at whatever we guessed for it. All ranks evaluate the same
    candidates in the same order and agree on the winner by all-reducing the
    timings, which they must -- these are collectives. Set CUTLASS_AUTOTUNE=0 to
    skip it and take swizzle_size directly.

    With autotune off, swizzle_size=None falls back to matched_swizzle(M), i.e.
    the same tile-grouping width our kernel uses -- without it CUTLASS would run
    unswizzled and the comparison would not be like for like.
    """
    _check(variant)
    ex = _load(variant)
    tensors = _tensors(ex, variant, M, N, K, rank, world_size, device)

    if autotune is None:
        autotune = os.environ.get(_ENV_AUTOTUNE, "1") not in ("0", "", "false")

    def compile_at(swz, raster):
        return _compile_one(ex, tensors, variant=variant, M=M, N=N, K=K,
                            rank=rank, world_size=world_size,
                            swizzle_size=swz, raster_order=raster)

    if not autotune:
        if swizzle_size is None:
            swizzle_size = matched_swizzle(M)
        launch = compile_at(swizzle_size, raster_order)
        if launch is None:
            raise RuntimeError(
                f"cutlass[{variant}] rejects swizzle_size={swizzle_size} "
                f"raster_order={raster_order} at M={M} N={N}")
        return launch

    best, best_ms = None, float("inf")
    tried = []
    for raster in AUTOTUNE_RASTERS:
        for swz in AUTOTUNE_SWIZZLES:
            cand = compile_at(swz, raster)
            if cand is None:
                continue
            ms = _time(cand)
            tried.append((swz, raster, ms))
            if ms < best_ms:
                best, best_ms = cand, ms
    if best is None:
        raise RuntimeError(
            f"no cutlass[{variant}] config is implementable at M={M} N={N}")
    best.autotune_log = tried
    return best


########## bench ##########

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = local_rank == 0
    mod = load_module.load("gemm_ar_blackwell")

    NUM_DEVICES = dist.get_world_size() if dist.is_initialized() else 8

    for n in SHAPES:
        M, K, N = n, n // NUM_DEVICES, n

        torch.manual_seed(42 + rank); torch.cuda.manual_seed(42 + rank)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_dbuf.data_.zero_()

        barrier = make_barrier(mod, local_rank, world_size)

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        dist.barrier()

        C_ref = torch.matmul(A, B).detach().float()
        local_ref = C_ref.clone()
        dist.all_reduce(C_ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Barrier was just zeroed, so this run is epoch 1.
        sync_ranks()
        mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final, 1)
        torch.cuda.synchronize()

        # C_dbuf is the local GEMM slice and C_final the all-reduced output, so
        # they are checked against the local and the all-reduced reference
        # respectively -- which isolates a comp-side bug from a comm-side one.
        gemm_ok = check_close(f"gemm M={M}", C_dbuf.data_, local_ref)
        ar_ok = check_close(f"gemm_ar_blackwell M={M}", C_final.data_, C_ref,
                            atol=0.55, rtol=0.12)

        if not gemm_ok:
            if is_chief:
                print(f"{M=} GEMM error :(")
            dist.destroy_process_group()
            return 1
        elif not ar_ok:
            if is_chief:
                print(f"{M=} AR Error :(((")
            dist.destroy_process_group()
            return 1

        if is_chief:
            print(f"{M=} correct :)")

        del C_dbuf, C_final, barrier, A, B, C_ref, local_ref
        dist.barrier()

    if is_chief:
        print("Correctness checks passed, benchmarking now...")

    for n in SHAPES:
        M, K, N = n, n // NUM_DEVICES, n

        torch.manual_seed(42 + rank); torch.cuda.manual_seed(42 + rank)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_dbuf.data_.zero_()

        barrier = make_barrier(mod, local_rank, world_size)

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        # NVIDIA's CuTeDSL GEMM+AR example, as a second external reference next
        # to cuBLAS+NCCL. It brings its own symmetric-memory buffers, so it is
        # timed but not correctness-checked here -- the upstream example checks
        # itself against the same dist.all_reduce reference we use.
        ok, why = cutlass_availability(DEFAULT_VARIANT)
        cutlass_run = None
        if ok:
            try:
                cutlass_run = cutlass_build(M=M, N=N, K=K, rank=rank,
                                            world_size=world_size,
                                            device=local_rank)
            except Exception as exc:
                ok, why = False, f"{type(exc).__name__}: {exc}"
        # Every rank must agree on whether cutlass is in, or the interleave
        # would put one rank in a collective the others are not running.
        # cutlass_availability() is local, so vote it to unanimous.
        vote = torch.tensor([1 if ok else 0], device="cuda")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        if not vote.item():
            cutlass_run = None
            if is_chief:
                print(f"  [skip] cutlass[{DEFAULT_VARIANT}] CuTeDSL GEMM+AR: "
                      f"{why or 'peer opted out'}", flush=True)
        elif is_chief:
            swz, raster = cutlass_run.config
            log = getattr(cutlass_run, "autotune_log", None)
            note = "autotuned" if log else "matched to SUPERGROUP_WIDTH"
            print(f"  cutlass[{DEFAULT_VARIANT}] config: swizzle_size={swz} "
                  f"raster_order={raster} ({note})", flush=True)
            # Fastest first, with the schedule our kernel uses marked, so it is
            # visible how much the tuned pick beat the like-for-like one by --
            # if it is a wash, the matched config was already the right
            # comparison and the autotune only cost time.
            for cswz, craster, cms in sorted(log or [], key=lambda r: r[2]):
                mark = " <- best" if (cswz, craster) == (swz, raster) else ""
                if (cswz, craster) == (matched_swizzle(M), "m"):
                    mark += " (matches our SUPERGROUP_WIDTH)"
                print(f"    [autotune] swizzle={cswz} raster={craster}: "
                      f"{cms:8.3f} ms{mark}", flush=True)

        # Interleave the conditions, ensure that interleaved order is the same among ranks
        BASELINE = ("baseline",)
        FUSED = ("fused",)
        CUTLASS = ("cutlass",)
        conditions = [BASELINE, FUSED]
        if cutlass_run is not None:
            conditions.append(CUTLASS)
        ORDERS = williams_orders(conditions)
        # Round the target iteration count to a whole number of orders so the
        # balancing is exact rather than approximate.
        iterations = max(1, round(BENCH_ITER / len(ORDERS))) * len(ORDERS)

        # The epoch is carried across warmup and timing because the barrier is
        # never cleared -- see make_barrier.
        epoch = 0

        def launch(cond):
            """Run one condition once, untimed.

            The single place a condition's launch lives, so the warmup and the
            timed loop cannot drift apart.
            """
            nonlocal epoch
            if cond == BASELINE:
                C_tmp = torch.matmul(A, B)
                dist.all_reduce(C_tmp)
                # C_tmp dies with this frame, so the baseline's output is back
                # in the allocator before the next condition runs rather than
                # sitting on the device through it -- at M=N=32768 that is
                # another 2 GB of headroom. The caching allocator is
                # stream-ordered, so releasing it before the work it belongs to
                # completes is safe.
            elif cond == CUTLASS:
                cutlass_run()
            else:
                epoch += 1
                mod.gemm_ar_intranode_blackwell(
                    A, B, C_dbuf, barrier, C_final, epoch)

        # Drain, line the ranks up, then soak. The sleep only resets temperature
        # if the GPU is already idle when it starts, so it has to come after the
        # synchronize -- otherwise the queue is still draining through it.
        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # Warm on the same rotation the timed loop uses. Every condition is
        # warmed before ANY of them is timed, so none pays another's cold-start
        # cost once the measured loop begins -- and because the rotation is the
        # same, no condition ends the warmup with a cache or clock state the
        # others did not also get a turn at.
        for it in range(WARMUP):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                launch(cond)

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        samples = {c: [] for c in conditions}

        for it in range(iterations):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                launch(cond)
                e.record()
                samples[cond].append((s, e))

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # events are only readable once the stream has drained
        if is_chief:
            print(f"M={M} K={K} N={N}", flush=True)

        baseline_ms = median_then_max_cuda(
            elapsed_ms(samples[BASELINE]), label="cublas+nccl")
        fused_ms = median_then_max_cuda(
            elapsed_ms(samples[FUSED]), label="mkernel")
        cutlass_ms = (
            median_then_max_cuda(elapsed_ms(samples[CUTLASS]),
                                 label=f"cutlass[{DEFAULT_VARIANT}]")
            if cutlass_run is not None else None)

        # 2*M*K*N per rank for the local GEMM slice
        flops = 2.0 * M * K * N
        if is_chief:
            def tflops(ms):
                return flops / (ms * 1e9) if ms > 0 else float("nan")

            print(f"  {'cublas+nccl':<16}: {baseline_ms:8.3f} ms  "
                  f"({tflops(baseline_ms):7.1f} TFLOP/s)", flush=True)
            if cutlass_ms is not None:
                print(f"  {'cutlass[' + DEFAULT_VARIANT + ']':<16}: "
                      f"{cutlass_ms:8.3f} ms  ({tflops(cutlass_ms):7.1f} TFLOP/s)  "
                      f"{baseline_ms / cutlass_ms:6.3f}x vs cublas+nccl",
                      flush=True)
            line = (f"  {'mkernel':<16}: {fused_ms:8.3f} ms  "
                    f"({tflops(fused_ms):7.1f} TFLOP/s)  "
                    f"{baseline_ms / fused_ms:6.3f}x vs cublas+nccl")
            if cutlass_ms is not None and fused_ms > 0:
                verdict = "BEATS" if fused_ms < cutlass_ms else "behind"
                line += (f"  {cutlass_ms / fused_ms:6.3f}x vs "
                         f"cutlass[{DEFAULT_VARIANT}] ({verdict} it by "
                         f"{abs(1 - cutlass_ms / fused_ms) * 100:.1f}%)")
            print(line, flush=True)

        # Rebound to None rather than `del`eted: launch() closes over these,
        # and `del` on a captured name empties the cell, so a stray later call
        # would raise NameError rather than simply be wrong. Both free the
        # memory, which is the point -- the next shape is up to 4x larger.
        A = B = C_dbuf = barrier = C_final = None
        cutlass_run = None
        cutlass_release()
        dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
