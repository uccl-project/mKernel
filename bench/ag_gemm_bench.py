"""ag_gemm All-Gather + GEMM bench (release version).

Hopper Benchmarks (supports intranode and internode):
Default EFA sweep: M∈{4096,8192,16384,24576,32768}.
SM split: --num-comm-sms 64 (50/50 split intra/inter inside the kernel).

Blackwell Benchmarks (only intranode supported):
Sweep Attention Projections: M∈{2048,3072,3584,4096,8192,16384,32768}, K=7168, N∈{6284,3648}
To include cutlass and TK's distributed kernels, specify the following environment variables:

CUTLASS_PATH=<path to cutlass root folder>
THUNDERKITTENS_PATH=<path to TK root folder>

E.g. THUNDERKITTENS_PATH=/home/ThunderKittens CUTLASS_PATH=/home/cutlass python -m torch.distributed.run \
    --standalone --nproc-per-node=8 ag_gemm_bench.py --arch blackwell --intranode-only
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol

os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
import load_thirdparty  #noqa: E402
from common import (  # noqa: E402
    check_close,
    check_deterministic_rerun,
    compare_named_results,
    gather_cpu_tensors,
    get_peer_ips,
    get_peer_ports,
    is_peermem_backing,
    make_dist_buffer,
    rdma_backing,
    rdma_policy_label,
)

from common import get_num_nodes  # noqa: E402

class HopperBenchConfig:
    arch = "hopper"
    kernel_name = "ag_gemm"
    num_nodes = get_num_nodes()
    row_block = 128
    col_block = 256
    red_block = 64
    chunk_bytes = 64 * 1024  # baked from AG_CHUNK_BYTES=65536

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    node_idx = int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)
    peer_ip = os.environ.get("PEER_IP")
    use_ngt2_fallback = os.environ.get("MKERNEL_AG_GEMM_USE_TORCH_FALLBACK") == "1"
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    global_world = num_nodes * world_size
    global_gpu_idx = node_idx * world_size + local_rank
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank
    # Canonical: NCCL-style no-sync timing — N back-to-back iters with a
    # SINGLE sync after, divide by N. Mirrors nccl_16gpu_baseline.py's
    # default --steady-state. Set MKERNEL_BENCH_LEGACY_SYNC=1 to opt
    # back into per-iter sync (kept for A/B and source-of-truth debugging).
    legacy_sync = os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1"
    # Back-compat: MKERNEL_BENCH_NO_SYNC=0 also forces legacy.
    if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
        legacy_sync = True

    # Per-shape num_comm_sms override. Smaller values reduce coordination overhead
    # at small M (NCCL has minimal launch overhead and beats the fused path there
    # unless we cut the comm-CTA budget). The 64-sms default oversubscribes comm
    # CTAs at medium M where the GEMM wave count is lower.
    sm_per_shape = {4096: 8, 6144: 8, 8192: 8, 12288: 8, 16384: 8,  # noqa: RUF012
                 24576: 8, 49152: 8, 65536: 8, 57344: 8, 73728: 8}
    # M=57344 hangs during default-warmup 4-node sweeps on H200x.
    # 4-node M=65536 sits at the u32 overflow boundary for the per-peer
    # offset fix (a_half_bytes = M²/2 = 2 GiB; sap=2 * 2 GiB = 4 GiB).
    # M=98304 pushes past the boundary to confirm the u64 path is correct
    # at payloads above 4 GiB.
    shapes_to_test = (
        [8192, 16384, 32768, 49152, 65536, 98304]
        if num_nodes == 4 else
        [6144, 12288, 24576, 49152, 73728]
        if num_nodes == 3 else
        [4096, 8192, 16384, 24576, 32768]
    )

class DistBufferLike(Protocol):
    """Structural type for mod.DistBuffer, which is loaded dynamically so there's no static class to import."""
    data_: torch.Tensor

@dataclass
class HopperBenchVars:
    """Kernel-input state for one hopper shape; reset_state()/run_once() mutate these in place each iter."""
    a_tk: DistBufferLike | None = None
    start_row: int | None = None
    a_recv_tk: DistBufferLike | None = None
    barrier: DistBufferLike | None = None
    epoch: int | None = None
    a_rdma_src: DistBufferLike | None = None
    a_recv_rdma: DistBufferLike | None = None

class BlackwellBenchConfig:
    arch = "blackwell"
    kernel_name = "ag_gemm_warp_specialized"
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    is_chief = local_rank == 0
    num_nodes = 1

    projections = ( 
        # KDA proj_qkvgfab, (4 * 12288 + 96) / TP + 128.
        ("KDA", (4 * 12288 + 96) // world_size + 128),
        # MLA qkvg proj, 576 + 1536 + 12288 / TP.
        ("MLA", 576 + 1536 + 12288 // world_size),
    )
    shapes_to_test = [2048, 3072, 3584, 4096, 8192, 16384, 32768]
    default_k = 7168
    
    # mkernel configs
    mkernel_per_shape_config = {  # noqa: RUF012
        # (projection, logical global M): (COL_BLOCK, NUM_CTA)
        ("KDA", 2048): (128, 2),
        ("KDA", 3072): (256, 2),
        ("KDA", 3584): (256, 2),
        ("KDA", 4096): (256, 2),
        ("KDA", 8192): (256, 2),
        ("KDA", 16384): (256, 2),
        ("KDA", 32768): (256, 2),
        ("MLA", 2048): (128, 2),
        ("MLA", 3072): (128, 1),
        ("MLA", 3584): (256, 2),
        ("MLA", 4096): (256, 2),
        ("MLA", 8192): (256, 2),
        ("MLA", 16384): (256, 2),
        ("MLA", 32768): (256, 2),
    }
    # cutlass configs
    cutlass_autotune_configs = (
        ((256, 128), (2, 1), True),
        ((256, 256), (2, 1), True),
        ((128, 128), (1, 1), False),
        ((128, 256), (1, 1), False),
    )
    # tk configs
    tk_tile_granularity = 256
    tk_comm_sms = (2, 4, 8, 16, 32, 64)

@dataclass
class BlackwellBenchVars:
    """Per-(projection, M) kernel-input state, filled in progressively as each candidate is set up -- every field defaults to None so BlackwellBenchVars() can be built empty and filled in later."""
    logical_m: int | None = None
    logical_n: int | None = None

    # per kernel args could either be fresh tensors or pointers to the same underlying tensor,
    # depending on padding
    # baseline args
    baseline_padded_m: int | None = None
    baseline_padded_n: int | None = None
    baseline_a: torch.Tensor | None = None
    baseline_a_buf: torch.Tensor | None = None
    baseline_b: torch.Tensor | None = None
    baseline_c: torch.Tensor | None = None

    # mkernel specific args
    mkernel_padded_m: int | None = None
    mkernel_padded_n: int | None = None
    mkernel_a_dist: DistBufferLike | None = None
    mkernel_a_local_buf: torch.Tensor | None = None
    mkernel_b_buf: torch.Tensor | None = None
    mkernel_c_buf: torch.Tensor | None = None

    # TK specific args
    tk_padded_m: int | None = None
    tk_padded_n: int | None = None
    tk_a_dist: DistBufferLike | None = None
    tk_b_transposed: torch.Tensor | None = None
    tk_c_buf: torch.Tensor | None = None
    tk_barrier: DistBufferLike | None = None

CONFIGS = {
    "hopper": HopperBenchConfig,
    "blackwell": BlackwellBenchConfig
}

def avg_then_max_cuda(samples):
    # Median-then-max for robustness against outlier iters (matches gemm_rs).
    median = sorted(float(x) for x in samples)[len(samples) // 2]
    if os.environ.get("MKERNEL_BENCH_DUMP_RANK_MS") == "1":
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, {
            "rank": dist.get_rank(),
            "ms": median,
            "host": os.uname().nodename,
        })
        if dist.get_rank() == 0:
            print(f"[ag_gemm-rank-ms] {gathered}", flush=True)
    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def benchmark_cuda(
    run_once: Callable[[], None], warmup: int, iters: int
) -> float:
    """Return average CUDA time in ms, taking the slowest rank's result."""
    for _ in range(warmup):
        run_once()

    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / iters

    # End-to-end distributed latency is gated by the slowest rank. This
    # reduction is outside the timed region for both implementations.
    rank_ms = torch.tensor(local_ms, device="cuda", dtype=torch.float64)
    dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
    dist.barrier()
    return float(rank_ms.item())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str, default=None,
                   help="Comma-separated M values; defaults to the arch's shapes_to_test.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--num-comm-sms", type=int, default=64)
    p.add_argument("--num-intra-comm-sms", type=int, default=0)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    p.add_argument("--arch", type=str, default="hopper", choices=["hopper", "blackwell"])
    p.add_argument("--intranode-only", action='store_true', default=False)

    args = p.parse_args()
    if args.arch == "blackwell":
        assert args.intranode_only, "Blackwell kernel only supports intranode"
    return args

def ag_gemm_hopper_prepare(config: HopperBenchConfig, mod, base_n: int, source_backing: str, num_comm_sms: int, intra_comm_sms: int, iters: int) -> tuple[Callable[[], list[float] | None], str, bool, Callable[[], bool], Callable[[], bool]]: 
    """
    Setup performs correctness checks and sets up functions to bench
    """
    M, K, N = base_n, base_n, base_n // config.global_world
    M_node = M // config.num_nodes
    M_local = M_node // config.world_size
    assert M % config.row_block == 0 and K % config.red_block == 0 and N % config.col_block == 0
    os.environ.pop("AG_GEMM_ROW_STRIDE_BYTES", None)
    n_peers = config.num_nodes - 1
    ring_recv_banks = n_peers

    if config.is_chief:
        print(f"\n[ag_gemm] M={M} K={K} N={N} M_node={M_node} M_local={M_local}", flush=True)

    run_config = HopperBenchVars()

    torch.manual_seed(42 + config.global_gpu_idx); torch.cuda.manual_seed(42 + config.global_gpu_idx)
    A_local = torch.randn((M_local, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
    torch.manual_seed(100); torch.cuda.manual_seed(100)
    B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
    C_tmp = torch.zeros((M_local, N), device="cuda", dtype=torch.bfloat16)    

    if config.use_ngt2_fallback:
        if config.is_chief:
            print("[ag_gemm] using explicit torch fallback; unset "
                "MKERNEL_AG_GEMM_USE_TORCH_FALLBACK to test fused path",
                flush=True)

        def bench_ngt2_fallback():
            torch.matmul(A_local, B, out=C_tmp)

        # No all-gather happens here, so there's nothing to check against a
        # reference -- same as the original torch-fallback path, which just
        # timed and reported without a correctness check.
        return (bench_ngt2_fallback, "torch_fallback", True, lambda: True, lambda: True)
    else:
        run_config.a_tk = mod.DistBuffer(
            (M_node, K), dtype=torch.bfloat16,
            local_rank=config.local_rank, local_world_size=config.world_size, multicast=True
        )
        run_config.start_row = config.local_rank * M_local
        run_config.a_tk.data_[run_config.start_row:run_config.start_row + M_local].copy_(A_local)
        run_config.a_rdma_src = None
        if is_peermem_backing(source_backing):
            run_config.a_rdma_src = make_dist_buffer(
                mod, (M_node, K), dtype=torch.bfloat16,
                local_rank=config.local_rank, local_world_size=config.world_size,
                multicast=False, backing=source_backing)
            run_config.a_rdma_src.data_.zero_()
            run_config.a_rdma_src.data_[run_config.start_row:run_config.start_row + M_local].copy_(A_local)

        run_config.a_recv_tk = mod.DistBuffer((M_node * n_peers * ring_recv_banks, K), dtype=torch.bfloat16,
                        local_rank=config.local_rank, local_world_size=config.world_size, multicast=True)
        run_config.a_recv_tk.data_.zero_()
        a_recv_rdma = None
        target_backing = source_backing
        if is_peermem_backing(target_backing):
            a_recv_rdma = make_dist_buffer(
                mod, (M_node * n_peers * ring_recv_banks, K),
                dtype=torch.bfloat16,
                local_rank=config.local_rank, local_world_size=config.world_size,
                multicast=False, backing=target_backing)
            a_recv_rdma.data_.zero_()

        run_config.barrier = mod.DistBuffer((3, 1024, 1024), dtype=torch.int,
                       local_rank=config.local_rank, local_world_size=config.world_size, multicast=True)
        run_config.barrier.data_.zero_()

        C = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)        
        a_half_bytes = M_node * K * 2
        total_chunks = (a_half_bytes + config.chunk_bytes - 1) // config.chunk_bytes

        # Per-peer recv_buf / arrival flag scaling. At N == 2 n_peers == 1,
        # so this collapses to a single peer-sized slot.
        recv_buf_bytes = n_peers * a_half_bytes * ring_recv_banks
        recv_buf_chunks = n_peers * total_chunks * ring_recv_banks

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_chunks * 2: fifo_cap *= 2
        a_tk_ptr = int((run_config.a_rdma_src if run_config.a_rdma_src is not None else run_config.a_tk).data_.data_ptr())
        send_buf_ptr = int((a_recv_rdma if a_recv_rdma is not None else run_config.a_recv_tk).data_.data_ptr())
        send_buf_size = recv_buf_bytes
        # A_recv is registered as MR0 (src_view=0) so received shards can be
        # forwarded to the next node after phase-2 republishes them.
        peer_ips = get_peer_ips(config.node_idx, config.num_nodes)
        mod.create_session(
            config.node_idx, config.peer_ip, config.tcp_port,
            send_buf_ptr, send_buf_size, recv_buf_bytes,
            recv_buf_chunks, fifo_cap, config.local_rank,
            clocal_buf_ptr=a_tk_ptr, clocal_buf_size=a_half_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(config.node_idx, config.num_nodes, config.tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        run_config.epoch = 1
        mod.set_epoch(run_config.epoch)
        dist.barrier(); time.sleep(0.5)

        def reset_state():
            run_config.barrier.data_.zero_()
            run_config.a_tk.data_[run_config.start_row:run_config.start_row + M_local].copy_(A_local)
            if run_config.a_rdma_src is not None:
                run_config.a_rdma_src.data_.zero_()
                run_config.a_rdma_src.data_[run_config.start_row:run_config.start_row + M_local].copy_(A_local)
            run_config.a_recv_tk.data_.zero_()
            if a_recv_rdma is not None:
                a_recv_rdma.data_.zero_()
            C.zero_()

        def _bench_mkernel():
            default_active_sms = torch.cuda.get_device_properties(config.local_rank).multi_processor_count
            active_sms = int(os.environ.get("AG_GEMM_ACTIVE_SMS", str(default_active_sms)))
            mod.ag_gemm_multinode(
                run_config.a_tk, B, C, run_config.barrier,
                recv_ptr,
                int(fifo[0]), int(fifo[1]), int(fifo[2]), int(fifo[3]), int(fifo[4]),
                arrival_ptr, run_config.epoch, config.node_idx, num_comm_sms, a_half_bytes,
                run_config.a_recv_tk, active_sms, intra_comm_sms, config.num_nodes,
            )

        def check():
            gathered_a = gather_cpu_tensors(A_local)
            A_ref = torch.cat(gathered_a, dim=0).to(device="cuda")
            C_ref = torch.matmul(A_ref, B)
            if config.is_chief:
                rows_per_node = M // config.num_nodes
                for nr in range(config.num_nodes):
                    lo = nr * rows_per_node
                    hi = lo + rows_per_node
                    shard_abs = (C[lo:hi].float() - C_ref[lo:hi].float()).abs().max().item()
                    print(f"[ag_gemm-correctness] node_shard={nr} max_abs={shard_abs:.6f}",
                        flush=True)
            return check_close(
                f"ag_gemm M={M}", C, C_ref, atol=0.45, rtol=0.10
            )

        def invariant_check():
            torch.cuda.synchronize(); dist.barrier(); time.sleep(0.1)
            reset_state(); run_config.epoch += 1; mod.set_epoch(run_config.epoch)
            dist.barrier(); time.sleep(0.05)
            _bench_mkernel(); torch.cuda.synchronize()
            det_out_a = C.detach().clone()
            reset_state(); run_config.epoch += 1; mod.set_epoch(run_config.epoch)
            dist.barrier(); time.sleep(0.05)
            _bench_mkernel(); torch.cuda.synchronize()
            det_out_b = C.detach().clone()
            return check_deterministic_rerun(
                f"ag_gemm M={M}", det_out_a, det_out_b, config.is_chief
            )


        if config.legacy_sync or config.num_nodes > 2:
            # Legacy: full reset + fresh epoch + individual per-iter event
            # timing. Slower, but kept for A/B and source-of-truth debugging,
            # and forced on for >2 nodes (multi-node timing needs the
            # per-iter barrier/sleep to stay stable -- see the M=57344 hang
            # note above).
            def bench_mkernel():
                samples = []
                for _ in range(iters):
                    reset_state(); run_config.epoch += 1; mod.set_epoch(run_config.epoch)
                    dist.barrier(); time.sleep(0.05)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(); _bench_mkernel(); e.record(); torch.cuda.synchronize()
                    samples.append(s.elapsed_time(e))
                    dist.barrier()
                return samples

            return (bench_mkernel, "mKernel", False, check, invariant_check)
        else:
            # Canonical: NCCL-style no-sync timing -- N back-to-back iters
            # under one epoch, single sync after, divide by N.
            def bench_mkernel():
                n_iters = max(iters, 32)
                # Pre-flip enough epochs so all N iters have unique epochs without
                # an inter-iter set_epoch that would re-issue prepare_epoch's
                # drain_proxy + barrier (which is itself a sync). We reuse a single
                # epoch across the back-to-back run; reset_state restores buffers.
                reset_state(); run_config.epoch += 1; mod.set_epoch(run_config.epoch)
                dist.barrier(); time.sleep(0.05)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                for _ in range(n_iters):
                    _bench_mkernel()
                e.record()
                torch.cuda.synchronize()
                avg_ms = s.elapsed_time(e) / n_iters
                samples = [avg_ms] * iters  # reuse downstream reduce path
                if config.is_chief:
                    print(f"[ag_gemm-nosync] M={M} N={n_iters} avg={avg_ms:.4f} ms",
                        flush=True)
                dist.barrier()
                return samples

            return (bench_mkernel, "mKernel", False, check, invariant_check)

def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple

def gemm_tflops(m: int, n: int, k: int, elapsed_ms: float) -> float:
    """Return GEMM throughput using 2*M*N*K floating-point operations."""
    return 2.0 * m * n * k / (elapsed_ms * 1.0e9)

def useful_tflops(m: int, logical_n: int, k: int, elapsed_ms: float) -> float:
    """Return throughput over the logical problem only.

    Every candidate pads M and N to whatever its own tiler requires, and each
    pads by a different amount. Charging each implementation for the rows and
    columns it happened to pad rewards the one that pads most, so score all of
    them on the 2*M*logical_n*K the model actually needs, where M is the
    unpadded global sequence length.
    """
    return gemm_tflops(m, logical_n, k, elapsed_ms)

def cutlass_config_label(config: tuple[tuple[int, int], tuple[int, int], bool]) -> str:
    (tile_m, tile_n), cluster, two_cta = config
    return (
        f"mma={tile_m}x{tile_n} cluster={cluster[0]}x{cluster[1]} "
        f"{'2-CTA' if two_cta else '1-CTA'}"
    )

def cutlass_padded_shape(m: int, n: int) -> tuple[int, int]:
    """The (M, N) CUTLASS actually computes, regardless of tile config.

    CUTLASS only needs M/N aligned to 16 for good tensor-core utilization --
    unlike mkernel/TK it isn't tied to a per-config CTA tile size. Rounding
    here ourselves, rather than handing it a raw shape and trusting it to pad
    internally, is what lets us ground its TFLOP/s in a shape we actually know
    it ran.
    """
    return round_up(m, 16), round_up(n, 16)

def pad_rows(config: BlackwellBenchConfig, a_local: torch.Tensor, padded_local_m: int) -> torch.Tensor:
    """Zero-pad a rank's A shard up to padded_local_m rows."""
    local_m = a_local.size(0)
    if local_m == padded_local_m:
        return a_local
    padded = torch.zeros(
        (padded_local_m, config.default_k), device="cuda", dtype=torch.bfloat16
    )
    padded[:local_m].copy_(a_local)
    return padded

def pad_cols(config: BlackwellBenchConfig, b: torch.Tensor, padded_n: int) -> torch.Tensor:
    """Zero-pad B up to padded_n columns."""
    logical_n = b.size(1)
    if logical_n == padded_n:
        return b
    padded = torch.zeros((config.default_k, padded_n), device="cuda", dtype=torch.bfloat16)
    padded[:, :logical_n].copy_(b)
    return padded

def unpad_rows(
    c: torch.Tensor,
    local_m: int,
    padded_local_m: int,
    world_size: int,
    logical_n: int,
) -> torch.Tensor:
    """Return the logical [M, logical_n] block of a row/column padded C.

    Each rank contributes padded_local_m rows to the all-gathered output but
    only the first local_m of them carry real data, so the padding rows sit
    between rank shards rather than after the last one.
    """
    if padded_local_m == local_m:
        return c[:, :logical_n]
    rows = c.view(world_size, padded_local_m, -1)[:, :local_m, :logical_n]
    return rows.reshape(world_size * local_m, logical_n)

def check_correctness_ag_gemm_blackwell(config: BlackwellBenchConfig, mod):
    all_correct = True
    for (projection, logical_n), m in product(config.projections, config.shapes_to_test):
        if m % config.world_size != 0:
            raise ValueError(f"global M={m} is not divisible by {config.world_size=}")

        local_m = m // config.world_size
        col_block, num_cta = config.mkernel_per_shape_config[(projection, m)]
        padded_local_m = round_up(local_m, 128 * num_cta)
        padded_m = padded_local_m * config.world_size
        padded_n = round_up(logical_n, col_block)

        torch.manual_seed(42 + config.local_rank)
        torch.cuda.manual_seed(42 + config.local_rank)
        A_ref_local = torch.randn(
            (local_m, config.default_k), device="cuda", dtype=torch.bfloat16
        ) / (config.default_k**0.25)
        A_ref = torch.empty((m, config.default_k), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (config.default_k, logical_n), device="cuda", dtype=torch.bfloat16
        ) / (config.default_k ** 0.25)
        C_ref = torch.empty(
            (m, logical_n), device="cuda", dtype=torch.bfloat16
        )

        dist.all_gather_into_tensor(A_ref, A_ref_local)
        torch.mm(A_ref, B_ref, out=C_ref)

        # The kernel gets its own tensors, padded to the granularity its
        # dispatched schedule tiles at. The padding rows and columns are zero,
        # so they contribute zero to C and cost only the tiles spent on them.
        A_kernel = mod.DistBuffer(
            (padded_local_m, config.default_k),
            dtype=torch.bfloat16,
            local_rank=config.local_rank,
            local_world_size=config.world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(pad_rows(config, A_ref_local, padded_local_m))
        A_local_buf = torch.empty(
            (padded_m, config.default_k), device="cuda", dtype=torch.bfloat16
        )
        # ag_gemm_warp_specialized takes B pre-transposed to [N, K] (contiguous K reads
        # per N-tile); see the same transform in ag_gemm_blackwell_prepare.
        B_kernel = pad_cols(config, B_ref, padded_n).T.contiguous()
        C_kernel = torch.zeros(
            (padded_m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        # The kernel's first act is to pull every peer's shard out of their
        # DistBuffer, so no rank may launch until all of them have finished
        # filling theirs -- and the fill is stream-ordered work that a bare
        # dist.barrier() does not wait on.
        torch.cuda.synchronize()
        dist.barrier()

        C_kernel.zero_()
        mod.ag_gemm_warp_specialized(A_kernel, A_local_buf, B_kernel, C_kernel, m)
        torch.cuda.synchronize()

        # Drop the padded rows and columns and compare the logical
        # M x logical_n result against the unpadded PyTorch reference.
        is_correct = check_close(
            f"ag-gemm-warp-specialized {projection} M={m} N={logical_n} "
            f"padded_m={padded_m} padded_n={padded_n}",
            unpad_rows(
                C_kernel, local_m, padded_local_m, config.world_size, logical_n
            ),
            C_ref,
        )
        all_correct = all_correct and is_correct

        if config.is_chief:
            status = "passed :)" if is_correct else "FAILED :("
            print(
                f"ag-gemm-warp-specialized {projection} M={m} local_m={local_m} N={logical_n} "
                f"padded_m={padded_m} padded_n={padded_n}: {status}",
                flush=True,
            )

        # ---- ThunderKittens: check every comm_sms candidate up front ----
        # Every value in config.tk_comm_sms gets autotuned later, so every
        # value gets verified here -- not just one fixed comm_sms -- so a
        # candidate that's wrong (not just slow) can't win the sweep and
        # get reported before anyone's checked its output.
        tk_ok, tk_why = load_thirdparty.tk_availability(config.world_size, "ag_gemm")
        tk_vote = torch.tensor([1 if tk_ok else 0], device="cuda")
        dist.all_reduce(tk_vote, op=dist.ReduceOp.MIN)
        tk_ok = bool(tk_vote.item())
        if not tk_ok:
            if config.is_chief:
                print(
                    f"[skip] ThunderKittens correctness check {projection} M={m}: "
                    f"{tk_why or 'unavailable on a peer'}",
                    flush=True,
                )
        else:
            tk_local_m = round_up(local_m, config.tk_tile_granularity)
            tk_m = tk_local_m * config.world_size
            tk_n = round_up(logical_n, config.tk_tile_granularity)

            tk_module = load_thirdparty.load_tk_extension("ag_gemm")
            A_tk = tk_module.TKParallelTensor(
                (tk_m, config.default_k),
                dtype=torch.bfloat16,
                local_rank=config.local_rank,
                local_world_size=config.world_size,
                multicast=True,
            )
            A_tk.data_[
                config.local_rank * tk_local_m : (config.local_rank + 1) * tk_local_m
            ].copy_(pad_rows(config, A_ref_local, tk_local_m))
            # The Blackwell TK kernel declares B as [N, K] and computes A @ B^T.
            B_tk_transposed = pad_cols(config, B_ref, tk_n).T.contiguous()
            tk_barrier = tk_module.TKParallelTensor(
                (2, 1024, 1024),
                dtype=torch.int,
                local_rank=config.local_rank,
                local_world_size=config.world_size,
                multicast=True,
            )

            # Same multicast-init race as mkernel above -- no rank may launch
            # until every rank has finished filling its own TKParallelTensor.
            torch.cuda.synchronize()
            dist.barrier()

            all_tk_correct = True
            for comm_sms in config.tk_comm_sms:
                # Fresh barrier and output per candidate -- reusing either
                # across launches is exactly the stale-state risk this check
                # exists to catch, not something to introduce into it.
                tk_barrier.data_.zero_()
                C_tk = torch.zeros((tk_m, tk_n), device="cuda", dtype=torch.bfloat16)
                torch.cuda.synchronize()
                dist.barrier()
                tk_module.all_gather_matmul(A_tk, B_tk_transposed, C_tk, tk_barrier, comm_sms)
                torch.cuda.synchronize()

                is_correct_tk = check_close(
                    f"ThunderKittens ag-gemm {projection} M={m} N={logical_n} "
                    f"padded_m={tk_m} padded_n={tk_n} num_comm_sms={comm_sms}",
                    unpad_rows(C_tk, local_m, tk_local_m, config.world_size, logical_n),
                    C_ref,
                )
                all_tk_correct = all_tk_correct and is_correct_tk

                if config.is_chief:
                    status = "passed :)" if is_correct_tk else "FAILED :("
                    print(
                        f"ThunderKittens {projection} M={m} local_m={local_m} N={logical_n} "
                        f"padded_m={tk_m} padded_n={tk_n} num_comm_sms={comm_sms}: {status}",
                        flush=True,
                    )
                del C_tk

            all_correct = all_correct and all_tk_correct
            del A_tk, B_tk_transposed, tk_barrier

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, A_local_buf, B_kernel, C_kernel
        dist.barrier()

    if not all_correct:
        if config.is_chief:
            print("Correctness checks failed; skipping benchmarks.", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

def report_blackwell_result(
    config: BlackwellBenchConfig, projection: str, global_m: int, logical_n: int,
    results: list[tuple[str, float]],
) -> None:
    """Print one shape's candidate timings/TFLOP-s, mirroring ag_gemm_warp_specialized_bench.py's report."""
    if not config.is_chief:
        return

    ms_by_name = dict(results)
    baseline_ms = ms_by_name.get("baseline")
    cutlass_ms = ms_by_name.get("cutlass")
    tk_ms = ms_by_name.get("TK")
    mkernel_ms = ms_by_name.get("mkernel")

    def tflops_for(ms: float) -> float:
        return useful_tflops(global_m, logical_n, config.default_k, ms)

    print(
        f"\n===== {projection} M={global_m} N={logical_n} "
        f"(TFLOP/s scored on the logical M={global_m} N={logical_n}) =====",
        flush=True,
    )

    if baseline_ms is not None:
        print(
            f"  {'baseline (NCCL + cuBLAS)':<26} {baseline_ms:8.3f} ms  "
            f"{tflops_for(baseline_ms):8.2f} TFLOP/s",
            flush=True,
        )
    if cutlass_ms is not None:
        vs_baseline = f"  ({baseline_ms / cutlass_ms:6.3f}x vs baseline)" if baseline_ms else ""
        print(
            f"  {'CUTLASS AG-GEMM':<26} {cutlass_ms:8.3f} ms  "
            f"{tflops_for(cutlass_ms):8.2f} TFLOP/s{vs_baseline}",
            flush=True,
        )
    if tk_ms is not None:
        vs_baseline = f"  ({baseline_ms / tk_ms:6.3f}x vs baseline)" if baseline_ms else ""
        print(
            f"  {'ThunderKittens AG-GEMM':<26} {tk_ms:8.3f} ms  "
            f"{tflops_for(tk_ms):8.2f} TFLOP/s{vs_baseline}",
            flush=True,
        )
    if mkernel_ms is not None:
        line = (
            f"  {'ag_gemm_warp_specialized':<26} {mkernel_ms:8.3f} ms  "
            f"{tflops_for(mkernel_ms):8.2f} TFLOP/s"
        )
        if baseline_ms is not None:
            line += f"  ({baseline_ms / mkernel_ms:6.3f}x vs baseline)"
        if cutlass_ms is not None:
            verdict = "BEATS" if mkernel_ms < cutlass_ms else "behind"
            line += f"  {cutlass_ms / mkernel_ms:6.3f}x vs CUTLASS ({verdict})"
        if tk_ms is not None:
            verdict = "BEATS" if mkernel_ms < tk_ms else "behind"
            line += f"  {tk_ms / mkernel_ms:6.3f}x vs ThunderKittens ({verdict})"
        print(line, flush=True)

def ag_gemm_blackwell_prepare(
    config: BlackwellBenchConfig, mod, projection: str, global_m: int, logical_n: int,
    warmup: int, iters: int,
) -> list[tuple[Callable[[], float | None], str, bool]]:
    """
    To be called per (projection, shape). Tunes each kernel, returning a list of
    (fn, name, should_wrap) to run for that shape. should_wrap tells main() whether
    to time fn via the generic benchmark_cuda loop (True, for a bare single-launch
    closure) or to call fn() once and use its own returned ms directly (False, for
    a candidate -- like CUTLASS -- that already times itself internally).
    """
    run_config = BlackwellBenchVars()

    assert global_m % config.world_size == 0, f"{global_m=} must be divisible by world size"
    run_config.logical_m = global_m // config.world_size
    run_config.logical_n = logical_n

    torch.manual_seed(42 + config.local_rank); torch.cuda.manual_seed(42 + config.local_rank)
    A_local = torch.randn((run_config.logical_m, config.default_k), device="cuda", dtype=torch.bfloat16) / (config.default_k ** 0.25)
    B_ref = torch.randn((config.default_k, run_config.logical_n), device="cuda", dtype=torch.bfloat16) / (config.default_k ** 0.25)

    fns = []

    # ---- baseline: NCCL all-gather + cuBLAS matmul, timed together ----
    # NOTE: even for cublas, we must pad the attention shapes to a multiple of 16, else performance will be bad
    run_config.baseline_padded_m = round_up(run_config.logical_m, 16)
    run_config.baseline_padded_n = round_up(run_config.logical_n, 16)
    run_config.baseline_a = pad_rows(config, A_local, run_config.baseline_padded_m)
    run_config.baseline_b = pad_cols(config, B_ref, run_config.baseline_padded_n)
    run_config.baseline_a_buf = torch.zeros(
        (config.world_size * run_config.baseline_padded_m, config.default_k),
        device="cuda", dtype=torch.bfloat16,
    )
    run_config.baseline_c = torch.zeros(
        (config.world_size * run_config.baseline_padded_m, run_config.baseline_padded_n),
        device="cuda", dtype=torch.bfloat16,
    )

    def bench_baseline():
        dist.all_gather_into_tensor(run_config.baseline_a_buf, run_config.baseline_a)
        torch.matmul(run_config.baseline_a_buf, run_config.baseline_b, out=run_config.baseline_c)

    fns.append((bench_baseline, "baseline", True))

    # ---- mkernel: ag_gemm_warp_specialized dispatch ----
    col_block, num_cta = config.mkernel_per_shape_config[(projection, global_m)]
    mk_local_m = round_up(run_config.logical_m, 128 * num_cta)
    mk_m = mk_local_m * config.world_size
    mk_n = round_up(run_config.logical_n, col_block)
    run_config.mkernel_padded_m = mk_m
    run_config.mkernel_padded_n = mk_n

    A_mk_local = pad_rows(config, A_local, mk_local_m)
    # ag_gemm_warp_specialized takes B pre-transposed to [N, K]: contiguous K reads per
    # N-tile match the reduction axis, instead of the [K, N] layout's strided
    # per-K-step access across N.
    run_config.mkernel_b_buf = pad_cols(config, B_ref, mk_n).T.contiguous()
    run_config.mkernel_a_dist = mod.DistBuffer(
        (mk_local_m, config.default_k), dtype=torch.bfloat16,
        local_rank=config.local_rank, local_world_size=config.world_size, multicast=True,
    )
    run_config.mkernel_a_dist.data_.copy_(A_mk_local)
    run_config.mkernel_a_local_buf = torch.empty(
        (mk_m, config.default_k), device="cuda", dtype=torch.bfloat16
    )
    run_config.mkernel_c_buf = torch.zeros((mk_m, mk_n), device="cuda", dtype=torch.bfloat16)

    # The kernel's first act is to pull every peer's shard out of their
    # DistBuffer, so no rank may launch until all of them have finished
    # filling theirs -- and the fill is stream-ordered work that a bare
    # dist.barrier() does not wait on.
    torch.cuda.synchronize()
    dist.barrier()

    tune_warmup = 2
    tune_iterations = 5

    def run_mkernel():
        mod.ag_gemm_warp_specialized(
            run_config.mkernel_a_dist, run_config.mkernel_a_local_buf,
            run_config.mkernel_b_buf, run_config.mkernel_c_buf, global_m,
        )

    fns.append((run_mkernel, "mkernel", True))

    # ---- CUTLASS: distributed_all_gather_gemm_blackwell.py ----
    cutlass_kernel_name = "distributed_all_gather_gemm_blackwell.py"
    cutlass_ok, cutlass_why = load_thirdparty.cutlass_availability(cutlass_kernel_name)
    cutlass_vote = torch.tensor([1 if cutlass_ok else 0], device="cuda")
    dist.all_reduce(cutlass_vote, op=dist.ReduceOp.MIN)
    cutlass_ok = bool(cutlass_vote.item())
    if not cutlass_ok:
        if config.is_chief:
            print(
                f"[skip] CUTLASS all-gather GEMM: "
                f"{cutlass_why or 'unavailable on a peer'}",
                flush=True,
            )
    else:
        # CUTLASS only needs M/N aligned to 16, regardless of which tile
        # config ends up winning, so this is the same for every candidate.
        cutlass_padded_m, cutlass_padded_n = cutlass_padded_shape(global_m, logical_n)

        timings = []
        best_config = None
        best_ms = float("inf")
        for cutlass_config in config.cutlass_autotune_configs:
            try:
                ms = load_thirdparty.run_cutlass_once(
                    cutlass_kernel_name,
                    m=cutlass_padded_m,
                    n=cutlass_padded_n,
                    k=config.default_k,
                    config=cutlass_config,
                    warmup=tune_warmup,
                    iterations=tune_iterations,
                )
            except Exception:
                # can_implement rejects some tile/shape pairs. Drop the candidate
                # rather than the whole shape, but only in lockstep: a rank that
                # kept a config its peers dropped would hang in the next launch.
                ms = None
            vote = torch.tensor([1 if ms is not None else 0], device="cuda")
            dist.all_reduce(vote, op=dist.ReduceOp.MIN)
            if not vote.item():
                continue
            timings.append((cutlass_config, ms))
            if ms < best_ms:
                best_config, best_ms = cutlass_config, ms

        if best_config is None:
            if config.is_chief:
                print(
                    f"[skip] CUTLASS all-gather GEMM: no config ran at "
                    f"M={global_m} N={logical_n}",
                    flush=True,
                )
        else:
            if config.is_chief:
                print(
                    f"  CUTLASS config {projection} M={global_m}: "
                    f"{cutlass_config_label(best_config)} (autotuned)",
                    flush=True,
                )
                for cand_config, tune_ms in sorted(timings, key=lambda item: item[1]):
                    mark = " <- best" if cand_config == best_config else ""
                    tune_tflops = useful_tflops(global_m, logical_n, config.default_k, tune_ms)
                    print(
                        f"    [autotune] {cutlass_config_label(cand_config)}: "
                        f"{tune_ms:8.3f} ms  {tune_tflops:8.2f} TFLOP/s{mark}",
                        flush=True,
                    )

            def bench_cutlass():
                # Upstream's run() builds a private CUDA graph per call and
                # cannot hand the tuned launcher back to us, so rebuilding it
                # many times over an outer benchmark_cuda loop is both wasted
                # work and a real hang risk. Rebuild it exactly once here,
                # driving its internal warmup/iterations with the real
                # values, and return the ms it already measured.
                return load_thirdparty.run_cutlass_once(
                    cutlass_kernel_name,
                    m=cutlass_padded_m,
                    n=cutlass_padded_n,
                    k=config.default_k,
                    config=best_config,
                    warmup=warmup,
                    iterations=iters,
                )

            fns.append((bench_cutlass, "cutlass", False))

    # ---- ThunderKittens: all_gather_matmul ----
    tk_ok, tk_why = load_thirdparty.tk_availability(config.world_size, "ag_gemm")
    tk_vote = torch.tensor([1 if tk_ok else 0], device="cuda")
    dist.all_reduce(tk_vote, op=dist.ReduceOp.MIN)
    tk_ok = bool(tk_vote.item())
    if not tk_ok:
        if config.is_chief:
            print(
                f"[skip] ThunderKittens all-gather GEMM: "
                f"{tk_why or 'unavailable on a peer'}",
                flush=True,
            )
    else:
        tk_local_m = round_up(run_config.logical_m, config.tk_tile_granularity)
        tk_m = tk_local_m * config.world_size
        tk_n = round_up(run_config.logical_n, config.tk_tile_granularity)
        run_config.tk_padded_m = tk_m
        run_config.tk_padded_n = tk_n

        # Reuse the mkernel's padded A/B when the two candidates land on the
        # same tile granularity, instead of allocating a second copy.
        A_tk_local = (
            A_mk_local if tk_local_m == mk_local_m
            else pad_rows(config, A_local, tk_local_m)
        )
        # mkernel_b_buf is already [N, K] contiguous (see above) -- reuse it
        # directly when the two candidates land on the same N padding,
        # instead of transposing a second copy.
        run_config.tk_b_transposed = (
            run_config.mkernel_b_buf if tk_n == mk_n
            else pad_cols(config, B_ref, tk_n).T.contiguous()
        )

        tk_module = load_thirdparty.load_tk_extension("ag_gemm")
        run_config.tk_a_dist = tk_module.TKParallelTensor(
            (tk_m, config.default_k), dtype=torch.bfloat16,
            local_rank=config.local_rank, local_world_size=config.world_size,
            multicast=True,
        )
        run_config.tk_a_dist.data_[
            config.local_rank * tk_local_m : (config.local_rank + 1) * tk_local_m
        ].copy_(A_tk_local)
        run_config.tk_c_buf = torch.zeros((tk_m, tk_n), device="cuda", dtype=torch.bfloat16)
        run_config.tk_barrier = tk_module.TKParallelTensor(
            (2, 1024, 1024), dtype=torch.int,
            local_rank=config.local_rank, local_world_size=config.world_size,
            multicast=True,
        )
        run_config.tk_barrier.data_.zero_()

        # ParallelKittens requires all ranks to finish initializing its
        # multicast barrier before the first fused launch -- same race as
        # mkernel's DistBuffer fill above, just for the TKParallelTensor
        # buffers instead.
        torch.cuda.synchronize()
        dist.barrier()

        def run_tk(num_comm_sms):
            tk_module.all_gather_matmul(
                run_config.tk_a_dist, run_config.tk_b_transposed,
                run_config.tk_c_buf, run_config.tk_barrier, num_comm_sms,
            )

        timings = []
        best_comm_sms = None
        best_ms = float("inf")
        for comm_sms in config.tk_comm_sms:
            ms = benchmark_cuda(
                lambda comm_sms=comm_sms: run_tk(comm_sms),
                tune_warmup,
                tune_iterations,
            )
            timings.append((comm_sms, ms))
            if ms < best_ms:
                best_comm_sms, best_ms = comm_sms, ms
        assert best_comm_sms is not None

        if config.is_chief:
            print(
                f"  ThunderKittens config {projection} M={global_m}: "
                f"num_comm_sms={best_comm_sms} (autotuned)",
                flush=True,
            )
            for comm_sms, tune_ms in sorted(timings, key=lambda item: item[1]):
                mark = " <- best" if comm_sms == best_comm_sms else ""
                tune_tflops = useful_tflops(global_m, logical_n, config.default_k, tune_ms)
                print(
                    f"    [autotune] num_comm_sms={comm_sms}: "
                    f"{tune_ms:8.3f} ms  {tune_tflops:8.2f} TFLOP/s{mark}",
                    flush=True,
                )

        def bench_tk():
            run_tk(best_comm_sms)

        fns.append((bench_tk, "TK", True))

    return fns

def main():
    # common setup
    args = parse_args()
    config = CONFIGS.get(args.arch)
    if config is None:
        raise NotImplementedError(
            f"--arch {args.arch!r} has no bench path in main() yet; only 'hopper' and 'blackwell' have been implemented."
        )
    if args.shapes is None:
        args.shapes = ",".join(str(s) for s in config.shapes_to_test)
    # Preserve explicit --num-intra-comm-sms from the CLI (e.g. AG_GEMM_BENCH_EXTRA
    # profile runs). The per-shape loop used to force 0 here, which silently ignored
    # tuned intra splits unless INTRA_OVERRIDE was populated.
    cli_num_intra_comm_sms = int(args.num_intra_comm_sms)

    torch.cuda.set_device(config.local_rank)
    dist_backend = os.environ.get("MKERNEL_DIST_BACKEND", "nccl")
    if dist_backend == "nccl":
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{config.local_rank}"))
    else:
        dist.init_process_group(dist_backend)

    mod = load_module.load(config.kernel_name)
    result_sizes, result_fused = [], []
    correctness_ok = True

    if args.arch == "hopper":
        # config modification
        if not config.peer_ip:
            peer_node = 1 if config.node_idx == 0 else 0
            peer_ip = os.environ.get(f"NODE{peer_node}_IP")
            if not peer_ip:
                raise RuntimeError(f"NODE{peer_node}_IP must be set, or set PEER_IP explicitly")
            config.peer_ip = peer_ip

        source_backing = rdma_backing()
        target_backing = source_backing
        if config.is_chief:
            print(f"[ag_gemm] world={config.world_size*config.num_nodes} shapes={args.shapes}", flush=True)
            print(rdma_policy_label(
                config.kernel_name, source=source_backing, target=target_backing), flush=True)

        shapes = [int(x) for x in args.shapes.split(",") if x.strip()]

        # Per-shape intra override that bypasses the max(4) floor in the kernel.
        INTRA_OVERRIDE = {}
        # Per-shape override format: AG_GEMM_INTRA_OVERRIDE_<M>=<intra>.
        # Applied after the conditional defaults so the env value always wins.
        for base_n in shapes:
            env_key = f"AG_GEMM_INTRA_OVERRIDE_{base_n}"
            if env_key in os.environ:
                INTRA_OVERRIDE[base_n] = int(os.environ[env_key])
                if config.is_chief:
                    print(f"[ag_gemm] env override {env_key}={os.environ[env_key]}", flush=True)
        for base_n in shapes:
            # Per-shape num_comm_sms override (small-M overhead reduction).
            if base_n in config.sm_per_shape:
                args.num_comm_sms = config.sm_per_shape[base_n]
                if config.is_chief:
                    print(f"[ag_gemm] M={base_n}: per-shape num_comm_sms={args.num_comm_sms}",
                        flush=True)
            if base_n in INTRA_OVERRIDE:
                args.num_intra_comm_sms = INTRA_OVERRIDE[base_n]
                if config.is_chief:
                    print(f"[ag_gemm] M={base_n}: per-shape num_intra_comm_sms={args.num_intra_comm_sms}",
                        flush=True)
            else:
                args.num_intra_comm_sms = cli_num_intra_comm_sms

            samples = []

            fn, _, should_wrap, check_fn, invariant_check = ag_gemm_hopper_prepare(config, mod, base_n, source_backing, args.num_comm_sms, args.num_intra_comm_sms, args.iters)

            samples = None
            if not should_wrap:
                samples = fn()
            else:
                samples = [benchmark_cuda(fn, args.warmup, args.iters)]

            wall_ms = avg_then_max_cuda(samples)

            correctness_ok = check_fn() and correctness_ok

            if os.environ.get("MKERNEL_INVARIANT_DETERMINISTIC", "0") == "1":
                correctness_ok = invariant_check() and correctness_ok

            if config.is_chief:
                print(f"[ag_gemm] M={base_n} wall={wall_ms:.3f} ms", flush=True)
            

            result_sizes.append(f"M={base_n}")
            result_fused.append(wall_ms)

    else:
        check_correctness_ag_gemm_blackwell(config, mod)
        for (projection, logical_n), m in product(config.projections, config.shapes_to_test):
            fns_to_run = ag_gemm_blackwell_prepare(config, mod, projection, m, logical_n, args.warmup, args.iters)

            results = []
            for i, (fn, name, should_wrap) in enumerate(fns_to_run):
                if i > 0:
                    # Let the GPU settle between candidates rather than
                    # measuring one right on the heels of another.
                    time.sleep(5)
                ms = benchmark_cuda(fn, args.warmup, args.iters) if should_wrap else fn()
                results.append((name, ms))
            report_blackwell_result(config, projection, m, logical_n, results)

            # A plain string, not a tuple: write_results_json merges with a
            # prior run's JSON, and a size that round-trips through JSON as a
            # list (json has no tuple type) is unhashable as a dict key there.
            result_sizes.append(f"{projection} M={m} N={logical_n}")
            for name, res in results:
                if name == "mkernel":
                    result_fused.append(res)

    if config.is_chief and args.save_json:
        # MERGE with existing JSON so a single-shape bench doesn't erase the
        # other shapes the chart needs.
        from common import write_results_json
        write_results_json(Path(args.save_json), "ag_gemm",
                           result_sizes, result_fused,
                           note=f"release ag_gemm bench (world={config.world_size*config.num_nodes})")
        print(f"[ag_gemm] wrote {args.save_json}", flush=True)

    if config.is_chief and args.compare_to:
        ok = compare_named_results("ag_gemm", result_sizes, result_fused, args.compare_to)
        ok = ok and correctness_ok
        dist.destroy_process_group()
        return 0 if ok else 1
    if not correctness_ok:
        dist.destroy_process_group()
        return 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    