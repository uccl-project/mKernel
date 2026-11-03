"""Single-launch driver for Nsight Compute profiling of the mKernel GEMM + RS.

Not meant to be run directly: bench/ncu_gemm_rs_blackwell.sh wraps every rank in
ncu and selects the one launch inside the NVTX range below. Each rank warms up
the chosen config, then issues exactly one launch of it inside the push/pop
range "ncu_target". Uses the release build (no timing ring), so build with
    make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell
Under ncu application replay the whole process reruns once per metric pass, so
every pass starts from freshly prepared buffers. Ranks still reach the profiled
launch at different times, so by default the ready counters are pre-filled and
the RS never waits on peers (see presatisfy_ready_counters); use the in-kernel
timing profiler to study those waits.
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))

from gemm_rs_blackwell_bench import BlackwellBenchConfig, make_blackwell_buffers, run_mkernel
import load_module

NVTX_RANGE = "ncu_target"


def next_pass_index(local_rank: int) -> int:
    """Count this rank's launches, which under ncu application replay are its
    passes. Every rank relaunches once per pass, so the counts agree across ranks.
    Without NCU_PASS_DIR (a plain torchrun), every launch is pass 0."""
    pass_dir = os.environ.get("NCU_PASS_DIR")
    if not pass_dir:
        return 0
    counter = Path(pass_dir) / f"rank{local_rank}.pass"
    index = int(counter.read_text()) + 1 if counter.exists() else 0
    counter.write_text(str(index))
    return index


def presatisfy_ready_counters(buffers, world_size: int):
    """Fill this rank's ready counters with the count the next launch waits for,
    so its RS warps never spin on peers. Each ncu rank reaches the profiled launch
    at a different time, and without this the profile is mostly that spin. The RS
    then reduces partially written tiles, so the output is wrong; nothing checks it.
    Call only after every rank has drained its previous launch (sync + barrier)."""
    next_epoch = buffers.epochs_mkernel.data_.view(-1)[:1] + 1
    buffers.ready_mkernel.data_.copy_(next_epoch * world_size)


def init_process_group(config, pass_index: int):
    """Join the torchrun agent's store under a per-pass prefix. Application replay
    relaunches the worker inside one torchrun, so the default env:// rendezvous
    would read the previous pass's NCCL id and barrier keys from the same store."""
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), config.world_size,
        is_master=False, wait_for_workers=False,
    )
    dist.init_process_group(
        "nccl", store=dist.PrefixStore(f"ncu_pass{pass_index}", store),
        rank=int(os.environ["RANK"]), world_size=config.world_size,
        device_id=torch.device("cuda", config.local_rank),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--swizzle", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--signals-per-tile", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--rs-strip-rows", type=int, default=16, choices=(8, 16, 32))
    parser.add_argument("--mma-warps", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--keep-waits", action="store_true",
        help="Leave the ready counters alone, so the RS warps spin until peers publish. "
        "Under ncu that profiles cross-rank launch skew, not the kernel",
    )
    args = parser.parse_args()

    config = BlackwellBenchConfig()
    M, N, K = args.M, config.output_dim, config.split_inner_dim
    if config.world_size not in (2, 4, 8):
        parser.error("Launch with torchrun using 2, 4, or 8 GPUs on one node")
    if int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size)) != config.world_size:
        parser.error("This driver requires a single node")
    if M <= 0 or M % (config.world_size * 256) or M % 512:
        parser.error("mKernel requires M divisible by both WORLD_SIZE * 256 and 512")
    if args.mma_warps == 1 and args.signals_per_tile == 4:
        parser.error("--mma-warps 1 supports --signals-per-tile 1 or 2")

    pass_index = next_pass_index(config.local_rank)
    # The broker's shm/socket keys and the store keys must not collide with a
    # previous application-replay pass whose peers may still be tearing down.
    os.environ["DIST_BROKER_KEY"] = f"mkernel_ncu_{os.environ['MASTER_PORT']}_{pass_index}"
    module = load_module.load("gemm_rs_blackwell")
    if module.gemm_rs_blackwell_world_size != config.world_size:
        parser.error(
            f"Rebuild mKernel with INTRA_NUM_DEVICES={config.world_size} to match WORLD_SIZE"
        )

    torch.cuda.set_device(config.local_rank)
    init_process_group(config, pass_index)
    try:
        buffers = make_blackwell_buffers(config, M, N, K, module)
        candidate = (args.swizzle, args.signals_per_tile, args.rs_strip_rows, args.mma_warps)
        for _ in range(args.warmup):
            run_mkernel(module, buffers, candidate)
        torch.cuda.synchronize()
        dist.barrier()
        if not args.keep_waits:
            presatisfy_ready_counters(buffers, config.world_size)

        torch.cuda.nvtx.range_push(NVTX_RANGE)
        run_mkernel(module, buffers, candidate)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if config.is_chief:
            print(
                f"[ncu driver] profiled M={M} N={N} K_local={K} swizzle={args.swizzle} "
                f"signals={args.signals_per_tile} rs_rows={args.rs_strip_rows} "
                f"mma_warps={args.mma_warps}",
                flush=True,
            )
        del buffers
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
