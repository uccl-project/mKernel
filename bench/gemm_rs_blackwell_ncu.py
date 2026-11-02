"""Single-launch driver for Nsight Compute profiling of the mKernel GEMM + RS.

Not meant to be run directly: bench/ncu_gemm_rs_blackwell.sh wraps every rank in
ncu and selects the one launch inside the NVTX range below. Each rank warms up
the chosen config, then issues exactly one launch of it inside the push/pop
range "ncu_target". Uses the release build (no timing ring), so build with
    make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell
Kernel replay restores device memory between passes, so this driver does not
check the output: a replayed launch can leave C or the ready counters stale.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--swizzle", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--signals-per-tile", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--rs-strip-rows", type=int, default=16, choices=(8, 16, 32))
    parser.add_argument("--mma-warps", type=int, default=2, choices=(1, 2))
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

    module = load_module.load("gemm_rs_blackwell")
    if module.gemm_rs_blackwell_world_size != config.world_size:
        parser.error(
            f"Rebuild mKernel with INTRA_NUM_DEVICES={config.world_size} to match WORLD_SIZE"
        )

    torch.cuda.set_device(config.local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", config.local_rank))
    try:
        buffers = make_blackwell_buffers(config, M, N, K, module)
        candidate = (args.swizzle, args.signals_per_tile, args.rs_strip_rows, args.mma_warps)
        for _ in range(args.warmup):
            run_mkernel(module, buffers, candidate)
        torch.cuda.synchronize()
        dist.barrier()

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
