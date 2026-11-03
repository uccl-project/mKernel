"""In-kernel timing profile of mKernel GEMM + reduce-scatter on one node.

Build with make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell-profile, then
    torchrun --standalone --nproc-per-node=8 bench/gemm_rs_blackwell_profile.py --shapes 8192
Each dumping rank warms up without the ring, records exactly one launch at the
end of a back-to-back chain (--lead-in), checks it against cuBLAS + NCCL, and
writes <out-dir>/gemm_rs_blackwell_M<M>_rank<r>.npz plus a Gantt PDF. Re-render or zoom without a GPU:
    python python/timings.py traces/gemm_rs_blackwell_M8192_rank0.npz --blocks 0-7
%globaltimer is per GPU, so traces from different ranks share no time axis.
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

from common import check_close
from gemm_rs_blackwell_bench import (
    BlackwellBenchConfig,
    make_blackwell_buffers,
    mkernel_output,
    run_baseline,
    run_mkernel,
)
import load_module
import timings as timing_lib

KERNEL = "gemm_rs_blackwell"


def profile_shape(config, module, M, N, K, args, dumps: bool) -> bool:
    buffers = make_blackwell_buffers(config, M, N, K, module)
    num_blocks = module.TIMING_NUM_BLOCKS
    events_per_block = module.EVENTS_PER_BLOCK
    # Zeroed: a zero timestamp marks an unwritten record, so no head array is needed.
    ring = (
        torch.zeros(num_blocks * events_per_block * 2, dtype=torch.int64, device="cuda")
        if dumps else None
    )

    # Warm up without the ring: every launch would restart each CTA's head at 0.
    candidate = (args.swizzle, args.signals_per_tile, args.rs_strip_rows, args.mma_warps)
    for _ in range(args.warmup):
        run_mkernel(module, buffers, candidate)
    torch.cuda.synchronize()
    dist.barrier()

    # Leaving the host barrier skews each rank's start by tens of us. Queue a chain
    # of unrecorded launches with no host sync, so the recorded launch starts when
    # the previous one drains on its own GPU, and the RS tail couples those ends.
    for _ in range(args.lead_in):
        run_mkernel(module, buffers, candidate)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    module.gemm_rs_blackwell_launch(
        buffers.baseline_a, buffers.baseline_b, buffers.c_mkernel,
        buffers.ready_mkernel, buffers.epochs_mkernel, swizzle_size=args.swizzle,
        signals_per_tile=args.signals_per_tile, rs_strip_rows=args.rs_strip_rows,
        mma_warps=args.mma_warps, timings=ring,
    )
    end.record()
    end.synchronize()
    launch_us = start.elapsed_time(end) * 1000.0

    run_baseline(buffers)
    torch.cuda.synchronize()
    correct = check_close(
        f"gemm_rs mKernel profile M={M}", mkernel_output(config, buffers), buffers.baseline_c_rs,
    )

    if dumps:
        records, heads = timing_lib.unpack(ring.cpu().numpy(), num_blocks, events_per_block)
        rank = config.local_rank
        full = int((heads >= events_per_block).sum())
        if full:
            print(
                f"[rank {rank}] warning: {full} CTAs filled all {events_per_block} records; "
                f"their timelines are truncated",
                flush=True,
            )
        title = (
            f"{KERNEL} M={M} N={N} K_local={K} rank {rank}/{config.world_size} "
            f"(launch {launch_us:.1f} us)"
        )
        out = Path(args.out_dir) / f"{KERNEL}_M{M}_rank{rank}.npz"
        timing_lib.save(
            out, records, module.TIMING_EVENTS, kernel=KERNEL,
            M=M, N=N, K=K, rank=rank, world_size=config.world_size,
            num_blocks=num_blocks, events_per_block=events_per_block,
            heads=heads, launch_us=launch_us, title=title,
        )
        layout = timing_lib.LAYOUTS[KERNEL]
        spans = timing_lib.spans_for_layout(records, module.TIMING_EVENTS, layout)
        print(
            f"[rank {rank}] M={M}: {records.shape[0]} events, trace span "
            f"{timing_lib.wall_span_us(records):.1f} us, launch incl. exit barrier "
            f"{launch_us:.1f} us\n{timing_lib.summarize(spans)}\n"
            f"{timing_lib.summarize_rs_counters(timing_lib.rs_counters(records, module.TIMING_EVENTS), config.world_size)}\n"
            f"{timing_lib.summarize_ksteps(timing_lib.kstep_table(records, module.TIMING_EVENTS, K // timing_lib.RED_BLOCK))}",
            flush=True,
        )
        if not args.no_render:
            pdf = timing_lib.plot(
                records, out.with_suffix(".pdf"), module.TIMING_EVENTS, layout, title=title,
            )
            print(f"[rank {rank}] wrote {out} and {pdf}", flush=True)
        else:
            print(f"[rank {rank}] wrote {out}", flush=True)

    del buffers, ring
    torch.cuda.synchronize()
    dist.barrier()
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="8192", help="Comma-separated M values")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--lead-in", type=int, default=8,
        help="Unrecorded launches queued back-to-back before the recorded one; 0 "
        "records the first launch after the host barrier",
    )
    parser.add_argument(
        "--ranks", default="0", help="Comma-separated ranks that dump a trace, or 'all'",
    )
    parser.add_argument("--swizzle", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--signals-per-tile", type=int, default=1, choices=(1, 2, 4),
        help="Ready-counter publications per 256x256 tile",
    )
    parser.add_argument("--rs-strip-rows", type=int, default=16, choices=(8, 16, 32))
    parser.add_argument(
        "--mma-warps", type=int, default=2, choices=(1, 2),
        help="MMA warps per CTA; 1 gives 256-row cluster tiles and needs --signals-per-tile <= 2",
    )
    parser.add_argument("--out-dir", default=str(HERE.parent / "traces"))
    parser.add_argument("--no-render", action="store_true", help="Write the .npz only")
    args = parser.parse_args()
    config = BlackwellBenchConfig()
    shapes = [int(value) for value in args.shapes.split(",")]
    if config.world_size not in (2, 4, 8):
        parser.error("Launch with torchrun using 2, 4, or 8 GPUs on one node")
    if int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size)) != config.world_size:
        parser.error("This profiler requires a single node")
    if any(m <= 0 or m % (config.world_size * 256) or m % 512 for m in shapes):
        parser.error("mKernel requires M divisible by both WORLD_SIZE * 256 and 512")
    if args.mma_warps == 1 and args.signals_per_tile == 4:
        parser.error("--mma-warps 1 supports --signals-per-tile 1 or 2")
    if args.warmup < 0 or args.lead_in < 0:
        parser.error("--warmup and --lead-in must be nonnegative")
    dump_ranks = (
        set(range(config.world_size)) if args.ranks == "all"
        else {int(r) for r in args.ranks.split(",")}
    )

    module = load_module.load(f"{KERNEL}_profile")
    if not hasattr(module, "EVENTS_PER_BLOCK"):
        parser.error(f"build/lib{KERNEL}_profile.so was not built with -DPROFILE_TIMINGS")
    if module.gemm_rs_blackwell_world_size != config.world_size:
        parser.error(
            f"Rebuild mKernel with INTRA_NUM_DEVICES={config.world_size} to match WORLD_SIZE"
        )

    torch.cuda.set_device(config.local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This profiler requires Blackwell GPUs")
    dist.init_process_group("nccl", device_id=torch.device("cuda", config.local_rank))
    try:
        all_correct = True
        for M in shapes:
            all_correct = profile_shape(
                config, module, M, config.output_dim, config.split_inner_dim, args,
                config.local_rank in dump_ranks,
            ) and all_correct
        if not all_correct:
            raise RuntimeError("profile build produced wrong GEMM+RS output")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
