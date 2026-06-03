"""Same-functionality baseline for dispatch_gemm_glu_combine.

Standard EP-MoE via NCCL + cuBLAS (tensor cores), matching the fused kernel's
work: dispatch (all_to_all) -> gemm1(2I) -> SwiGLU -> gemm2(H) -> combine
(all_to_all) -> weighted sum. Uniform routing (each token -> TOP_K experts,
balanced) so splits are even — same FLOPs/comm as the kernel.

TFLOPS uses the SAME formula as the kernel plot:
    per_rank_flops = 6 * tokens * TOPK * H * I / WORLD     (gemm1 + gemm2)

Run (NCCL uses EFA inter-node):
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=R \
        --master_addr=NODE0 --master_port=PORT baseline_glu_combine.py \
        --H 256 --I 128 --topk 8 --shapes 256,512,1024,2048,4096
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.distributed as dist


def max_across_ranks(v):
    t = torch.tensor([v], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def timed(run_once, warmup, iters):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(); dist.barrier()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run_once()
    e.record(); torch.cuda.synchronize()
    return max_across_ranks(s.elapsed_time(e) / max(iters, 1))


def bench_shape(tokens, rank, world, H, I, topk, warmup, iters):
    assert tokens % world == 0
    local_tokens = tokens // world
    rows = local_tokens * topk                      # dispatched rows this rank emits
    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    send = (torch.randn((rows, H), generator=g, device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25))
    # Uniform all_to_all splits (balanced routing).
    base, rem = divmod(rows, world)
    in_splits = [base + (p < rem) for p in range(world)]
    out_splits = [in_splits[rank]] * world
    recv = torch.empty((sum(out_splits), H), device="cuda", dtype=torch.bfloat16)
    R = recv.shape[0]
    # Expert FFN weights (one fused GEMM per layer = cuBLAS best case).
    w1 = (torch.randn((H, 2 * I), generator=g, device="cuda", dtype=torch.bfloat16)
          * (H ** -0.5))
    w2 = (torch.randn((I, H), generator=g, device="cuda", dtype=torch.bfloat16)
          * (I ** -0.5))
    back = torch.empty((rows, H), device="cuda", dtype=torch.bfloat16)

    def run_once():
        dist.all_to_all_single(recv, send, out_splits, in_splits)   # dispatch
        h1 = recv @ w1                                              # gemm1 -> [R, 2I]
        act = torch.nn.functional.silu(h1[:, :I]) * h1[:, I:]       # SwiGLU -> [R, I]
        y = act @ w2                                                # gemm2 -> [R, H]
        dist.all_to_all_single(back, y, in_splits, out_splits)     # combine
        _ = back.view(local_tokens, topk, H).sum(1)                # weighted sum (w=1)

    ms = timed(run_once, warmup, iters)
    per_rank_flops = 6.0 * tokens * topk * H * I / world
    tflops = per_rank_flops / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--I", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--shapes", type=str, default="256,512,1024,2048,4096")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = rank == 0
    if is_chief:
        print(f"[baseline] NCCL+cuBLAS EP-MoE  world={world} H={args.H} I={args.I} "
              f"topk={args.topk}", flush=True)
    for tok in [int(s) for s in args.shapes.split(",") if s]:
        ms, tf = bench_shape(tok, rank, world, args.H, args.I, args.topk,
                             args.warmup, args.iters)
        if is_chief:
            print(f"[baseline] tokens={tok} wall={ms:.3f}ms tflops_per_gpu={tf:.3f}",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
