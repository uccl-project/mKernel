"""Matched direct-NCCL + direct-cuBLAS baseline for dispatch GEMM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import load_module  # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dispatch_gemm_blackwell_bench import (  # noqa: E402
    DEFAULT_SHAPES,
    H,
    I,
    NUM_EXPERTS,
    build_choices,
    build_local_routes,
    int_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=int_list, default=DEFAULT_SHAPES)
    parser.add_argument("--routing", choices=("uniform", "multinomial"),
                        default=os.environ.get(
                            "MKERNEL_DISPATCH_GEMM_ROUTING", "multinomial"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def make_problem(num_tokens: int, rank: int, world: int, routing: str):
    local_tokens = num_tokens // world
    local_experts = NUM_EXPERTS // world
    choices = build_choices(num_tokens, routing, rank)
    pull, padded, actual_rows = build_local_routes(choices, rank, world)
    padded_rows = pull.size(0)

    send_indices = []
    input_splits = []
    token_begin = rank * local_tokens
    token_end = token_begin + local_tokens
    for dst in range(world):
        expert_begin = dst * local_experts
        chunk_begin = len(send_indices)
        for expert in range(expert_begin, expert_begin + local_experts):
            for token in range(token_begin, token_end):
                if expert in choices[token].tolist():
                    send_indices.append(token - token_begin)
        input_splits.append(len(send_indices) - chunk_begin)

    pull_cpu = pull.cpu()
    recv_positions = []
    output_splits = []
    for src in range(world):
        positions = torch.nonzero(
            pull_cpu[:, 0] == src, as_tuple=False).flatten().tolist()
        recv_positions.extend(positions)
        output_splits.append(len(positions))

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    pre_tokens = torch.randn(
        (local_tokens, H), device="cuda", dtype=torch.bfloat16)
    pre_tokens.div_(H ** 0.25)
    torch.manual_seed(100 + rank)
    torch.cuda.manual_seed(100 + rank)
    weights = torch.randn(
        (local_experts, H, I), device="cuda", dtype=torch.bfloat16)
    weights.div_(H ** 0.25)

    send_indices = torch.tensor(send_indices, device="cuda", dtype=torch.int64)
    recv_positions = torch.tensor(
        recv_positions, device="cuda", dtype=torch.int64)
    send = torch.empty(
        (send_indices.numel(), H), device="cuda", dtype=torch.bfloat16)
    recv = torch.empty(
        (recv_positions.numel(), H), device="cuda", dtype=torch.bfloat16)
    post_tokens = torch.zeros(
        (padded_rows, H), device="cuda", dtype=torch.bfloat16)
    outputs = torch.empty(
        (padded_rows, I), device="cuda", dtype=torch.bfloat16)
    offsets = padded.cumsum(0).cpu().tolist()
    return {
        "pre_tokens": pre_tokens,
        "weights": weights,
        "send_indices": send_indices,
        "recv_positions": recv_positions,
        "send": send,
        "recv": recv,
        "post_tokens": post_tokens,
        "outputs": outputs,
        "pull": pull,
        "padded": padded,
        "offsets": offsets,
        "input_splits": input_splits,
        "output_splits": output_splits,
        "actual_rows": actual_rows,
    }


def launch(session, problem) -> None:
    torch.index_select(
        problem["pre_tokens"], 0, problem["send_indices"],
        out=problem["send"])
    session.communicate(problem["send"], problem["recv"])
    problem["post_tokens"].index_copy_(
        0, problem["recv_positions"], problem["recv"])
    session.gemm(
        problem["post_tokens"], problem["weights"], problem["outputs"])


def check_result(problem, rank: int, world: int) -> None:
    gathered = torch.empty(
        (world * problem["pre_tokens"].size(0), H),
        device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(gathered, problem["pre_tokens"])
    expected_post = torch.zeros_like(problem["post_tokens"])
    pull = problem["pull"]
    valid = pull[:, 0] >= 0
    local_tokens = problem["pre_tokens"].size(0)
    global_rows = (pull[valid, 0].to(torch.int64) * local_tokens +
                   pull[valid, 1].to(torch.int64))
    expected_post[valid] = gathered[global_rows]
    if not torch.equal(problem["post_tokens"], expected_post):
        raise AssertionError(f"rank {rank}: NCCL dispatch layout mismatch")

    reference = torch.zeros_like(problem["outputs"])
    row_begin = 0
    for expert, row_end in enumerate(problem["offsets"]):
        reference[row_begin:row_end] = (
            expected_post[row_begin:row_end] @ problem["weights"][expert])
        row_begin = row_end
    torch.testing.assert_close(
        problem["outputs"], reference, atol=0.55, rtol=0.12,
        msg=lambda msg: f"rank {rank}: cuBLAS output mismatch: {msg}")


def main() -> None:
    args = parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    if world != 8:
        raise RuntimeError(f"expected 8 GPUs, got {world}")

    mod = load_module.load("dispatch_gemm_blackwell_nccl_cublas")
    unique_id = [mod.get_nccl_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(
        unique_id, src=0, device=torch.device(f"cuda:{rank}"))
    session = mod.Session(unique_id[0], rank, world, rank)

    if rank == 0:
        print(
            f"[optimized-pack+direct-nccl+direct-cublas] "
            f"world={world} routing={args.routing}")
        print("global_tokens max_actual max_padded avg_ms useful_TF/GPU padded_TF/GPU")

    for num_tokens in args.shapes:
        problem = make_problem(num_tokens, rank, world, args.routing)
        session.configure(
            problem["input_splits"], problem["output_splits"],
            problem["offsets"])
        counts = torch.tensor(
            [problem["actual_rows"], problem["post_tokens"].size(0)],
            device="cuda", dtype=torch.int64)
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
        max_actual, max_padded = counts.cpu().tolist()

        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(args.warmup):
            launch(session, problem)
            torch.cuda.synchronize()
        dist.barrier()

        if num_tokens <= 16384:
            n_iters = max(args.iters, 64)
        elif num_tokens <= 65536:
            n_iters = max(args.iters, 32)
        else:
            n_iters = max(args.iters, 24)

        trial_ms = []
        for _ in range(args.trials):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(n_iters):
                launch(session, problem)
            end.record()
            end.synchronize()
            rank_avg = torch.tensor(
                [start.elapsed_time(end) / n_iters],
                device="cuda", dtype=torch.float64)
            dist.all_reduce(rank_avg, op=dist.ReduceOp.MAX)
            trial_ms.append(rank_avg.item())
        avg_ms = sum(trial_ms) / len(trial_ms)
        useful_tflops = 2.0 * max_actual * H * I / (avg_ms * 1e9)
        padded_tflops = 2.0 * max_padded * H * I / (avg_ms * 1e9)

        if args.check:
            check_result(problem, rank, world)
        if rank == 0:
            print(
                f"{num_tokens:13d} {max_actual:10d} {max_padded:10d} "
                f"{avg_ms:7.4f} {useful_tflops:12.2f} {padded_tflops:12.2f}")

        del problem
        torch.cuda.empty_cache()
        dist.barrier()

    if args.check and rank == 0:
        print("PASS: matched direct NCCL + direct cuBLAS baseline")
    del session
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
