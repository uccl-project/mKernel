"""User-facing 4-GPU benchmark for Blackwell LSA dispatch + GEMM.

The workload matches dispatch_gemm_bench.py: global input tokens, top-8
routing over 256 experts, 128-row expert padding, and 64 local experts per
GPU.  Only the transport differs: this benchmark stays inside one NVLink
domain and calls moe_dispatch_gemm_blackwell directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import load_module  # noqa: E402


H = 7168
I = 2048
NUM_EXPERTS = 256
TOP_K = 8
ROW_BLOCK = 128
DEFAULT_SHAPES = [8192, 16384, 32768, 65536, 131072]


def int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=int_list, default=DEFAULT_SHAPES)
    parser.add_argument("--routing", choices=("uniform", "multinomial"),
                        default=os.environ.get(
                            "MKERNEL_DISPATCH_GEMM_ROUTING", "multinomial"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dispatch-sms", type=int, default=28)
    parser.add_argument("--gemm-sms", type=int, default=124)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_choices(num_tokens: int, routing: str, rank: int) -> torch.Tensor:
    if routing == "uniform":
        return (torch.arange(num_tokens * TOP_K, dtype=torch.int64) %
                NUM_EXPERTS).to(torch.int32).view(num_tokens, TOP_K)

    if rank == 0:
        generator = torch.Generator(device="cuda").manual_seed(num_tokens)
        routing_weights = torch.rand(
            NUM_EXPERTS, device="cuda", generator=generator)
        choices = torch.multinomial(
            routing_weights.repeat(num_tokens, 1), TOP_K,
            replacement=False, generator=generator).to(torch.int32)
    else:
        choices = torch.empty(
            (num_tokens, TOP_K), device="cuda", dtype=torch.int32)
    dist.broadcast(choices, 0)
    return choices.cpu()


def build_local_routes(
    choices: torch.Tensor, rank: int, world: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_tokens = choices.size(0)
    if num_tokens % world != 0:
        raise ValueError("global token count must be divisible by world size")
    if NUM_EXPERTS % world != 0:
        raise ValueError("expert count must be divisible by world size")

    local_tokens = num_tokens // world
    local_experts = NUM_EXPERTS // world
    expert_begin = rank * local_experts
    expert_routes: list[list[tuple[int, int]]] = [
        [] for _ in range(local_experts)
    ]

    for token in range(num_tokens):
        src_gpu = token // local_tokens
        src_token = token % local_tokens
        for expert in choices[token].tolist():
            if expert_begin <= expert < expert_begin + local_experts:
                expert_routes[expert - expert_begin].append(
                    (src_gpu, src_token))

    rows: list[tuple[int, int]] = []
    padded_counts = []
    actual_rows = 0
    for routes in expert_routes:
        actual_rows += len(routes)
        padded = ((len(routes) + ROW_BLOCK - 1) // ROW_BLOCK) * ROW_BLOCK
        padded_counts.append(padded)
        rows.extend(routes)
        rows.extend([(-1, -1)] * (padded - len(routes)))

    pull = torch.tensor(rows, device="cuda", dtype=torch.int32)
    padded = torch.tensor(padded_counts, device="cuda", dtype=torch.int32)
    return pull, padded, actual_rows


def make_problem(mod, num_tokens: int, rank: int, world: int,
                 routing: str):
    local_tokens = num_tokens // world
    local_experts = NUM_EXPERTS // world
    choices = build_choices(num_tokens, routing, rank)
    pull, padded, actual_rows = build_local_routes(choices, rank, world)
    padded_rows = pull.size(0)

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    pre_tokens = mod.DistBuffer(
        (local_tokens, H), dtype=torch.bfloat16,
        local_rank=rank, local_world_size=world,
        multicast=False, backing="vmm")
    pre_tokens.data_.normal_()
    pre_tokens.data_.div_(H ** 0.25)

    torch.manual_seed(100 + rank)
    torch.cuda.manual_seed(100 + rank)
    weights = torch.randn(
        (local_experts, H, I), device="cuda", dtype=torch.bfloat16)
    weights.div_(H ** 0.25)

    post_tokens = torch.zeros(
        (padded_rows, H), device="cuda", dtype=torch.bfloat16)
    outputs = torch.empty(
        (padded_rows, I), device="cuda", dtype=torch.bfloat16)
    row_ready = torch.zeros(
        (padded_rows + ROW_BLOCK - 1) // ROW_BLOCK,
        device="cuda", dtype=torch.int32)
    return (pre_tokens, post_tokens, pull, row_ready, weights, outputs,
            padded, actual_rows)


def launch(mod, problem, dispatch_sms: int, gemm_sms: int) -> None:
    pre_tokens, post_tokens, pull, row_ready, weights, outputs, padded, _ = problem
    mod.moe_dispatch_gemm_blackwell(
        pre_tokens, post_tokens, pull, row_ready, weights, outputs, padded,
        num_dispatch_sms=dispatch_sms, num_gemm_sms=gemm_sms)


def check_result(problem, rank: int, world: int) -> None:
    pre_tokens, post_tokens, pull, _, weights, outputs, padded, _ = problem
    gathered = torch.empty(
        (world * pre_tokens.data_.size(0), H),
        device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(gathered, pre_tokens.data_)
    valid = pull[:, 0] >= 0
    expected = torch.zeros_like(post_tokens)
    global_rows = (pull[valid, 0].to(torch.int64) *
                   pre_tokens.data_.size(0) +
                   pull[valid, 1].to(torch.int64))
    expected[valid] = gathered[global_rows]
    if not torch.equal(post_tokens, expected):
        raise AssertionError(f"rank {rank}: dispatch output mismatch")

    reference = torch.zeros_like(outputs)
    row_begin = 0
    for expert, rows in enumerate(padded.cpu().tolist()):
        if rows:
            row_end = row_begin + rows
            reference[row_begin:row_end] = (
                expected[row_begin:row_end] @ weights[expert])
            row_begin = row_end
    torch.testing.assert_close(
        outputs, reference, atol=0.55, rtol=0.12,
        msg=lambda msg: f"rank {rank}: GEMM output mismatch: {msg}")


def main() -> None:
    args = parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    if world != 4:
        raise RuntimeError(f"expected one 4-GPU NVLink domain, got {world} GPUs")

    mod = load_module.load("dispatch_gemm_blackwell")
    if rank == 0:
        print(
            f"[dispatch_gemm_blackwell] world={world} routing={args.routing} "
            f"sm(dispatch,gemm)=({args.dispatch_sms},{args.gemm_sms})")
        print("global_tokens local_tokens max_actual max_padded avg_ms useful_TF/GPU padded_TF/GPU")

    for num_tokens in args.shapes:
        problem = make_problem(mod, num_tokens, rank, world, args.routing)
        row_ready = problem[3]
        actual_rows = problem[7]
        padded_rows = problem[1].size(0)
        row_counts = torch.tensor(
            [actual_rows, padded_rows], device="cuda", dtype=torch.int64)
        dist.all_reduce(row_counts, op=dist.ReduceOp.MAX)
        max_actual_rows, max_padded_rows = row_counts.cpu().tolist()

        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(args.warmup):
            row_ready.zero_()
            launch(mod, problem, args.dispatch_sms, args.gemm_sms)
            torch.cuda.synchronize()
        dist.barrier()

        trial_ms = []
        for _ in range(args.trials):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                row_ready.zero_()
                launch(mod, problem, args.dispatch_sms, args.gemm_sms)
            end.record()
            end.synchronize()
            rank_avg = torch.tensor(
                [start.elapsed_time(end) / args.iters],
                device="cuda", dtype=torch.float64)
            dist.all_reduce(rank_avg, op=dist.ReduceOp.MAX)
            trial_ms.append(rank_avg.item())
        avg_ms = sorted(trial_ms)[len(trial_ms) // 2]
        useful_tflops = 2.0 * max_actual_rows * H * I / (avg_ms * 1e9)
        padded_tflops = 2.0 * max_padded_rows * H * I / (avg_ms * 1e9)

        if args.check:
            row_ready.zero_()
            problem[1].zero_()
            launch(mod, problem, args.dispatch_sms, args.gemm_sms)
            torch.cuda.synchronize()
            check_result(problem, rank, world)

        if rank == 0:
            print(
                f"{num_tokens:13d} {num_tokens // world:12d} "
                f"{max_actual_rows:10d} {max_padded_rows:10d} {avg_ms:7.4f} "
                f"{useful_tflops:12.2f} {padded_tflops:12.2f}")

        del problem
        torch.cuda.empty_cache()
        dist.barrier()

    if args.check and rank == 0:
        print("PASS: user-style dispatch routing")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
