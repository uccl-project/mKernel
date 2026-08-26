"""Benchmark the Blackwell dispatch+GEMM specialization variants.

Both variants use the same 8-GPU BF16 MoE workload. ``warp`` keeps
communication and GEMM roles in every persistent CTA, while ``sm`` assigns
separate CTA/SM pools to dispatch and GEMM.
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
MODULES = {
    "warp": "dispatch_gemm_warp_specialization",
    "sm": "dispatch_gemm_blackwell",
}


def int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    specialization = parser.add_mutually_exclusive_group()
    specialization.add_argument(
        "--specialization", choices=("warp", "sm"),
        default=os.environ.get("MKERNEL_SPECIALIZATION", "sm"),
        help="execution strategy (default: sm)",
    )
    specialization.add_argument(
        "--warp-specialization", action="store_const", const="warp",
        dest="specialization", help="alias for --specialization warp",
    )
    specialization.add_argument(
        "--sm-specialization", action="store_const", const="sm",
        dest="specialization", help="alias for --specialization sm",
    )
    parser.add_argument("--shapes", type=int_list, default=DEFAULT_SHAPES)
    parser.add_argument(
        "--routing", choices=("uniform", "multinomial"),
        default=os.environ.get("MKERNEL_DISPATCH_GEMM_ROUTING", "multinomial"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--ring-blocks", type=int, default=128,
        help="warp specialization: number of global ring slots",
    )
    parser.add_argument(
        "--num-sms", type=int, default=0,
        help="warp specialization: persistent CTAs (0 uses every SM)",
    )
    parser.add_argument(
        "--dispatch-sms", type=int, default=24,
        help="SM specialization: dispatch CTA count",
    )
    parser.add_argument(
        "--gemm-sms", type=int, default=124,
        help="SM specialization: GEMM CTA count",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_choices(num_tokens: int, routing: str, rank: int) -> torch.Tensor:
    if routing == "uniform":
        return (
            torch.arange(num_tokens * TOP_K, dtype=torch.int64) % NUM_EXPERTS
        ).to(torch.int32).view(num_tokens, TOP_K)

    if rank == 0:
        generator = torch.Generator(device="cuda").manual_seed(num_tokens)
        routing_weights = torch.rand(
            NUM_EXPERTS, device="cuda", generator=generator)
        choices = torch.multinomial(
            routing_weights.repeat(num_tokens, 1), TOP_K,
            replacement=False, generator=generator,
        ).to(torch.int32)
    else:
        choices = torch.empty(
            (num_tokens, TOP_K), device="cuda", dtype=torch.int32)
    dist.broadcast(choices, 0)
    return choices.cpu()


def build_local_routes(
    choices: torch.Tensor, rank: int, world: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_tokens = choices.size(0)
    if num_tokens % world or NUM_EXPERTS % world:
        raise ValueError("tokens and experts must divide evenly across ranks")

    local_tokens = num_tokens // world
    local_experts = NUM_EXPERTS // world
    expert_begin = rank * local_experts
    expert_routes: list[list[tuple[int, int]]] = [
        [] for _ in range(local_experts)
    ]
    for token in range(num_tokens):
        src_gpu, src_token = token // local_tokens, token % local_tokens
        for expert in choices[token].tolist():
            if expert_begin <= expert < expert_begin + local_experts:
                expert_routes[expert - expert_begin].append(
                    (src_gpu, src_token))

    rows: list[tuple[int, int]] = []
    padded_counts: list[int] = []
    row_experts: list[int] = []
    actual_rows = 0
    for expert, routes in enumerate(expert_routes):
        actual_rows += len(routes)
        padded = ((len(routes) + ROW_BLOCK - 1) // ROW_BLOCK) * ROW_BLOCK
        padded_counts.append(padded)
        rows.extend(routes)
        rows.extend([(-1, -1)] * (padded - len(routes)))
        row_experts.extend([expert] * (padded // ROW_BLOCK))

    return (
        torch.tensor(rows, device="cuda", dtype=torch.int32),
        torch.tensor(padded_counts, device="cuda", dtype=torch.int32),
        torch.tensor(row_experts, device="cuda", dtype=torch.int32),
        actual_rows,
    )


def make_problem(
    mod, num_tokens: int, rank: int, world: int, routing: str,
    specialization: str, ring_blocks: int,
) -> dict[str, object]:
    choices = build_choices(num_tokens, routing, rank)
    pull, padded, row_to_expert, actual_rows = build_local_routes(
        choices, rank, world)
    local_tokens = num_tokens // world
    local_experts = NUM_EXPERTS // world
    padded_rows = pull.size(0)

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    pre_tokens = mod.DistBuffer(
        (local_tokens, H), dtype=torch.bfloat16,
        local_rank=rank, local_world_size=world,
        multicast=False, backing="vmm",
    )
    pre_tokens.data_.normal_().div_(H ** 0.25)

    torch.manual_seed(100 + rank)
    torch.cuda.manual_seed(100 + rank)
    weights = torch.randn(
        (local_experts, H, I), device="cuda", dtype=torch.bfloat16)
    weights.div_(H ** 0.25)

    problem: dict[str, object] = {
        "pre_tokens": pre_tokens,
        "pull": pull,
        "padded": padded,
        "row_to_expert": row_to_expert,
        "actual_rows": actual_rows,
        "weights": weights,
        "outputs": torch.empty(
            (padded_rows, I), device="cuda", dtype=torch.bfloat16),
    }
    if specialization == "warp":
        ring_state = torch.zeros(
            (3, ring_blocks), device="cuda", dtype=torch.int32)
        problem.update({
            "ring_tokens": torch.empty(
                (ring_blocks * ROW_BLOCK, H),
                device="cuda", dtype=torch.bfloat16),
            "ring_state": ring_state,
            "ring_full": ring_state[0],
            "ring_empty": ring_state[1],
            "ring_done": ring_state[2],
        })
    else:
        problem.update({
            "post_tokens": torch.zeros(
                (padded_rows, H), device="cuda", dtype=torch.bfloat16),
            "row_ready": torch.zeros(
                (padded_rows + ROW_BLOCK - 1) // ROW_BLOCK,
                device="cuda", dtype=torch.int32),
        })
    return problem


def reset_workspace(problem: dict[str, object], specialization: str) -> None:
    if specialization == "warp":
        problem["ring_state"].zero_()
    else:
        problem["row_ready"].zero_()


def launch(mod, problem: dict[str, object], args: argparse.Namespace,
           num_sms: int) -> None:
    if args.specialization == "warp":
        mod.moe_dispatch_gemm_warp_specialization(
            problem["pre_tokens"], problem["ring_tokens"], problem["pull"],
            problem["ring_full"], problem["ring_empty"],
            problem["ring_done"], problem["row_to_expert"],
            problem["weights"], problem["outputs"], num_sms=num_sms,
        )
    else:
        mod.moe_dispatch_gemm_blackwell(
            problem["pre_tokens"], problem["post_tokens"], problem["pull"],
            problem["row_ready"], problem["weights"], problem["outputs"],
            problem["padded"], num_dispatch_sms=args.dispatch_sms,
            num_gemm_sms=args.gemm_sms,
        )


def check_result(problem: dict[str, object], rank: int, world: int,
                 specialization: str) -> None:
    pre_tokens = problem["pre_tokens"]
    pull = problem["pull"]
    gathered = torch.empty(
        (world * pre_tokens.data_.size(0), H),
        device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(gathered, pre_tokens.data_)

    dispatched = torch.zeros(
        (pull.size(0), H), device="cuda", dtype=torch.bfloat16)
    valid = pull[:, 0] >= 0
    global_rows = (
        pull[valid, 0].long() * pre_tokens.data_.size(0)
        + pull[valid, 1].long()
    )
    dispatched[valid] = gathered[global_rows]
    if specialization == "sm" and not torch.equal(
            problem["post_tokens"], dispatched):
        raise AssertionError(f"rank {rank}: dispatch output mismatch")

    reference = torch.zeros_like(problem["outputs"])
    row_begin = 0
    for expert, rows in enumerate(problem["padded"].cpu().tolist()):
        if rows:
            row_end = row_begin + rows
            reference[row_begin:row_end] = (
                dispatched[row_begin:row_end] @ problem["weights"][expert])
            row_begin = row_end
    torch.testing.assert_close(
        problem["outputs"], reference, atol=0.55, rtol=0.12,
        msg=lambda msg: f"rank {rank}: GEMM output mismatch: {msg}",
    )


def main() -> None:
    args = parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{rank}"))
    if world != 8:
        raise RuntimeError(f"these Blackwell builds expect 8 GPUs, got {world}")

    module_name = MODULES[args.specialization]
    mod = load_module.load(module_name)
    device_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    num_sms = args.num_sms or device_sms
    if rank == 0:
        config = (
            f"sms={num_sms} ring_blocks={args.ring_blocks}"
            if args.specialization == "warp"
            else f"sm(dispatch,gemm)=({args.dispatch_sms},{args.gemm_sms})"
        )
        print(
            f"[{module_name}] specialization={args.specialization} "
            f"world={world} routing={args.routing} {config}")
        print(
            "global_tokens local_tokens max_actual max_padded avg_ms "
            "useful_TF/GPU padded_TF/GPU")

    for num_tokens in args.shapes:
        problem = make_problem(
            mod, num_tokens, rank, world, args.routing,
            args.specialization, args.ring_blocks)
        row_counts = torch.tensor(
            [problem["actual_rows"], problem["pull"].size(0)],
            device="cuda", dtype=torch.int64)
        dist.all_reduce(row_counts, op=dist.ReduceOp.MAX)
        max_actual_rows, max_padded_rows = row_counts.cpu().tolist()

        for _ in range(args.warmup):
            reset_workspace(problem, args.specialization)
            launch(mod, problem, args, num_sms)
        torch.cuda.synchronize()
        dist.barrier()

        trial_ms = []
        for _ in range(args.trials):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                reset_workspace(problem, args.specialization)
                launch(mod, problem, args, num_sms)
            end.record()
            end.synchronize()
            rank_avg = torch.tensor(
                [start.elapsed_time(end) / args.iters],
                device="cuda", dtype=torch.float64)
            dist.all_reduce(rank_avg, op=dist.ReduceOp.MAX)
            trial_ms.append(rank_avg.item())

        avg_ms = sum(trial_ms) / len(trial_ms)
        useful_tflops = 2.0 * max_actual_rows * H * I / (avg_ms * 1e9)
        padded_tflops = 2.0 * max_padded_rows * H * I / (avg_ms * 1e9)

        if args.check:
            reset_workspace(problem, args.specialization)
            if args.specialization == "sm":
                problem["post_tokens"].zero_()
            launch(mod, problem, args, num_sms)
            torch.cuda.synchronize()
            check_result(problem, rank, world, args.specialization)

        if rank == 0:
            print(
                f"{num_tokens:13d} {num_tokens // world:12d} "
                f"{max_actual_rows:10d} {max_padded_rows:10d} "
                f"{avg_ms:7.4f} {useful_tflops:12.2f} "
                f"{padded_tflops:12.2f}")

        del problem
        torch.cuda.empty_cache()
        dist.barrier()

    if args.check and rank == 0:
        print(f"PASS: {args.specialization} specialization")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
