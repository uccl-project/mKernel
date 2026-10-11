"""GPU regression for AG-GEMM epochs (run on one Blackwell NVLink node).

make GPU=blackwell ag-gemm-warp-specialized
python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ag_gemm_epoch_test.py

Covers both transports, back-to-back eager/graph launches with fixed inputs,
input changes between globally completed batches, rank skew, and uint32 epoch
wraparound. The world size must match the extension's INTRA_NUM_DEVICES build
setting. Pull source shards must remain unchanged until every peer is done.
"""
import os
from datetime import timedelta
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import load_module


def check_shape(mod, rank, world, n, initial_epoch):
    local_m, k = 128, 64
    a = mod.DistBuffer(
        (world, local_m, k), dtype=torch.bfloat16,
        local_rank=rank, local_world_size=world, multicast=True,
    )
    ready = mod.DistBuffer(
        (mod.ag_gemm_warp_specialized_ready_words,), dtype=torch.int32,
        local_rank=rank, local_world_size=world, multicast=True,
    )
    # Negative seeds exercise uint32 wraparound in eager launches and graph replay.
    ready.data_.fill_(initial_epoch)
    a.data_.zero_()
    b = torch.ones((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((world, local_m, n), dtype=torch.bfloat16, device="cuda")
    snapshots = [torch.empty_like(c) for _ in range(4)]
    replay_results = [torch.empty_like(c) for _ in range(12)]
    expected_rows = torch.arange(1, world + 1, device="cuda", dtype=torch.float32)
    expected_rows = expected_rows[:, None, None]
    torch.cuda.synchronize()
    dist.barrier()

    def launch():
        mod.ag_gemm_warp_specialized_prepare(ready, world * local_m, n)
        mod.ag_gemm_warp_specialized_launch(a, ready, b, c)

    def check(tensor, offset):
        expected = ((expected_rows + offset) * k).to(torch.bfloat16).expand_as(tensor)
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    def check_epoch(calls):
        expected_epoch = (initial_epoch + calls + 2**31) % 2**32 - 2**31
        assert ready.data_[world].item() == expected_epoch

    graph = None
    for batch, offset in enumerate((0, 8)):
        # The preceding synchronize + barrier protects old source shards.
        # Deliberately delay this batch's input write on one rank. There is no
        # host barrier after the write: prepare must establish input readiness.
        if rank == batch % world:
            torch.cuda._sleep(100_000)
        a.data_[rank].fill_(rank + 1 + offset)

        # No per-invocation host synchronization. Only the next prepare's
        # entry barrier prevents a fast multicast sender overwriting a peer.
        for i in range(4):
            if rank == i % world:
                torch.cuda._sleep(100_000)
            c.zero_()
            launch()
            snapshots[i].copy_(c)
        torch.cuda.synchronize()
        for snapshot in snapshots:
            check(snapshot, offset)
        check_epoch(batch * 16 + 4)
        dist.barrier()

        if graph is None:
            # Capture must not consume an epoch; each replay consumes two.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                c.zero_()
                launch()
                snapshots[0].copy_(c)
                c.zero_()
                launch()
                snapshots[1].copy_(c)
            torch.cuda.synchronize()
            dist.barrier()
            check_epoch(4)

        for i in range(6):
            if rank == i % world:
                torch.cuda._sleep(100_000)
            graph.replay()
            replay_results[2 * i].copy_(snapshots[0])
            replay_results[2 * i + 1].copy_(snapshots[1])
        torch.cuda.synchronize()
        for result in replay_results:
            check(result, offset)
        check_epoch((batch + 1) * 16)
        # All peers must finish pulling before changing or freeing sources.
        dist.barrier()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    mod = load_module.load("ag_gemm_warp_specialized")
    assert mod.ag_gemm_warp_specialized_ready_words == world + 1
    for n in (256, 6400):  # multicast push, then pull
        for initial_epoch in (0, -2, -6):
            check_shape(mod, rank, world, n, initial_epoch)
    if rank == 0:
        print("AG-GEMM epoch regression passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
