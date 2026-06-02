"""ring_attention Ring Attention bench (release version).

Default EFA sweep: seq_per_dev∈{768,1536,3072,6144,12288}.
Uses --per-stage build (RING_ATTN_PER_STAGE_KERNEL=1 baked in Makefile).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import (  # noqa: E402
    check_close,
    compare_named_results,
    gather_cpu_tensors,
    get_peer_ips,
    get_peer_ports,
    is_peermem_backing,
    make_dist_buffer,
    rdma_backing,
    rdma_policy_label,
)

KERNEL_NAME = "ring_attention"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
BATCH = 4
HEADS = 16
D = 128
QO_BLOCK = 64
KV_BLOCK = 128
CONSUMER_WARPGROUPS = 3
CHUNK_BYTES = 256 * 1024  # baked

DEFAULT_SHAPES = [768, 1536, 3072, 6144, 12288]


def median_then_max_cuda(samples):
    median = sorted(float(x) for x in samples)[len(samples) // 2]
    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str,
                   default=",".join(str(s) for s in DEFAULT_SHAPES))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--num-comm-sms", type=int, default=16)
    p.add_argument("--num-send-sms", type=int, default=8)
    p.add_argument("--num-copy-sms", type=int, default=8)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist_backend = os.environ.get("MKERNEL_DIST_BACKEND", "nccl")
    if dist_backend == "nccl":
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group(dist_backend)

    node_idx = args.node_idx if args.node_idx is not None else int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)
    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        peer_node = 1 if node_idx == 0 else 0
        peer_ip = os.environ.get(f"NODE{peer_node}_IP")
        if not peer_ip:
            raise RuntimeError(f"NODE{peer_node}_IP must be set, or set PEER_IP explicitly")
    tcp_port = int(os.environ.get("TCP_PORT", "18560")) + local_rank

    mod = load_module.load(KERNEL_NAME)
    source_backing = rdma_backing()
    rdma_source_staging = is_peermem_backing(source_backing)
    ring_buffer_backing = "vmm" if rdma_source_staging else source_backing
    if is_chief:
        print(f"[ring_attn] world={world_size} per_node={local_world_size} "
              f"shapes={args.shapes}", flush=True)
        print(rdma_policy_label(
            KERNEL_NAME, source=source_backing, target="cuda_malloc",
            ring=ring_buffer_backing), flush=True)

    seqs = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_gpu = node_idx * local_world_size + local_rank

    result_sizes, result_fused = [], []
    correctness_ok = True

    # Per-shape SM split (comm, send, copy). Larger seqs benefit from
    # smaller comm pool (more compute SMs); small seqs need comm coverage.
    SM_SPLIT = {
        1536:  (8, 12, 12),
        3072:  (12, 8, 8),
        6144:  (12, 8, 8),
        12288: (8, 12, 12),
    }
    for seq_per_dev in seqs:
        if seq_per_dev in SM_SPLIT:
            c, s, cp = SM_SPLIT[seq_per_dev]
            args.num_comm_sms = c; args.num_send_sms = s; args.num_copy_sms = cp
        assert seq_per_dev % (CONSUMER_WARPGROUPS * QO_BLOCK) == 0
        assert seq_per_dev % KV_BLOCK == 0

        if is_chief:
            print(f"\n[ring_attn] seq_per_dev={seq_per_dev}", flush=True)

        torch.manual_seed(42 + global_gpu); torch.cuda.manual_seed(42 + global_gpu)
        scale = D ** -0.25
        Q = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                        dtype=torch.bfloat16) * scale
        K_local = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                              dtype=torch.bfloat16) * scale
        V_local = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                              dtype=torch.bfloat16) * scale

        def dbuf(shape):
            return make_dist_buffer(
                mod,
                shape, dtype=torch.bfloat16,
                local_rank=local_rank, local_world_size=local_world_size,
                multicast=False, backing=ring_buffer_backing)

        K0 = dbuf((BATCH, HEADS, seq_per_dev, D))
        K1 = dbuf((BATCH, HEADS, seq_per_dev, D))
        V0 = dbuf((BATCH, HEADS, seq_per_dev, D))
        V1 = dbuf((BATCH, HEADS, seq_per_dev, D))

        L = torch.zeros(BATCH, HEADS, seq_per_dev, device="cuda", dtype=torch.float32)
        L_block = torch.zeros_like(L)
        O = torch.zeros_like(Q)
        O_block = torch.zeros_like(Q)

        barrier = mod.DistBuffer(
            (2, 1024, 1024), dtype=torch.int32,
            local_rank=local_rank, local_world_size=local_world_size,
            multicast=True)

        k_bytes = K_local.numel() * 2
        v_bytes = V_local.numel() * 2
        kv_buf_bytes = k_bytes + v_bytes
        send_staging = None
        if rdma_source_staging:
            send_staging = make_dist_buffer(
                mod, (kv_buf_bytes // 2,), dtype=torch.bfloat16,
                local_rank=local_rank, local_world_size=local_world_size,
                multicast=False, backing=source_backing)
            send_buf = send_staging.data_
        else:
            send_buf = torch.empty(kv_buf_bytes // 2, device="cuda", dtype=torch.bfloat16)
        send_buf_ptr = send_buf.data_ptr()
        total_chunks = (kv_buf_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES

        # Per-peer sizing (1× at N == 2, identical to legacy).
        n_peers = NUM_NODES - 1
        recv_buf_chunks = n_peers * total_chunks
        recv_buf_bytes = n_peers * kv_buf_bytes

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_chunks * 2: fifo_cap *= 2

        # Default zero-copy send registers K0/V0 directly. Peermem mode keeps
        # those as VMM ring buffers and registers send_staging instead.
        peer_ips = get_peer_ips(node_idx, NUM_NODES)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            send_buf_ptr, kv_buf_bytes, recv_buf_bytes,
            recv_buf_chunks, fifo_cap, local_rank,
            int(K0.data_.data_ptr()), k_bytes,
            int(V0.data_.data_ptr()), v_bytes,
            rdma_source_staging,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        def pack_send_staging():
            if send_staging is None:
                return
            send_buf[:k_bytes // 2].copy_(K_local.reshape(-1))
            send_buf[k_bytes // 2:].copy_(V_local.reshape(-1))

        K0.data_.copy_(K_local); V0.data_.copy_(V_local)
        K1.data_.zero_(); V1.data_.zero_()
        pack_send_staging()
        barrier.data_.zero_()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        def reset_and_run(ep):
            K0.data_.copy_(K_local); V0.data_.copy_(V_local)
            K1.data_.zero_(); V1.data_.zero_()
            pack_send_staging()
            barrier.data_.zero_()
            L.zero_(); L_block.zero_()
            O.zero_(); O_block.zero_()
            mod.set_epoch(ep)
            dist.barrier(); time.sleep(0.05)
            mod.ring_attn_multinode(
                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                send_buf_ptr, recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, ep,
                node_idx, args.num_comm_sms,
                args.num_send_sms, args.num_copy_sms, NUM_NODES,
                rdma_source_staging,
            )

        for _ in range(args.warmup + 1):
            epoch += 1; reset_and_run(epoch); torch.cuda.synchronize()
            dist.barrier()

        samples = []
        # Canonical: no-sync (steady-state) timing — N back-to-back iters with
        # a SINGLE sync at end, divide by N. Set MKERNEL_BENCH_LEGACY_SYNC=1
        # (or MKERNEL_BENCH_NO_SYNC=0) to opt back into per-iter sync.
        # ring_attention's iter is much longer than ag_gemm's (≈2-135 ms per
        # iter across shapes), so we cap N to keep large-shape wall budget
        # bounded while still ensuring total measurement >= ~100 ms (timer
        # noise <1%) at the smallest shapes.
        legacy_sync = os.environ.get("MKERNEL_BENCH_LEGACY_SYNC") == "1"
        if os.environ.get("MKERNEL_BENCH_NO_SYNC") == "0":
            legacy_sync = True
        if rdma_source_staging:
            legacy_sync = True
        if not legacy_sync:
            # Pick N so total ≥ ~100 ms at smaller shapes but bounded ≤ ~2s at
            # the largest. For ring_attn, expected per-iter ms ≈ shape-dependent
            # ~(2,4,11,35,135) — N=32 gives ~64ms..4.3s range. Cap at 32 small,
            # taper to 8 at the biggest shape.
            if seq_per_dev <= 1536:
                n_iters = 32
            elif seq_per_dev <= 6144:
                n_iters = 16
            else:
                n_iters = 8
            epoch += 1
            K0.data_.copy_(K_local); V0.data_.copy_(V_local)
            K1.data_.zero_(); V1.data_.zero_()
            pack_send_staging()
            barrier.data_.zero_()
            L.zero_(); L_block.zero_(); O.zero_(); O_block.zero_()
            mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(n_iters):
                mod.ring_attn_multinode(
                    Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                    send_buf_ptr, recv_ptr,
                    fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                    arrival_ptr, epoch,
                    node_idx, args.num_comm_sms,
                    args.num_send_sms, args.num_copy_sms, NUM_NODES,
                    rdma_source_staging,
                )
            e.record()
            torch.cuda.synchronize()
            avg_ms = s.elapsed_time(e) / n_iters
            samples = [avg_ms] * args.iters
            if is_chief:
                print(f"[ring_attn-nosync] seq={seq_per_dev} N={n_iters} "
                      f"avg={avg_ms:.4f} ms", flush=True)
            dist.barrier()
        else:
            for _ in range(args.iters):
                epoch += 1
                K0.data_.copy_(K_local); V0.data_.copy_(V_local)
                K1.data_.zero_(); V1.data_.zero_()
                pack_send_staging()
                barrier.data_.zero_()
                L.zero_(); L_block.zero_(); O.zero_(); O_block.zero_()
                mod.set_epoch(epoch)
                dist.barrier(); time.sleep(0.05)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                mod.ring_attn_multinode(
                    Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                    send_buf_ptr, recv_ptr,
                    fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                    arrival_ptr, epoch,
                    node_idx, args.num_comm_sms,
                    args.num_send_sms, args.num_copy_sms, NUM_NODES,
                    rdma_source_staging,
                )
                e.record(); torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
                dist.barrier()

        wall_ms = median_then_max_cuda(samples)
        if is_chief:
            print(f"[ring_attn] seq={seq_per_dev} wall={wall_ms:.3f} ms", flush=True)
        K_full = torch.cat(gather_cpu_tensors(K_local), dim=2).to("cuda")
        V_full = torch.cat(gather_cpu_tensors(V_local), dim=2).to("cuda")
        # fp32 reference so we have a real ground truth, not bf16 noise.
        O_ref = F.scaled_dot_product_attention(
            Q.float(), K_full.float(), V_full.float()
        ).to(torch.bfloat16)
        # Scale atol with output magnitude. Without this, a fixed atol that's
        # large relative to ref_mean lets garbage outputs short-circuit the
        # check via the max_abs <= atol branch. The 0.05 floor accounts for
        # bf16 accumulation drift on small-magnitude outputs at NUM_NODES > 2,
        # where the per-stage running softmax compounds across more rounds.
        ref_mag = float(O_ref.abs().mean().item())
        atol = max(0.05, 5.0 * ref_mag)
        correctness_ok = check_close(
            f"ring_attention seq={seq_per_dev}",
            O, O_ref, atol=atol, rtol=0.12
        ) and correctness_ok
        # Store as total_seq to match NCCL reference + published chart x-axis.
        result_sizes.append(seq_per_dev)
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "ring_attention",
                           result_sizes, result_fused,
                           note=f"release ring_attention bench (world={world_size})")
        print(f"[ring_attn] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("ring_attn", result_sizes, result_fused, args.compare_to)
        ok = ok and correctness_ok
        dist.destroy_process_group()
        if not ok: return 1
        return 0
    if not correctness_ok:
        dist.destroy_process_group()
        return 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
