# Ring Attention

Is the optimal SM split between KV-ring communication and local attention
workload-dependent, and can a dynamic queue policy match the best static split?

## Kernel design

Each device sends its local KV tiles to the next device in the ring while
computing partial attention over the KV it currently holds. Both phases read
from the same double-buffered KV stores (K0/K1, V0/V1), so there is no
producer→consumer dependency between them and CTAs can interleave freely.

| Variant | SM allocation |
|---------|---------------|
| `static` | First `num_comm_sms` CTAs (must be even) transfer KV; rest compute partial attention |
| `dynamic` | `comp_only_ctas` CTAs always compute; `reserved_comm_ctas` CTAs always transfer; middle CTAs claim whichever queue has work |

Comm slots come in pairs — even slot transfers K, odd slot transfers V —
preserving the static kernel's K/V pairing convention.

## Files

- `ring_attn.cu` — static baseline
- `dynamic_ring_attn.cu` — dynamic SM partitioning kernel
- `benchmark_ring_attn_dynamic_sm.py` — sweep and timing harness

## Run

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29611 \
  dynamic_sm_allocation/ring_attn/benchmark_ring_attn_dynamic_sm.py \
  --arch sm_90a
```
