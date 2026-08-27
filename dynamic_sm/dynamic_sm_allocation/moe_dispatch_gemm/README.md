# MoE Dispatch+GEMM

Does dynamically partitioning SMs between token dispatch (cross-device
all-to-all) and expert GEMM ever beat a statically tuned split?

## Kernel design

| Variant | SM allocation |
|---------|---------------|
| `static` | Fixed `num_comm_sms` CTAs for dispatch, rest for GEMM |
| `dynamic` | `comp_only_ctas` CTAs always do GEMM; `reserved_comm_ctas` CTAs always dispatch; middle CTAs steal a dispatch task if one is available, else do GEMM |

The dynamic variant assigns dispatch tasks through a global atomic counter
(`next_dispatch`) so any CTA can claim any block of tokens, with GEMM tasks
claimed from a second counter (`next_gemm`). This removes the static kernel's
fixed SM-index-to-token-range coupling.

## Files

- `moe_dispatch_gemm.cu` — static baseline
- `dynamic_moe_dispatch_gemm.cu` — dynamic SM partitioning kernel
- `moe_dispatch_gemm_dynamic_sm_allocation.cu` — adaptive-tile variant
- `benchmark_moe_dynamic_sm.py` — sweep and timing harness

## Run

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29610 \
  dynamic_sm_allocation/moe_dispatch_gemm/benchmark_moe_dynamic_sm.py \
  --arch sm_90a
```
