# Fused AG+GEMM

A fused ThunderKittens all-gather + GEMM kernel, benchmarked with a static
compute/comm CTA partition against dynamic CTA reassignment.

## Files

- `ag_gemm_h100_common.cuh` — shared fused-kernel helpers and TK primitives
- `ag_gemm_h100_static.cu` — static CTA partition baseline
- `ag_gemm_h100_dynamic.cu` — dynamic CTA scheduler
- `build_utils.py` — extension builder/importer
- `benchmark_fused_dynamic_sm_ag_gemm.py` — correctness check and performance sweep

## Run

```bash
torchrun --nproc_per_node=8 \
  dynamic_sm_allocation/ag_gemm/benchmark_fused_dynamic_sm_ag_gemm.py \
  --arch sm_90a --check
```

Compilation takes several minutes; set `MAX_JOBS` to parallelize it. The
benchmark writes a row per static and dynamic configuration into `results/`,
plus a summary CSV comparing the best dynamic result against the best static
split for each shape and synthetic delay setting.
