# GEMM+AR

Measures how sensitive the fused `gemm_ar` kernel is to its static
`num_comm_sms` partition, and compares that sweep against the dynamic
scheduler. Both `gemm_ar.cu` and `dynamic_gemm_ar.cu` live here so the static
and dynamic baselines are built from the same sources.

## Files

- `gemm_ar.cu` — static baseline
- `dynamic_gemm_ar.cu` — dynamic CTA scheduler
- `benchmark_gemm_ar_sm_sweep.py` — sweep and timing harness

## Run

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  dynamic_sm_allocation/gemm_ar/benchmark_gemm_ar_sm_sweep.py --arch sm_90a
```

Per-configuration rows go to `results/gemm_ar_sm_sweep.csv`; the shape-level
best-static-vs-best-dynamic comparison goes to
`results/gemm_ar_sm_sweep_summary.csv`.
