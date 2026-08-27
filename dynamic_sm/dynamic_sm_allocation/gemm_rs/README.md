# GEMM+RS

Can dynamic SM allocation between GEMM compute and reduce-scatter comm beat a
strong static split across the shape and GPU-count sweeps?

## Files

- `gemm_rs.cu` — static baseline
- `dynamic_gemm_rs.cu` — dynamic SM partitioning kernel
- `benchmark_gemm_rs.py` — build, check, and benchmark harness

## Run

```bash
torchrun --nproc_per_node=8 \
  dynamic_sm_allocation/gemm_rs/benchmark_gemm_rs.py --arch sm_90a
```

Row-level data goes to `results/gemm_rs.csv` and the summary to
`results/gemm_rs_summary.csv`.
