#!/usr/bin/env python3
"""Single-GPU correctness + microbench for the Gen-15 modular grouped GEMM.

Gate 3 (correctness): bit-close vs fp32 torch per-expert grouped reference, and
super_m decode invariance (super_m in {1,8} produce identical C).
Gate 4 (microbench): isolated gemm1 rewrite (mode 1) vs current grouped_gemm
schedule (mode 0=union, byte-identical to the fused kernel). KILL if not >=~2%.
"""
import os, sys, time, importlib.util, torch

H, I2 = 7168, 4096
ROW_BLOCK, COL_BLOCK, RED_BLOCK = 128, 256, 64
E = 16

_SO = os.environ.get("GEMM_CHECK_SO", "/tmp/gemm_sm90_check.so")
_spec = importlib.util.spec_from_file_location("gemm_sm90_check", _SO)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def build_layout(rows_per_expert):
    """rows_per_expert: list[E] multiples of ROW_BLOCK. Returns padded[E],
    local_rb[E] (=total_rb, all local), total rows M."""
    padded = list(rows_per_expert)
    # local_rb = full block count per expert (all rows local -> pass1 empty)
    cum = 0
    local_rb = []
    for e in range(E):
        rb_start = cum // ROW_BLOCK
        cum += padded[e]
        rb_end = (cum + ROW_BLOCK - 1) // ROW_BLOCK
        local_rb.append(rb_end - rb_start)
    M = cum
    return padded, local_rb, M


def ref_gemm(A, Bw, padded):
    """fp32 per-expert grouped reference: C[rows_e] = A[rows_e] @ Bw[e]."""
    C = torch.zeros((A.shape[0], I2), dtype=torch.float32, device=A.device)
    off = 0
    for e in range(E):
        n = padded[e]
        if n == 0:
            continue
        C[off:off+n] = A[off:off+n].float() @ Bw[e].float()
        off += n
    return C


def run(rows_per_expert, label):
    dev = "cuda:0"
    padded, local_rb, M = build_layout(rows_per_expert)
    torch.manual_seed(0)
    A = (torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.1)
    Bw = (torch.randn(E, H, I2, device=dev, dtype=torch.bfloat16) * 0.1)
    ptpe = torch.tensor(padded, dtype=torch.int32, device=dev)
    lrb = torch.tensor(local_rb, dtype=torch.int32, device=dev)

    Cref = ref_gemm(A, Bw, padded)

    results = {}
    for mode in (0, 1):
        C = torch.zeros(M, I2, device=dev, dtype=torch.bfloat16)
        mod.run_gemm(A, Bw, C, ptpe, lrb, 8, mode)
        torch.cuda.synchronize()
        # only valid token rows are checked (all rows are valid here: padded are
        # multiples of ROW_BLOCK so no partial-block garbage).
        err = (C.float() - Cref).abs().max().item()
        results[mode] = (C, err)

    # super_m invariance for the reg path (mode 1): super_m in {1,8} identical
    C1 = torch.zeros(M, I2, device=dev, dtype=torch.bfloat16)
    mod.run_gemm(A, Bw, C1, ptpe, lrb, 1, 1)
    C8 = torch.zeros(M, I2, device=dev, dtype=torch.bfloat16)
    mod.run_gemm(A, Bw, C8, ptpe, lrb, 8, 1)
    torch.cuda.synchronize()
    sm_inv = (C1 - C8).abs().max().item()

    print(f"[{label}] M={M}  union_err={results[0][1]:.5f}  reg_err={results[1][1]:.5f}  "
          f"super_m(1vs8)_maxabs={sm_inv:.6f}")
    return results, sm_inv


def bench(rows_per_expert, label, reps=50):
    dev = "cuda:0"
    padded, local_rb, M = build_layout(rows_per_expert)
    torch.manual_seed(1)
    A = (torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.1)
    Bw = (torch.randn(E, H, I2, device=dev, dtype=torch.bfloat16) * 0.1)
    ptpe = torch.tensor(padded, dtype=torch.int32, device=dev)
    lrb = torch.tensor(local_rb, dtype=torch.int32, device=dev)
    C = torch.zeros(M, I2, device=dev, dtype=torch.bfloat16)

    def timed(mode):
        for _ in range(5):
            mod.run_gemm(A, Bw, C, ptpe, lrb, 8, mode)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(reps):
                mod.run_gemm(A, Bw, C, ptpe, lrb, 8, mode)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / reps * 1e6)
        return best

    us0 = timed(0)
    us1 = timed(1)
    speedup = (us0 - us1) / us0 * 100.0
    print(f"[{label}] union(base)={us0:.1f}us  reg(perf)={us1:.1f}us  "
          f"reg speedup={speedup:+.2f}%")
    return us0, us1, speedup


if __name__ == "__main__":
    # Skewed multinomial-ish per-expert load (the scored workload is skewed).
    # ~65k floor shape: pick a per-expert row distribution summing to a realistic
    # per-GPU token count. Use multiples of ROW_BLOCK to avoid partial blocks.
    rows = [ROW_BLOCK * n for n in [8, 6, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]]
    print("=== correctness (Gate 3) ===")
    run(rows, "skewed")
    run([ROW_BLOCK * 4] * E, "uniform")
    print("=== microbench (Gate 4) ===")
    bench(rows, "skewed-65k-ish")
    bench([ROW_BLOCK * 4] * E, "uniform")
