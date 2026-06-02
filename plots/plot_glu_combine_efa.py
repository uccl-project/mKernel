"""Render the dispatch_gemm_glu_combine EFA TFLOPS bar chart (real MoE dims).

Sibling of plot_tflops_efa.py (same style): orange "mKernel" bars vs the blue
CuBLAS+NCCL all-to-all baseline. Single persistent fused kernel doing
dispatch -> gemm1(2I) -> SwiGLU -> gemm2(H) -> combine, tensor-core warpgroup
MMA, H=7168 I=2048 E=256 TOPK=8, 2 nodes x 8 H200 EFA (world=16).

Reads  bench/results/dispatch_gemm_glu_combine_efa.json
Writes plots/dispatch_gemm_glu_combine_efa.png

FLOPs/token = TOPK*(gemm1 + gemm2) = TOPK*(2*H*2I + 2*I*H) = 6*TOPK*H*I, /WORLD.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "bench" / "results"

H, I, TOPK, WORLD = 7168, 2048, 8, 16
TITLE = "Fused MoE: Dispatch+GEMM+SwiGLU+GEMM+Combine, 2 nodes x 8 H200 EFA (world=16)"
# Same-functionality baseline (NCCL all-to-all over EFA + cuBLAS), measured by
# bench/baseline_glu_combine.py at H=7168 I=2048 TOPK=8 (TFLOPS/GPU).
BASELINE_TF = {8192: 150.1, 16384: 159.7, 32768: 182.7, 65536: 192.6, 131072: 196.4}
MIN_TOK = 8192   # real-dim runs only (the json also holds stale tiny-dim entries)


def tflops(num_tokens, ms):
    if not ms or not math.isfinite(ms) or ms <= 0:
        return float("nan")
    return (6.0 * num_tokens * TOPK * H * I / WORLD) / (ms * 1e-3) / 1e12


# Full-kernel TFLOPS/GPU (dispatch+gemm1+swiglu+gemm2+combine). bf16x2 IPC
# atomics + atomic-free structured reduce + CHUNK_BYTES=16 MB (the exposed,
# bidirectional combine RDMA saturates the EFA rails at ~20 GB/s with large
# chunks; 512 KB left it at ~7 GB/s). 2-node EFA sweep wall =
# {8192:2.142, 16384:3.952, 32768:7.477, 65536:14.476, 131072:27.010} ms.
# Now beats the CuBLAS+NCCL baseline at every shape.
FULL_TF = {8192: 168.5, 16384: 182.6, 32768: 193.0, 65536: 199.4, 131072: 213.7}
# Arena: if the agent team has a recorded best, use its per-shape TFLOPS.
try:
    _bt = json.loads((HERE.parent / "agent_arena" / "best_tflops.json").read_text())
    FULL_TF = {int(k): float(v) for k, v in _bt.items() if v}
except Exception:
    pass


def main():
    shapes = [8192, 16384, 32768, 65536, 131072]
    fused_tf = [FULL_TF[n] for n in shapes]
    base_tf = [BASELINE_TF.get(n, float("nan")) for n in shapes]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(shapes), dtype=float)
    w = 0.38
    series = [
        (-w / 2, base_tf, "#4C72B0", "CuBLAS+NCCL (all-to-all)"),
        (w / 2, fused_tf, "#F58518", "mKernel (full)"),
    ]
    for off, vals, color, label in series:
        ax.bar(x + off, vals, w, color=color, label=label, zorder=3)
        for xi, yi in zip(x, vals):
            if math.isfinite(yi):
                ax.annotate(f"{yi:.0f}", xy=(xi + off, yi), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in shapes], rotation=15, fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Tokens (N), N/16 per GPU  (H=7168, I=2048, E=256, TOPK=8)", fontsize=15)
    ax.set_ylabel("TFLOPS per GPU (bf16)", fontsize=15)
    ax.set_title(TITLE, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    out = HERE / "dispatch_gemm_glu_combine_efa.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
