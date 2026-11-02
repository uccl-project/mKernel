#!/bin/bash
# Nsight Compute A/B of the mKernel GEMM+RS main kernel across MMA warp counts.
#
# Usage (from the repo root, after make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell):
#   bash bench/ncu_gemm_rs_blackwell.sh
#   M=8192 SWIZZLE=1 SIGNALS=2 RS_ROWS=16 MMA_WARPS_LIST="1 2" bash bench/ncu_gemm_rs_blackwell.sh
#
# Writes one report per MMA warp count, taken from rank $REPORT_RANK:
#   $OUT_DIR/gemm_rs_M8192_sw1_sig2_rs16_mma1.ncu-rep
#   $OUT_DIR/gemm_rs_M8192_sw1_sig2_rs16_mma2.ncu-rep
# Every rank's report lands in $OUT_DIR/<tag>/rank<r>.ncu-rep. Compare with
#   ncu-ui <mma1>.ncu-rep <mma2>.ncu-rep   (then set one as the baseline)
#
# Every rank runs under its own ncu, since the kernel spins on peers' ready
# counters. Kernel replay cannot save the multicast/VMM DistBuffers ("Failed to
# save memory for replay"), so the default is application replay: each ncu
# reruns its whole rank once per metric pass, and the driver gives every pass
# fresh store/broker keys. REPLAY_MODE=kernel keeps the lockstep kernel-replay
# path for builds where memory save works. Only fused_kernel_stub inside the
# "ncu_target" NVTX range is profiled; exit_epoch_barrier and warmup are not.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/bench/$(basename "${BASH_SOURCE[0]}")"

M=${M:-8192}
SWIZZLE=${SWIZZLE:-1}
SIGNALS=${SIGNALS:-2}
RS_ROWS=${RS_ROWS:-16}
MMA_WARPS_LIST=${MMA_WARPS_LIST:-"1 2"}
NPROC=${NPROC:-8}
WARMUP=${WARMUP:-5}
REPORT_RANK=${REPORT_RANK:-0}
NCU_SET=${NCU_SET:-full}
NCU=${NCU:-ncu}
PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-$ROOT/reports/ncu}
NCU_PORT=${NCU_PORT:-49217}
REPLAY_MODE=${REPLAY_MODE:-application}

# Per-rank mode: torchrun runs this script once per rank with LOCAL_RANK set.
if [[ -n "${NCU_RANK_MODE:-}" ]]; then
    if [[ "$REPLAY_MODE" == kernel ]]; then
        REPLAY_ARGS=(--replay-mode kernel --communicator tcp
            --communicator-tcp-num-peers "$WORLD_SIZE"
            --communicator-tcp-port "$NCU_PORT" --lockstep-kernel-launch)
    else
        REPLAY_ARGS=(--replay-mode application --app-replay-match grid)
    fi
    exec "$NCU" \
        --set "$NCU_SET" \
        --import-source yes \
        --nvtx --nvtx-include "ncu_target/" \
        --kernel-name regex:fused_kernel_stub \
        --launch-count 1 \
        "${REPLAY_ARGS[@]}" \
        --force-overwrite \
        -o "$NCU_TAG_DIR/rank${LOCAL_RANK}" \
        "$PYTHON" "$ROOT/bench/gemm_rs_blackwell_ncu.py" \
        --M "$M" --warmup "$WARMUP" --swizzle "$SWIZZLE" \
        --signals-per-tile "$SIGNALS" --rs-strip-rows "$RS_ROWS" --mma-warps "$MMA_WARPS"
fi

cd "$ROOT"
mkdir -p "$OUT_DIR"
for MMA_WARPS in $MMA_WARPS_LIST; do
    TAG="gemm_rs_M${M}_sw${SWIZZLE}_sig${SIGNALS}_rs${RS_ROWS}_mma${MMA_WARPS}"
    TAG_DIR="$OUT_DIR/$TAG"
    mkdir -p "$TAG_DIR"
    rm -f "$TAG_DIR"/rank*.pass
    echo "=== $TAG ===" >&2
    NCU_RANK_MODE=1 NCU_TAG_DIR="$TAG_DIR" NCU_PASS_DIR="$TAG_DIR" MMA_WARPS="$MMA_WARPS" \
        REPLAY_MODE="$REPLAY_MODE" \
        "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node="$NPROC" --max-restarts=0 --no-python \
        bash "$SELF"
    cp "$TAG_DIR/rank${REPORT_RANK}.ncu-rep" "$OUT_DIR/$TAG.ncu-rep"
    echo "wrote $OUT_DIR/$TAG.ncu-rep" >&2
done
