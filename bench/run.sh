#!/bin/bash
# bench/run.sh — multi-node launcher for the 5 release kernels.
#
# Usage:
#   bash run.sh <kernel|all> [check|bench] [num_nodes] [shapes_csv]
#
# Examples:
#   bash run.sh dispatch_gemm bench               # default 2 nodes
#   bash run.sh ag_gemm check 2
#   bash run.sh all bench 2 4096,8192
#   NUM_NODES=2 bash run.sh all bench             # via env, equivalent
#
# Multi-node WIP: only num_nodes=2 is fully validated. The bench layer
# resolves NUM_NODES (CLI arg > env var > default 2) and prints a WIP
# warning for >2 (see bench/common.py:get_num_nodes); the launcher itself
# refuses >2 because it only knows about NODE0/NODE1 SSH endpoints.
#
# Cluster (override via env if it changes):
#   NODE0_IP   — private IP of this (master) node
#   NODE1_IP   — private IP of the peer node (used as torchrun MASTER target)
#   NODE1_SSH  — host this script SSHs into to launch node 1 (public or private)
set -u

KERNEL=${1:?usage: $0 <kernel|all> [check|bench] [num_nodes] [shapes_csv]}
MODE=${2:-bench}
# Third positional is num_nodes if numeric, else treated as shapes_csv (back-compat
# with the old `<kernel> <mode> [shapes_csv]` 3-arg form).
if [[ "${3:-}" =~ ^[0-9]+$ ]]; then
    NUM_NODES=${3}
    SHAPES=${4:-}
else
    NUM_NODES=${NUM_NODES:-2}
    SHAPES=${3:-}
fi
export NUM_NODES

HERE=$(cd "$(dirname "$0")" && pwd)
RELEASE=$(dirname "$HERE")

# === Cluster topology ===
# 2-node defaults preserved. For N > 2 set NODE{i}_IP and NODE{i}_SSH for
# every i in [0, NUM_NODES). NODE0_SSH defaults to "" because node 0 is the
# launching host (we run its torchrun locally instead of SSHing to it).
NODE0_IP=${NODE0_IP:-172.31.1.237}
NODE0_SSH=${NODE0_SSH:-}
NODE1_IP=${NODE1_IP:-172.31.11.6}
NODE1_SSH=${NODE1_SSH:-15.164.130.63}
# Validate that every peer node has its IP + SSH endpoint configured.
for ((i=0; i<NUM_NODES; i++)); do
    ip_var="NODE${i}_IP"
    ssh_var="NODE${i}_SSH"
    if [[ -z "${!ip_var:-}" ]]; then
        echo "[run] $ip_var must be set for node $i (NUM_NODES=$NUM_NODES)." >&2
        exit 1
    fi
    if (( i > 0 )) && [[ -z "${!ssh_var:-}" ]]; then
        echo "[run] $ssh_var must be set for node $i (NUM_NODES=$NUM_NODES)." >&2
        exit 1
    fi
done

# Python venv with torch installed (EFS-shared, both nodes see this path).
PY=${PY:-/home/ubuntu/efs/yzhou/uccl/.venv/bin/python3}
TORCHRUN=${TORCHRUN:-/home/ubuntu/efs/yzhou/uccl/.venv/bin/torchrun}

# === Common env propagated to both nodes ===
COMMON_ENV=(
    "INTERNODE_BACKEND=efa"
    "LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
    "NCCL_IB_DISABLE=1"
    "NCCL_P2P_DISABLE=0"
    "NCCL_SOCKET_IFNAME=enp71s0"
    # gemm_ar (and gemm_rs) need verbs notify mode for proxy publication path.
    "MKERNEL_EFA_VERBS_NOTIFY_MODE=remote_flag"
    # gemm_ar release defaults that are still consumed by the host harness.
    "GEMM_AR_ARRIVAL_QUEUE=1"
    "GEMM_AR_INTER_SEND_SMS=4"
    "GEMM_AR_NUM_INTRA_COMM_SMS=14"
    "GEMM_AR_K_DIV=global_world"
    # NCCL's NVLS multicast claims the multicast capability for its own
    # collectives; that conflicts with DistBuffer's multicast bind
    # in the 2-node setup. Disable NCCL multicast so our user-mode bind
    # has the multicast feature exclusively.
    "NCCL_NVLS_ENABLE=0"
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    "TORCH_CUDA_ARCH_LIST=9.0a"
    "MASTER_ADDR=$NODE0_IP"
    "NUM_NODES=$NUM_NODES"
)

KERNELS_ALL="dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention"
STALE_BENCH_PATTERN='[t]orchrun|[d]ispatch_gemm_bench.py|[g]emm_rs_bench.py|[a]g_gemm_bench.py|[g]emm_ar_bench.py|[r]ing_attention_bench.py|[b]enchmark_gemm_ar_multinode.py'

if [[ "$KERNEL" == "all" ]]; then
    TARGETS="$KERNELS_ALL"
else
    TARGETS="$KERNEL"
fi

cleanup_stale_benches() {
    pkill -9 -f "$STALE_BENCH_PATTERN" 2>/dev/null || true
    for ((i=1; i<NUM_NODES; i++)); do
        local ssh_var="NODE${i}_SSH"
        local ssh_host="${!ssh_var:-}"
        [[ -z "$ssh_host" ]] && continue
        ssh -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$ssh_host" \
            "pkill -9 -f '$STALE_BENCH_PATTERN' 2>/dev/null || true" 2>/dev/null || true
    done
    sleep "${CLEANUP_SETTLE_SLEEP:-1}"
}

run_one_2node() {
    local kernel=$1
    local script="$HERE/${kernel}_bench.py"
    if [[ ! -f "$script" ]]; then
        echo "[run] no bench script for $kernel ($script)"
        return 1
    fi

    local extra_args="--mode $MODE"
    if [[ -n "$SHAPES" ]]; then
        extra_args="$extra_args --shapes $SHAPES"
    fi
    if [[ -n "${WARMUP:-}" ]]; then
        extra_args="$extra_args --warmup $WARMUP"
    fi
    if [[ -n "${ITERS:-}" ]]; then
        extra_args="$extra_args --iters $ITERS"
    fi
    if [[ "$MODE" == "bench" ]]; then
        local out_json="$HERE/results/${kernel}_efa.json"
        extra_args="$extra_args --save-json $out_json"
        # Self-contained regression check against the in-tree source-of-truth JSON.
        local ref_json="$HERE/source_of_truth/${kernel}.json"
        if [[ -f "$ref_json" ]]; then
            extra_args="$extra_args --compare-to $ref_json"
        fi
    fi

    local default_master_port=29890
    local default_tcp_port=19790
    case "$kernel" in
        dispatch_gemm)
            default_master_port=29890
            default_tcp_port=19790
            ;;
        gemm_rs)
            default_master_port=29850
            default_tcp_port=19750
            ;;
        ag_gemm)
            default_master_port=29880
            default_tcp_port=19780
            ;;
        gemm_ar)
            default_master_port=29830
            default_tcp_port=19730
            ;;
        ring_attention)
            default_master_port=29860
            default_tcp_port=18560
            ;;
    esac
    local master_port=${MPORT:-$default_master_port}
    local tcp_port_base=${TCP_PORT:-$default_tcp_port}
    local bind_retained=${MKERNEL_BIND_RETAINED_HANDLE:-1}
    local efa_num_qps_env=" MKERNEL_EFA_NUM_QPS=${MKERNEL_EFA_NUM_QPS:-8}"
    if [[ "$kernel" == "gemm_ar" && -z "${MKERNEL_EFA_NUM_QPS+x}" ]]; then
        # Match the gemm_ar source-of-truth experiment harness: leave this unset so
        # the gemm_ar module's session config uses its in-code default of 4 QPs.
        efa_num_qps_env=""
    fi
    local best_of_env=" MKERNEL_BENCH_BEST_OF_N=${MKERNEL_BENCH_BEST_OF_N:-0}"
    local env_str="${COMMON_ENV[*]} MASTER_PORT=$master_port MKERNEL_BIND_RETAINED_HANDLE=$bind_retained$efa_num_qps_env$best_of_env"
    # Optional: forward MKERNEL_DUMP_DIAG so the bench scripts can dump
    # per-proxy diagnostics.
    if [[ -n "${MKERNEL_DUMP_DIAG:-}" ]]; then
        env_str="$env_str MKERNEL_DUMP_DIAG=$MKERNEL_DUMP_DIAG"
    fi
    # No-sync (steady-state) timing is the canonical default. Forward
    # MKERNEL_BENCH_NO_SYNC if explicitly set (back-compat) and
    # MKERNEL_BENCH_LEGACY_SYNC=1 for users who want to opt into per-iter sync.
    if [[ -n "${MKERNEL_BENCH_NO_SYNC:-}" ]]; then
        env_str="$env_str MKERNEL_BENCH_NO_SYNC=$MKERNEL_BENCH_NO_SYNC"
    fi
    if [[ -n "${MKERNEL_BENCH_LEGACY_SYNC:-}" ]]; then
        env_str="$env_str MKERNEL_BENCH_LEGACY_SYNC=$MKERNEL_BENCH_LEGACY_SYNC"
    fi
    if [[ -n "${MKERNEL_GEMM_AR_ITERS:-}" ]]; then
        env_str="$env_str MKERNEL_GEMM_AR_ITERS=$MKERNEL_GEMM_AR_ITERS"
    fi
    if [[ -n "${MKERNEL_GEMM_AR_WARMUP:-}" ]]; then
        env_str="$env_str MKERNEL_GEMM_AR_WARMUP=$MKERNEL_GEMM_AR_WARMUP"
    fi
    if [[ -n "${MKERNEL_PREP_EPOCH_FAST:-}" ]]; then
        env_str="$env_str MKERNEL_PREP_EPOCH_FAST=$MKERNEL_PREP_EPOCH_FAST"
    fi
    if [[ -n "${MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET:-}" ]]; then
        env_str="$env_str MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET=$MKERNEL_COMMIT_EPOCH_SKIP_ARRIVAL_RESET"
    fi
    # Allow per-shape SM-split overrides for gemm_ar tuning sweeps. These
    # take precedence over the COMMON_ENV defaults because later assignments
    # in the env_str win.
    if [[ -n "${GEMM_AR_NUM_INTRA_COMM_SMS_OVERRIDE:-}" ]]; then
        env_str="$env_str GEMM_AR_NUM_INTRA_COMM_SMS=$GEMM_AR_NUM_INTRA_COMM_SMS_OVERRIDE"
    fi
    if [[ -n "${GEMM_AR_INTER_SEND_SMS_OVERRIDE:-}" ]]; then
        env_str="$env_str GEMM_AR_INTER_SEND_SMS=$GEMM_AR_INTER_SEND_SMS_OVERRIDE"
    fi
    if [[ -n "${GEMM_AR_NUM_COMM_SMS_OVERRIDE:-}" ]]; then
        env_str="$env_str GEMM_AR_NUM_COMM_SMS=$GEMM_AR_NUM_COMM_SMS_OVERRIDE"
    fi
    if [[ -n "${GEMM_AR_RDMA_CHUNK_TILES_RT:-}" ]]; then
        env_str="$env_str GEMM_AR_RDMA_CHUNK_TILES_RT=$GEMM_AR_RDMA_CHUNK_TILES_RT"
    fi
    if [[ -n "${MKERNEL_DISPATCH_GEMM_ROUTING:-}" ]]; then
        env_str="$env_str MKERNEL_DISPATCH_GEMM_ROUTING=$MKERNEL_DISPATCH_GEMM_ROUTING"
    fi
    if [[ -n "${MKERNEL_DISPATCH_GEMM_ROUTING_SEED:-}" ]]; then
        env_str="$env_str MKERNEL_DISPATCH_GEMM_ROUTING_SEED=$MKERNEL_DISPATCH_GEMM_ROUTING_SEED"
    fi
    if [[ -n "${Q2_EPOCH_TIMING:-}" ]]; then
        env_str="$env_str Q2_EPOCH_TIMING=$Q2_EPOCH_TIMING"
    fi
    # team_v8: forward MKERNEL_GEMM_RS_SPLIT_<M> overrides for sweep tuning.
    for var in $(compgen -e | grep '^MKERNEL_GEMM_RS_SPLIT_' || true); do
        env_str="$env_str ${var}=${!var}"
    done
    # team_v13: forward AG_GEMM_INTRA_OVERRIDE_<M> for ag_gemm sweep tuning.
    for var in $(compgen -e | grep '^AG_GEMM_INTRA_OVERRIDE_' || true); do
        env_str="$env_str ${var}=${!var}"
    done
    # Hard timeout (default 5 min per kernel sweep — adjust via TIMEOUT env).
    local TIMEOUT_S=${TIMEOUT:-300}
    echo "==== $kernel ($MODE) timeout=${TIMEOUT_S}s master_port=${master_port} tcp_port=${tcp_port_base} ===="
    cleanup_stale_benches

    # Launch peer nodes (rank 1..NUM_NODES-1) over SSH; node 0 runs locally.
    declare -a peer_pids=()
    for ((i=1; i<NUM_NODES; i++)); do
        local ssh_var="NODE${i}_SSH"
        local ssh_host="${!ssh_var}"
        local launch_peer="cd '$RELEASE' && $env_str NODE_IDX=$i \
            TCP_PORT=$tcp_port_base \
            '$TORCHRUN' --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=$i \
                --master_addr=$NODE0_IP --master_port=$master_port \
                '$script' $extra_args"
        timeout "${TIMEOUT_S}" ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
            "$ssh_host" "bash -c \"$launch_peer\"" \
            > "/tmp/${kernel}_node${i}.log" 2>&1 &
        peer_pids+=("$!")
    done
    sleep "${NODE1_LAUNCH_SLEEP:-2}"

    # Launch node 0 locally.
    local launch_node0="cd '$RELEASE' && $env_str NODE_IDX=0 \
        TCP_PORT=$tcp_port_base \
        '$TORCHRUN' --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=0 \
            --master_addr=$NODE0_IP --master_port=$master_port \
            '$script' $extra_args"
    timeout "${TIMEOUT_S}" bash -c "$launch_node0" 2>&1 | tee "/tmp/${kernel}_node0.log"
    local RC0=${PIPESTATUS[0]}

    # Wait for peer nodes and collect their return codes.
    local rc_sum=$RC0
    local timed_out_any=$([[ "$RC0" == "124" ]] && echo 1 || echo 0)
    local rc_summary="rc0=$RC0"
    for ((i=1; i<NUM_NODES; i++)); do
        local pid="${peer_pids[$((i-1))]}"
        wait "$pid"
        local rc=$?
        rc_sum=$((rc_sum + rc))
        rc_summary="$rc_summary  rc${i}=$rc"
        [[ "$rc" == "124" ]] && timed_out_any=1
    done
    if (( timed_out_any )); then
        echo "==== $kernel TIMEOUT after ${TIMEOUT_S}s ($rc_summary) ===="
        # Hard kill any stragglers locally and on every peer.
        pkill -9 -f "${kernel}_bench.py" 2>/dev/null
        for ((i=1; i<NUM_NODES; i++)); do
            local ssh_var="NODE${i}_SSH"
            ssh -o BatchMode=yes "${!ssh_var}" "pkill -9 -f ${kernel}_bench.py" 2>/dev/null || true
        done
    fi
    cleanup_stale_benches
    echo "==== $kernel done ($rc_summary) ===="
    return $rc_sum
}

OVERALL=0
for t in $TARGETS; do
    run_one_2node "$t" || OVERALL=$?
done
exit $OVERALL
