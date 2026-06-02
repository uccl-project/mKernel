#!/bin/bash
# bench/run.sh — multi-node launcher for the 5 release kernels.
#
# Usage:
# bash run.sh <kernel|all> [check|bench] [num_nodes] [shapes_csv]
# bash run.sh nccl_baseline bench 3 55296   # NCCL+GEMM baseline (nccl_baseline_bench.py)
#
# Examples:
# bash run.sh dispatch_gemm bench               # default NUM_NODES=2
# bash run.sh ag_gemm check 2
# bash run.sh all bench 2 4096,8192
# NUM_NODES=2 bash run.sh all bench             # via env, equivalent
#
# Cluster configuration:
# NODE{i}_IP   — data-plane IP for node i
# NODE{i}_SSH  — SSH target for peer node i (i > 0)
# NODE{i}_SSH_PORT — optional SSH port for peer node i
#
# H200 etiquette:
# Point NODE0_IP / NODE{i}_SSH at GPU-idle nodes only (no shared SGLang/training).
#
# Runtime triage (during a run): if per-GPU utilization stays >99% for ≥15s
# with no log / iter progress, treat the job as stuck (e.g. ring wait) — stop
# torchrun and debug; do not wait indefinitely.
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
# Set NODE{i}_IP for every i in [0, NUM_NODES). For peer nodes also set
# NODE{i}_SSH; NODE0_SSH defaults to "" because node 0 is launched locally.
NODE0_IP=${NODE0_IP:-}
NODE0_SSH=${NODE0_SSH:-}
NODE1_IP=${NODE1_IP:-}
NODE1_SSH=${NODE1_SSH:-}
BACKEND_ENV_SET=${BACKEND+x}
BACKEND=${BACKEND:-efa}
if [[ "${MKERNEL_H200:-0}" == "1" ]]; then
    if [[ -z "$BACKEND_ENV_SET" ]]; then
        BACKEND=cx7
    fi
    NODE0_SSH=${NODE0_SSH:-}
    for ((i=1; i<NUM_NODES; i++)); do
        port_var="NODE${i}_SSH_PORT"
        export "${port_var}=${!port_var:-2222}"
    done
    RESULT_SUFFIX=${RESULT_SUFFIX:-h200_n${NUM_NODES}}
    # H200 RoCE bootstrap must use the bond interfaces, not eth0.
    _MK_NCCL_BOND_IFS=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
    NCCL_SOCKET_IFNAME=${MKERNEL_BASELINE_NCCL_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-$_MK_NCCL_BOND_IFS}}
    NCCL_SOCKET_FAMILY=AF_INET
    NCCL_IB_ADDR_FAMILY=${MKERNEL_BASELINE_NCCL_IB_ADDR_FAMILY:-AF_INET}
    # Gloo (host broker) stays on eth0 unless overridden — do not tie to NCCL bonds.
    GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
    # Allow NCCL_IB_DISABLE=1 (e.g. profile script socket baseline) to override RoCE.
    NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    if [[ "$NCCL_IB_DISABLE" != "1" ]]; then
        NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7}
        NCCL_IB_GID_INDEX=3
        NCCL_IB_MERGE_NICS=0
        NCCL_NET_MERGE_LEVEL=LOC
        MKERNEL_ROCE_GID_INDEX=3
    fi
    MKERNEL_CHANNELIZE_GPU_PEERS=${MKERNEL_CHANNELIZE_GPU_PEERS:-1}
    MKERNEL_INTERNODE_NUM_QPS=${MKERNEL_INTERNODE_NUM_QPS:-16}
fi
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

if [[ "${MKERNEL_H200:-0}" == "1" ]]; then
    echo "[run] policy: NUM_NODES=${NUM_NODES} — use GPU-idle nodes only; set NODE0_IP to an idle chief (not a shared head)." >&2
    echo "[run] WARNING: if any selected node is busy, benchmark results may hang or be misleading." >&2
fi

# Python with torch installed. Override PY/TORCHRUN if using a virtualenv.
PY=${PY:-python3}
TORCHRUN=${TORCHRUN:-torchrun}

# === Common env propagated to both nodes ===
COMMON_ENV=(
    "INTERNODE_BACKEND=$BACKEND"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
    "NCCL_P2P_DISABLE=0"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp71s0}"
    "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}"
    "NCCL_IB_ADDR_FAMILY=${NCCL_IB_ADDR_FAMILY:-AF_INET}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-enp71s0}}"
    # gemm_ar release defaults that are still consumed by the host harness.
    "GEMM_AR_ARRIVAL_QUEUE=${GEMM_AR_ARRIVAL_QUEUE:-1}"
    "GEMM_AR_K_DIV=${GEMM_AR_K_DIV:-global_world}"
    # NCCL's NVLS multicast claims the multicast capability for its own
    # collectives; that conflicts with DistBuffer's multicast bind
    # in this setup. Disable NCCL multicast so our user-mode bind
    # has the multicast feature exclusively.
    "NCCL_NVLS_ENABLE=0"
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    "TORCH_CUDA_ARCH_LIST=9.0a"
    "MASTER_ADDR=$NODE0_IP"
    "NUM_NODES=$NUM_NODES"
    "MKERNEL_H200=${MKERNEL_H200:-0}"
)
for ((i=0; i<NUM_NODES; i++)); do
    ip_var="NODE${i}_IP"
    COMMON_ENV+=("${ip_var}=${!ip_var}")
done
if [[ -n "${GEMM_AR_INTER_SEND_SMS:-}" ]]; then
    COMMON_ENV+=("GEMM_AR_INTER_SEND_SMS=$GEMM_AR_INTER_SEND_SMS")
fi
if [[ -n "${GEMM_AR_NUM_INTRA_COMM_SMS:-}" ]]; then
    COMMON_ENV+=("GEMM_AR_NUM_INTRA_COMM_SMS=$GEMM_AR_NUM_INTRA_COMM_SMS")
fi
# Super-M L2-rasterization decode factor for dispatch_gemm_glu_combine. Forwarded
# to BOTH node launches only when explicitly set by the caller; unset => the kernel
# defaults to 8. Set DGC_SUPER_M=1 for plain row-major decode.
if [[ -n "${DGC_SUPER_M:-}" ]]; then
    COMMON_ENV+=("DGC_SUPER_M=$DGC_SUPER_M")
fi
KERNELS_ALL="dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention"
STALE_BENCH_PATTERN='[d]ispatch_gemm_bench.py|[g]emm_rs_bench.py|[a]g_gemm_bench.py|[g]emm_ar_bench.py|[r]ing_attention_bench.py|[n]ccl_baseline_bench.py|[b]enchmark_gemm_ar_multinode.py'

if [[ "$KERNEL" == "all" ]]; then
    TARGETS="$KERNELS_ALL"
else
    TARGETS="$KERNEL"
fi

cleanup_stale_benches() {
    pkill -9 -f "$STALE_BENCH_PATTERN" 2>/dev/null || true
    for ((i=1; i<NUM_NODES; i++)); do
        local ssh_var="NODE${i}_SSH"
        local ssh_port_var="NODE${i}_SSH_PORT"
        local ssh_host="${!ssh_var:-}"
        [[ -z "$ssh_host" ]] && continue
        local ssh_port_args=()
        if [[ -n "${!ssh_port_var:-}" ]]; then
            ssh_port_args=(-p "${!ssh_port_var}")
        fi
        ssh -o BatchMode=yes \
            -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "${ssh_port_args[@]}" "$ssh_host" \
            "pkill -9 -f '$STALE_BENCH_PATTERN' 2>/dev/null || true" 2>/dev/null || true
    done
    sleep "${CLEANUP_SETTLE_SLEEP:-1}"
}

run_one_2node() {
    local kernel=$1
    local script
    local extra_args=""
    # write_imm avoids the EFA SRD unordered-delivery race (data WR vs flag
    # WR), but gemm_ar's kernel polls an arrival queue and the write_imm CQE
    # handler only publishes to the flat array — so gemm_ar stays on
    # remote_flag for now.
    local kernel_notify_mode="${MKERNEL_EFA_VERBS_NOTIFY_MODE:-}"
    if [[ "$BACKEND" == "efa" && -z "$kernel_notify_mode" ]]; then
        if [[ "$kernel" == "gemm_ar" ]]; then
            kernel_notify_mode=remote_flag
        else
            kernel_notify_mode=write_imm
        fi
    fi
    if [[ "$kernel" == "nccl_baseline" ]]; then
        script="$HERE/nccl_baseline_bench.py"
        if [[ ! -f "$script" ]]; then
            echo "[run] missing $script"
            return 1
        fi
        # First positional to nccl_baseline_bench.py is kernel name (ag_gemm, ...).
        extra_args="ag_gemm"
        if [[ -n "$SHAPES" ]]; then
            extra_args="$extra_args --shapes $SHAPES"
        fi
        if [[ -n "${WARMUP:-}" ]]; then
            extra_args="$extra_args --warmup $WARMUP"
        fi
        if [[ -n "${ITERS:-}" ]]; then
            extra_args="$extra_args --iters $ITERS"
        fi
        extra_args="$extra_args --save-dir $HERE/results --tag ${NCCL_BASELINE_TAG:-nccl}"
    else
        script="$HERE/${kernel}_bench.py"
        if [[ ! -f "$script" ]]; then
            echo "[run] no bench script for $kernel ($script)"
            return 1
        fi
        extra_args="--mode $MODE"
        if [[ -n "$SHAPES" ]]; then
            extra_args="$extra_args --shapes $SHAPES"
        fi
        if [[ -n "${WARMUP:-}" ]]; then
            extra_args="$extra_args --warmup $WARMUP"
        fi
        if [[ -n "${ITERS:-}" ]]; then
            extra_args="$extra_args --iters $ITERS"
        fi
        # Optional extra CLI for ag_gemm_bench.py (e.g. --num-intra-comm-sms 32).
        if [[ "$kernel" == "ag_gemm" && -n "${AG_GEMM_BENCH_EXTRA:-}" ]]; then
            extra_args="$extra_args ${AG_GEMM_BENCH_EXTRA}"
        fi
        if [[ "$MODE" == "bench" ]]; then
            local out_suffix="${RESULT_SUFFIX:-$BACKEND}"
            local out_json="$HERE/results/${kernel}_${out_suffix}.json"
            extra_args="$extra_args --save-json $out_json"
            # Optional regression check against an in-tree reference JSON.
            local ref_json="$HERE/source_of_truth/${kernel}.json"
            if [[ -f "$ref_json" && -z "${SKIP_BENCH_COMPARE_TO:-}" ]]; then
                extra_args="$extra_args --compare-to $ref_json"
            fi
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
        nccl_baseline)
            default_master_port=29845
            default_tcp_port=19745
            ;;
    esac
    local master_port=${MPORT:-$default_master_port}
    local tcp_port_base=${TCP_PORT:-$default_tcp_port}
    local bind_retained=${MKERNEL_BIND_RETAINED_HANDLE:-1}
    local efa_num_qps_env=""
    if [[ "$BACKEND" == "efa" ]]; then
        efa_num_qps_env=" MKERNEL_EFA_NUM_QPS=${MKERNEL_EFA_NUM_QPS:-8}"
    fi
    if [[ "$BACKEND" == "efa" && "$kernel" == "gemm_ar" && -z "${MKERNEL_EFA_NUM_QPS+x}" ]]; then
        # Leave this unset so the gemm_ar module uses its in-code default of 4 QPs.
        efa_num_qps_env=""
    fi
    local best_of_env=" MKERNEL_BENCH_BEST_OF_N=${MKERNEL_BENCH_BEST_OF_N:-0}"
    local broker_key="${DIST_BROKER_KEY:-mkernel_${kernel}_${master_port}_${tcp_port_base}}"
    local env_str="${COMMON_ENV[*]} MASTER_PORT=$master_port DIST_BROKER_KEY=$broker_key MKERNEL_BIND_RETAINED_HANDLE=$bind_retained$efa_num_qps_env$best_of_env"
    if [[ -n "$kernel_notify_mode" ]]; then
        env_str="$env_str MKERNEL_EFA_VERBS_NOTIFY_MODE=$kernel_notify_mode"
    fi
    local allow_profiler_logging=1
    if [[ "$MODE" == "bench" && "${MKERNEL_ALLOW_PROFILER_LOGGING:-0}" != "1" ]]; then
        # Plot-generating benchmark runs should not carry trace/profiler output
        # from the caller's shell; opt in explicitly for diagnostic runs.
        allow_profiler_logging=0
    fi
    if [[ "$allow_profiler_logging" == "1" && -n "${MKERNEL_EFA_PROXY_DIAG:-}" ]]; then
        env_str="$env_str MKERNEL_EFA_PROXY_DIAG=$MKERNEL_EFA_PROXY_DIAG"
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
    if [[ "$allow_profiler_logging" == "1" && -n "${MKERNEL_BENCH_DUMP_RANK_MS:-}" ]]; then
        env_str="$env_str MKERNEL_BENCH_DUMP_RANK_MS=$MKERNEL_BENCH_DUMP_RANK_MS"
    fi
    if [[ -n "${MKERNEL_GEMM_AR_ITERS:-}" ]]; then
        env_str="$env_str MKERNEL_GEMM_AR_ITERS=$MKERNEL_GEMM_AR_ITERS"
    fi
    if [[ -n "${MKERNEL_GEMM_AR_WARMUP:-}" ]]; then
        env_str="$env_str MKERNEL_GEMM_AR_WARMUP=$MKERNEL_GEMM_AR_WARMUP"
    fi
    if [[ -n "${GEMM_RS_SPLIT:-}" ]]; then
        env_str="$env_str GEMM_RS_SPLIT=$GEMM_RS_SPLIT"
    fi
    if [[ -n "${GEMM_RS_NUM_COMP_SMS:-}" ]]; then
        env_str="$env_str GEMM_RS_NUM_COMP_SMS=$GEMM_RS_NUM_COMP_SMS"
    fi
    if [[ -n "${GEMM_RS_NUM_INTRA_COMM_SMS:-}" ]]; then
        env_str="$env_str GEMM_RS_NUM_INTRA_COMM_SMS=$GEMM_RS_NUM_INTRA_COMM_SMS"
    fi
    if [[ -n "${GEMM_RS_NUM_SEND_SMS:-}" ]]; then
        env_str="$env_str GEMM_RS_NUM_SEND_SMS=$GEMM_RS_NUM_SEND_SMS"
    fi
    if [[ -n "${GEMM_RS_NUM_REDUCE_SMS:-}" ]]; then
        env_str="$env_str GEMM_RS_NUM_REDUCE_SMS=$GEMM_RS_NUM_REDUCE_SMS"
    fi
    if [[ -n "${GEMM_RS_CHUNK_TILES:-}" ]]; then
        env_str="$env_str GEMM_RS_CHUNK_TILES=$GEMM_RS_CHUNK_TILES"
    fi
    if [[ -n "${GEMM_RS_READY_REDUCE_QUEUE:-}" ]]; then
        env_str="$env_str GEMM_RS_READY_REDUCE_QUEUE=$GEMM_RS_READY_REDUCE_QUEUE"
    fi
    if [[ -n "${GEMM_RS_TRANSPORT_ARRIVAL_QUEUE:-}" ]]; then
        env_str="$env_str GEMM_RS_TRANSPORT_ARRIVAL_QUEUE=$GEMM_RS_TRANSPORT_ARRIVAL_QUEUE"
    fi
    if [[ -n "${GEMM_RS_INCREMENTAL_PEER_REDUCE:-}" ]]; then
        env_str="$env_str GEMM_RS_INCREMENTAL_PEER_REDUCE=$GEMM_RS_INCREMENTAL_PEER_REDUCE"
    fi
    if [[ -n "${GEMM_RS_RECEIVER_OWNER_RS:-}" ]]; then
        env_str="$env_str GEMM_RS_RECEIVER_OWNER_RS=$GEMM_RS_RECEIVER_OWNER_RS"
    fi
    if [[ -n "${GEMM_RS_RECV_PROGRESS_SMS:-}" ]]; then
        env_str="$env_str GEMM_RS_RECV_PROGRESS_SMS=$GEMM_RS_RECV_PROGRESS_SMS"
    fi
    if [[ -n "${AG_GEMM_ACTIVE_SMS:-}" ]]; then
        env_str="$env_str AG_GEMM_ACTIVE_SMS=$AG_GEMM_ACTIVE_SMS"
    fi
    if [[ -n "${AG_GEMM_RING_PROXY_FORWARD:-}" ]]; then
        env_str="$env_str AG_GEMM_RING_PROXY_FORWARD=$AG_GEMM_RING_PROXY_FORWARD"
    fi
    if [[ -n "${AG_GEMM_REMOTE_READY_PER_COL:-}" ]]; then
        env_str="$env_str AG_GEMM_REMOTE_READY_PER_COL=$AG_GEMM_REMOTE_READY_PER_COL"
    fi
    if [[ -n "${AG_GEMM_SKIP_REMOTE_COMPUTE:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_REMOTE_COMPUTE=$AG_GEMM_SKIP_REMOTE_COMPUTE"
    fi
    if [[ -n "${AG_GEMM_SKIP_PHASE1:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_PHASE1=$AG_GEMM_SKIP_PHASE1"
    fi
    if [[ -n "${AG_GEMM_SKIP_PHASE1_GATE:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_PHASE1_GATE=$AG_GEMM_SKIP_PHASE1_GATE"
    fi
    if [[ -n "${AG_GEMM_SKIP_PHASE2:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_PHASE2=$AG_GEMM_SKIP_PHASE2"
    fi
    if [[ -n "${AG_GEMM_SKIP_COMPUTE:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_COMPUTE=$AG_GEMM_SKIP_COMPUTE"
    fi
    if [[ -n "${AG_GEMM_SKIP_RESET:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_RESET=$AG_GEMM_SKIP_RESET"
    fi
    if [[ -n "${AG_GEMM_PROLOGUE_SIDE_STREAM:-}" ]]; then
        env_str="$env_str AG_GEMM_PROLOGUE_SIDE_STREAM=$AG_GEMM_PROLOGUE_SIDE_STREAM"
    fi
    if [[ -n "${AG_GEMM_SKIP_PROLOGUE:-}" ]]; then
        env_str="$env_str AG_GEMM_SKIP_PROLOGUE=$AG_GEMM_SKIP_PROLOGUE"
    fi
    if [[ -n "${AG_GEMM_ADAPTIVE_CAP_LARGE_M:-}" ]]; then
        env_str="$env_str AG_GEMM_ADAPTIVE_CAP_LARGE_M=$AG_GEMM_ADAPTIVE_CAP_LARGE_M"
    fi
    if [[ -n "${AG_GEMM_LOGICAL_QUEUES_PER_QP:-}" ]]; then
        env_str="$env_str AG_GEMM_LOGICAL_QUEUES_PER_QP=$AG_GEMM_LOGICAL_QUEUES_PER_QP"
    fi
    if [[ -n "${AG1_ADAPTIVE_COMM_SMS:-}" ]]; then
        env_str="$env_str AG1_ADAPTIVE_COMM_SMS=$AG1_ADAPTIVE_COMM_SMS"
    fi
    if [[ -n "${MKERNEL_PREP_EPOCH_FAST:-}" ]]; then
        env_str="$env_str MKERNEL_PREP_EPOCH_FAST=$MKERNEL_PREP_EPOCH_FAST"
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
    # Forward all DGC_* gather/debug knobs to both nodes.
    for var in $(compgen -e | grep '^DGC_' || true); do
        env_str="$env_str ${var}=${!var}"
    done
    if [[ -n "${MKERNEL_DISPATCH_GEMM_ROUTING:-}" ]]; then
        env_str="$env_str MKERNEL_DISPATCH_GEMM_ROUTING=$MKERNEL_DISPATCH_GEMM_ROUTING"
    fi
    if [[ -n "${MKERNEL_DISPATCH_GEMM_ROUTING_SEED:-}" ]]; then
        env_str="$env_str MKERNEL_DISPATCH_GEMM_ROUTING_SEED=$MKERNEL_DISPATCH_GEMM_ROUTING_SEED"
    fi
    if [[ -n "${Q2_EPOCH_TIMING:-}" ]]; then
        env_str="$env_str Q2_EPOCH_TIMING=$Q2_EPOCH_TIMING"
    fi
    if [[ -n "${MKERNEL_DIST_BACKEND:-}" ]]; then
        env_str="$env_str MKERNEL_DIST_BACKEND=$MKERNEL_DIST_BACKEND"
    fi
    if [[ -n "${MKERNEL_ROCE_GID_INDEX:-}" ]]; then
        env_str="$env_str MKERNEL_ROCE_GID_INDEX=$MKERNEL_ROCE_GID_INDEX"
    fi
    if [[ -n "${NCCL_IB_HCA:-}" ]]; then
        env_str="$env_str NCCL_IB_HCA=$NCCL_IB_HCA"
    fi
    if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
        env_str="$env_str NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
    fi
    if [[ -n "${NCCL_IB_MERGE_NICS:-}" ]]; then
        env_str="$env_str NCCL_IB_MERGE_NICS=$NCCL_IB_MERGE_NICS"
    fi
    if [[ -n "${NCCL_NET_MERGE_LEVEL:-}" ]]; then
        env_str="$env_str NCCL_NET_MERGE_LEVEL=$NCCL_NET_MERGE_LEVEL"
    fi
    if [[ -n "${NCCL_DEBUG:-}" ]]; then
        env_str="$env_str NCCL_DEBUG=$NCCL_DEBUG"
    fi
    if [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]]; then
        env_str="$env_str NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
    fi
    if [[ -n "${MKERNEL_CHANNELIZE_GPU_PEERS:-}" ]]; then
        env_str="$env_str MKERNEL_CHANNELIZE_GPU_PEERS=$MKERNEL_CHANNELIZE_GPU_PEERS"
    fi
    if [[ -n "${MKERNEL_INTERNODE_NUM_QPS:-}" ]]; then
        env_str="$env_str MKERNEL_INTERNODE_NUM_QPS=$MKERNEL_INTERNODE_NUM_QPS MKERNEL_IB_NUM_QPS=$MKERNEL_INTERNODE_NUM_QPS MKERNEL_EFA_NUM_QPS=$MKERNEL_INTERNODE_NUM_QPS"
    fi
    if [[ -n "${MKERNEL_PROXY_THREADS:-}" ]]; then
        env_str="$env_str MKERNEL_PROXY_THREADS=$MKERNEL_PROXY_THREADS"
    fi
    if [[ -n "${AG_GEMM_PROXY_THREADS:-}" ]]; then
        env_str="$env_str AG_GEMM_PROXY_THREADS=$AG_GEMM_PROXY_THREADS"
    fi
    if [[ -n "${GEMM_AR_PROXY_THREADS:-}" ]]; then
        env_str="$env_str GEMM_AR_PROXY_THREADS=$GEMM_AR_PROXY_THREADS"
    fi
    if [[ -n "${GEMM_AR_EARLY_REMOTE_ACCUM:-}" ]]; then
        env_str="$env_str GEMM_AR_EARLY_REMOTE_ACCUM=$GEMM_AR_EARLY_REMOTE_ACCUM"
    fi
    if [[ -n "${GEMM_AR_RECV_PROGRESS_SMS:-}" ]]; then
        env_str="$env_str GEMM_AR_RECV_PROGRESS_SMS=$GEMM_AR_RECV_PROGRESS_SMS"
    fi
    if [[ -n "${GEMM_AR_INTRA_READY_MULTIMEM:-}" ]]; then
        env_str="$env_str GEMM_AR_INTRA_READY_MULTIMEM=$GEMM_AR_INTRA_READY_MULTIMEM"
    fi
    if [[ -n "${GEMM_AR_CHUNK_TILES:-}" ]]; then
        env_str="$env_str GEMM_AR_CHUNK_TILES=$GEMM_AR_CHUNK_TILES"
    fi
    if [[ -n "${GEMM_AR_LOGICAL_QUEUES_PER_QP:-}" ]]; then
        env_str="$env_str GEMM_AR_LOGICAL_QUEUES_PER_QP=$GEMM_AR_LOGICAL_QUEUES_PER_QP"
    fi
    if [[ -n "${MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP:-}" ]]; then
        env_str="$env_str MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP=$MKERNEL_INTERNODE_LOGICAL_QUEUES_PER_QP"
    fi
    if [[ -n "${MKERNEL_MAX_INFLIGHT:-}" ]]; then
        env_str="$env_str MKERNEL_MAX_INFLIGHT=$MKERNEL_MAX_INFLIGHT"
    fi
    # Forward per-shape GEMM_RS split overrides.
    for var in $(compgen -e | grep '^MKERNEL_GEMM_RS_SPLIT_' || true); do
        env_str="$env_str ${var}=${!var}"
    done
    # Forward per-shape ag_gemm intra-CTA overrides.
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
        local ssh_port_var="NODE${i}_SSH_PORT"
        local ssh_host="${!ssh_var}"
        local ssh_port_args=()
        if [[ -n "${!ssh_port_var:-}" ]]; then
            ssh_port_args=(-p "${!ssh_port_var}")
        fi
        local launch_peer="cd '$RELEASE' && $env_str NODE_IDX=$i \
            TCP_PORT=$tcp_port_base \
            '$TORCHRUN' --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=$i \
                --master_addr=$NODE0_IP --master_port=$master_port \
                '$script' $extra_args"
        timeout "${TIMEOUT_S}" ssh \
            -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
            "${ssh_port_args[@]}" "$ssh_host" "bash -c \"$launch_peer\"" \
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
        pkill -9 -f "${kernel}_bench.py" 2>/dev/null || true
        for ((i=1; i<NUM_NODES; i++)); do
            local ssh_var="NODE${i}_SSH"
            local ssh_port_var="NODE${i}_SSH_PORT"
            local ssh_port_args=()
            if [[ -n "${!ssh_port_var:-}" ]]; then
                ssh_port_args=(-p "${!ssh_port_var}")
            fi
            ssh -o BatchMode=yes \
                "${ssh_port_args[@]}" "${!ssh_var}" "pkill -9 -f ${kernel}_bench.py; pkill -9 -f nccl_baseline_bench.py" 2>/dev/null || true
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
