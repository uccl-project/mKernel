#!/bin/bash
# Emulated 4-node × 4-GPU correctness launcher.
#
# Spawns 4 torchrun process groups (2 per physical host) so the kernels
# run as if they were on 4 physical nodes. Used for testing N>2 code
# paths on limited hardware.
#
# Build: dispatch_gemm hardcodes its peer count via TK_MOE_NUM_NODES, so
# the .so must be built with both INTRA_NUM_DEVICES=4 and
# TK_MOE_NUM_NODES=4:
#
#     make INTRA_NUM_DEVICES=4 TK_MOE_NUM_NODES=4 all
#
# Usage:
#     bash run_emulated_4x4.sh <kernel|all>
#
# Required env:
#     NODE_A_IP    — IP of host A (this machine, becomes nodes 0 and 1)
#     NODE_B_IP    — IP of host B (becomes nodes 2 and 3)
#     NODE_B_SSH   — SSH target for host B (e.g. ubuntu@1.2.3.4)
#     TORCHRUN     — full path to torchrun
#     EFA_HOME     — EFA install root (default /opt/amazon/efa)
#     PY           — full path to python3 (optional, mostly informational)
#
# Optional env:
#     SHAPES       — CSV of shapes to test (kernel-specific)
#     MASTER_PORT  — c10d rendezvous port (default 29980 to avoid clashes)
#     TCP_PORT     — kernel TCP bootstrap base (default 19880)
#     NODE_B_SSH_PORT — SSH port for host B
#     TIMEOUT      — per-kernel timeout in seconds (default 300)
#
# Mode is always `check` — this launcher is correctness-only.
set -u

KERNEL=${1:?usage: $0 <kernel|all>}

NODE_A_IP=${NODE_A_IP:?NODE_A_IP must be set (IP of this host)}
NODE_B_IP=${NODE_B_IP:?NODE_B_IP must be set (IP of peer host)}
NODE_B_SSH=${NODE_B_SSH:?NODE_B_SSH must be set (SSH target for peer host)}

TORCHRUN=${TORCHRUN:?TORCHRUN must point at torchrun (use venv path)}
EFA_HOME=${EFA_HOME:-/opt/amazon/efa}
MASTER_PORT=${MASTER_PORT:-29980}
TCP_PORT_BASE=${TCP_PORT:-19880}
TIMEOUT_S=${TIMEOUT:-300}

HERE=$(cd "$(dirname "$0")" && pwd)
RELEASE=$(dirname "$HERE")

KERNELS_ALL="dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention"
if [[ "$KERNEL" == "all" ]]; then
    TARGETS="$KERNELS_ALL"
else
    TARGETS="$KERNEL"
fi

cleanup() {
    pkill -9 -f "_bench.py" 2>/dev/null || true
    local ssh_port_args=()
    if [[ -n "${NODE_B_SSH_PORT:-}" ]]; then
        ssh_port_args=(-p "${NODE_B_SSH_PORT}")
    fi
    ssh -o BatchMode=yes "${ssh_port_args[@]}" "$NODE_B_SSH" \
        "pkill -9 -f _bench.py" 2>/dev/null || true
    rm -f /tmp/mkernel_*.sock* 2>/dev/null || true
    ssh -o BatchMode=yes "${ssh_port_args[@]}" "$NODE_B_SSH" \
        "rm -f /tmp/mkernel_*.sock*" 2>/dev/null || true
    sleep 2
}

run_one() {
    local kernel=$1
    local script="$HERE/${kernel}_bench.py"
    if [[ ! -f "$script" ]]; then
        echo "[run_4x4] missing $script"
        return 1
    fi

    # Per-kernel master port offset so multiple sweeps don't collide.
    local default_master_port=$MASTER_PORT
    local default_tcp_port=$TCP_PORT_BASE
    case "$kernel" in
        dispatch_gemm)   default_master_port=$((MASTER_PORT+0));   default_tcp_port=$((TCP_PORT_BASE+0));   ;;
        gemm_rs)         default_master_port=$((MASTER_PORT+10));  default_tcp_port=$((TCP_PORT_BASE+100)); ;;
        ag_gemm)         default_master_port=$((MASTER_PORT+20));  default_tcp_port=$((TCP_PORT_BASE+200)); ;;
        gemm_ar)         default_master_port=$((MASTER_PORT+30));  default_tcp_port=$((TCP_PORT_BASE+300)); ;;
        ring_attention)  default_master_port=$((MASTER_PORT+40));  default_tcp_port=$((TCP_PORT_BASE+400)); ;;
    esac
    local mport=$default_master_port
    local tcp_port=$default_tcp_port

    echo "==== $kernel (check 4x4) master_port=${mport} tcp_port=${tcp_port} ===="

    # Common env propagated to every group.
    local COMMON_ENV=(
        "INTERNODE_BACKEND=efa"
        "LD_LIBRARY_PATH=${EFA_HOME}/lib:${LD_LIBRARY_PATH:-}"
        "NCCL_IB_DISABLE=1"
        "NCCL_P2P_DISABLE=0"
        "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp71s0}"
        "NCCL_NVLS_ENABLE=0"
        "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-enp71s0}}"
        "GEMM_AR_ARRIVAL_QUEUE=1"
        "GEMM_AR_K_DIV=global_world"
        "MKERNEL_BIND_RETAINED_HANDLE=1"
        "MKERNEL_EFA_NUM_QPS=${MKERNEL_EFA_NUM_QPS:-4}"
        "MKERNEL_BENCH_BEST_OF_N=0"
        "TORCH_CUDA_ARCH_LIST=9.0a"
        "NUM_NODES=4"
        "MASTER_ADDR=$NODE_A_IP"
        "MASTER_PORT=$mport"
        "NODE0_IP=$NODE_A_IP"
        "NODE1_IP=$NODE_A_IP"
        "NODE2_IP=$NODE_B_IP"
        "NODE3_IP=$NODE_B_IP"
    )
    local env_str="${COMMON_ENV[*]}"

    local extra_args="--mode check"
    if [[ -n "${SHAPES:-}" ]]; then extra_args="$extra_args --shapes $SHAPES"; fi
    if [[ -n "${WARMUP:-}" ]]; then extra_args="$extra_args --warmup $WARMUP"; fi
    if [[ -n "${ITERS:-}" ]]; then extra_args="$extra_args --iters $ITERS"; fi

    # Per-kernel notify mode: gemm_ar uses arrival queue + the write_imm CQE
    # handler only publishes to the flat array → hangs. Other kernels use
    # write_imm to dodge EFA SRD's unordered data/flag delivery race.
    local notify_mode="${MKERNEL_EFA_VERBS_NOTIFY_MODE:-}"
    if [[ -z "$notify_mode" ]]; then
        if [[ "$kernel" == "gemm_ar" ]]; then notify_mode=remote_flag; else notify_mode=write_imm; fi
    fi
    COMMON_ENV+=("MKERNEL_EFA_VERBS_NOTIFY_MODE=$notify_mode")
    env_str="${COMMON_ENV[*]}"

    cleanup

    # Launch each group: NODE_IDX determines GPUs + node_rank.
    #   NODE_IDX=0 → host A, GPUs 0-3
    #   NODE_IDX=1 → host A, GPUs 4-7
    #   NODE_IDX=2 → host B, GPUs 0-3
    #   NODE_IDX=3 → host B, GPUs 4-7
    declare -a pids=()
    local ssh_port_args=()
    if [[ -n "${NODE_B_SSH_PORT:-}" ]]; then
        ssh_port_args=(-p "${NODE_B_SSH_PORT}")
    fi

    for NODE_IDX in 0 1 2 3; do
        case "$NODE_IDX" in
            0|1) target=local ;;
            2|3) target=peer ;;
        esac
        case "$NODE_IDX" in
            0|2) cvd="0,1,2,3" ;;
            1|3) cvd="4,5,6,7" ;;
        esac
        # Per-NODE_IDX DIST_BROKER_KEY so two same-host groups don't share
        # the default /dist_broker.sock / /dist_broker_shm and cross-
        # contaminate each other's IPC handle exchanges (which manifests as
        # `cuMemMap operation not supported` when a rank tries to map an
        # IPC handle from a foreign address space).
        local broker_key="mkernel_${kernel}_4x4_n${NODE_IDX}_${mport}"
        local launch="cd '$RELEASE' && \
            $env_str CUDA_VISIBLE_DEVICES=$cvd NODE_IDX=$NODE_IDX TCP_PORT=$tcp_port \
            DIST_BROKER_KEY=$broker_key \
            '$TORCHRUN' --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_IDX \
                --master_addr=$NODE_A_IP --master_port=$mport \
                '$script' $extra_args"
        if [[ "$target" == "local" ]]; then
            timeout "${TIMEOUT_S}" bash -c "$launch" \
                > "/tmp/${kernel}_4x4_node${NODE_IDX}.log" 2>&1 &
        else
            timeout "${TIMEOUT_S}" ssh \
                -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
                "${ssh_port_args[@]}" "$NODE_B_SSH" "bash -c \"$launch\"" \
                > "/tmp/${kernel}_4x4_node${NODE_IDX}.log" 2>&1 &
        fi
        pids+=("$!")
        sleep 1   # Stagger launches a bit so all groups don't hammer rendezvous at once.
    done

    local rc=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            rc=$?
        fi
    done

    # Concatenate logs for quick inspection.
    echo "==== $kernel 4x4 results ===="
    for NODE_IDX in 0 1 2 3; do
        echo "---- node $NODE_IDX ----"
        grep -E "correctness|PASS|FAIL|cuMulticast|Traceback|Error|Aborted" \
            "/tmp/${kernel}_4x4_node${NODE_IDX}.log" 2>/dev/null | head -30
    done

    cleanup
    return $rc
}

OVERALL_RC=0
for kernel in $TARGETS; do
    if ! run_one "$kernel"; then
        echo "[run_4x4] $kernel: FAILED"
        OVERALL_RC=1
    fi
done

exit $OVERALL_RC
