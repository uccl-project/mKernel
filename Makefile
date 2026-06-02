# Makefile — single-config build for 5 multinode kernels
#
# Usage:
#   make all       — build all 5 .so's into build/
#   make check     — run correctness check across all 5 kernels
#   make bench     — run wall-time bench across all 5 kernels
#   make plots     — regenerate TFLOPS bar charts under plots/
#   make clean     — remove build/

# === Backend selection ===
#
# Two backends are supported:
#   BACKEND=efa  → AWS EFA SRD via libibverbs+efadv (default)
#   BACKEND=cx7  → ConnectX-7 RC via libibverbs (InfiniBand / RoCE)
#
# Override with: `make BACKEND=cx7 all`
BACKEND ?= efa
ifeq ($(BACKEND),efa)
    BACKEND_DEFINES := -DINTERNODE_BACKEND_EFA
    BACKEND_LIBS    := -L$(EFA_HOME)/lib -lfabric -libverbs -lefa
else ifeq ($(BACKEND),cx7)
    BACKEND_DEFINES := -DINTERNODE_BACKEND_IBVERBS
    BACKEND_LIBS    := -libverbs
else
    $(error Unknown BACKEND=$(BACKEND). Use BACKEND=efa or BACKEND=cx7.)
endif

# === Tooling ===
CUDA_HOME       ?= /usr/local/cuda-12.9
EFA_HOME        ?= /opt/amazon/efa
NVCC            := $(CUDA_HOME)/bin/nvcc
# Python with torch installed. Override with `PYTHON=/path/to/python`.
PYTHON          ?= python3

# === Include paths (must precede LDFLAGS — TORCH_LIB feeds both) ===
HERE            := $(abspath .)
INC_RELEASE     := -I$(HERE)/include
ifeq ($(BACKEND),efa)
    INC_EFA     := -I$(EFA_HOME)/include
else
    INC_EFA     :=
endif
PY_INC          := $(shell $(PYTHON) -c "import sysconfig; print('-I'+sysconfig.get_path('include'))")
TORCH_INC       := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as e; print(' '.join('-I'+p for p in e.include_paths()))")
TORCH_LIB       := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as e; print(e.library_paths()[0])")

# === Common compile flags ===
ARCH            := -gencode arch=compute_90a,code=sm_90a
# INTRA_NUM_DEVICES = GPUs per logical node (multicast group size). Default 8
# matches an 8-GPU-per-node deployment. Override to test emulated multinode
# (e.g. `make INTRA_NUM_DEVICES=4 all` for 4 GPUs / "node").
INTRA_NUM_DEVICES ?= 8
COMMON_DEFINES  := -DKITTENS_HOPPER -DINTRA_NUM_DEVICES=$(INTRA_NUM_DEVICES) $(BACKEND_DEFINES)
COMMON_FLAGS    := -O3 -std=c++20 --use_fast_math --extended-lambda --expt-relaxed-constexpr $(ARCH)
LDFLAGS         := -shared -lcuda $(BACKEND_LIBS) \
                   -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
                   -Xlinker -rpath -Xlinker $(TORCH_LIB)

COMMON_INC      := $(INC_RELEASE) $(INC_EFA) $(TORCH_INC) $(PY_INC)

# === Per-kernel constants (passed via -D, no env-var lookups) ===
#
# Note: keep per-kernel compile-time constants here until the corresponding
# source paths no longer need build-time specialization.
#
# Failed-experiment flags are NOT defined here (HYBRID, MERGED_COMM,
# PUSH_NVL_FANOUT, DISPATCH_DONATE_INTER_SEND, ACTIVITY_TRACE, etc.) so
# their #ifdef branches stay disabled.
DEFS_ag_gemm        :=
# Arrival-flag layout is now a runtime flag (SessionConfig.use_arrival_queue);
# gemm_ar's session shim sets it to true. No compile-time switch needed.
DEFS_gemm_ar        :=

TK_MOE_NUM_NODES ?= 2
DEFS_dispatch_gemm  := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256 -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)
DEFS_ring_attention :=
DEFS_gemm_rs        :=
# dispatch_gemm_glu_combine (fused MoE, M2): clones dispatch_gemm's RDMA dispatch
# machinery + a naive FFN compute (gemm1->swiglu->gemm2) + combine. Same
# compile-time shape knobs as dispatch_gemm; override for fast naive-compute
# validation, e.g. `make DGC_H=1024 DGC_I=512 build/libdispatch_gemm_glu_combine.so`.
DGC_H           ?= 7168
DGC_I           ?= 2048
DGC_TOP_K       ?= 8
DGC_NUM_EXPERTS ?= 256
DEFS_dispatch_gemm_glu_combine := -DTK_MOE_H=$(DGC_H) -DTK_MOE_I=$(DGC_I) -DTK_MOE_TOP_K=$(DGC_TOP_K) -DTK_MOE_NUM_EXPERTS=$(DGC_NUM_EXPERTS) -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)

# === Build targets ===
BUILD := build
SRC   := src

KERNELS := dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention dispatch_gemm_glu_combine

all: $(addprefix $(BUILD)/lib,$(addsuffix .so,$(KERNELS)))

$(BUILD)/lib%.so: $(SRC)/%.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_$* $(DEFS_$*) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

bench: all
	cd bench && bash run.sh all bench

check: all
	cd bench && bash run.sh all check

# Host-only unit test for internode slot math (peer_rank_for_slot,
# slot_at_peer, ring origin). Pins down the N>2 invariants without needing
# real multi-node hardware.
test-slot-math: tests/test_internode_slot_math.cpp | $(BUILD)
	g++ -std=c++17 -O2 -I include -D__host__= -D__device__= $< -o $(BUILD)/test_internode_slot_math
	$(BUILD)/test_internode_slot_math

plots:
	cd plots && python3 plot_tflops_efa.py

.PHONY: all clean bench check test-slot-math plots
