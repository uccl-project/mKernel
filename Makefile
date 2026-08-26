# Makefile — single-config build for 5 multinode kernels
#
# Usage:
#   make all       — build all 5 .so's into build/
#   make ENABLE_DISPATCH_GEMM_BLACKWELL=1 all
#                  — also build the intra-node Blackwell dispatch+GEMM kernel
#   make dispatch-gemm-blackwell SPECIALIZATION=sm|warp
#                  — build the selected 8-GPU B300 implementation
#   make run-dispatch-gemm-blackwell SPECIALIZATION=sm|warp
#                  — build and benchmark the selected implementation
#   make dispatch-gemm-warp-specialization
#                  — build the 8-GPU B300 dispatch + warp-specialized kernel
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
CUDA_HOME       ?= /usr/local/cuda
CUDA_DRIVER_LIB ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/lib,/lib/x86_64-linux-gnu)
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
# dispatch_gemm_blackwell uses tcgen05 and must be compiled for the native
# Blackwell architecture. B300 is SM 10.3; override BLACKWELL_SM for another
# Blackwell GPU (for example BLACKWELL_SM=100).
BLACKWELL_SM ?= 103
ARCH_dispatch_gemm_blackwell := \
    -gencode arch=compute_$(BLACKWELL_SM)a,code=sm_$(BLACKWELL_SM)a
ARCH_dispatch_gemm_warp_specialization := $(ARCH_dispatch_gemm_blackwell)
# INTRA_NUM_DEVICES = GPUs per logical node (multicast group size). Default 8
# matches an 8-GPU-per-node deployment. Override to test emulated multinode
# (e.g. `make INTRA_NUM_DEVICES=4 all` for 4 GPUs / "node").
INTRA_NUM_DEVICES ?= 8
BLACKWELL_INTRA_NUM_DEVICES ?= 8
INTRA_NUM_DEVICES_dispatch_gemm_blackwell := $(BLACKWELL_INTRA_NUM_DEVICES)
INTRA_NUM_DEVICES_dispatch_gemm_warp_specialization := $(BLACKWELL_INTRA_NUM_DEVICES)

# $* is available while expanding the pattern-rule recipe below. Per-target
# values keep existing kernels on ARCH/INTRA_NUM_DEVICES while allowing the
# optional Blackwell kernel to select its native architecture and domain.
TARGET_ARCH     = $(or $(ARCH_$*),$(ARCH))
TARGET_INTRA_NUM_DEVICES = $(or $(INTRA_NUM_DEVICES_$*),$(INTRA_NUM_DEVICES))
# The Blackwell kernel is intra-node-only. Keep its direct target independent
# of the repository-wide EFA default so it works on the B300/CX7 machine with
# a plain make dispatch-gemm-blackwell.
BLACKWELL_INTRANODE_KERNELS := dispatch_gemm_blackwell dispatch_gemm_warp_specialization
TARGET_BACKEND_DEFINES = $(if $(filter $*,$(BLACKWELL_INTRANODE_KERNELS)),-DINTERNODE_BACKEND_IBVERBS,$(BACKEND_DEFINES))
TARGET_BACKEND_LIBS = $(if $(filter $*,$(BLACKWELL_INTRANODE_KERNELS)),-libverbs,$(BACKEND_LIBS))
COMMON_DEFINES  = -DKITTENS_HOPPER -DINTRA_NUM_DEVICES=$(TARGET_INTRA_NUM_DEVICES) $(TARGET_BACKEND_DEFINES)
COMMON_FLAGS    = -O3 -std=c++20 --use_fast_math --extended-lambda --expt-relaxed-constexpr $(TARGET_ARCH)
LDFLAGS         = -shared -L$(CUDA_DRIVER_LIB) -lcuda $(TARGET_BACKEND_LIBS) \
                   -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
                   -Xlinker -rpath -Xlinker $(CUDA_DRIVER_LIB) \
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
DEFS_dispatch_gemm_blackwell := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256
DEFS_dispatch_gemm_warp_specialization := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256
DEFS_ring_attention :=
DEFS_gemm_rs        :=
DEFS_dispatch_gemm_glu_combine := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256 -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)

# === Build targets ===
BUILD := build
SRC   := src

KERNELS := dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention dispatch_gemm_glu_combine

SPECIALIZATION ?= sm
ifeq ($(SPECIALIZATION),warp)
BLACKWELL_SPECIALIZATION_KERNEL := dispatch_gemm_warp_specialization
else ifeq ($(SPECIALIZATION),sm)
BLACKWELL_SPECIALIZATION_KERNEL := dispatch_gemm_blackwell
else
$(error Unknown SPECIALIZATION=$(SPECIALIZATION). Use SPECIALIZATION=warp or SPECIALIZATION=sm.)
endif

ENABLE_DISPATCH_GEMM_BLACKWELL ?= 0
ifeq ($(ENABLE_DISPATCH_GEMM_BLACKWELL),1)
KERNELS += $(BLACKWELL_SPECIALIZATION_KERNEL)
endif
all: $(addprefix $(BUILD)/lib,$(addsuffix .so,$(KERNELS)))

dispatch-gemm-blackwell: $(BUILD)/lib$(BLACKWELL_SPECIALIZATION_KERNEL).so

dispatch-gemm-sm-specialization: $(BUILD)/libdispatch_gemm_blackwell.so

dispatch-gemm-warp-specialization: $(BUILD)/libdispatch_gemm_warp_specialization.so

BLACKWELL_BENCH_ARGS ?= --check
run-dispatch-gemm-blackwell: dispatch-gemm-blackwell
	$(PYTHON) -m torch.distributed.run --standalone \
	    --nproc-per-node=$(BLACKWELL_INTRA_NUM_DEVICES) \
	    bench/dispatch_gemm_blackwell_bench.py \
	        --specialization $(SPECIALIZATION) $(BLACKWELL_BENCH_ARGS)

DISPATCH_GEMM_BLACKWELL_HEADERS := \
	include/operators/dispatch_gemm_blackwell/dispatch_gemm_blackwell.cuh \
	include/operators/dispatch_gemm_blackwell/session.cuh \
	include/profiling/timings.cuh

$(BUILD)/libdispatch_gemm_blackwell.so: $(DISPATCH_GEMM_BLACKWELL_HEADERS)

DISPATCH_GEMM_WARP_SPECIALIZATION_HEADERS := \
	include/operators/dispatch_gemm_warp_specialization/dispatch_gemm_warp_specialization.cuh \
	include/operators/dispatch_gemm_warp_specialization/session.cuh \
	include/profiling/timings.cuh

$(BUILD)/libdispatch_gemm_warp_specialization.so: $(DISPATCH_GEMM_WARP_SPECIALIZATION_HEADERS)

$(BUILD)/lib%.so: $(SRC)/%.cu Makefile | $(BUILD)
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

.PHONY: all dispatch-gemm-blackwell dispatch-gemm-sm-specialization \
	dispatch-gemm-warp-specialization run-dispatch-gemm-blackwell \
	clean bench check test-slot-math plots
