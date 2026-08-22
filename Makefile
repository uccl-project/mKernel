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

# === Target GPU ===
#   GPU=hopper    → sm_90a, wgmma MMA path (default, upstream behaviour)
#   GPU=blackwell → sm_103a, tcgen05 MMA path (B300; gemm_rs and ag_gemm)
GPU ?= hopper
ifeq ($(GPU),blackwell)
    ARCH              := -gencode arch=compute_103a,code=sm_103a
    ARCH_DEFINES      := -DKITTENS_SM10X -DKITTENS_BLACKWELL -DMKERNEL_TCGEN05
    DEFAULT_CUDA_HOME := /usr/local/cuda-13.2
    # conda forces a host compiler through NVCC_PREPEND_FLAGS/CXX on some boxes,
    # which makes nvcc miss system headers; pin the system g++.
    CCBIN             := -ccbin /usr/bin/g++
else ifeq ($(GPU),hopper)
    ARCH              := -gencode arch=compute_90a,code=sm_90a
    ARCH_DEFINES      := -DKITTENS_HOPPER
    DEFAULT_CUDA_HOME := /usr/local/cuda-12.9
    CCBIN             :=
else
    $(error Unknown GPU=$(GPU). Use GPU=hopper or GPU=blackwell.)
endif

# === Tooling ===
CUDA_HOME       ?= $(DEFAULT_CUDA_HOME)
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

# INTRA_NUM_DEVICES = GPUs per logical node (multicast group size). Default 8
# matches an 8-GPU-per-node deployment. Override to test emulated multinode
# (e.g. `make INTRA_NUM_DEVICES=4 all` for 4 GPUs / "node").
INTRA_NUM_DEVICES ?= 8
COMMON_DEFINES  := $(ARCH_DEFINES) -DINTRA_NUM_DEVICES=$(INTRA_NUM_DEVICES) $(BACKEND_DEFINES)
COMMON_FLAGS    := -O3 -std=c++20 --use_fast_math --extended-lambda --expt-relaxed-constexpr $(ARCH) $(CCBIN)
LDFLAGS         := -shared -lcuda $(BACKEND_LIBS) \
                   -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
                   -Xlinker -rpath -Xlinker $(TORCH_LIB) -L$(CUDA_HOME)/lib

COMMON_INC      := $(INC_RELEASE) $(INC_EFA) $(TORCH_INC) $(PY_INC)

# === Per-kernel constants (passed via -D, no env-var lookups) ===
#
# Note: keep per-kernel compile-time constants here until the corresponding
# source paths no longer need build-time specialization.
#
# Failed-experiment flags are NOT defined here (HYBRID, MERGED_COMM,
# PUSH_NVL_FANOUT, DISPATCH_DONATE_INTER_SEND, ACTIVITY_TRACE, etc.) so
# their #ifdef branches stay disabled.
# AG_GEMM_TRACE=1 turns on the per-task activity trace (device-side %globaltimer
# + clock64 records, read back via trace_read()). Profiling only -- it perturbs
# timing slightly and costs 4 MB of static device memory.
AG_GEMM_TRACE ?= 0
# AG_GEMM_ROWPERM=1 orders compute's row blocks to match phase-1's production
# order (see decode_comp_task). Perf change only -- it is a bijection.
ifeq ($(GPU),blackwell)
AG_GEMM_ROWPERM ?= 1
else
AG_GEMM_ROWPERM ?= 0
endif
DEFS_ag_gemm        :=
ifeq ($(AG_GEMM_TRACE),1)
DEFS_ag_gemm        += -DAG_GEMM_TRACE
endif
ifeq ($(AG_GEMM_ROWPERM),1)
DEFS_ag_gemm        += -DAG_GEMM_ROWPERM
endif
# AG_GEMM_FASTPOLL=1 lets compute skip the per-K-strip readiness poll once
# phase-1 has completed on every device.
ifeq ($(GPU),blackwell)
AG_GEMM_FASTPOLL ?= 1
else
AG_GEMM_FASTPOLL ?= 0
endif
ifeq ($(AG_GEMM_FASTPOLL),1)
DEFS_ag_gemm        += -DAG_GEMM_FASTPOLL
endif
# AG_GEMM_OWNSHARD=1 skips the readiness wait for rows this device owns, whose
# bytes are already in place before phase 1 runs. MEASURED SLOWER (M=32768
# 6.556 -> 6.612 ms, M=16384 1.043 -> 1.081): letting those CTAs run ahead
# breaks the lockstep tile order that the multicast read sharing depends on.
# Kept off by repository convention for failed experiments.
AG_GEMM_OWNSHARD ?= 0
ifeq ($(AG_GEMM_OWNSHARD),1)
DEFS_ag_gemm        += -DAG_GEMM_OWNSHARD
endif
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
	    --nproc-per-node=$(INTRA_NUM_DEVICES) \
	    bench/dispatch_gemm_blackwell_bench.py \
	        --specialization $(SPECIALIZATION) $(BLACKWELL_BENCH_ARGS)

DISPATCH_GEMM_BLACKWELL_HEADERS := \
	include/operators/dispatch_gemm_blackwell/dispatch_gemm_blackwell.cuh \
	include/operators/dispatch_gemm_blackwell/session.cuh

$(BUILD)/libdispatch_gemm_blackwell.so: $(DISPATCH_GEMM_BLACKWELL_HEADERS)

DISPATCH_GEMM_WARP_SPECIALIZATION_HEADERS := \
	include/operators/dispatch_gemm_warp_specialization/dispatch_gemm_warp_specialization.cuh \
	include/operators/dispatch_gemm_warp_specialization/session.cuh

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
	gemm-ar-blackwell run-gemm-ar-blackwell clean bench check \
	test-slot-math plots

run-gemm-ar-blackwell : gemm_ar_blackwell
	python -m torch.distributed.run --standalone --nproc-per-node=$(INTRA_NUM_DEVICES) bench/gemm_ar_blackwell_bench.py

gemm-ar-blackwell : $(BUILD)/libgemm_ar_blackwell.so

$(BUILD)/libgemm_ar_blackwell.so : $(SRC)/gemm_ar_blackwell.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(GEMM_AR_BLACKWELL_SANITIZE) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_gemm_ar_blackwell $(DEFS_gemm_ar_blackwell) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@
