# =====================================================================
# Makefile -- selectable VCS simulation flow for hw/verif testbenches.
#
#   make                         # compile + run TB=layer_norm
#   make TB=multihead_attention  # choose a testbench
#   make TB=feed_forward_network all
#   make TB=numerical_feature_tokenizer all
#   make comp                    # compile selected TB only
#   make run                     # run selected TB from sim/
#   make verdi                   # open selected TB FSDB with KDB hierarchy
#   make clean
#
# Compatibility aliases are kept for the existing DPI benches:
#   make all_mha / run_mha / verdi_mha / cmodel_mha
#   make all_ffn / run_ffn / verdi_ffn / cmodel_ffn
# =====================================================================

VCS      ?= vcs
VERDI    ?= verdi
GCC      ?= gcc

# repo root (this Makefile lives in hw/verif/)
PROJ     := $(abspath ../..)
MODELDIR := $(PROJ)/hw/model

TB_DIR    := tb
FLIST_DIR := flist
UTILS_DIR := utils
SIM_DIR   := sim

TB ?= layer_norm
VALID_TBS := layer_norm multihead_attention feed_forward_network numerical_feature_tokenizer mlp

ifeq ($(filter $(TB),$(VALID_TBS)),)
$(error Unsupported TB="$(TB)". Valid TB values: $(VALID_TBS))
endif

ifeq ($(TB),layer_norm)
FLIST        := $(FLIST_DIR)/layer_norm_filelist.f
DPI_C        := $(UTILS_DIR)/layer_norm_dpi.c
CMODEL_C     := $(MODELDIR)/layer_norm_cmodel.c
CMODEL_DEFINE := LN_STANDALONE
CMODEL_BIN   := $(SIM_DIR)/lncm
endif

ifeq ($(TB),multihead_attention)
FLIST        := $(FLIST_DIR)/multihead_attention_filelist.f
DPI_C        := $(UTILS_DIR)/multihead_attention_dpi.c
CMODEL_C     := $(MODELDIR)/multihead_attention_cmodel.c
CMODEL_DEFINE := MA_STANDALONE
CMODEL_BIN   := $(SIM_DIR)/macm
endif

ifeq ($(TB),feed_forward_network)
FLIST        := $(FLIST_DIR)/feed_forward_network_filelist.f
DPI_C        := $(UTILS_DIR)/feed_forward_network_dpi.c
CMODEL_C     := $(MODELDIR)/feed_forward_network_cmodel.c
CMODEL_DEFINE := FFN_STANDALONE
CMODEL_BIN   := $(SIM_DIR)/ffncm
endif

ifeq ($(TB),numerical_feature_tokenizer)
FLIST        := $(FLIST_DIR)/numerical_feature_tokenizer_filelist.f
DPI_C        :=
CMODEL_C     :=
CMODEL_DEFINE :=
CMODEL_BIN   :=
endif

ifeq ($(TB),mlp)
FLIST        := $(FLIST_DIR)/mlp_filelist.f
DPI_C        := $(UTILS_DIR)/mlp_dpi.c
CMODEL_C     := $(MODELDIR)/mlp_cmodel.c
CMODEL_DEFINE := MLP_STANDALONE
CMODEL_BIN   := $(SIM_DIR)/mlpcm
endif

SIMV      := $(SIM_DIR)/simv_$(TB)
COMP_LOG  := $(SIM_DIR)/comp_$(TB).log
RUN_LOG   := run_$(TB).log
FSDB      := $(SIM_DIR)/$(TB).fsdb

# -kdb pairs with -debug_access+all: writes the Verdi Knowledge DB
# (simv.daidir/kdb) so the FSDB loads with full source / code hierarchy.
VCS_FLAGS := -full64 -sverilog -debug_access+all -kdb -timescale=1ns/1ps
CFLAGS    := -std=c11 -O2 -I$(MODELDIR)
VCS_DEFINES ?=
SIMARGS   ?=

# If $fsdbDump* tasks do not resolve on an older VCS/Verdi, link the Verdi PLI:
#   NOVAS_FLAGS = -P $(VERDI_HOME)/share/PLI/VCS/LINUX64/novas.tab \
#                    $(VERDI_HOME)/share/PLI/VCS/LINUX64/pli.a +vpi +memcbk +vcsd
NOVAS_FLAGS ?=

VCS_CFLAGS  := $(if $(CMODEL_C),-CFLAGS "$(CFLAGS)")
VCS_CSOURCES := $(DPI_C) $(CMODEL_C)

.PHONY: all comp run verdi cmodel clean \
        all_ln comp_ln run_ln verdi_ln cmodel_ln \
        all_mha comp_mha run_mha verdi_mha cmodel_mha \
        all_ffn comp_ffn run_ffn verdi_ffn cmodel_ffn \
        all_nft comp_nft run_nft verdi_nft \
        all_mlp comp_mlp run_mlp verdi_mlp cmodel_mlp

all: comp run

comp:
	mkdir -p "$(SIM_DIR)"
	$(VCS) $(VCS_FLAGS) $(VCS_DEFINES) $(VCS_CFLAGS) \
	    -f $(FLIST) $(VCS_CSOURCES) $(NOVAS_FLAGS) \
	    -Mdir=$(SIM_DIR)/csrc_$(TB) \
	    -l $(COMP_LOG) -o $(SIMV)

run:
	cd "$(SIM_DIR)" && ./simv_$(TB) +fsdb+autoflush $(SIMARGS) -l $(RUN_LOG)

verdi:
	$(VERDI) -dbdir $(SIMV).daidir/kdb -ssf $(FSDB) &

ifneq ($(CMODEL_C),)
cmodel:
	mkdir -p "$(SIM_DIR)"
	$(GCC) -D$(CMODEL_DEFINE) -O2 -std=c11 $(CMODEL_C) -o $(CMODEL_BIN) -lm && ./$(CMODEL_BIN)
else
cmodel:
	@echo "No simulator-free C-model target is defined for TB=$(TB)."
endif

all_ln:
	$(MAKE) TB=layer_norm all

comp_ln:
	$(MAKE) TB=layer_norm comp

run_ln:
	$(MAKE) TB=layer_norm run

verdi_ln:
	$(MAKE) TB=layer_norm verdi

cmodel_ln:
	$(MAKE) TB=layer_norm cmodel

all_mha:
	$(MAKE) TB=multihead_attention all

comp_mha:
	$(MAKE) TB=multihead_attention comp

run_mha:
	$(MAKE) TB=multihead_attention run

verdi_mha:
	$(MAKE) TB=multihead_attention verdi

cmodel_mha:
	$(MAKE) TB=multihead_attention cmodel

all_ffn:
	$(MAKE) TB=feed_forward_network all

comp_ffn:
	$(MAKE) TB=feed_forward_network comp

run_ffn:
	$(MAKE) TB=feed_forward_network run

verdi_ffn:
	$(MAKE) TB=feed_forward_network verdi

cmodel_ffn:
	$(MAKE) TB=feed_forward_network cmodel

all_nft:
	$(MAKE) TB=numerical_feature_tokenizer all

comp_nft:
	$(MAKE) TB=numerical_feature_tokenizer comp

run_nft:
	$(MAKE) TB=numerical_feature_tokenizer run

verdi_nft:
	$(MAKE) TB=numerical_feature_tokenizer verdi

all_mlp:
	$(MAKE) TB=mlp all

comp_mlp:
	$(MAKE) TB=mlp comp

run_mlp:
	$(MAKE) TB=mlp run

verdi_mlp:
	$(MAKE) TB=mlp verdi

cmodel_mlp:
	$(MAKE) TB=mlp cmodel

clean:
	rm -rf "$(SIM_DIR)"/* \
	       csrc simv simv_* simv*.daidir ucli.key vc_hdrs.h \
	       *.log *.fsdb *.vpd novas.* verdiLog* lncm macm ffncm mlpcm DVEfiles .inter.vpd.uvm
