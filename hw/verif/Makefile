# =====================================================================
# Makefile -- VCS-only end-to-end check of hw/rtl/layer_norm.v against the
# behavioral C model (src/models/layer_norm_cmodel.c) over DPI-C, with FSDB
# + KDB waveform dumping for Verdi.
#
#   make            # compile + run the end-to-end DPI check (alias: make all)
#   make comp       # compile only  -> ./simv (+ simv.daidir/kdb)
#   make run        # run ./simv     -> layer_norm.fsdb
#   make verdi      # open the FSDB in Verdi with full KDB code hierarchy
#   make cmodel     # simulator-free C-model self-test (math sanity, needs gcc)
#   make clean
#
#   make all VCS_DEFINES=+define+LN_DT=64   # sweep d_token
#   make run SIMARGS=+seed=12345            # pick a random seed
# =====================================================================

VCS      ?= vcs
VERDI    ?= verdi
GCC      ?= gcc

# repo root (this Makefile lives in hw/verif/)
PROJ     := $(abspath ../..)
MODELDIR := $(PROJ)/src/models

RTL      := ../rtl/layer_norm.v
TB       := tb_layer_norm.sv
DPI_C    := layer_norm_dpi.c
CMODEL_C := $(MODELDIR)/layer_norm_cmodel.c

SIMV     := simv
FSDB     := layer_norm.fsdb

# ---- MultiheadAttention block (self-attention) -----------------------------
RTL_MHA     := ../rtl/multihead_attention.v
TB_MHA      := tb_multihead_attention.sv
DPI_MHA     := multihead_attention_dpi.c
CMODEL_MHA  := $(MODELDIR)/multihead_attention_cmodel.c
FLIST_MHA   := multihead_attention_filelist.f
SIMV_MHA    := simv_mha
FSDB_MHA    := multihead_attention.fsdb

# ---- FeedForwardNetwork block (position-wise FFN, ReLU activation) ---------
RTL_FFN     := ../rtl/feed_forward_network.v
TB_FFN      := tb_feed_forward_network.sv
DPI_FFN     := feed_forward_network_dpi.c
CMODEL_FFN  := $(MODELDIR)/feed_forward_network_cmodel.c
FLIST_FFN   := feed_forward_network_filelist.f
SIMV_FFN    := simv_ffn
FSDB_FFN    := feed_forward_network.fsdb

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

.PHONY: all comp run verdi cmodel \
        all_mha comp_mha run_mha verdi_mha cmodel_mha \
        all_ffn comp_ffn run_ffn verdi_ffn cmodel_ffn clean

all: comp run

comp:
	$(VCS) $(VCS_FLAGS) $(VCS_DEFINES) \
	    -CFLAGS "$(CFLAGS)" \
	    -f filelist.f \
	    $(DPI_C) $(CMODEL_C) \
	    $(NOVAS_FLAGS) \
	    -l comp.log -o $(SIMV)

run:
	./$(SIMV) +fsdb+autoflush $(SIMARGS) -l run.log

verdi:
	$(VERDI) -dbdir $(SIMV).daidir/kdb -ssf $(FSDB) &

# Simulator-free: validate the C model's int8 math vs the float nn.LayerNorm
# ideal (<= 1 LSB). Useful on a box without VCS.
cmodel:
	$(GCC) -DLN_STANDALONE -O2 -std=c11 $(CMODEL_C) -o lncm -lm && ./lncm

# ---- MultiheadAttention targets --------------------------------------------
#   make all_mha                              # compile + run the DPI check
#   make verdi_mha                            # open multihead_attention.fsdb
#   make all_mha VCS_DEFINES=+define+MA_S=8   # sweep seq_len
#   make run_mha SIMARGS=+seed=12345          # pick a random seed
all_mha: comp_mha run_mha

comp_mha:
	$(VCS) $(VCS_FLAGS) $(VCS_DEFINES) \
	    -CFLAGS "$(CFLAGS)" \
	    -f $(FLIST_MHA) \
	    $(DPI_MHA) $(CMODEL_MHA) \
	    $(NOVAS_FLAGS) \
	    -l comp_mha.log -o $(SIMV_MHA)

run_mha:
	./$(SIMV_MHA) +fsdb+autoflush $(SIMARGS) -l run_mha.log

verdi_mha:
	$(VERDI) -dbdir $(SIMV_MHA).daidir/kdb -ssf $(FSDB_MHA) &

# Simulator-free: integer attention vs float reference (needs gcc).
cmodel_mha:
	$(GCC) -DMA_STANDALONE -O2 -std=c11 $(CMODEL_MHA) -o macm -lm && ./macm

# ---- FeedForwardNetwork targets --------------------------------------------
#   make all_ffn                                  # compile + run the DPI check
#   make verdi_ffn                                # open feed_forward_network.fsdb
#   make all_ffn VCS_DEFINES=+define+FFN_DFFN=128 # sweep d_ffn
#   make run_ffn SIMARGS=+seed=12345              # pick a random seed
all_ffn: comp_ffn run_ffn

comp_ffn:
	$(VCS) $(VCS_FLAGS) $(VCS_DEFINES) \
	    -CFLAGS "$(CFLAGS)" \
	    -f $(FLIST_FFN) \
	    $(DPI_FFN) $(CMODEL_FFN) \
	    $(NOVAS_FLAGS) \
	    -l comp_ffn.log -o $(SIMV_FFN)

run_ffn:
	./$(SIMV_FFN) +fsdb+autoflush $(SIMARGS) -l run_ffn.log

verdi_ffn:
	$(VERDI) -dbdir $(SIMV_FFN).daidir/kdb -ssf $(FSDB_FFN) &

# Simulator-free: int8 ReLU FFN vs an independent fixed-point reference (gcc).
cmodel_ffn:
	$(GCC) -DFFN_STANDALONE -O2 -std=c11 $(CMODEL_FFN) -o ffncm -lm && ./ffncm

clean:
	rm -rf $(SIMV) $(SIMV_MHA) $(SIMV_FFN) \
	       $(SIMV).daidir $(SIMV_MHA).daidir $(SIMV_FFN).daidir \
	       csrc ucli.key vc_hdrs.h \
	       *.log *.fsdb *.vpd novas.* verdiLog* lncm macm ffncm DVEfiles .inter.vpd.uvm
