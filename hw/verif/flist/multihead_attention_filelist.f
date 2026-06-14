// HDL sources for the MultiheadAttention end-to-end DPI-C check.
// The DPI glue (utils/multihead_attention_dpi.c) and the C model
// (hw/model/multihead_attention_cmodel.c) are passed directly on the vcs
// command line by the Makefile.
../rtl/multihead_attention.v
tb/tb_multihead_attention.sv
