// HDL sources for the LayerNorm end-to-end DPI-C check.
// The DPI glue (utils/layer_norm_dpi.c) and the C model (src/models/layer_norm_cmodel.c)
// are passed directly on the vcs command line by the Makefile.
../rtl/layer_norm.v
tb/tb_layer_norm.sv
