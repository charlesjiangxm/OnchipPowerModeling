// HDL sources for the MLP end-to-end DPI-C check.
// The DPI glue (utils/mlp_dpi.c) and the C model (hw/model/mlp_cmodel.c) are
// passed directly on the vcs command line by the Makefile.
// requant_rne -> dyn_quant -> mlp (instantiation order).
../rtl/requant_rne.v
../rtl/dyn_quant.v
../rtl/mlp.v
tb/tb_mlp.sv
