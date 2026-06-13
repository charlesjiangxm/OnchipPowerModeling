// HDL sources for the full FT-Transformer end-to-end DPI-C check (VCS -f).
// The DPI glue (ft_transformer_dpi.c) and the C models
// (src/models/ft_transformer_cmodel.c + the four block twins) are passed
// directly on the vcs command line by the Makefile.
../rtl/numerical_feature_tokenizer.v
../rtl/layer_norm.v
../rtl/feed_forward_network.v
../rtl/multihead_attention.v
../rtl/residual_add.v
../rtl/head.v
../rtl/ft_transformer_top.v
tb_ft_transformer.sv
