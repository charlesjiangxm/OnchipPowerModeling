// HDL sources for the FeedForwardNetwork end-to-end DPI-C check.
// The DPI glue (utils/feed_forward_network_dpi.c) and the C model
// (hw/model/feed_forward_network_cmodel.c) are passed directly on the vcs
// command line by the Makefile.
../rtl/requant.v
../rtl/align_bias.v
../rtl/feed_forward_network.v
tb/tb_feed_forward_network.sv
