// =====================================================================
// head.v                                                   (Verilog-2005*)
//
// Final regression head of the FT-Transformer (FTTransformer.forward in
// src/models/ft_transformer.py):
//
//     y = head( relu(cls_out) )     # head = nn.Linear(d_token, 1)
//
// Operates on the final_norm output of the CLS token (token 0), a length
// D_TOKEN int8 Q1.7 vector. Applies relu, the length-D_TOKEN dot product with
// head.weight (Q1.7) plus head.bias (Q1.7, << FRAC_BITS aligned), and emits the
// WIDE accumulator -- signed Q(2*FRAC_BITS) -- NOT a requantized int8. A single
// regression value quantized to int8 Q1.7 ([-1,1)) would be far too coarse, so
// the full-precision accumulator is the output; dequantize as
// y_float = y_out / 2^(2*FRAC_BITS), then Standardizer.inverse_y downstream.
//
// Bit-exact twin: ft_head_int8() in src/models/ft_transformer_cmodel.c.
//   acc = sum_k max(0,x[k])*head_w[k] + (head_b << FRAC_BITS);   y_out = acc.
//
// Coefficients live in an FF register file loaded through a write-only port,
// exactly like the other blocks (wr_is_bias selects head_w vs head_b).
//
// (*) Verilog-2005 dialect using always_ff/always_comb (compile -sverilog);
// reg/wire only; clk / rst_n (async assert, sync deassert).
// =====================================================================

`default_nettype none
module head #(
    parameter D_TOKEN    = 32,   // input vector length (= d_token)
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    // ---- derived (do not override) ----
    parameter ADDR_W     = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN),
    parameter ACC_W      = 2*DATA_WIDTH + (($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN)) + 2,
    parameter OUT_W      = 32    // wide signed output bus (holds ACC_W, room to spare)
) (
    input  wire                          clk,
    input  wire                          rst_n,      // async assert, sync deassert
    input  wire                          wr_en,      // coefficient write strobe
    input  wire                          wr_is_bias, // 0 = head_w[wr_addr], 1 = head_b
    input  wire [ADDR_W-1:0]             wr_addr,    // weight lane index k
    input  wire [DATA_WIDTH-1:0]         wr_data,    // signed int8 coefficient
    input  wire                          in_valid,   // x_vec valid this cycle
    input  wire [D_TOKEN*DATA_WIDTH-1:0] x_vec,      // cls_out (Q1.7): x[k]=x_vec[k*W +: W]
    output wire                          out_valid,  // y_out valid
    output wire signed [OUT_W-1:0]       y_out       // wide signed Q(2*FRAC_BITS) result
);

    // ---- coefficient register file (FF-based, write-only port) ------------
    reg signed [DATA_WIDTH-1:0] hw_mem [0:D_TOKEN-1];   // head.weight (D_TOKEN)
    reg signed [DATA_WIDTH-1:0] hb_reg;                 // head.bias   (scalar)

    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (wr_is_bias) hb_reg          <= wr_data;
            else            hw_mem[wr_addr] <= wr_data;
        end
    end

    // ---- valid pipeline (2 stages; datapath self-flushes via valid) -------
    reg v1, v2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1 <= 1'b0; v2 <= 1'b0; end
        else        begin v1 <= in_valid; v2 <= v1; end
    end
    assign out_valid = v2;

    // ---- stage 1: latch relu(x) -------------------------------------------
    reg signed [DATA_WIDTH-1:0] rq [0:D_TOKEN-1];  // relu(x), non-negative
    genvar gk;
    generate
        for (gk = 0; gk < D_TOKEN; gk = gk + 1) begin : g_relu
            wire signed [DATA_WIDTH-1:0] xs = $signed(x_vec[gk*DATA_WIDTH +: DATA_WIDTH]);
            always_ff @(posedge clk)
                rq[gk] <= (xs < 0) ? {DATA_WIDTH{1'b0}} : xs;
        end
    endgenerate

    // ---- stage 2: acc = relu(x).head_w + (head_b<<FRAC); emit wide --------
    reg signed [ACC_W-1:0] acc_c;
    reg signed [ACC_W-1:0] bacc;
    reg signed [OUT_W-1:0] y_q;
    integer kk;
    always_comb begin
        acc_c = {ACC_W{1'b0}};
        for (kk = 0; kk < D_TOKEN; kk = kk + 1)
            acc_c = acc_c + rq[kk] * hw_mem[kk];
        bacc  = hb_reg;                 // sign-extend int8 bias to ACC_W
        acc_c = acc_c + (bacc <<< FRAC_BITS);
    end
    always_ff @(posedge clk)
        y_q <= {{(OUT_W-ACC_W){acc_c[ACC_W-1]}}, acc_c};   // sign-extend to OUT_W

    assign y_out = y_q;

endmodule
`default_nettype wire
