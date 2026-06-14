//-----------------------------------------------------------------------------
// numerical_feature_tokenizer.v                               (Verilog-2005*)
//
// title    : FT-Transformer NumericalFeatureTokenizer (int8 inference)
//            (src/models/ft_transformer.py NumericalFeatureTokenizer.forward)
// math     : out[j][k] = x[j] * weight[j][k] + bias[j][k]
//            elementwise broadcast multiply-add, no reduction over features.
// numeric  : int8 = signed Q1.FRAC_BITS (zero-point 0). product is Q.(2*FRAC),
//            bias is left-shifted FRAC_BITS to align, sum requantized to int8
//            (round-half-up, arithmetic >> FRAC_BITS, saturate to [-128,127]).
// datapath : FF-based coefficient regfile -> N_FEATURE*D_TOKEN parallel lanes
//            (latch x -> multiply -> bias-add -> requant).
// schedule : one x_row in / full token matrix out per clock, latency 3, II = 1.
// params   : N_FEATURE, D_TOKEN, DATA_WIDTH, FRAC_BITS (DEPTH/ADDR_W derived).
// language : Verilog-2005 + SystemVerilog always_ff/always_comb (no logic);
//            compile with `vcs -sverilog`, analyze with -format sverilog.
//            clk / rst_n (async assert, sync deassert).
//-----------------------------------------------------------------------------
`default_nettype none
module numerical_feature_tokenizer #(
    parameter N_FEATURE  = 20,   // number of features (rows of weight/bias)
    parameter D_TOKEN    = 32,   // token dimension   (cols of weight/bias)
    parameter DATA_WIDTH = 8,    // int8 element width (x, weight, bias, out)
    parameter FRAC_BITS  = 7,    // fractional bits => int8 read as Q1.7
    // ---- derived (do not override) ----
    parameter DEPTH      = N_FEATURE * D_TOKEN,                    // regfile depth
    parameter ADDR_W     = ($clog2(DEPTH) < 1) ? 1 : $clog2(DEPTH) // write-address width
) (
    // control port
    input  wire                                     clk,            // clock
    input  wire                                     rst_n,          // async assert, sync deassert
    input  wire                                     i_wr_en,        // coefficient write strobe
    input  wire                                     i_wr_is_bias,   // 0=weight, 1=bias
    input  wire [ADDR_W     -1:0]                   i_wr_addr,      // linear index j*D_TOKEN+k
    input  wire [DATA_WIDTH -1:0]                   i_wr_data,      // signed int8 coefficient
    input  wire                                     i_valid,        // x_row valid this cycle
    // data port
    input  wire [N_FEATURE*DATA_WIDTH -1:0]         i_x_row,        // packed: x[j] = i_x_row[j*W +: W]
    output wire                                     o_valid,        // o_tokens valid
    output wire [N_FEATURE*D_TOKEN*DATA_WIDTH -1:0] o_tokens        // packed: out[j][k] at (j*D+k)
);

    genvar j, k;   // feature index / token index

    localparam PIPE   = 3;             // valid-pipeline / latency depth
    localparam PROD_W = 2 * DATA_WIDTH; // one int8*int8 product
    // accumulator wide enough for the product, the shifted bias, and the
    // round-half-up margin.
    localparam ACC_W  = ((PROD_W > (DATA_WIDTH + FRAC_BITS)) ?
                          PROD_W : (DATA_WIDTH + FRAC_BITS)) + 2;

    // ---- coefficient register file (FF-based, write-only port) --------------
    reg signed [DATA_WIDTH -1:0] weight_mem [DEPTH-1 :0];   // x*weight coefficients
    reg signed [DATA_WIDTH -1:0] bias_mem   [DEPTH-1 :0];   // additive bias coefficients

    always_ff @(posedge clk) begin : DFF_WR
        if (i_wr_en) begin
            if (i_wr_is_bias) bias_mem[i_wr_addr]   <= i_wr_data;
            else              weight_mem[i_wr_addr] <= i_wr_data;
        end
    end

    // ---- valid pipeline (resettable; datapath self-flushes via valid) -------
    reg [PIPE -1:0] valid_ff;   // PIPE-stage valid shift register

    always_ff @(posedge clk or negedge rst_n) begin : DFF_VLD
        if (!rst_n) valid_ff <= {PIPE{1'b0}};
        else        valid_ff <= {valid_ff[PIPE-2:0], i_valid};
    end
    assign o_valid = valid_ff[PIPE-1];

    // ---- pipeline registers -------------------------------------------------
    reg signed [DATA_WIDTH -1:0] x_ff    [N_FEATURE-1 :0];   // stage 1: latched x (per feature)
    reg signed [PROD_W     -1:0] prod_ff [DEPTH-1     :0];   // stage 2: x*W product
    reg signed [DATA_WIDTH -1:0] tok_ff  [DEPTH-1     :0];   // stage 3: requantized output

    // ---- stage 1: latch one x per feature (shared across all D_TOKEN lanes) -
    generate
        for (j = 0; j < N_FEATURE; j = j + 1) begin : G_XLATCH
            always_ff @(posedge clk) begin : DFF_X
                x_ff[j] <= $signed(i_x_row[j*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endgenerate

    // ---- stages 2-3: per (feature j, token k) multiply -> bias-add -> requant
    generate
        for (j = 0; j < N_FEATURE; j = j + 1) begin : G_LANE_J
            for (k = 0; k < D_TOKEN; k = k + 1) begin : G_LANE_K
                wire signed [PROD_W     -1:0] prod_c;   // stage 2 product x*W
                wire signed [ACC_W      -1:0] bias_al;  // aligned bias (bias << FRAC)
                wire signed [ACC_W      -1:0] acc_c;    // stage 3 bias-add accumulator
                wire signed [DATA_WIDTH -1:0] tok_c;    // requantized lane output

                // stage 2: product (freshly latched x, static weight)
                assign prod_c = x_ff[j] * weight_mem[j*D_TOKEN + k];
                always_ff @(posedge clk) begin : DFF_PROD
                    prod_ff[j*D_TOKEN + k] <= prod_c;
                end

                // stage 3: add aligned bias, requantize to int8
                align_bias #(
                    .IN_W  (DATA_WIDTH),
                    .OUT_W (ACC_W),
                    .SH    (FRAC_BITS)
                ) U_ALIGN_BIAS (
                    .i_b       (bias_mem[j*D_TOKEN + k]),
                    .o_aligned (bias_al)
                );
                assign acc_c = prod_ff[j*D_TOKEN + k] + bias_al;
                requant #(
                    .ACC_W      (ACC_W),
                    .DATA_WIDTH (DATA_WIDTH),
                    .SHIFT      (FRAC_BITS)
                ) U_REQUANT (
                    .i_acc (acc_c),
                    .o_q   (tok_c)
                );
                always_ff @(posedge clk) begin : DFF_TOK
                    tok_ff[j*D_TOKEN + k] <= tok_c;
                end

                assign o_tokens[(j*D_TOKEN + k)*DATA_WIDTH +: DATA_WIDTH] =
                        tok_ff[j*D_TOKEN + k];
            end
        end
    endgenerate

endmodule
`default_nettype wire
