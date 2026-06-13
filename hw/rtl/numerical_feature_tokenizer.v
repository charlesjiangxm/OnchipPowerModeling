// =====================================================================
// numerical_feature_tokenizer.v                             (Verilog-2005*)
//
// Hardware implementation of the FT-Transformer NumericalFeatureTokenizer
// (src/models/ft_transformer.py). For an input row x of N_FEATURE elements
// and learned weight/bias of shape (N_FEATURE, D_TOKEN):
//
//     out[j][k] = x[j] * weight[j][k] + bias[j][k]
//
// This is a purely element-wise broadcast multiply-add (no reduction over
// features), so each of the N_FEATURE*D_TOKEN outputs is one multiply plus
// one bias add. One input row is consumed per clock and the full
// N_FEATURE*D_TOKEN token matrix is produced every clock after a fixed
// pipeline latency  =>  initiation interval (II) = 1.
//
// Numeric model (int8 inference, symmetric quantization, zero-point = 0):
//   * int8 values are read as signed Q1.FRAC_BITS fixed-point (default Q1.7,
//     scale 2^-7, range [-1, +0.992...]).
//   * product x*W is Q2.(2*FRAC_BITS); bias (Q1.FRAC_BITS) is left-shifted by
//     FRAC_BITS to align, then added.
//   * the sum is requantized back to int8: round-half-up, arithmetic right
//     shift by FRAC_BITS, then saturate to the signed DATA_WIDTH range.
//
// Weights/biases live in an FF-based register file (so every coefficient can
// be read in parallel each cycle) and are loaded through a write-only port.
//
// ANSI ports + $clog2 size the write-address port; the "*" operator lets
// the synthesis tool infer hard multipliers / DSP blocks.
//
// (*) Verilog-2005 dialect, but uses the SystemVerilog procedural keywords
// always_ff / always_comb (compile with `vcs -sverilog`, analyze with
// `analyze -format sverilog`). No `logic` is used -- every signal is
// `reg`/`wire` -- and the design uses clk / rst_n.
// =====================================================================

`default_nettype none
module numerical_feature_tokenizer #(
    parameter N_FEATURE  = 20,   // number of features (rows of weight/bias)
    parameter D_TOKEN    = 32,   // token dimension   (cols of weight/bias)
    parameter DATA_WIDTH = 8,    // int8 element width (x, weight, bias, out)
    parameter FRAC_BITS  = 7,    // fractional bits => int8 read as Q1.7
    // ---- derived (do not override) ----
    parameter DEPTH      = N_FEATURE * D_TOKEN,
    parameter ADDR_W     = ($clog2(DEPTH) < 1) ? 1 : $clog2(DEPTH)
) (
    input  wire                                    clk,
    input  wire                                    rst_n,       // async assert, sync deassert
    input  wire                                    wr_en,       // coefficient write strobe
    input  wire                                    wr_is_bias,  // 0=weight, 1=bias
    input  wire [ADDR_W-1:0]                       wr_addr,     // linear index j*D_TOKEN+k
    input  wire [DATA_WIDTH-1:0]                   wr_data,     // signed int8 coefficient
    input  wire                                    in_valid,    // x_row valid this cycle
    input  wire [N_FEATURE*DATA_WIDTH-1:0]         x_row,       // packed: x[j] = x_row[j*W +: W]
    output wire                                    out_valid,   // out_tokens valid
    output wire [N_FEATURE*D_TOKEN*DATA_WIDTH-1:0] out_tokens   // packed: out[j][k] at (j*D+k)
);

    // ---- derived sizes ------------------------------------------------
    localparam PROD_W = 2 * DATA_WIDTH;
    // accumulator wide enough for the product, the shifted bias, and the
    // round-half-up margin.
    localparam ACC_W  = ((PROD_W > (DATA_WIDTH + FRAC_BITS)) ?
                          PROD_W : (DATA_WIDTH + FRAC_BITS)) + 2;

    // requant rounding constant and signed int saturation bounds.
    localparam signed [ACC_W-1:0] ROUND   = (FRAC_BITS == 0) ? 0 : (1 <<< (FRAC_BITS-1));
    localparam signed [ACC_W-1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam signed [ACC_W-1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));

    // ---- helper functions ---------------------------------------------
    // Sign-extend an int8 bias to ACC_W, then shift left by FRAC_BITS so it
    // aligns with the product's fractional scale (Q.(2*FRAC_BITS)).
    function signed [ACC_W-1:0] align_bias;
        input signed [DATA_WIDTH-1:0] b;
        reg   signed [ACC_W-1:0] be;
        begin
            be         = b;                 // sign-extend to ACC_W
            align_bias = be <<< FRAC_BITS;  // shift inside ACC_W width (no loss)
        end
    endfunction

    // Round-half-up, arithmetic right shift by FRAC_BITS, saturate to int8.
    function signed [DATA_WIDTH-1:0] requant;
        input signed [ACC_W-1:0] acc;
        reg   signed [ACC_W-1:0] s;
        reg   signed [ACC_W-1:0] r;
        begin
            s = acc + ROUND;
            r = s >>> FRAC_BITS;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            requant = r[DATA_WIDTH-1:0];
        end
    endfunction

    // ---- coefficient register file (FF-based, write-only port) --------
    reg signed [DATA_WIDTH-1:0] weight_mem [0:DEPTH-1];
    reg signed [DATA_WIDTH-1:0] bias_mem   [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (wr_is_bias) bias_mem[wr_addr]   <= wr_data;
            else            weight_mem[wr_addr] <= wr_data;
        end
    end

    // ---- pipeline registers -------------------------------------------
    reg signed [DATA_WIDTH-1:0] x_q    [0:N_FEATURE-1];  // stage 1: latched x (one per feature)
    reg signed [PROD_W-1:0]     prod_q [0:DEPTH-1];      // stage 2: x*W product
    reg signed [DATA_WIDTH-1:0] tok_q  [0:DEPTH-1];      // stage 3: requantized output
    reg                         v1, v2, v3;              // valid pipeline

    // Valid pipeline is resettable; the datapath self-flushes via valid
    // gating (no need to reset the wide data registers).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2;
        end
    end
    assign out_valid = v3;

    // ---- datapath: N_FEATURE * D_TOKEN parallel lanes -----------------
    genvar gj, gk;
    generate
        // Stage 1: latch one x per feature (shared across all D_TOKEN lanes).
        for (gj = 0; gj < N_FEATURE; gj = gj + 1) begin : g_xlatch
            always_ff @(posedge clk)
                x_q[gj] <= $signed(x_row[gj*DATA_WIDTH +: DATA_WIDTH]);
        end

        // Stages 2-3: per (feature j, token k) multiply -> bias add -> requant.
        for (gj = 0; gj < N_FEATURE; gj = gj + 1) begin : g_lane_j
            for (gk = 0; gk < D_TOKEN; gk = gk + 1) begin : g_lane_k
                reg signed [PROD_W-1:0] prod_c;  // stage 2 combinational product
                reg signed [ACC_W-1:0]  acc_c;   // stage 3 combinational bias-add

                // Stage 2: product (uses freshly latched x and static weight).
                always_comb prod_c = x_q[gj] * weight_mem[gj*D_TOKEN + gk];
                always_ff @(posedge clk)
                    prod_q[gj*D_TOKEN + gk] <= prod_c;

                // Stage 3: add aligned bias, requantize to int8.
                always_comb acc_c =
                    prod_q[gj*D_TOKEN + gk] + align_bias(bias_mem[gj*D_TOKEN + gk]);
                always_ff @(posedge clk)
                    tok_q[gj*D_TOKEN + gk] <= requant(acc_c);

                assign out_tokens[(gj*D_TOKEN + gk)*DATA_WIDTH +: DATA_WIDTH] =
                        tok_q[gj*D_TOKEN + gk];
            end
        end
    endgenerate

endmodule
`default_nettype wire
