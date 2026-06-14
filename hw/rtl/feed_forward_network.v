//-----------------------------------------------------------------------------
// feed_forward_network.v                                       (Verilog-2005*)
//
// title    : FT-Transformer position-wise feed-forward network (int8 inference)
//            (TransformerBlock.ffn: Linear1 -> act -> Linear2; dropout=0)
// math     : h[o] = relu( requant( sum_k x[k]*W1[o][k] + (b1[o]<<FRAC) ) )
//            y[o] =       requant( sum_k h[k]*W2[o][k] + (b2[o]<<FRAC) )
//            W1 = Linear1.weight (F,E), W2 = Linear2.weight (E,F), row-major.
// numeric  : int8 = signed Q1.FRAC_BITS (zero-point 0). matmul accumulates
//            int8*int8 exactly, adds bias (int8<<FRAC), requantizes to int8
//            (round-half-up, arithmetic >> FRAC, saturate). ACTIVATION = ReLU
//            (a sign-bit clamp) in place of the model's GELU, matching
//            src/models/feed_forward_network_cmodel.c bit-for-bit.
// datapath : FF weight regfile -> Linear1 (U_REQUANT) -> ReLU -> Linear2.
// schedule : one x_vec in / one y_vec out per clock, latency 4, II = 1.
// params   : D_TOKEN (E), D_FFN (F), DATA_WIDTH, FRAC_BITS.
// language : Verilog-2005 + SystemVerilog always_ff/always_comb (no logic);
//            compile with `vcs -sverilog`. clk / rst_n (async assert, sync deassert).
//-----------------------------------------------------------------------------
`default_nettype none
module feed_forward_network #(
    parameter D_TOKEN    = 32,   // E: Linear1 in-dim / Linear2 out-dim
    parameter D_FFN      = 64,   // F: hidden width (Linear1 out / Linear2 in)
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    // ---- derived (do not override) ----
    parameter W1_DEPTH   = D_FFN * D_TOKEN,                                  // Linear1.weight (F,E)
    parameter W2_DEPTH   = D_TOKEN * D_FFN,                                  // Linear2.weight (E,F)
    parameter WSEL_W     = 2,                                                // wr_sel width
    parameter MAXW_DEPTH = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,      // widest weight array
    parameter WADDR_W    = ($clog2(MAXW_DEPTH) < 1) ? 1 : $clog2(MAXW_DEPTH) // write-address width
) (
    // control port
    input  wire                           clk,        // clock
    input  wire                           rst_n,      // async assert, sync deassert
    input  wire                           i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]            i_wr_sel,   // 0=W1 1=b1 2=W2 3=b2
    input  wire [WADDR_W -1:0]            i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]         i_wr_data,  // signed int8 coefficient
    input  wire                           i_valid,    // x_vec valid this cycle
    // data port
    input  wire [D_TOKEN*DATA_WIDTH -1:0] i_x_vec,    // packed: x[k] = i_x_vec[k*W +: W]
    output wire                           o_valid,    // y_vec valid
    output wire [D_TOKEN*DATA_WIDTH -1:0] o_y_vec     // packed: y[o] = o_y_vec[o*W +: W]
);

    integer m;       // procedural matmul accumulation index
    genvar  k, o;    // input / output generate indices

    localparam PIPE     = 4;                            // valid-pipeline / latency depth
    localparam CLOG2_DT = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam CLOG2_DF = ($clog2(D_FFN)   < 1) ? 1 : $clog2(D_FFN);
    // Linear1 sums D_TOKEN products + aligned bias; Linear2 sums D_FFN products.
    localparam ACC1_W   = 2*DATA_WIDTH + CLOG2_DT + 2;  // Linear1 accumulator
    localparam ACC2_W   = 2*DATA_WIDTH + CLOG2_DF + 2;  // Linear2 accumulator

    // ---- coefficient register file (FF-based, write-only port) --------------
    reg signed [DATA_WIDTH -1:0] w1_mem [W1_DEPTH-1 :0];   // Linear1.weight (F,E) row-major
    reg signed [DATA_WIDTH -1:0] b1_mem [D_FFN-1    :0];   // Linear1.bias   (F)
    reg signed [DATA_WIDTH -1:0] w2_mem [W2_DEPTH-1 :0];   // Linear2.weight (E,F) row-major
    reg signed [DATA_WIDTH -1:0] b2_mem [D_TOKEN-1  :0];   // Linear2.bias   (E)

    always_ff @(posedge clk) begin : DFF_WR
        if (i_wr_en) begin
            case (i_wr_sel)
                2'd0: w1_mem[i_wr_addr] <= i_wr_data;
                2'd1: b1_mem[i_wr_addr] <= i_wr_data;
                2'd2: w2_mem[i_wr_addr] <= i_wr_data;
                2'd3: b2_mem[i_wr_addr] <= i_wr_data;
                default: ;
            endcase
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
    reg signed [DATA_WIDTH -1:0] xq_ff [D_TOKEN-1 :0];   // stage 1: latched x
    reg signed [DATA_WIDTH -1:0] hq_ff [D_FFN-1   :0];   // stage 2: Linear1 (pre-ReLU)
    reg signed [DATA_WIDTH -1:0] gq_ff [D_FFN-1   :0];   // stage 3: ReLU(Linear1)
    reg signed [DATA_WIDTH -1:0] yq_ff [D_TOKEN-1 :0];   // stage 4: Linear2 (output)

    // ---- stage 1: unpack x_vec -> xq_ff -------------------------------------
    generate
        for (k = 0; k < D_TOKEN; k = k + 1) begin : G_XLATCH
            always_ff @(posedge clk) begin : DFF_X
                xq_ff[k] <= $signed(i_x_vec[k*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endgenerate

    // ---- stage 2: Linear1  h = requant( x.W1 + (b1<<FRAC) ) -----------------
    generate
        for (o = 0; o < D_FFN; o = o + 1) begin : G_L1
            reg  signed [ACC1_W     -1:0] a1;     // matmul accumulator
            wire signed [ACC1_W     -1:0] b1_al;  // aligned bias (b1 << FRAC)
            wire signed [ACC1_W     -1:0] acc1;   // matmul + bias
            wire signed [DATA_WIDTH -1:0] h_c;    // requantized Linear1 output

            always_comb begin : CMB_L1
                a1 = {ACC1_W{1'b0}};
                for (m = 0; m < D_TOKEN; m = m + 1)
                    a1 = a1 + xq_ff[m] * w1_mem[o*D_TOKEN + m];
            end
            align_bias #(
                .IN_W  (DATA_WIDTH),
                .OUT_W (ACC1_W),
                .SH    (FRAC_BITS)
            ) U_ALIGN_B1 (
                .i_b       (b1_mem[o]),
                .o_aligned (b1_al)
            );
            assign acc1 = a1 + b1_al;
            requant #(
                .ACC_W      (ACC1_W),
                .DATA_WIDTH (DATA_WIDTH),
                .SHIFT      (FRAC_BITS)
            ) U_REQUANT_L1 (
                .i_acc (acc1),
                .o_q   (h_c)
            );
            always_ff @(posedge clk) begin : DFF_H
                hq_ff[o] <= h_c;
            end
        end
    endgenerate

    // ---- stage 3: ReLU  g = max(0, h) (negatives -> 0; output stays Q1.7) ---
    generate
        for (o = 0; o < D_FFN; o = o + 1) begin : G_RELU
            always_ff @(posedge clk) begin : DFF_G
                gq_ff[o] <= (hq_ff[o] < 0) ? {DATA_WIDTH{1'b0}} : hq_ff[o];
            end
        end
    endgenerate

    // ---- stage 4: Linear2  y = requant( g.W2 + (b2<<FRAC) ) -----------------
    generate
        for (o = 0; o < D_TOKEN; o = o + 1) begin : G_L2
            reg  signed [ACC2_W     -1:0] a2;     // matmul accumulator
            wire signed [ACC2_W     -1:0] b2_al;  // aligned bias (b2 << FRAC)
            wire signed [ACC2_W     -1:0] acc2;   // matmul + bias
            wire signed [DATA_WIDTH -1:0] y_c;    // requantized Linear2 output

            always_comb begin : CMB_L2
                a2 = {ACC2_W{1'b0}};
                for (m = 0; m < D_FFN; m = m + 1)
                    a2 = a2 + gq_ff[m] * w2_mem[o*D_FFN + m];
            end
            align_bias #(
                .IN_W  (DATA_WIDTH),
                .OUT_W (ACC2_W),
                .SH    (FRAC_BITS)
            ) U_ALIGN_B2 (
                .i_b       (b2_mem[o]),
                .o_aligned (b2_al)
            );
            assign acc2 = a2 + b2_al;
            requant #(
                .ACC_W      (ACC2_W),
                .DATA_WIDTH (DATA_WIDTH),
                .SHIFT      (FRAC_BITS)
            ) U_REQUANT_L2 (
                .i_acc (acc2),
                .o_q   (y_c)
            );
            always_ff @(posedge clk) begin : DFF_Y
                yq_ff[o] <= y_c;
            end

            assign o_y_vec[o*DATA_WIDTH +: DATA_WIDTH] = yq_ff[o];
        end
    endgenerate

endmodule
`default_nettype wire
