// =====================================================================
// feed_forward_network.v                                    (Verilog-2005*)
//
// Hardware implementation of the FT-Transformer position-wise feed-forward
// network (TransformerBlock.ffn in src/models/ft_transformer.py; inference,
// dropout=0, bias=True):
//
//     nn.Sequential(nn.Linear(d_token, d_ffn), <act>, nn.Dropout,
//                   nn.Linear(d_ffn, d_token))
//
// Dropout is identity at inference. The FFN is applied independently to each
// token, so one D_TOKEN-element int8 vector is consumed per clock (packed
// x_vec) and one D_TOKEN-element int8 vector is produced per clock after a
// fixed latency => II = 1. This is the same per-token x_vec/y_vec interface as
// layer_norm.v (NOT the whole-sequence interface of multihead_attention.v).
//
// ACTIVATION: the model uses GELU; this hardware uses ReLU in its place (a
// cheaper sign-bit clamp: relu(h) = max(0, h)). The pure-C twin
// src/models/feed_forward_network_cmodel.c runs the IDENTICAL integer datapath
// (ReLU too), so this RTL matches it bit-for-bit.
//
// Math (per token x of length E=D_TOKEN, hidden width F=D_FFN):
//   h[o] = relu( requant( sum_k x[k]*W1[o][k] + (b1[o]<<FRAC) ) )   o in [0,F)
//   y[o] =       requant( sum_k h[k]*W2[o][k] + (b2[o]<<FRAC) )      o in [0,E)
// where W1 = Linear1.weight (F,E) and W2 = Linear2.weight (E,F), row-major,
// exactly PyTorch's y = x @ W^T + b convention.
//
// Numeric model (int8 inference, symmetric quantization, zero-point 0):
//   * int8 values are signed Q1.FRAC_BITS (default Q1.7, scale 2^-7).
//   * each matmul accumulates int8*int8 products exactly, adds the bias
//     (int8 << FRAC_BITS), then requantizes to int8: round-half-up, arithmetic
//     right shift by FRAC_BITS, saturate to [-128,127].
//   * ReLU clamps the (already int8 Q1.7) Linear1 output's negatives to 0; it
//     needs no requant -- it is exact in both the RTL and the C model.
//
// Weights live in an FF-based register file (read in parallel each cycle)
// loaded through a write-only port, exactly like numerical_feature_tokenizer,
// layer_norm and multihead_attention.
//
// (*) Verilog-2005 dialect using the SystemVerilog procedural keywords
// always_ff / always_comb (compile with `vcs -sverilog`). No `logic` is used
// -- every signal is reg/wire -- and the design uses clk / rst_n.
// =====================================================================

`default_nettype none
module feed_forward_network #(
    parameter D_TOKEN    = 32,   // E: Linear1 in-dim / Linear2 out-dim
    parameter D_FFN      = 64,   // F: hidden width (Linear1 out / Linear2 in)
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    // ---- derived (do not override) ----
    parameter W1_DEPTH   = D_FFN * D_TOKEN,   // Linear1.weight (F,E) row-major
    parameter W2_DEPTH   = D_TOKEN * D_FFN,   // Linear2.weight (E,F) row-major
    parameter WSEL_W     = 2,
    parameter MAXW_DEPTH = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,
    parameter WADDR_W    = ($clog2(MAXW_DEPTH) < 1) ? 1 : $clog2(MAXW_DEPTH)
) (
    input  wire                          clk,
    input  wire                          rst_n,    // async assert, sync deassert
    input  wire                          wr_en,    // coefficient write strobe
    input  wire [WSEL_W-1:0]             wr_sel,   // 0=W1 1=b1 2=W2 3=b2
    input  wire [WADDR_W-1:0]            wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH-1:0]         wr_data,  // signed int8 coefficient
    input  wire                          in_valid, // x_vec valid this cycle
    input  wire [D_TOKEN*DATA_WIDTH-1:0] x_vec,    // packed: x[k] = x_vec[k*W +: W]
    output wire                          out_valid,// y_vec valid
    output wire [D_TOKEN*DATA_WIDTH-1:0] y_vec     // packed: y[o] = y_vec[o*W +: W]
);

    // ---- derived sizes (sized to hold EXACT values, no truncation) --------
    localparam CLOG2_DT = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam CLOG2_DF = ($clog2(D_FFN)   < 1) ? 1 : $clog2(D_FFN);

    localparam PROD_W   = 2 * DATA_WIDTH;                  // one int8*int8 product
    // Linear1 sums D_TOKEN products + aligned bias; Linear2 sums D_FFN products.
    localparam ACC1_W   = 2*DATA_WIDTH + CLOG2_DT + 2;
    localparam ACC2_W   = 2*DATA_WIDTH + CLOG2_DF + 2;
    // one shared requant() input width (widest accumulator), the MHA idiom.
    localparam RQ_W     = (ACC1_W > ACC2_W) ? ACC1_W : ACC2_W;

    localparam        [RQ_W-1:0] RQ_ONE  = {{(RQ_W-1){1'b0}}, 1'b1};
    localparam signed [RQ_W-1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;   // +127
    localparam signed [RQ_W-1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));      // -128

    // ---- helper functions -------------------------------------------------
    // sign-extend an int8 bias to RQ_W, then <<FRAC_BITS to align with the
    // Q(2*FRAC) product accumulator (like layer_norm's align_beta / MHA's
    // align_bias).
    function signed [RQ_W-1:0] align_bias;
        input signed [DATA_WIDTH-1:0] b;
        reg   signed [RQ_W-1:0] be;
        begin
            be         = b;
            align_bias = be <<< FRAC_BITS;
        end
    endfunction

    // round-half-up, arithmetic right shift by `shift`, saturate to int8.
    function signed [DATA_WIDTH-1:0] requant;
        input signed [RQ_W-1:0] acc;
        input integer           shift;
        reg   signed [RQ_W-1:0] rndc, s, r;
        begin
            rndc = (shift > 0) ? (RQ_ONE <<< (shift-1)) : {RQ_W{1'b0}};
            s    = acc + rndc;
            r    = s >>> shift;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            requant = r[DATA_WIDTH-1:0];
        end
    endfunction

    // ---- coefficient register file (FF-based, write-only port) ------------
    reg signed [DATA_WIDTH-1:0] w1_mem [0:W1_DEPTH-1];   // Linear1.weight (F,E)
    reg signed [DATA_WIDTH-1:0] b1_mem [0:D_FFN-1];      // Linear1.bias   (F)
    reg signed [DATA_WIDTH-1:0] w2_mem [0:W2_DEPTH-1];   // Linear2.weight (E,F)
    reg signed [DATA_WIDTH-1:0] b2_mem [0:D_TOKEN-1];    // Linear2.bias   (E)

    always_ff @(posedge clk) begin
        if (wr_en) begin
            case (wr_sel)
                2'd0: w1_mem[wr_addr] <= wr_data;
                2'd1: b1_mem[wr_addr] <= wr_data;
                2'd2: w2_mem[wr_addr] <= wr_data;
                2'd3: b2_mem[wr_addr] <= wr_data;
                default: ;
            endcase
        end
    end

    // ---- valid pipeline (4 stages; datapath self-flushes via valid) -------
    reg v1, v2, v3, v4;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3;
        end
    end
    assign out_valid = v4;

    // ---- pipeline registers -----------------------------------------------
    reg signed [DATA_WIDTH-1:0] xq [0:D_TOKEN-1];   // stage 1: latched x
    reg signed [DATA_WIDTH-1:0] hq [0:D_FFN-1];     // stage 2: Linear1 (pre-ReLU)
    reg signed [DATA_WIDTH-1:0] gq [0:D_FFN-1];     // stage 3: ReLU(Linear1)
    reg signed [DATA_WIDTH-1:0] yq [0:D_TOKEN-1];   // stage 4: Linear2 (output)

    genvar gk, go;

    // ---- stage 1: unpack x_vec -> xq --------------------------------------
    generate
        for (gk = 0; gk < D_TOKEN; gk = gk + 1) begin : gA
            always_ff @(posedge clk)
                xq[gk] <= $signed(x_vec[gk*DATA_WIDTH +: DATA_WIDTH]);
        end
    endgenerate

    // ---- stage 2: Linear1  h = requant( x.W1 + (b1<<FRAC) ) ----------------
    generate
        for (go = 0; go < D_FFN; go = go + 1) begin : gB
            reg signed [RQ_W-1:0] a1;
            integer kk;
            always_comb begin
                a1 = {RQ_W{1'b0}};
                for (kk = 0; kk < D_TOKEN; kk = kk + 1)
                    a1 = a1 + xq[kk] * w1_mem[go*D_TOKEN + kk];
                a1 = a1 + align_bias(b1_mem[go]);
            end
            always_ff @(posedge clk)
                hq[go] <= requant(a1, FRAC_BITS);
        end
    endgenerate

    // ---- stage 3: ReLU  g = max(0, h) (negatives -> 0; output stays Q1.7) --
    generate
        for (go = 0; go < D_FFN; go = go + 1) begin : gC
            always_ff @(posedge clk)
                gq[go] <= (hq[go] < 0) ? {DATA_WIDTH{1'b0}} : hq[go];
        end
    endgenerate

    // ---- stage 4: Linear2  y = requant( g.W2 + (b2<<FRAC) ) ----------------
    generate
        for (go = 0; go < D_TOKEN; go = go + 1) begin : gD
            reg signed [RQ_W-1:0] a2;
            integer kk;
            always_comb begin
                a2 = {RQ_W{1'b0}};
                for (kk = 0; kk < D_FFN; kk = kk + 1)
                    a2 = a2 + gq[kk] * w2_mem[go*D_FFN + kk];
                a2 = a2 + align_bias(b2_mem[go]);
            end
            always_ff @(posedge clk)
                yq[go] <= requant(a2, FRAC_BITS);
            assign y_vec[go*DATA_WIDTH +: DATA_WIDTH] = yq[go];
        end
    endgenerate

endmodule
`default_nettype wire
