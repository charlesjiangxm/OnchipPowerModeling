//-----------------------------------------------------------------------------
// mlp.v                                                        (Verilog-2005*)
//
// title    : 3-layer MLP accelerator (SmallMLP.forward, int8 dynamic-quant)
//            src/models/mlp.py: relu(fc1) -> relu(fc2) -> fc3 (dropout off)
// math     : fc1  a1[j] = relu( b1[j] + sum_i ( x[i] ? W1[j][i] : 0 ) ) // x is 1-bit {0,1}
//            req1 h1    = dyn_quant(a1)                                  // block-FP, RNE, int8
//            fc2  a2[k] = relu( b2[k] + sum_i h1[i]*W2[k][i] )
//            req2 h2    = dyn_quant(a2)
//            fc3  a3    =       b3[0] + sum_i h2[i]*W3[i]
//            req3 y     = dyn_quant(a3)  -> int8 + shift
//            W1(H1,NF) W2(H2,H1) W3(1,H2) row-major; bias added directly (no <<).
// numeric  : pure integer. x in {0,1} (unipolar) so fc1 is a gated adder tree
//            (no multipliers). weights/biases signed int8. After each VMM the
//            full-precision result is requantized to int8 with ONE data-dependent
//            shift per vector (max|v| -> shift) and round-to-nearest-ties-to-even
//            (dyn_quant -> requant_rne). fc3's shift is exported on o_shift so
//            the host can recover the integer VMM3 magnitude (o_y << o_shift).
// datapath : FF coeff regfile -> fc1(gated) -> req1 -> fc2 -> req2 -> fc3 -> req3.
// schedule : one 1-bit x in / one int8 y out per clock, latency 7, II = 1.
// params   : N_FEATURES, HIDDEN1, HIDDEN2, DATA_WIDTH.
// language : Verilog-2005 + SystemVerilog always_ff/always_comb (no logic);
//            compile with `vcs -sverilog`. clk / rst_n (async assert, sync deassert).
//-----------------------------------------------------------------------------
`default_nettype none
module mlp #(
    parameter N_FEATURES = 32,   // fc1 in-dim (1-bit input vector length)
    parameter HIDDEN1    = 16,   // fc1 out-dim / fc2 in-dim
    parameter HIDDEN2    = 16,   // fc2 out-dim / fc3 in-dim
    parameter DATA_WIDTH = 8,    // signed int8 weights / bias / output
    // ---- derived (do not override) ----
    parameter W1_DEPTH = HIDDEN1 * N_FEATURES,                            // fc1.weight (H1,NF)
    parameter W2_DEPTH = HIDDEN2 * HIDDEN1,                               // fc2.weight (H2,H1)
    parameter W3_DEPTH = HIDDEN2,                                         // fc3.weight (1,H2)
    parameter WSEL_W   = 3,                                               // wr_sel width (6 arrays)
    parameter MAXW_D1  = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,
    parameter MAXW_DEP = (MAXW_D1 > W3_DEPTH) ? MAXW_D1 : W3_DEPTH,       // widest array
    parameter WADDR_W  = ($clog2(MAXW_DEP) < 1) ? 1 : $clog2(MAXW_DEP),   // write-address width
    parameter CLOG2_NF = ($clog2(N_FEATURES) < 1) ? 1 : $clog2(N_FEATURES),
    parameter CLOG2_H1 = ($clog2(HIDDEN1)    < 1) ? 1 : $clog2(HIDDEN1),
    parameter CLOG2_H2 = ($clog2(HIDDEN2)    < 1) ? 1 : $clog2(HIDDEN2),
    parameter ACC1_W   = DATA_WIDTH   + CLOG2_NF + 2,                     // fc1 acc (gated sum + bias)
    parameter ACC2_W   = 2*DATA_WIDTH + CLOG2_H1 + 2,                     // fc2 acc (int8 matmul + bias)
    parameter ACC3_W   = 2*DATA_WIDTH + CLOG2_H2 + 2,                     // fc3 acc (int8 matmul + bias)
    parameter SHW3     = ($clog2(ACC3_W+1) < 1) ? 1 : $clog2(ACC3_W+1)    // o_shift width (fc3)
) (
    // control port
    input  wire                          clk,        // clock
    input  wire                          rst_n,      // async assert, sync deassert
    input  wire                          i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]           i_wr_sel,   // 0=W1 1=b1 2=W2 3=b2 4=W3 5=b3
    input  wire [WADDR_W -1:0]           i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]        i_wr_data,  // signed int8 coefficient
    input  wire                          i_valid,    // x valid this cycle
    // data port
    input  wire [N_FEATURES -1:0]        i_x,        // 1-bit input vector: x[i]=i_x[i]
    output wire                          o_valid,    // o_y / o_shift valid
    output wire signed [DATA_WIDTH -1:0] o_y,        // int8 result (fc3, dynamic-quant)
    output wire [SHW3 -1:0]              o_shift     // fc3 dynamic right-shift
);

    integer m;            // procedural matmul accumulation index
    genvar  j, k;         // output generate indices

    localparam PIPE = 7;                                                // latency / valid depth
    localparam SHW1 = ($clog2(ACC1_W+1) < 1) ? 1 : $clog2(ACC1_W+1);    // req1 shift width
    localparam SHW2 = ($clog2(ACC2_W+1) < 1) ? 1 : $clog2(ACC2_W+1);    // req2 shift width

    // ---- coefficient register file (FF-based, write-only port) -------------
    reg signed [DATA_WIDTH -1:0] w1_mem [W1_DEPTH-1 :0];   // fc1.weight (H1,NF) row-major
    reg signed [DATA_WIDTH -1:0] b1_mem [HIDDEN1-1  :0];   // fc1.bias   (H1)
    reg signed [DATA_WIDTH -1:0] w2_mem [W2_DEPTH-1 :0];   // fc2.weight (H2,H1) row-major
    reg signed [DATA_WIDTH -1:0] b2_mem [HIDDEN2-1  :0];   // fc2.bias   (H2)
    reg signed [DATA_WIDTH -1:0] w3_mem [W3_DEPTH-1 :0];   // fc3.weight (1,H2)
    reg signed [DATA_WIDTH -1:0] b3_mem [0:0];             // fc3.bias   (1)

    always_ff @(posedge clk) begin : DFF_WR
        if (i_wr_en) begin
            case (i_wr_sel)
                3'd0: w1_mem[i_wr_addr] <= i_wr_data;
                3'd1: b1_mem[i_wr_addr] <= i_wr_data;
                3'd2: w2_mem[i_wr_addr] <= i_wr_data;
                3'd3: b2_mem[i_wr_addr] <= i_wr_data;
                3'd4: w3_mem[i_wr_addr] <= i_wr_data;
                3'd5: b3_mem[i_wr_addr] <= i_wr_data;
                default: ;
            endcase
        end
    end

    // ---- valid pipeline (resettable; datapath self-flushes via valid) ------
    reg [PIPE -1:0] valid_ff;   // PIPE-stage valid shift register

    always_ff @(posedge clk or negedge rst_n) begin : DFF_VLD
        if (!rst_n) valid_ff <= {PIPE{1'b0}};
        else        valid_ff <= {valid_ff[PIPE-2:0], i_valid};
    end
    assign o_valid = valid_ff[PIPE-1];

    // ---- pipeline registers ------------------------------------------------
    reg        [N_FEATURES -1:0] x_ff;                   // stage 1: latched 1-bit x
    reg signed [ACC1_W     -1:0] acc1_ff [HIDDEN1-1:0];  // stage 2: fc1 (post-ReLU) accumulators
    reg signed [DATA_WIDTH -1:0] h1q_ff  [HIDDEN1-1:0];  // stage 3: req1 int8
    reg signed [ACC2_W     -1:0] acc2_ff [HIDDEN2-1:0];  // stage 4: fc2 (post-ReLU) accumulators
    reg signed [DATA_WIDTH -1:0] h2q_ff  [HIDDEN2-1:0];  // stage 5: req2 int8
    reg signed [ACC3_W     -1:0] acc3_ff;                // stage 6: fc3 accumulator (scalar)
    reg signed [DATA_WIDTH -1:0] y_ff;                   // stage 7: req3 int8 output
    reg        [SHW3       -1:0] shift_ff;               // stage 7: fc3 dynamic shift

    // ---- stage 1: latch x --------------------------------------------------
    always_ff @(posedge clk) begin : DFF_X
        x_ff <= i_x;
    end

    // ---- stage 2: fc1  a1 = relu( b1 + sum_i ( x[i] ? W1[j][i] : 0 ) ) -----
    generate
        for (j = 0; j < HIDDEN1; j = j + 1) begin : G_FC1
            reg signed [ACC1_W -1:0] a1;   // gated-add accumulator + bias

            always_comb begin : CMB_FC1
                a1 = $signed(b1_mem[j]);
                for (m = 0; m < N_FEATURES; m = m + 1)
                    if (x_ff[m]) a1 = a1 + $signed(w1_mem[j*N_FEATURES + m]);
            end
            always_ff @(posedge clk) begin : DFF_A1
                acc1_ff[j] <= (a1 < 0) ? {ACC1_W{1'b0}} : a1;   // ReLU (full precision)
            end
        end
    endgenerate

    // ---- stage 3: req1  h1 = dyn_quant(acc1)  (block-FP, RNE, int8) --------
    wire [HIDDEN1*ACC1_W     -1:0] acc1_pk;    // packed fc1 accumulators
    wire [HIDDEN1*DATA_WIDTH -1:0] h1q_pk;     // packed req1 int8
    wire [SHW1               -1:0] s1_nc;      // req1 shift (not exported)

    generate
        for (j = 0; j < HIDDEN1; j = j + 1) begin : G_PK1
            assign acc1_pk[j*ACC1_W +: ACC1_W] = acc1_ff[j];
        end
    endgenerate
    dyn_quant #(.M(HIDDEN1), .ACC_W(ACC1_W), .DATA_WIDTH(DATA_WIDTH)) U_REQ1 (
        .i_vec   (acc1_pk),
        .o_vec   (h1q_pk),
        .o_shift (s1_nc)
    );
    generate
        for (j = 0; j < HIDDEN1; j = j + 1) begin : G_H1
            always_ff @(posedge clk) begin : DFF_H1
                h1q_ff[j] <= $signed(h1q_pk[j*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endgenerate

    // ---- stage 4: fc2  a2 = relu( b2 + sum_i h1[i]*W2[k][i] ) --------------
    generate
        for (k = 0; k < HIDDEN2; k = k + 1) begin : G_FC2
            reg signed [ACC2_W -1:0] a2;   // int8 matmul accumulator + bias

            always_comb begin : CMB_FC2
                a2 = $signed(b2_mem[k]);
                for (m = 0; m < HIDDEN1; m = m + 1)
                    a2 = a2 + h1q_ff[m] * $signed(w2_mem[k*HIDDEN1 + m]);
            end
            always_ff @(posedge clk) begin : DFF_A2
                acc2_ff[k] <= (a2 < 0) ? {ACC2_W{1'b0}} : a2;   // ReLU
            end
        end
    endgenerate

    // ---- stage 5: req2  h2 = dyn_quant(acc2) -------------------------------
    wire [HIDDEN2*ACC2_W     -1:0] acc2_pk;    // packed fc2 accumulators
    wire [HIDDEN2*DATA_WIDTH -1:0] h2q_pk;     // packed req2 int8
    wire [SHW2               -1:0] s2_nc;      // req2 shift (not exported)

    generate
        for (k = 0; k < HIDDEN2; k = k + 1) begin : G_PK2
            assign acc2_pk[k*ACC2_W +: ACC2_W] = acc2_ff[k];
        end
    endgenerate
    dyn_quant #(.M(HIDDEN2), .ACC_W(ACC2_W), .DATA_WIDTH(DATA_WIDTH)) U_REQ2 (
        .i_vec   (acc2_pk),
        .o_vec   (h2q_pk),
        .o_shift (s2_nc)
    );
    generate
        for (k = 0; k < HIDDEN2; k = k + 1) begin : G_H2
            always_ff @(posedge clk) begin : DFF_H2
                h2q_ff[k] <= $signed(h2q_pk[k*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endgenerate

    // ---- stage 6: fc3  a3 = b3 + sum_i h2[i]*W3[i]  (no activation) --------
    reg signed [ACC3_W -1:0] a3;   // int8 matmul accumulator + bias (scalar)

    always_comb begin : CMB_FC3
        a3 = $signed(b3_mem[0]);
        for (m = 0; m < HIDDEN2; m = m + 1)
            a3 = a3 + h2q_ff[m] * $signed(w3_mem[m]);
    end
    always_ff @(posedge clk) begin : DFF_A3
        acc3_ff <= a3;
    end

    // ---- stage 7: req3  y = dyn_quant(acc3) -> int8 + shift ----------------
    wire [DATA_WIDTH -1:0] y_c;    // req3 int8
    wire [SHW3       -1:0] s3_c;   // req3 shift

    dyn_quant #(.M(1), .ACC_W(ACC3_W), .DATA_WIDTH(DATA_WIDTH)) U_REQ3 (
        .i_vec   (acc3_ff),
        .o_vec   (y_c),
        .o_shift (s3_c)
    );
    always_ff @(posedge clk) begin : DFF_Y
        y_ff     <= $signed(y_c);
        shift_ff <= s3_c;
    end

    assign o_y     = y_ff;
    assign o_shift = shift_ff;

endmodule
`default_nettype wire
