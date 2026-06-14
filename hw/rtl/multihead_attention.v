//-----------------------------------------------------------------------------
// multihead_attention.v                                       (Verilog-2005*)
//
// title    : FT-Transformer self-attention block (int8 inference)
//            (nn.MultiheadAttention(d_token, n_heads, batch_first), q=k=v=x)
// math     : in_proj : Q=x@Wq^T+bq, K=x@Wk^T+bk, V=x@Wv^T+bv   (per head split)
//            scores  : raw[qi][kj] = sum_d Qh[qi][d]*Kh[kj][d]; sm = raw/sqrt(HD)
//            softmax : a = exp(sm-rowmax) / sum exp(sm-rowmax)  (integer-only)
//            context : Ch[qi][d] = sum_kj a[qi][kj]*Vh[kj][d]
//            out     : y = concat_h(Ch) @ Wo^T + bo
// numeric  : int8 = signed Q1.FRAC_BITS (zero-point 0). matmuls requantize to
//            int8; scale = round(2^SCALE_FRAC/sqrt(HD)), score kept at Q(SM_FRAC);
//            exp(d<=0) = 2^-z * (Q16 quadratic); inv = round(2^RECIP_FRAC/Se);
//            context = requant( (sum e*Vh) * inv ) >> RECIP_FRAC. Matches
//            src/models/multihead_attention_cmodel.c bit-for-bit.
// datapath : FF weight regfile -> in_proj -> per-head scores+scale -> integer
//            softmax (U_EXP per key) -> context -> out_proj. Fully parallel.
// schedule : one (S,E) example in / one out per clock, latency 6, II = 1.
// params   : D_TOKEN(E), N_HEADS(H), SEQ_LEN(S), DATA_WIDTH, FRAC_BITS,
//            SCALE_FRAC, SM_FRAC, RECIP_FRAC, SCALE.
// language : Verilog-2005 + SystemVerilog always_ff/always_comb (no logic);
//            compile with `vcs -sverilog`. clk / rst_n (async assert, sync deassert).
//-----------------------------------------------------------------------------
`default_nettype none
module multihead_attention #(
    parameter D_TOKEN    = 32,   // E: embedding dim
    parameter N_HEADS    = 8,    // H: number of heads
    parameter SEQ_LEN    = 16,   // S: sequence length (= 1 + n_features)
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    parameter SCALE_FRAC = 14,   // fractional bits of SCALE
    parameter SM_FRAC    = 8,    // fractional bits of the softmax-input score
    parameter RECIP_FRAC = 24,   // reciprocal (1/Se) fractional bits
    parameter SCALE      = 8192, // round(2^SCALE_FRAC / sqrt(D_TOKEN/N_HEADS)); HD=4
    // ---- derived (do not override) ----
    parameter HD         = D_TOKEN / N_HEADS,                            // per-head dim
    parameter IPW_DEPTH  = 3 * D_TOKEN * D_TOKEN,                        // in_proj_weight depth
    parameter WSEL_W     = 2,                                            // wr_sel width
    parameter WADDR_W    = ($clog2(IPW_DEPTH) < 1) ? 1 : $clog2(IPW_DEPTH) // write-address width
) (
    // control port
    input  wire                                  clk,        // clock
    input  wire                                  rst_n,      // async assert, sync deassert
    input  wire                                  i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]                   i_wr_sel,   // 0=ipw 1=ipb 2=opw 3=opb
    input  wire [WADDR_W -1:0]                   i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]                i_wr_data,  // signed int8 coefficient
    input  wire                                  i_valid,    // x_seq valid this cycle
    // data port
    input  wire [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] i_x_seq,   // x[s][e]=i_x_seq[(s*E+e)*W +: W]
    output wire                                  o_valid,    // y_seq valid
    output wire [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] o_y_seq    // y[s][e]=o_y_seq[(s*E+e)*W +: W]
);

    integer n;                       // procedural matmul / reduction index
    genvar  s, e, h, q, k, d;        // seq / embed / head / query / key / head-dim indices

    localparam PIPE     = 6;                            // valid-pipeline / latency depth
    localparam CLOG2_E  = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam CLOG2_S  = ($clog2(SEQ_LEN) < 1) ? 1 : $clog2(SEQ_LEN);
    localparam CLOG2_HD = ($clog2(HD)      < 1) ? 1 : $clog2(HD);

    // in_proj / out_proj matmul accumulator (sum of E int8*int8 + aligned bias)
    localparam PROJ_ACC_W  = 2*DATA_WIDTH + CLOG2_E + 2;

    // scores + scale
    localparam SC_RAW_W    = 2*DATA_WIDTH + CLOG2_HD + 1;        // signed sum of HD products
    localparam SCALE_W     = SCALE_FRAC + 2;                     // SCALE (positive) width
    localparam SC_SCALED_W = SC_RAW_W + SCALE_W + 1;            // raw*SCALE
    localparam SC_SHIFT    = 2*FRAC_BITS + SCALE_FRAC - SM_FRAC; // Q(2*FRAC) -> Q(SM_FRAC)
    localparam SM_W        = CLOG2_HD + SM_FRAC + 3;             // signed score, sized > max

    // exp / softmax
    localparam EXP_FRAC    = 16;          // exp output Q(EXP_FRAC)  (matches U_EXP)
    localparam EXP_W       = EXP_FRAC + 1; // width of e (max 2^16)
    localparam D_W         = SM_W + 1;     // d = sm - rowmax (<= 0)

    // reciprocal + context
    localparam SE_W        = CLOG2_S + EXP_FRAC + 1;            // sum of S exp values
    localparam INV_W       = RECIP_FRAC + 1;
    localparam RNUM_W      = ((RECIP_FRAC > SE_W) ? RECIP_FRAC : SE_W) + 2;
    localparam CTX_ACC_W   = EXP_W + DATA_WIDTH + CLOG2_S + 1;  // signed sum e*V
    localparam CTX_PROD_W  = CTX_ACC_W + INV_W + 1;            // signed (e*V)*inv

    localparam signed [SCALE_W -1:0] SCALE_S   = SCALE;        // SCALE as signed positive
    localparam        [RECIP_FRAC:0] RECIP_ONE = (1 <<< RECIP_FRAC);

    // ---- coefficient register file (FF-based, write-only port) --------------
    reg signed [DATA_WIDTH -1:0] ipw_mem [3*D_TOKEN*D_TOKEN-1 :0];  // in_proj_weight (3E,E)
    reg signed [DATA_WIDTH -1:0] ipb_mem [3*D_TOKEN-1         :0];  // in_proj_bias   (3E)
    reg signed [DATA_WIDTH -1:0] opw_mem [D_TOKEN*D_TOKEN-1   :0];  // out_proj.weight(E,E)
    reg signed [DATA_WIDTH -1:0] opb_mem [D_TOKEN-1           :0];  // out_proj.bias  (E)

    always_ff @(posedge clk) begin : DFF_WR
        if (i_wr_en) begin
            case (i_wr_sel)
                2'd0: ipw_mem[i_wr_addr] <= i_wr_data;
                2'd1: ipb_mem[i_wr_addr] <= i_wr_data;
                2'd2: opw_mem[i_wr_addr] <= i_wr_data;
                2'd3: opb_mem[i_wr_addr] <= i_wr_data;
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
    reg signed [DATA_WIDTH -1:0] xq_ff  [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // A: latched x
    reg signed [DATA_WIDTH -1:0] Qr_ff  [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // B: in_proj Q
    reg signed [DATA_WIDTH -1:0] Kr_ff  [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // B: in_proj K
    reg signed [DATA_WIDTH -1:0] Vr_ff  [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // B: in_proj V
    reg signed [DATA_WIDTH -1:0] Vpc_ff [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // C: V aligned to scores
    reg signed [DATA_WIDTH -1:0] Vpd_ff [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // D: V aligned to exp/inv
    reg signed [SM_W       -1:0] smr_ff [N_HEADS-1 :0][SEQ_LEN-1 :0][SEQ_LEN-1 :0]; // C: scores
    reg        [EXP_W      -1:0] er_ff  [N_HEADS-1 :0][SEQ_LEN-1 :0][SEQ_LEN-1 :0]; // D: exp
    reg        [INV_W      -1:0] invr_ff[N_HEADS-1 :0][SEQ_LEN-1 :0];               // D: 1/Se
    reg signed [DATA_WIDTH -1:0] ctxr_ff[SEQ_LEN-1 :0][D_TOKEN-1 :0];  // E: concat-of-heads ctx
    reg signed [DATA_WIDTH -1:0] yq_ff  [SEQ_LEN-1 :0][D_TOKEN-1 :0];  // F: out_proj

    // ---- stage A: unpack x_seq -> xq_ff -------------------------------------
    generate
        for (s = 0; s < SEQ_LEN; s = s + 1) begin : G_A_S
            for (e = 0; e < D_TOKEN; e = e + 1) begin : G_A_E
                always_ff @(posedge clk) begin : DFF_A
                    xq_ff[s][e] <= $signed(i_x_seq[(s*D_TOKEN+e)*DATA_WIDTH +: DATA_WIDTH]);
                end
            end
        end
    endgenerate

    // ---- stage B: in_proj  Q/K/V = requant( x.W + (b<<FRAC) ) ----------------
    generate
        for (s = 0; s < SEQ_LEN; s = s + 1) begin : G_B_S
            for (e = 0; e < D_TOKEN; e = e + 1) begin : G_B_E
                reg  signed [PROJ_ACC_W -1:0] aq, ak, av;        // matmul accumulators
                wire signed [PROJ_ACC_W -1:0] bq_al, bk_al, bv_al; // aligned biases
                wire signed [PROJ_ACC_W -1:0] acq, ack, acv;     // matmul + bias
                wire signed [DATA_WIDTH -1:0] q_c, k_c, v_c;     // requantized Q/K/V

                always_comb begin : CMB_IPROJ
                    aq = {PROJ_ACC_W{1'b0}};
                    ak = {PROJ_ACC_W{1'b0}};
                    av = {PROJ_ACC_W{1'b0}};
                    for (n = 0; n < D_TOKEN; n = n + 1) begin
                        aq = aq + xq_ff[s][n] * ipw_mem[e*D_TOKEN + n];
                        ak = ak + xq_ff[s][n] * ipw_mem[(D_TOKEN+e)*D_TOKEN + n];
                        av = av + xq_ff[s][n] * ipw_mem[(2*D_TOKEN+e)*D_TOKEN + n];
                    end
                end
                align_bias #(.IN_W(DATA_WIDTH), .OUT_W(PROJ_ACC_W), .SH(FRAC_BITS))
                    U_ALIGN_BQ (.i_b(ipb_mem[e]),             .o_aligned(bq_al));
                align_bias #(.IN_W(DATA_WIDTH), .OUT_W(PROJ_ACC_W), .SH(FRAC_BITS))
                    U_ALIGN_BK (.i_b(ipb_mem[D_TOKEN+e]),     .o_aligned(bk_al));
                align_bias #(.IN_W(DATA_WIDTH), .OUT_W(PROJ_ACC_W), .SH(FRAC_BITS))
                    U_ALIGN_BV (.i_b(ipb_mem[2*D_TOKEN+e]),   .o_aligned(bv_al));
                assign acq = aq + bq_al;
                assign ack = ak + bk_al;
                assign acv = av + bv_al;
                requant #(.ACC_W(PROJ_ACC_W), .DATA_WIDTH(DATA_WIDTH), .SHIFT(FRAC_BITS))
                    U_REQ_Q (.i_acc(acq), .o_q(q_c));
                requant #(.ACC_W(PROJ_ACC_W), .DATA_WIDTH(DATA_WIDTH), .SHIFT(FRAC_BITS))
                    U_REQ_K (.i_acc(ack), .o_q(k_c));
                requant #(.ACC_W(PROJ_ACC_W), .DATA_WIDTH(DATA_WIDTH), .SHIFT(FRAC_BITS))
                    U_REQ_V (.i_acc(acv), .o_q(v_c));

                always_ff @(posedge clk) begin : DFF_B
                    Qr_ff [s][e] <= q_c;
                    Kr_ff [s][e] <= k_c;
                    Vr_ff [s][e] <= v_c;
                    Vpc_ff[s][e] <= Vr_ff[s][e];    // pipeline V to the score stage
                    Vpd_ff[s][e] <= Vpc_ff[s][e];   // pipeline V to the exp/inv stage
                end
            end
        end
    endgenerate

    // ---- stage C: per-head scores Qh.Kh, scaled to Q(SM_FRAC) ---------------
    generate
        for (h = 0; h < N_HEADS; h = h + 1) begin : G_C_H
            for (q = 0; q < SEQ_LEN; q = q + 1) begin : G_C_Q
                for (k = 0; k < SEQ_LEN; k = k + 1) begin : G_C_K
                    reg  signed [SC_RAW_W    -1:0] raw;     // sum of HD products
                    wire signed [SC_SCALED_W -1:0] scaled;  // raw * SCALE
                    wire signed [SM_W        -1:0] sm_c;    // shifted score

                    always_comb begin : CMB_SCORE
                        raw = {SC_RAW_W{1'b0}};
                        for (n = 0; n < HD; n = n + 1)
                            raw = raw + Qr_ff[q][h*HD+n] * Kr_ff[k][h*HD+n];
                    end
                    assign scaled = raw * SCALE_S;
                    score_shift #(
                        .IN_W  (SC_SCALED_W),
                        .OUT_W (SM_W),
                        .SH    (SC_SHIFT)
                    ) U_SCORE_SHIFT (
                        .i_v (scaled),
                        .o_s (sm_c)
                    );
                    always_ff @(posedge clk) begin : DFF_C
                        smr_ff[h][q][k] <= sm_c;
                    end
                end
            end
        end
    endgenerate

    // ---- stage D: row max -> exp -> sum -> reciprocal -----------------------
    generate
        for (h = 0; h < N_HEADS; h = h + 1) begin : G_D_H
            for (q = 0; q < SEQ_LEN; q = q + 1) begin : G_D_Q
                reg  signed [SM_W   -1:0] mx;            // row max
                reg         [SE_W   -1:0] se;            // sum of exp
                reg         [RNUM_W -1:0] inum;          // reciprocal numerator
                reg         [INV_W  -1:0] iv;            // reciprocal 1/Se
                wire signed [D_W    -1:0] dsub [SEQ_LEN-1 :0];  // score - rowmax per key
                wire        [EXP_W  -1:0] e_w  [SEQ_LEN-1 :0];  // exp(dsub) per key

                // row max over the S key scores
                always_comb begin : CMB_MAX
                    mx = smr_ff[h][q][0];
                    for (n = 1; n < SEQ_LEN; n = n + 1)
                        if (smr_ff[h][q][n] > mx) mx = smr_ff[h][q][n];
                end

                // per key: d = score - rowmax (<= 0), then e = exp(d)
                for (k = 0; k < SEQ_LEN; k = k + 1) begin : G_D_K
                    assign dsub[k] = smr_ff[h][q][k] - mx;   // computed in D_W context
                    exp_neg #(
                        .D_W     (D_W),
                        .SM_FRAC (SM_FRAC)
                    ) U_EXP (
                        .i_d (dsub[k]),
                        .o_e (e_w[k])
                    );
                end

                // sum of exp, clamp >= 1, reciprocal = round(2^RECIP_FRAC / Se)
                always_comb begin : CMB_INV
                    se = {SE_W{1'b0}};
                    for (n = 0; n < SEQ_LEN; n = n + 1)
                        se = se + e_w[n];
                    if (se == {SE_W{1'b0}}) se = {{(SE_W-1){1'b0}}, 1'b1};  // clamp >= 1
                    inum = RECIP_ONE + (se >> 1);
                    iv   = inum / se;
                end

                always_ff @(posedge clk) begin : DFF_D
                    for (n = 0; n < SEQ_LEN; n = n + 1)
                        er_ff[h][q][n] <= e_w[n];
                    invr_ff[h][q] <= iv;
                end
            end
        end
    endgenerate

    // ---- stage E: context = requant( (sum_kj e*Vh) * inv ) >> RECIP_FRAC ----
    generate
        for (h = 0; h < N_HEADS; h = h + 1) begin : G_E_H
            for (q = 0; q < SEQ_LEN; q = q + 1) begin : G_E_Q
                for (d = 0; d < HD; d = d + 1) begin : G_E_D
                    reg  signed [CTX_ACC_W  -1:0] cacc;   // sum e*V
                    wire signed [CTX_PROD_W -1:0] cprod;  // cacc * inv
                    wire signed [DATA_WIDTH -1:0] ctx_c;  // requantized context

                    always_comb begin : CMB_CTX
                        cacc = {CTX_ACC_W{1'b0}};
                        for (n = 0; n < SEQ_LEN; n = n + 1)
                            cacc = cacc + $signed({1'b0, er_ff[h][q][n]})
                                        * Vpd_ff[n][h*HD+d];
                    end
                    assign cprod = cacc * $signed({1'b0, invr_ff[h][q]});
                    requant #(
                        .ACC_W      (CTX_PROD_W),
                        .DATA_WIDTH (DATA_WIDTH),
                        .SHIFT      (RECIP_FRAC)
                    ) U_REQ_CTX (
                        .i_acc (cprod),
                        .o_q   (ctx_c)
                    );
                    always_ff @(posedge clk) begin : DFF_E
                        ctxr_ff[q][h*HD+d] <= ctx_c;
                    end
                end
            end
        end
    endgenerate

    // ---- stage F: out_proj  y = requant( concat.Wo + (bo<<FRAC) ) -----------
    generate
        for (s = 0; s < SEQ_LEN; s = s + 1) begin : G_F_S
            for (e = 0; e < D_TOKEN; e = e + 1) begin : G_F_E
                reg  signed [PROJ_ACC_W -1:0] ao;     // matmul accumulator
                wire signed [PROJ_ACC_W -1:0] bo_al;  // aligned bias (bo << FRAC)
                wire signed [PROJ_ACC_W -1:0] aco;    // matmul + bias
                wire signed [DATA_WIDTH -1:0] y_c;    // requantized output

                always_comb begin : CMB_OPROJ
                    ao = {PROJ_ACC_W{1'b0}};
                    for (n = 0; n < D_TOKEN; n = n + 1)
                        ao = ao + ctxr_ff[s][n] * opw_mem[e*D_TOKEN + n];
                end
                align_bias #(.IN_W(DATA_WIDTH), .OUT_W(PROJ_ACC_W), .SH(FRAC_BITS))
                    U_ALIGN_BO (.i_b(opb_mem[e]), .o_aligned(bo_al));
                assign aco = ao + bo_al;
                requant #(.ACC_W(PROJ_ACC_W), .DATA_WIDTH(DATA_WIDTH), .SHIFT(FRAC_BITS))
                    U_REQ_O (.i_acc(aco), .o_q(y_c));

                always_ff @(posedge clk) begin : DFF_F
                    yq_ff[s][e] <= y_c;
                end
                assign o_y_seq[(s*D_TOKEN+e)*DATA_WIDTH +: DATA_WIDTH] = yq_ff[s][e];
            end
        end
    endgenerate

endmodule
`default_nettype wire
