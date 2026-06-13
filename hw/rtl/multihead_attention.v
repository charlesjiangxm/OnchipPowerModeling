// =====================================================================
// multihead_attention.v                                     (Verilog-2005*)
//
// Hardware implementation of the FT-Transformer self-attention block
// (nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads,
// batch_first=True) called as self.attn(x, x, x, ...) in
// src/models/ft_transformer.py; inference, dropout=0, bias=True).
//
// One full (SEQ_LEN x D_TOKEN) int8 example is consumed per clock (packed
// x_seq, with q = k = v = x) and one full (SEQ_LEN x D_TOKEN) int8 result
// is produced per clock after a fixed latency => II = 1. Only the attention
// output is produced (not the attention-weight tensor).
//
// Math (per example x of shape (S, E), E=D_TOKEN, S=SEQ_LEN, H=N_HEADS,
// HD=E/H):
//   in_proj : Q = x@Wq^T+bq, K = x@Wk^T+bk, V = x@Wv^T+bv      (each int8)
//   scores  : per head h, raw[qi][kj] = sum_d Qh[qi][d]*Kh[kj][d]
//   scale   : sm = raw / sqrt(HD)
//   softmax : a[qi][kj] = exp(sm-rowmax) / sum_kj exp(sm-rowmax)
//   context : Ch[qi][d] = sum_kj a[qi][kj] * Vh[kj][d]
//   out     : y = concat_h(Ch) @ Wo^T + bo                     (int8)
//
// Numeric model (int8 inference, symmetric quantization, zero-point 0) -- the
// pure-C twin src/models/multihead_attention_cmodel.c runs the IDENTICAL
// integer datapath, so this RTL matches it bit-for-bit:
//   * int8 values are signed Q1.FRAC_BITS (default Q1.7, scale 2^-7).
//   * each matmul accumulates int8*int8 products exactly, adds bias
//     (int8<<FRAC_BITS), then requantizes to int8: round-half-up, arithmetic
//     right shift, saturate to [-128,127].
//   * the 1/sqrt(HD) scale is the integer constant SCALE = round(2^SCALE_FRAC
//     / sqrt(HD)); the scaled score is requantized to Q(SM_FRAC) (kept wide,
//     not int8) before softmax.
//   * softmax is integer-only: subtract the row max (=> d <= 0), then
//       e = exp(d) via 2^(d*log2e) = 2^-z * 2^-f, with z = floor(.) a shift
//       and 2^-f a fitted quadratic (C2*f^2+C1*f+C0, Q16; endpoint-exact),
//       Se = sum e, inv = round(2^RECIP_FRAC / Se) (one unsigned reciprocal
//       per (head,row), the layer_norm idiom).
//   * context = requant( (sum_kj e[kj]*Vh[kj][d]) * inv ) >> RECIP_FRAC.
//     The EXP_FRAC scale in e and in Se cancel, so the shift is RECIP_FRAC.
//
// Weights live in an FF-based register file (read in parallel each cycle)
// loaded through a write-only port, exactly like numerical_feature_tokenizer
// and layer_norm.
//
// (*) Verilog-2005 dialect using the SystemVerilog procedural keywords
// always_ff / always_comb (compile with `vcs -sverilog`). No `logic` is used
// -- every signal is reg/wire -- and the design uses clk / rst_n.
// =====================================================================

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
    // SCALE = round(2^SCALE_FRAC / sqrt(D_TOKEN/N_HEADS)); default is for HD=4.
    // Override (with SEQ_LEN/N_HEADS) when N_HEADS changes; the TB / DC script
    // recompute it. Pass the SAME value to multihead_attention_int8().
    parameter SCALE      = 8192,
    // ---- derived (do not override) ----
    parameter HD         = D_TOKEN / N_HEADS,
    parameter IPW_DEPTH  = 3 * D_TOKEN * D_TOKEN,
    parameter WSEL_W     = 2,
    parameter WADDR_W    = ($clog2(IPW_DEPTH) < 1) ? 1 : $clog2(IPW_DEPTH)
) (
    input  wire                                  clk,
    input  wire                                  rst_n,      // async assert, sync deassert
    input  wire                                  wr_en,      // coefficient write strobe
    input  wire [WSEL_W-1:0]                     wr_sel,     // 0=ipw 1=ipb 2=opw 3=opb
    input  wire [WADDR_W-1:0]                    wr_addr,    // linear index in selected array
    input  wire [DATA_WIDTH-1:0]                 wr_data,    // signed int8 coefficient
    input  wire                                  in_valid,   // x_seq valid this cycle
    input  wire [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] x_seq,      // x[s][e]=x_seq[(s*E+e)*W +: W]
    output wire                                  out_valid,  // y_seq valid
    output wire [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] y_seq       // y[s][e]=y_seq[(s*E+e)*W +: W]
);

    // ---- derived sizes (sized to hold EXACT values, no truncation) --------
    localparam CLOG2_E  = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam CLOG2_S  = ($clog2(SEQ_LEN) < 1) ? 1 : $clog2(SEQ_LEN);
    localparam CLOG2_HD = ($clog2(HD)      < 1) ? 1 : $clog2(HD);

    // in_proj / out_proj matmul accumulator (sum of E int8*int8 + aligned bias)
    localparam PROJ_ACC_W   = 2*DATA_WIDTH + CLOG2_E + 2;

    // scores + scale
    localparam SC_RAW_W     = 2*DATA_WIDTH + CLOG2_HD + 1;        // signed sum of HD products
    localparam SCALE_W      = SCALE_FRAC + 2;                     // SCALE (positive) width
    localparam SC_SCALED_W  = SC_RAW_W + SCALE_W + 1;            // raw*SCALE
    localparam SC_SHIFT     = 2*FRAC_BITS + SCALE_FRAC - SM_FRAC; // Q(2*FRAC) -> Q(SM_FRAC)
    localparam SM_W         = CLOG2_HD + SM_FRAC + 3;             // signed score, sized > max

    // exp (fixed Q16 constants; MUST match the C model)
    localparam LOG2E_FRAC = 16;
    localparam EXP_FRAC   = 16;
    localparam LOG2E      = 94548;        // round(log2(e)*2^16)
    localparam EXP_C2     = 11279;        // round( 0.172100*2^16)
    localparam EXP_C0     = 65536;        // 2^16
    localparam ZMAX       = EXP_FRAC + 1; // exp -> 0 past here
    localparam SH_EXP     = SM_FRAC + LOG2E_FRAC;
    localparam EXP_W      = EXP_FRAC + 1; // width of e (max 2^16)
    localparam D_W        = SM_W + 1;     // d = sm - rowmax (<= 0)
    localparam U_W        = D_W;          // -d
    localparam M_W        = U_W + 18;     // u*LOG2E (LOG2E < 2^17)
    localparam Z_W        = (M_W > SH_EXP) ? (M_W - SH_EXP) : 1;

    // reciprocal + context
    localparam SE_W       = CLOG2_S + EXP_FRAC + 1;              // sum of S exp values
    localparam INV_W      = RECIP_FRAC + 1;
    localparam RNUM_W     = ((RECIP_FRAC > SE_W) ? RECIP_FRAC : SE_W) + 2;
    localparam CTX_ACC_W  = EXP_W + DATA_WIDTH + CLOG2_S + 1;    // signed sum e*V
    localparam CTX_PROD_W = CTX_ACC_W + INV_W + 1;              // signed (e*V)*inv

    // widest accumulator -> one shared requant() input width
    localparam RQ_W       = CTX_PROD_W + 1;

    // constants
    localparam signed [17:0]      EXP_C1    = -44047;            // round(-0.672100*2^16)
    localparam signed [SCALE_W-1:0] SCALE_S = SCALE;            // SCALE as signed positive
    localparam        [RECIP_FRAC:0] RECIP_ONE = (1 <<< RECIP_FRAC);
    localparam        [RQ_W-1:0]  RQ_ONE    = {{(RQ_W-1){1'b0}}, 1'b1};
    localparam signed [SC_SCALED_W-1:0] SC_RND =
                 (SC_SHIFT > 0) ? (1 <<< (SC_SHIFT-1)) : 0;
    localparam signed [RQ_W-1:0]  OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;   // +127
    localparam signed [RQ_W-1:0]  OUT_MIN = -(1 <<< (DATA_WIDTH-1));      // -128

    // ---- helper functions -------------------------------------------------
    // sign-extend an int8 bias to PROJ_ACC_W, then <<FRAC_BITS to align with
    // the Q(2*FRAC) product accumulator (like layer_norm's align_beta).
    function signed [PROJ_ACC_W-1:0] align_bias;
        input signed [DATA_WIDTH-1:0] b;
        reg   signed [PROJ_ACC_W-1:0] be;
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

    // scaled-score requant: round-half-up + arithmetic shift, NO int8 saturate
    // (SM_W is sized to hold the result exactly).
    function signed [SM_W-1:0] score_shift;
        input signed [SC_SCALED_W-1:0] v;
        reg   signed [SC_SCALED_W-1:0] s;
        begin
            s           = v + SC_RND;
            score_shift = s >>> SC_SHIFT;
        end
    endfunction

    // e = exp(d), d <= 0 in Q(SM_FRAC); returns Q(EXP_FRAC=16), >= 0.
    // 2^(d*log2e) = 2^-z * 2^-f ; 2^-f ~= C2*f^2 + C1*f + C0 (Q16, endpoint-exact).
    function [EXP_W-1:0] exp_neg;
        input signed [D_W-1:0]  d;
        reg        [U_W-1:0]    u;
        reg        [M_W-1:0]    m;
        reg        [Z_W-1:0]    z;
        reg        [EXP_FRAC-1:0] f;
        reg        [63:0]       t2_full;
        reg        [31:0]       t2;
        reg signed [47:0]       t1_full;
        reg signed [31:0]       t1;
        reg signed [31:0]       p;
        begin
            if (d < 0) u = -d;          // d <= 0 by construction; d==0 -> u=0
            else       u = {U_W{1'b0}};
            m  = u * LOG2E;             // >= 0, exact
            z  = m[M_W-1 : SH_EXP];     // m >> SH_EXP
            f  = m[SH_EXP-1 -: EXP_FRAC];                 // (m mod 2^SH) >> (SH-EXP_FRAC)
            t2_full = EXP_C2 * f * f;                      // unsigned
            t2 = t2_full >> (2*EXP_FRAC);                  // C2*f^2 in Q16
            t1_full = EXP_C1 * $signed({1'b0, f});         // signed, C1<0
            t1 = t1_full >>> EXP_FRAC;                      // arithmetic
            p  = $signed({1'b0, t2}) + t1 + EXP_C0;        // 2^-f, Q16, in (0.5,1]
            if (z >= ZMAX) exp_neg = {EXP_W{1'b0}};
            else           exp_neg = p[EXP_W-1:0] >> z;    // p >= 0
        end
    endfunction

    // ---- coefficient register file (FF-based, write-only port) ------------
    reg signed [DATA_WIDTH-1:0] ipw_mem [0:3*D_TOKEN*D_TOKEN-1];  // in_proj_weight (3E,E)
    reg signed [DATA_WIDTH-1:0] ipb_mem [0:3*D_TOKEN-1];          // in_proj_bias   (3E)
    reg signed [DATA_WIDTH-1:0] opw_mem [0:D_TOKEN*D_TOKEN-1];    // out_proj.weight(E,E)
    reg signed [DATA_WIDTH-1:0] opb_mem [0:D_TOKEN-1];            // out_proj.bias  (E)

    always_ff @(posedge clk) begin
        if (wr_en) begin
            case (wr_sel)
                2'd0: ipw_mem[wr_addr] <= wr_data;
                2'd1: ipb_mem[wr_addr] <= wr_data;
                2'd2: opw_mem[wr_addr] <= wr_data;
                2'd3: opb_mem[wr_addr] <= wr_data;
                default: ;
            endcase
        end
    end

    // ---- valid pipeline (6 stages; datapath self-flushes via valid) -------
    reg v1, v2, v3, v4, v5, v6;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; v6 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; v6 <= v5;
        end
    end
    assign out_valid = v6;

    // ---- pipeline registers -----------------------------------------------
    reg signed [DATA_WIDTH-1:0] xq  [0:SEQ_LEN-1][0:D_TOKEN-1];  // A: latched x
    reg signed [DATA_WIDTH-1:0] Qr  [0:SEQ_LEN-1][0:D_TOKEN-1];  // B: in_proj Q
    reg signed [DATA_WIDTH-1:0] Kr  [0:SEQ_LEN-1][0:D_TOKEN-1];  // B: in_proj K
    reg signed [DATA_WIDTH-1:0] Vr  [0:SEQ_LEN-1][0:D_TOKEN-1];  // B: in_proj V
    reg signed [DATA_WIDTH-1:0] Vpc [0:SEQ_LEN-1][0:D_TOKEN-1];  // C: V aligned to scores
    reg signed [DATA_WIDTH-1:0] Vpd [0:SEQ_LEN-1][0:D_TOKEN-1];  // D: V aligned to exp/inv
    reg signed [SM_W-1:0]       smr [0:N_HEADS-1][0:SEQ_LEN-1][0:SEQ_LEN-1]; // C
    reg        [EXP_W-1:0]      er  [0:N_HEADS-1][0:SEQ_LEN-1][0:SEQ_LEN-1]; // D
    reg        [INV_W-1:0]      invr[0:N_HEADS-1][0:SEQ_LEN-1];              // D
    reg signed [DATA_WIDTH-1:0] ctxr[0:SEQ_LEN-1][0:D_TOKEN-1];  // E: concat-of-heads context
    reg signed [DATA_WIDTH-1:0] yq  [0:SEQ_LEN-1][0:D_TOKEN-1];  // F: out_proj

    genvar gs, ge, gh, gq, gk, gd;

    // ---- stage A: unpack x_seq -> xq --------------------------------------
    generate
        for (gs = 0; gs < SEQ_LEN; gs = gs + 1) begin : gA_s
            for (ge = 0; ge < D_TOKEN; ge = ge + 1) begin : gA_e
                always_ff @(posedge clk)
                    xq[gs][ge] <= $signed(x_seq[(gs*D_TOKEN+ge)*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endgenerate

    // ---- stage B: in_proj  Q/K/V = requant( x.W + (b<<FRAC) ) -------------
    generate
        for (gs = 0; gs < SEQ_LEN; gs = gs + 1) begin : gB_s
            for (ge = 0; ge < D_TOKEN; ge = ge + 1) begin : gB_e
                reg signed [PROJ_ACC_W-1:0] aq, ak, av;
                integer kk;
                always_comb begin
                    aq = {PROJ_ACC_W{1'b0}};
                    ak = {PROJ_ACC_W{1'b0}};
                    av = {PROJ_ACC_W{1'b0}};
                    for (kk = 0; kk < D_TOKEN; kk = kk + 1) begin
                        aq = aq + xq[gs][kk] * ipw_mem[ge*D_TOKEN + kk];
                        ak = ak + xq[gs][kk] * ipw_mem[(D_TOKEN+ge)*D_TOKEN + kk];
                        av = av + xq[gs][kk] * ipw_mem[(2*D_TOKEN+ge)*D_TOKEN + kk];
                    end
                    aq = aq + align_bias(ipb_mem[ge]);
                    ak = ak + align_bias(ipb_mem[D_TOKEN+ge]);
                    av = av + align_bias(ipb_mem[2*D_TOKEN+ge]);
                end
                always_ff @(posedge clk) begin
                    Qr [gs][ge] <= requant(aq, FRAC_BITS);
                    Kr [gs][ge] <= requant(ak, FRAC_BITS);
                    Vr [gs][ge] <= requant(av, FRAC_BITS);
                    Vpc[gs][ge] <= Vr[gs][ge];     // pipeline V to the score stage
                    Vpd[gs][ge] <= Vpc[gs][ge];    // pipeline V to the exp/inv stage
                end
            end
        end
    endgenerate

    // ---- stage C: per-head scores Qh.Kh, scaled to Q(SM_FRAC) -------------
    generate
        for (gh = 0; gh < N_HEADS; gh = gh + 1) begin : gC_h
            for (gq = 0; gq < SEQ_LEN; gq = gq + 1) begin : gC_q
                for (gk = 0; gk < SEQ_LEN; gk = gk + 1) begin : gC_k
                    reg signed [SC_RAW_W-1:0]    raw;
                    reg signed [SC_SCALED_W-1:0] scaled;
                    integer dd;
                    always_comb begin
                        raw = {SC_RAW_W{1'b0}};
                        for (dd = 0; dd < HD; dd = dd + 1)
                            raw = raw + Qr[gq][gh*HD+dd] * Kr[gk][gh*HD+dd];
                        scaled = raw * SCALE_S;
                    end
                    always_ff @(posedge clk)
                        smr[gh][gq][gk] <= score_shift(scaled);
                end
            end
        end
    endgenerate

    // ---- stage D: row max -> exp -> sum -> reciprocal ---------------------
    generate
        for (gh = 0; gh < N_HEADS; gh = gh + 1) begin : gD_h
            for (gq = 0; gq < SEQ_LEN; gq = gq + 1) begin : gD_q
                reg signed [SM_W-1:0] mx;
                reg signed [D_W-1:0]  dsub;
                reg        [EXP_W-1:0] e_tmp [0:SEQ_LEN-1];
                reg        [SE_W-1:0]  se;
                reg        [RNUM_W-1:0] inum;
                reg        [INV_W-1:0] iv;
                integer kj;
                always_comb begin
                    mx = smr[gh][gq][0];
                    for (kj = 1; kj < SEQ_LEN; kj = kj + 1)
                        if (smr[gh][gq][kj] > mx) mx = smr[gh][gq][kj];
                    se = {SE_W{1'b0}};
                    for (kj = 0; kj < SEQ_LEN; kj = kj + 1) begin
                        dsub      = smr[gh][gq][kj] - mx;     // computed in D_W (LHS) context
                        e_tmp[kj] = exp_neg(dsub);
                        se        = se + e_tmp[kj];
                    end
                    if (se == {SE_W{1'b0}}) se = {{(SE_W-1){1'b0}}, 1'b1};  // clamp >=1
                    inum = RECIP_ONE + (se >> 1);
                    iv   = inum / se;                          // round(2^RECIP_FRAC / Se)
                end
                always_ff @(posedge clk) begin
                    for (kj = 0; kj < SEQ_LEN; kj = kj + 1)
                        er[gh][gq][kj] <= e_tmp[kj];
                    invr[gh][gq] <= iv;
                end
            end
        end
    endgenerate

    // ---- stage E: context = requant( (sum_kj e*Vh) * inv ) >> RECIP_FRAC --
    generate
        for (gh = 0; gh < N_HEADS; gh = gh + 1) begin : gE_h
            for (gq = 0; gq < SEQ_LEN; gq = gq + 1) begin : gE_q
                for (gd = 0; gd < HD; gd = gd + 1) begin : gE_d
                    reg signed [CTX_ACC_W-1:0]  cacc;
                    reg signed [CTX_PROD_W-1:0] cprod;
                    integer kj;
                    always_comb begin
                        cacc = {CTX_ACC_W{1'b0}};
                        for (kj = 0; kj < SEQ_LEN; kj = kj + 1)
                            cacc = cacc + $signed({1'b0, er[gh][gq][kj]})
                                        * Vpd[kj][gh*HD+gd];
                        cprod = cacc * $signed({1'b0, invr[gh][gq]});
                    end
                    always_ff @(posedge clk)
                        ctxr[gq][gh*HD+gd] <= requant(cprod, RECIP_FRAC);
                end
            end
        end
    endgenerate

    // ---- stage F: out_proj  y = requant( concat.Wo + (bo<<FRAC) ) ---------
    generate
        for (gs = 0; gs < SEQ_LEN; gs = gs + 1) begin : gF_s
            for (ge = 0; ge < D_TOKEN; ge = ge + 1) begin : gF_e
                reg signed [PROJ_ACC_W-1:0] ao;
                integer kk;
                always_comb begin
                    ao = {PROJ_ACC_W{1'b0}};
                    for (kk = 0; kk < D_TOKEN; kk = kk + 1)
                        ao = ao + ctxr[gs][kk] * opw_mem[ge*D_TOKEN + kk];
                    ao = ao + align_bias(opb_mem[ge]);
                end
                always_ff @(posedge clk)
                    yq[gs][ge] <= requant(ao, FRAC_BITS);
                assign y_seq[(gs*D_TOKEN+ge)*DATA_WIDTH +: DATA_WIDTH] = yq[gs][ge];
            end
        end
    endgenerate

endmodule
`default_nettype wire
