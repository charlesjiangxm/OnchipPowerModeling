// =====================================================================
// layer_norm.v                                              (Verilog-2005*)
//
// Hardware implementation of the FT-Transformer LayerNorm
// (nn.LayerNorm(d_token) in src/models/ft_transformer.py; norm1/norm2/
// final_norm). For one token x of D_TOKEN int8 elements with learned int8
// affine gamma/beta (elementwise_affine=True), this computes, over the
// D_TOKEN axis (the same math nn.LayerNorm does, eps default 1e-5):
//
//     mean = (1/D) * sum(x)
//     var  = (1/D) * sum((x-mean)^2)            // biased / population
//     y[i] = (x[i]-mean)/sqrt(var+eps) * gamma[i] + beta[i]
//
// Unlike the tokenizer this needs a reduction (mean/var) over the whole
// token, so one full token is consumed per clock (packed x_vec) and one
// full token is produced per clock after a fixed latency => II = 1.
//
// Numeric model (int8 inference, symmetric quantization, zero-point = 0):
//   * int8 values are signed Q1.FRAC_BITS (default Q1.7, scale 2^-7).
//   * the 2^-FRAC input scale and the 1/D mean factor cancel exactly, so
//     in pure integers:  y_norm[i] = (D*x[i] - S) / sqrt(V + EPS_V),
//     where  S = sum(x),  V = D*sum(x^2) - S^2  (= D^2 * var_int, >= 0),
//     EPS_V = round(eps * 2^(2*FRAC) * D^2)  (default 168 for D=32).
//   * 1/sqrt is done as  r = floor(sqrt(V+EPS_V))  (integer sqrt, clamp >=1)
//     then  inv = round(2^RECIP_FRAC / r)  (one unsigned reciprocal/token).
//   * per lane:  acc = (D*x[i]-S)*inv * gamma[i] + (beta[i] << RECIP_FRAC),
//     then requantize to int8: round-half-up, arithmetic right shift by
//     SHIFT = FRAC_BITS + RECIP_FRAC - OUT_FRAC, saturate to int8 range.
//
// gamma/beta live in an FF-based register file (read in parallel each cycle)
// loaded through a write-only port, exactly like numerical_feature_tokenizer.
//
// (*) Verilog-2005 dialect, but uses the SystemVerilog procedural keywords
// always_ff / always_comb (compile with `vcs -sverilog`). No `logic` is used
// -- every signal is `reg`/`wire` -- and the design uses clk / rst_n.
// hw/model: behavioral twin in src/models/layer_norm_cmodel.c (bit-exact).
// =====================================================================

`default_nettype none
module layer_norm #(
    parameter D_TOKEN    = 32,   // token dimension (length of the normalized axis)
    parameter DATA_WIDTH = 8,    // int8 element width (x, gamma, beta, y)
    parameter FRAC_BITS  = 7,    // input/affine fractional bits => Q1.7
    parameter RECIP_FRAC = 24,   // reciprocal fractional bits (1/std precision)
    parameter OUT_FRAC   = 7,    // output fractional bits (=FRAC_BITS => strict Q1.7)
    parameter EPS_V      = 168,  // integer epsilon = round(eps*2^(2*FRAC)*D^2), eps=1e-5
    // ---- derived (do not override) ----
    parameter ADDR_W     = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN)
) (
    input  wire                              clk,
    input  wire                              rst_n,       // async assert, sync deassert
    input  wire                              wr_en,       // coefficient write strobe
    input  wire                              wr_is_beta,  // 0=gamma, 1=beta
    input  wire [ADDR_W-1:0]                 wr_addr,     // lane index i
    input  wire [DATA_WIDTH-1:0]             wr_data,     // signed int8 coefficient
    input  wire                              in_valid,    // x_vec valid this cycle
    input  wire [D_TOKEN*DATA_WIDTH-1:0]     x_vec,       // packed: x[i] = x_vec[i*W +: W]
    output wire                              out_valid,   // y_vec valid
    output wire [D_TOKEN*DATA_WIDTH-1:0]     y_vec        // packed: y[i] = y_vec[i*W +: W]
);

    // ---- derived sizes (sized to hold EXACT values, no truncation) ----
    localparam CLOG2_D  = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam S_W      = DATA_WIDTH + CLOG2_D + 1;        // signed sum S = sum(x)
    localparam SS_W     = 2*DATA_WIDTH + CLOG2_D;          // unsigned sum of squares
    localparam VEPS_W   = 2*DATA_WIDTH + 2*CLOG2_D;        // unsigned V+EPS_V (>= D^2*var bits +1)
    localparam R_W      = DATA_WIDTH + CLOG2_D;            // unsigned r = floor(sqrt(Veps))
    localparam INV_W    = RECIP_FRAC + 1;                  // unsigned inv (max 2^RECIP_FRAC at r=1)
    localparam NUM_W    = DATA_WIDTH + CLOG2_D + 2;        // signed num = D*x[i] - S
    localparam ZNORM_W  = NUM_W + INV_W + 1;               // signed num*inv
    localparam ACC_W    = ZNORM_W + DATA_WIDTH + 2;        // signed znorm*gamma + (beta<<RECIP_FRAC)
    localparam SHIFT    = FRAC_BITS + RECIP_FRAC - OUT_FRAC; // requant right-shift amount

    localparam ISQRT_ITERS = (VEPS_W + 1) / 2;            // fixed (synthesizable) sqrt iterations
    localparam SQW         = VEPS_W + 2;                  // working width for the sqrt datapath

    // requant constants (in the wide signed accumulator domain).
    localparam signed [ACC_W-1:0] ROUND   = (SHIFT > 0) ? (1 <<< (SHIFT-1)) : 0;
    localparam signed [ACC_W-1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;   // +127
    localparam signed [ACC_W-1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));      // -128
    localparam        [INV_W+1:0] RECIP_ONE = (1 <<< RECIP_FRAC);         // 2^RECIP_FRAC

    // ---- helper functions ---------------------------------------------
    // floor(sqrt(n)) by the classic bit-by-bit method, fully unrolled to a
    // constant ISQRT_ITERS for synthesis. All-unsigned (the >= compare must
    // be unsigned). Matches isqrt_floor() in layer_norm_cmodel.c bit-for-bit.
    function [R_W-1:0] isqrt;
        input [VEPS_W-1:0] n_in;
        reg [SQW-1:0] n, one, t, res;
        integer it;
        begin
            n   = n_in;
            res = {SQW{1'b0}};
            one = {{(SQW-1){1'b0}}, 1'b1} <<< (2*(ISQRT_ITERS-1));
            for (it = 0; it < ISQRT_ITERS; it = it + 1) begin
                t = res + one;
                if (n >= t) begin
                    n   = n - t;
                    res = (res >> 1) + one;
                end else begin
                    res = res >> 1;
                end
                one = one >> 2;
            end
            isqrt = res[R_W-1:0];
        end
    endfunction

    // sign-extend an int8 beta to ACC_W, then shift left by RECIP_FRAC to
    // align with the (num*inv*gamma) fractional scale.
    function signed [ACC_W-1:0] align_beta;
        input signed [DATA_WIDTH-1:0] b;
        reg   signed [ACC_W-1:0] be;
        begin
            be         = b;                  // sign-extend to ACC_W
            align_beta = be <<< RECIP_FRAC;  // shift inside ACC_W (no loss)
        end
    endfunction

    // round-half-up, arithmetic right shift by SHIFT, saturate to int8.
    function signed [DATA_WIDTH-1:0] requant;
        input signed [ACC_W-1:0] acc;
        reg   signed [ACC_W-1:0] s, r;
        begin
            s = acc + ROUND;
            r = s >>> SHIFT;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            requant = r[DATA_WIDTH-1:0];
        end
    endfunction

    // ---- coefficient register file (FF-based, write-only port) --------
    reg signed [DATA_WIDTH-1:0] gamma_mem [0:D_TOKEN-1];
    reg signed [DATA_WIDTH-1:0] beta_mem  [0:D_TOKEN-1];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (wr_is_beta) beta_mem[wr_addr]  <= wr_data;
            else            gamma_mem[wr_addr] <= wr_data;
        end
    end

    // ---- valid pipeline (resettable; datapath self-flushes via valid) -
    reg v1, v2, v3, v4, v5;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4;
        end
    end
    assign out_valid = v5;

    // ---- stage 0 (combinational): unpack x, reduce to S and SS --------
    reg signed [DATA_WIDTH-1:0] x_in  [0:D_TOKEN-1];
    reg signed [S_W-1:0]        s_comb;
    reg        [SS_W-1:0]       ss_comb;
    reg        [2*DATA_WIDTH-1:0] sq;
    integer ci;
    always_comb begin
        s_comb  = {S_W{1'b0}};
        ss_comb = {SS_W{1'b0}};
        for (ci = 0; ci < D_TOKEN; ci = ci + 1) begin
            x_in[ci] = $signed(x_vec[ci*DATA_WIDTH +: DATA_WIDTH]);
            sq       = x_in[ci] * x_in[ci];           // >= 0, exact in 2*DATA_WIDTH bits
            s_comb   = s_comb + x_in[ci];
            ss_comb  = ss_comb + sq;
        end
    end

    // ---- stage 1 regs: latch x, S, SS ---------------------------------
    reg signed [DATA_WIDTH-1:0] xq1 [0:D_TOKEN-1];
    reg signed [S_W-1:0]        sr1;
    reg        [SS_W-1:0]       ssr1;
    integer i1;
    always_ff @(posedge clk) begin
        sr1  <= s_comb;
        ssr1 <= ss_comb;
        for (i1 = 0; i1 < D_TOKEN; i1 = i1 + 1) xq1[i1] <= x_in[i1];
    end

    // ---- stage 2 (combinational): V, Veps, r = floor(sqrt(Veps)) ------
    reg [VEPS_W-1:0] dss_comb, ssq_comb, v_comb, veps_comb;
    reg [R_W-1:0]    r_comb;
    always_comb begin
        dss_comb  = D_TOKEN * ssr1;          // D * sum(x^2)
        ssq_comb  = sr1 * sr1;               // S^2 (>= 0)
        v_comb    = dss_comb - ssq_comb;     // V = D*SS - S^2 >= 0 (Cauchy-Schwarz)
        veps_comb = v_comb + EPS_V;
        r_comb    = isqrt(veps_comb);
        if (r_comb == {R_W{1'b0}}) r_comb = {{(R_W-1){1'b0}}, 1'b1};  // clamp r >= 1
    end

    // ---- stage 2 regs: pipe x, S; register r --------------------------
    reg signed [DATA_WIDTH-1:0] xq2 [0:D_TOKEN-1];
    reg signed [S_W-1:0]        sr2;
    reg        [R_W-1:0]        rr2;
    integer i2;
    always_ff @(posedge clk) begin
        sr2 <= sr1;
        rr2 <= r_comb;
        for (i2 = 0; i2 < D_TOKEN; i2 = i2 + 1) xq2[i2] <= xq1[i2];
    end

    // ---- stage 3 (combinational): inv = round(2^RECIP_FRAC / r) -------
    reg [INV_W+1:0] recip_num;
    reg [INV_W-1:0] inv_comb;
    always_comb begin
        recip_num = RECIP_ONE + (rr2 >> 1);    // rr2 zero-extends to recip_num width
        inv_comb  = recip_num / rr2;           // unsigned reciprocal = round(2^RECIP_FRAC / r)
    end

    // ---- stage 3 regs: pipe x, S; register inv ------------------------
    reg signed [DATA_WIDTH-1:0] xq3 [0:D_TOKEN-1];
    reg signed [S_W-1:0]        sr3;
    reg        [INV_W-1:0]      invr3;
    integer i3;
    always_ff @(posedge clk) begin
        sr3   <= sr2;
        invr3 <= inv_comb;
        for (i3 = 0; i3 < D_TOKEN; i3 = i3 + 1) xq3[i3] <= xq2[i3];
    end

    // ---- stage 4 regs: per-lane num = D*x-S, znorm = num*inv ----------
    reg signed [ZNORM_W-1:0] znormr4 [0:D_TOKEN-1];

    // ---- stage 5 regs: per-lane affine + requant ----------------------
    reg signed [DATA_WIDTH-1:0] y_q [0:D_TOKEN-1];

    genvar gi;
    generate
        for (gi = 0; gi < D_TOKEN; gi = gi + 1) begin : g_lane
            reg signed [NUM_W-1:0]  num_c;
            reg signed [ACC_W-1:0]  acc_c;

            // stage 4: num and normalized value (scaled by 2^RECIP_FRAC)
            always_comb num_c = D_TOKEN * xq3[gi] - sr3;
            always_ff @(posedge clk)
                znormr4[gi] <= $signed(num_c) * $signed({1'b0, invr3});

            // stage 5: affine multiply-add then requantize to int8
            always_comb acc_c = $signed(znormr4[gi]) * $signed(gamma_mem[gi])
                                + align_beta(beta_mem[gi]);
            always_ff @(posedge clk)
                y_q[gi] <= requant(acc_c);

            assign y_vec[gi*DATA_WIDTH +: DATA_WIDTH] = y_q[gi];
        end
    endgenerate

endmodule
`default_nettype wire
