//-----------------------------------------------------------------------------
// layer_norm.v                                                 (Verilog-2005*)
//
// title    : FT-Transformer LayerNorm over D_TOKEN axis (int8 inference)
//            (nn.LayerNorm(d_token): norm1/norm2/final_norm, eps=1e-5)
// math     : mean = (1/D)*sum(x); var = (1/D)*sum((x-mean)^2)   (population)
//            y[i] = (x[i]-mean)/sqrt(var+eps) * gamma[i] + beta[i]
// numeric  : 2^-FRAC and 1/D cancel -> y_norm[i] = (D*x[i]-S)/sqrt(V+EPS_V),
//            S = sum(x), V = D*sum(x^2)-S^2 (= D^2*var_int >= 0),
//            EPS_V = round(eps*2^(2*FRAC)*D^2). r = floor(sqrt(V+EPS_V)) (>=1),
//            inv = round(2^RECIP_FRAC / r). per lane:
//            acc = (D*x[i]-S)*inv*gamma[i] + (beta[i]<<RECIP_FRAC), then requant.
// datapath : reduce S/SS -> V -> integer sqrt (U_ISQRT) -> reciprocal
//            -> per-lane affine multiply-add + requant (U_REQUANT/U_ALIGN_BETA).
// schedule : one x_vec in / one y_vec out per clock, latency 5, II = 1.
// params   : D_TOKEN, DATA_WIDTH, FRAC_BITS, RECIP_FRAC, OUT_FRAC, EPS_V.
// language : Verilog-2005 + SystemVerilog always_ff/always_comb (no logic);
//            compile with `vcs -sverilog`. clk / rst_n (async assert, sync deassert).
// hw/model : behavioral twin src/models/layer_norm_cmodel.c (bit-exact).
//-----------------------------------------------------------------------------
`default_nettype none
module layer_norm #(
    parameter D_TOKEN    = 32,   // token dimension (length of the normalized axis)
    parameter DATA_WIDTH = 8,    // int8 element width (x, gamma, beta, y)
    parameter FRAC_BITS  = 7,    // input/affine fractional bits => Q1.7
    parameter RECIP_FRAC = 24,   // reciprocal fractional bits (1/std precision)
    parameter OUT_FRAC   = 7,    // output fractional bits (=FRAC_BITS => strict Q1.7)
    parameter EPS_V      = 168,  // integer epsilon = round(eps*2^(2*FRAC)*D^2), eps=1e-5
    // ---- derived (do not override) ----
    parameter ADDR_W     = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN)  // write-address width
) (
    // control port
    input  wire                           clk,           // clock
    input  wire                           rst_n,         // async assert, sync deassert
    input  wire                           i_wr_en,       // coefficient write strobe
    input  wire                           i_wr_is_beta,  // 0=gamma, 1=beta
    input  wire [ADDR_W     -1:0]         i_wr_addr,     // lane index i
    input  wire [DATA_WIDTH -1:0]         i_wr_data,     // signed int8 coefficient
    input  wire                           i_valid,       // x_vec valid this cycle
    // data port
    input  wire [D_TOKEN*DATA_WIDTH -1:0] i_x_vec,       // packed: x[i] = i_x_vec[i*W +: W]
    output wire                           o_valid,       // y_vec valid
    output wire [D_TOKEN*DATA_WIDTH -1:0] o_y_vec        // packed: y[i] = o_y_vec[i*W +: W]
);

    integer i;     // procedural loop index
    genvar  g;     // per-lane generate index

    localparam PIPE     = 5;                                // valid-pipeline / latency depth
    localparam CLOG2_D  = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN);
    localparam S_W      = DATA_WIDTH + CLOG2_D + 1;         // signed sum S = sum(x)
    localparam SS_W     = 2*DATA_WIDTH + CLOG2_D;           // unsigned sum of squares
    localparam VEPS_W   = 2*DATA_WIDTH + 2*CLOG2_D;         // unsigned V+EPS_V
    localparam R_W      = DATA_WIDTH + CLOG2_D;             // unsigned r = floor(sqrt(Veps))
    localparam INV_W    = RECIP_FRAC + 1;                   // unsigned inv (max 2^RECIP_FRAC at r=1)
    localparam NUM_W    = DATA_WIDTH + CLOG2_D + 2;         // signed num = D*x[i] - S
    localparam ZNORM_W  = NUM_W + INV_W + 1;                // signed num*inv
    localparam ACC_W    = ZNORM_W + DATA_WIDTH + 2;         // signed znorm*gamma + (beta<<RECIP_FRAC)
    localparam SHIFT    = FRAC_BITS + RECIP_FRAC - OUT_FRAC; // requant right-shift amount

    localparam [INV_W+1 :0] RECIP_ONE = (1 <<< RECIP_FRAC); // 2^RECIP_FRAC

    // ---- coefficient register file (FF-based, write-only port) --------------
    reg signed [DATA_WIDTH -1:0] gamma_mem [D_TOKEN-1 :0];   // affine scale gamma
    reg signed [DATA_WIDTH -1:0] beta_mem  [D_TOKEN-1 :0];   // affine shift  beta

    always_ff @(posedge clk) begin : DFF_WR
        if (i_wr_en) begin
            if (i_wr_is_beta) beta_mem[i_wr_addr]  <= i_wr_data;
            else              gamma_mem[i_wr_addr] <= i_wr_data;
        end
    end

    // ---- valid pipeline (resettable; datapath self-flushes via valid) -------
    reg [PIPE -1:0] valid_ff;   // PIPE-stage valid shift register

    always_ff @(posedge clk or negedge rst_n) begin : DFF_VLD
        if (!rst_n) valid_ff <= {PIPE{1'b0}};
        else        valid_ff <= {valid_ff[PIPE-2:0], i_valid};
    end
    assign o_valid = valid_ff[PIPE-1];

    // ---- stage 0 (combinational): unpack x, reduce to S and SS --------------
    reg signed [DATA_WIDTH   -1:0] x_in [D_TOKEN-1 :0];   // unpacked x
    reg signed [S_W          -1:0] s_comb;                // sum(x)
    reg        [SS_W         -1:0] ss_comb;               // sum(x^2)
    reg        [2*DATA_WIDTH -1:0] sq;                    // x[i]^2

    always_comb begin : CMB_RED
        s_comb  = {S_W{1'b0}};
        ss_comb = {SS_W{1'b0}};
        for (i = 0; i < D_TOKEN; i = i + 1) begin
            x_in[i] = $signed(i_x_vec[i*DATA_WIDTH +: DATA_WIDTH]);
            sq      = x_in[i] * x_in[i];   // >= 0, exact in 2*DATA_WIDTH bits
            s_comb  = s_comb + x_in[i];
            ss_comb = ss_comb + sq;
        end
    end

    // ---- stage 1 regs: latch x, S, SS ---------------------------------------
    reg signed [DATA_WIDTH -1:0] xq1_ff [D_TOKEN-1 :0];   // x carried to stage 1
    reg signed [S_W        -1:0] sr1_ff;                  // S  at stage 1
    reg        [SS_W       -1:0] ssr1_ff;                 // SS at stage 1

    always_ff @(posedge clk) begin : DFF_S1
        sr1_ff  <= s_comb;
        ssr1_ff <= ss_comb;
        for (i = 0; i < D_TOKEN; i = i + 1) xq1_ff[i] <= x_in[i];
    end

    // ---- stage 2 (combinational): V, Veps, r = floor(sqrt(Veps)) ------------
    wire [VEPS_W -1:0] dss;      // D * sum(x^2)
    wire [VEPS_W -1:0] ssq;      // S^2 (>= 0)
    wire [VEPS_W -1:0] v_int;    // V = D*SS - S^2 >= 0 (Cauchy-Schwarz)
    wire [VEPS_W -1:0] veps;     // V + EPS_V
    wire [R_W    -1:0] r_isqrt;  // floor(sqrt(Veps))
    wire [R_W    -1:0] r_comb;   // r clamped >= 1

    assign dss   = D_TOKEN * ssr1_ff;   // D * sum(x^2)
    assign ssq   = sr1_ff * sr1_ff;     // S^2 (signed*signed, result >= 0)
    assign v_int = dss - ssq;
    assign veps  = v_int + EPS_V;
    isqrt #(
        .VEPS_W (VEPS_W),
        .R_W    (R_W)
    ) U_ISQRT (
        .i_n (veps),
        .o_r (r_isqrt)
    );
    assign r_comb = (r_isqrt == {R_W{1'b0}}) ? {{(R_W-1){1'b0}}, 1'b1} : r_isqrt; // clamp r >= 1

    // ---- stage 2 regs: pipe x, S; register r --------------------------------
    reg signed [DATA_WIDTH -1:0] xq2_ff [D_TOKEN-1 :0];   // x carried to stage 2
    reg signed [S_W        -1:0] sr2_ff;                  // S at stage 2
    reg        [R_W        -1:0] rr2_ff;                  // r at stage 2

    always_ff @(posedge clk) begin : DFF_S2
        sr2_ff <= sr1_ff;
        rr2_ff <= r_comb;
        for (i = 0; i < D_TOKEN; i = i + 1) xq2_ff[i] <= xq1_ff[i];
    end

    // ---- stage 3 (combinational): inv = round(2^RECIP_FRAC / r) -------------
    wire [INV_W+1 :0] recip_num;   // 2^RECIP_FRAC + r/2 (round numerator)
    wire [INV_W -1:0] inv_comb;    // unsigned reciprocal round(2^RECIP_FRAC / r)

    assign recip_num = RECIP_ONE + (rr2_ff >> 1);   // rr2_ff zero-extends to recip_num width
    assign inv_comb  = recip_num / rr2_ff;

    // ---- stage 3 regs: pipe x, S; register inv ------------------------------
    reg signed [DATA_WIDTH -1:0] xq3_ff [D_TOKEN-1 :0];   // x carried to stage 3
    reg signed [S_W        -1:0] sr3_ff;                  // S at stage 3
    reg        [INV_W      -1:0] invr3_ff;                // inv at stage 3

    always_ff @(posedge clk) begin : DFF_S3
        sr3_ff   <= sr2_ff;
        invr3_ff <= inv_comb;
        for (i = 0; i < D_TOKEN; i = i + 1) xq3_ff[i] <= xq2_ff[i];
    end

    // ---- stage 4/5 regs: per-lane num/znorm then affine + requant -----------
    reg signed [ZNORM_W    -1:0] znorm4_ff [D_TOKEN-1 :0];   // num*inv at stage 4
    reg signed [DATA_WIDTH -1:0] y_ff      [D_TOKEN-1 :0];   // requantized output

    generate
        for (g = 0; g < D_TOKEN; g = g + 1) begin : G_LANE
            wire signed [NUM_W      -1:0] num_c;    // D*x[g] - S
            wire signed [ACC_W      -1:0] beta_al;  // aligned beta (beta << RECIP_FRAC)
            wire signed [ACC_W      -1:0] acc_c;    // affine accumulator
            wire signed [DATA_WIDTH -1:0] y_c;      // requantized lane output

            // stage 4: num and normalized value (scaled by 2^RECIP_FRAC)
            assign num_c = D_TOKEN * xq3_ff[g] - sr3_ff;
            always_ff @(posedge clk) begin : DFF_ZNORM
                znorm4_ff[g] <= $signed(num_c) * $signed({1'b0, invr3_ff});
            end

            // stage 5: affine multiply-add then requantize to int8
            align_bias #(
                .IN_W  (DATA_WIDTH),
                .OUT_W (ACC_W),
                .SH    (RECIP_FRAC)
            ) U_ALIGN_BETA (
                .i_b       (beta_mem[g]),
                .o_aligned (beta_al)
            );
            assign acc_c = $signed(znorm4_ff[g]) * $signed(gamma_mem[g]) + beta_al;
            requant #(
                .ACC_W      (ACC_W),
                .DATA_WIDTH (DATA_WIDTH),
                .SHIFT      (SHIFT)
            ) U_REQUANT (
                .i_acc (acc_c),
                .o_q   (y_c)
            );
            always_ff @(posedge clk) begin : DFF_Y
                y_ff[g] <= y_c;
            end

            assign o_y_vec[g*DATA_WIDTH +: DATA_WIDTH] = y_ff[g];
        end
    endgenerate

endmodule
`default_nettype wire
