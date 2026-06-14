//-----------------------------------------------------------------------------
// exp_neg.v                                                    (Verilog-2005*)
//
// title    : integer exp(d) for d <= 0  (base-2 shift + Q16 quadratic)
// spec     : e = exp(d) = 2^(d*log2e) = 2^-z * 2^-f , d in Q(SM_FRAC), e in Q16
//            2^-f ~= EXP_C2*f^2 + EXP_C1*f + EXP_C0   (Q16, endpoint-exact)
// datapath : combinational  u=-d -> m=u*LOG2E -> {z,f}=split(m)
//            -> 2^-f poly (p) -> e = p >> z   (0 past ZMAX)
// schedule : combinational; no clk / rst_n
// params   : D_W     signed input width (d = sm - rowmax, <= 0)
//            SM_FRAC softmax-input score fractional bits
//            (EXP_FRAC..Z_W derived; must not be overridden)
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Shared helper. Replaces multihead_attention's exp_neg() function; the fixed
// Q16 constants MUST match multihead_attention_cmodel.c, so it is bit-identical.
//-----------------------------------------------------------------------------
`default_nettype none
module exp_neg #(
    parameter D_W      = 14,   // signed input width (d = sm - rowmax, <= 0)
    parameter SM_FRAC  = 8,    // softmax-input score fractional bits
    // ---- derived (do not override) ----
    parameter EXP_FRAC = 16,                  // output Q(EXP_FRAC)
    parameter EXP_W    = EXP_FRAC + 1,        // width of e (max 2^16)
    parameter LOG2E_FR = 16,                  // log2(e) constant fractional bits
    parameter U_W      = D_W,                 // width of u = -d
    parameter M_W      = U_W + 18,            // width of m = u*LOG2E (LOG2E < 2^17)
    parameter SH_EXP   = SM_FRAC + LOG2E_FR,  // m split point: integer | fraction
    parameter ZMAX     = EXP_FRAC + 1,        // e -> 0 for z >= ZMAX
    parameter Z_W      = (M_W > SH_EXP) ? (M_W - SH_EXP) : 1
) (
    input  wire signed [D_W   -1:0] i_d,   // d <= 0, Q(SM_FRAC)
    output wire        [EXP_W -1:0] o_e    // e = exp(d), Q(EXP_FRAC), >= 0
);

    localparam               LOG2E  = 94548;     // round(log2(e) * 2^16)
    localparam               EXP_C2 = 11279;     // round( 0.172100 * 2^16)
    localparam               EXP_C0 = 65536;     // 2^16
    localparam signed [18 -1:0] EXP_C1 = -44047; // round(-0.672100 * 2^16)

    wire        [U_W      -1:0] u;        // -d (>= 0)
    wire        [M_W      -1:0] m;        // u * LOG2E
    wire        [Z_W      -1:0] z;        // integer part of m (a shift count)
    wire        [EXP_FRAC -1:0] f;        // fractional part of m
    wire        [64       -1:0] t2_full;  // EXP_C2 * f * f (unsigned)
    wire        [32       -1:0] t2;       // C2*f^2 in Q16
    wire signed [48       -1:0] t1_full;  // EXP_C1 * f (signed)
    wire signed [32       -1:0] t1;       // C1*f in Q16
    wire signed [32       -1:0] p;        // 2^-f in Q16, in (0.5, 1]

    assign u       = (i_d < 0) ? (-i_d) : {U_W{1'b0}};
    assign m       = u * LOG2E;
    assign z       = m[M_W-1 : SH_EXP];                 // m >> SH_EXP
    assign f       = m[SH_EXP-1 -: EXP_FRAC];           // (m mod 2^SH_EXP) >> (SH_EXP-EXP_FRAC)
    assign t2_full = EXP_C2 * f * f;                    // unsigned
    assign t2      = t2_full >> (2*EXP_FRAC);           // C2*f^2 in Q16
    assign t1_full = EXP_C1 * $signed({1'b0, f});       // signed, EXP_C1 < 0
    assign t1      = t1_full >>> EXP_FRAC;              // arithmetic
    assign p       = $signed({1'b0, t2}) + t1 + EXP_C0; // 2^-f, Q16, in (0.5,1]
    assign o_e     = (z >= ZMAX) ? {EXP_W{1'b0}} : (p[EXP_W -1:0] >> z);

endmodule
`default_nettype wire
