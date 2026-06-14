//-----------------------------------------------------------------------------
// align_bias.v                                                 (Verilog-2005*)
//
// title    : sign-extend an int8 bias and left-align it to the accumulator scale
// spec     : o_aligned = sign_extend(i_b) <<< SH
// datapath : pure combinational  sign-extend -> shift-left
// schedule : combinational; no clk / rst_n
// params   : IN_W  signed bias input width (int8)
//            OUT_W aligned output width (= accumulator width)
//            SH    left-shift to match the product fractional scale
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Shared helper. Replaces the per-file align_bias() / align_beta() function. The
// sign-extension happens by assigning the signed input to the wider signed wire
// (both signed), exactly as the original `be = b; be <<< SH;` did.
//-----------------------------------------------------------------------------
`default_nettype none
module align_bias #(
    parameter IN_W  = 8,    // signed bias input width (int8)
    parameter OUT_W = 32,   // aligned output width (= accumulator width)
    parameter SH    = 7     // left-shift to match the product fractional scale
) (
    input  wire signed [IN_W  -1:0] i_b,        // signed int8 bias
    output wire signed [OUT_W -1:0] o_aligned   // sign-extended, left-shifted bias
);

    wire signed [OUT_W -1:0] be;   // bias sign-extended to OUT_W

    assign be        = i_b;        // sign-extend (both signed, OUT_W wider)
    assign o_aligned = be <<< SH;  // shift inside OUT_W (no loss)

endmodule
`default_nettype wire
