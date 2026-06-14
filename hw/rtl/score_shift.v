//-----------------------------------------------------------------------------
// score_shift.v                                                (Verilog-2005*)
//
// title    : scaled-score requantizer (round-half-up + arithmetic shift, no sat)
// spec     : o_s = (i_v + (1<<(SH-1))) >>> SH
// datapath : pure combinational  sum -> arithmetic-shift (truncated to OUT_W)
// schedule : combinational; no clk / rst_n
// params   : IN_W  signed scaled-score input width
//            OUT_W signed output width (sized to hold the result exactly)
//            SH    round-half-up + arithmetic right-shift amount
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Shared helper. Replaces multihead_attention's score_shift() function. Unlike
// requant there is NO int8 saturation: OUT_W is sized to hold the shifted score
// exactly, so the assignment's truncation to OUT_W matches the function return.
//-----------------------------------------------------------------------------
`default_nettype none
module score_shift #(
    parameter IN_W  = 40,   // signed scaled-score input width
    parameter OUT_W = 14,   // signed output width (sized exact; no saturation)
    parameter SH    = 13    // round-half-up + arithmetic right-shift amount
) (
    input  wire signed [IN_W  -1:0] i_v,   // raw*SCALE, Q(2*FRAC+SCALE_FRAC)
    output wire signed [OUT_W -1:0] o_s    // shifted score, Q(SM_FRAC)
);

    localparam signed [IN_W -1:0] RND = (SH > 0) ? (1 <<< (SH-1)) : 0;   // round-half-up

    wire signed [IN_W -1:0] sum;   // i_v + round-half-up constant

    assign sum = i_v + RND;
    assign o_s = sum >>> SH;       // computed in IN_W, truncated to OUT_W on assign

endmodule
`default_nettype wire
