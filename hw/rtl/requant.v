//-----------------------------------------------------------------------------
// requant.v                                                    (Verilog-2005*)
//
// title    : symmetric int8 requantizer (round-half-up, shift, saturate)
// spec     : o_q = sat_int8( (i_acc + (1<<(SHIFT-1))) >>> SHIFT )
//            arithmetic right shift, then clamp to [-2^(W-1), 2^(W-1)-1]
// datapath : pure combinational  sum -> arithmetic-shift -> saturate -> slice
// schedule : combinational; no clk / rst_n
// params   : ACC_W  signed accumulator (input) width
//            DATA_W signed output width (int8)
//            SHIFT  arithmetic right-shift amount (>= 0)
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Shared helper. Replaces the per-file requant() function (numerical_feature_
// tokenizer / layer_norm / feed_forward_network / multihead_attention) and is
// bit-identical: the round / shift / saturate is computed in the connected
// accumulator width, which always carries the value with headroom, so the
// result matches the original wide-domain computation exactly.
//-----------------------------------------------------------------------------
`default_nettype none
module requant #(
    parameter ACC_W      = 32,   // signed accumulator (input) width
    parameter DATA_WIDTH = 8,    // signed output width (int8)
    parameter SHIFT      = 7     // arithmetic right-shift amount (>= 0)
) (
    input  wire signed [ACC_W      -1:0] i_acc,   // wide signed accumulator
    output wire signed [DATA_WIDTH -1:0] o_q      // requantized int8 result
);

    localparam signed [ACC_W -1:0] ROUND   = (SHIFT > 0) ? (1 <<< (SHIFT-1)) : 0; // round-half-up
    localparam signed [ACC_W -1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;          // +127
    localparam signed [ACC_W -1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));             // -128

    wire signed [ACC_W -1:0] sum;   // i_acc + round-half-up constant
    wire signed [ACC_W -1:0] sht;   // arithmetic right shift by SHIFT
    wire signed [ACC_W -1:0] sat;   // clamped to the signed int8 range

    assign sum = i_acc + ROUND;
    assign sht = sum >>> SHIFT;
    assign sat = (sht > OUT_MAX) ? OUT_MAX :
                 (sht < OUT_MIN) ? OUT_MIN : sht;
    assign o_q = sat[DATA_WIDTH -1:0];

endmodule
`default_nettype wire
