//-----------------------------------------------------------------------------
// requant_rne.v                                               (Verilog-2005*)
//
// title    : dynamic-shift int8 requantizer (round-to-nearest-ties-to-even)
// spec     : q_floor = i_acc >>> i_shift                 (arithmetic floor)
//            rem     = i_acc - (q_floor <<< i_shift)      in [0, 2^i_shift)
//            half    = (i_shift==0) ? 0 : 1 << (i_shift-1)
//            up      = (rem > half) || (i_shift!=0 && rem==half && q_floor[0])
//            o_q     = sat_int8( q_floor + up )
//            i_shift==0 => no rounding (the i_shift!=0 guard kills the false tie
//            at rem==half==0); exact halves round to EVEN.
// datapath : pure combinational; variable arithmetic shift -> round -> saturate
// schedule : combinational; no clk / rst_n
// params   : ACC_W      signed accumulator (input) width
//            DATA_WIDTH signed output width (int8)
//            SHW        i_shift port width
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Dynamic-shift sibling of requant.v: the shift is a data-dependent PORT (block
// floating-point, driven by dyn_quant.v), and rounding is ties-to-even rather
// than the round-half-up requant.v uses. Saturation matches requant.v.
//-----------------------------------------------------------------------------
`default_nettype none
module requant_rne #(
    parameter ACC_W      = 24,   // signed accumulator (input) width
    parameter DATA_WIDTH = 8,    // signed output width (int8)
    parameter SHW        = 5     // i_shift port width
) (
    input  wire signed [ACC_W      -1:0] i_acc,    // wide signed accumulator
    input  wire        [SHW        -1:0] i_shift,  // dynamic right-shift amount
    output wire signed [DATA_WIDTH -1:0] o_q       // requantized int8 result
);

    localparam signed [ACC_W -1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;  // +127
    localparam signed [ACC_W -1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));     // -128

    wire signed [ACC_W -1:0] q_floor;  // arithmetic floor (i_acc >>> i_shift)
    wire signed [ACC_W -1:0] back;     // q_floor re-expanded to the acc scale
    wire signed [ACC_W -1:0] rem;      // dropped low bits, in [0, 2^i_shift)
    wire signed [ACC_W -1:0] half;     // tie threshold 2^(i_shift-1)
    wire                     tie;      // exact halfway case
    wire                     up;       // round-up decision
    wire signed [ACC_W -1:0] up_ext;   // signed 0/1 increment (keeps add signed)
    wire signed [ACC_W -1:0] q;        // rounded accumulator
    wire signed [ACC_W -1:0] sat;      // clamped to the signed int8 range

    assign q_floor = i_acc >>> i_shift;
    assign back    = q_floor <<< i_shift;
    assign rem     = i_acc - back;                                       // >= 0
    assign half    = (i_shift == 0) ? {ACC_W{1'b0}}
                                    : ({{(ACC_W-1){1'b0}}, 1'b1} <<< (i_shift-1));
    assign tie     = (i_shift != 0) && (rem == half);
    assign up      = (rem > half) || (tie && q_floor[0]);                // ties -> even
    assign up_ext  = up ? {{(ACC_W-1){1'b0}}, 1'b1} : {ACC_W{1'b0}};
    assign q       = q_floor + up_ext;
    assign sat     = (q > OUT_MAX) ? OUT_MAX :
                     (q < OUT_MIN) ? OUT_MIN : q;
    assign o_q     = sat[DATA_WIDTH -1:0];

endmodule
`default_nettype wire
