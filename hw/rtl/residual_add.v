// =====================================================================
// residual_add.v                                           (Verilog-2005*)
//
// Elementwise residual add for the FT-Transformer top level. The pre-norm
// residual STREAM is carried at the coarser Q1.RES_FRAC (Q3.5, range +-4) so
// it does not clip as it grows across blocks; the MHA/FFN branch outputs are
// strict Q1.FRAC_BITS (Q1.7). This block aligns the Q1.7 branch output down to
// the stream's Q3.5 scale (round-half-up) and adds, saturating to int8.
//
// Bit-exact twin: ft_residual_add() in src/models/ft_transformer_cmodel.c.
//   out[i] = sat8( stream[i] + ((module[i] + 2^(SH-1)) >>> SH) ),  SH = FRAC-RES.
//
// Purely combinational (no clock): one full vector of VEC_LEN int8 lanes per
// call. The top level registers the result into the sequence buffer; doing the
// whole sequence in parallel keeps a residual add to ~1 cycle (it is only
// adders + clamps, no multipliers). Requires RES_FRAC <= FRAC_BITS (SH >= 0),
// which holds for the Q3.5 stream / Q1.7 modules.
//
// (*) Verilog-2005 dialect; vendor-neutral; uses reg/wire only.
// =====================================================================

`default_nettype none
module residual_add #(
    parameter VEC_LEN    = 544,   // int8 lanes (e.g. SEQ_LEN*D_TOKEN = 17*32)
    parameter DATA_WIDTH = 8,     // int8
    parameter FRAC_BITS  = 7,     // module_vec fractional bits (Q1.7)
    parameter RES_FRAC   = 5      // stream/out fractional bits (Q3.5)
) (
    input  wire [VEC_LEN*DATA_WIDTH-1:0] stream_vec,  // Q1.RES_FRAC residual stream
    input  wire [VEC_LEN*DATA_WIDTH-1:0] module_vec,  // Q1.FRAC_BITS branch output
    output wire [VEC_LEN*DATA_WIDTH-1:0] out_vec      // Q1.RES_FRAC, saturated int8
);

    localparam integer SH    = FRAC_BITS - RES_FRAC;       // >= 0
    localparam integer ACC_W = DATA_WIDTH + 4;             // headroom for align + add
    localparam signed [ACC_W-1:0] RND     = (SH > 0) ? (1 <<< (SH-1)) : 0;
    localparam signed [ACC_W-1:0] OUT_MAX = (1 <<< (DATA_WIDTH-1)) - 1;  // +127
    localparam signed [ACC_W-1:0] OUT_MIN = -(1 <<< (DATA_WIDTH-1));     // -128

    // align an int8 Q1.FRAC value down to Q1.RES_FRAC (round-half-up shift).
    function signed [ACC_W-1:0] align_down;
        input signed [DATA_WIDTH-1:0] m;
        reg   signed [ACC_W-1:0] me;
        begin
            me         = m;
            align_down = (me + RND) >>> SH;
        end
    endfunction

    // saturate a wide signed value to int8.
    function signed [DATA_WIDTH-1:0] sat8;
        input signed [ACC_W-1:0] v;
        reg   signed [ACC_W-1:0] r;
        begin
            r = v;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            sat8 = r[DATA_WIDTH-1:0];
        end
    endfunction

    genvar i;
    generate
        for (i = 0; i < VEC_LEN; i = i + 1) begin : g_lane
            wire signed [DATA_WIDTH-1:0] s_el =
                 $signed(stream_vec[i*DATA_WIDTH +: DATA_WIDTH]);
            wire signed [DATA_WIDTH-1:0] m_el =
                 $signed(module_vec[i*DATA_WIDTH +: DATA_WIDTH]);
            assign out_vec[i*DATA_WIDTH +: DATA_WIDTH] =
                 sat8($signed(s_el) + align_down(m_el));
        end
    endgenerate

endmodule
`default_nettype wire
