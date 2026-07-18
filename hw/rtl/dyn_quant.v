//-----------------------------------------------------------------------------
// dyn_quant.v                                                  (Verilog-2005*)
//
// title    : per-vector dynamic (block-floating-point) int8 requantizer
// math     : s     = max(0, bitlen(max_p |v_p|) - (DATA_WIDTH-1))   (one shared
//                                                                    shift)
//            o_v_p = sat_int8( RNE( v_p >> s ) )                     (per element)
// datapath : abs + max-reduce + bit-length -> shift  (combinational), then M
//            parallel requant_rne lanes sharing the one dynamic shift.
// schedule : combinational; no clk / rst_n
// params   : M          number of vector elements
//            ACC_W      signed element (input) width
//            DATA_WIDTH signed output width (int8)
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// The two's-complement negation of the most-negative element yields the correct
// magnitude when read as the same-width UNSIGNED value (|-2^(W-1)| = 2^(W-1) =
// the bit pattern 1000..0), so MAXABS width = ACC_W is exact; in this design the
// ACC_W headroom keeps the rail unreachable anyway.
//-----------------------------------------------------------------------------
`default_nettype none
module dyn_quant #(
    parameter M          = 16,   // number of vector elements
    parameter ACC_W      = 24,   // signed element (input) width
    parameter DATA_WIDTH = 8,    // signed output width (int8)
    // ---- derived (do not override) ----
    parameter SHW        = ($clog2(ACC_W+1) < 1) ? 1 : $clog2(ACC_W+1)  // shift width
) (
    input  wire [M*ACC_W      -1:0] i_vec,    // packed signed: v[p]=i_vec[p*ACC_W +: ACC_W]
    output wire [M*DATA_WIDTH -1:0] o_vec,    // packed signed int8: o[p]=o_vec[p*W +: W]
    output wire [SHW          -1:0] o_shift   // shared dynamic right-shift
);

    integer p, q;     // procedural reduce / bit-scan indices
    genvar  g;        // per-element lane index

    reg signed [ACC_W   -1:0] v_s;       // scratch: signed element
    reg        [ACC_W   -1:0] a_u;       // scratch: |element| (unsigned magnitude)
    reg        [ACC_W   -1:0] maxabs;    // max |v_p| over the vector
    reg        [SHW     -1:0] shift_c;   // derived shared shift
    reg        [ACC_W      :0] bl;       // bit-length of maxabs (0..ACC_W)

    // ---- abs + max-reduce + bit-length -> shift ----------------------------
    always_comb begin : CMB_SHIFT
        maxabs = {ACC_W{1'b0}};
        for (p = 0; p < M; p = p + 1) begin
            v_s = $signed(i_vec[p*ACC_W +: ACC_W]);
            a_u = v_s[ACC_W-1] ? (-v_s) : v_s;             // |v_p|
            if (a_u > maxabs) maxabs = a_u;
        end
        bl = {(ACC_W+1){1'b0}};
        for (q = 0; q < ACC_W; q = q + 1)
            if (maxabs[q]) bl = q + 1;                     // highest set bit + 1
        shift_c = (bl > (DATA_WIDTH-1)) ? (bl - (DATA_WIDTH-1)) : {SHW{1'b0}};
    end
    assign o_shift = shift_c;

    // ---- M parallel requant lanes sharing the one dynamic shift ------------
    generate
        for (g = 0; g < M; g = g + 1) begin : G_LANE
            requant_rne #(
                .ACC_W      (ACC_W),
                .DATA_WIDTH (DATA_WIDTH),
                .SHW        (SHW)
            ) U_RQ (
                .i_acc   ($signed(i_vec[g*ACC_W +: ACC_W])),
                .i_shift (shift_c),
                .o_q     (o_vec[g*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule
`default_nettype wire
