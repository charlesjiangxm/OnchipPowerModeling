//-----------------------------------------------------------------------------
// isqrt.v                                                      (Verilog-2005*)
//
// title    : unsigned integer floor square root (classic bit-by-bit method)
// spec     : o_r = floor(sqrt(i_n))
// datapath : pure combinational, fully unrolled to ITERS iterations
// schedule : combinational; no clk / rst_n
// params   : VEPS_W unsigned radicand width
//            R_W    result width (floor-sqrt bits)
//            ITERS  fixed (unrolled) iteration count   -- derived
//            SQW    sqrt datapath working width        -- derived
// language : Verilog-2005 + always_ff/always_comb dialect (no logic).
//
// Shared helper. Replaces layer_norm's isqrt() function and matches
// isqrt_floor() in layer_norm_cmodel.c bit-for-bit. All arithmetic is unsigned
// (the >= compare must be unsigned).
//-----------------------------------------------------------------------------
`default_nettype none
module isqrt #(
    parameter VEPS_W = 24,   // unsigned radicand width
    parameter R_W    = 13,   // result width (floor-sqrt bits)
    // ---- derived (do not override) ----
    parameter ITERS  = (VEPS_W + 1) / 2,   // fixed unrolled iteration count
    parameter SQW    = VEPS_W + 2           // sqrt datapath working width
) (
    input  wire [VEPS_W -1:0] i_n,   // unsigned radicand (V + EPS_V)
    output wire [R_W    -1:0] o_r    // floor(sqrt(i_n))
);

    integer i;   // unroll index

    reg [SQW -1:0] n;     // running remainder
    reg [SQW -1:0] one;   // current bit weight (4^k)
    reg [SQW -1:0] t;     // trial value res + one
    reg [SQW -1:0] res;   // accumulated root

    always_comb begin : CMB_ISQRT
        n   = i_n;
        res = {SQW{1'b0}};
        one = {{(SQW-1){1'b0}}, 1'b1} <<< (2*(ITERS-1));
        for (i = 0; i < ITERS; i = i + 1) begin
            t = res + one;
            if (n >= t) begin
                n   = n - t;
                res = (res >> 1) + one;
            end else begin
                res = res >> 1;
            end
            one = one >> 2;
        end
    end

    assign o_r = res[R_W -1:0];

endmodule
`default_nettype wire
