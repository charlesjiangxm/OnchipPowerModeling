// =====================================================================
// layer_norm_registered.v                                  (Verilog-2005*)
//
// Synthesis wrapper for layer_norm. Preserves the DUT interface while adding
// input and output registers so Design Compiler sees a clean flop-to-flop
// boundary around the combinational port logic.
//
// (*) Instantiates the SystemVerilog layer_norm (always_ff/always_comb), so
// analyze both files as -format sverilog (see hw/syn/script/dc_layer_norm.tcl).
// No `logic` is used here -- only reg/wire.
// =====================================================================

`default_nettype none
module layer_norm_registered #(
    parameter D_TOKEN    = 32,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS  = 7,
    parameter RECIP_FRAC = 24,
    parameter OUT_FRAC   = 7,
    parameter EPS_V      = 168,
    // ---- derived (do not override) ----
    parameter ADDR_W     = ($clog2(D_TOKEN) < 1) ? 1 : $clog2(D_TOKEN)
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          wr_en,
    input  wire                          wr_is_beta,
    input  wire [ADDR_W-1:0]             wr_addr,
    input  wire [DATA_WIDTH-1:0]         wr_data,
    input  wire                          in_valid,
    input  wire [D_TOKEN*DATA_WIDTH-1:0] x_vec,
    output wire                          out_valid,
    output wire [D_TOKEN*DATA_WIDTH-1:0] y_vec
);

    reg                          wr_en_q;
    reg                          wr_is_beta_q;
    reg  [ADDR_W-1:0]            wr_addr_q;
    reg  [DATA_WIDTH-1:0]        wr_data_q;
    reg                          in_valid_q;
    reg  [D_TOKEN*DATA_WIDTH-1:0] x_vec_q;

    wire                         dut_out_valid;
    wire [D_TOKEN*DATA_WIDTH-1:0] dut_y_vec;

    reg                          out_valid_q;
    reg  [D_TOKEN*DATA_WIDTH-1:0] y_vec_q;

    // Reset only control-valid flops. Data flops are don't-care while their
    // associated valid/write-enable controls are low.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_en_q     <= 1'b0;
            in_valid_q  <= 1'b0;
            out_valid_q <= 1'b0;
        end else begin
            wr_en_q     <= wr_en;
            in_valid_q  <= in_valid;
            out_valid_q <= dut_out_valid;
        end
    end

    always @(posedge clk) begin
        wr_is_beta_q <= wr_is_beta;
        wr_addr_q    <= wr_addr;
        wr_data_q    <= wr_data;
        x_vec_q      <= x_vec;
        y_vec_q      <= dut_y_vec;
    end

    layer_norm #(
        .D_TOKEN(D_TOKEN),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .RECIP_FRAC(RECIP_FRAC),
        .OUT_FRAC(OUT_FRAC),
        .EPS_V(EPS_V)
    ) u_layer_norm (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_q),
        .wr_is_beta(wr_is_beta_q),
        .wr_addr(wr_addr_q),
        .wr_data(wr_data_q),
        .in_valid(in_valid_q),
        .x_vec(x_vec_q),
        .out_valid(dut_out_valid),
        .y_vec(dut_y_vec)
    );

    assign out_valid = out_valid_q;
    assign y_vec     = y_vec_q;

endmodule
`default_nettype wire
