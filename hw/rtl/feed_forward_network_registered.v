// =====================================================================
// feed_forward_network_registered.v                        (Verilog-2005*)
//
// Synthesis wrapper for feed_forward_network. Preserves the DUT interface while
// adding input and output registers so Design Compiler sees a clean
// flop-to-flop boundary around the combinational port logic.
//
// (*) Instantiates the SystemVerilog feed_forward_network (always_ff/always_comb),
// so analyze both files as -format sverilog (see
// hw/syn/script/dc_feed_forward_network.tcl). No `logic` is used here.
// =====================================================================

`default_nettype none
module feed_forward_network_registered #(
    parameter D_TOKEN    = 32,
    parameter D_FFN      = 64,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS  = 7,
    // ---- derived (do not override) ----
    parameter W1_DEPTH   = D_FFN * D_TOKEN,
    parameter W2_DEPTH   = D_TOKEN * D_FFN,
    parameter WSEL_W     = 2,
    parameter MAXW_DEPTH = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,
    parameter WADDR_W    = ($clog2(MAXW_DEPTH) < 1) ? 1 : $clog2(MAXW_DEPTH)
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          wr_en,
    input  wire [WSEL_W-1:0]             wr_sel,
    input  wire [WADDR_W-1:0]            wr_addr,
    input  wire [DATA_WIDTH-1:0]         wr_data,
    input  wire                          in_valid,
    input  wire [D_TOKEN*DATA_WIDTH-1:0] x_vec,
    output wire                          out_valid,
    output wire [D_TOKEN*DATA_WIDTH-1:0] y_vec
);

    reg                          wr_en_q;
    reg  [WSEL_W-1:0]            wr_sel_q;
    reg  [WADDR_W-1:0]           wr_addr_q;
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
        wr_sel_q  <= wr_sel;
        wr_addr_q <= wr_addr;
        wr_data_q <= wr_data;
        x_vec_q   <= x_vec;
        y_vec_q   <= dut_y_vec;
    end

    feed_forward_network #(
        .D_TOKEN(D_TOKEN),
        .D_FFN(D_FFN),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) u_feed_forward_network (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_q),
        .wr_sel(wr_sel_q),
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
