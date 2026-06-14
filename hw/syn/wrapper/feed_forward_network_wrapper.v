//-----------------------------------------------------------------------------
// feed_forward_network_wrapper.v                              (Verilog-2005*)
//
// title    : synthesis wrapper for feed_forward_network
// purpose  : preserve the DUT interface while adding input and output registers
//            so Design Compiler sees a clean flop-to-flop boundary around the
//            combinational port logic.
// schedule : input regs -> feed_forward_network -> output regs.
// language : Verilog-2005 + SystemVerilog always_ff (no logic); instantiates the
//            always_ff/always_comb DUT, analyze both as -format sverilog
//            (hw/syn/script/dc_feed_forward_network.tcl).
//-----------------------------------------------------------------------------
`default_nettype none
module feed_forward_network_wrapper #(
    parameter D_TOKEN    = 32,   // E: Linear1 in-dim / Linear2 out-dim
    parameter D_FFN      = 64,   // F: hidden width
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    // ---- derived (do not override) ----
    parameter W1_DEPTH   = D_FFN * D_TOKEN,
    parameter W2_DEPTH   = D_TOKEN * D_FFN,
    parameter WSEL_W     = 2,
    parameter MAXW_DEPTH = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,
    parameter WADDR_W    = ($clog2(MAXW_DEPTH) < 1) ? 1 : $clog2(MAXW_DEPTH)
) (
    // control port
    input  wire                           clk,        // clock
    input  wire                           rst_n,      // async assert, sync deassert
    input  wire                           i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]            i_wr_sel,   // 0=W1 1=b1 2=W2 3=b2
    input  wire [WADDR_W -1:0]            i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]         i_wr_data,  // signed int8 coefficient
    input  wire                           i_valid,    // x_vec valid this cycle
    // data port
    input  wire [D_TOKEN*DATA_WIDTH -1:0] i_x_vec,    // packed input token
    output wire                           o_valid,    // y_vec valid
    output wire [D_TOKEN*DATA_WIDTH -1:0] o_y_vec     // packed output token
);

    // ---- input registers ----------------------------------------------------
    reg                            wr_en_ff;   // registered i_wr_en
    reg  [WSEL_W  -1:0]            wr_sel_ff;  // registered i_wr_sel
    reg  [WADDR_W -1:0]            wr_addr_ff; // registered i_wr_addr
    reg  [DATA_WIDTH -1:0]         wr_data_ff; // registered i_wr_data
    reg                            valid_ff;   // registered i_valid
    reg  [D_TOKEN*DATA_WIDTH -1:0] x_vec_ff;   // registered i_x_vec

    // ---- DUT outputs ---------------------------------------------------------
    wire                           dut_valid; // DUT o_valid
    wire [D_TOKEN*DATA_WIDTH -1:0] dut_y_vec; // DUT o_y_vec

    // ---- output registers ----------------------------------------------------
    reg                            o_valid_ff; // registered DUT o_valid
    reg  [D_TOKEN*DATA_WIDTH -1:0] y_vec_ff;   // registered DUT o_y_vec

    // control/valid flops are reset; data flops are don't-care while their
    // associated valid/write-enable controls are low.
    always_ff @(posedge clk or negedge rst_n) begin : DFF_CTRL
        if (!rst_n) begin
            wr_en_ff   <= 1'b0;
            valid_ff   <= 1'b0;
            o_valid_ff <= 1'b0;
        end else begin
            wr_en_ff   <= i_wr_en;
            valid_ff   <= i_valid;
            o_valid_ff <= dut_valid;
        end
    end

    always_ff @(posedge clk) begin : DFF_DATA
        wr_sel_ff  <= i_wr_sel;
        wr_addr_ff <= i_wr_addr;
        wr_data_ff <= i_wr_data;
        x_vec_ff   <= i_x_vec;
        y_vec_ff   <= dut_y_vec;
    end

    feed_forward_network #(
        .D_TOKEN    (D_TOKEN),
        .D_FFN      (D_FFN),
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (FRAC_BITS)
    ) U_DUT (
        .clk       (clk),
        .rst_n     (rst_n),
        .i_wr_en   (wr_en_ff),
        .i_wr_sel  (wr_sel_ff),
        .i_wr_addr (wr_addr_ff),
        .i_wr_data (wr_data_ff),
        .i_valid   (valid_ff),
        .i_x_vec   (x_vec_ff),
        .o_valid   (dut_valid),
        .o_y_vec   (dut_y_vec)
    );

    assign o_valid = o_valid_ff;
    assign o_y_vec = y_vec_ff;

endmodule
`default_nettype wire
