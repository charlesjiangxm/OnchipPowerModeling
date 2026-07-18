//-----------------------------------------------------------------------------
// mlp_wrapper.v                                                (Verilog-2005*)
//
// title    : synthesis wrapper for mlp
// purpose  : preserve the DUT interface while adding input and output registers
//            so Design Compiler sees a clean flop-to-flop boundary around the
//            combinational port logic.
// schedule : input regs -> mlp -> output regs.
// language : Verilog-2005 + SystemVerilog always_ff (no logic); instantiates the
//            always_ff/always_comb DUT, analyze both as -format sverilog
//            (hw/syn/script/dc_mlp.tcl).
//-----------------------------------------------------------------------------
`default_nettype none
module mlp_wrapper #(
    parameter N_FEATURES = 32,   // fc1 in-dim (1-bit input vector length)
    parameter HIDDEN1    = 16,   // fc1 out-dim / fc2 in-dim
    parameter HIDDEN2    = 16,   // fc2 out-dim / fc3 in-dim
    parameter DATA_WIDTH = 8,    // signed int8 weights / bias / output
    // ---- derived (do not override) ----
    parameter W1_DEPTH = HIDDEN1 * N_FEATURES,
    parameter W2_DEPTH = HIDDEN2 * HIDDEN1,
    parameter W3_DEPTH = HIDDEN2,
    parameter WSEL_W   = 3,
    parameter MAXW_D1  = (W1_DEPTH > W2_DEPTH) ? W1_DEPTH : W2_DEPTH,
    parameter MAXW_DEP = (MAXW_D1 > W3_DEPTH) ? MAXW_D1 : W3_DEPTH,
    parameter WADDR_W  = ($clog2(MAXW_DEP) < 1) ? 1 : $clog2(MAXW_DEP),
    parameter CLOG2_H2 = ($clog2(HIDDEN2) < 1) ? 1 : $clog2(HIDDEN2),
    parameter ACC3_W   = 2*DATA_WIDTH + CLOG2_H2 + 2,
    parameter SHW3     = ($clog2(ACC3_W+1) < 1) ? 1 : $clog2(ACC3_W+1)
) (
    // control port
    input  wire                          clk,        // clock
    input  wire                          rst_n,      // async assert, sync deassert
    input  wire                          i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]           i_wr_sel,   // 0=W1 1=b1 2=W2 3=b2 4=W3 5=b3
    input  wire [WADDR_W -1:0]           i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]        i_wr_data,  // signed int8 coefficient
    input  wire                          i_valid,    // x valid this cycle
    // data port
    input  wire [N_FEATURES -1:0]        i_x,        // 1-bit input vector
    output wire                          o_valid,    // o_y / o_shift valid
    output wire signed [DATA_WIDTH -1:0] o_y,        // int8 result
    output wire [SHW3 -1:0]              o_shift     // fc3 dynamic right-shift
);

    // ---- input registers ----------------------------------------------------
    reg                    wr_en_ff;    // registered i_wr_en
    reg  [WSEL_W  -1:0]    wr_sel_ff;   // registered i_wr_sel
    reg  [WADDR_W -1:0]    wr_addr_ff;  // registered i_wr_addr
    reg  [DATA_WIDTH -1:0] wr_data_ff;  // registered i_wr_data
    reg                    valid_ff;    // registered i_valid
    reg  [N_FEATURES -1:0] x_ff;        // registered i_x

    // ---- DUT outputs ---------------------------------------------------------
    wire                     dut_valid;  // DUT o_valid
    wire signed [DATA_WIDTH -1:0] dut_y; // DUT o_y
    wire [SHW3 -1:0]         dut_shift;  // DUT o_shift

    // ---- output registers ----------------------------------------------------
    reg                    o_valid_ff;  // registered DUT o_valid
    reg  signed [DATA_WIDTH -1:0] y_ff; // registered DUT o_y
    reg  [SHW3 -1:0]       shift_ff;    // registered DUT o_shift

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
        x_ff       <= i_x;
        y_ff       <= dut_y;
        shift_ff   <= dut_shift;
    end

    mlp #(
        .N_FEATURES (N_FEATURES),
        .HIDDEN1    (HIDDEN1),
        .HIDDEN2    (HIDDEN2),
        .DATA_WIDTH (DATA_WIDTH)
    ) U_DUT (
        .clk       (clk),
        .rst_n     (rst_n),
        .i_wr_en   (wr_en_ff),
        .i_wr_sel  (wr_sel_ff),
        .i_wr_addr (wr_addr_ff),
        .i_wr_data (wr_data_ff),
        .i_valid   (valid_ff),
        .i_x       (x_ff),
        .o_valid   (dut_valid),
        .o_y       (dut_y),
        .o_shift   (dut_shift)
    );

    assign o_valid = o_valid_ff;
    assign o_y     = y_ff;
    assign o_shift = shift_ff;

endmodule
`default_nettype wire
