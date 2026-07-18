//-----------------------------------------------------------------------------
// numerical_feature_tokenizer_wrapper.v                       (Verilog-2005*)
//
// title    : synthesis wrapper for numerical_feature_tokenizer
// purpose  : preserve the DUT interface while adding input and output registers
//            so Design Compiler sees a clean flop-to-flop boundary around the
//            combinational port logic.
// schedule : input regs -> numerical_feature_tokenizer -> output regs.
// language : Verilog-2005 + SystemVerilog always_ff (no logic); instantiates the
//            always_ff/always_comb DUT, analyze both as -format sverilog
//            (hw/syn/script/dc_numerical_feature_tokenizer.tcl).
//-----------------------------------------------------------------------------
`default_nettype none
module numerical_feature_tokenizer_wrapper #(
    parameter N_FEATURE  = 20,   // number of features
    parameter D_TOKEN    = 32,   // token dimension
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    // ---- derived (do not override) ----
    parameter DEPTH      = N_FEATURE * D_TOKEN,
    parameter ADDR_W     = ($clog2(DEPTH) < 1) ? 1 : $clog2(DEPTH)
) (
    // control port
    input  wire                                     clk,           // clock
    input  wire                                     rst_n,         // async assert, sync deassert
    input  wire                                     i_wr_en,       // coefficient write strobe
    input  wire                                     i_wr_is_bias,  // 0=weight, 1=bias
    input  wire [ADDR_W     -1:0]                   i_wr_addr,     // linear index j*D_TOKEN+k
    input  wire [DATA_WIDTH -1:0]                   i_wr_data,     // signed int8 coefficient
    input  wire                                     i_valid,       // x_row valid this cycle
    // data port
    input  wire [N_FEATURE*DATA_WIDTH -1:0]         i_x_row,       // packed input row
    output wire                                     o_valid,       // o_tokens valid
    output wire [N_FEATURE*D_TOKEN*DATA_WIDTH -1:0] o_tokens       // packed token matrix
);

    // ---- input registers ----------------------------------------------------
    reg                              wr_en_ff;      // registered i_wr_en
    reg                              wr_is_bias_ff; // registered i_wr_is_bias
    reg  [ADDR_W     -1:0]           wr_addr_ff;    // registered i_wr_addr
    reg  [DATA_WIDTH -1:0]           wr_data_ff;    // registered i_wr_data
    reg                              valid_ff;      // registered i_valid
    reg  [N_FEATURE*DATA_WIDTH -1:0] x_row_ff;      // registered i_x_row

    // ---- DUT outputs ---------------------------------------------------------
    wire                                     dut_valid;  // DUT o_valid
    wire [N_FEATURE*D_TOKEN*DATA_WIDTH -1:0] dut_tokens; // DUT o_tokens

    // ---- output registers ----------------------------------------------------
    reg                                      o_valid_ff; // registered DUT o_valid
    reg  [N_FEATURE*D_TOKEN*DATA_WIDTH -1:0] tokens_ff;  // registered DUT o_tokens

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
        wr_is_bias_ff <= i_wr_is_bias;
        wr_addr_ff    <= i_wr_addr;
        wr_data_ff    <= i_wr_data;
        x_row_ff      <= i_x_row;
        tokens_ff     <= dut_tokens;
    end

    numerical_feature_tokenizer #(
        .N_FEATURE  (N_FEATURE),
        .D_TOKEN    (D_TOKEN),
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (FRAC_BITS)
    ) U_DUT (
        .clk          (clk),
        .rst_n        (rst_n),
        .i_wr_en      (wr_en_ff),
        .i_wr_is_bias (wr_is_bias_ff),
        .i_wr_addr    (wr_addr_ff),
        .i_wr_data    (wr_data_ff),
        .i_valid      (valid_ff),
        .i_x_row      (x_row_ff),
        .o_valid      (dut_valid),
        .o_tokens     (dut_tokens)
    );

    assign o_valid  = o_valid_ff;
    assign o_tokens = tokens_ff;

endmodule
`default_nettype wire
