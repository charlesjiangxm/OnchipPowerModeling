//-----------------------------------------------------------------------------
// multihead_attention_wrapper.v                              (Verilog-2005*)
//
// title    : synthesis wrapper for multihead_attention
// purpose  : preserve the DUT interface while adding input and output registers
//            so Design Compiler sees a clean flop-to-flop boundary around the
//            combinational port logic.
// schedule : input regs -> multihead_attention -> output regs.
// language : Verilog-2005 + SystemVerilog always_ff (no logic); instantiates the
//            always_ff/always_comb DUT, analyze both as -format sverilog
//            (hw/syn/script/dc_multihead_attention.tcl).
//-----------------------------------------------------------------------------
`default_nettype none
module multihead_attention_wrapper #(
    parameter D_TOKEN    = 32,   // E: embedding dim
    parameter N_HEADS    = 8,    // H: number of heads
    parameter SEQ_LEN    = 16,   // S: sequence length
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // Q1.7
    parameter SCALE_FRAC = 14,   // fractional bits of SCALE
    parameter SM_FRAC    = 8,    // softmax-input score fractional bits
    parameter RECIP_FRAC = 24,   // reciprocal fractional bits
    parameter SCALE      = 8192, // round(2^SCALE_FRAC / sqrt(HD))
    // ---- derived (do not override) ----
    parameter HD         = D_TOKEN / N_HEADS,
    parameter IPW_DEPTH  = 3 * D_TOKEN * D_TOKEN,
    parameter WSEL_W     = 2,
    parameter WADDR_W    = ($clog2(IPW_DEPTH) < 1) ? 1 : $clog2(IPW_DEPTH)
) (
    // control port
    input  wire                                   clk,        // clock
    input  wire                                   rst_n,      // async assert, sync deassert
    input  wire                                   i_wr_en,    // coefficient write strobe
    input  wire [WSEL_W  -1:0]                    i_wr_sel,   // 0=ipw 1=ipb 2=opw 3=opb
    input  wire [WADDR_W -1:0]                    i_wr_addr,  // linear index in selected array
    input  wire [DATA_WIDTH -1:0]                 i_wr_data,  // signed int8 coefficient
    input  wire                                   i_valid,    // x_seq valid this cycle
    // data port
    input  wire [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] i_x_seq,    // packed input sequence
    output wire                                   o_valid,    // y_seq valid
    output wire [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] o_y_seq     // packed output sequence
);

    // ---- input registers ----------------------------------------------------
    reg                                    wr_en_ff;   // registered i_wr_en
    reg  [WSEL_W  -1:0]                    wr_sel_ff;  // registered i_wr_sel
    reg  [WADDR_W -1:0]                    wr_addr_ff; // registered i_wr_addr
    reg  [DATA_WIDTH -1:0]                 wr_data_ff; // registered i_wr_data
    reg                                    valid_ff;   // registered i_valid
    reg  [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] x_seq_ff;   // registered i_x_seq

    // ---- DUT outputs ---------------------------------------------------------
    wire                                   dut_valid; // DUT o_valid
    wire [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] dut_y_seq; // DUT o_y_seq

    // ---- output registers ----------------------------------------------------
    reg                                    o_valid_ff; // registered DUT o_valid
    reg  [SEQ_LEN*D_TOKEN*DATA_WIDTH -1:0] y_seq_ff;   // registered DUT o_y_seq

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
        x_seq_ff   <= i_x_seq;
        y_seq_ff   <= dut_y_seq;
    end

    multihead_attention #(
        .D_TOKEN    (D_TOKEN),
        .N_HEADS    (N_HEADS),
        .SEQ_LEN    (SEQ_LEN),
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (FRAC_BITS),
        .SCALE_FRAC (SCALE_FRAC),
        .SM_FRAC    (SM_FRAC),
        .RECIP_FRAC (RECIP_FRAC),
        .SCALE      (SCALE)
    ) U_DUT (
        .clk       (clk),
        .rst_n     (rst_n),
        .i_wr_en   (wr_en_ff),
        .i_wr_sel  (wr_sel_ff),
        .i_wr_addr (wr_addr_ff),
        .i_wr_data (wr_data_ff),
        .i_valid   (valid_ff),
        .i_x_seq   (x_seq_ff),
        .o_valid   (dut_valid),
        .o_y_seq   (dut_y_seq)
    );

    assign o_valid = o_valid_ff;
    assign o_y_seq = y_seq_ff;

endmodule
`default_nettype wire
