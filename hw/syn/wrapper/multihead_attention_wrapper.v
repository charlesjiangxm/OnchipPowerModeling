// =====================================================================
// multihead_attention_wrapper.v                            (Verilog-2005*)
//
// Synthesis wrapper for multihead_attention. Preserves the DUT interface while
// adding input and output registers so Design Compiler sees a clean
// flop-to-flop boundary around the combinational port logic.
//
// (*) Instantiates the SystemVerilog multihead_attention (always_ff/always_comb),
// so analyze both files as -format sverilog (see
// hw/syn/script/dc_multihead_attention.tcl). No `logic` is used here.
// =====================================================================

`default_nettype none
module multihead_attention_wrapper #(
    parameter D_TOKEN    = 32,
    parameter N_HEADS    = 8,
    parameter SEQ_LEN    = 16,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS  = 7,
    parameter SCALE_FRAC = 14,
    parameter SM_FRAC    = 8,
    parameter RECIP_FRAC = 24,
    parameter SCALE      = 8192,
    // ---- derived (do not override) ----
    parameter HD         = D_TOKEN / N_HEADS,
    parameter IPW_DEPTH  = 3 * D_TOKEN * D_TOKEN,
    parameter WSEL_W     = 2,
    parameter WADDR_W    = ($clog2(IPW_DEPTH) < 1) ? 1 : $clog2(IPW_DEPTH)
) (
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  wr_en,
    input  wire [WSEL_W-1:0]                     wr_sel,
    input  wire [WADDR_W-1:0]                    wr_addr,
    input  wire [DATA_WIDTH-1:0]                 wr_data,
    input  wire                                  in_valid,
    input  wire [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] x_seq,
    output wire                                  out_valid,
    output wire [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] y_seq
);

    reg                                  wr_en_q;
    reg  [WSEL_W-1:0]                     wr_sel_q;
    reg  [WADDR_W-1:0]                    wr_addr_q;
    reg  [DATA_WIDTH-1:0]                 wr_data_q;
    reg                                  in_valid_q;
    reg  [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] x_seq_q;

    wire                                  dut_out_valid;
    wire [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] dut_y_seq;

    reg                                  out_valid_q;
    reg  [SEQ_LEN*D_TOKEN*DATA_WIDTH-1:0] y_seq_q;

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
        x_seq_q   <= x_seq;
        y_seq_q   <= dut_y_seq;
    end

    multihead_attention #(
        .D_TOKEN(D_TOKEN),
        .N_HEADS(N_HEADS),
        .SEQ_LEN(SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .SCALE_FRAC(SCALE_FRAC),
        .SM_FRAC(SM_FRAC),
        .RECIP_FRAC(RECIP_FRAC),
        .SCALE(SCALE)
    ) u_multihead_attention (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_q),
        .wr_sel(wr_sel_q),
        .wr_addr(wr_addr_q),
        .wr_data(wr_data_q),
        .in_valid(in_valid_q),
        .x_seq(x_seq_q),
        .out_valid(dut_out_valid),
        .y_seq(dut_y_seq)
    );

    assign out_valid = out_valid_q;
    assign y_seq     = y_seq_q;

endmodule
`default_nettype wire
