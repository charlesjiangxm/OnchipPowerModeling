// =====================================================================
// numerical_feature_tokenizer_registered.v                   (Verilog-2005)
//
// Synthesis wrapper for numerical_feature_tokenizer.  The wrapper preserves
// the DUT interface while adding input and output registers so Design Compiler
// sees a clean flop-to-flop boundary around the combinational port logic.
// =====================================================================

`default_nettype none
module numerical_feature_tokenizer_registered #(
    parameter N_FEATURE  = 20,
    parameter D_TOKEN    = 32,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS  = 7,
    // ---- derived (do not override) ----
    parameter DEPTH      = N_FEATURE * D_TOKEN,
    parameter ADDR_W     = ($clog2(DEPTH) < 1) ? 1 : $clog2(DEPTH)
) (
    input  wire                                    clk,
    input  wire                                    rst_n,
    input  wire                                    wr_en,
    input  wire                                    wr_is_bias,
    input  wire [ADDR_W-1:0]                       wr_addr,
    input  wire [DATA_WIDTH-1:0]                   wr_data,
    input  wire                                    in_valid,
    input  wire [N_FEATURE*DATA_WIDTH-1:0]         x_row,
    output wire                                    out_valid,
    output wire [N_FEATURE*D_TOKEN*DATA_WIDTH-1:0] out_tokens
);

    reg                                    wr_en_q;
    reg                                    wr_is_bias_q;
    reg  [ADDR_W-1:0]                      wr_addr_q;
    reg  [DATA_WIDTH-1:0]                  wr_data_q;
    reg                                    in_valid_q;
    reg  [N_FEATURE*DATA_WIDTH-1:0]        x_row_q;

    wire                                   dut_out_valid;
    wire [N_FEATURE*D_TOKEN*DATA_WIDTH-1:0] dut_out_tokens;

    reg                                    out_valid_q;
    reg  [N_FEATURE*D_TOKEN*DATA_WIDTH-1:0] out_tokens_q;

    // Reset only control-valid flops.  Data flops are don't-care while their
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
        wr_is_bias_q <= wr_is_bias;
        wr_addr_q    <= wr_addr;
        wr_data_q    <= wr_data;
        x_row_q      <= x_row;
        out_tokens_q <= dut_out_tokens;
    end

    numerical_feature_tokenizer #(
        .N_FEATURE(N_FEATURE),
        .D_TOKEN(D_TOKEN),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) u_numerical_feature_tokenizer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_q),
        .wr_is_bias(wr_is_bias_q),
        .wr_addr(wr_addr_q),
        .wr_data(wr_data_q),
        .in_valid(in_valid_q),
        .x_row(x_row_q),
        .out_valid(dut_out_valid),
        .out_tokens(dut_out_tokens)
    );

    assign out_valid  = out_valid_q;
    assign out_tokens = out_tokens_q;

endmodule
`default_nettype wire
