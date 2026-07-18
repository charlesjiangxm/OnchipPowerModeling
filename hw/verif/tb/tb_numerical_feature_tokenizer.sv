// =====================================================================
// tb_numerical_feature_tokenizer.sv
//
// Self-checking testbench for numerical_feature_tokenizer.
//   1. Loads known int8 weight/bias into every (j,k) via the write port.
//   2. Streams NROWS x-rows back-to-back with in_valid held high to prove
//      II = 1 (a new row every cycle, outputs contiguous).
//   3. Compares each int8 output against a wide-integer golden model that
//      mirrors the DUT's align/round/saturate semantics.
//
// Build/run: see hw/verif/Makefile.
//   make TB=numerical_feature_tokenizer all
// =====================================================================

`timescale 1ns/1ps
`default_nettype none
module tb_numerical_feature_tokenizer;

    localparam NF    = 4;            // N_FEATURE
    localparam DT    = 4;            // D_TOKEN
    localparam DW    = 8;            // DATA_WIDTH (int8)
    localparam FRAC  = 7;            // FRAC_BITS  (Q1.7)
    localparam DEPTH = NF*DT;
    localparam AW    = $clog2(DEPTH); // write-address width
    localparam LAT   = 3;     // pipeline latency (cycles)
    localparam NROWS = 12;    // number of streamed rows

    reg                  clk, rst_n;
    reg                  wr_en, wr_is_bias;
    reg  [AW-1:0]        wr_addr;
    reg  [DW-1:0]        wr_data;
    reg                  in_valid;
    reg  [NF*DW-1:0]     x_row;
    wire                 out_valid;
    wire [NF*DT*DW-1:0]  out_tokens;

    // Golden copies of the loaded coefficients and the driven input rows.
    reg signed [DW-1:0] w_mem [0:DEPTH-1];
    reg signed [DW-1:0] b_mem [0:DEPTH-1];
    reg [NF*DW-1:0]     xhist [0:NROWS-1];

    integer errors, out_count, cyc, first_out_cyc, last_out_cyc;
    integer i, j, k, idx, row;
    reg     checking;
    reg  [NF*DW-1:0] xrow_cur;
    integer dut_v, ref_v;

    // ---- DUT ----------------------------------------------------------
    numerical_feature_tokenizer #(
        .N_FEATURE(NF), .D_TOKEN(DT), .DATA_WIDTH(DW), .FRAC_BITS(FRAC)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .i_wr_en(wr_en), .i_wr_is_bias(wr_is_bias), .i_wr_addr(wr_addr), .i_wr_data(wr_data),
        .i_valid(in_valid), .i_x_row(x_row),
        .o_valid(out_valid), .o_tokens(out_tokens)
    );

    // ---- clock and free-running cycle counter -------------------------
    initial clk = 0;
    always #5 clk = ~clk;
    initial cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---- golden model (wide signed integer; matches DUT semantics) ----
    function integer ref_tok;
        input integer xj, w, b;
        integer prod, acc, s, r;
        begin
            prod = xj * w;
            acc  = prod + (b <<< FRAC);   // align bias to product scale
            s    = acc + (1 <<< (FRAC-1)); // round half up
            r    = s >>> FRAC;             // arithmetic right shift
            if      (r >  127) r =  127;   // saturate to int8
            else if (r < -128) r = -128;
            ref_tok = r;
        end
    endfunction

    // ---- output monitor / checker -------------------------------------
    always @(posedge clk) begin
        if (checking && out_valid) begin
            if (out_count == 0) first_out_cyc = cyc;
            last_out_cyc = cyc;
            xrow_cur = xhist[out_count];
            for (j = 0; j < NF; j = j + 1) begin
                for (k = 0; k < DT; k = k + 1) begin
                    idx   = j*DT + k;
                    dut_v = $signed(out_tokens[idx*DW +: DW]);
                    ref_v = ref_tok($signed(xrow_cur[j*DW +: DW]),
                                    $signed(w_mem[idx]), $signed(b_mem[idx]));
                    if (dut_v !== ref_v) begin
                        errors = errors + 1;
                        if (errors <= 20)
                            $display("  MISMATCH row=%0d j=%0d k=%0d  dut=%0d  ref=%0d",
                                     out_count, j, k, dut_v, ref_v);
                    end
                end
            end
            out_count = out_count + 1;
        end
    end

    // ---- stimulus -----------------------------------------------------
    initial begin
        rst_n = 0; wr_en = 0; wr_is_bias = 0; wr_addr = 0; wr_data = 0;
        in_valid = 0; x_row = 0; errors = 0; out_count = 0;
        first_out_cyc = 0; last_out_cyc = 0; checking = 0;

        $fsdbDumpfile("numerical_feature_tokenizer.fsdb");
        $fsdbDumpvars(0, tb_numerical_feature_tokenizer, "+mda");

        // Choose signed coefficient patterns (wrapped into int8).
        for (i = 0; i < DEPTH; i = i + 1) begin
            w_mem[i] = (i*7 - 13);
            b_mem[i] = (5 - i*3);
        end

        // Reset.
        repeat (3) @(negedge clk);
        rst_n    = 1;
        @(negedge clk);
        checking = 1;

        // Load weights, then biases (one coefficient per cycle).
        for (i = 0; i < DEPTH; i = i + 1) begin
            wr_en = 1; wr_is_bias = 0; wr_addr = i[AW-1:0]; wr_data = w_mem[i];
            @(negedge clk);
        end
        for (i = 0; i < DEPTH; i = i + 1) begin
            wr_en = 1; wr_is_bias = 1; wr_addr = i[AW-1:0]; wr_data = b_mem[i];
            @(negedge clk);
        end
        wr_en = 0; wr_is_bias = 0; wr_addr = 0; wr_data = 0;
        @(negedge clk);

        // Stream NROWS rows back-to-back (in_valid high every cycle => II=1).
        for (row = 0; row < NROWS; row = row + 1) begin
            in_valid = 1;
            if (row == 0) begin                 // force +saturation
                for (j = 0; j < NF; j = j + 1) x_row[j*DW +: DW] =  8'sd127;
            end else if (row == 1) begin        // force -saturation
                for (j = 0; j < NF; j = j + 1) x_row[j*DW +: DW] = -8'sd128;
            end else begin                      // pseudo-random coverage
                for (j = 0; j < NF; j = j + 1) x_row[j*DW +: DW] = $random;
            end
            xhist[row] = x_row;
            @(negedge clk);
        end
        in_valid = 0;
        x_row    = 0;

        // Drain the pipeline.
        repeat (LAT + 3) @(negedge clk);

        // ---- report ---------------------------------------------------
        $display("----------------------------------------------------------");
        if (out_count !== NROWS) begin
            errors = errors + 1;
            $display("FAIL: produced %0d output beats, expected %0d", out_count, NROWS);
        end
        if (out_count == NROWS &&
            (last_out_cyc - first_out_cyc + 1) !== NROWS) begin
            errors = errors + 1;
            $display("FAIL: outputs not contiguous (II != 1): span=%0d for %0d beats",
                     (last_out_cyc - first_out_cyc + 1), NROWS);
        end
        if (errors == 0)
            $display("PASS: %0d rows x %0d tokens match; II=1 (contiguous outputs).",
                     NROWS, DEPTH);
        else
            $display("FAIL: %0d total error(s).", errors);
        $display("----------------------------------------------------------");
        $finish;
    end

    // ---- safety timeout ----------------------------------------------
    initial begin
        #100000;
        $display("FAIL: simulation timeout");
        $finish;
    end

endmodule
`default_nettype wire
