// =====================================================================
// tb_feed_forward_network.sv
//
// End-to-end self-checking testbench for hw/rtl/feed_forward_network.v, run
// under VCS.
//   1. Loads random int8 weights (Linear1.weight/bias, Linear2.weight/bias)
//      into the DUT via the write port and keeps golden copies in SystemVerilog.
//   2. Streams random int8 tokens x_vec: a back-to-back burst (proves II = 1)
//      plus a gapped phase (random bubbles). Each driven token is pushed to a
//      scoreboard queue. The weights are static after load, so the golden
//      arrays ARE the per-token weight snapshot -- no per-token copy needed.
//   3. On every out_valid it pops the matching input, calls the behavioral C
//      reference model through DPI-C (feed_forward_network_cmodel ->
//      feed_forward_network_int8 in src/models/feed_forward_network_cmodel.c),
//      and compares the int8 output element by element. The RTL and the C model
//      run the identical integer datapath (Linear1 -> ReLU -> Linear2), so the
//      check is BIT-EXACT. (The ReLU activation is internal; the TB compares
//      only the end-to-end y bytes.)
//   4. Dumps an FSDB (KDB written at compile via `-kdb`) so Verdi opens the
//      waveform with full source / code hierarchy.
//
// Build/run: see hw/verif/Makefile (VCS only).
//   make all_ffn                         # compile + run, expect "PASS: ... II=1"
//   make verdi_ffn                       # open feed_forward_network.fsdb with KDB
//   make all_ffn VCS_DEFINES=+define+FFN_DFFN=128
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

`ifndef FFN_DT
  `define FFN_DT 32
`endif
`ifndef FFN_DFFN
  `define FFN_DFFN 64
`endif

module tb_feed_forward_network;

    // ---- parameters (must match the DUT instance) --------------------
    localparam int DT   = `FFN_DT;     // d_token  (Linear1 in / Linear2 out)
    localparam int DFFN = `FFN_DFFN;   // d_ffn    (hidden width)
    localparam int DW   = 8;           // int8
    localparam int FRAC = 7;           // Q1.7

    localparam int W1X  = DFFN * DT;   // Linear1.weight depth (F,E) row-major
    localparam int B1X  = DFFN;        // Linear1.bias depth
    localparam int W2X  = DT * DFFN;   // Linear2.weight depth (E,F) row-major
    localparam int B2X  = DT;          // Linear2.bias depth

    localparam int NBURST = 64;   // back-to-back tokens (II = 1 proof)
    localparam int NGAP   = 48;   // gapped tokens (extra coverage)
    localparam int LAT    = 4;    // pipeline latency (informational; TB is latency-agnostic)
    localparam int WADDR_W = ($clog2(W1X) < 1) ? 1 : $clog2(W1X);  // sized to largest array

    // ---- DPI-C import: the behavioral reference model ----------------
    import "DPI-C" function void feed_forward_network_cmodel(
        input  int  d_token,
        input  int  d_ffn,
        input  int  frac_bits,
        input  byte x  [DT],
        input  byte w1 [W1X],
        input  byte b1 [B1X],
        input  byte w2 [W2X],
        input  byte b2 [B2X],
        output byte y  [DT]
    );

    // ---- DUT I/O -----------------------------------------------------
    reg                  clk, rst_n;
    reg                  wr_en;
    reg  [1:0]           wr_sel;            // 0=W1 1=b1 2=W2 3=b2
    reg  [WADDR_W-1:0]   wr_addr;
    reg  [DW-1:0]        wr_data;
    reg                  in_valid;
    reg  [DT*DW-1:0]     x_vec;
    wire                 out_valid;
    wire [DT*DW-1:0]     y_vec;

    // ---- golden coefficients (kept in SV, mirrored into the DUT) -----
    byte w1_g [W1X];
    byte b1_g [B1X];
    byte w2_g [W2X];
    byte b2_g [B2X];

    // ---- scoreboard --------------------------------------------------
    bit [DT*DW-1:0] inq [$];      // driven tokens awaiting their output
    byte x_arr [DT];
    byte y_exp [DT];

    integer errors, out_count, drive_count;
    integer cyc, first_out_cyc, last_out_cyc, burst_outs;
    integer i, n;
    reg     checking, burst_phase;

    // ---- DUT (the core module, like tb_layer_norm / tb_multihead_attention) --
    feed_forward_network #(
        .D_TOKEN(DT), .D_FFN(DFFN), .DATA_WIDTH(DW), .FRAC_BITS(FRAC)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
        .in_valid(in_valid), .x_vec(x_vec),
        .out_valid(out_valid), .y_vec(y_vec)
    );

    // ---- clock + free-running cycle counter --------------------------
    initial clk = 0;
    always #5 clk = ~clk;
    initial cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---- output monitor / checker ------------------------------------
    always @(posedge clk) begin : checker
        bit [DT*DW-1:0] xp;
        byte got;
        if (checking && out_valid) begin
            if (inq.size() == 0) begin
                errors = errors + 1;
                $display("  ERROR: out_valid with empty scoreboard at cyc=%0d", cyc);
            end else begin
                xp = inq.pop_front();
                for (i = 0; i < DT; i = i + 1) x_arr[i] = xp[i*DW +: DW];
                feed_forward_network_cmodel(DT, DFFN, FRAC,
                                            x_arr, w1_g, b1_g, w2_g, b2_g, y_exp);
                for (i = 0; i < DT; i = i + 1) begin
                    got = y_vec[i*DW +: DW];
                    if (got !== y_exp[i]) begin
                        errors = errors + 1;
                        if (errors <= 20)
                            $display("  MISMATCH out#%0d i=%0d  dut=%0d  cmodel=%0d",
                                     out_count, i, $signed(got), y_exp[i]);
                    end
                end
            end
            if (burst_phase) begin
                if (burst_outs == 0) first_out_cyc = cyc;
                last_out_cyc = cyc;
                burst_outs = burst_outs + 1;
            end
            out_count = out_count + 1;
        end
    end

    // ---- drive one token at negedge; push to scoreboard if valid -----
    task drive_token(input bit valid);
        integer j;
        begin
            if (valid) begin
                for (j = 0; j < DT; j = j + 1) x_vec[j*DW +: DW] = byte'($urandom);
                in_valid = 1'b1;
                inq.push_back(x_vec);
                drive_count = drive_count + 1;
            end else begin
                in_valid = 1'b0;
                x_vec    = '0;
            end
            @(negedge clk);
        end
    endtask

    // ---- load one coefficient array through the write port -----------
    task load_coeffs(input int unsigned sel, input int unsigned depth);
        integer j;
        begin
            for (j = 0; j < depth; j = j + 1) begin
                wr_en   = 1'b1;
                wr_sel  = sel[1:0];
                wr_addr = j[WADDR_W-1:0];
                case (sel)
                    2'd0: wr_data = w1_g[j];
                    2'd1: wr_data = b1_g[j];
                    2'd2: wr_data = w2_g[j];
                    default: wr_data = b2_g[j];
                endcase
                @(negedge clk);
            end
        end
    endtask

    // ---- stimulus ----------------------------------------------------
    initial begin : stim
        int seed;
        errors = 0; out_count = 0; drive_count = 0; burst_outs = 0;
        first_out_cyc = 0; last_out_cyc = 0; checking = 0; burst_phase = 0;
        wr_en = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        in_valid = 0; x_vec = '0; rst_n = 0;
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));

        // FSDB dump ("+mda" captures the unpacked weight memories / pipe arrays).
        $fsdbDumpfile("feed_forward_network.fsdb");
        $fsdbDumpvars(0, tb_feed_forward_network, "+mda");

        // random int8 weights
        for (i = 0; i < W1X; i = i + 1) w1_g[i] = byte'($urandom);
        for (i = 0; i < B1X; i = i + 1) b1_g[i] = byte'($urandom);
        for (i = 0; i < W2X; i = i + 1) w2_g[i] = byte'($urandom);
        for (i = 0; i < B2X; i = i + 1) b2_g[i] = byte'($urandom);

        // reset
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        checking = 1;

        // load weights through the write-only port (one coefficient per cycle)
        load_coeffs(2'd0, W1X);   // Linear1.weight
        load_coeffs(2'd1, B1X);   // Linear1.bias
        load_coeffs(2'd2, W2X);   // Linear2.weight
        load_coeffs(2'd3, B2X);   // Linear2.bias
        wr_en = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        @(negedge clk);

        // phase A: back-to-back burst (in_valid high every cycle => II = 1)
        burst_phase = 1;
        for (n = 0; n < NBURST; n = n + 1) drive_token(1'b1);
        burst_phase = 0;

        // phase B: gapped tokens with random bubbles
        for (n = 0; n < NGAP; n = n + 1) begin
            drive_token(1'b1);
            if (($urandom % 3) == 0) drive_token(1'b0);   // insert a bubble
        end

        // drain the pipeline
        in_valid = 0; x_vec = '0;
        repeat (LAT + 4) @(negedge clk);

        // ---- report --------------------------------------------------
        $display("----------------------------------------------------------");
        if (out_count !== drive_count) begin
            errors = errors + 1;
            $display("FAIL: produced %0d output beats, expected %0d", out_count, drive_count);
        end
        if (burst_outs == NBURST &&
            (last_out_cyc - first_out_cyc + 1) !== NBURST) begin
            errors = errors + 1;
            $display("FAIL: burst outputs not contiguous (II != 1): span=%0d for %0d beats",
                     (last_out_cyc - first_out_cyc + 1), NBURST);
        end
        if (errors == 0)
            $display("PASS: %0d tokens (DT=%0d, DFFN=%0d) match C-model bit-for-bit; "
                     "II=1 over %0d-beat burst.", out_count, DT, DFFN, NBURST);
        else
            $display("FAIL: %0d total error(s).", errors);
        $display("----------------------------------------------------------");
        $finish;
    end

    // ---- safety timeout ----------------------------------------------
    initial begin
        #2000000;
        $display("FAIL: simulation timeout");
        $finish;
    end

endmodule
`default_nettype wire
