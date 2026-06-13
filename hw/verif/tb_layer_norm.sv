// =====================================================================
// tb_layer_norm.sv
//
// End-to-end self-checking testbench for hw/rtl/layer_norm.v, run under VCS.
//   1. Loads random int8 gamma/beta into the DUT via the write port and keeps
//      golden copies in SystemVerilog.
//   2. Streams random int8 tokens: a back-to-back burst (proves II = 1) plus a
//      gapped phase (random bubbles) for extra coverage. Each driven token is
//      pushed to a scoreboard queue.
//   3. On every out_valid it pops the matching input, calls the behavioral C
//      reference model through DPI-C (layer_norm_cmodel -> layer_norm_int8 in
//      src/models/layer_norm_cmodel.c), and compares the int8 output element by
//      element. The RTL and the C model run the identical integer datapath, so
//      the check is BIT-EXACT.
//   4. Dumps an FSDB (KDB written at compile via `-kdb`) so Verdi opens the
//      waveform with full source / code hierarchy.
//
// Build/run: see hw/verif/Makefile (VCS only).
//   make all                       # compile + run, expect "PASS: ... II=1"
//   make verdi                     # open layer_norm.fsdb with KDB hierarchy
//   make all VCS_DEFINES=+define+LN_DT=64
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

`ifndef LN_DT
  `define LN_DT 32
`endif

module tb_layer_norm;

    // ---- parameters (must match the DUT instance) --------------------
    localparam int DT         = `LN_DT;   // d_token
    localparam int DW         = 8;        // int8
    localparam int FRAC       = 7;        // Q1.7
    localparam int RECIP_FRAC = 24;       // reciprocal fractional bits
    localparam int OUT_FRAC   = 7;        // strict Q1.7 output
    // EPS_V = round(eps * 2^(2*FRAC) * DT^2), eps = 1e-5, via integer math.
    localparam longint EPS_V  = ((longint'(1) << (2*FRAC)) * DT * DT + 50000) / 100000;

    localparam int NBURST = 64;   // back-to-back tokens (II = 1 proof)
    localparam int NGAP   = 48;   // gapped tokens (extra coverage)
    localparam int LAT    = 5;    // pipeline latency (informational; TB is latency-agnostic)
    localparam int ADDR_W = (DT <= 1) ? 1 : $clog2(DT);

    // ---- DPI-C import: the behavioral reference model ----------------
    import "DPI-C" function void layer_norm_cmodel(
        input  int     D,
        input  int     frac_bits,
        input  longint eps_v,
        input  int     recip_frac,
        input  int     out_frac,
        input  byte    x     [`LN_DT],
        input  byte    gamma [`LN_DT],
        input  byte    beta  [`LN_DT],
        output byte    y     [`LN_DT]
    );

    // ---- DUT I/O -----------------------------------------------------
    reg                  clk, rst_n;
    reg                  wr_en, wr_is_beta;
    reg  [ADDR_W-1:0]    wr_addr;
    reg  [DW-1:0]        wr_data;
    reg                  in_valid;
    reg  [DT*DW-1:0]     x_vec;
    wire                 out_valid;
    wire [DT*DW-1:0]     y_vec;

    // ---- golden coefficients (kept in SV, mirrored into the DUT) -----
    byte g_arr [DT];
    byte b_arr [DT];

    // ---- scoreboard --------------------------------------------------
    bit [DT*DW-1:0] inq [$];     // driven tokens awaiting their output
    byte x_arr [DT];
    byte y_exp [DT];

    integer errors, out_count, drive_count;
    integer cyc, first_out_cyc, last_out_cyc, burst_outs;
    integer i, n;
    reg     checking, burst_phase;

    // ---- DUT ---------------------------------------------------------
    layer_norm #(
        .D_TOKEN(DT), .DATA_WIDTH(DW), .FRAC_BITS(FRAC),
        .RECIP_FRAC(RECIP_FRAC), .OUT_FRAC(OUT_FRAC), .EPS_V(EPS_V)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_is_beta(wr_is_beta), .wr_addr(wr_addr), .wr_data(wr_data),
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
                layer_norm_cmodel(DT, FRAC, EPS_V, RECIP_FRAC, OUT_FRAC,
                                  x_arr, g_arr, b_arr, y_exp);
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

    // ---- stimulus ----------------------------------------------------
    initial begin : stim
        int seed;
        errors = 0; out_count = 0; drive_count = 0; burst_outs = 0;
        first_out_cyc = 0; last_out_cyc = 0; checking = 0; burst_phase = 0;
        wr_en = 0; wr_is_beta = 0; wr_addr = 0; wr_data = 0;
        in_valid = 0; x_vec = '0; rst_n = 0;
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));

        // FSDB dump ("+mda" captures the unpacked gamma/beta coefficient memories).
        $fsdbDumpfile("layer_norm.fsdb");
        $fsdbDumpvars(0, tb_layer_norm, "+mda");

        // random int8 coefficients
        for (i = 0; i < DT; i = i + 1) begin
            g_arr[i] = byte'($urandom);
            b_arr[i] = byte'($urandom);
        end

        // reset
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        checking = 1;

        // load gamma then beta, one coefficient per cycle
        for (i = 0; i < DT; i = i + 1) begin
            wr_en = 1; wr_is_beta = 0; wr_addr = i[ADDR_W-1:0]; wr_data = g_arr[i];
            @(negedge clk);
        end
        for (i = 0; i < DT; i = i + 1) begin
            wr_en = 1; wr_is_beta = 1; wr_addr = i[ADDR_W-1:0]; wr_data = b_arr[i];
            @(negedge clk);
        end
        wr_en = 0; wr_is_beta = 0; wr_addr = 0; wr_data = 0;
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
            $display("PASS: %0d tokens (DT=%0d, OUT_FRAC=%0d) match C-model bit-for-bit; "
                     "II=1 over %0d-beat burst.", out_count, DT, OUT_FRAC, NBURST);
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
