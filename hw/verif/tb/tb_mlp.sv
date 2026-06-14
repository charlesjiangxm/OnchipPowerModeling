// =====================================================================
// tb_mlp.sv
//
// End-to-end self-checking testbench for hw/rtl/mlp.v, run under VCS.
//   1. Loads random int8 coefficients (fc1/fc2/fc3 weight+bias, 6 arrays) into
//      the DUT via the write port and keeps golden copies in SystemVerilog.
//   2. Streams random 1-bit input vectors i_x: a back-to-back burst (proves
//      II = 1) plus a gapped phase (random bubbles). Each driven vector is
//      pushed to a scoreboard queue. Weights are static after load, so the
//      golden arrays ARE the per-vector weight snapshot -- no per-vector copy.
//   3. On every out_valid it pops the matching input, calls the behavioral C
//      reference model through DPI-C (mlp_cmodel -> mlp_int8 in
//      hw/model/mlp_cmodel.c), and compares BOTH the int8 result o_y and the
//      fc3 dynamic shift o_shift. The RTL and the C model run the identical
//      integer datapath (gated fc1 -> dyn-quant -> fc2 -> dyn-quant -> fc3 ->
//      dyn-quant), so the check is BIT-EXACT.
//   4. Dumps an FSDB (KDB written at compile via `-kdb`) so Verdi opens the
//      waveform with full source / code hierarchy.
//
// Build/run: see hw/verif/Makefile (VCS only).
//   make all_mlp                         # compile + run, expect "PASS: ... II=1"
//   make verdi_mlp                       # open mlp.fsdb with KDB
//   make all_mlp VCS_DEFINES=+define+MLP_H1=32+define+MLP_H2=24
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

`ifndef MLP_NF
  `define MLP_NF 32
`endif
`ifndef MLP_H1
  `define MLP_H1 16
`endif
`ifndef MLP_H2
  `define MLP_H2 16
`endif

module tb_mlp;

    // ---- parameters (must match the DUT instance) --------------------
    localparam int NF = `MLP_NF;   // n_features (1-bit input length)
    localparam int H1 = `MLP_H1;   // hidden1
    localparam int H2 = `MLP_H2;   // hidden2
    localparam int DW = 8;         // int8

    localparam int W1X = H1 * NF;  // fc1.weight (H1,NF) row-major
    localparam int B1X = H1;       // fc1.bias
    localparam int W2X = H2 * H1;  // fc2.weight (H2,H1) row-major
    localparam int B2X = H2;       // fc2.bias
    localparam int W3X = H2;       // fc3.weight (1,H2)
    localparam int B3X = 1;        // fc3.bias

    localparam int NBURST = 64;    // back-to-back vectors (II = 1 proof)
    localparam int NGAP   = 48;    // gapped vectors (extra coverage)
    localparam int LAT    = 7;     // pipeline latency (informational; TB is latency-agnostic)

    // largest coefficient array -> write-address width
    localparam int MAXW   = (W1X > W2X) ? ((W1X > W3X) ? W1X : W3X)
                                        : ((W2X > W3X) ? W2X : W3X);
    localparam int WADDR_W = ($clog2(MAXW) < 1) ? 1 : $clog2(MAXW);

    // fc3 accumulator / dynamic-shift width (matches mlp.v derivation)
    localparam int CL_H2  = ($clog2(H2) < 1) ? 1 : $clog2(H2);
    localparam int ACC3_W = 2*DW + CL_H2 + 2;
    localparam int SHW3   = ($clog2(ACC3_W+1) < 1) ? 1 : $clog2(ACC3_W+1);

    // ---- DPI-C import: the behavioral reference model ----------------
    import "DPI-C" function void mlp_cmodel(
        input  int  n_features,
        input  int  hidden1,
        input  int  hidden2,
        input  byte x  [NF],
        input  byte w1 [W1X],
        input  byte b1 [B1X],
        input  byte w2 [W2X],
        input  byte b2 [B2X],
        input  byte w3 [W3X],
        input  byte b3 [B3X],
        output byte y  [1],
        output int  o_shift
    );

    // ---- DUT I/O -----------------------------------------------------
    reg                  clk, rst_n;
    reg                  wr_en;
    reg  [2:0]           wr_sel;            // 0=W1 1=b1 2=W2 3=b2 4=W3 5=b3
    reg  [WADDR_W-1:0]   wr_addr;
    reg  [DW-1:0]        wr_data;
    reg                  in_valid;
    reg  [NF-1:0]        x_vec;
    wire                 out_valid;
    wire signed [DW-1:0] y_out;
    wire [SHW3-1:0]      shift_out;

    // ---- golden coefficients (kept in SV, mirrored into the DUT) -----
    byte w1_g [W1X];
    byte b1_g [B1X];
    byte w2_g [W2X];
    byte b2_g [B2X];
    byte w3_g [W3X];
    byte b3_g [B3X];

    // ---- scoreboard --------------------------------------------------
    bit [NF-1:0] inq [$];     // driven input vectors awaiting their output
    byte x_arr [NF];          // unpacked 0/1 input for the C model
    byte y_exp [1];           // expected int8 result
    int  shift_exp;           // expected fc3 shift

    integer errors, out_count, drive_count;
    integer cyc, first_out_cyc, last_out_cyc, burst_outs;
    integer i, n;
    reg     checking, burst_phase;

    // ---- DUT ---------------------------------------------------------
    mlp #(
        .N_FEATURES(NF), .HIDDEN1(H1), .HIDDEN2(H2), .DATA_WIDTH(DW)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .i_wr_en(wr_en), .i_wr_sel(wr_sel), .i_wr_addr(wr_addr), .i_wr_data(wr_data),
        .i_valid(in_valid), .i_x(x_vec),
        .o_valid(out_valid), .o_y(y_out), .o_shift(shift_out)
    );

    // ---- clock + free-running cycle counter --------------------------
    initial clk = 0;
    always #5 clk = ~clk;
    initial cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---- output monitor / checker ------------------------------------
    always @(posedge clk) begin : checker
        bit [NF-1:0] xp;
        byte gy;
        integer gs;
        if (checking && out_valid) begin
            if (inq.size() == 0) begin
                errors = errors + 1;
                $display("  ERROR: out_valid with empty scoreboard at cyc=%0d", cyc);
            end else begin
                xp = inq.pop_front();
                for (i = 0; i < NF; i = i + 1) x_arr[i] = xp[i] ? 8'sd1 : 8'sd0;
                mlp_cmodel(NF, H1, H2, x_arr,
                           w1_g, b1_g, w2_g, b2_g, w3_g, b3_g,
                           y_exp, shift_exp);
                gy = y_out;
                gs = shift_out;
                if (gy !== y_exp[0]) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("  MISMATCH y out#%0d  dut=%0d  cmodel=%0d",
                                 out_count, $signed(gy), y_exp[0]);
                end
                if (gs !== shift_exp) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("  MISMATCH shift out#%0d  dut=%0d  cmodel=%0d",
                                 out_count, gs, shift_exp);
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

    // ---- drive one vector at negedge; push to scoreboard if valid ----
    task drive_vec(input bit valid);
        integer j;
        begin
            if (valid) begin
                for (j = 0; j < NF; j = j + 1) x_vec[j] = $urandom & 1'b1;
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
                wr_sel  = sel[2:0];
                wr_addr = j[WADDR_W-1:0];
                case (sel)
                    3'd0: wr_data = w1_g[j];
                    3'd1: wr_data = b1_g[j];
                    3'd2: wr_data = w2_g[j];
                    3'd3: wr_data = b2_g[j];
                    3'd4: wr_data = w3_g[j];
                    default: wr_data = b3_g[j];
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
        $fsdbDumpfile("mlp.fsdb");
        $fsdbDumpvars(0, tb_mlp, "+mda");

        // random int8 coefficients
        for (i = 0; i < W1X; i = i + 1) w1_g[i] = byte'($urandom);
        for (i = 0; i < B1X; i = i + 1) b1_g[i] = byte'($urandom);
        for (i = 0; i < W2X; i = i + 1) w2_g[i] = byte'($urandom);
        for (i = 0; i < B2X; i = i + 1) b2_g[i] = byte'($urandom);
        for (i = 0; i < W3X; i = i + 1) w3_g[i] = byte'($urandom);
        for (i = 0; i < B3X; i = i + 1) b3_g[i] = byte'($urandom);

        // reset
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        checking = 1;

        // load coefficients through the write-only port (one per cycle)
        load_coeffs(3'd0, W1X);   // fc1.weight
        load_coeffs(3'd1, B1X);   // fc1.bias
        load_coeffs(3'd2, W2X);   // fc2.weight
        load_coeffs(3'd3, B2X);   // fc2.bias
        load_coeffs(3'd4, W3X);   // fc3.weight
        load_coeffs(3'd5, B3X);   // fc3.bias
        wr_en = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        @(negedge clk);

        // phase A: back-to-back burst (in_valid high every cycle => II = 1)
        burst_phase = 1;
        for (n = 0; n < NBURST; n = n + 1) drive_vec(1'b1);
        burst_phase = 0;

        // phase B: gapped vectors with random bubbles
        for (n = 0; n < NGAP; n = n + 1) begin
            drive_vec(1'b1);
            if (($urandom % 3) == 0) drive_vec(1'b0);   // insert a bubble
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
            $display("PASS: %0d vectors (NF=%0d, H1=%0d, H2=%0d) match C-model bit-for-bit "
                     "(y + shift); II=1 over %0d-beat burst.", out_count, NF, H1, H2, NBURST);
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
