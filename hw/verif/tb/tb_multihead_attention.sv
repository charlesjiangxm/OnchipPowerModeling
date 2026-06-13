// =====================================================================
// tb_multihead_attention.sv
//
// End-to-end self-checking testbench for hw/rtl/multihead_attention.v, run under
// VCS.
//   1. Loads random int8 weights (in_proj_weight/bias, out_proj.weight/bias) into
//      the DUT via the write port and keeps golden copies in SystemVerilog.
//   2. Streams random int8 sequences x_seq (q=k=v=x): a back-to-back burst (proves
//      II = 1) plus a gapped phase (random bubbles). Each driven sequence is
//      pushed to a scoreboard queue.
//   3. On every out_valid it pops the matching input, calls the behavioral C
//      reference model through DPI-C (multihead_attention_cmodel ->
//      multihead_attention_int8 in src/models/multihead_attention_cmodel.c), and
//      compares the int8 output element by element. The RTL and the C model run
//      the identical integer datapath, so the check is BIT-EXACT.
//   4. Dumps an FSDB (KDB written at compile via `-kdb`) so Verdi opens the
//      waveform with full source / code hierarchy.
//
// Build/run: see hw/verif/Makefile (VCS only).
//   make all_mha                         # compile + run, expect "PASS: ... II=1"
//   make verdi_mha                       # open multihead_attention.fsdb with KDB
//   make all_mha VCS_DEFINES=+define+MA_S=8
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

`ifndef MA_E
  `define MA_E 32
`endif
`ifndef MA_H
  `define MA_H 8
`endif
`ifndef MA_S
  `define MA_S 16
`endif

module tb_multihead_attention;

    // ---- parameters (must match the DUT instance) --------------------
    localparam int E          = `MA_E;   // d_token
    localparam int H          = `MA_H;   // n_heads
    localparam int S          = `MA_S;   // seq_len
    localparam int HD         = E / H;
    localparam int DW         = 8;       // int8
    localparam int FRAC       = 7;       // Q1.7
    localparam int SCALE_FRAC = 14;
    localparam int SM_FRAC    = 8;
    localparam int RECIP_FRAC = 24;
    // SCALE = round(2^SCALE_FRAC / sqrt(HD)); override with +define+MA_SCALE=...
`ifdef MA_SCALE
    localparam int SCALE = `MA_SCALE;
`else
    localparam int SCALE = int'((2.0 ** SCALE_FRAC) / $sqrt(real'(HD)) + 0.5);
`endif

    localparam int SX     = S * E;          // packed sequence elements
    localparam int IPWX   = 3 * E * E;       // in_proj_weight depth
    localparam int IPBX   = 3 * E;           // in_proj_bias depth
    localparam int OPWX   = E * E;           // out_proj.weight depth
    localparam int OPBX   = E;               // out_proj.bias depth

    localparam int NBURST = 24;   // back-to-back sequences (II = 1 proof)
    localparam int NGAP   = 24;   // gapped sequences (extra coverage)
    localparam int LAT    = 6;    // pipeline latency (informational; TB is latency-agnostic)
    localparam int WADDR_W = ($clog2(IPWX) < 1) ? 1 : $clog2(IPWX);

    // ---- DPI-C import: the behavioral reference model ----------------
    import "DPI-C" function void multihead_attention_cmodel(
        input  int     d_token,
        input  int     n_heads,
        input  int     seq_len,
        input  int     frac_bits,
        input  longint scale,
        input  int     scale_frac,
        input  int     sm_frac,
        input  int     recip_frac,
        input  byte    x   [SX],
        input  byte    ipw [IPWX],
        input  byte    ipb [IPBX],
        input  byte    opw [OPWX],
        input  byte    opb [OPBX],
        output byte    y   [SX]
    );

    // ---- DUT I/O -----------------------------------------------------
    reg                  clk, rst_n;
    reg                  wr_en;
    reg  [1:0]           wr_sel;
    reg  [WADDR_W-1:0]   wr_addr;
    reg  [DW-1:0]        wr_data;
    reg                  in_valid;
    reg  [SX*DW-1:0]     x_seq;
    wire                 out_valid;
    wire [SX*DW-1:0]     y_seq;

    // ---- golden coefficients (kept in SV, mirrored into the DUT) -----
    byte ipw_g [IPWX];
    byte ipb_g [IPBX];
    byte opw_g [OPWX];
    byte opb_g [OPBX];

    // ---- scoreboard --------------------------------------------------
    bit [SX*DW-1:0] inq [$];      // driven sequences awaiting their output
    byte x_arr [SX];
    byte y_exp [SX];

    integer errors, out_count, drive_count;
    integer cyc, first_out_cyc, last_out_cyc, burst_outs;
    integer i, n;
    reg     checking, burst_phase;

    // ---- DUT ---------------------------------------------------------
    multihead_attention #(
        .D_TOKEN(E), .N_HEADS(H), .SEQ_LEN(S), .DATA_WIDTH(DW),
        .FRAC_BITS(FRAC), .SCALE_FRAC(SCALE_FRAC), .SM_FRAC(SM_FRAC),
        .RECIP_FRAC(RECIP_FRAC), .SCALE(SCALE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
        .in_valid(in_valid), .x_seq(x_seq),
        .out_valid(out_valid), .y_seq(y_seq)
    );

    // ---- clock + free-running cycle counter --------------------------
    initial clk = 0;
    always #5 clk = ~clk;
    initial cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---- output monitor / checker ------------------------------------
    always @(posedge clk) begin : checker
        bit [SX*DW-1:0] xp;
        byte got;
        if (checking && out_valid) begin
            if (inq.size() == 0) begin
                errors = errors + 1;
                $display("  ERROR: out_valid with empty scoreboard at cyc=%0d", cyc);
            end else begin
                xp = inq.pop_front();
                for (i = 0; i < SX; i = i + 1) x_arr[i] = xp[i*DW +: DW];
                multihead_attention_cmodel(E, H, S, FRAC, longint'(SCALE),
                                           SCALE_FRAC, SM_FRAC, RECIP_FRAC,
                                           x_arr, ipw_g, ipb_g, opw_g, opb_g, y_exp);
                for (i = 0; i < SX; i = i + 1) begin
                    got = y_seq[i*DW +: DW];
                    if (got !== y_exp[i]) begin
                        errors = errors + 1;
                        if (errors <= 20)
                            $display("  MISMATCH out#%0d i=%0d (s=%0d,e=%0d)  dut=%0d  cmodel=%0d",
                                     out_count, i, i/E, i%E, $signed(got), y_exp[i]);
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

    // ---- drive one sequence at negedge; push to scoreboard if valid --
    task drive_seq(input bit valid);
        integer j;
        begin
            if (valid) begin
                for (j = 0; j < SX; j = j + 1) x_seq[j*DW +: DW] = byte'($urandom);
                in_valid = 1'b1;
                inq.push_back(x_seq);
                drive_count = drive_count + 1;
            end else begin
                in_valid = 1'b0;
                x_seq    = '0;
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
                    2'd0: wr_data = ipw_g[j];
                    2'd1: wr_data = ipb_g[j];
                    2'd2: wr_data = opw_g[j];
                    default: wr_data = opb_g[j];
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
        in_valid = 0; x_seq = '0; rst_n = 0;
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));

        // FSDB dump ("+mda" captures the unpacked weight memories / pipe arrays).
        $fsdbDumpfile("multihead_attention.fsdb");
        $fsdbDumpvars(0, tb_multihead_attention, "+mda");

        // random int8 weights
        for (i = 0; i < IPWX; i = i + 1) ipw_g[i] = byte'($urandom);
        for (i = 0; i < IPBX; i = i + 1) ipb_g[i] = byte'($urandom);
        for (i = 0; i < OPWX; i = i + 1) opw_g[i] = byte'($urandom);
        for (i = 0; i < OPBX; i = i + 1) opb_g[i] = byte'($urandom);

        // reset
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        checking = 1;

        // load weights through the write-only port (one coefficient per cycle)
        load_coeffs(2'd0, IPWX);   // in_proj_weight
        load_coeffs(2'd1, IPBX);   // in_proj_bias
        load_coeffs(2'd2, OPWX);   // out_proj.weight
        load_coeffs(2'd3, OPBX);   // out_proj.bias
        wr_en = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        @(negedge clk);

        // phase A: back-to-back burst (in_valid high every cycle => II = 1)
        burst_phase = 1;
        for (n = 0; n < NBURST; n = n + 1) drive_seq(1'b1);
        burst_phase = 0;

        // phase B: gapped sequences with random bubbles
        for (n = 0; n < NGAP; n = n + 1) begin
            drive_seq(1'b1);
            if (($urandom % 3) == 0) drive_seq(1'b0);   // insert a bubble
        end

        // drain the pipeline
        in_valid = 0; x_seq = '0;
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
            $display("PASS: %0d sequences (E=%0d, H=%0d, S=%0d) match C-model bit-for-bit; "
                     "II=1 over %0d-beat burst.", out_count, E, H, S, NBURST);
        else
            $display("FAIL: %0d total error(s).", errors);
        $display("----------------------------------------------------------");
        $finish;
    end

    // ---- safety timeout ----------------------------------------------
    initial begin
        #5000000;
        $display("FAIL: simulation timeout");
        $finish;
    end

endmodule
`default_nettype wire
