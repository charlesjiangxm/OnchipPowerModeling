// =====================================================================
// tb_ft_transformer.sv
//
// End-to-end self-checking testbench for hw/rtl/ft_transformer_top.v, run under
// VCS. The integrated top composes the four banked cores + cls/residual/head
// glue; this TB proves it equals the full-model C reference bit-for-bit.
//   1. Generates random int8 coefficients for every logical array (tokenizer,
//      cls, the 6 LayerNorms, per-block MHA + FFN, final_norm, head) and keeps
//      golden copies in SystemVerilog.
//   2. Loads them into the DUT through the unified write port in the canonical
//      order (wr_layer / wr_bank / wr_sel / wr_addr), matching
//      script/pack_ft_transformer_int8.py.
//   3. Drives random input feature rows. One input is processed at a time
//      (latency-irrelevant reuse design); the next is issued after out_valid.
//   4. On each out_valid, calls ft_transformer_cmodel() over DPI-C on the same
//      coefficients + input and checks the WIDE int32 result bit-for-bit. The
//      RTL and C model run the identical integer datapath.
//
// Build/run: see hw/verif/Makefile.
//   make all_ft                          # compile + run, expect "PASS"
//   make verdi_ft                        # open ft_transformer.fsdb with KDB
//   make run_ft SIMARGS=+seed=12345
//
// NOTE the bit-exact check is independent of quantization semantics: identical
// int8 bytes feed both sides, so random coefficients suffice (the cls "Q3.5"
// vs weights "Q1.7" distinction only matters for accuracy on a trained model).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_ft_transformer;

    // ---- parameters (must match the DUT instance) --------------------
    localparam int F          = 16;
    localparam int E          = 32;   // d_token
    localparam int DF         = 64;   // d_ffn
    localparam int H          = 4;    // n_heads (HD = 8)
    localparam int NB         = 3;    // n_blocks
    localparam int S          = 1 + F;// seq_len = 17
    localparam int DW         = 8;
    localparam int FRAC       = 7;
    localparam int RES_FRAC   = 5;
    localparam int SCALE_FRAC = 14;
    localparam int SM_FRAC    = 8;
    localparam int RECIP_FRAC = 24;
    localparam int SCALE      = 5793; // round(2^14/sqrt(8))
    localparam int EPS_V      = 168;  // round(1e-5*2^14*32^2)
    localparam int OUT_W      = 32;

    // wr_layer codes (match ft_transformer_top.v / the packer)
    localparam int LAYER_TOK=0, LAYER_LN=1, LAYER_MHA=2, LAYER_FFN=3,
                   LAYER_HEAD=4, LAYER_CLS=5;

    // ---- array sizes ----
    localparam int TOKX = F*E;
    localparam int N_E  = NB*E;            // per-block LayerNorm gamma/beta store
    localparam int IPWX = NB*3*E*E, IPBX = NB*3*E, OPWX = NB*E*E, OPBX = NB*E;
    localparam int W1X  = NB*DF*E,  B1X  = NB*DF,  W2X  = NB*E*DF, B2X  = NB*E;

    localparam int NSAMPLES = 8;       // random inputs to check
    localparam int TOP_ADDR_W = ($clog2(3*E*E) < 1) ? 1 : $clog2(3*E*E);
    localparam int TOP_BANK_W = ($clog2(2*NB+1) < 1) ? 1 : $clog2(2*NB+1);

    // ---- DPI-C import: the full-model reference -----------------------
    import "DPI-C" function void ft_transformer_cmodel(
        input  int     F, input int seq_len, input int d_token, input int d_ffn,
        input  int     n_heads, input int n_blocks, input int frac_bits,
        input  int     res_frac, input int scale_frac, input int sm_frac,
        input  int     recip_frac, input int out_frac,
        input  longint scale, input longint eps_v,
        input  byte    tok_w [TOKX], input byte tok_b [TOKX], input byte cls [E],
        input  byte    n1g [N_E], input byte n1b [N_E],
        input  byte    ipw [IPWX], input byte ipb [IPBX],
        input  byte    opw [OPWX], input byte opb [OPBX],
        input  byte    n2g [N_E], input byte n2b [N_E],
        input  byte    w1 [W1X], input byte b1 [B1X],
        input  byte    w2 [W2X], input byte b2 [B2X],
        input  byte    fng [E], input byte fnb [E],
        input  byte    hw [E], input byte hb,
        input  byte    x_feat [F],
        output int     y
    );

    // ---- DUT I/O -----------------------------------------------------
    reg                     clk, rst_n;
    reg                     wr_en;
    reg  [2:0]              wr_layer;
    reg  [TOP_BANK_W-1:0]   wr_bank;
    reg  [1:0]              wr_sel;
    reg  [TOP_ADDR_W-1:0]   wr_addr;
    reg  [DW-1:0]           wr_data;
    reg                     in_valid;
    reg  [F*DW-1:0]         x_row;
    wire                    out_valid;
    wire signed [OUT_W-1:0] y;

    // ---- golden coefficients (kept in SV) ----------------------------
    byte tok_w_g [TOKX], tok_b_g [TOKX], cls_g [E];
    byte n1g_g [N_E], n1b_g [N_E], n2g_g [N_E], n2b_g [N_E];
    byte ipw_g [IPWX], ipb_g [IPBX], opw_g [OPWX], opb_g [OPBX];
    byte w1_g [W1X], b1_g [B1X], w2_g [W2X], b2_g [B2X];
    byte fng_g [E], fnb_g [E], hw_g [E], hb_g;

    byte x_arr [F];
    integer errors, checked, i, b, k;

    // ---- DUT ---------------------------------------------------------
    ft_transformer_top #(
        .F(F), .D_TOKEN(E), .D_FFN(DF), .N_HEADS(H), .N_BLOCKS(NB),
        .DATA_WIDTH(DW), .FRAC_BITS(FRAC), .RES_FRAC(RES_FRAC),
        .SCALE_FRAC(SCALE_FRAC), .SM_FRAC(SM_FRAC), .RECIP_FRAC(RECIP_FRAC),
        .SCALE(SCALE), .EPS_V(EPS_V), .OUT_W(OUT_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_layer(wr_layer), .wr_bank(wr_bank), .wr_sel(wr_sel),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .in_valid(in_valid), .x_row(x_row),
        .out_valid(out_valid), .y(y)
    );

    // ---- clock -------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // ---- one coefficient write per cycle -----------------------------
    task wr1(input int layer, input int bank, input int sel, input int addr,
             input byte data);
        begin
            wr_en    = 1'b1;
            wr_layer = layer[2:0];
            wr_bank  = bank[TOP_BANK_W-1:0];
            wr_sel   = sel[1:0];
            wr_addr  = addr[TOP_ADDR_W-1:0];
            wr_data  = data;
            @(negedge clk);
        end
    endtask

    // load a contiguous slice arr[off +: depth] into (layer,bank,sel) addr 0..
    task load_arr(input int layer, input int bank, input int sel,
                  const ref byte arr [], input int off, input int depth);
        integer j;
        begin
            for (j = 0; j < depth; j = j + 1)
                wr1(layer, bank, sel, j, arr[off + j]);
        end
    endtask

    // ---- drive one input and check the result ------------------------
    task drive_and_check(input [F*DW-1:0] xr);
        int y_exp;
        begin
            @(negedge clk); in_valid = 1'b1; x_row = xr;
            @(negedge clk); in_valid = 1'b0; x_row = '0;
            // wait for the (single in-flight) result
            do @(posedge clk); while (out_valid !== 1'b1);
            for (k = 0; k < F; k = k + 1) x_arr[k] = xr[k*DW +: DW];
            ft_transformer_cmodel(F, S, E, DF, H, NB, FRAC, RES_FRAC, SCALE_FRAC,
                SM_FRAC, RECIP_FRAC, FRAC, longint'(SCALE), longint'(EPS_V),
                tok_w_g, tok_b_g, cls_g, n1g_g, n1b_g, ipw_g, ipb_g, opw_g, opb_g,
                n2g_g, n2b_g, w1_g, b1_g, w2_g, b2_g, fng_g, fnb_g, hw_g, hb_g,
                x_arr, y_exp);
            if (y !== y_exp) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  MISMATCH sample#%0d  dut=%0d  cmodel=%0d",
                             checked, y, y_exp);
            end
            checked = checked + 1;
        end
    endtask

    // ---- stimulus ----------------------------------------------------
    initial begin : stim
        int seed;
        errors = 0; checked = 0;
        wr_en = 0; wr_layer = 0; wr_bank = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        in_valid = 0; x_row = '0; rst_n = 0;
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));

        $fsdbDumpfile("ft_transformer.fsdb");
        $fsdbDumpvars(0, tb_ft_transformer, "+mda");

        // random int8 coefficients
        for (i = 0; i < TOKX; i = i + 1) begin tok_w_g[i]=byte'($urandom); tok_b_g[i]=byte'($urandom); end
        for (i = 0; i < E;    i = i + 1) begin cls_g[i]=byte'($urandom); fng_g[i]=byte'($urandom);
                                              fnb_g[i]=byte'($urandom); hw_g[i]=byte'($urandom); end
        hb_g = byte'($urandom);
        for (i = 0; i < N_E;  i = i + 1) begin n1g_g[i]=byte'($urandom); n1b_g[i]=byte'($urandom);
                                              n2g_g[i]=byte'($urandom); n2b_g[i]=byte'($urandom); end
        for (i = 0; i < IPWX; i = i + 1) ipw_g[i]=byte'($urandom);
        for (i = 0; i < IPBX; i = i + 1) ipb_g[i]=byte'($urandom);
        for (i = 0; i < OPWX; i = i + 1) opw_g[i]=byte'($urandom);
        for (i = 0; i < OPBX; i = i + 1) opb_g[i]=byte'($urandom);
        for (i = 0; i < W1X;  i = i + 1) w1_g[i]=byte'($urandom);
        for (i = 0; i < B1X;  i = i + 1) b1_g[i]=byte'($urandom);
        for (i = 0; i < W2X;  i = i + 1) w2_g[i]=byte'($urandom);
        for (i = 0; i < B2X;  i = i + 1) b2_g[i]=byte'($urandom);

        // reset
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // ---- load all coefficients (canonical order) ----
        load_arr(LAYER_TOK, 0, 0, tok_w_g, 0, TOKX);    // tokenizer weight
        load_arr(LAYER_TOK, 0, 1, tok_b_g, 0, TOKX);    // tokenizer bias
        load_arr(LAYER_CLS, 0, 0, cls_g, 0, E);         // cls token (Q3.5)
        for (b = 0; b < NB; b = b + 1) begin
            if (b >= 1) begin                            // block 0 skips norm1
                load_arr(LAYER_LN, b, 0, n1g_g, b*E, E);
                load_arr(LAYER_LN, b, 1, n1b_g, b*E, E);
            end
            load_arr(LAYER_LN, NB+b, 0, n2g_g, b*E, E);
            load_arr(LAYER_LN, NB+b, 1, n2b_g, b*E, E);
        end
        load_arr(LAYER_LN, 2*NB, 0, fng_g, 0, E);        // final_norm
        load_arr(LAYER_LN, 2*NB, 1, fnb_g, 0, E);
        for (b = 0; b < NB; b = b + 1) begin
            load_arr(LAYER_MHA, b, 0, ipw_g, b*3*E*E, 3*E*E);
            load_arr(LAYER_MHA, b, 1, ipb_g, b*3*E,   3*E);
            load_arr(LAYER_MHA, b, 2, opw_g, b*E*E,   E*E);
            load_arr(LAYER_MHA, b, 3, opb_g, b*E,     E);
        end
        for (b = 0; b < NB; b = b + 1) begin
            load_arr(LAYER_FFN, b, 0, w1_g, b*DF*E, DF*E);
            load_arr(LAYER_FFN, b, 1, b1_g, b*DF,   DF);
            load_arr(LAYER_FFN, b, 2, w2_g, b*E*DF, E*DF);
            load_arr(LAYER_FFN, b, 3, b2_g, b*E,    E);
        end
        load_arr(LAYER_HEAD, 0, 0, hw_g, 0, E);          // head weight
        wr1(LAYER_HEAD, 0, 1, 0, hb_g);                  // head bias (scalar)
        wr_en = 0; wr_layer = 0; wr_bank = 0; wr_sel = 0; wr_addr = 0; wr_data = 0;
        @(negedge clk);

        // ---- drive random inputs ----
        for (i = 0; i < NSAMPLES; i = i + 1) begin
            for (k = 0; k < F; k = k + 1) x_row[k*DW +: DW] = byte'($urandom);
            drive_and_check(x_row);
        end

        // ---- report ----
        $display("----------------------------------------------------------");
        if (errors == 0)
            $display("PASS: %0d FT-Transformer inferences (F=%0d,E=%0d,DF=%0d,H=%0d,NB=%0d) "
                     "match the C model bit-for-bit.", checked, F, E, DF, H, NB);
        else
            $display("FAIL: %0d error(s) over %0d inferences.", errors, checked);
        $display("----------------------------------------------------------");
        $finish;
    end

    // ---- safety timeout ----------------------------------------------
    initial begin
        #50000000;
        $display("FAIL: simulation timeout");
        $finish;
    end

endmodule
`default_nettype wire
