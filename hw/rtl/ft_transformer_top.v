// =====================================================================
// ft_transformer_top.v                                     (Verilog-2005*)
//
// Full FT-Transformer int8 inference datapath (FTTransformer.forward in
// src/models/ft_transformer.py), composing the four reusable int8 cores by
// TIME-MULTIPLEXING one physical instance of each across the N_BLOCKS
// transformer blocks. One input feature row is accepted per CYCLE_BUDGET
// cycles and one scalar y is produced after a fixed (latency-irrelevant) delay.
//
// Bit-exact twin: ft_transformer_int8() in src/models/ft_transformer_cmodel.c.
// The top mirrors that exact composition and numeric format:
//   * residual STREAM carried at Q1.RES_FRAC (Q3.5); cores stay Q1.FRAC_BITS.
//   * cls injected at row 0 (Q3.5); tokenizer rows rescaled Q1.7 -> Q3.5.
//   * block 0 is_first: MHA reads the stream rescaled Q3.5 -> Q1.7 (no norm1).
//   * LayerNorm reads the Q3.5 stream directly (scale-equivariant), emits Q1.7.
//   * residual adds done in parallel over the whole sequence (residual_add.v).
//   * head emits the WIDE accumulator (signed Q(2*FRAC_BITS)), not int8.
//
// Reuse: ONE tokenizer (1 bank), ONE multihead_attention (N_BLOCKS banks),
// ONE feed_forward_network (N_BLOCKS banks), ONE layer_norm (2*N_BLOCKS+1
// banks: norm1[b], norm2[b], final), ONE head, plus seq buffer, cls regfile,
// one residual_add, and the run FSM below. Per-token LayerNorm/FFN are streamed
// one of the SEQ_LEN tokens per cycle; norm2->FFN is wired through (II=1).
//
// Weight load: a single write port fans out to each sub-block's write port,
// selected by wr_layer; wr_bank chooses the resident bank. Load all
// coefficients once before asserting in_valid (see the packer / DPI TB).
//
// (*) Verilog-2005 dialect using always_ff/always_comb (compile -sverilog);
// reg/wire only; clk / rst_n (async assert, sync deassert).
// =====================================================================

`default_nettype none
module ft_transformer_top #(
    parameter F          = 16,   // input features
    parameter D_TOKEN    = 32,   // embedding dim E
    parameter D_FFN      = 64,   // FFN hidden width
    parameter N_HEADS    = 4,    // attention heads (HD = D_TOKEN/N_HEADS = 8)
    parameter N_BLOCKS   = 3,    // transformer blocks
    parameter DATA_WIDTH = 8,    // int8
    parameter FRAC_BITS  = 7,    // module I/O Q1.7
    parameter RES_FRAC   = 5,    // residual-stream Q3.5
    parameter SCALE_FRAC = 14,
    parameter SM_FRAC    = 8,
    parameter RECIP_FRAC = 24,
    parameter SCALE      = 5793, // round(2^SCALE_FRAC / sqrt(HD)); HD=8 -> 5793
    parameter EPS_V      = 168,  // round(1e-5 * 2^(2*FRAC) * D^2); D=32 -> 168
    parameter OUT_W      = 32,   // wide head output bus
    parameter CYCLE_BUDGET = 192,// informational: input cadence (>= compute depth ~163)
    // ---- derived (do not override) ----
    parameter SEQ_LEN    = 1 + F,                 // 17
    parameter N_LN_BANK  = 2*N_BLOCKS + 1,        // norm1[b], norm2[b], final
    parameter MHA_BANK_W = ($clog2(N_BLOCKS)   < 1) ? 1 : $clog2(N_BLOCKS),
    parameter LN_BANK_W  = ($clog2(N_LN_BANK)  < 1) ? 1 : $clog2(N_LN_BANK),
    parameter TOK_ADDR_W = ($clog2(F*D_TOKEN)  < 1) ? 1 : $clog2(F*D_TOKEN),
    parameter LN_ADDR_W  = ($clog2(D_TOKEN)    < 1) ? 1 : $clog2(D_TOKEN),
    parameter MHA_ADDR_W = ($clog2(3*D_TOKEN*D_TOKEN) < 1) ? 1 : $clog2(3*D_TOKEN*D_TOKEN),
    parameter FFN_ADDR_W = ($clog2(D_FFN*D_TOKEN)     < 1) ? 1 : $clog2(D_FFN*D_TOKEN),
    parameter TOP_ADDR_W = MHA_ADDR_W,            // widest write address
    parameter TOP_BANK_W = LN_BANK_W,             // widest bank index
    parameter SIDX_W     = ($clog2(SEQ_LEN+1) < 1) ? 1 : $clog2(SEQ_LEN+1)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- one-time coefficient load (fans out to each sub-block) ----------
    input  wire                          wr_en,
    input  wire [2:0]                     wr_layer,  // 0=TOK 1=LN 2=MHA 3=FFN 4=HEAD 5=CLS
    input  wire [TOP_BANK_W-1:0]          wr_bank,   // resident bank within the layer
    input  wire [1:0]                     wr_sel,    // MHA/FFN: wr_sel; LN/TOK/HEAD: bit0 = is_beta/is_bias
    input  wire [TOP_ADDR_W-1:0]          wr_addr,   // linear coeff index within selected array
    input  wire [DATA_WIDTH-1:0]          wr_data,   // signed int8 coefficient

    // ---- inference ----
    input  wire                          in_valid,  // start: x_row valid (when idle)
    input  wire [F*DATA_WIDTH-1:0]        x_row,     // packed int8 Q1.7 feature row
    output wire                          out_valid, // y valid (one-cycle pulse)
    output wire signed [OUT_W-1:0]        y          // wide signed Q(2*FRAC_BITS) result
);

    localparam W = DATA_WIDTH;

    // wr_layer codes
    localparam [2:0] LAYER_TOK=3'd0, LAYER_LN=3'd1, LAYER_MHA=3'd2,
                     LAYER_FFN=3'd3, LAYER_HEAD=3'd4, LAYER_CLS=3'd5;

    // LayerNorm bank assignment
    function [LN_BANK_W-1:0] ln_bank_norm1; input integer b; ln_bank_norm1 = b[LN_BANK_W-1:0]; endfunction
    function [LN_BANK_W-1:0] ln_bank_norm2; input integer b; ln_bank_norm2 = (N_BLOCKS + b); endfunction
    localparam [LN_BANK_W-1:0] LN_BANK_FINAL = 2*N_BLOCKS;

    // -------- int8 fixed-point rescales (match ft_rescale in the c-model) --
    localparam integer SH = FRAC_BITS - RES_FRAC;   // 2
    function signed [W-1:0] q17_to_q35;             // Q1.7 -> Q3.5, round-half-up >>SH, sat
        input signed [W-1:0] v;
        reg signed [W+4-1:0] t;
        begin
            t = ($signed(v) + (1 <<< (SH-1))) >>> SH;
            if      (t >  127) t =  127;
            else if (t < -128) t = -128;
            q17_to_q35 = t[W-1:0];
        end
    endfunction
    function signed [W-1:0] q35_to_q17;             // Q3.5 -> Q1.7, <<SH, sat
        input signed [W-1:0] v;
        reg signed [W+4-1:0] t;
        begin
            t = $signed(v) <<< SH;
            if      (t >  127) t =  127;
            else if (t < -128) t = -128;
            q35_to_q17 = t[W-1:0];
        end
    endfunction

    // =====================================================================
    // Storage
    // =====================================================================
    reg signed [W-1:0] seq [0:SEQ_LEN-1][0:D_TOKEN-1];  // residual stream, Q3.5
    reg signed [W-1:0] xb  [0:SEQ_LEN-1][0:D_TOKEN-1];  // MHA input, Q1.7
    reg signed [W-1:0] mb  [0:SEQ_LEN-1][0:D_TOKEN-1];  // MHA/FFN output, Q1.7
    reg signed [W-1:0] cls_reg  [0:D_TOKEN-1];          // learned cls token, Q3.5
    reg signed [W-1:0] cls_norm [0:D_TOKEN-1];          // final_norm(token 0), Q1.7
    reg        [F*W-1:0] x_lat;                         // latched input row

    // =====================================================================
    // FSM
    // =====================================================================
    localparam [3:0] S_IDLE=4'd0, S_TOK=4'd1, S_N1=4'd2, S_B0=4'd3, S_MHA=4'd4,
                     S_RES1=4'd5, S_N2F=4'd6, S_RES2=4'd7, S_FINAL=4'd8,
                     S_HEAD=4'd9, S_DONE=4'd10;
    reg [3:0]            state;
    reg [SIDX_W-1:0]     si;          // input index (driven into a core)
    reg [SIDX_W-1:0]     so;          // output index (collected from a core)
    reg [MHA_BANK_W-1:0] blk;         // current block 0..N_BLOCKS-1
    reg signed [OUT_W-1:0] y_q;
    reg                  ov_q;

    // -------- core wiring --------
    // tokenizer (single bank; original ports)
    wire                          tok_iv  = (state == S_TOK) && (si == 0);
    wire                          tok_ov;
    wire [F*D_TOKEN*W-1:0]        tok_out;

    // layer_norm (banked): used for norm1, norm2, final
    reg                           ln_iv;
    reg  [D_TOKEN*W-1:0]          ln_x;
    reg  [LN_BANK_W-1:0]          ln_bank;
    wire                          ln_ov;
    wire [D_TOKEN*W-1:0]          ln_y;

    // feed_forward_network (banked): fed by LN during S_N2F
    wire                          ffn_iv  = (state == S_N2F) ? ln_ov : 1'b0;
    wire [D_TOKEN*W-1:0]          ffn_x   = ln_y;
    wire                          ffn_ov;
    wire [D_TOKEN*W-1:0]          ffn_y;

    // multihead_attention (banked)
    wire                          mha_iv  = (state == S_MHA) && (si == 0);
    wire                          mha_ov;
    wire [SEQ_LEN*D_TOKEN*W-1:0]  xb_flat;
    wire [SEQ_LEN*D_TOKEN*W-1:0]  mha_y;

    // head
    wire                          head_iv = (state == S_HEAD) && (si == 0);
    wire [D_TOKEN*W-1:0]          cls_norm_flat;
    wire                          head_ov;
    wire signed [OUT_W-1:0]       head_y;

    // residual_add (combinational, whole sequence): out = seq + mb
    wire [SEQ_LEN*D_TOKEN*W-1:0]  seq_flat, mb_flat, ra_out;

    // flatten arrays for the whole-sequence cores
    genvar gs, ge;
    generate
        for (gs = 0; gs < SEQ_LEN; gs = gs + 1) begin : g_flat
            for (ge = 0; ge < D_TOKEN; ge = ge + 1) begin : g_flat_e
                assign xb_flat [(gs*D_TOKEN+ge)*W +: W] = xb [gs][ge];
                assign seq_flat[(gs*D_TOKEN+ge)*W +: W] = seq[gs][ge];
                assign mb_flat [(gs*D_TOKEN+ge)*W +: W] = mb [gs][ge];
            end
        end
        for (ge = 0; ge < D_TOKEN; ge = ge + 1) begin : g_clsn
            assign cls_norm_flat[ge*W +: W] = cls_norm[ge];
        end
    endgenerate

    // -------- LayerNorm input/bank mux (combinational) --------
    integer e0;
    always_comb begin
        ln_iv   = 1'b0;
        ln_bank = {LN_BANK_W{1'b0}};
        ln_x    = {D_TOKEN*W{1'b0}};
        case (state)
            S_N1: begin
                ln_iv   = (si < SEQ_LEN);
                ln_bank = ln_bank_norm1(blk);
                for (e0 = 0; e0 < D_TOKEN; e0 = e0 + 1)
                    ln_x[e0*W +: W] = seq[si][e0];     // Q3.5 stream token
            end
            S_N2F: begin
                ln_iv   = (si < SEQ_LEN);
                ln_bank = ln_bank_norm2(blk);
                for (e0 = 0; e0 < D_TOKEN; e0 = e0 + 1)
                    ln_x[e0*W +: W] = seq[si][e0];
            end
            S_FINAL: begin
                ln_iv   = (si == 0);
                ln_bank = LN_BANK_FINAL;
                for (e0 = 0; e0 < D_TOKEN; e0 = e0 + 1)
                    ln_x[e0*W +: W] = seq[0][e0];      // token 0 only
            end
            default: ;
        endcase
    end

    // =====================================================================
    // Core instances
    // =====================================================================
    numerical_feature_tokenizer #(
        .N_FEATURE(F), .D_TOKEN(D_TOKEN), .DATA_WIDTH(W), .FRAC_BITS(FRAC_BITS)
    ) u_tok (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en && (wr_layer == LAYER_TOK)),
        .wr_is_bias(wr_sel[0]),
        .wr_addr(wr_addr[TOK_ADDR_W-1:0]),
        .wr_data(wr_data),
        .in_valid(tok_iv), .x_row(x_lat),
        .out_valid(tok_ov), .out_tokens(tok_out)
    );

    layer_norm #(
        .D_TOKEN(D_TOKEN), .DATA_WIDTH(W), .FRAC_BITS(FRAC_BITS),
        .RECIP_FRAC(RECIP_FRAC), .OUT_FRAC(FRAC_BITS), .EPS_V(EPS_V),
        .N_BANK(N_LN_BANK)
    ) u_ln (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en && (wr_layer == LAYER_LN)),
        .wr_is_beta(wr_sel[0]),
        .wr_bank(wr_bank[LN_BANK_W-1:0]),
        .wr_addr(wr_addr[LN_ADDR_W-1:0]),
        .wr_data(wr_data),
        .bank_sel(ln_bank),
        .in_valid(ln_iv), .x_vec(ln_x),
        .out_valid(ln_ov), .y_vec(ln_y)
    );

    feed_forward_network #(
        .D_TOKEN(D_TOKEN), .D_FFN(D_FFN), .DATA_WIDTH(W), .FRAC_BITS(FRAC_BITS),
        .N_BANK(N_BLOCKS)
    ) u_ffn (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en && (wr_layer == LAYER_FFN)),
        .wr_sel(wr_sel),
        .wr_bank(wr_bank[MHA_BANK_W-1:0]),
        .wr_addr(wr_addr[FFN_ADDR_W-1:0]),
        .wr_data(wr_data),
        .bank_sel(blk),
        .in_valid(ffn_iv), .x_vec(ffn_x),
        .out_valid(ffn_ov), .y_vec(ffn_y)
    );

    multihead_attention #(
        .D_TOKEN(D_TOKEN), .N_HEADS(N_HEADS), .SEQ_LEN(SEQ_LEN), .DATA_WIDTH(W),
        .FRAC_BITS(FRAC_BITS), .SCALE_FRAC(SCALE_FRAC), .SM_FRAC(SM_FRAC),
        .RECIP_FRAC(RECIP_FRAC), .SCALE(SCALE), .N_BANK(N_BLOCKS)
    ) u_mha (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en && (wr_layer == LAYER_MHA)),
        .wr_sel(wr_sel),
        .wr_bank(wr_bank[MHA_BANK_W-1:0]),
        .wr_addr(wr_addr[MHA_ADDR_W-1:0]),
        .wr_data(wr_data),
        .bank_sel(blk),
        .in_valid(mha_iv), .x_seq(xb_flat),
        .out_valid(mha_ov), .y_seq(mha_y)
    );

    head #(
        .D_TOKEN(D_TOKEN), .DATA_WIDTH(W), .FRAC_BITS(FRAC_BITS), .OUT_W(OUT_W)
    ) u_head (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en && (wr_layer == LAYER_HEAD)),
        .wr_is_bias(wr_sel[0]),
        .wr_addr(wr_addr[LN_ADDR_W-1:0]),
        .wr_data(wr_data),
        .in_valid(head_iv), .x_vec(cls_norm_flat),
        .out_valid(head_ov), .y_out(head_y)
    );

    residual_add #(
        .VEC_LEN(SEQ_LEN*D_TOKEN), .DATA_WIDTH(W),
        .FRAC_BITS(FRAC_BITS), .RES_FRAC(RES_FRAC)
    ) u_resid (
        .stream_vec(seq_flat), .module_vec(mb_flat), .out_vec(ra_out)
    );

    // =====================================================================
    // cls regfile load (Q3.5)
    // =====================================================================
    always_ff @(posedge clk) begin
        if (wr_en && (wr_layer == LAYER_CLS))
            cls_reg[wr_addr[LN_ADDR_W-1:0]] <= wr_data;
    end

    // =====================================================================
    // Sequencer
    // =====================================================================
    integer s, e;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; si <= 0; so <= 0; blk <= 0; ov_q <= 1'b0;
        end else begin
            ov_q <= 1'b0;
            case (state)
                // ---- wait for an input, latch it, kick the tokenizer ----
                S_IDLE: begin
                    if (in_valid) begin
                        x_lat <= x_row;
                        si <= 0; so <= 0; blk <= 0;
                        state <= S_TOK;
                    end
                end

                // ---- tokenize, then build the Q3.5 stream (cls + tokens) ----
                S_TOK: begin
                    si <= 1;                          // one-shot in_valid (si==0)
                    if (tok_ov) begin
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            seq[0][e] <= cls_reg[e];  // row 0 = cls (Q3.5)
                        for (s = 1; s < SEQ_LEN; s = s + 1)
                            for (e = 0; e < D_TOKEN; e = e + 1)
                                seq[s][e] <= q17_to_q35(
                                    $signed(tok_out[((s-1)*D_TOKEN+e)*W +: W]));
                        si <= 0; so <= 0;
                        state <= S_B0;                // block 0 skips norm1
                    end
                end

                // ---- block 0 (is_first): xb = rescale(seq) Q3.5 -> Q1.7 ----
                S_B0: begin
                    for (s = 0; s < SEQ_LEN; s = s + 1)
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            xb[s][e] <= q35_to_q17(seq[s][e]);
                    si <= 0;
                    state <= S_MHA;
                end

                // ---- norm1 (blocks > 0): stream seq -> LN -> xb ----
                S_N1: begin
                    if (si < SEQ_LEN) si <= si + 1'b1;
                    if (ln_ov) begin
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            xb[so][e] <= $signed(ln_y[e*W +: W]);
                        so <= so + 1'b1;
                        if (so == SEQ_LEN-1) begin si <= 0; state <= S_MHA; end
                    end
                end

                // ---- attention: present whole seq once, capture y_seq ----
                S_MHA: begin
                    si <= 1;                          // one-shot in_valid (si==0)
                    if (mha_ov) begin
                        for (s = 0; s < SEQ_LEN; s = s + 1)
                            for (e = 0; e < D_TOKEN; e = e + 1)
                                mb[s][e] <= $signed(mha_y[(s*D_TOKEN+e)*W +: W]);
                        state <= S_RES1;
                    end
                end

                // ---- residual #1: seq += mha_out (parallel) ----
                S_RES1: begin
                    for (s = 0; s < SEQ_LEN; s = s + 1)
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            seq[s][e] <= $signed(ra_out[(s*D_TOKEN+e)*W +: W]);
                    si <= 0; so <= 0;
                    state <= S_N2F;
                end

                // ---- norm2 -> FFN chained stream -> mb ----
                S_N2F: begin
                    if (si < SEQ_LEN) si <= si + 1'b1;
                    if (ffn_ov) begin
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            mb[so][e] <= $signed(ffn_y[e*W +: W]);
                        so <= so + 1'b1;
                        if (so == SEQ_LEN-1) state <= S_RES2;
                    end
                end

                // ---- residual #2: seq += ffn_out; next block or finish ----
                S_RES2: begin
                    for (s = 0; s < SEQ_LEN; s = s + 1)
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            seq[s][e] <= $signed(ra_out[(s*D_TOKEN+e)*W +: W]);
                    si <= 0; so <= 0;
                    if (blk == N_BLOCKS-1) state <= S_FINAL;
                    else begin blk <= blk + 1'b1; state <= S_N1; end
                end

                // ---- final_norm on token 0 -> cls_norm ----
                S_FINAL: begin
                    si <= 1;                          // one-shot in_valid (si==0)
                    if (ln_ov) begin
                        for (e = 0; e < D_TOKEN; e = e + 1)
                            cls_norm[e] <= $signed(ln_y[e*W +: W]);
                        si <= 0;
                        state <= S_HEAD;
                    end
                end

                // ---- head -> wide scalar ----
                S_HEAD: begin
                    si <= 1;                          // one-shot in_valid (si==0)
                    if (head_ov) begin
                        y_q  <= head_y;
                        ov_q <= 1'b1;
                        state <= S_DONE;
                    end
                end

                S_DONE: state <= S_IDLE;              // ready for the next input
                default: state <= S_IDLE;
            endcase
        end
    end

    assign out_valid = ov_q;
    assign y         = y_q;

endmodule
`default_nettype wire
