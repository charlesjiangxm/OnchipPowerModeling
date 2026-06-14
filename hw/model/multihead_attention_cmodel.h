/* =====================================================================
 * multihead_attention_cmodel.h
 *
 * Behavioral C reference model for the int8 FT-Transformer self-attention
 * block (nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads,
 * batch_first=True) used as self.attn(x, x, x, ...) in
 * src/models/ft_transformer.py).
 *
 * This is a PURE NUMERIC model: it has no notion of clocks, pipelines,
 * valid signals, or any other hardware detail. It computes exactly the
 * same integer arithmetic that hw/rtl/multihead_attention.v computes, so
 * the RTL output is bit-for-bit identical to multihead_attention_int8()
 * for every element.
 *
 * The DPI-C glue that exposes this to a SystemVerilog testbench lives
 * separately in hw/verif/utils/multihead_attention_dpi.c -- nothing simulator-
 * specific leaks into this file.
 * =====================================================================
 */
#ifndef MULTIHEAD_ATTENTION_CMODEL_H
#define MULTIHEAD_ATTENTION_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Integer 1/sqrt(head_dim) scale the RTL's SCALE parameter must share:
 *   SCALE = round(2^scale_frac / sqrt(d_token / n_heads)).
 * (e.g. d_token=32, n_heads=8 -> head_dim=4 -> 2^14/2 = 8192 at scale_frac=14.)
 * Pass the SAME value to multihead_attention_int8() and the RTL so results
 * match bit-for-bit. */
long long mha_scale(int n_heads, int d_token, int scale_frac);

/* Pure integer self-attention over one (S, E) example.
 *
 *   d_token (E)  : embedding dim   (RTL D_TOKEN)
 *   n_heads (H)  : number of heads (RTL N_HEADS); head_dim HD = E / H
 *   seq_len (S)  : sequence length = 1 + n_features (RTL SEQ_LEN)
 *   frac_bits    : int8 fractional bits, signed Q1.frac (RTL FRAC_BITS, e.g. 7)
 *   scale        : round(2^scale_frac / sqrt(HD))  (see mha_scale)
 *   scale_frac   : fractional bits of `scale`      (RTL SCALE_FRAC, e.g. 14)
 *   sm_frac      : fractional bits of the softmax-input score (RTL SM_FRAC, e.g. 8)
 *   recip_frac   : fractional bits of the row-sum reciprocal (RTL RECIP_FRAC, e.g. 24)
 *
 *   x            : [S*E] int8, row-major  x[s*E + e]            (q = k = v = x)
 *   in_proj_w    : [3E*E] int8, PyTorch in_proj_weight (3E, E), row-major
 *                  rows 0..E-1 = Wq, E..2E-1 = Wk, 2E..3E-1 = Wv;  W[row][col]=in_proj_w[row*E+col]
 *   in_proj_b    : [3E]   int8, in_proj_bias (bq | bk | bv)
 *   out_proj_w   : [E*E]  int8, out_proj.weight (E, E), row-major  Wo[oe][k]=out_proj_w[oe*E+k]
 *   out_proj_b   : [E]    int8, out_proj.bias
 *   y            : [S*E]  int8 output, row-major  y[s*E + e]
 *
 * Internals (all integer; identical in hw/rtl/multihead_attention.v):
 *   in_proj : Q/K/V[s][e] = sat8( rnd( sum_k x[s][k]*W[e][k] + (b[e]<<frac) ) >> frac )
 *   scores  : raw[h][qi][kj] = sum_d Qh[qi][d]*Kh[kj][d]            (Q2.2frac, kept wide)
 *   scale   : sm[kj] = rnd( raw*scale ) >> (2*frac + scale_frac - sm_frac)  (Q.sm_frac)
 *   softmax : e[kj] = exp(sm[kj]-max) in Q16 (base-2 + fitted quadratic),
 *             Se = sum e, inv = round(2^recip / Se)
 *   context : ctx[h][qi][d] = sat8( rnd( (sum_kj e[kj]*Vh[kj][d]) * inv ) >> recip )
 *   out_proj: y[s][e] = sat8( rnd( sum_k ctx[s][k]*Wo[e][k] + (bo[e]<<frac) ) >> frac )
 *
 * The exp fixed-point fractional width is fixed at 16 (EXP_FRAC), matching the
 * fitted-quadratic constants; see the .c file.
 */
void multihead_attention_int8(
    int d_token, int n_heads, int seq_len, int frac_bits,
    long long scale, int scale_frac, int sm_frac, int recip_frac,
    const int8_t *x,
    const int8_t *in_proj_w, const int8_t *in_proj_b,
    const int8_t *out_proj_w, const int8_t *out_proj_b,
    int8_t *y);

#ifdef __cplusplus
}
#endif

#endif /* MULTIHEAD_ATTENTION_CMODEL_H */
