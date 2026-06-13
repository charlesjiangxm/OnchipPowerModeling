/* =====================================================================
 * ft_transformer_cmodel.h
 *
 * Behavioral C reference model for the FULL int8 FT-Transformer forward pass
 * (FTTransformer.forward in src/models/ft_transformer.py), composing the four
 * per-block twins into one end-to-end network:
 *
 *   tokenizer -> (+cls) -> n_blocks x { (norm1) -> MHA -> +resid
 *                                       -> norm2 -> FFN -> +resid }
 *             -> final_norm(token 0) -> relu -> head(Linear D->1) -> scalar
 *
 * This is the GOLDEN composition spec the integrated RTL (hw/rtl/
 * ft_transformer_top.v) must match bit-for-bit. It calls the existing block
 * twins (numerical_feature_tokenizer_int8, layer_norm_int8,
 * multihead_attention_int8, feed_forward_network_int8) UNMODIFIED, and adds
 * the composition glue: the cls token, the residual adds, and the head.
 *
 * NUMERIC FORMAT (see the design plan):
 *   - The four block cores read/write strict Q1.frac (Q1.7) int8.
 *   - The RESIDUAL STREAM is carried at a coarser Q1.res_frac (Q3.5, range +-4)
 *     so the growing pre-norm stream does not clip. Conversions happen only at
 *     the seams (cls inject, block-0 MHA input, the two residual adds). The
 *     LayerNorm cores consume the Q3.5 stream directly (mean/var are
 *     scale-equivariant) and emit Q1.7; MHA/FFN only ever see Q1.7.
 *   - The HEAD emits the WIDE accumulator (signed int32 at Q(2*frac)=Q14), not
 *     a requantized int8 -- an int8 regression output would be far too coarse.
 *     Dequantize as y_float = y_out / 2^(2*frac), then Standardizer.inverse_y.
 *
 * NOTE the int8 path diverges from float PyTorch BY CONSTRUCTION (FFN uses
 * ReLU not GELU; softmax is the integer base-2 approximation; sqrt/reciprocal
 * are integer). The float model is an accuracy sanity check only -- the int8
 * composition here is the authoritative spec for the hardware.
 * =====================================================================
 */
#ifndef FT_TRANSFORMER_CMODEL_H
#define FT_TRANSFORMER_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- configuration (shared by the C model and the RTL parameters) ---- */
typedef struct {
    int F;            /* number of input features (RTL N_FEATURE)            */
    int seq_len;      /* 1 + F (RTL SEQ_LEN)                                 */
    int d_token;      /* embedding dim E (RTL D_TOKEN)                       */
    int d_ffn;        /* FFN hidden width (RTL D_FFN)                        */
    int n_heads;      /* attention heads H; head_dim HD = d_token/n_heads    */
    int n_blocks;     /* number of transformer blocks                       */
    int frac_bits;    /* int8 fractional bits for module I/O (Q1.frac, 7)    */
    int res_frac;     /* residual-stream fractional bits (Q1.res_frac, 5)    */
    int scale_frac;   /* MHA score-scale fractional bits (RTL SCALE_FRAC,14) */
    int sm_frac;      /* MHA softmax-input fractional bits (RTL SM_FRAC, 8)  */
    int recip_frac;   /* LN/MHA reciprocal fractional bits (RTL RECIP_FRAC)  */
    int out_frac;     /* LayerNorm output fractional bits (=frac_bits => Q1.7)*/
    long long scale;  /* round(2^scale_frac / sqrt(HD)); see mha_scale()     */
    long long eps_v;  /* LayerNorm integer epsilon for D=d_token; see        */
                      /*   layer_norm_eps_v()                                */
} ft_config;

/* ---- weights (all int8 unless noted; per-block arrays are block-major) ----
 * Block stride is the per-block element count; block b lives at b*stride.
 * For LayerNorm, norm1 of block 0 is unused (is_first skips it) but the slot
 * is still present in the array for a regular stride. */
typedef struct {
    const int8_t *tok_w;      /* [F*d_token]                 Q1.7, addr j*D+k  */
    const int8_t *tok_b;      /* [F*d_token]                 Q1.7, addr j*D+k  */
    const int8_t *cls;        /* [d_token]                   Q1.res_frac (Q3.5)*/
    const int8_t *norm1_g;    /* [n_blocks*d_token]          Q1.7 (b=0 unused) */
    const int8_t *norm1_b;    /* [n_blocks*d_token]          Q1.7 (b=0 unused) */
    const int8_t *mha_ipw;    /* [n_blocks*3*E*E]            Q1.7 Wq|Wk|Wv     */
    const int8_t *mha_ipb;    /* [n_blocks*3*E]              Q1.7 bq|bk|bv     */
    const int8_t *mha_opw;    /* [n_blocks*E*E]              Q1.7              */
    const int8_t *mha_opb;    /* [n_blocks*E]                Q1.7              */
    const int8_t *norm2_g;    /* [n_blocks*d_token]          Q1.7              */
    const int8_t *norm2_b;    /* [n_blocks*d_token]          Q1.7              */
    const int8_t *ffn_w1;     /* [n_blocks*d_ffn*d_token]    Q1.7 addr o*E+k   */
    const int8_t *ffn_b1;     /* [n_blocks*d_ffn]            Q1.7              */
    const int8_t *ffn_w2;     /* [n_blocks*d_token*d_ffn]    Q1.7 addr o*F+k   */
    const int8_t *ffn_b2;     /* [n_blocks*d_token]          Q1.7              */
    const int8_t *fnorm_g;    /* [d_token]                   Q1.7              */
    const int8_t *fnorm_b;    /* [d_token]                   Q1.7              */
    const int8_t *head_w;     /* [d_token]                   Q1.7              */
    int8_t        head_b;     /* scalar                      Q1.7              */
} ft_weights;

/* ---- composition-glue primitives (each has a 1:1 RTL twin) ----
 * Exposed so residual_add.v / head.v can be unit-checked over DPI too. */

/* Rescale an int8 value between fixed-point fractions, saturating to int8.
 *   from_frac > to_frac : round-half-up arithmetic right shift by the diff.
 *   from_frac < to_frac : left shift by the diff.
 *   from_frac == to_frac: identity.  */
int8_t ft_rescale(int8_t v, int from_frac, int to_frac);

/* One elementwise residual add: stream is Q1.res_frac, module_out is Q1.frac.
 * Aligns module_out down to res_frac (round-half-up) and adds, saturating to
 * int8 in the Q1.res_frac domain. */
int8_t ft_residual_add(int8_t stream, int8_t module_out, int res_frac, int frac_bits);

/* Head: relu the length-d_token Q1.frac vector, dot with head_w (Q1.frac),
 * add (head_b << frac_bits); return the WIDE accumulator (signed, Q(2*frac)).
 * No requant/shift -- the regression value keeps full precision. */
int32_t ft_head_int8(int d_token, int frac_bits,
                     const int8_t *x, const int8_t *head_w, int8_t head_b);

/* ---- full forward ----
 * x_feat : [F] int8 Q1.frac (a standardized + quantized feature row).
 * Returns the head's wide accumulator (signed int32 Q(2*frac_bits)). */
int32_t ft_transformer_int8(const ft_config *cfg, const ft_weights *w,
                            const int8_t *x_feat);

#ifdef __cplusplus
}
#endif

#endif /* FT_TRANSFORMER_CMODEL_H */
