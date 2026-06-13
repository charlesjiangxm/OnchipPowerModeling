/* =====================================================================
 * numerical_feature_tokenizer_cmodel.h
 *
 * Behavioral C reference model for the int8 FT-Transformer
 * NumericalFeatureTokenizer (src/models/ft_transformer.py):
 *
 *     out = x.unsqueeze(-1) * weight.unsqueeze(0) + bias.unsqueeze(0)
 *     #  elementwise:  out[j][k] = x[j] * weight[j][k] + bias[j][k]
 *
 * Purely element-wise broadcast multiply-add -- no reduction across features.
 * This is a PURE NUMERIC model: no clocks, pipelines, valid signals, or other
 * hardware detail. It computes exactly the integer arithmetic that
 * hw/rtl/numerical_feature_tokenizer.v computes, so the RTL output is
 * bit-for-bit identical to numerical_feature_tokenizer_int8() for every
 * element. (The cycle-accurate pipeline twin lives in hw/model/ref_model.py;
 * this C twin closes the gap so the tokenizer can join the DPI flow and be
 * called from ft_transformer_cmodel.c.)
 * =====================================================================
 */
#ifndef NUMERICAL_FEATURE_TOKENIZER_CMODEL_H
#define NUMERICAL_FEATURE_TOKENIZER_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pure integer numerical-feature tokenizer.
 *
 *   n_feature (F) : number of features (rows of weight/bias)  (RTL N_FEATURE)
 *   d_token   (D) : token dimension    (cols of weight/bias)  (RTL D_TOKEN)
 *   frac_bits     : int8 fractional bits, signed Q1.frac      (RTL FRAC_BITS, e.g. 7)
 *
 *   x      : [F]    int8, one quantized feature value per feature (Q1.frac)
 *   weight : [F*D]  int8, weight (F, D) row-major  W[j][k] = weight[j*D + k]
 *   bias   : [F*D]  int8, bias   (F, D) row-major  b[j][k] = bias[j*D + k]
 *   out    : [F*D]  int8 output (Q1.frac), row-major  out[j][k] = out[j*D + k]
 *
 * Algorithm (identical in hw/rtl/numerical_feature_tokenizer.v):
 *   acc      = x[j]*W[j][k] + (b[j][k] << frac_bits);
 *   out[j][k]= sat8( round_half_up(acc >> frac_bits) ).
 */
void numerical_feature_tokenizer_int8(int n_feature, int d_token, int frac_bits,
                                      const int8_t *x,
                                      const int8_t *weight, const int8_t *bias,
                                      int8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* NUMERICAL_FEATURE_TOKENIZER_CMODEL_H */
