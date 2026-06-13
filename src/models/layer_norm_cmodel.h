/* =====================================================================
 * layer_norm_cmodel.h
 *
 * Behavioral C reference model for the int8 FT-Transformer LayerNorm
 * (nn.LayerNorm(d_token) in src/models/ft_transformer.py).
 *
 * This is a PURE NUMERIC model: it has no notion of clocks, pipelines,
 * valid signals, or any other hardware detail. It computes exactly the
 * same integer arithmetic that hw/rtl/layer_norm.v computes, so the RTL
 * output is bit-for-bit identical to layer_norm_int8() for every element.
 *
 * The DPI-C glue that exposes this to a SystemVerilog testbench lives
 * separately in hw/verif/utils/layer_norm_dpi.c -- nothing simulator-specific
 * leaks into this file.
 * =====================================================================
 */
#ifndef LAYER_NORM_CMODEL_H
#define LAYER_NORM_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Integer epsilon in the V = D^2 * var(int) domain, matching nn.LayerNorm's
 * float eps (default 1e-5):  EPS_V = round(eps * 2^(2*frac) * D^2).
 * Returned as the value the RTL's EPS_V parameter and the model must share. */
long long layer_norm_eps_v(int d_token, int frac_bits, double eps);

/* Pure integer LayerNorm over one length-D token.
 *
 *   x, gamma, beta : length-D int8 vectors, read as signed Q1.frac fixed point
 *   y              : length-D int8 output (Q1.out_frac), round-half-up + saturate
 *
 *   eps_v      : integer epsilon (see layer_norm_eps_v); pass the SAME value the
 *                RTL was parameterized with so results match bit-for-bit.
 *   recip_frac : fractional bits of the internal reciprocal (RTL RECIP_FRAC, e.g. 24)
 *   out_frac   : fractional bits of the int8 output (RTL OUT_FRAC; out_frac=frac_bits
 *                gives strict Q1.7 and saturates when |z*gamma+beta| >= 1)
 *
 * Algorithm (identical in hw/rtl/layer_norm.v):
 *   S=sum(x); SS=sum(x*x); V=D*SS-S*S; Veps=V+eps_v;
 *   r=floor(sqrt(Veps)) (clamped >=1); inv=round(2^recip_frac / r);
 *   per lane: num=D*x[i]-S; znorm=num*inv;
 *             acc=znorm*gamma[i] + (beta[i] << recip_frac);
 *             y[i]=sat8( round_half_up(acc >> (frac_bits+recip_frac-out_frac)) ).
 */
void layer_norm_int8(int d_token, int frac_bits, long long eps_v,
                     int recip_frac, int out_frac,
                     const int8_t *x, const int8_t *gamma, const int8_t *beta,
                     int8_t *y);

#ifdef __cplusplus
}
#endif

#endif /* LAYER_NORM_CMODEL_H */
