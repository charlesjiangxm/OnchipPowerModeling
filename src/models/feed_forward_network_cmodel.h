/* =====================================================================
 * feed_forward_network_cmodel.h
 *
 * Behavioral C reference model for the int8 FT-Transformer position-wise
 * feed-forward network (TransformerBlock.ffn in src/models/ft_transformer.py):
 *
 *     nn.Sequential(nn.Linear(d_token, d_ffn), <act>, nn.Dropout,
 *                   nn.Linear(d_ffn, d_token))
 *
 * Dropout is identity at inference. ACTIVATION: the model uses GELU; this
 * hardware/model uses ReLU in its place (relu(h) = max(0, h)) -- a cheaper
 * sign-bit clamp with no LUT/multiplier.
 *
 * This is a PURE NUMERIC model: it has no notion of clocks, pipelines, valid
 * signals, or any other hardware detail. It computes exactly the same integer
 * arithmetic that hw/rtl/feed_forward_network.v computes, so the RTL output is
 * bit-for-bit identical to feed_forward_network_int8() for every element.
 *
 * The DPI-C glue that exposes this to a SystemVerilog testbench lives
 * separately in hw/verif/feed_forward_network_dpi.c -- nothing simulator-
 * specific leaks into this file.
 * =====================================================================
 */
#ifndef FEED_FORWARD_NETWORK_CMODEL_H
#define FEED_FORWARD_NETWORK_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pure integer feed-forward network over one length-d_token token.
 *
 *   x  : [d_token]          int8, read as signed Q1.frac fixed point
 *   w1 : [d_ffn*d_token]    Linear1.weight (d_ffn, d_token) row-major,
 *                           W1[o][k] = w1[o*d_token + k]
 *   b1 : [d_ffn]            Linear1.bias
 *   w2 : [d_token*d_ffn]    Linear2.weight (d_token, d_ffn) row-major,
 *                           W2[o][k] = w2[o*d_ffn + k]
 *   b2 : [d_token]          Linear2.bias
 *   y  : [d_token]          int8 output (Q1.frac), round-half-up + saturate
 *
 *   frac_bits : input/weight/output fractional bits (RTL FRAC_BITS, e.g. 7).
 *               Pass the SAME value the RTL was parameterized with.
 *
 * Algorithm (identical in hw/rtl/feed_forward_network.v):
 *   Linear1: acc = sum_k x[k]*W1[o][k] + (b1[o] << frac_bits);
 *            h[o] = relu( sat8( round_half_up(acc >> frac_bits) ) );  o in [0,d_ffn)
 *   Linear2: acc = sum_k h[k]*W2[o][k] + (b2[o] << frac_bits);
 *            y[o] =       sat8( round_half_up(acc >> frac_bits) );    o in [0,d_token)
 * ReLU forces negative Linear1 outputs to 0; it is exact (no requant).
 */
void feed_forward_network_int8(int d_token, int d_ffn, int frac_bits,
                               const int8_t *x,
                               const int8_t *w1, const int8_t *b1,
                               const int8_t *w2, const int8_t *b2,
                               int8_t *y);

#ifdef __cplusplus
}
#endif

#endif /* FEED_FORWARD_NETWORK_CMODEL_H */
