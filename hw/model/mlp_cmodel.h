/* =====================================================================
 * mlp_cmodel.h
 *
 * Behavioral C reference model for the int8 MLP accelerator that realizes
 * SmallMLP.forward in src/models/mlp.py:
 *
 *     h = relu(fc1(x)); h = relu(fc2(h)); y = fc3(h).squeeze(-1)
 *     fc1: Linear(n_features, hidden1)   fc2: Linear(hidden1, hidden2)
 *     fc3: Linear(hidden2, 1)            (dropout is identity at inference)
 *
 * HARDWARE NUMERIC MODEL (confirmed with the user):
 *   - x is a 1-BIT, UNIPOLAR {0,1} vector, so fc1 is a gated adder tree (no
 *     multipliers): h1[j] = b1[j] + sum_i ( x[i] ? W1[j][i] : 0 ).
 *   - Weights and biases are signed int8; bias is added DIRECTLY to the
 *     accumulator (pure integer arithmetic, no fractional alignment / FRAC_BITS).
 *   - After each VMM the full-precision result is DYNAMICALLY (block-floating-
 *     point) requantized back to int8: one shared right-shift per result vector,
 *       s = max(0, bitlen(max_i |v_i|) - (OUT_BITS-1)),
 *     then q_i = sat8( RNE(v_i >> s) ).  Rounding is ROUND-TO-NEAREST,
 *     TIES-TO-EVEN.  OUT_BITS is 8 (int8).
 *   - fc1 / fc2 apply ReLU to the full-precision accumulator BEFORE the dynamic
 *     requant (so the shift is derived from the post-ReLU, non-negative vector);
 *     fc3 has no activation and is requantized from the raw signed accumulator.
 *   - The final scalar y is int8; *shift returns fc3's dynamic shift so the host
 *     can recover the integer VMM3 magnitude (y << shift ~= acc3).
 *
 * This is a PURE NUMERIC model: no clocks, pipelines, valid signals, or sim
 * detail. It computes exactly the integer arithmetic hw/rtl/mlp.v computes, so
 * the RTL output is bit-for-bit identical to mlp_int8() (the SV/DPI testbench
 * checks both y and shift). The DPI-C glue lives separately in
 * hw/verif/utils/mlp_dpi.c -- nothing simulator-specific leaks into this file.
 * =====================================================================
 */
#ifndef MLP_CMODEL_H
#define MLP_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pure integer 3-layer MLP over one 1-bit input vector.
 *
 *   x  : [n_features]          1-bit input, each element 0 or 1
 *   w1 : [hidden1*n_features]  fc1.weight (hidden1, n_features) row-major,
 *                              W1[j][i] = w1[j*n_features + i]
 *   b1 : [hidden1]             fc1.bias
 *   w2 : [hidden2*hidden1]     fc2.weight (hidden2, hidden1) row-major,
 *                              W2[k][i] = w2[k*hidden1 + i]
 *   b2 : [hidden2]             fc2.bias
 *   w3 : [hidden2]             fc3.weight (1, hidden2) row-major, W3[i] = w3[i]
 *   b3 : [1]                   fc3.bias
 *   y     : [1] int8 output (round-to-nearest-ties-to-even + saturate)
 *   shift : [1] fc3 dynamic right-shift (so acc3 ~= (*y) << (*shift))
 *
 * All accumulators are int64 and sized never to truncate, so the integer sums
 * match the RTL's exact-width adders regardless of order.  Algorithm (identical
 * in hw/rtl/mlp.v):
 *   fc1: a1[j] = relu( b1[j] + sum_i ( x[i] ? W1[j][i] : 0 ) );   j in [0,hidden1)
 *        s1 = dyn_shift(a1, hidden1);  h1[j] = rne_sat(a1[j], s1);
 *   fc2: a2[k] = relu( b2[k] + sum_i h1[i]*W2[k][i] );            k in [0,hidden2)
 *        s2 = dyn_shift(a2, hidden2);  h2[k] = rne_sat(a2[k], s2);
 *   fc3: a3   =        b3[0] + sum_i h2[i]*W3[i];
 *        s3 = dyn_shift(&a3, 1);       *y = rne_sat(a3, s3);  *shift = s3;
 */
void mlp_int8(int n_features, int hidden1, int hidden2,
              const int8_t *x,
              const int8_t *w1, const int8_t *b1,
              const int8_t *w2, const int8_t *b2,
              const int8_t *w3, const int8_t *b3,
              int8_t *y, int *shift);

#ifdef __cplusplus
}
#endif

#endif /* MLP_CMODEL_H */
