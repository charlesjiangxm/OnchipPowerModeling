/* =====================================================================
 * feed_forward_network_cmodel.c
 *
 * Behavioral C reference model for the int8 FT-Transformer position-wise
 * feed-forward network. See feed_forward_network_cmodel.h for the contract.
 * This file is the GOLDEN model: hw/rtl/feed_forward_network.v runs the
 * identical integer datapath (ReLU activation), so the RTL output equals
 * feed_forward_network_int8() bit-for-bit (the SV/DPI testbench checks this).
 *
 * Pure math only -- no clocks, pipelines, valid signals, or sim/DPI detail.
 * Every accumulator is int64 and sized to never truncate, so the (associative)
 * integer sums match the RTL's exact-width adders regardless of order.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DFFN_STANDALONE -O2 -std=c11 src/models/feed_forward_network_cmodel.c -o /tmp/ffncm -lm
 *     /tmp/ffncm
 *   -> cross-checks the integer model against an INDEPENDENT fixed-point
 *      reference (double-precision accumulation + floor-requant, same int8
 *      quantization points). They must agree bit-for-bit; this catches a wrong
 *      shift / sign / weight layout. (A "full-precision ideal" is not a useful
 *      reference for random int8 weights: the Q1.7 datapath saturates almost
 *      everywhere, so that residual measures the Q1.7 range limit, not a bug.)
 *      The ultimate gate is the bit-exact RTL-vs-C-model DPI testbench.
 * =====================================================================
 */
#include "feed_forward_network_cmodel.h"

#include <stdint.h>

/* The round-half-up requant relies on an arithmetic (sign-replicating) right
 * shift of a negative signed value, which is what GCC/Clang -- the compilers
 * VCS uses for DPI -- implement, and what Verilog `>>>` on a signed reg does.
 * Fail loudly at compile time if that ever stops holding. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "feed_forward_network_cmodel requires arithmetic right shift on int64_t");
#endif

/* compile-time scratch bound for the hidden vector (generous; override -D) */
#ifndef FFN_MAX_DFFN
#define FFN_MAX_DFFN 2048
#endif

/* round-half-up, arithmetic right shift by `shift`, saturate to int8. */
static int8_t ffn_requant(int64_t acc, int shift)
{
    int64_t round_c = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (acc + round_c) >> shift;          /* arithmetic shift */
    if      (r >  127) r =  127;
    else if (r < -128) r = -128;
    return (int8_t)r;
}

void feed_forward_network_int8(int d_token, int d_ffn, int frac_bits,
                               const int8_t *x,
                               const int8_t *w1, const int8_t *b1,
                               const int8_t *w2, const int8_t *b2,
                               int8_t *y)
{
    const int DT = d_token;
    const int DF = d_ffn;
    const int FB = frac_bits;

    /* static (single-threaded DPI): keeps the stack small for big overrides */
    static int8_t h[FFN_MAX_DFFN];

    /* ---- Linear1 -> ReLU : h = relu(requant(x@W1^T + b1)) ---- */
    for (int o = 0; o < DF; o++) {
        int64_t acc = 0;
        for (int k = 0; k < DT; k++)
            acc += (int64_t)x[k] * (int64_t)w1[o * DT + k];
        acc += (int64_t)b1[o] << FB;
        int8_t hpre = ffn_requant(acc, FB);
        h[o] = (hpre < 0) ? 0 : hpre;              /* ReLU (exact) */
    }

    /* ---- Linear2 : y = requant(h@W2^T + b2) ---- */
    for (int o = 0; o < DT; o++) {
        int64_t acc = 0;
        for (int k = 0; k < DF; k++)
            acc += (int64_t)h[k] * (int64_t)w2[o * DF + k];
        acc += (int64_t)b2[o] << FB;
        y[o] = ffn_requant(acc, FB);
    }
}

#ifdef FFN_STANDALONE
/* ---------------------------------------------------------------------
 * Standalone self-test (compiled only with -DFFN_STANDALONE).
 * Cross-checks the integer model against an INDEPENDENT fixed-point reference:
 * the same dataflow and the same int8 quantization points, but accumulating in
 * double and requantizing via floor(acc/2^frac + 0.5). Because int8*int8
 * products and their sums are exact in double, and an arithmetic right shift
 * equals floor division by a power of two, the two implementations must agree
 * BIT-FOR-BIT. A wrong shift / sign / weight layout makes them diverge.
 *
 * Also reports (informational only) the divergence from a full-precision ReLU
 * FFN, which is dominated by Q1.7 saturation on random int8 weights -- so it is
 * NOT gated.
 * ------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef FFN_MAX_DT
#define FFN_MAX_DT 256
#endif

/* tiny deterministic LCG so the self-test is reproducible without <random> */
static uint32_t lcg_state = 0x13579bdfu;
static int rand_i8(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 24) & 0xff) - 128;   /* [-128, 127] */
}

static int sat8(long v)
{
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int)v;
}

/* small int8 in [-(2^bits), 2^bits): keeps the datapath out of saturation so
 * the requant rounding is exercised across the whole range, not just the rails. */
static int rand_i8_small(int bits)
{
    int m = (1 << (bits + 1)) - 1;          /* mask to [0, 2^(bits+1)-1] */
    return (rand_i8() & m) - (1 << bits);
}

/* Independent fixed-point reference: accumulate the SAME int8 products in
 * double, add the aligned bias, requantize with floor(acc/2^frac + 0.5) and
 * saturate; ReLU the int8 hidden vector. Must equal feed_forward_network_int8()
 * bit-for-bit. */
static void ffn_fixedpoint_ref(int DT, int DF, int frac,
                               const int8_t *x,
                               const int8_t *w1, const int8_t *b1,
                               const int8_t *w2, const int8_t *b2,
                               int8_t *y)
{
    double scale = (double)(1 << frac);
    static int8_t h[FFN_MAX_DFFN];

    for (int o = 0; o < DF; o++) {
        double acc = 0.0;
        for (int k = 0; k < DT; k++)
            acc += (double)x[k] * (double)w1[o * DT + k];     /* exact int product */
        acc += (double)((int64_t)b1[o] << frac);              /* bias aligned, exact */
        int hpre = sat8((long)floor(acc / scale + 0.5));      /* round-half-up requant */
        h[o] = (int8_t)((hpre < 0) ? 0 : hpre);               /* ReLU */
    }
    for (int o = 0; o < DT; o++) {
        double acc = 0.0;
        for (int k = 0; k < DF; k++)
            acc += (double)h[k] * (double)w2[o * DF + k];
        acc += (double)((int64_t)b2[o] << frac);
        y[o] = (int8_t)sat8((long)floor(acc / scale + 0.5));
    }
}

/* Informational only: a full-precision ReLU FFN (no intermediate quantization).
 * The residual vs the int8 model is dominated by Q1.7 saturation on random
 * int8 weights and is NOT a correctness signal -- printed for context. */
static void ffn_float_ideal(int DT, int DF, int frac,
                            const int8_t *x,
                            const int8_t *w1, const int8_t *b1,
                            const int8_t *w2, const int8_t *b2,
                            int8_t *y)
{
    double scale = (double)(1 << frac);
    static double h[FFN_MAX_DFFN];

    for (int o = 0; o < DF; o++) {
        double acc = 0.0;
        for (int k = 0; k < DT; k++)
            acc += ((double)x[k] / scale) * ((double)w1[o * DT + k] / scale);
        acc += (double)b1[o] / scale;
        h[o] = (acc < 0.0) ? 0.0 : acc;            /* true ReLU, no quantization */
    }
    for (int o = 0; o < DT; o++) {
        double acc = 0.0;
        for (int k = 0; k < DF; k++)
            acc += h[k] * ((double)w2[o * DF + k] / scale);
        acc += (double)b2[o] / scale;
        y[o] = (int8_t)sat8((long)floor(acc * scale + 0.5));
    }
}

/* wbits < 0 -> full-range int8 weights (heavy saturation); wbits >= 0 -> small
 * weights in [-(2^wbits), 2^wbits) so the output stays largely in-range. */
static int run_case(int DT, int DF, int frac, int n_tokens, int wbits)
{
    static int8_t w1[FFN_MAX_DFFN * FFN_MAX_DT], b1[FFN_MAX_DFFN];
    static int8_t w2[FFN_MAX_DT * FFN_MAX_DFFN], b2[FFN_MAX_DT];
    static int8_t x[FFN_MAX_DT], yi[FFN_MAX_DT], yr[FFN_MAX_DT], yf[FFN_MAX_DT];
    int err_fp = 0, err_ideal = 0;
    #define RW() ((int8_t)(wbits < 0 ? rand_i8() : rand_i8_small(wbits)))

    for (int i = 0; i < DF * DT; i++) w1[i] = RW();
    for (int i = 0; i < DF; i++)      b1[i] = RW();
    for (int i = 0; i < DT * DF; i++) w2[i] = RW();
    for (int i = 0; i < DT; i++)      b2[i] = RW();
    #undef RW

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < DT; i++) x[i] = (int8_t)rand_i8();
        feed_forward_network_int8(DT, DF, frac, x, w1, b1, w2, b2, yi);
        ffn_fixedpoint_ref(DT, DF, frac, x, w1, b1, w2, b2, yr);
        ffn_float_ideal(DT, DF, frac, x, w1, b1, w2, b2, yf);
        for (int i = 0; i < DT; i++) {
            int ef = yi[i] - yr[i]; if (ef < 0) ef = -ef;
            int ei = yi[i] - yf[i]; if (ei < 0) ei = -ei;
            if (ef > err_fp)    err_fp = ef;
            if (ei > err_ideal) err_ideal = ei;
        }
    }
    printf("  DT=%-4d DFFN=%-4d frac=%d %-12s: fixed-point ref max |diff| = %d LSB  [%s]"
           "   (vs ideal: %3d LSB)\n",
           DT, DF, frac, (wbits < 0) ? "(full-range)" : "(small-w)",
           err_fp, (err_fp == 0) ? "bit-exact" : "BAD", err_ideal);
    return (err_fp == 0) ? 0 : 1;
}

int main(void)
{
    const int frac = 7;
    int fails = 0;

    printf("feed_forward_network_cmodel self-test "
           "(int8 ReLU FFN vs independent fixed-point reference)\n");
    printf("------------------------------------------------------------------\n");
    /* full-range weights: heavy Q1.7 saturation (rails exercised) */
    fails += run_case(32, 64,  frac, 4000, -1);   /* model defaults */
    fails += run_case(16, 32,  frac, 4000, -1);
    fails += run_case(64, 128, frac, 2000, -1);
    fails += run_case(32, 256, frac, 2000, -1);
    /* small weights: outputs stay in-range, so requant ROUNDING is exercised */
    fails += run_case(32, 64,  frac, 4000,  3);
    fails += run_case(64, 128, frac, 2000,  2);
    printf("------------------------------------------------------------------\n");
    if (fails == 0)
        printf("PASS: integer model matches the fixed-point reference bit-for-bit.\n");
    else
        printf("FAIL: %d case(s) diverged from the fixed-point reference.\n", fails);
    return fails ? 1 : 0;
}
#endif /* FFN_STANDALONE */
