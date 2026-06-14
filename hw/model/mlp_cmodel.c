/* =====================================================================
 * mlp_cmodel.c
 *
 * Behavioral C reference model for the int8 MLP accelerator (SmallMLP.forward
 * in src/models/mlp.py). See mlp_cmodel.h for the contract. This file is the
 * GOLDEN model: hw/rtl/mlp.v runs the identical integer datapath (gated fc1,
 * ReLU, dynamic block-floating-point requant with round-to-nearest-ties-to-even,
 * int8 fc2/fc3), so the RTL output equals mlp_int8() bit-for-bit -- both the
 * int8 result y and the fc3 dynamic shift -- which the SV/DPI testbench checks.
 *
 * Pure math only -- no clocks, pipelines, valid signals, or sim/DPI detail.
 * Every accumulator is int64 and sized never to truncate.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DMLP_STANDALONE -O2 -std=c11 hw/model/mlp_cmodel.c -o /tmp/mlpcm -lm
 *     /tmp/mlpcm
 *   -> cross-checks the integer model against an INDEPENDENT reference that runs
 *      the same dataflow but does the ties-to-even rounding with the FPU's native
 *      round-to-nearest-even mode (nearbyint under FE_TONEAREST). int8 products
 *      and their sums are exact in double, so the two must agree BIT-FOR-BIT;
 *      a wrong shift / RNE / sign / weight layout makes them diverge.
 * =====================================================================
 */
#include "mlp_cmodel.h"

#include <stdint.h>

/* The dynamic requant relies on an arithmetic (sign-replicating) right shift of
 * a negative signed value -- what GCC/Clang (the compilers VCS uses for DPI)
 * implement, and what Verilog `>>>` on a signed reg does. Fail loudly at compile
 * time if that ever stops holding. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "mlp_cmodel requires arithmetic right shift on int64_t");
#endif

/* compile-time scratch bound for the hidden vectors (generous; override -D) */
#ifndef MLP_MAX_H
#define MLP_MAX_H 4096
#endif

#define MLP_OUT_BITS 8   /* int8 output */

/* number of significant bits of an unsigned magnitude (bitlen(0)=0). */
static int mlp_bitlen(int64_t m)
{
    int bl = 0;
    while (m) { bl++; m >>= 1; }
    return bl;
}

/* block-floating-point shift shared by a result vector: the minimal right-shift
 * s such that max|v| fits in (OUT_BITS-1) magnitude bits. */
static int mlp_dyn_shift(const int64_t *v, int n)
{
    int64_t m = 0;
    for (int i = 0; i < n; i++) {
        int64_t a = (v[i] < 0) ? -v[i] : v[i];
        if (a > m) m = a;
    }
    int s = mlp_bitlen(m) - (MLP_OUT_BITS - 1);
    return (s < 0) ? 0 : s;
}

/* round-to-nearest-ties-to-even arithmetic right shift by s, saturate to int8.
 * s==0 means no rounding (just saturate). */
static int8_t mlp_rne_sat(int64_t v, int s)
{
    int64_t q;
    if (s <= 0) {
        q = v;
    } else {
        q = v >> s;                       /* arithmetic floor */
        int64_t rem  = v - (q << s);      /* in [0, 2^s) */
        int64_t half = (int64_t)1 << (s - 1);
        if      (rem >  half) q++;
        else if (rem == half) { if (q & 1) q++; }   /* tie -> even */
    }
    const int64_t hi = ((int64_t)1 << (MLP_OUT_BITS - 1)) - 1;   /* +127 */
    const int64_t lo = -((int64_t)1 << (MLP_OUT_BITS - 1));      /* -128 */
    if      (q > hi) q = hi;
    else if (q < lo) q = lo;
    return (int8_t)q;
}

void mlp_int8(int n_features, int hidden1, int hidden2,
              const int8_t *x,
              const int8_t *w1, const int8_t *b1,
              const int8_t *w2, const int8_t *b2,
              const int8_t *w3, const int8_t *b3,
              int8_t *y, int *shift)
{
    const int NF = n_features, H1 = hidden1, H2 = hidden2;

    /* static (single-threaded DPI): keeps the stack small for big overrides */
    static int64_t a1[MLP_MAX_H], a2[MLP_MAX_H];
    static int8_t  h1[MLP_MAX_H], h2[MLP_MAX_H];

    /* ---- fc1 -> ReLU : gated adder tree (x in {0,1}), then dynamic requant -- */
    for (int j = 0; j < H1; j++) {
        int64_t acc = (int64_t)b1[j];
        for (int i = 0; i < NF; i++)
            if (x[i]) acc += (int64_t)w1[j * NF + i];
        a1[j] = (acc < 0) ? 0 : acc;                 /* ReLU on full precision */
    }
    int s1 = mlp_dyn_shift(a1, H1);
    for (int j = 0; j < H1; j++) h1[j] = mlp_rne_sat(a1[j], s1);

    /* ---- fc2 -> ReLU : int8 matmul, then dynamic requant ------------------- */
    for (int k = 0; k < H2; k++) {
        int64_t acc = (int64_t)b2[k];
        for (int i = 0; i < H1; i++)
            acc += (int64_t)h1[i] * (int64_t)w2[k * H1 + i];
        a2[k] = (acc < 0) ? 0 : acc;                 /* ReLU */
    }
    int s2 = mlp_dyn_shift(a2, H2);
    for (int k = 0; k < H2; k++) h2[k] = mlp_rne_sat(a2[k], s2);

    /* ---- fc3 (no activation) : int8 matmul -> scalar, dynamic requant ------ */
    int64_t a3 = (int64_t)b3[0];
    for (int i = 0; i < H2; i++)
        a3 += (int64_t)h2[i] * (int64_t)w3[i];
    int s3 = mlp_dyn_shift(&a3, 1);
    *y     = mlp_rne_sat(a3, s3);
    *shift = s3;
}

#ifdef MLP_STANDALONE
/* ---------------------------------------------------------------------
 * Standalone self-test (compiled only with -DMLP_STANDALONE).
 * Cross-checks the integer model against an INDEPENDENT reference: the same
 * dataflow, but the ties-to-even rounding is done with the FPU's native
 * round-to-nearest-even mode via nearbyint() under FE_TONEAREST. int8*int8
 * products and their sums are exact in double (|acc| << 2^53), so the two
 * implementations must agree BIT-FOR-BIT in both y and the fc3 shift. A wrong
 * shift / RNE / sign / weight layout makes them diverge.
 * ------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>

#ifndef MLP_MAX_NF
#define MLP_MAX_NF 256
#endif

/* tiny deterministic LCG so the self-test is reproducible without <random> */
static uint32_t lcg_state = 0x13579bdfu;
static int rand_i8(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 24) & 0xff) - 128;   /* [-128, 127] */
}
static int rand_bit(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 31) & 0x1);
}
/* small int8 in [-(2^bits), 2^bits): keeps the datapath out of saturation so
 * the requant rounding (incl. exact ties) is exercised, not just the rails. */
static int rand_i8_small(int bits)
{
    int m = (1 << (bits + 1)) - 1;
    return (rand_i8() & m) - (1 << bits);
}

static int sat8(long v) { return (v > 127) ? 127 : (v < -128) ? -128 : (int)v; }

/* independent block-floating-point shift (same definition; trivial integer op) */
static int ref_dyn_shift(const double *v, int n)
{
    long long m = 0;
    for (int i = 0; i < n; i++) {
        long long a = (long long)llabs((long long)v[i]);
        if (a > m) m = a;
    }
    int bl = 0; long long t = m; while (t) { bl++; t >>= 1; }
    int s = bl - 7;
    return (s < 0) ? 0 : s;
}
/* independent RNE via the FPU's round-to-nearest-even mode + saturate */
static int ref_rne_sat(double v, int s)
{
    double q = (s <= 0) ? v : nearbyint(v / (double)((int64_t)1 << s));
    return sat8((long)q);
}

static void mlp_ref(int NF, int H1, int H2,
                    const int8_t *x,
                    const int8_t *w1, const int8_t *b1,
                    const int8_t *w2, const int8_t *b2,
                    const int8_t *w3, const int8_t *b3,
                    int8_t *y, int *shift)
{
    static double a1[MLP_MAX_H], a2[MLP_MAX_H];
    static int8_t h1[MLP_MAX_H], h2[MLP_MAX_H];

    for (int j = 0; j < H1; j++) {
        double acc = (double)b1[j];
        for (int i = 0; i < NF; i++) if (x[i]) acc += (double)w1[j * NF + i];
        a1[j] = (acc < 0.0) ? 0.0 : acc;
    }
    int s1 = ref_dyn_shift(a1, H1);
    for (int j = 0; j < H1; j++) h1[j] = (int8_t)ref_rne_sat(a1[j], s1);

    for (int k = 0; k < H2; k++) {
        double acc = (double)b2[k];
        for (int i = 0; i < H1; i++) acc += (double)h1[i] * (double)w2[k * H1 + i];
        a2[k] = (acc < 0.0) ? 0.0 : acc;
    }
    int s2 = ref_dyn_shift(a2, H2);
    for (int k = 0; k < H2; k++) h2[k] = (int8_t)ref_rne_sat(a2[k], s2);

    double a3 = (double)b3[0];
    for (int i = 0; i < H2; i++) a3 += (double)h2[i] * (double)w3[i];
    int s3 = ref_dyn_shift(&a3, 1);
    *y     = (int8_t)ref_rne_sat(a3, s3);
    *shift = s3;
}

/* wbits < 0 -> full-range int8 weights; wbits >= 0 -> small weights so outputs
 * stay in-range and the requant rounding is exercised across its whole domain. */
static int run_case(int NF, int H1, int H2, int n_vec, int wbits)
{
    static int8_t w1[MLP_MAX_H * MLP_MAX_NF], b1[MLP_MAX_H];
    static int8_t w2[MLP_MAX_H * MLP_MAX_H],  b2[MLP_MAX_H];
    static int8_t w3[MLP_MAX_H], b3[1];
    static int8_t x[MLP_MAX_NF];
    int err = 0, shift_err = 0;
    #define RW() ((int8_t)(wbits < 0 ? rand_i8() : rand_i8_small(wbits)))

    for (int i = 0; i < H1 * NF; i++) w1[i] = RW();
    for (int i = 0; i < H1; i++)      b1[i] = RW();
    for (int i = 0; i < H2 * H1; i++) w2[i] = RW();
    for (int i = 0; i < H2; i++)      b2[i] = RW();
    for (int i = 0; i < H2; i++)      w3[i] = RW();
    b3[0] = RW();
    #undef RW

    for (int t = 0; t < n_vec; t++) {
        for (int i = 0; i < NF; i++) x[i] = (int8_t)rand_bit();
        int8_t yi, yr; int si, sr;
        mlp_int8(NF, H1, H2, x, w1, b1, w2, b2, w3, b3, &yi, &si);
        mlp_ref (NF, H1, H2, x, w1, b1, w2, b2, w3, b3, &yr, &sr);
        if (yi != yr) err++;
        if (si != sr) shift_err++;
    }
    int bad = (err != 0) || (shift_err != 0);
    printf("  NF=%-4d H1=%-4d H2=%-4d %-12s: y mismatch=%d  shift mismatch=%d  [%s]\n",
           NF, H1, H2, (wbits < 0) ? "(full-range)" : "(small-w)",
           err, shift_err, bad ? "BAD" : "bit-exact");
    return bad ? 1 : 0;
}

int main(void)
{
    int fails = 0;
    fesetround(FE_TONEAREST);   /* ties-to-even, matching mlp_rne_sat */

    printf("mlp_cmodel self-test "
           "(int8 dynamic-quant MLP vs independent nearbyint-RNE reference)\n");
    printf("------------------------------------------------------------------\n");
    /* full-range weights: large accumulators, big dynamic shifts */
    fails += run_case(32, 16, 16, 20000, -1);   /* model defaults */
    fails += run_case(16,  8,  8, 20000, -1);
    fails += run_case(64, 32, 24, 10000, -1);
    fails += run_case(20, 12, 10, 10000, -1);
    /* small weights: outputs stay in-range, so requant ROUNDING (incl. exact
     * ties to even) is exercised, not just the saturation rails */
    fails += run_case(32, 16, 16, 20000,  2);
    fails += run_case(64, 32, 24, 10000,  3);
    fails += run_case(48, 24, 16, 10000,  1);
    printf("------------------------------------------------------------------\n");
    if (fails == 0)
        printf("PASS: integer model matches the independent reference bit-for-bit.\n");
    else
        printf("FAIL: %d case(s) diverged.\n", fails);
    return fails ? 1 : 0;
}
#endif /* MLP_STANDALONE */
