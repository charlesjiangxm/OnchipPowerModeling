/* =====================================================================
 * layer_norm_cmodel.c
 *
 * Behavioral C reference model for the int8 FT-Transformer LayerNorm.
 * See layer_norm_cmodel.h for the contract. This file is the GOLDEN model:
 * hw/rtl/layer_norm.v runs the identical integer datapath, so the RTL output
 * equals layer_norm_int8() bit-for-bit (the SV/DPI testbench checks this).
 *
 * Pure math only -- no clocks, pipelines, valid signals, or sim/DPI detail.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DLN_STANDALONE -O2 src/models/layer_norm_cmodel.c -o /tmp/lncm -lm
 *     /tmp/lncm
 *   -> checks the integer model against the float nn.LayerNorm ideal
 *      (<= 1 LSB) for several d_token values and random tokens.
 * =====================================================================
 */
#include "layer_norm_cmodel.h"

#include <stdint.h>

/* The round-half-up requant relies on an arithmetic (sign-replicating) right
 * shift of a negative signed value, which is what GCC/Clang -- the compilers
 * VCS uses for DPI -- implement, and what Verilog `>>>` on a signed reg does.
 * Fail loudly at compile time if that ever stops holding. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "layer_norm_cmodel requires arithmetic right shift on int64_t");
#endif

/* floor(sqrt(n)) by the classic bit-by-bit method. Exact for any n.
 * hw/rtl/layer_norm.v uses the fully-unrolled fixed-iteration form of the
 * same algorithm; both yield the exact floor sqrt, so they agree bit-for-bit. */
static uint64_t isqrt_floor(uint64_t n)
{
    uint64_t res = 0;
    uint64_t one = (uint64_t)1 << 62;   /* highest even bit of a 64-bit word */
    while (one > n) one >>= 2;           /* data-dependent prune is fine in C */
    while (one != 0) {
        uint64_t t = res + one;
        if (n >= t) {
            n  -= t;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }
    return res;
}

long long layer_norm_eps_v(int d_token, int frac_bits, double eps)
{
    /* EPS_V = round(eps * 2^(2*frac) * D^2). */
    double scale = (double)((int64_t)1 << (2 * frac_bits));
    double v = eps * scale * (double)d_token * (double)d_token;
    return (long long)(v + 0.5);
}

void layer_norm_int8(int d_token, int frac_bits, long long eps_v,
                     int recip_frac, int out_frac,
                     const int8_t *x, const int8_t *gamma, const int8_t *beta,
                     int8_t *y)
{
    const int    D     = d_token;
    const int    SHIFT = frac_bits + recip_frac - out_frac;
    const int64_t OUT_MAX = ((int64_t)1 << (8 - 1)) - 1;   /* +127 */
    const int64_t OUT_MIN = -((int64_t)1 << (8 - 1));       /* -128 */

    /* --- reduction: S = sum(x), SS = sum(x*x) --- */
    int64_t S  = 0;
    int64_t SS = 0;
    for (int i = 0; i < D; i++) {
        int64_t xi = (int64_t)x[i];
        S  += xi;
        SS += xi * xi;
    }

    /* --- variance numerator V = D^2 * var(int), epsilon, inverse std --- */
    int64_t V    = (int64_t)D * SS - S * S;     /* >= 0, exact */
    uint64_t Veps = (uint64_t)(V + eps_v);
    uint64_t r    = isqrt_floor(Veps);
    if (r == 0) r = 1;                          /* avoid divide-by-zero */
    uint64_t recip_one = (uint64_t)1 << recip_frac;
    uint64_t inv  = (recip_one + (r >> 1)) / r; /* round(2^recip_frac / r) */

    /* --- per-lane normalize + affine + requantize to int8 --- */
    int64_t round_const = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    for (int i = 0; i < D; i++) {
        int64_t num   = (int64_t)D * (int64_t)x[i] - S;   /* D*(x[i] - mean) */
        int64_t znorm = num * (int64_t)inv;               /* ~ normalized * 2^recip_frac */
        int64_t acc   = znorm * (int64_t)gamma[i]
                        + ((int64_t)beta[i] << recip_frac);
        int64_t res   = (acc + round_const) >> SHIFT;     /* round-half-up, arithmetic */
        if      (res > OUT_MAX) res = OUT_MAX;
        else if (res < OUT_MIN) res = OUT_MIN;
        y[i] = (int8_t)res;
    }
}

/* ---------------------------------------------------------------------
 * Standalone self-test (compiled only with -DLN_STANDALONE).
 * Confirms the integer model tracks the float nn.LayerNorm ideal to <=1 LSB.
 * ------------------------------------------------------------------- */
#ifdef LN_STANDALONE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* tiny deterministic LCG so the self-test is reproducible without <random> */
static uint32_t lcg_state = 0x1234567u;
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

/* float nn.LayerNorm ideal, then quantize to int8 Q1.out_frac (round-half-up). */
static void float_ideal(int D, int frac, int out_frac, double eps,
                        const int8_t *x, const int8_t *g, const int8_t *b,
                        int8_t *y)
{
    double scale = (double)(1 << frac);
    double mean = 0.0;
    for (int i = 0; i < D; i++) mean += (double)x[i] / scale;
    mean /= D;
    double var = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)x[i] / scale - mean;
        var += d * d;
    }
    var /= D;                                   /* biased / population variance */
    double inv_std = 1.0 / sqrt(var + eps);
    double oscale = (double)(1 << out_frac);
    for (int i = 0; i < D; i++) {
        double xn = ((double)x[i] / scale - mean) * inv_std;
        double out = xn * ((double)g[i] / scale) + (double)b[i] / scale;
        y[i] = (int8_t)sat8((long)floor(out * oscale + 0.5));
    }
}

static int run_case(int D, int frac, int recip_frac, int out_frac,
                    double eps, int n_tokens, int *max_err_out)
{
    long long eps_v = layer_norm_eps_v(D, frac, eps);
    int8_t x[256], g[256], b[256], yi[256], yf[256];
    int max_err = 0, n_over = 0;

    for (int i = 0; i < D; i++) { g[i] = (int8_t)rand_i8(); b[i] = (int8_t)rand_i8(); }

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < D; i++) x[i] = (int8_t)rand_i8();
        layer_norm_int8(D, frac, eps_v, recip_frac, out_frac, x, g, b, yi);
        float_ideal(D, frac, out_frac, eps, x, g, b, yf);
        for (int i = 0; i < D; i++) {
            int e = yi[i] - yf[i];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            if (e > 1) n_over++;
        }
    }
    *max_err_out = max_err;
    printf("  D=%-4d frac=%d recip_frac=%d out_frac=%d eps_v=%-6lld : "
           "max |int-float| = %d LSB, #(>1 LSB)=%d  [%s]\n",
           D, frac, recip_frac, out_frac, eps_v, max_err, n_over,
           (max_err <= 1) ? "ok" : "BAD");
    return (max_err <= 1) ? 0 : 1;
}

int main(void)
{
    const double eps = 1e-5;
    const int frac = 7, recip_frac = 24;
    int fails = 0, me;

    printf("layer_norm_cmodel self-test (int8 model vs float nn.LayerNorm ideal)\n");
    printf("------------------------------------------------------------------\n");
    /* strict Q1.7 output (out_frac=7): saturates a lot, still <=1 LSB vs ideal */
    int dims[] = {16, 32, 64, 128};
    for (int k = 0; k < 4; k++) fails += run_case(dims[k], frac, recip_frac, 7, eps, 4000, &me);
    /* coarser Q3.5 output (out_frac=5): little/no saturation */
    for (int k = 0; k < 4; k++) fails += run_case(dims[k], frac, recip_frac, 5, eps, 4000, &me);
    printf("------------------------------------------------------------------\n");
    if (fails == 0) printf("PASS: all cases within 1 LSB of the float ideal.\n");
    else            printf("FAIL: %d case(s) exceeded 1 LSB.\n", fails);
    return fails ? 1 : 0;
}
#endif /* LN_STANDALONE */
