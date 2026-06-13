/* =====================================================================
 * numerical_feature_tokenizer_cmodel.c
 *
 * Behavioral C reference model for the int8 FT-Transformer numerical-feature
 * tokenizer. See numerical_feature_tokenizer_cmodel.h for the contract. This
 * file is the GOLDEN model: hw/rtl/numerical_feature_tokenizer.v runs the
 * identical integer datapath, so the RTL output equals
 * numerical_feature_tokenizer_int8() bit-for-bit (a DPI testbench can check
 * this, mirroring the layer_norm / ffn / mha benches).
 *
 * Pure math only -- no clocks, pipelines, valid signals, or sim/DPI detail.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DNFT_STANDALONE -O2 -std=c11 \
 *         src/models/numerical_feature_tokenizer_cmodel.c -o /tmp/nftcm -lm
 *     /tmp/nftcm
 *   -> cross-checks the integer model against an INDEPENDENT fixed-point
 *      reference (double accumulation + floor-requant, same int8 quantization
 *      points); they must agree bit-for-bit. Also reports (informational) the
 *      dequantized divergence from the float x*W+b op, bounded by 0.5 LSB on
 *      non-saturated cells.
 * =====================================================================
 */
#include "numerical_feature_tokenizer_cmodel.h"

#include <stdint.h>

/* The round-half-up requant relies on an arithmetic (sign-replicating) right
 * shift of a negative signed value, which is what GCC/Clang implement and what
 * Verilog `>>>` on a signed reg does. Fail loudly if that ever stops holding. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "numerical_feature_tokenizer_cmodel requires arithmetic right shift on int64_t");
#endif

/* round-half-up, arithmetic right shift by `shift`, saturate to int8. */
static int8_t nft_requant(int64_t acc, int shift)
{
    int64_t round_c = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (acc + round_c) >> shift;          /* arithmetic shift */
    if      (r >  127) r =  127;
    else if (r < -128) r = -128;
    return (int8_t)r;
}

void numerical_feature_tokenizer_int8(int n_feature, int d_token, int frac_bits,
                                      const int8_t *x,
                                      const int8_t *weight, const int8_t *bias,
                                      int8_t *out)
{
    const int F  = n_feature;
    const int D  = d_token;
    const int FB = frac_bits;

    for (int j = 0; j < F; j++) {
        const int64_t xj = (int64_t)x[j];
        for (int k = 0; k < D; k++) {
            int64_t acc = xj * (int64_t)weight[j * D + k]
                        + ((int64_t)bias[j * D + k] << FB);
            out[j * D + k] = nft_requant(acc, FB);
        }
    }
}

#ifdef NFT_STANDALONE
/* ---------------------------------------------------------------------
 * Standalone self-test (compiled only with -DNFT_STANDALONE).
 * ------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef NFT_MAX_F
#define NFT_MAX_F 128
#endif
#ifndef NFT_MAX_D
#define NFT_MAX_D 128
#endif

static uint32_t lcg_state = 0x2468ace0u;
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

/* Independent fixed-point reference: same int8 products in double, aligned
 * bias, floor(acc/2^frac + 0.5), saturate. Must equal the integer model
 * bit-for-bit. */
static void nft_fixedpoint_ref(int F, int D, int frac,
                               const int8_t *x, const int8_t *w, const int8_t *b,
                               int8_t *out)
{
    double scale = (double)(1 << frac);
    for (int j = 0; j < F; j++)
        for (int k = 0; k < D; k++) {
            double acc = (double)x[j] * (double)w[j * D + k];
            acc += (double)((int64_t)b[j * D + k] << frac);
            out[j * D + k] = (int8_t)sat8((long)floor(acc / scale + 0.5));
        }
}

static int run_case(int F, int D, int frac, int n_rows)
{
    static int8_t w[NFT_MAX_F * NFT_MAX_D], b[NFT_MAX_F * NFT_MAX_D];
    static int8_t x[NFT_MAX_F], yi[NFT_MAX_F * NFT_MAX_D], yr[NFT_MAX_F * NFT_MAX_D];
    int err_fp = 0;
    double max_lsb = 0.0;

    for (int i = 0; i < F * D; i++) { w[i] = (int8_t)rand_i8(); b[i] = (int8_t)rand_i8(); }

    for (int t = 0; t < n_rows; t++) {
        for (int j = 0; j < F; j++) x[j] = (int8_t)rand_i8();
        numerical_feature_tokenizer_int8(F, D, frac, x, w, b, yi);
        nft_fixedpoint_ref(F, D, frac, x, w, b, yr);
        for (int i = 0; i < F * D; i++) {
            int e = yi[i] - yr[i]; if (e < 0) e = -e;
            if (e > err_fp) err_fp = e;
        }
        /* informational: dequantized vs float op on non-saturated cells */
        double s = (double)(1 << frac);
        for (int j = 0; j < F; j++)
            for (int k = 0; k < D; k++) {
                int v = yi[j * D + k];
                if (v > -128 && v < 127) {
                    double fout = ((double)x[j] / s) * ((double)w[j * D + k] / s)
                                + (double)b[j * D + k] / s;
                    double lsb = fabs((double)v / s - fout) * s;
                    if (lsb > max_lsb) max_lsb = lsb;
                }
            }
    }
    printf("  F=%-4d D=%-4d frac=%d : fixed-point ref max |diff| = %d LSB  [%s]"
           "   (vs float op: %.3f LSB)\n",
           F, D, frac, err_fp, (err_fp == 0) ? "bit-exact" : "BAD", max_lsb);
    return (err_fp == 0 && max_lsb <= 0.5 + 1e-9) ? 0 : 1;
}

int main(void)
{
    const int frac = 7;
    int fails = 0;

    printf("numerical_feature_tokenizer_cmodel self-test "
           "(int8 model vs independent fixed-point reference)\n");
    printf("------------------------------------------------------------------\n");
    fails += run_case(16, 32, frac, 4000);   /* hardware target (F=16, D=32) */
    fails += run_case(20, 32, frac, 4000);   /* RTL default N_FEATURE=20 */
    fails += run_case(4,  4,  frac, 4000);   /* tb_numerical_feature_tokenizer.v shape */
    fails += run_case(64, 64, frac, 1000);
    printf("------------------------------------------------------------------\n");
    if (fails == 0) printf("PASS: integer model matches the fixed-point reference bit-for-bit.\n");
    else            printf("FAIL: %d case(s) diverged.\n", fails);
    return fails ? 1 : 0;
}
#endif /* NFT_STANDALONE */
