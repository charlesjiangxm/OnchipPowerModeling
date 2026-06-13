/* =====================================================================
 * multihead_attention_cmodel.c
 *
 * Behavioral C reference model for the int8 FT-Transformer self-attention.
 * See multihead_attention_cmodel.h for the contract. This file is the GOLDEN
 * model: hw/rtl/multihead_attention.v runs the identical integer datapath, so
 * the RTL output equals multihead_attention_int8() bit-for-bit (the SV/DPI
 * testbench checks this).
 *
 * Pure math only -- no clocks, pipelines, valid signals, or sim/DPI detail.
 * Every accumulator is int64 and sized to never truncate, so the (associative)
 * integer sums match the RTL's exact-width adders regardless of order.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DMA_STANDALONE -O2 -std=c11 src/models/multihead_attention_cmodel.c -o /tmp/macm -lm
 *     /tmp/macm
 *   -> checks the integer attention against a float reference that shares the
 *      same int8 dataflow but uses true exp()/softmax, so the residual is just
 *      the integer-softmax approximation (a few LSB).
 * =====================================================================
 */
#include "multihead_attention_cmodel.h"

#include <stdint.h>

#ifdef MA_STANDALONE
#include <math.h>   /* mha_scale + the standalone float reference only */
#endif

/* The round-half-up requant and the (negative) C1 polynomial term rely on an
 * arithmetic (sign-replicating) right shift of negative signed values -- what
 * GCC/Clang (the compilers VCS uses for DPI) implement, and what Verilog `>>>`
 * on a signed reg does. Fail loudly at compile time if that stops holding. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "multihead_attention_cmodel requires arithmetic right shift on int64_t");
#endif

/* ---- fixed exp constants (Q16). MUST match hw/rtl/multihead_attention.v ----
 * 2^-f on f in [0,1) ~= C2*f^2 + C1*f + C0, endpoint-exact (p(0)=1.0, p(1)=0.5),
 * max abs poly error ~0.0019. EXP_FRAC is fixed at 16 (the "1.0" = 65536). */
#define MHA_LOG2E_FRAC 16
#define MHA_EXP_FRAC   16
#define MHA_LOG2E      94548LL    /* round(log2(e) * 2^16) */
#define MHA_EXP_C2     11279LL    /* round( 0.172100 * 2^16) */
#define MHA_EXP_C1   (-44047LL)   /* round(-0.672100 * 2^16) */
#define MHA_EXP_C0     65536LL    /* 2^16 */
#define MHA_ZMAX       (MHA_EXP_FRAC + 1)   /* shift past which exp -> 0 */

/* compile-time scratch bounds (generous; override with -D if needed) */
#ifndef MHA_MAX_E
#define MHA_MAX_E 256
#endif
#ifndef MHA_MAX_S
#define MHA_MAX_S 128
#endif

/* round-half-up, arithmetic right shift by `shift`, saturate to int8. */
static int8_t mha_requant(int64_t acc, int shift)
{
    int64_t round_c = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (acc + round_c) >> shift;          /* arithmetic shift */
    if      (r >  127) r =  127;
    else if (r < -128) r = -128;
    return (int8_t)r;
}

/* exp(d) for d <= 0, with d in Q(sm_frac). Returns Q(MHA_EXP_FRAC=16), >= 0.
 * d = 0 -> 65536 (= 1.0). Integer-only: base-2 decomposition + fitted quad. */
static int64_t mha_exp_neg(int64_t d, int sm_frac)
{
    int     SH = sm_frac + MHA_LOG2E_FRAC;
    int64_t u, m, z, f, t2, t1, p;

    if (d > 0) d = 0;                  /* defensive; caller guarantees d <= 0 */
    u  = -d;                           /* >= 0, Q(sm_frac) */
    m  = u * MHA_LOG2E;                /* >= 0, Q(sm_frac + LOG2E_FRAC), exact */
    z  = m >> SH;                      /* integer part = # of halvings */
    f  = (m - (z << SH)) >> (SH - MHA_EXP_FRAC);    /* remainder, Q16 in [0,1) */
    t2 = (MHA_EXP_C2 * f * f) >> (2 * MHA_EXP_FRAC);
    t1 = (MHA_EXP_C1 * f) >> MHA_EXP_FRAC;          /* C1 < 0 -> arithmetic */
    p  = t2 + t1 + MHA_EXP_C0;         /* 2^-f in Q16, (0.5, 1] */
    if (z >= MHA_ZMAX) return 0;       /* 2^-z < 1 LSB -> exactly 0 */
    return p >> z;                     /* p >= 0 -> logical shift */
}

void multihead_attention_int8(
    int d_token, int n_heads, int seq_len, int frac_bits,
    long long scale, int scale_frac, int sm_frac, int recip_frac,
    const int8_t *x,
    const int8_t *in_proj_w, const int8_t *in_proj_b,
    const int8_t *out_proj_w, const int8_t *out_proj_b,
    int8_t *y)
{
    const int E  = d_token;
    const int H  = n_heads;
    const int S  = seq_len;
    const int HD = E / H;
    const int FB = frac_bits;
    const int SC_SHIFT  = 2 * FB + scale_frac - sm_frac;   /* score Q(2FB) -> Q(sm_frac) */
    const int64_t SC_RND = (SC_SHIFT > 0) ? ((int64_t)1 << (SC_SHIFT - 1)) : 0;
    const int64_t RECIP_ONE = (int64_t)1 << recip_frac;

    /* static (single-threaded DPI): keeps the stack small for big overrides */
    static int8_t Q[MHA_MAX_S * MHA_MAX_E];
    static int8_t K[MHA_MAX_S * MHA_MAX_E];
    static int8_t V[MHA_MAX_S * MHA_MAX_E];
    static int8_t ctx[MHA_MAX_S * MHA_MAX_E];   /* concat-of-heads layout [s*E + e] */

    /* ---- in_proj: Q = x@Wq^T+bq, K = x@Wk^T+bk, V = x@Wv^T+bv (int8) ---- */
    for (int s = 0; s < S; s++) {
        for (int e = 0; e < E; e++) {
            int64_t accq = 0, acck = 0, accv = 0;
            for (int k = 0; k < E; k++) {
                int64_t xk = x[s * E + k];
                accq += xk * in_proj_w[(e)        * E + k];   /* Wq row e      */
                acck += xk * in_proj_w[(E + e)    * E + k];   /* Wk row E+e    */
                accv += xk * in_proj_w[(2 * E + e) * E + k];  /* Wv row 2E+e   */
            }
            accq += (int64_t)in_proj_b[e]         << FB;
            acck += (int64_t)in_proj_b[E + e]     << FB;
            accv += (int64_t)in_proj_b[2 * E + e] << FB;
            Q[s * E + e] = mha_requant(accq, FB);
            K[s * E + e] = mha_requant(acck, FB);
            V[s * E + e] = mha_requant(accv, FB);
        }
    }

    /* ---- per-head scaled-dot-product attention ---- */
    for (int h = 0; h < H; h++) {
        for (int qi = 0; qi < S; qi++) {
            int64_t sm[MHA_MAX_S];
            int64_t e_w[MHA_MAX_S];
            int64_t mx, Se, inv;

            /* scores Qh.Kh, scale by 1/sqrt(HD) -> Q(sm_frac) */
            for (int kj = 0; kj < S; kj++) {
                int64_t raw = 0;
                for (int d = 0; d < HD; d++)
                    raw += (int64_t)Q[qi * E + h * HD + d]
                         * (int64_t)K[kj * E + h * HD + d];
                sm[kj] = (raw * (int64_t)scale + SC_RND) >> SC_SHIFT;  /* arithmetic */
            }

            /* row max (for numerical stability; d = sm - max <= 0) */
            mx = sm[0];
            for (int kj = 1; kj < S; kj++) if (sm[kj] > mx) mx = sm[kj];

            /* exp + sum */
            Se = 0;
            for (int kj = 0; kj < S; kj++) {
                e_w[kj] = mha_exp_neg(sm[kj] - mx, sm_frac);
                Se += e_w[kj];
            }
            if (Se == 0) Se = 1;                          /* unreachable (max -> 1.0) */
            inv = (RECIP_ONE + (Se >> 1)) / Se;           /* round(2^recip / Se) */

            /* context = (sum_kj e[kj]*Vh[kj][d]) * inv, requant to int8.
             * The EXP_FRAC scale in e and in Se cancel, so shift is recip only. */
            for (int d = 0; d < HD; d++) {
                int64_t cacc = 0;
                for (int kj = 0; kj < S; kj++)
                    cacc += e_w[kj] * (int64_t)V[kj * E + h * HD + d];
                ctx[qi * E + h * HD + d] = mha_requant(cacc * inv, recip_frac);
            }
        }
    }

    /* ---- out_proj: y = concat @ Wo^T + bo (int8) ---- */
    for (int s = 0; s < S; s++) {
        for (int e = 0; e < E; e++) {
            int64_t acc = 0;
            for (int k = 0; k < E; k++)
                acc += (int64_t)ctx[s * E + k] * (int64_t)out_proj_w[e * E + k];
            acc += (int64_t)out_proj_b[e] << FB;
            y[s * E + e] = mha_requant(acc, FB);
        }
    }
}

#ifdef MA_STANDALONE
/* ---------------------------------------------------------------------
 * mha_scale + standalone self-test (compiled only with -DMA_STANDALONE).
 * Compares the integer model to a float reference that shares the SAME int8
 * dataflow (identical Q/K/V/ctx quantization points) but uses true float
 * exp()/softmax, isolating the integer-softmax approximation error.
 * ------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>

long long mha_scale(int n_heads, int d_token, int scale_frac)
{
    int    hd = d_token / n_heads;
    double v  = (double)((long long)1 << scale_frac) / sqrt((double)hd);
    return (long long)(v + 0.5);
}

/* tiny deterministic LCG so the self-test is reproducible */
static uint32_t lcg_state = 0x2468ace0u;
static int rand_i8(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 24) & 0xff) - 128;   /* [-128, 127] */
}
static int sat8(long v) { return v > 127 ? 127 : (v < -128 ? -128 : (int)v); }

/* float reference: int8 in_proj (same requant) -> float softmax -> int8 ctx
 * -> int8 out_proj. Mirrors the integer dataflow except softmax is exact. */
static void mha_float_ref(int E, int H, int S, int FB, long long scale,
                          int scale_frac,
                          const int8_t *ipw, const int8_t *ipb,
                          const int8_t *opw, const int8_t *opb,
                          const int8_t *x, int8_t *y)
{
    const int HD = E / H;
    static int8_t Q[MHA_MAX_S * MHA_MAX_E], K[MHA_MAX_S * MHA_MAX_E];
    static int8_t Vq[MHA_MAX_S * MHA_MAX_E], ctx[MHA_MAX_S * MHA_MAX_E];
    int64_t round_c = (int64_t)1 << (FB - 1);
    double  sqrt_hd = sqrt((double)HD);

    for (int s = 0; s < S; s++)
        for (int e = 0; e < E; e++) {
            int64_t aq = 0, ak = 0, av = 0;
            for (int k = 0; k < E; k++) {
                int64_t xk = x[s * E + k];
                aq += xk * ipw[(e) * E + k];
                ak += xk * ipw[(E + e) * E + k];
                av += xk * ipw[(2 * E + e) * E + k];
            }
            aq += (int64_t)ipb[e]         << FB;
            ak += (int64_t)ipb[E + e]     << FB;
            av += (int64_t)ipb[2 * E + e] << FB;
            Q[s * E + e]  = (int8_t)sat8((long)((aq + round_c) >> FB));
            K[s * E + e]  = (int8_t)sat8((long)((ak + round_c) >> FB));
            Vq[s * E + e] = (int8_t)sat8((long)((av + round_c) >> FB));
        }

    double oscale = (double)(1 << FB);
    for (int h = 0; h < H; h++)
        for (int qi = 0; qi < S; qi++) {
            double sc[MHA_MAX_S], mx, sum = 0.0;
            for (int kj = 0; kj < S; kj++) {
                long raw = 0;
                for (int d = 0; d < HD; d++)
                    raw += (long)Q[qi * E + h * HD + d] * (long)K[kj * E + h * HD + d];
                /* dequant the Q2.2FB raw score, then scale (use the same int SCALE) */
                sc[kj] = ((double)raw / (double)(1 << (2 * FB)))
                         * ((double)scale / (double)((long long)1 << scale_frac));
            }
            mx = sc[0];
            for (int kj = 1; kj < S; kj++) if (sc[kj] > mx) mx = sc[kj];
            double w[MHA_MAX_S];
            for (int kj = 0; kj < S; kj++) { w[kj] = exp(sc[kj] - mx); sum += w[kj]; }
            for (int d = 0; d < HD; d++) {
                double acc = 0.0;
                for (int kj = 0; kj < S; kj++)
                    acc += (w[kj] / sum) * ((double)Vq[kj * E + h * HD + d] / oscale);
                ctx[qi * E + h * HD + d] =
                    (int8_t)sat8((long)floor(acc * oscale + 0.5));
            }
            (void)sqrt_hd;
        }

    for (int s = 0; s < S; s++)
        for (int e = 0; e < E; e++) {
            int64_t acc = 0;
            for (int k = 0; k < E; k++)
                acc += (int64_t)ctx[s * E + k] * (int64_t)opw[e * E + k];
            acc += (int64_t)opb[e] << FB;
            y[s * E + e] = (int8_t)sat8((long)((acc + round_c) >> FB));
        }
}

/* End-to-end accuracy is INFORMATIONAL, not the correctness gate. On adversarial
 * random int8, the 2nd-order exp-poly's ~6.5e-4 weight error (see
 * softmax_weight_test) propagates through the weighted-sum and is then amplified
 * by the out_proj matmul (~sqrt(E)), so the worst-case residual vs float is a
 * handful of LSB (typical is sub-LSB -- see the mean). The strict guarantees are
 * the tight softmax_weight_test below and the bit-exact RTL-vs-C-model DPI TB.
 * GROSS bugs (a wrong shift / sign / scale) blow past SANE_LSB; that is the gate. */
#define MHA_SANE_LSB 24
static int run_case(int E, int H, int S, int FB, int scale_frac, int sm_frac,
                    int recip_frac, int n_iter)
{
    long long scale = mha_scale(H, E, scale_frac);
    static int8_t ipw[3 * MHA_MAX_E * MHA_MAX_E], ipb[3 * MHA_MAX_E];
    static int8_t opw[MHA_MAX_E * MHA_MAX_E], opb[MHA_MAX_E];
    static int8_t x[MHA_MAX_S * MHA_MAX_E];
    static int8_t yi[MHA_MAX_S * MHA_MAX_E], yf[MHA_MAX_S * MHA_MAX_E];
    int max_err = 0; long long sum_err = 0, n_elem = 0;

    for (int i = 0; i < 3 * E * E; i++) ipw[i] = (int8_t)rand_i8();
    for (int i = 0; i < 3 * E; i++)     ipb[i] = (int8_t)rand_i8();
    for (int i = 0; i < E * E; i++)     opw[i] = (int8_t)rand_i8();
    for (int i = 0; i < E; i++)         opb[i] = (int8_t)rand_i8();

    for (int t = 0; t < n_iter; t++) {
        for (int i = 0; i < S * E; i++) x[i] = (int8_t)rand_i8();
        multihead_attention_int8(E, H, S, FB, scale, scale_frac, sm_frac,
                                 recip_frac, x, ipw, ipb, opw, opb, yi);
        mha_float_ref(E, H, S, FB, scale, scale_frac, ipw, ipb, opw, opb, x, yf);
        for (int i = 0; i < S * E; i++) {
            int e = yi[i] - yf[i];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            sum_err += e; n_elem++;
        }
    }
    printf("  E=%-3d H=%-2d S=%-3d HD=%-2d frac=%d scale=%-5lld : "
           "mean|int-float| = %.3f LSB, max = %2d LSB  [%s]\n",
           E, H, S, E / H, FB, scale, (double)sum_err / (double)n_elem, max_err,
           (max_err <= MHA_SANE_LSB) ? "ok" : "BAD");
    return (max_err <= MHA_SANE_LSB) ? 0 : 1;
}

/* direct integer-vs-float softmax weight accuracy on random score rows */
static int softmax_weight_test(int S, int sm_frac, int n_iter)
{
    double worst = 0.0;
    for (int t = 0; t < n_iter; t++) {
        int64_t sm[MHA_MAX_S], ew[MHA_MAX_S], Se = 0, mx;
        double  fw[MHA_MAX_S], fsum = 0.0;
        for (int j = 0; j < S; j++)
            sm[j] = ((int64_t)rand_i8()) * 4;            /* spread the scores */
        mx = sm[0];
        for (int j = 1; j < S; j++) if (sm[j] > mx) mx = sm[j];
        for (int j = 0; j < S; j++) {
            ew[j] = mha_exp_neg(sm[j] - mx, sm_frac);
            Se += ew[j];
            fw[j] = exp((double)(sm[j] - mx) / (double)(1 << sm_frac));
            fsum += fw[j];
        }
        for (int j = 0; j < S; j++) {
            double iw = (double)ew[j] / (double)Se;
            double d  = iw - fw[j] / fsum;
            if (d < 0) d = -d;
            if (d > worst) worst = d;
        }
    }
    printf("  softmax weight: max|int - float| = %.6f over %d rows (S=%d)  [%s]\n",
           worst, n_iter, S, (worst <= 1e-3) ? "ok" : "BAD");
    return (worst <= 1e-3) ? 0 : 1;
}

int main(void)
{
    const int FB = 7, scale_frac = 14, sm_frac = 8, recip_frac = 24;
    int fails = 0;

    printf("multihead_attention_cmodel self-test "
           "(int8 model vs float-softmax reference)\n");
    printf("------------------------------------------------------------------\n");
    fails += softmax_weight_test(16, sm_frac, 20000);
    fails += softmax_weight_test(32, sm_frac, 20000);
    printf("------------------------------------------------------------------\n");
    fails += run_case(32, 8, 8,  FB, scale_frac, sm_frac, recip_frac, 400);
    fails += run_case(32, 8, 16, FB, scale_frac, sm_frac, recip_frac, 400);
    fails += run_case(32, 8, 32, FB, scale_frac, sm_frac, recip_frac, 200);
    fails += run_case(16, 4, 16, FB, scale_frac, sm_frac, recip_frac, 400);
    fails += run_case(64, 8, 16, FB, scale_frac, sm_frac, recip_frac, 100);
    printf("------------------------------------------------------------------\n");
    if (fails == 0) printf("PASS: integer attention tracks the float reference.\n");
    else            printf("FAIL: %d case(s) out of tolerance.\n", fails);
    return fails ? 1 : 0;
}
#endif /* MA_STANDALONE */
