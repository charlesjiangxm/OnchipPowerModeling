/* =====================================================================
 * ft_transformer_cmodel.c
 *
 * Behavioral C reference model for the FULL int8 FT-Transformer forward pass.
 * See ft_transformer_cmodel.h for the contract and numeric format. This file
 * is the GOLDEN composition spec: hw/rtl/ft_transformer_top.v runs the
 * identical integer datapath, so its output equals ft_transformer_int8()
 * bit-for-bit (the SV/DPI testbench checks this).
 *
 * It calls the four per-block twins unmodified and adds the composition glue
 * (cls inject, Q3.5 residual stream, head). Pure math only.
 *
 * Standalone self-test (no HDL simulator needed):
 *     gcc -DFT_STANDALONE -O2 -std=c11 \
 *         src/models/ft_transformer_cmodel.c \
 *         src/models/numerical_feature_tokenizer_cmodel.c \
 *         src/models/layer_norm_cmodel.c \
 *         src/models/multihead_attention_cmodel.c \
 *         src/models/feed_forward_network_cmodel.c -o /tmp/ftcm -lm
 *     /tmp/ftcm
 *   -> drives random (well-scaled) weights + inputs through the int8 model and
 *      a float ideal that shares the same dequantized weights; reports
 *      mean/max abs error and a pseudo-R^2, plus a residual-saturation monitor.
 *      Runs BOTH the Q3.5 stream and the Q1.7-saturating baseline so the
 *      accuracy trade-off is quantified directly.
 * =====================================================================
 */
#include "ft_transformer_cmodel.h"

#include "numerical_feature_tokenizer_cmodel.h"
#include "layer_norm_cmodel.h"
#include "multihead_attention_cmodel.h"
#include "feed_forward_network_cmodel.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(((int64_t)-1 >> 1) == (int64_t)-1,
               "ft_transformer_cmodel requires arithmetic right shift on int64_t");
#endif

/* compile-time scratch bounds (generous; override with -D) */
#ifndef FT_MAX_SEQ
#define FT_MAX_SEQ 64
#endif
#ifndef FT_MAX_D
#define FT_MAX_D 128
#endif

static int8_t ft_sat8(int64_t v)
{
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* residual-saturation monitor: zero-cost unless the self-test is compiled. */
#ifdef FT_STANDALONE
long ft_dbg_resid_total = 0;
long ft_dbg_resid_sat   = 0;
#endif

/* ---- composition-glue primitives (1:1 RTL twins) ---- */

int8_t ft_rescale(int8_t v, int from_frac, int to_frac)
{
    if (from_frac > to_frac) {
        int sh = from_frac - to_frac;                  /* round-half-up right shift */
        int64_t round_c = (int64_t)1 << (sh - 1);
        return ft_sat8(((int64_t)v + round_c) >> sh);
    } else if (from_frac < to_frac) {
        int sh = to_frac - from_frac;                  /* exact left shift */
        return ft_sat8((int64_t)v << sh);
    }
    return v;
}

int8_t ft_residual_add(int8_t stream, int8_t module_out, int res_frac, int frac_bits)
{
    /* align the Q1.frac module output down to the Q1.res_frac stream, add, sat */
    int sh = frac_bits - res_frac;                     /* = 2 for Q1.7 -> Q3.5 */
    int64_t m;
    if (sh > 0) {
        int64_t round_c = (int64_t)1 << (sh - 1);
        m = ((int64_t)module_out + round_c) >> sh;
    } else if (sh < 0) {
        m = (int64_t)module_out << (-sh);
    } else {
        m = (int64_t)module_out;
    }
    int64_t s = (int64_t)stream + m;
#ifdef FT_STANDALONE
    ft_dbg_resid_total++;
    if (s > 127 || s < -128) ft_dbg_resid_sat++;
#endif
    return ft_sat8(s);
}

int32_t ft_head_int8(int d_token, int frac_bits,
                     const int8_t *x, const int8_t *head_w, int8_t head_b)
{
    int64_t acc = 0;
    for (int k = 0; k < d_token; k++) {
        int xr = (x[k] < 0) ? 0 : x[k];                /* relu (exact) */
        acc += (int64_t)xr * (int64_t)head_w[k];
    }
    acc += (int64_t)head_b << frac_bits;               /* align bias to Q(2*frac) */
    return (int32_t)acc;                               /* wide output, no requant */
}

/* ---- full forward ---- */
int32_t ft_transformer_int8(const ft_config *cfg, const ft_weights *w,
                            const int8_t *x_feat)
{
    const int F  = cfg->F;
    const int S  = cfg->seq_len;
    const int E  = cfg->d_token;
    const int DF = cfg->d_ffn;
    const int FB = cfg->frac_bits;
    const int RF = cfg->res_frac;

    static int8_t seq[FT_MAX_SEQ * FT_MAX_D];          /* residual stream, Q1.res_frac */
    static int8_t tmp_in[FT_MAX_SEQ * FT_MAX_D];       /* per-stage Q1.frac input  */
    static int8_t tmp_out[FT_MAX_SEQ * FT_MAX_D];      /* per-stage Q1.frac output */
    static int8_t cls_norm[FT_MAX_D];

    /* 1. tokenizer (Q1.7) into rows 1..F of a temporary, then 2. inject. */
    static int8_t tok[FT_MAX_SEQ * FT_MAX_D];
    numerical_feature_tokenizer_int8(F, E, FB, x_feat, w->tok_w, w->tok_b, tok);

    /* 2. cls (already Q1.res_frac) at row 0; tokens rescaled Q1.7 -> Q1.res_frac. */
    for (int k = 0; k < E; k++) seq[0 * E + k] = w->cls[k];
    for (int j = 0; j < F; j++)
        for (int k = 0; k < E; k++)
            seq[(1 + j) * E + k] = ft_rescale(tok[j * E + k], FB, RF);

    /* 3. transformer blocks */
    for (int b = 0; b < cfg->n_blocks; b++) {
        const int8_t *ipw = w->mha_ipw + (size_t)b * 3 * E * E;
        const int8_t *ipb = w->mha_ipb + (size_t)b * 3 * E;
        const int8_t *opw = w->mha_opw + (size_t)b * E * E;
        const int8_t *opb = w->mha_opb + (size_t)b * E;

        /* --- attention branch: x_attn (Q1.7) --- */
        if (b == 0) {
            /* is_first: no norm1; rescale the Q3.5 stream down to Q1.7 */
            for (int i = 0; i < S * E; i++)
                tmp_in[i] = ft_rescale(seq[i], RF, FB);
        } else {
            const int8_t *g = w->norm1_g + (size_t)b * E;
            const int8_t *bt = w->norm1_b + (size_t)b * E;
            for (int s = 0; s < S; s++)
                layer_norm_int8(E, FB, cfg->eps_v, cfg->recip_frac, cfg->out_frac,
                                &seq[s * E], g, bt, &tmp_in[s * E]);
        }
        multihead_attention_int8(E, cfg->n_heads, S, FB,
                                 cfg->scale, cfg->scale_frac, cfg->sm_frac, cfg->recip_frac,
                                 tmp_in, ipw, ipb, opw, opb, tmp_out);
        /* residual add #1: seq (Q3.5) += mha_out (Q1.7) */
        for (int i = 0; i < S * E; i++)
            seq[i] = ft_residual_add(seq[i], tmp_out[i], RF, FB);

        /* --- FFN branch: norm2 (Q1.7) -> FFN (Q1.7) --- */
        {
            const int8_t *g = w->norm2_g + (size_t)b * E;
            const int8_t *bt = w->norm2_b + (size_t)b * E;
            for (int s = 0; s < S; s++)
                layer_norm_int8(E, FB, cfg->eps_v, cfg->recip_frac, cfg->out_frac,
                                &seq[s * E], g, bt, &tmp_in[s * E]);
        }
        {
            const int8_t *w1 = w->ffn_w1 + (size_t)b * DF * E;
            const int8_t *b1 = w->ffn_b1 + (size_t)b * DF;
            const int8_t *w2 = w->ffn_w2 + (size_t)b * E * DF;
            const int8_t *b2 = w->ffn_b2 + (size_t)b * E;
            for (int s = 0; s < S; s++)
                feed_forward_network_int8(E, DF, FB, &tmp_in[s * E],
                                          w1, b1, w2, b2, &tmp_out[s * E]);
        }
        /* residual add #2 */
        for (int i = 0; i < S * E; i++)
            seq[i] = ft_residual_add(seq[i], tmp_out[i], RF, FB);
    }

    /* 4. final_norm on token 0 only (Q3.5 stream -> Q1.7) */
    layer_norm_int8(E, FB, cfg->eps_v, cfg->recip_frac, cfg->out_frac,
                    &seq[0], w->fnorm_g, w->fnorm_b, cls_norm);

    /* 5. head: relu + dot + bias -> wide int32 Q(2*frac) */
    return ft_head_int8(E, FB, cls_norm, w->head_w, w->head_b);
}

#ifdef FT_STANDALONE
/* ---------------------------------------------------------------------
 * Standalone self-test (compiled only with -DFT_STANDALONE).
 *
 * Generates well-scaled random float weights, quantizes them (Q1.7; cls at
 * Q1.res_frac), and compares the int8 model's dequantized scalar against a
 * float ideal that uses the SAME dequantized weights (so the comparison
 * isolates the integer arithmetic + residual clipping + softmax/sqrt
 * approximations, not the weight-quantization noise). Runs the Q3.5 stream
 * and the Q1.7-saturating baseline so the residual trade-off is explicit.
 * ------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern long ft_dbg_resid_total, ft_dbg_resid_sat;

/* ---- config under test ---- */
#define TF   16        /* features        */
#define TE   32        /* d_token         */
#define TFFN 64        /* d_ffn           */
#define TH   4         /* n_heads (HD=8)  */
#define TB   3         /* n_blocks        */
#define TS   (1 + TF)  /* seq_len = 17    */
#define FRAC 7

/* deterministic LCG -> uniform(-1,1) and approx-normal via sum of uniforms */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static double urand(void)
{
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    uint32_t hi = (uint32_t)(rng_state >> 32);   /* top 32 bits */
    return (double)hi / 4294967296.0;            /* [0,1) */
}
static double uniform(double a, double b) { return a + (b - a) * urand(); }
static double nrand(double sd)                /* ~N(0, sd^2) via CLT(12) */
{
    double s = 0.0;
    for (int i = 0; i < 12; i++) s += urand();
    return (s - 6.0) * sd;
}

static int8_t qf(double v, int frac)
{
    long r = (long)floor(v * (double)(1 << frac) + 0.5);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* ---- float reference blocks (use dequantized weights) ---- */
static void layernorm_f(int D, const double *x, const double *g, const double *b,
                        double *y, double eps)
{
    double mean = 0.0;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;
    double var = 0.0;
    for (int i = 0; i < D; i++) { double d = x[i] - mean; var += d * d; }
    var /= D;
    double inv = 1.0 / sqrt(var + eps);
    for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

/* float self-attention, q=k=v=x; weights laid out like PyTorch in_proj. */
static void mha_f(int S, int E, int H, const double *x,
                  const double *ipw, const double *ipb,
                  const double *opw, const double *opb, double *y)
{
    int HD = E / H;
    static double Q[TS * TE], K[TS * TE], V[TS * TE], ctx[TS * TE];
    for (int s = 0; s < S; s++)
        for (int e = 0; e < E; e++) {
            double aq = ipb[e], ak = ipb[E + e], av = ipb[2 * E + e];
            for (int k = 0; k < E; k++) {
                double xv = x[s * E + k];
                aq += xv * ipw[(e) * E + k];
                ak += xv * ipw[(E + e) * E + k];
                av += xv * ipw[(2 * E + e) * E + k];
            }
            Q[s * E + e] = aq; K[s * E + e] = ak; V[s * E + e] = av;
        }
    double inv_sqrt_hd = 1.0 / sqrt((double)HD);
    for (int h = 0; h < H; h++)
        for (int qi = 0; qi < S; qi++) {
            double sc[TS], mx = -1e300, sum = 0.0;
            for (int kj = 0; kj < S; kj++) {
                double dot = 0.0;
                for (int d = 0; d < HD; d++)
                    dot += Q[qi * E + h * HD + d] * K[kj * E + h * HD + d];
                sc[kj] = dot * inv_sqrt_hd;
                if (sc[kj] > mx) mx = sc[kj];
            }
            for (int kj = 0; kj < S; kj++) { sc[kj] = exp(sc[kj] - mx); sum += sc[kj]; }
            for (int d = 0; d < HD; d++) {
                double acc = 0.0;
                for (int kj = 0; kj < S; kj++)
                    acc += (sc[kj] / sum) * V[kj * E + h * HD + d];
                ctx[qi * E + h * HD + d] = acc;
            }
        }
    for (int s = 0; s < S; s++)
        for (int e = 0; e < E; e++) {
            double acc = opb[e];
            for (int k = 0; k < E; k++) acc += ctx[s * E + k] * opw[e * E + k];
            y[s * E + e] = acc;
        }
}

static void ffn_f(int E, int DF, const double *x,
                  const double *w1, const double *b1,
                  const double *w2, const double *b2, double *y)
{
    static double h[4096];
    for (int o = 0; o < DF; o++) {
        double acc = b1[o];
        for (int k = 0; k < E; k++) acc += x[k] * w1[o * E + k];
        h[o] = (acc < 0.0) ? 0.0 : acc;                /* ReLU */
    }
    for (int o = 0; o < E; o++) {
        double acc = b2[o];
        for (int k = 0; k < DF; k++) acc += h[k] * w2[o * DF + k];
        y[o] = acc;
    }
}

/* float forward mirroring ft_transformer_int8's structure (ReLU FFN, real
 * softmax). Uses dequantized weights so only arithmetic/approx error remains. */
static double forward_f(const double *x, const double *tok_w, const double *tok_b,
                        const double *cls,
                        const double *n1g, const double *n1b,
                        const double *ipw, const double *ipb,
                        const double *opw, const double *opb,
                        const double *n2g, const double *n2b,
                        const double *w1, const double *b1,
                        const double *w2, const double *b2,
                        const double *fng, const double *fnb,
                        const double *hw, double hb)
{
    static double seq[TS * TE], a[TS * TE], o[TS * TE], cn[TE];
    const double eps = 1e-5;
    for (int k = 0; k < TE; k++) seq[k] = cls[k];
    for (int j = 0; j < TF; j++)
        for (int k = 0; k < TE; k++)
            seq[(1 + j) * TE + k] = x[j] * tok_w[j * TE + k] + tok_b[j * TE + k];

    for (int b = 0; b < TB; b++) {
        if (b == 0) memcpy(a, seq, sizeof(double) * TS * TE);
        else for (int s = 0; s < TS; s++)
            layernorm_f(TE, &seq[s * TE], &n1g[b * TE], &n1b[b * TE], &a[s * TE], eps);
        mha_f(TS, TE, TH, a, &ipw[(size_t)b * 3 * TE * TE], &ipb[(size_t)b * 3 * TE],
              &opw[(size_t)b * TE * TE], &opb[(size_t)b * TE], o);
        for (int i = 0; i < TS * TE; i++) seq[i] += o[i];
        for (int s = 0; s < TS; s++)
            layernorm_f(TE, &seq[s * TE], &n2g[b * TE], &n2b[b * TE], &a[s * TE], eps);
        for (int s = 0; s < TS; s++)
            ffn_f(TE, TFFN, &a[s * TE], &w1[(size_t)b * TFFN * TE], &b1[(size_t)b * TFFN],
                  &w2[(size_t)b * TE * TFFN], &b2[(size_t)b * TE], &o[s * TE]);
        for (int i = 0; i < TS * TE; i++) seq[i] += o[i];
    }
    layernorm_f(TE, &seq[0], fng, fnb, cn, eps);
    double y = hb;
    for (int k = 0; k < TE; k++) { double r = (cn[k] < 0) ? 0 : cn[k]; y += r * hw[k]; }
    return y;
}

/* float buffers (module weights) and their int8 quantizations */
static double f_tok_w[TF * TE], f_tok_b[TF * TE], f_cls[TE];
static double f_n1g[TB * TE], f_n1b[TB * TE], f_n2g[TB * TE], f_n2b[TB * TE];
static double f_ipw[TB * 3 * TE * TE], f_ipb[TB * 3 * TE];
static double f_opw[TB * TE * TE], f_opb[TB * TE];
static double f_w1[TB * TFFN * TE], f_b1[TB * TFFN], f_w2[TB * TE * TFFN], f_b2[TB * TE];
static double f_fng[TE], f_fnb[TE], f_hw[TE], f_hb;

static int8_t q_tok_w[TF * TE], q_tok_b[TF * TE], q_cls[TE];
static int8_t q_n1g[TB * TE], q_n1b[TB * TE], q_n2g[TB * TE], q_n2b[TB * TE];
static int8_t q_ipw[TB * 3 * TE * TE], q_ipb[TB * 3 * TE];
static int8_t q_opw[TB * TE * TE], q_opb[TB * TE];
static int8_t q_w1[TB * TFFN * TE], q_b1[TB * TFFN], q_w2[TB * TE * TFFN], q_b2[TB * TE];
static int8_t q_fng[TE], q_fnb[TE], q_hw[TE], q_hb;

static void gen_weights(void)
{
    double we = 1.0 / sqrt((double)TE);
    double wf = 1.0 / sqrt((double)TFFN);
    for (int i = 0; i < TF * TE; i++) { f_tok_w[i] = nrand(0.5); f_tok_b[i] = nrand(0.1); }
    for (int k = 0; k < TE; k++) f_cls[k] = nrand(0.3);
    for (int i = 0; i < TB * TE; i++) {
        f_n1g[i] = 1.0 + nrand(0.05); f_n1b[i] = nrand(0.05);
        f_n2g[i] = 1.0 + nrand(0.05); f_n2b[i] = nrand(0.05);
    }
    for (int i = 0; i < TB * 3 * TE * TE; i++) f_ipw[i] = nrand(we);
    for (int i = 0; i < TB * 3 * TE; i++)      f_ipb[i] = nrand(0.05);
    for (int i = 0; i < TB * TE * TE; i++)     f_opw[i] = nrand(we);
    for (int i = 0; i < TB * TE; i++)          f_opb[i] = nrand(0.05);
    for (int i = 0; i < TB * TFFN * TE; i++)   f_w1[i]  = nrand(we);
    for (int i = 0; i < TB * TFFN; i++)        f_b1[i]  = nrand(0.05);
    for (int i = 0; i < TB * TE * TFFN; i++)   f_w2[i]  = nrand(wf);
    for (int i = 0; i < TB * TE; i++)          f_b2[i]  = nrand(0.05);
    for (int k = 0; k < TE; k++) { f_fng[k] = 1.0 + nrand(0.05); f_fnb[k] = nrand(0.05); }
    for (int k = 0; k < TE; k++) f_hw[k] = nrand(we);
    f_hb = nrand(0.05);
}

static void quantize_weights(int res_frac)
{
    for (int i = 0; i < TF * TE; i++) { q_tok_w[i] = qf(f_tok_w[i], FRAC); q_tok_b[i] = qf(f_tok_b[i], FRAC); }
    for (int k = 0; k < TE; k++) q_cls[k] = qf(f_cls[k], res_frac);   /* cls at stream frac */
    for (int i = 0; i < TB * TE; i++) {
        q_n1g[i] = qf(f_n1g[i], FRAC); q_n1b[i] = qf(f_n1b[i], FRAC);
        q_n2g[i] = qf(f_n2g[i], FRAC); q_n2b[i] = qf(f_n2b[i], FRAC);
    }
    for (int i = 0; i < TB * 3 * TE * TE; i++) q_ipw[i] = qf(f_ipw[i], FRAC);
    for (int i = 0; i < TB * 3 * TE; i++)      q_ipb[i] = qf(f_ipb[i], FRAC);
    for (int i = 0; i < TB * TE * TE; i++)     q_opw[i] = qf(f_opw[i], FRAC);
    for (int i = 0; i < TB * TE; i++)          q_opb[i] = qf(f_opb[i], FRAC);
    for (int i = 0; i < TB * TFFN * TE; i++)   q_w1[i]  = qf(f_w1[i], FRAC);
    for (int i = 0; i < TB * TFFN; i++)        q_b1[i]  = qf(f_b1[i], FRAC);
    for (int i = 0; i < TB * TE * TFFN; i++)   q_w2[i]  = qf(f_w2[i], FRAC);
    for (int i = 0; i < TB * TE; i++)          q_b2[i]  = qf(f_b2[i], FRAC);
    for (int k = 0; k < TE; k++) { q_fng[k] = qf(f_fng[k], FRAC); q_fnb[k] = qf(f_fnb[k], FRAC); }
    for (int k = 0; k < TE; k++) q_hw[k] = qf(f_hw[k], FRAC);
    q_hb = qf(f_hb, FRAC);
}

/* dequantized-weight float views (so the float ideal uses identical weights) */
static double d_tok_w[TF * TE], d_tok_b[TF * TE], d_cls[TE];
static double d_n1g[TB * TE], d_n1b[TB * TE], d_n2g[TB * TE], d_n2b[TB * TE];
static double d_ipw[TB * 3 * TE * TE], d_ipb[TB * 3 * TE];
static double d_opw[TB * TE * TE], d_opb[TB * TE];
static double d_w1[TB * TFFN * TE], d_b1[TB * TFFN], d_w2[TB * TE * TFFN], d_b2[TB * TE];
static double d_fng[TE], d_fnb[TE], d_hw[TE], d_hb;

#define DEQ(dst, src, n, frac) do { for (int _i = 0; _i < (n); _i++) (dst)[_i] = (double)(src)[_i] / (double)(1 << (frac)); } while (0)

static void dequantize_weights(int res_frac)
{
    DEQ(d_tok_w, q_tok_w, TF * TE, FRAC); DEQ(d_tok_b, q_tok_b, TF * TE, FRAC);
    DEQ(d_cls, q_cls, TE, res_frac);
    DEQ(d_n1g, q_n1g, TB * TE, FRAC); DEQ(d_n1b, q_n1b, TB * TE, FRAC);
    DEQ(d_n2g, q_n2g, TB * TE, FRAC); DEQ(d_n2b, q_n2b, TB * TE, FRAC);
    DEQ(d_ipw, q_ipw, TB * 3 * TE * TE, FRAC); DEQ(d_ipb, q_ipb, TB * 3 * TE, FRAC);
    DEQ(d_opw, q_opw, TB * TE * TE, FRAC); DEQ(d_opb, q_opb, TB * TE, FRAC);
    DEQ(d_w1, q_w1, TB * TFFN * TE, FRAC); DEQ(d_b1, q_b1, TB * TFFN, FRAC);
    DEQ(d_w2, q_w2, TB * TE * TFFN, FRAC); DEQ(d_b2, q_b2, TB * TE, FRAC);
    DEQ(d_fng, q_fng, TE, FRAC); DEQ(d_fnb, q_fnb, TE, FRAC);
    DEQ(d_hw, q_hw, TE, FRAC); d_hb = (double)q_hb / (double)(1 << FRAC);
}

static void run_variant(int res_frac, const char *label, int n_samples)
{
    quantize_weights(res_frac);
    dequantize_weights(res_frac);

    ft_config cfg = {
        .F = TF, .seq_len = TS, .d_token = TE, .d_ffn = TFFN, .n_heads = TH,
        .n_blocks = TB, .frac_bits = FRAC, .res_frac = res_frac,
        .scale_frac = 14, .sm_frac = 8, .recip_frac = 24, .out_frac = FRAC,
        .scale = mha_scale(TH, TE, 14),
        .eps_v = layer_norm_eps_v(TE, FRAC, 1e-5),
    };
    ft_weights w = {
        .tok_w = q_tok_w, .tok_b = q_tok_b, .cls = q_cls,
        .norm1_g = q_n1g, .norm1_b = q_n1b,
        .mha_ipw = q_ipw, .mha_ipb = q_ipb, .mha_opw = q_opw, .mha_opb = q_opb,
        .norm2_g = q_n2g, .norm2_b = q_n2b,
        .ffn_w1 = q_w1, .ffn_b1 = q_b1, .ffn_w2 = q_w2, .ffn_b2 = q_b2,
        .fnorm_g = q_fng, .fnorm_b = q_fnb, .head_w = q_hw, .head_b = q_hb,
    };

    ft_dbg_resid_total = 0; ft_dbg_resid_sat = 0;
    double sse = 0.0, sabs = 0.0, maxabs = 0.0;
    double sum_yf = 0.0, sum_yf2 = 0.0, sum_yi = 0.0, sum_yi2 = 0.0, sum_yiyf = 0.0;
    double xf[TF];
    int8_t xq[TF];

    for (int n = 0; n < n_samples; n++) {
        for (int j = 0; j < TF; j++) { xf[j] = nrand(1.0); xq[j] = qf(xf[j], FRAC); }
        /* the int8 path sees the QUANTIZED input; the float ideal sees the
         * same dequantized input so only the network arithmetic differs. */
        double xd[TF];
        for (int j = 0; j < TF; j++) xd[j] = (double)xq[j] / (double)(1 << FRAC);

        int32_t yo = ft_transformer_int8(&cfg, &w, xq);
        double yi = (double)yo / (double)(1 << (2 * FRAC));   /* dequant Q14 */
        double yf = forward_f(xd, d_tok_w, d_tok_b, d_cls, d_n1g, d_n1b,
                              d_ipw, d_ipb, d_opw, d_opb, d_n2g, d_n2b,
                              d_w1, d_b1, d_w2, d_b2, d_fng, d_fnb, d_hw, d_hb);
        double e = yi - yf; if (e < 0) e = -e;
        sse += (yi - yf) * (yi - yf); sabs += e; if (e > maxabs) maxabs = e;
        sum_yf += yf; sum_yf2 += yf * yf;
        sum_yi += yi; sum_yi2 += yi * yi; sum_yiyf += yi * yf;
    }
    double mean_yf = sum_yf / n_samples, mean_yi = sum_yi / n_samples;
    double var_yf = sum_yf2 / n_samples - mean_yf * mean_yf;
    double var_yi = sum_yi2 / n_samples - mean_yi * mean_yi;
    double cov = sum_yiyf / n_samples - mean_yi * mean_yf;
    double corr = (var_yi > 1e-18 && var_yf > 1e-18) ? cov / sqrt(var_yi * var_yf) : 0.0;
    double sstot = sum_yf2 - n_samples * mean_yf * mean_yf;
    double r2 = (sstot > 1e-12) ? (1.0 - sse / sstot) : 0.0;
    double sat_pct = (ft_dbg_resid_total > 0)
                   ? 100.0 * (double)ft_dbg_resid_sat / (double)ft_dbg_resid_total : 0.0;

    printf("  %-22s : mean|err|=%.4f max|err|=%.4f R2=%.4f corr=%.4f "
           "std(yi)=%.4f std(yf)=%.4f resid-sat=%.2f%%\n",
           label, sabs / n_samples, maxabs, r2, corr,
           sqrt(var_yi > 0 ? var_yi : 0), sqrt(var_yf > 0 ? var_yf : 0), sat_pct);
}

/* ---- bit-exact gate: glue primitives vs hand-computed constants ---- */
static int check_glue(void)
{
    struct { int8_t v; int from, to; int8_t exp; } rc[] = {
        {127, 7, 5,  32}, { -1, 7, 5,   0}, { -2, 7, 5,   0}, { -6, 7, 5,  -1},
        { 31, 5, 7, 124}, { 40, 5, 7, 127}, {-40, 5, 7,-128}, {  5, 7, 7,   5},
    };
    int fails = 0;
    for (unsigned i = 0; i < sizeof(rc) / sizeof(rc[0]); i++) {
        int8_t got = ft_rescale(rc[i].v, rc[i].from, rc[i].to);
        if (got != rc[i].exp) {
            printf("  FAIL ft_rescale(%d,%d,%d)=%d exp %d\n",
                   rc[i].v, rc[i].from, rc[i].to, got, rc[i].exp);
            fails++;
        }
    }
    struct { int8_t s, m; int8_t exp; } ra[] = {
        {100, 127,  127}, {-100, -128, -128}, {10, 4, 11}, {0, 2, 1},
        {-3, -2, -3},   /* (-2+2)>>2 = 0  -> -3+0  */
        {-3, -6, -4},   /* (-6+2)>>2 = -1 -> -3-1  */
    };
    for (unsigned i = 0; i < sizeof(ra) / sizeof(ra[0]); i++) {
        int8_t got = ft_residual_add(ra[i].s, ra[i].m, 5, 7);
        if (got != ra[i].exp) {
            printf("  FAIL ft_residual_add(%d,%d,5,7)=%d exp %d\n",
                   ra[i].s, ra[i].m, got, ra[i].exp);
            fails++;
        }
    }
    {   /* head: relu([10,-5,20,30])=[10,0,20,30] . [2,3,4,5] + (8<<7) = 250+1024 */
        int8_t x[4] = {10, -5, 20, 30}, hw[4] = {2, 3, 4, 5};
        int32_t got = ft_head_int8(4, 7, x, hw, 8), exp = 1274;
        if (got != exp) { printf("  FAIL ft_head_int8 = %d exp %d\n", got, exp); fails++; }
    }
    printf("  glue primitives (rescale / residual_add / head) : %s\n",
           fails ? "FAIL" : "bit-exact vs hand-computed");
    return fails;
}

/* ---- bit-exact gate: zeroed attn/FFN must pass the stream through, so the
 * output equals head(final_norm(cls)) computed independently. Validates the
 * tokenizer/cls-inject/final-norm-on-row-0/head wiring and residual identity. */
static int check_passthrough(void)
{
    quantize_weights(5);
    /* zero every attention + FFN weight/bias so each block is identity on seq */
    memset(q_ipw, 0, sizeof q_ipw); memset(q_ipb, 0, sizeof q_ipb);
    memset(q_opw, 0, sizeof q_opw); memset(q_opb, 0, sizeof q_opb);
    memset(q_w1,  0, sizeof q_w1);  memset(q_b1,  0, sizeof q_b1);
    memset(q_w2,  0, sizeof q_w2);  memset(q_b2,  0, sizeof q_b2);

    ft_config cfg = {
        .F = TF, .seq_len = TS, .d_token = TE, .d_ffn = TFFN, .n_heads = TH,
        .n_blocks = TB, .frac_bits = FRAC, .res_frac = 5,
        .scale_frac = 14, .sm_frac = 8, .recip_frac = 24, .out_frac = FRAC,
        .scale = mha_scale(TH, TE, 14), .eps_v = layer_norm_eps_v(TE, FRAC, 1e-5),
    };
    ft_weights w = {
        .tok_w = q_tok_w, .tok_b = q_tok_b, .cls = q_cls,
        .norm1_g = q_n1g, .norm1_b = q_n1b,
        .mha_ipw = q_ipw, .mha_ipb = q_ipb, .mha_opw = q_opw, .mha_opb = q_opb,
        .norm2_g = q_n2g, .norm2_b = q_n2b,
        .ffn_w1 = q_w1, .ffn_b1 = q_b1, .ffn_w2 = q_w2, .ffn_b2 = q_b2,
        .fnorm_g = q_fng, .fnorm_b = q_fnb, .head_w = q_hw, .head_b = q_hb,
    };

    int fails = 0;
    int8_t xq[TF];
    for (int t = 0; t < 64; t++) {
        for (int j = 0; j < TF; j++) xq[j] = qf(nrand(1.0), FRAC);
        int32_t got = ft_transformer_int8(&cfg, &w, xq);
        /* independent expected: blocks are identity, so seq[0] stays = cls */
        int8_t cls_norm[TE];
        layer_norm_int8(TE, FRAC, cfg.eps_v, cfg.recip_frac, cfg.out_frac,
                        q_cls, q_fng, q_fnb, cls_norm);
        int32_t exp = ft_head_int8(TE, FRAC, cls_norm, q_hw, q_hb);
        if (got != exp) {
            printf("  FAIL passthrough t=%d: got %d exp %d\n", t, got, exp);
            if (++fails > 4) break;
        }
    }
    printf("  zeroed-block pass-through (cls -> final_norm -> head)   : %s\n",
           fails ? "FAIL" : "bit-exact");
    return fails;
}

int main(void)
{
    const int N = 2000;
    printf("ft_transformer_cmodel self-test (int8 composition vs float ideal, "
           "shared dequantized weights)\n");
    printf("config: F=%d seq_len=%d d_token=%d d_ffn=%d n_heads=%d n_blocks=%d "
           "frac=%d  SCALE=%lld EPS_V=%lld\n",
           TF, TS, TE, TFFN, TH, TB, FRAC, mha_scale(TH, TE, 14),
           layer_norm_eps_v(TE, FRAC, 1e-5));
    printf("------------------------------------------------------------------\n");
    gen_weights();

    printf("[1] bit-exact correctness gate\n");
    int fails = 0;
    fails += check_glue();
    fails += check_passthrough();

    printf("[2] accuracy sanity (informational; random UNTRAINED weights -> "
           "low-signal cls output, so corr/R2 are weak by nature)\n");
    run_variant(5, "Q3.5 stream (res_frac=5)", N);   /* recommended */
    run_variant(7, "Q1.7 baseline (res_frac=7)", N); /* saturating lower bound */
    printf("------------------------------------------------------------------\n");
    printf("note: [2] error is int8-vs-float (ReLU/softmax/sqrt approx + residual\n"
           "clipping). The Q3.5 vs Q1.7 resid-sat gap is the design evidence; the\n"
           "real RTL gate is the bit-exact DPI testbench against this c-model.\n");
    if (fails == 0) printf("PASS: composition bit-exact gate clean.\n");
    else            printf("FAIL: %d composition error(s).\n", fails);
    return fails ? 1 : 0;
}
#endif /* FT_STANDALONE */
