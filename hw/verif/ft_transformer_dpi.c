/* =====================================================================
 * ft_transformer_dpi.c
 *
 * DPI-C glue between tb_ft_transformer.sv and the full-model C reference
 * (src/models/ft_transformer_cmodel.c). This is the ONLY place that knows about
 * the simulator (svdpi.h); the model stays hardware/sim agnostic. It assembles
 * the ft_config / ft_weights structs from the flat SV byte arrays and calls
 * ft_transformer_int8(), returning the wide int32 head accumulator.
 *
 * Fixed-size SV `byte` arrays map to C `char*` (signed 8-bit on every VCS
 * platform); we reinterpret as int8_t. The scalar head_b is a single `byte`.
 * =====================================================================
 */
#include "svdpi.h"
#include <stdint.h>
#include "ft_transformer_cmodel.h"

void ft_transformer_cmodel(
    int F, int seq_len, int d_token, int d_ffn, int n_heads, int n_blocks,
    int frac_bits, int res_frac, int scale_frac, int sm_frac, int recip_frac,
    int out_frac, long long scale, long long eps_v,
    const char *tok_w, const char *tok_b, const char *cls,
    const char *n1g, const char *n1b,
    const char *ipw, const char *ipb, const char *opw, const char *opb,
    const char *n2g, const char *n2b,
    const char *w1, const char *b1, const char *w2, const char *b2,
    const char *fng, const char *fnb,
    const char *hw, char hb,
    const char *x_feat,
    int *y)
{
    ft_config cfg = {
        .F = F, .seq_len = seq_len, .d_token = d_token, .d_ffn = d_ffn,
        .n_heads = n_heads, .n_blocks = n_blocks, .frac_bits = frac_bits,
        .res_frac = res_frac, .scale_frac = scale_frac, .sm_frac = sm_frac,
        .recip_frac = recip_frac, .out_frac = out_frac,
        .scale = scale, .eps_v = eps_v,
    };
    ft_weights w = {
        .tok_w = (const int8_t *)tok_w, .tok_b = (const int8_t *)tok_b,
        .cls = (const int8_t *)cls,
        .norm1_g = (const int8_t *)n1g, .norm1_b = (const int8_t *)n1b,
        .mha_ipw = (const int8_t *)ipw, .mha_ipb = (const int8_t *)ipb,
        .mha_opw = (const int8_t *)opw, .mha_opb = (const int8_t *)opb,
        .norm2_g = (const int8_t *)n2g, .norm2_b = (const int8_t *)n2b,
        .ffn_w1 = (const int8_t *)w1, .ffn_b1 = (const int8_t *)b1,
        .ffn_w2 = (const int8_t *)w2, .ffn_b2 = (const int8_t *)b2,
        .fnorm_g = (const int8_t *)fng, .fnorm_b = (const int8_t *)fnb,
        .head_w = (const int8_t *)hw, .head_b = (int8_t)hb,
    };
    *y = (int)ft_transformer_int8(&cfg, &w, (const int8_t *)x_feat);
}
