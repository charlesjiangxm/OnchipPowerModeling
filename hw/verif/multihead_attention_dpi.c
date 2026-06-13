/* =====================================================================
 * multihead_attention_dpi.c
 *
 * DPI-C glue between tb_multihead_attention.sv and the pure C reference model
 * (src/models/multihead_attention_cmodel.c). This is the ONLY place that knows
 * about the simulator (svdpi.h); the model itself stays hardware/sim agnostic.
 *
 * SystemVerilog import (see tb_multihead_attention.sv):
 *   import "DPI-C" function void multihead_attention_cmodel(
 *       input  int     d_token, input int n_heads, input int seq_len,
 *       input  int     frac_bits, input longint scale, input int scale_frac,
 *       input  int     sm_frac, input int recip_frac,
 *       input  byte    x   [S*E],
 *       input  byte    ipw [3*E*E], input byte ipb [3*E],
 *       input  byte    opw [E*E],   input byte opb [E],
 *       output byte    y   [S*E]);
 *
 * A fixed-size SV `byte` array maps to a C `char*` (signed 8-bit on every VCS
 * platform), so we just reinterpret as int8_t and call the core. The output
 * pointer aliases the SV array, so writing through it returns the result --
 * no copy-back call needed.
 * =====================================================================
 */
#include "svdpi.h"
#include <stdint.h>
#include "multihead_attention_cmodel.h"

/* Exported to SystemVerilog. Name must match the `import "DPI-C"` declaration. */
void multihead_attention_cmodel(int d_token, int n_heads, int seq_len,
                                int frac_bits, long long scale, int scale_frac,
                                int sm_frac, int recip_frac,
                                const char *x,
                                const char *ipw, const char *ipb,
                                const char *opw, const char *opb,
                                char *y)
{
    multihead_attention_int8(d_token, n_heads, seq_len, frac_bits,
                             scale, scale_frac, sm_frac, recip_frac,
                             (const int8_t *)x,
                             (const int8_t *)ipw, (const int8_t *)ipb,
                             (const int8_t *)opw, (const int8_t *)opb,
                             (int8_t *)y);
}
