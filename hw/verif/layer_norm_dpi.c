/* =====================================================================
 * layer_norm_dpi.c
 *
 * DPI-C glue between tb_layer_norm.sv and the pure C reference model
 * (src/models/layer_norm_cmodel.c). This is the ONLY place that knows about
 * the simulator (svdpi.h); the model itself stays hardware/sim agnostic.
 *
 * SystemVerilog import (see tb_layer_norm.sv):
 *   import "DPI-C" function void layer_norm_cmodel(
 *       input  int     D, input int frac_bits, input longint eps_v,
 *       input  int     recip_frac, input int out_frac,
 *       input  byte    x [DT], input byte gamma [DT], input byte beta [DT],
 *       output byte    y [DT]);
 *
 * A fixed-size SV `byte` array maps to a C `char*` (signed 8-bit on every VCS
 * platform), so we just reinterpret as int8_t and call the core. The output
 * pointer aliases the SV array, so writing through it returns the result --
 * no copy-back call needed.
 * =====================================================================
 */
#include "svdpi.h"
#include <stdint.h>
#include "layer_norm_cmodel.h"

/* Exported to SystemVerilog. Name must match the `import "DPI-C"` declaration. */
void layer_norm_cmodel(int D, int frac_bits, long long eps_v,
                       int recip_frac, int out_frac,
                       const char *x, const char *gamma, const char *beta,
                       char *y)
{
    layer_norm_int8(D, frac_bits, eps_v, recip_frac, out_frac,
                    (const int8_t *)x, (const int8_t *)gamma,
                    (const int8_t *)beta, (int8_t *)y);
}
