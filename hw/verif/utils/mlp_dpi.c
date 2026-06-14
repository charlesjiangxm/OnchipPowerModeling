/* =====================================================================
 * mlp_dpi.c
 *
 * DPI-C glue between tb_mlp.sv and the pure C reference model
 * (hw/model/mlp_cmodel.c). This is the ONLY place that knows about the
 * simulator (svdpi.h); the model itself stays hardware/sim agnostic.
 *
 * SystemVerilog import (see tb_mlp.sv):
 *   import "DPI-C" function void mlp_cmodel(
 *       input  int  n_features, input int hidden1, input int hidden2,
 *       input  byte x  [NF],
 *       input  byte w1 [W1X], input byte b1 [H1],
 *       input  byte w2 [W2X], input byte b2 [H2],
 *       input  byte w3 [W3X], input byte b3 [1],
 *       output byte y  [1],   output int  o_shift);
 *
 * A fixed-size SV `byte` array maps to a C `char*` (signed 8-bit on every VCS
 * platform), so we reinterpret as int8_t and call the core. x carries one 0/1
 * per feature (the 1-bit input). The y/o_shift outputs alias the SV objects, so
 * writing through them returns the result -- no copy-back call needed.
 * =====================================================================
 */
#include "svdpi.h"
#include <stdint.h>
#include "mlp_cmodel.h"

/* Exported to SystemVerilog. Name must match the `import "DPI-C"` declaration. */
void mlp_cmodel(int n_features, int hidden1, int hidden2,
                const char *x,
                const char *w1, const char *b1,
                const char *w2, const char *b2,
                const char *w3, const char *b3,
                char *y, int *o_shift)
{
    mlp_int8(n_features, hidden1, hidden2,
             (const int8_t *)x,
             (const int8_t *)w1, (const int8_t *)b1,
             (const int8_t *)w2, (const int8_t *)b2,
             (const int8_t *)w3, (const int8_t *)b3,
             (int8_t *)y, o_shift);
}
