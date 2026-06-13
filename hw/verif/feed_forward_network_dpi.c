/* =====================================================================
 * feed_forward_network_dpi.c
 *
 * DPI-C glue between tb_feed_forward_network.sv and the pure C reference model
 * (src/models/feed_forward_network_cmodel.c). This is the ONLY place that knows
 * about the simulator (svdpi.h); the model itself stays hardware/sim agnostic.
 *
 * SystemVerilog import (see tb_feed_forward_network.sv):
 *   import "DPI-C" function void feed_forward_network_cmodel(
 *       input  int  d_token, input int d_ffn, input int frac_bits,
 *       input  byte x  [DT],
 *       input  byte w1 [W1X], input byte b1 [B1X],
 *       input  byte w2 [W2X], input byte b2 [B2X],
 *       output byte y  [DT]);
 *
 * A fixed-size SV `byte` array maps to a C `char*` (signed 8-bit on every VCS
 * platform), so we just reinterpret as int8_t and call the core. The output
 * pointer aliases the SV array, so writing through it returns the result --
 * no copy-back call needed.
 * =====================================================================
 */
#include "svdpi.h"
#include <stdint.h>
#include "feed_forward_network_cmodel.h"

/* Exported to SystemVerilog. Name must match the `import "DPI-C"` declaration. */
void feed_forward_network_cmodel(int d_token, int d_ffn, int frac_bits,
                                 const char *x,
                                 const char *w1, const char *b1,
                                 const char *w2, const char *b2,
                                 char *y)
{
    feed_forward_network_int8(d_token, d_ffn, frac_bits,
                              (const int8_t *)x,
                              (const int8_t *)w1, (const int8_t *)b1,
                              (const int8_t *)w2, (const int8_t *)b2,
                              (int8_t *)y);
}
