#!/usr/bin/env python3
"""Cycle-accurate behavioral twin of numerical_feature_tokenizer.v.

This is a software mirror of the RTL's *integer* datapath and 3-stage
pipeline. It exists so the int8 arithmetic and the II=1 streaming behaviour
can be validated without an HDL simulator, and so the same quantization can
be reused to generate / check coefficient values for the hardware.

It checks three things:
  1. The pipeline emits one output row per input row, contiguously (II = 1).
  2. The integer result matches an independent wide-integer golden.
  3. The int8 result, dequantized, matches the float NumericalFeatureTokenizer
     op  out = x*W + b  in Q1.FRAC space to within one LSB (2**-FRAC).

Run:  python3 hw/ref_model.py
"""
from __future__ import annotations
import numpy as np

DW = 8          # DATA_WIDTH (int8)
FRAC = 7        # FRAC_BITS  (Q1.7)
OUT_MAX = (1 << (DW - 1)) - 1     # +127
OUT_MIN = -(1 << (DW - 1))        # -128
ROUND = (1 << (FRAC - 1)) if FRAC > 0 else 0


def requant(acc: int) -> int:
    """Round-half-up, arithmetic right shift by FRAC, saturate to int8.

    Matches the RTL `requant` function exactly (Python `>>` on ints is an
    arithmetic floor shift, so (acc+ROUND)>>FRAC is round-half-up)."""
    r = (acc + ROUND) >> FRAC
    if r > OUT_MAX:
        r = OUT_MAX
    elif r < OUT_MIN:
        r = OUT_MIN
    return r


def lane(x_j: int, w: int, b: int) -> int:
    """One (j,k) lane: product + aligned bias, then requantize -> int8."""
    acc = x_j * w + (b << FRAC)   # == align_bias(b) + product, in the RTL
    return requant(acc)


class TokenizerPipe:
    """3-stage pipeline twin: latch x -> multiply -> bias+requant.

    step() consumes one (in_valid, x_row) per call and returns the registered
    (out_valid, out_tokens) *after* this clock edge -- the same observation
    point as a `@(posedge clk)` monitor in the testbench."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.NF, self.DT = weight.shape
        self.W = weight.astype(np.int64)
        self.B = bias.astype(np.int64)
        self.x_q = np.zeros(self.NF, dtype=np.int64)
        self.prod_q = np.zeros((self.NF, self.DT), dtype=np.int64)
        self.tok_q = np.zeros((self.NF, self.DT), dtype=np.int64)
        self.v1 = self.v2 = self.v3 = 0

    def step(self, in_valid: int, x_row: np.ndarray):
        # next-state (all reads use current registers, like nonblocking <=)
        nx = x_row.astype(np.int64).copy()
        nprod = self.x_q[:, None] * self.W
        nacc = self.prod_q + (self.B << FRAC)
        ntok = np.vectorize(requant)(nacc)
        nv1, nv2, nv3 = in_valid, self.v1, self.v2
        # commit
        self.x_q, self.prod_q, self.tok_q = nx, nprod, ntok
        self.v1, self.v2, self.v3 = nv1, nv2, nv3
        return self.v3, self.tok_q.copy()


def main() -> int:
    rng = np.random.default_rng(0)
    NF, DT = 4, 4
    NROWS = 12

    # Same coefficient patterns as the Verilog testbench (wrapped into int8).
    idx = np.arange(NF * DT)
    W = ((idx * 7 - 13 + 128) % 256 - 128).reshape(NF, DT).astype(np.int64)
    B = ((5 - idx * 3 + 128) % 256 - 128).reshape(NF, DT).astype(np.int64)

    # Stimulus: +sat row, -sat row, then random int8.
    rows = [np.full(NF, 127), np.full(NF, -128)]
    rows += [rng.integers(-128, 128, size=NF) for _ in range(NROWS - 2)]

    pipe = TokenizerPipe(W, B)
    outputs, out_cycles = [], []
    drive = rows + [None] * 6           # trailing bubbles to drain
    for cyc, r in enumerate(drive):
        ov, tok = pipe.step(1 if r is not None else 0,
                            r if r is not None else np.zeros(NF, np.int64))
        if ov:
            outputs.append(tok)
            out_cycles.append(cyc)

    errors = 0

    # (1) II = 1: one output per input, contiguous.
    if len(outputs) != NROWS:
        errors += 1
        print(f"FAIL: {len(outputs)} outputs, expected {NROWS}")
    elif out_cycles[-1] - out_cycles[0] + 1 != NROWS:
        errors += 1
        print(f"FAIL: outputs not contiguous (II != 1): span="
              f"{out_cycles[-1]-out_cycles[0]+1} for {NROWS} beats")

    # (2) integer golden + (3) float faithfulness.
    max_lsb_err = 0.0
    for n in range(min(len(outputs), NROWS)):
        x = rows[n]
        for j in range(NF):
            for k in range(DT):
                ref = lane(int(x[j]), int(W[j, k]), int(B[j, k]))
                hw = int(outputs[n][j, k])
                if hw != ref:
                    errors += 1
                    if errors <= 20:
                        print(f"  MISMATCH row={n} j={j} k={k} hw={hw} ref={ref}")
                # float NumericalFeatureTokenizer op in Q1.FRAC space
                xf, wf, bf = x[j] / 2**FRAC, W[j, k] / 2**FRAC, B[j, k] / 2**FRAC
                fout = xf * wf + bf
                # compare dequantized hw to the float op (skip saturated cells)
                if OUT_MIN < ref < OUT_MAX:
                    max_lsb_err = max(max_lsb_err,
                                      abs(hw / 2**FRAC - fout) * 2**FRAC)

    if max_lsb_err > 0.5 + 1e-9:
        errors += 1
        print(f"FAIL: dequantized error {max_lsb_err:.3f} LSB exceeds 0.5 "
              f"(rounding bound)")

    print("-" * 58)
    if errors == 0:
        print(f"PASS: {NROWS} rows x {NF*DT} tokens match integer golden; "
              f"II=1; max non-saturated error {max_lsb_err:.3f} LSB "
              f"(<= 0.5 round bound).")
    else:
        print(f"FAIL: {errors} error(s).")
    print("-" * 58)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
