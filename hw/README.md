# NumericalFeatureTokenizer — Verilog-2005 (int8 inference)

Hardware implementation of the FT-Transformer feature tokenizer in
[`src/models/ft_transformer.py`](../src/models/ft_transformer.py)
(`NumericalFeatureTokenizer.forward`):

```python
out = x.unsqueeze(-1) * weight.unsqueeze(0) + bias.unsqueeze(0)   # (B,F) -> (B,F,D)
#  elementwise:  out[j][k] = x[j] * weight[j][k] + bias[j][k]
```

It is a purely **element-wise** broadcast multiply-add (no reduction across
features): each of the `N_FEATURE × D_TOKEN` outputs is one multiply plus one
bias add. The block consumes **one input row `x` per clock** and emits the full
`N_FEATURE × D_TOKEN` token matrix every clock after a fixed pipeline latency —
**initiation interval (II) = 1**.

## Files

| File | Purpose |
|------|---------|
| `numerical_feature_tokenizer.v` | Synthesizable DUT (FF register file + `N_FEATURE×D_TOKEN` pipelined int8 multiply-add lanes). |
| `tb_numerical_feature_tokenizer.v` | Self-checking Verilog testbench (loads coefficients, streams rows back-to-back, checks values + II=1). |
| `ref_model.py` | Cycle-accurate Python twin of the integer datapath/pipeline; validates the arithmetic and II=1 **without an HDL simulator**, and is reusable for coefficient quantization. |

## Numeric model (int8, inference only)

- Inputs are **already quantized** int8 (no z-score / normalization in hardware —
  that is done upstream). Symmetric quantization is assumed (**zero-point = 0**),
  so there are no zero-point correction terms — just the algorithm.
- int8 is interpreted as signed **Q1.`FRAC_BITS`** fixed-point (default Q1.7,
  scale `2^-7`, range `[-1, +0.992…]`).
- Datapath per lane: `prod = x·W` (Q2.`2·FRAC`, signed 16-bit) → add bias
  left-shifted by `FRAC_BITS` to align → **requantize** back to int8
  (round-half-up, arithmetic right shift by `FRAC_BITS`, saturate to
  `[-128, 127]`).
- **Dialect:** Verilog-2005 (ANSI ports, `$clog2` sizes `wr_addr`),
  vendor-neutral and synthesizable.

Convert trained float coefficients to what the write port expects, and
dequantize the output:

```python
q  = lambda v: int(np.clip(np.round(v * 2**FRAC_BITS), -128, 127))   # float -> int8
xf = out_int8 / 2**FRAC_BITS                                          # int8 -> float
```

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `N_FEATURE`  | 20 | number of features (rows of weight/bias), typically < 128 |
| `D_TOKEN`    | 32 | token dimension (cols of weight/bias), typically < 128 |
| `DATA_WIDTH` | 8  | int8 element width (x, weight, bias, output) |
| `FRAC_BITS`  | 7  | fractional bits → int8 read as Q1.7; sets bias alignment + requant shift |

## Ports

| Dir | Name | Width | Notes |
|-----|------|-------|-------|
| in  | `clk`        | 1 | |
| in  | `rst_n`      | 1 | async assert / sync deassert; clears the valid pipeline |
| in  | `wr_en`      | 1 | coefficient write strobe |
| in  | `wr_is_bias` | 1 | `0` = write weight, `1` = write bias |
| in  | `wr_addr`    | `clog2(N_FEATURE*D_TOKEN)` | linear index `j*D_TOKEN + k` |
| in  | `wr_data`    | `DATA_WIDTH` | signed int8 coefficient |
| in  | `in_valid`   | 1 | a valid `x_row` is present this cycle |
| in  | `x_row`      | `N_FEATURE*DATA_WIDTH` | packed: `x[j] = x_row[j*DATA_WIDTH +: DATA_WIDTH]` |
| out | `out_valid`  | 1 | high when `out_tokens` is valid (latency cycles after `in_valid`) |
| out | `out_tokens` | `N_FEATURE*D_TOKEN*DATA_WIDTH` | packed: `out[j][k] = out_tokens[(j*D_TOKEN+k)*DATA_WIDTH +: DATA_WIDTH]` |

**Packing / addressing:** the linear index `j*D_TOKEN + k` is identical for
`wr_addr` and for the `out_tokens` slice — feature-major, token-minor.

## Timing

- **Latency:** 3 clocks (`in_valid` → corresponding `out_valid`):
  stage 1 latches `x`, stage 2 multiplies, stage 3 adds bias + requantizes.
- **Throughput:** II = 1 — assert `in_valid` every cycle to stream a new row
  each clock. No backpressure (the block always accepts).

## Loading coefficients

Write-only port. Hold weights static during inference. One coefficient per
clock:

```
# weights
wr_en=1, wr_is_bias=0, wr_addr = j*D_TOKEN+k, wr_data = W[j][k]   (int8)
# biases
wr_en=1, wr_is_bias=1, wr_addr = j*D_TOKEN+k, wr_data = b[j][k]   (int8)
```

## Verification

**Python twin (runs anywhere, no simulator):**

```
python3 hw/ref_model.py
# PASS: 12 rows x 16 tokens match integer golden; II=1; max non-saturated error 0.500 LSB ...
```

It reproduces the RTL's exact integer semantics (align/round/saturate) and
3-stage pipeline, runs the same stimulus as the Verilog TB (a `+127` row, a
`-128` row, then random int8), and additionally confirms the dequantized output
matches the float `x·W + b` op to within the 0.5-LSB rounding bound.

**HDL simulation** (Icarus Verilog or Verilator — *not installed in this
environment*, so the RTL was not simulated here; install one, e.g.
`brew install icarus-verilog`, then run):

```
cd hw
iverilog -g2005 -o tb.out numerical_feature_tokenizer.v tb_numerical_feature_tokenizer.v && vvp tb.out
# expect: PASS: 12 rows x 16 tokens match; II=1 (contiguous outputs).

# quick syntax/lint check with Verilator:
verilator --lint-only -Wall numerical_feature_tokenizer.v
```

## Resource note

Fully parallel to sustain II = 1 with a full token-matrix output every cycle:

- **Multipliers/adders:** `N_FEATURE × D_TOKEN` int8 lanes (worst case
  128×128 = 16 384; typical 20×32 = 640).
- **Register file:** `2 × N_FEATURE × D_TOKEN × DATA_WIDTH` flip-flops, FF-based
  so all coefficients read in parallel (worst case ~262 k; typical ~10 k).

If a worst-case 128×128 instance is too large, a folded variant (process the row
over several cycles, II > 1) can reuse the same lane datapath — out of scope here
given the 1-cycle-per-x requirement.
