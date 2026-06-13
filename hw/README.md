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

---

# LayerNorm — Verilog (int8 inference)

Hardware implementation of the FT-Transformer `nn.LayerNorm(d_token)` in
[`src/models/ft_transformer.py`](../src/models/ft_transformer.py) (`norm1`,
`norm2`, `final_norm`; `elementwise_affine=True`, `eps=1e-5`). Over the
`D_TOKEN` axis of one token `x` with learned int8 affine `gamma`/`beta`:

```python
mean = x.mean(-1); var = x.var(-1, unbiased=False)        # population variance
y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
```

Unlike the tokenizer this needs a **reduction** (mean/variance) over the whole
token before any output can be produced, so one full token (packed `x_vec`) is
consumed per clock and one full normalized token is produced per clock after a
fixed latency — **II = 1**, latency **5**.

## Files

| File | Purpose |
|------|---------|
| `rtl/layer_norm.v` | Synthesizable DUT (S/SS reduction → V → integer `sqrt` → reciprocal → per-lane affine + requant). Verilog-2005 dialect but uses `always_ff`/`always_comb` (no `logic`); compile with `vcs -sverilog`. |
| `rtl/layer_norm_registered.v` | Synthesis wrapper adding input/output registers (flop-to-flop boundary for DC). |
| `../src/models/layer_norm_cmodel.{c,h}` | **Pure** behavioral C reference (no hardware/sim detail). The golden model — the RTL output equals it bit-for-bit. Has a `-DLN_STANDALONE` self-test vs the float ideal. |
| `verif/layer_norm_dpi.c` | Thin DPI-C glue exposing `layer_norm_cmodel` to SystemVerilog (the only file that includes `svdpi.h`). |
| `verif/tb_layer_norm.sv` | End-to-end self-checking SV testbench: random tokens + random γ/β, scoreboard, **bit-exact** compare vs the C model over DPI-C, II=1 check, FSDB dump. |
| `verif/Makefile`, `verif/filelist.f` | VCS-only compile/run/verdi flow. |
| `syn/dc_layer_norm.tcl`, `syn/run_dc_layer_norm.csh` | Design Compiler scaffold (mirrors the tokenizer's). |

## Numeric model (int8, inference only)

Same Q1.`FRAC_BITS` symmetric-quant conventions as the tokenizer (zero-point 0).
The `2^-FRAC` input scale and the `1/D` mean factor **cancel exactly**, so the
whole datapath is integer:

```
S    = sum(x)                                  # signed
V    = D*sum(x^2) - S^2   (= D^2 * var_int, >= 0, exact)
Veps = V + EPS_V          # EPS_V = round(eps * 2^(2*FRAC) * D^2), eps=1e-5
r    = floor(sqrt(Veps))  # integer sqrt, clamped >= 1
inv  = round(2^RECIP_FRAC / r)
y[i] = sat_int8( round_half_up( ((D*x[i]-S)*inv*gamma[i] + (beta[i]<<RECIP_FRAC))
                                >> (FRAC_BITS + RECIP_FRAC - OUT_FRAC) ) )
```

The integer `sqrt` is the classic bit-by-bit method, fully unrolled to a constant
iteration count for synthesis; the reciprocal is one unsigned divide per token.
This tracks the float LayerNorm to **≤ 1 LSB** (the residual is floor-sqrt +
reciprocal rounding); see the self-test below.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `D_TOKEN`    | 32  | token dimension = length of the normalized axis |
| `DATA_WIDTH` | 8   | int8 element width (x, gamma, beta, y) |
| `FRAC_BITS`  | 7   | input/affine fractional bits → Q1.7 |
| `RECIP_FRAC` | 24  | reciprocal fractional bits (1/std precision); 24 keeps reciprocal error ≪ 0.5 LSB |
| `OUT_FRAC`   | 7   | output fractional bits; `=FRAC_BITS` → strict Q1.7 |
| `EPS_V`      | 168 | integer epsilon = `round(eps*2^(2*FRAC)*D^2)`; **D-dependent** (168 @ D=32, 2684 @ D=128) |

> **Output saturation:** LayerNorm outputs are z-scores, so with the default
> `OUT_FRAC=7` (strict Q1.7) elements with `|z*gamma + beta| ≥ 1` **saturate** to
> ±127. This is expected and identical on the RTL and C sides (so the check still
> passes). For a non-saturating output scale set `OUT_FRAC` lower, e.g. `5` (Q3.5).
> `EPS_V` defaults to the D=32 value; the testbench and DC script recompute it
> from `D_TOKEN`. γ/β are Q1.7, so a trained γ≈1.0 maps to 127/128≈0.992.

## Ports

| Dir | Name | Width | Notes |
|-----|------|-------|-------|
| in  | `clk`/`rst_n` | 1 | async assert / sync deassert; clears the valid pipeline |
| in  | `wr_en`       | 1 | coefficient write strobe |
| in  | `wr_is_beta`  | 1 | `0` = write gamma, `1` = write beta |
| in  | `wr_addr`     | `clog2(D_TOKEN)` | lane index `i` |
| in  | `wr_data`     | `DATA_WIDTH` | signed int8 coefficient |
| in  | `in_valid`    | 1 | a valid `x_vec` is present this cycle |
| in  | `x_vec`       | `D_TOKEN*DATA_WIDTH` | packed: `x[i] = x_vec[i*DATA_WIDTH +: DATA_WIDTH]` |
| out | `out_valid`   | 1 | high when `y_vec` is valid (5 cycles after `in_valid`) |
| out | `y_vec`       | `D_TOKEN*DATA_WIDTH` | packed: `y[i] = y_vec[i*DATA_WIDTH +: DATA_WIDTH]` |

Coefficients load exactly like the tokenizer (write-only port, one per clock):
`wr_en=1, wr_is_beta=0/1, wr_addr=i, wr_data=gamma[i]/beta[i]` (int8).

## Verification

**C-model self-test (runs anywhere, no simulator):**

```
cd hw/verif && make cmodel
# PASS: all cases within 1 LSB of the float ideal.   (D = 16/32/64/128)
```

**End-to-end RTL-vs-C-model over DPI-C (VCS + Verdi):**

```
cd hw/verif
make all                                 # compile + run; expect:
#   PASS: N tokens (DT=32, OUT_FRAC=7) match C-model bit-for-bit; II=1 over 64-beat burst.
make verdi                               # open layer_norm.fsdb with full KDB code hierarchy
make all VCS_DEFINES=+define+LN_DT=64    # sweep d_token
make run SIMARGS=+seed=12345             # pick a random seed
```

The TB drives random tokens and random γ/β, captures each output after the
pipeline latency, calls the C model through DPI-C on the same inputs, and
requires **every int8 element to match exactly** (the RTL and C run the identical
integer path). `-kdb -debug_access+all` write the Verdi knowledge database under
`simv.daidir/kdb` so the FSDB loads with source/hierarchy navigation. If
`$fsdbDump*` ever fail to resolve on an older toolchain, set `NOVAS_FLAGS` in the
Makefile to link the Verdi PLI (`novas.tab`/`pli.a` `+vpi +memcbk +vcsd`).

**Synthesis (Design Compiler):**

```
cd hw/syn && ./run_dc_layer_norm.csh -mode syn -d_token 32
```
