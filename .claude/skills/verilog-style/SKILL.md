---
name: verilog-style
description: Coding-style conventions for synthesizable Verilog (.v) RTL in this repo (hw/rtl, hw/syn/wrapper). Use when writing, generating, or editing synthesizable .v modules — port/parameter naming, declaration order, width notation, always_ff/always_comb usage, reset, and layout. Does NOT apply to verification / testbench (.sv) code, which is exempt.
---

# Verilog style

Conventions for synthesizable `.v` RTL in this repo. Verification / testbench (`.sv`) code is exempt.

## Language
1. Write the Verilog-2005 (`.v`) dialect, but use the SystemVerilog procedural keywords `always_ff` / `always_comb` instead of bare `always`. Elaborate as SystemVerilog (e.g. `vcs -sverilog`).
2. Do not use `logic` — every signal is `reg` or `wire`.
3. `$clog2` is allowed (it is part of Verilog-2005).
4. Use `integer` for procedural loop variables inside `always` blocks. `generate`-`for` loops still require `genvar` — a `genvar` cannot be replaced by `integer`.
5. For module I/O ports, do not use multi-dimensional / unpacked arrays. Flatten to a single packed vector (e.g. `[N*W -1:0]`) and slice with `+:` (`vec[i*W +: W]`).
6. Do not use `function` or `task`. Factor helper / reusable logic into its own module in a separate `.v` file and instantiate it (`U_*`).

## Naming
1. Put a short trailing comment after every I/O port explaining its meaning.
2. Parameters are UPPERCASE with short names (`N`, `OW`), each with a short trailing comment.
3. Separate configurable parameters from derived parameters (which must not be overridden).
4. Module I/O signals are prefixed `i_` for inputs and `o_` for outputs (e.g. `i_vld`, `o_vld`). (`clk`/`rst_n` are the exception — see below.)
5. Use `_ff`, `_2ff` suffixes to mark pipelined signals inside the module.
6. Use `clk` and `rst_n`. Reset is always asynchronous active-low (`@(posedge clk or negedge rst_n)`).
7. Module instances are named `U_*`, e.g. `U_MULT`.
8. Generate blocks are named `G_*`, e.g. `G_FP4_MULT`.
9. `always_ff` blocks are named `DFF_*`, e.g. `DFF_VLD`.
10. `always_comb` blocks are named `CMB_*`, e.g. `CMB_CMP42`.
11. Loop variables are single letters: `integer i, j, k, m, n, p, q, ...` for procedural loops; `generate` loop variables use the same single-letter style but are declared `genvar`.

## Coding style
1. All `integer` and `genvar` declarations come right after the module I/O list.
2. All `localparam` declarations come right after the integers and genvars.
3. Internal signals are declared after the localparams.
4. The order of code and module instantiation should mostly follow the data flow.
5. Width is written as `[<W> -1:0]`, e.g. `[7   -1:0]`, `[OW  -1:0]` — not `[6:0]`.
6. The second (higher / array) dimension is written `[<N>-1 :0]`.
7. Column-align the type, `[width]`, name, `[array-dim]`, and trailing `//` comment within each small code chunk.
8. Every non-trivial internal signal gets a short trailing inline comment.
9. Zero-extend with replication: `{{(OW-7){1'b0}}, m_out[i]}`.
10. Named port connections only, one per line, aligned: `.name (expr)`. Params via `#(.W(OW))`.
11. One `always_ff` / `always_comb` block per function: nonblocking `<=` for sequential state, blocking `=` in `always_comb`.

## Functional correctness
1. Prefer `assign` over `always_comb` wherever possible.
2. The left- and right-hand widths of `=` and operators should match. If they intentionally differ, add a short trailing comment explaining why.
3. Handle enable / gating with a plain `if (i_vld)` (no `else`) to hold state.

## Layout
1. The file opens with a `//---` banner block: title, math / spec, datapath, schedule, parameters, language line.
2. Each logical section is preceded by a `//---...---` divider with a one-line title (e.g. `// stage 1 : 64 -> 32`).
3. Two leading spaces align the `input  wire` / `output reg` / `output wire` keyword columns; group ports under `// control port` / `// data port` comments.
