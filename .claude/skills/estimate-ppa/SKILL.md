---
name: estimate-ppa
description: Estimate rough PPA — critical-path delay, dynamic power, static (leakage) power, and cell area — of a given Verilog/SystemVerilog module using TSMC N6 LVT primitive-cell characterization data, without running synthesis. Use when asked to estimate PPA, area, power, timing, or gate count of RTL by hand.
---

# estimate-ppa

Produce a **rough, first-order** PPA estimate for a Verilog/SystemVerilog module
the user provides (pasted snippet or file path). This is a back-of-the-envelope
gate-mapping estimate to guide design decisions — it is **not** a substitute for
DC synthesis (use the `create-dc-syn-script` skill for real numbers).

Estimate four quantities:
- **Cell area** (µm²)
- **Static power** = leakage (nW)
- **Dynamic power** (µW) at a user-given clock frequency + switching activity
- **Critical-path delay** (ps) along the longest combinational logic path

## Characterization data

All numbers below are from the **TSMC N6** standard-cell library
`tcbn06_bwph240l11p57cpd_base_lvt` (H240 track, **LVT**), typical corner
**TT / 0.75 V / 85 °C** (the `$N6Datasheet` corner; the same library
`_flow/dc/tcl/setup.tcl` targets).

Per-cell delay and dynamic energy are characterized at a common operating point:
**input slew = 13.2 ps, output load = 5 fF**, taking the cell's worst
input→output arc. Leakage is the unconditioned VDD `cell_leakage_power`.
"Dyn. energy" is internal switching energy **per output transition** (fJ);
convert to power with `P_dyn = E · f_toggle` (see methodology).

| Function | Size | Cell | Area (µm²) | Leakage (nW) | Dyn. energy (fJ/tr) | Delay (ps) | Worst arc |
|---|---|---|---|---|---|---|---|
| **2-in AND** | small | AN2D1 | 0.0547 | 3.53 | 0.50 | 30.6 | A2→Z (r) |
| | mid | AN2D4 | 0.1231 | 9.78 | 1.22 | 19.9 | A2→Z (r) |
| | large | AN2D16 | 0.3830 | 36.01 | 3.90 | 18.8 | A2→Z (r) |
| **2-in OR** | small | OR2D1 | 0.0547 | 2.95 | 0.51 | 32.6 | A2→Z (f) |
| | mid | OR2D4 | 0.1231 | 8.10 | 1.22 | 20.0 | A2→Z (f) |
| | large | OR2D16 | 0.3830 | 28.51 | 3.98 | 18.7 | A2→Z (f) |
| **2-in XOR** | small | XOR2D1 | 0.1231 | 6.54 | 0.92 | 44.0 | A1→Z (f) |
| | mid | XOR2D4 | 0.2189 | 19.68 | 1.91 | 29.1 | A2→Z (f) |
| | large | XOR2D8 | 0.3283 | 29.56 | 2.87 | 26.9 | A2→Z (f) |
| **2-to-1 MUX** | small | MUX2D1 | 0.1094 | 5.26 | 0.84 | 39.8 | S→Z (f) |
| | mid | MUX2D4 | 0.2189 | 16.63 | 1.77 | 27.2 | S→Z (f) |
| | large | MUX2D8 | 0.3283 | 25.10 | 2.49 | 26.2 | S→Z (f) |
| **AOI21** (2-to-1 AOI) | small | AOI21D1 | 0.0547 | 2.12 | 0.38 | 39.7 | A1→ZN (f) |
| | mid | AOI21D4 | 0.1778 | 8.15 | 1.31 | 16.4 | A1→ZN (f) |
| | large | AOI21D8 | 0.3557 | 16.44 | 3.11 | 12.5 | A2→ZN (f) |
| **Half adder** | small | HA1D1 | 0.1778 | 10.60 | 0.85 | 42.5 | B→S (f) |
| | mid | HA1D2 | 0.1915 | 13.49 | 0.97 | 35.3 | B→S (f) |
| | large | HA1D4 | 0.3283 | 29.19 | 2.05 | 30.4 | A→S (f) |
| **Full adder / CMP32** | small | FA1D1 | 0.2189 | 8.68 | 0.85 | 51.5 | A→S (f) |
| | mid | FA1D2 | 0.2462 | 12.03 | 1.02 | 48.6 | A→S (f) |
| | large | FA1D4 | 0.3830 | 20.93 | 1.67 | 59.5 | A→S (r) |
| **CMP42** (4:2 compr.) | small | CMPE42D1 | 0.4378 | 17.41 | 1.79 | 85.9 | A→S (f) |
| | mid | CMPE42D2 | 0.4651 | 22.74 | 1.94 | 85.5 | C→S (f) |
| | large | CMPE42D4 | 0.7114 | 36.49 | 2.75 | 92.9 | A→S (r) |

All cells are suffixed `BWP240H11P57CPDLVT`. Data-table notes:
- **CMP32** (3:2 compressor) has no dedicated cell — it *is* the full adder, so
  it shares the **FA1** rows.
- **CMP42** → the `CMPE42` (4:2 compressor w/ carry-in/out) family; only D1/D2/D4
  exist; the worst arc reaches the sum output `S`.
- Delay for the complex adder/compressor cells is **not monotonic vs drive**
  (the worst arc changes edge/pin between sizes) — don't over-interpret it.
- **Default to the `mid` drive strength** unless the user asks otherwise; it is
  the most representative of what synthesis picks for average fanout.

## Methodology

Work in these steps and show the reasoning, not just the final numbers.

### 1. Map the RTL to primitive cells
Read the module and decompose each operator into the primitives above. Common maps:
- `&`,`~&` → AND (AOI21 for `~( a & b | c)`-style fused logic if it fits).
- `|`,`~|` → OR. `^`,`~^` → XOR. `? :` / `case` mux → MUX (an N-way mux ≈ N-1 MUX2).
- **Adder (ripple):** LSB = 1 HA, remaining `W-1` bits = FA each.
- **Adder (fast/CLA):** approximate as `W` FA-equivalents for area/power; for
  delay use `~2·log2(W)` gate levels instead of the ripple carry chain.
- **Multiplier `W×W`:** partial products ≈ `W²` AND2; reduction tree ≈ `W²`
  compressors (mix of CMP42/FA/HA); final ≈ one `2W`-bit adder.
- **Comparator / equality:** per bit an XOR + reduction (AND/OR tree).
- Count each primitive type and its instance count `n_cell`.

### 2. Cell area
`Area = Σ_cell ( n_cell · area_cell )`. Report µm² and, if useful, kGE
(1 GE ≈ NAND2 area; here ≈ AN2D1 area 0.0547 µm² as a rough gate equivalent).

### 3. Static (leakage) power
`P_leak = Σ_cell ( n_cell · leakage_cell )` in nW. Note this is LVT (leaky); SVT
would be markedly lower — flag if the user cares about leakage.

### 4. Dynamic power
Per-transition energy `E` is in the table (fJ). Average dynamic power:
```
P_dyn = Σ_cell ( n_cell · E_cell · α · f_clk )
```
- `f_clk` = target clock frequency (ask the user; state the value used).
- `α` = switching activity = avg output toggles per clock. **Default α = 0.15**
  for datapath logic (state the assumption; let the user override).
- Units: `E[fJ] · f[GHz]` gives µW directly (fJ·GHz = µW). So
  `P_dyn[µW] = Σ n_cell · E_cell[fJ] · α · f_clk[GHz]`.
- Add register/clock-tree dynamic power only if the module is sequential and the
  user wants it; otherwise state that this estimate is combinational-logic only.

### 5. Critical-path delay
Identify the **longest combinational path** between register/IO boundaries
(logic depth), then sum the per-cell delays along it:
```
t_crit = Σ_(cells on longest path) delay_cell   [ps]
```
- Ripple adder critical path ≈ `HA + (W-1)·FA_carry` — dominated by the carry
  chain; use the FA delay per bit.
- Multiplier ≈ `AND(pp) + (#compressor stages)·CMP42 + final_adder`.
- The table delays are per-cell at 5 fF / 13.2 ps slew; real paths add wire and
  fanout load, so treat this as a **lower bound / optimistic** number and say so.
- Report the path (which operators/bits dominate), not just the total.

## Output format
1. **Assumptions** — drive strength, `f_clk`, `α`, corner (state them up front).
2. **Cell mapping table** — primitive | count | area | leakage | dyn-energy.
3. **Totals** — Area (µm²), Static power (nW), Dynamic power (µW @ f_clk), and
   Critical-path delay (ps) with the dominating path described.
4. **Caveats** — this is a hand estimate; no wire load, no placement, no CTS, no
   real synthesis mapping/optimization. For sign-off numbers, run DC via the
   `create-dc-syn-script` skill.
