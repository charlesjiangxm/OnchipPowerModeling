# x-opm — feature engineering & dataset build (CP0 power modeling)

Turns the raw per-cycle signal database into ML-ready, feature-typed datasets for
power modeling, following `doc/x-opm trainning procedure.md` **steps 1–5**
(variance filter → A/B/C/D classification → transforms → scaling → save datasets +
report). The RuleFit model (steps 6–8) and cobit reference are a later phase.

## Run

```bash
# interpreter must be the anaconda base env (pandas + numpy + pyyaml)
~/anaconda3/bin/python x-opm/run.py all --config x-opm/configs/cp0.yaml
```

Stages (also runnable individually):

| command     | does                                                              | writes |
|-------------|------------------------------------------------------------------|--------|
| `fit`       | train-only decisions: variance filter, classify, denominators    | `out/x-opm/manifest.json` |
| `transform` | apply the manifest to every case (`--force` to rebuild, `--cases a,b`) | `out/x-opm/dataset/{trainset,testset}/<case>/type{A,B,C,D}.pkl` + `target.pkl` |
| `report`    | recompute scaled stats from the train pkls and assemble the report | `out/x-opm/reports/feature_report.csv` + `summary.json` |
| `all`       | `fit` → `transform` (all cases) → `report`                       | all of the above |

`transform` is resumable: a case whose `target.pkl` exists is skipped unless
`--force`. Small cases are processed first, the giant ones (conv_softmax, coremark)
last, so a crash costs at most one case.

Tests: `~/anaconda3/bin/python x-opm/tests/test_x_opm.py` (or `pytest x-opm/tests`).

## What each step does

1. **Variance filter.** A feature is dropped iff it is *constant across the whole
   training set* (`global_min == global_max`, computed by streaming the training
   cases — exact for 128-bit ints, no test leakage). Dropped signals are recorded
   in the report with their raw min/max.
2. **Classification** (full lowercased path, priority **B > A > C > D**):
   - **B** clock gating — contains `clk` **and** `_en`
   - **A** control — contains any of `_en` `_vld` `_stall` `_req` `_busy` `_idle`
   - **C** data bus — contains `data`
   - **D** control/status payload — everything else
3. **Transforms:**
   - A, B → **bit-split** to single-bit features (`path[lo+k]`, value `(int>>k)&1`).
   - A only → `_stall`/`_idle` bits **inverted** (`1-x`; negatively correlated with power).
   - C → per-cycle **toggle → Hamming distance** scalar
     (`toggle[t]=state[t]^state[t-1]`, `hamming[t]=popcount(toggle[t])`), replacing the bus.
   - D → kept as a single raw integer.
4. **Scaling to [0,1]:** single bits ÷1; Hamming ÷ bus-width `W`; type-D integer ÷ `2^width−1`.
   Stored as `float32`.
5. **Outputs:** per-(case,type) `.pkl` frames + one aligned `target.pkl` per case;
   `feature_report.csv` (one row per final feature: type, rule, widths, transform,
   divisor, raw & scaled min/max/mean; dropped signals flagged); `summary.json`.

## Design decisions (see the plan for rationale)

- **Raw values are signal STATES, not toggles** (verified empirically), so the
  XOR→Hamming transform is applied here. Do not confuse this DB with cobit's
  toggle-mask assumption.
- **Target** = `x_aq_core/Pc(x_aq_cp0_top)` from `pwr/<case>_pwr.pkl`; aligned by
  integer-index intersection (same cycle, power's trailing row dropped).
- **Split** = leave-benchmarks-out; `test = [conv_softmax]` (cobit schemeA). Change
  in the config.
- **Type-B rule** requires `_en` (not a bare `en`) so clocks like `fence_clk`
  aren't mis-tagged. Configurable via `rules` in the YAML.
- **Hamming applies to type C only**; type D is kept raw (the doc's C/D labels are
  garbled — this is the internally consistent reading). Flip `hamming_types: [C, D]`
  to also Hamming-encode the wide type-D payloads.
- **Duplicate handling**: the same physical net appears at multiple hierarchy
  depths. Exact-duplicate value vectors (across all training cycles) are grouped;
  with `dedup_identical: true` (the cp0 default) only a canonical representative per
  group is kept and the rest are recorded as dropped (`duplicate of <rep>`). Set it
  to `false` to keep every copy (groups are still reported in the manifest).

## Known modeling caveat

Type D is dominated by wide (up to 64-bit) register/address payloads. Scaling a
64-bit value by `2^64−1` collapses it into a near-zero float with little resolution.
This follows the procedure literally; if the model underperforms, try
`hamming_types: [C, D]` or `dedup_identical: true`.

A handful of `[63:0]` signals (`mepc_value`, `sepc_value`, `idu_cp0_ex1_src{0,1}_data`)
carry a dump artifact value of exactly `2^64` on ~0.003% of cycles (logged as a
width warning during `fit`). These are clipped to 1.0 after scaling.
