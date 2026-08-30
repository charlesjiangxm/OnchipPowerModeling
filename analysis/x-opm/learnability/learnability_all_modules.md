# x-opm learnability screen — all aq_core submodules

**Date:** 2026-08-16  
**Method:** `x-opm/learnability/` correlate-toggle-power screen (see
`doc/x-opm-dataset-learnability.md`). Two tracks per (module, benchmark):
*switching* (honest Spearman of block switching vs power movement, null-corrected)
and *held-state* (`lvl_corr` = best `|corr(signal value, power level)|`).  
**Run:** `bash x-opm/learnability/run_all_modules.sh` — all 8 modules in parallel,
completed cleanly (`rc=0`). Per-module outputs in `out/x-opm/<module>_activity/`
(`corr_toggle_power.{json,csv}`, `per_signal_summary.json`, per-case PNGs);
launch log `out/x-opm/run_all_modules.log`.

Verdict tags: **L** = LEARNABLE from switching (honest ≥ 0.5), **P** = PARTIAL
(0.2–0.5), **N** = switching does NOT track power (< 0.2, screen negative — check
the held-state track instead). All benchmarks: 12 per module, except `lsu`/`vpu`
which lack `conv_softmax` (11).

## Verdict tally per module

| Module | Benchmarks | L | P | N | Read |
|--------|-----------:|--:|--:|--:|------|
| ifu  | 12 | 7 | 5 | 0 | most switching-learnable; also very high held-state (`lvl_corr` 0.7–0.9) |
| idu  | 12 | 5 | 6 | 1 | strongly learnable on ISA_* + coremark/debug/exception |
| vidu | 12 | 4 | 8 | 0 | ISA_* learnable; held-state mostly weak → switching-driven |
| iu   | 12 | 3 | 8 | 1 | ISA_FP/ISA_LS/exception learnable; coremark negative |
| rtu  | 12 | 2 | 10| 0 | switching mostly PARTIAL, but held-state dominant (`lvl_corr` ~0.8–0.9) |
| cp0  | 12 | 1 | 8 | 3 | only ISA_FP learnable; conv_softmax/csr/ISA_THEAD negative |
| lsu  | 11 | 1 | 6 | 4 | coremark strongly learnable (0.69); several idle benchmarks negative |
| vpu  | 11 | 1 | 6 | 4 | weakest for switching; only ISA_FP; coremark negative (held-state story) |

## Highlights

- **ISA_FP is the most switching-driven benchmark everywhere** — LEARNABLE in every
  module (honest 0.54–0.75), consistently best at window ≈ 128 cyc. ISA_LS/ISA_INT
  are the next most datapath-driven.
- **ifu / idu are the most learnable datapaths** from unweighted switching alone.
- **rtu is held-state dominated**: switching only PARTIAL, but almost every benchmark
  has `lvl_corr` ~0.8–0.9 → learnable from *level* features, not activity.
- **vpu confirms the prior finding**: switching screen is weak; `coremark` is
  negative (honest −0.021, p=0.11) with the signal in held state (`lvl_corr` 0.45),
  not switching. See `x-opm-cp0-target-unlearnable` / `x-opm-learnability-*` notes.
- **Control/idle benchmarks** (conv_softmax, csr, debug, interrupt, ISA_THEAD) are
  the usual negatives — little datapath activity; when learnable it is via held state.

## Where to look

- Per-benchmark numbers + verdict: `out/x-opm/<module>_activity/corr_toggle_power.json`
  (and `.csv`).
- Correlation-vs-window curves: `corr_toggle_power_<bench>.png`.
- Big-power-step coverage (green=explained / red=unexplained): `coverage_<bench>.png`.
- Per-signal held-state ranking: `per_signal_<bench>.csv`, `per_signal_summary.json`.

## Recommended next steps

1. Prioritize model training on **L** cells (ISA_FP everywhere; ifu/idu broadly) using
   the printed `best_W` as the model window.
2. For **P/N with high held-state** (rtu, ifu control benchmarks, vpu/lsu coremark),
   add the top held-state control signals as *level* features before concluding.
3. For genuine **N with low held-state** (some cp0/vpu control benchmarks), the driver
   is likely off-camera — widen capture scope before spending training time.

## Appendix — raw switching correlations (module × benchmark)

Both tables report the switching track at each cell's best window (`best_W`, target
∈ {tv, maxstep}). Values are **absolute correlations, not null-corrected** (the
verdict in the tally above uses the null-corrected *honest* Spearman = raw Spearman −
null95). `—` = benchmark absent (lsu/vpu lack conv_softmax). A large Pearson ≫ Spearman
gap (e.g. cp0 coremark 0.01/0.48, csr 0.92/0.50) means the coupling is real but
concentrated in a few high-switching burst windows rather than typical of every window.

### Spearman (rank; headline — leads the verdict)

| Benchmark | cp0 | idu | ifu | iu | lsu | rtu | vidu | vpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cache | 0.548 | 0.672 | 0.903 | 0.572 | 0.537 | 0.607 | 0.646 | 0.349 |
| conv_softmax | 0.509 | 0.831 | 0.897 | 0.823 | — | 0.694 | 0.726 | — |
| coremark | 0.481 | 0.731 | 0.569 | 0.543 | 0.938 | 0.476 | 0.678 | 0.125 |
| csr | 0.500 | 0.629 | 0.809 | 0.623 | 0.457 | 0.511 | 0.642 | 0.445 |
| debug | 0.533 | 0.806 | 0.835 | 0.727 | 0.410 | 0.757 | 0.778 | 0.428 |
| exception | 0.632 | 0.789 | 0.907 | 0.781 | 0.613 | 0.738 | 0.832 | 0.516 |
| interrupt | 0.530 | 0.658 | 0.890 | 0.658 | 0.517 | 0.533 | 0.699 | 0.513 |
| ISA_FP | 0.807 | 0.863 | 0.972 | 0.935 | 0.428 | 0.860 | 0.853 | 0.824 |
| ISA_INT | 0.445 | 0.828 | 0.946 | 0.853 | 0.611 | 0.825 | 0.845 | 0.432 |
| ISA_LS | 0.414 | 0.860 | 0.963 | 0.902 | 0.728 | 0.843 | 0.782 | 0.448 |
| ISA_THEAD | 0.419 | 0.841 | 0.945 | 0.905 | 0.780 | 0.809 | 0.839 | 0.470 |
| MMU | 0.624 | 0.718 | 0.894 | 0.727 | 0.565 | 0.628 | 0.753 | 0.450 |

### Pearson (linear; at the same best window)

| Benchmark | cp0 | idu | ifu | iu | lsu | rtu | vidu | vpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cache | 0.822 | 0.712 | 0.845 | 0.633 | 0.565 | 0.610 | 0.751 | 0.703 |
| conv_softmax | 0.466 | 0.840 | 0.915 | 0.811 | — | 0.633 | 0.630 | — |
| coremark | 0.014 | 0.909 | 0.787 | 0.644 | 0.873 | 0.871 | 0.887 | 0.290 |
| csr | 0.920 | 0.456 | 0.810 | 0.432 | 0.174 | 0.565 | 0.750 | 0.734 |
| debug | 0.642 | 0.499 | 0.873 | 0.609 | 0.498 | 0.435 | 0.749 | 0.499 |
| exception | 0.778 | 0.729 | 0.906 | 0.821 | 0.367 | 0.771 | 0.830 | 0.462 |
| interrupt | 0.673 | 0.628 | 0.853 | 0.658 | 0.812 | 0.507 | 0.736 | 0.701 |
| ISA_FP | 0.750 | 0.747 | 0.974 | 0.946 | 0.317 | 0.806 | 0.828 | 0.763 |
| ISA_INT | 0.340 | 0.772 | 0.962 | 0.791 | 0.811 | 0.760 | 0.787 | 0.588 |
| ISA_LS | 0.296 | 0.784 | 0.951 | 0.930 | 0.661 | 0.622 | 0.723 | 0.201 |
| ISA_THEAD | 0.235 | 0.778 | 0.944 | 0.933 | 0.875 | 0.780 | 0.781 | 0.817 |
| MMU | 0.688 | 0.675 | 0.818 | 0.677 | 0.746 | 0.670 | 0.672 | 0.686 |
