# binary_fit

Staged per-cycle power modeling from **single-bit signal features**, merging the
former `cobit` (XGBoost) and `nn` (two-layer MLP) prior-art baselines into one
compact package with two interchangeable Stage-2 regressors.

The three stages are independent and resumable — each is one flag of
`src/binary_fit/run.py`, run directly (no package/`-m` needed):

```
# 0. Materialize the single-bit feature dataset to disk (once).
python src/binary_fit/run.py --build_db --config src/binary_fit/configs/cobit.yaml

# 1. LR-MCP proxy selection -> a ranked proxies.csv.
python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/cobit.yaml \
    --outdir analysis/binary_fit/run1

# 2. Fit a model per proxy count (-1 = all proxies; default [-1]).
python src/binary_fit/run.py --fit --config src/binary_fit/configs/cobit.yaml \
    --outdir analysis/binary_fit/run1 -q 250 -1 --model tree

python src/binary_fit/run.py --fit --config src/binary_fit/configs/nn.yaml \
    --outdir analysis/binary_fit/run1 -q 250 -1 --model nn
```

`cobit.yaml` and `nn.yaml` share the `build`/`data`/`split`/`selection` sections
and only differ in `hpo` (tree-search knobs vs `nn_n_trials`/`nn_n_jobs`); use
whichever matches `--model`.

## Stages

| Stage | Reads | Writes |
|-------|-------|--------|
| `--build_db` | `build.source_db_root/func/<module>/<bench>_func.pkl.zst` (raw per-cycle signal states, `path[hi:lo]` columns) | `build.out_root/func/<bench>_func.pkl.zst` (single-bit uint8, canonical order) |
| `--feature_select` | the single-bit `data.func_dir` / `data.pwr_dir` | `<outdir>/proxies.csv` (rank, name, col_id, mcp_weight) + `proxies.json` |
| `--fit` | `proxies.csv` + the single-bit dataset | `<outdir>/<model>/<q>/{model.*, coefficients.csv, result.json, trace_test.png}` + `report.md`, `metrics.json`, `figures/q_sweep.png` |

## Design notes

- **Single-bit construction** happens only in `--build_db`: each `path[hi:lo]`
  net is sliced to raw-state bits `(v>>k)&1` named `path[lo+k]` (bare name for
  1-bit nets). A two-pass scan drops bits that are globally constant across all
  benchmarks (split-independent), so the column set is identical everywhere.
- **Window averaging (`data.window_size`, default 32)** is applied in `data.py`,
  once, before *both* `--feature_select` and `--fit`, so the two stages agree on
  the design matrix. Every benchmark is reduced to non-overlapping
  `window_size`-cycle rows: a feature becomes the bit's density in the window (a
  float in `[0, 1]`) and the label becomes the mean power over the window. The
  trailing partial window of each benchmark is dropped and the train/val cut is
  taken on whole windows, so no window straddles the split. `window_size: 1`
  restores the per-cycle 0/1 matrix exactly. Override per run with
  `--window-size N`; use the same N for both stages (a mismatch against
  `proxies.json` is warned about, not corrected). Consequences worth knowing:
  - `eval.peak_window` / `eval.multicycle_windows` count *rows*, i.e. multiples
    of `window_size` cycles — a `multicycle` t of 8 at `window_size: 32` scores
    256-cycle means. Divide `eval.peak_window` by `window_size` to keep the
    peak-detection window at the same number of cycles; note the averaged trace
    is smoother either way, so fewer cycles clear the mean + 3σ peak threshold.
  - `data.min_toggle_count` still counts **per-cycle** ones over the training
    cycles, so the kept-bit set does not move with `window_size` (up to the few
    tail cycles a window boundary drops); bits that end up constant *after*
    averaging are dropped on top of that (they would be collinear with the
    selector's intercept).
  - `window_size` is part of `stage_hash("data", ...)`, so HPO studies never
    resume onto a differently aggregated matrix.
- **`-q` selects the top-q proxies** from `proxies.csv` (ranked by |MCP weight|);
  `-1` (or `q ≥ #proxies`) uses all.
- **`selection.tol` is load-bearing, not cosmetic.** skglm's `AndersonCD` stops on
  an *absolute* `stop_crit <= tol`, and for an excluded column MCP's
  subdifferential distance is `max(0, |grad_j| - alpha)`. Because the target is
  power in watts, this problem's entire gradient scale is bounded by
  `alpha_max = 1.8e-3` — only 18× skglm's default `tol=1e-4`. Left at that
  default the solver returns after ~7 outer iterations at *every* alpha, with 85%
  of excluded columns violating their entry condition, and reports convergence:
  Q sticks at 17 from `alpha=1.5e-5` down to `1.8e-10`, which makes `target_qs`,
  `max_iter`, `max_bisect` and `grid_decades` all dead configuration. With
  `tol: 1.0e-6` the same grid traverses Q = 152/229/410/532/950 and the bisection
  reaches every configured target. `selection.kkt_report` logs the stationarity
  residual per fit — `0 violations, max|grad|/alpha <= 1.0` is a real optimum.
- **Lower `gamma` reaches *more* proxies, not fewer** (an earlier note here said
  the opposite). `gamma -> infinity` *is* Lasso — maximum shrinkage, fewest
  entrants — and `gamma -> 1+` is hard thresholding. Measured at a converged
  `tol=1e-6`: gamma 1.5/3/10/30/100/1000/1e6 → Q 4559/4990/3459/3219/2174/1014/903.
  At `gamma=1000` every selected coefficient also sat inside MCP's quadratic
  branch, so the penalty was running as shrinkage-Lasso and its nonconvex
  de-biasing was never exercised.
- **`selector: fsr`** is greedy forward selection (orthogonal matching pursuit):
  it takes Q as a direct argument, so `target_qs` are hit *exactly* with no alpha
  grid, gamma or tolerance, and its residual orthogonalization stops it picking
  near-clones. On the held-out `{conv_softmax, coremark}` benchmarks it beat the
  penalized route at equal Q. `mcp` remains the config default because this
  package reproduces the CoBiT prior art; switch per run with a positional
  override, e.g. `... --feature_select --config ... selection.selector=fsr`.
- **`selection.dedup_proxies`** collapses exact-duplicate columns inside a
  selected support. 80.3% of the aq_core kept bits are exact copies of another
  bit — partly genuine bus/clock-gate fanout, partly a dataset defect where the
  same net is materialized twice, bare and path-prefixed (`cp0_mmu_tlb_asid[11]`
  *and* `x_aq_cp0_top/cp0_mmu_tlb_asid[11]`). Without dedup a reported Q
  overstates how many distinct signals were found: top-1000 by |corr| holds only
  259 distinct value vectors.
- **Test benchmarks** (`split.test_benchmarks` = `{conv_softmax, coremark}`) are
  read once per fit over the union of proxy columns and sliced per `-q`
  (`data.Union`).
- **Both backends share** the features, MCP proxies, evaluation, plots and report
  format; they differ only in the estimator (`models.py`).

## Requirements

`xgboost`, `optuna`, `skglm` (genuine LR-MCP; falls back to sklearn Lasso if
absent), `scikit-learn`, `pandas`, `pyarrow`/`zstandard` (zstd pkls), `matplotlib`,
`joblib`, `pytest`.
