# binary_fit

Staged per-cycle power modeling from **single-bit signal features**, merging the
former `cobit` (XGBoost) and `nn` (two-layer MLP) prior-art baselines into one
compact package with three interchangeable Stage-2 regressors — those two plus
`ridge`, the L2-penalized linear reference point.

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

python src/binary_fit/run.py --fit --config src/binary_fit/configs/ridge.yaml \
    --outdir analysis/binary_fit/run1 -q 250 -1 --model ridge
```

`cobit.yaml`, `nn.yaml` and `ridge.yaml` share the
`build`/`data`/`split`/`selection` sections and only differ in their
hyperparameter block (tree-search knobs vs `nn_n_trials`/`nn_n_jobs` vs the
`ridge:` alpha grid); use whichever matches `--model`. `--model both` fits all
three in one pass.

## Stages

| Stage | Reads | Writes |
|-------|-------|--------|
| `--build_db` | `build.source_db_root/func/<module>/<bench>_func.pkl.zst` (raw per-cycle signal states, `path[hi:lo]` columns) | `build.out_root/func/<bench>_func.pkl.zst` (single-bit uint8, canonical order) |
| `--feature_select` | the single-bit `data.func_dir` / `data.pwr_dir` | `<outdir>/proxies.csv` (rank, name, col_id, mcp_weight) + `proxies.json` |
| `--fit` | `proxies.csv` + the single-bit dataset | `<outdir>/<model>/{model.*, coefficients.csv, result.json, predictions.pkl.zst, residual_train_val_test.png, pred_vs_time_{train,val,test}.png}` (plus `ridge_coefficients.csv` for `--model ridge`) + `report.md`, `metrics.json`. Several `-q` values put each experiment in its own `<outdir>/<model>/<q>/` and add `q_sweep.png` — see [Result layout](#result-layout) |

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
<a id="result-layout"></a>
- **Result layout: a lone experiment *is* the model directory.** With one `-q`
  value — the default `-q -1`, which is every run in `analysis/` — everything
  lands flat in `<outdir>/<model>/`, with no single-child `all/` or `figures/`
  directory to descend through:

  ```
  analysis/ridge/2026-09-03-17-100proxy/       <- --outdir
  ├── proxies.csv / proxies.json               <- --feature_select, model-agnostic
  └── ridge/                                   <- --model ridge
      ├── model.joblib, coefficients.csv, ridge_coefficients.csv
      ├── result.json, predictions.pkl.zst
      ├── pred_vs_time_{train,val,test}.png, residual_train_val_test.png
      └── metrics.json, report.md
  ```

  Two or more `-q` values would overwrite each other there, so each keeps one
  `<outdir>/<model>/<q>/` subdirectory (`q250/`, `all/`) and the model directory
  holds only `metrics.json`, `report.md` and the cross-experiment
  `q_sweep.png` — which a single experiment skips, being one point.
  `proxies.csv` stays at the `--outdir` level either way: `--feature_select`
  never sees `--model`, and one proxy set is shared by all three backends of a
  `--model both` run.

  The experiment's **label** (`all`, `q250`) is unchanged by the flattening — it
  still names the optuna study (`rungs_all_R30_*.json`), `result.json`'s
  `label`, the `report.md` row and the figure titles.
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
- **Every `--fit` writes both figure types for every split** (`plots.py`),
  matching the layout of `analysis/x-opm/<ts>/<win>/<module>/`:
  `residual_train_val_test.png` (parity scatter, red dashed y = x, one panel per
  split) and `pred_vs_time_{train,val,test}.png` (label + prediction over the
  split's concatenated benchmarks, boundaries dotted and named). Three things
  worth knowing when reading them:
  - **Power is plotted in mW**; the target, `result.json` and every metric are
    in watts. R² and MAPE are scale-free either way.
  - **Nothing is decimated.** The full 63.6k-row test trace renders in ~1.5 s at
    ~74 KiB and the 2.04M-row `window_size: 1` worst case in ~3.4 s, so the old
    `eval.trace_plot_cycles` row cap is gone — it had been cutting the test
    figure at row 12000, inside `conv_softmax`, with `coremark` never drawn at
    all. Stride sampling would also delete exactly the spikes `eval.peak_window`
    scores.
  - **Read the panel titles before quoting a number.** Train is always training
    data, and *with HPO so is val*, because the final model is refit on
    train+val once the hyperparameters are chosen — so only the test panel is
    ever held out under `--fit` with HPO. Each in-sample panel is titled with
    the reason (`train (in-sample)`, `val (in HPO refit)`).
  The figure filename records which splits existed, so `residual_train_test.png`
  means the run had `split.val_fraction: 0`. `predictions.pkl.zst`
  (`bench, split, y_true, y_pred`) is written alongside so either figure can be
  redone without re-running the fit. Figures are rendered *after* `result.json`
  and wrapped so a matplotlib failure can never cost a completed fit. If you
  launch several `--fit` processes at once, `export MPLCONFIGDIR=<per-run dir>`
  first to avoid a matplotlib font-cache race.
- **`--model ridge` is the linear reference point.** L2-penalized linear
  regression on the same MCP proxies, answering how much of the power is just a
  weighted sum of the selected bits — and unlike the other two it hands back a
  per-bit coefficient in watts, which is what an on-chip monitor needs.
  - **Ridge, not OLS**, because the proxy set is strongly collinear: 80.3% of the
    aq_core kept bits are exact copies of another bit (see the `dedup_proxies`
    note above), so `XᵀX` is genuinely rank-deficient and the penalty is what
    makes the solve well-posed. Measured on duplicated columns, `alpha=0` returns
    `max|coef| = 6.8e4` where `alpha=1e-8` returns `1.5`; hence the grid floor is
    bounded away from zero (`ridge.grid_decades` below `ridge.alpha_rel_max`).
  - **X and y are both z-scored** before the fit, the opposite of the Stage-1
    choice in `selection.py`. That note ("standardizing would destroy sparsity and
    the toggle semantics") is about a sparse L1 *selector*; this is a dense L2 fit
    whose penalty is scale-dependent, so without standardization alpha would mean
    a different thing per feature and coefficients would not be comparable across
    bits. `ridge_coefficients.csv` therefore carries both: `coef_std`
    (dimensionless, comparable across bits) and `coef_watts` (watts per unit of
    the feature — per assertion of the bit at `window_size: 1`), with the matching
    `intercept_watts` in `result.json` under `best`.
  - **Alpha comes from `RidgeCV`'s leave-one-out generalized CV over the fitting
    rows**, not from an optuna study on the validation tail. So `hpo.py` is not
    involved, the knobs live in their own `ridge:` config section (putting them in
    `hpo` would change `hpo.study_stamp` and orphan the trials in an existing
    `optuna.db`), `result.json`'s `best.val_r2` is `null` because nothing here
    scored val, and `best.gcv_neg_mse` is negative MSE **on the standardized
    target** — not an R². With HPO on, train and val are fitted as one set, the
    same data the other two backends refit on, so the "in HPO refit" panel titles
    read correctly.
  - **What the GCV protocol difference actually costs.** The closed-form LOO/GCV
    path is a function of the Gram/SVD alone, so it is deterministic and
    *row-order independent* (verified: permuting the rows returns an identical
    `alpha_`); it does not shuffle anything. The real caveat is that leave-one-out
    leverage is optimistic on this data — consecutive `window_size`-cycle rows are
    temporally correlated and 80.3% of the columns are duplicates — so GCV can
    under-penalize relative to what scoring a contiguous held-out tail would pick,
    and ridge's alpha is therefore **not** on the same footing as the val-tail
    hyperparameters `tree` and `nn` get. Measured here it does not bite: GCV chose
    `alpha_rel` 2.15e-6 at Q=100 (absolute alpha 0.0186 over 8616 rows) and that
    model still reaches 0.9421 on the genuinely held-out benchmarks. If you want the val-tail protocol instead, score
    `models.fit_ridge_scaled(Xtr, ytr, alpha=a)` over `ridge_alphas(...)` against
    the val split yourself — the pieces are all public.
  - **Measured**, on the same 1000-proxy `proxies.csv` at `window_size: 32` with
    test `{conv_softmax, coremark}` (8616 fitting rows): test R² 0.9421 / MAPE
    3.04% at Q=100, and 0.9396 / 3.23% at Q=1000 — so **a linear model is
    competitive with the boosted trees on this data**, which is the useful finding
    rather than a win: separately tuned GBDTs on the same rows and split reach
    0.9394 (FSR proxies, Q=1000) and 0.9349 (bagged MCP, Q=1000). It does beat the
    tree HPO run stored in `analysis/cobit/.../tree/report.md` (0.8328 / 4.67% at
    Q=1000, 704 leaves), but that is one HPO outcome, not the tree's ceiling — the
    leaf-budget objective there lands on a different point than a plain
    `max_depth: 3` sweep does. Read ridge as a cheap, interpretable baseline that
    the non-linear backends have to actually beat, not as the better model.
    Plausible reason it does this well: at `window_size: 32` every feature is a bit
    *density*, and a window's mean power is close to additive in those densities.
  - **A `coef_watts` on a near-collinear bit is not individually meaningful**,
    even though the prediction is. In the Q=100 run above the top two rows are
    `mcic_arb_data_idx[13]` at +13.70 mW and `mcic_dc_addr[13]` at −13.69 mW: that
    pair is byte-identical in `cache` and differs in 22 of 83386 cycles in
    `ISA_LS` (r = 0.99947 per-cycle, 0.99997 at `window_size: 32`), so
    `dedup_proxies` — which collapses only *exact* duplicates — kept both, and
    ridge split a large weight across them with signs that cancel to 9e-6 W. Read
    such rows as a pair, not as two per-bit numbers, and if per-bit attribution is
    the goal prefer `selection.selector=fsr`, whose residual orthogonalization
    stops it picking near-clones in the first place.
  - **It is the one backend that runs at `split.val_fraction: 0` with HPO on** —
    `tree` and `nn` raise there, because their searches need the val split and
    ridge's does not.
  - **`ridge.max_rows` (default 200k) caps the fitting rows** with a seeded
    subsample. `RidgeCV`'s GCV path SVDs the design matrix, whose `U` factor is
    `n_rows × Q` float64: 160 MB at 200k × 100 but 16 GB at the `window_size: 1` ×
    `Q=1000` corner. A linear model over a few thousand proxies is fully
    determined long before 200k rows, and at `window_size: 32` the cap never fires
    on aq_core (measured: 6895 train + 1721 val = 8616 fitting rows).
- **All three backends share** the features, MCP proxies, evaluation, plots and
  report format; they differ only in the estimator (`models.py`).

## Requirements

`xgboost`, `optuna`, `skglm` (genuine LR-MCP; falls back to sklearn Lasso if
absent), `scikit-learn`, `pandas`, `pyarrow`/`zstandard` (zstd pkls), `matplotlib`,
`joblib`, `pytest`.
