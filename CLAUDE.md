# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Runtime on-chip power modeling for the OpenC906 RISC-V core. Trains a family of regressors (RuleFit, FT-Transformer, GBDT/XGBoost, MLP, Ridge) on cycle-aligned waveform features (`X`) versus measured power (`y = /Pc(openC906)`), and reports per-feature / per-rule importance. Background reading lives in `doc/` (APOLLO MICRO'21, DEEP ICCAD'22, ArchPower, FT-Transformer NeurIPS'21).

## Repository layout

- `src/` — one script per model family, all sharing the same loader, feature selector, and split scaffolding:
  - `rulefit.py`, `ft_transformer.py`, `gbdt.py`, `mlp.py`, `ridge.py` — runnable entry points (each has its own `main()`).
  - `rulefit_utils.py` — dataset loaders (`load_c906_pair`, `load_all`), `PREFIXES`, `TARGET_COL`, metrics (`compute_metrics`), rule parser, optional non-negative LassoCV refit for RuleFit.
  - `feature_selectors.py` — `FeatureSelector` with 8 methods: `pearson`, `variance`, `univariate`, `rfe`, `from_model`, `sequential`, `mcp`, `deep`. MCP/DEEP use a hand-rolled coordinate-descent with optional numba JIT (auto-detected). All entry scripts wire the same `--fs_method` plus the per-method knobs (`--fs_*`).
  - `ft_transformer_model.py` — self-contained PyTorch FT-Transformer plus `Standardizer`, `default_device(pref)` (CUDA > MPS > CPU), `describe_device`, `train_ft_transformer`, `predict`, `extract_attention`. Imported by other scripts purely for `Standardizer` / device helpers, not the model. Note the deliberate split between `ft_transformer.py` (entry script that mirrors the other model scripts) and `ft_transformer_model.py` (the actual PyTorch implementation).
  - `rulefit_utils.refit_nonneg_lasso` — rebuilds RuleFit's internal design matrix and refits LassoCV with `positive=True` (selected via `--lasso_mode nonneg` in `rulefit.py`).
- `db/` — the dataset (gitignored; ship via `db.zip`). The loader expects `db/c906-db/{<presim_subdir>,pwr}/<prefix>_func.pkl` + `<prefix>_pwr.pkl`. NOTE: the current working tree's `db/` instead contains `idu_input/`, `idu_net/`, `idu_output/`, `ifu_input/`, `ifu_net/`, `ifu_output/`, `pwr/` directly under `db/`. The scripts will not find data at this layout — either move the subdirectories under `db/c906-db/`, set `presim_subdir` to a real child of `db/c906-db/`, or pass `base_dir` explicitly to the loader.
- `doc/` — reference papers. Read these when changing modeling assumptions.
- `script/` — currently empty.

## How a training run is structured

Every entry script follows the same shape, so changes to the pipeline are usually mechanical across all five:

1. Parse `--split {loco, time_ordered}` plus shared flags (`--top_k`, `--seed`, `--presim_subdir`, `--fs_method` + per-method `--fs_*`, model-specific hyperparameters).
2. Resolve `out_dir` as `../../output/<model>_c906_<split>[_<fs_method>][_<extra>][_<presim_subdir>]/`. `fs_method=pearson` and `presim_subdir=presim` are the "default" and don't appear in the suffix.
3. Load via `load_c906_pair(prefix, presim_subdir=...)`. Each prefix yields `(X, y, time_ps)`; `X` is forced to `float64`, NaNs imputed to 0.0 (presim_large encodes unknown signal states as NaN).
4. Run one of two drivers:
   - `driver_loco`: 5-fold LOCO over `PREFIXES = ["MMU", "cache", "csr", "exception", "interrupt"]`.
   - `driver_time_ordered`: per-prefix, sort by `time_ps`, last `--test_ratio` rows held out.
5. Inside each fold, `FeatureSelector.from_args(args).fit_select(X_train, y_train)` returns column names (unless `--fs_method none`, which bypasses the selector and keeps every input column). Selection runs on the training fold only.
6. Standardize selected `X` and target `y` on the training-train split, train the model, inverse-transform predictions, compute metrics, write per-fold artifacts plus a `global/` aggregate and `report.md`.

When adding a new model, mirror this structure: the selector, loaders, metrics, and arg surface are designed to be shared verbatim. The per-script "plotting helpers" block is intentionally duplicated so each script is self-contained.

## Conventions worth knowing

- `TARGET_COL = "/Pc(openC906)"` and `PREFIXES = ["MMU", "cache", "csr", "exception", "interrupt"]` are hardcoded in `rulefit_utils.py`.
- `rulefit.py` shares its name with the PyPI `rulefit` package it imports. To avoid the local file shadowing the package when run as `python rulefit.py` (Python puts `src/` at `sys.path[0]`), the top of the file temporarily drops `src/` from `sys.path` for the `from rulefit import RuleFit` lookup and restores it immediately. Don't move that block.
- Feature selection always standardizes in float64 — this is deliberate. Several presim columns are wide-bus signals (e.g. 320-bit `data_in`) that overflow float32. Don't downcast in the loader; the previous `filter_features` had to wrap a float32 path in `np.errstate`, which the current selector replaces.
- The selector drops zero-variance columns first; constant columns are kept only when `--fs_method none` is used.
- `mcp` and `deep` coordinate-descent are sequential by algorithm — `--fs_n_jobs` does not affect them. They benefit instead from numba JIT, which is auto-used if `numba` is importable (10-50× speedup typical).
- `--fs_method sequential` is impractical at `--top_k=1000` (hours); pair it with a small `top_k` if you actually want it.
- `RuleFit`'s `rules.csv` thresholds and coefficients are in the standardized model space, not the raw feature space. The report notes this; don't translate them back without redoing standardization.
- `--lasso_mode nonneg` (RuleFit only) replaces `rf.lscv`/`rf.coef_`/`rf.intercept_` in place — `predict()`, `get_rules()`, and importance scores all reflect the refit.
- `--ft_device auto` resolves to CUDA > MPS > CPU. Explicit `cuda`/`mps` raises if unavailable rather than silently falling back. `--ft_amp` is a no-op on MPS/CPU.

## Running

There is no `Makefile`, `setup.py`, or `requirements.txt`. Install dependencies ad hoc:

```
pip install numpy pandas scikit-learn matplotlib torch xgboost rulefit
pip install numba    # optional, but ~10-50x faster MCP/DEEP feature selection
```

The scripts assume CWD is `src/`:

```
cd src
python rulefit.py        --split time_ordered --fs_method none
python ft_transformer.py --split time_ordered --fs_method pearson
python gbdt.py           --split loco          --fs_method mcp
python mlp.py            --split time_ordered --fs_method none
python ridge.py          --split time_ordered --fs_method pearson
```

Outputs land at `../../output/<model>_c906_<split>[...]/` — note the path is two levels above `src/`, NOT inside the repo. Either run from a checkout nested two levels deep (matching the original layout), or change `out_dir` construction at the bottom of each `main()` if you want outputs inside the repo.

There are no tests, lint config, or CI in this repo.

## Working with the data

`db.zip` is the packaged dataset. The loader expects this layout after extraction:

```
db/c906-db/
  presim/              # or another presim_subdir name; chosen via --presim_subdir
    MMU_func.pkl
    cache_func.pkl
    csr_func.pkl
    exception_func.pkl
    interrupt_func.pkl
  pwr/
    MMU_pwr.pkl
    cache_pwr.pkl
    ...
```

`validate_presim_subdir` rejects path traversal and nested paths — `presim_subdir` must resolve to a direct child of `db/c906-db/`. Pass `--presim_subdir presim_large` (or similar) to use a different signal-extraction variant.

`db/inspect.ipynb` is the dataset exploration notebook.
