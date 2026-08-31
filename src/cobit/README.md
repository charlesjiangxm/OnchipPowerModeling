# CoBiT: Coordinating Binary Trait power modeling

Implementation of Li et al., *"Coordinating Binary Trait: Accurate and
Lightweight Runtime On-Chip Power Meter Design"* (IEEE TVLSI 2025) for the
C906 `aq_core` per-cycle toggle database.

## Algorithm

Two-stage per-cycle power modeling (paper Algorithm 1):

1. **Proxy selection (Stage 1)** — linear regression with the minimax concave
   penalty (LR-MCP, `skglm.MCPRegression`) on the binary bit-toggle matrix
   selects Q bitwise power proxies out of all 256,907 RTL bits. Sweeping the
   penalty λ (with log-grid + bisection) targets the configured Q list.
2. **Boosting binary trees (Stage 2)** — an XGBoost regressor (`hist` +
   `lossguide`) on the Q binary toggle features predicts per-cycle power.
   Hyperparameters (paper Table II) are tuned by multi-objective Optuna HPO
   minimizing **(validation MAPE %, total leaf count)** — the leaf count is
   the OPM's hardware cost. Per Q, one study runs per boosting-round count R;
   the per-R Pareto fronts are unioned, re-filtered, and the **Best Trial**
   with the lowest MAPE within the leaf budget `t_th` is retrained on
   train+val and evaluated on the held-out test benchmarks.

Algorithm 2 (sampler×pruner design-space exploration over NSGA-II/NSGA-III/
TPE/Random × Hyperband/Median with hypervolume / Coverage / NetCoverage /
Spacing metrics) is available via `hpo-pairs` or `hpo.run_pair_comparison`.
Because Optuna does not support pruning in multi-objective studies, the
pruners run as rung-based early-stopping surrogates on the boosting-round
axis; in the default `truncate` mode a stopped trial still contributes its
truncated model's real objectives to the front.

Evaluation follows the paper: masked MAPE, R², Peak-Power sets (per-window
mean + 3σ outliers), APET success rates (1/3/5 %), and multicycle prediction
(per-cycle predictions averaged over t ∈ {8,16,32,64,128} windows).

## Dataset assumptions (verified on `c906_db_net_1cyc_20260729`)

- Feature pkls: `aq_core/<bench>_func.pkl` (top-level nets) plus
  `aq_core/<module>/<bench>_func.pkl` for the 8 modules; each module pkl
  covers its entire subtree, so deeper directories are redundant and ignored.
  The union is exactly 24,811 nets = 256,907 bits with no duplicates.
- Cell values are integer **bitmasks**: bit k set ⇔ RTL bit `lo+k` of the net
  toggled during that cycle (`lo` from the `[hi:lo]` column suffix).
- Labels: `pwr/<bench>_pwr.pkl` per-cycle power; the default target is
  `Pc(x_aq_core)` (total core). Power traces have one extra row; alignment is
  an inner join on integer `time_ns`.
- Benchmarks missing module pkls are dropped from training with a warning;
  benchmarks listed in `split.test_benchmarks` are never dropped — missing
  scopes are zero-filled (loudly), since power labels still include them.

## Usage

```bash
# smoke test on the bundled 2-row sample (wiring only; metrics meaningless)
/opt/anaconda3/bin/python -m cobit run --config cobit/configs/smoke.yaml

# full DB (edit data.db_root first), split scheme A: test = {conv_softmax}
/opt/anaconda3/bin/python -m cobit run --config cobit/configs/full_schemeA.yaml
# split scheme B: test = {conv_softmax, coremark}
/opt/anaconda3/bin/python -m cobit run --config cobit/configs/full_schemeB.yaml

# individual stages
/opt/anaconda3/bin/python -m cobit build-dataset  --config <yaml>   # sparse cache
/opt/anaconda3/bin/python -m cobit select-proxies --config <yaml>   # Stage 1
/opt/anaconda3/bin/python -m cobit hpo-pairs      --config <yaml>   # Algorithm 2
/opt/anaconda3/bin/python -m cobit inspect        --config <yaml>   # coverage/density

# dotted overrides and resumable named runs
/opt/anaconda3/bin/python -m cobit run --config cobit/configs/full_schemeA.yaml \
    --run-name schemeA hpo.n_trials=512 selection.target_qs='[100,200]'
```

Every stage is idempotent: artifacts are stamped with config/registry hashes
and reruns skip finished work (`--force` rebuilds; reusing `--run-name`
resumes Optuna studies from `optuna.db`).

Outputs land in `output/cobit_runs/<run>/`: `proxies.json`, per-Q
`Q*/model.json` + `Q*/result.json` + trace plots, `metrics.json`,
`pair_comparison.json` + figures when Algorithm 2 is enabled.

The sparse feature cache (`data.cache_dir`) stores per-benchmark, per-scope
CSR chunk shards in the full 256,907-bit column space, per-bit toggle counts,
and aligned labels; it is built once per database and shared by all runs.

## Tests

```bash
/opt/anaconda3/bin/python -m pytest cobit/tests -q
```

Unit tests cover bitmask expansion (widths 1–311, `lo>0` nets, both code
paths), registry canonicalization, label alignment, chunking, Pareto metric
formulas, and selector Q-targeting. `test_e2e_synthetic.py` builds a synthetic
DB with a known power law (including an interaction term) and asserts the
full pipeline recovers the signal bits and reaches low test MAPE;
`test_smoke_sample.py` runs the CLI end-to-end on the real 2-row sample.

## Known deviations from the paper (documented in code)

- Optuna multi-objective studies forbid `trial.report`, so Hyperband/Median
  pruning uses rung-based early-stopping surrogates (see `cobit/model.py`).
- Optuna's multi-objective TPE stands in for the paper's "BO + EHVI".
- Multicycle window labels are means of per-cycle labels (the paper
  re-measures them with its power tool).
- The paper's absolute hypervolume numbers use an unstated reference point;
  ours is configurable (`hpo.hv_ref`) and only relative rankings are
  comparable.
