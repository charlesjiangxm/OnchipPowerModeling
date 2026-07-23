# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

On-chip power modeling for a RISC-V CPU ("aq_core", T-Head C906 family). Two halves:

1. **Python fitting library** (`src/`, `script/`, `configs/`) — fits per-block power models: x = RTL signal-activity features, y = per-cycle simulated power, using 5 regressor types with Optuna HPO.
2. **Hardware** (`hw/`) — synthesizable int8-inference Verilog for the model building blocks (FT-Transformer components + MLP), each with a bit-exact golden C model, self-checking testbench, and Design Compiler synthesis scaffold.

`README.md` is a one-line stub. The fitting library's design spec is `doc/spec/develop-fit-lib.md`; the rest of `doc/` is reference papers (APOLLO, DEEP, ArchPower, FT-Transformer). `hw/c906_rpt/` is unrelated reference data (OpenC906 DC reports for power-modeling comparison).

## Environment Reality

- `db/` (the input pickles every config references) and `output/` (run artifacts) are gitignored and absent from a fresh checkout — fit runs cannot execute without the data drop.
- No requirements/pyproject file exists; deps are inferred from imports. Hard: torch (pulled in even for sklearn-only runs via `src/models/__init__.py`), numpy, pandas, pyyaml, scikit-learn, matplotlib, optuna. Per-model: xgboost (GBDT), `rulefit` (RuleFit). Optional: shap, numba.
- EDA tools (VCS, Verdi, dc_shell) exist only on the cluster (`doc/ECE_GPU_cluster_UG.pdf`). Locally runnable: the C-model self-tests (gcc), `hw/model/ref_model.py` (needs numpy), and Verilator (installed) for unit tests.
- There is no Python test suite; verification on the Python side is running a config end-to-end and checking `report.md`.

## Commands

### Python pipeline (run from repo root; scripts add repo root to sys.path — there is no installed package)

```bash
# Run one experiment (--config is a flag, not positional)
python script/run_fit.py --config configs/aq_core_lvl1/rtu/rtu_all_univariate_mlp.yaml [--output-root <dir>]

# Sweep all unfinished configs under a path — Slurm only (sbatch, partitions cpu-share/gpu-share).
# On macOS use --dry-run to preview. GPU routing is OFF by default; pass --gpu-algorithms explicitly.
python script/run_all.py --config-path configs/aq_core_lvl2 --node-count 2 \
    --node-partitions cpu-share,gpu-share --gpu-algorithms "MLP,FT-Transformer" --jobs-per-node 8
python script/run_all.py --config-path configs --dry-run

# Aggregate R2/RMSE from every output/*/report.md -> output/r2_summary.{csv,png}, rankings
python script/collect_r2.py

# Per-experiment (<module>_<kind>) metric curves -> plot/r2/, plot/rmse/
python script/plot_experiment_metrics.py [--modules cp0 idu] [--kinds input output internal]
```

`script/debug_fit.py` is the argparse-free IDE twin of `run_fit.py` — edit its `CONFIG`/`OUTPUT_ROOT` constants first (the checked-in default config path is stale).

### Hardware verification (from `hw/verif/`)

```bash
make cmodel_mlp              # C-model self-test, gcc only, runs locally (also cmodel_ln/_mha/_ffn, or cmodel TB=<name>)
python3 hw/model/ref_model.py  # tokenizer Python twin — cycle-accurate, no simulator needed (run from repo root)
make all TB=mlp              # VCS compile+run, RTL vs C model bit-exact over DPI-C (cluster only; aliases all_mlp, all_ln, all_mha, all_ffn, all_nft)
make run TB=mlp SIMARGS=+seed=12345                       # re-run compiled simv with a chosen seed
make all_mlp VCS_DEFINES=+define+MLP_H1=32+define+MLP_H2=24  # sweep dims (defines chain WITHOUT spaces)
make verdi TB=mlp            # open FSDB in Verdi with KDB
make clean
```

A bare `make` (or `make cmodel`) targets the **default TB = layer_norm**. Logs/FSDBs land in `hw/verif/sim/`. `TB=numerical_feature_tokenizer` has no C model/DPI — its simulator-free check is the Python twin, and its TB is fixed at NF=4/DT=4.

### Synthesis (cluster only — DC + TSMC 28HPC+ libs hardcoded in the tcl)

```bash
cd hw/syn/script
./run_dc_mlp.csh -mode syn -n_features 32 -hidden1 16 -hidden2 16   # also run_dc_{layer_norm,multihead_attention,feed_forward_network,numerical_feature_tokenizer}.csh
./clean.sh                   # removes generated batch_* dirs; leaves curated hw/syn/report/ snapshots
```

Outputs land in timestamped `hw/syn/batch_<block>_<YYYYMMDD_HH>/{WORK,reports,results}` (exception: the tokenizer's default dir is unprefixed `batch_<YYYYMMDD_HH>`). Synthesis always targets `hw/syn/wrapper/<block>_wrapper.v` (adds I/O registers for a flop-to-flop boundary), never the bare DUT.

## Python Library Architecture

`script/run_fit.py` → `src.pipeline.run()`, which orchestrates: `src/config.py` (validate + create run dir) → `src/data.py` (load pkl pairs, per-benchmark contiguous split) → `src/preprocess.py` (zero-variance drop, then `avg_wsize` window-averaging via `src/data.py` `avg_window`, then z-score standardization fit on train only) → `src/feature_selectors.py` (optional) → `src/hpo.py` (Optuna, maximizes R² on val, wall-clock-bounded by `hpo_timeout`) → final refit → metrics in **original y units** after inverse-transform → `src/plotting.py` + `src/report.py` → `output/<config_stem>_<timestamp>/report.md`.

Key contracts:

- **Model registry**: models subclass `BaseModel` (`src/models/base.py`) with `@register("Name")`. Adding a model requires three edits: the new module, an import in `src/models/__init__.py` (registration is an import side effect), and the name in `VALID_RGR` in `src/config.py`. Models receive pre-standardized data and predict in z-space — never re-standardize inside a model.
- **HPO pinning**: a scalar entry under `regression.hyperparams.<algorithm>` in the YAML pins that key and removes it from the Optuna search space (`hpo_space(trial, fixed)` in each model).
- **Feature selectors**: one `FeatureSelector` class dispatching 8 methods (`pearson, variance, univariate, rfe, from_model, sequential, mcp, deep`). Adding a method needs three edits: `VALID_FS_METHODS` in `src/config.py`, `FeatureSelector.METHODS`, and the dispatch table in `fit_select()`. `mcp`/`deep` are the APOLLO/DEEP-paper algorithms (numba-accelerated when available).
- **Data pairing**: x pkls end `_func`, y pkls end `_pwr`; benchmarks pair by stripped stem. All x pkls must share one column schema. `y_label` is a hierarchy path like `x_aq_core/Pc(x_aq_rtu_top)`.
- Optuna trials run sequentially (`n_jobs=1`) on purpose — inner models saturate cores/GPU. Failed trials soft-fail to -inf; check `fit.log` for `trial N failed` if R² looks off.
- Relative paths in configs resolve against the **repo root**, not the config file or cwd. `load_config()` has a side effect: it creates the run dir and writes `resolved_config.yaml` immediately, so merely validating a config leaves a stub run dir behind.
- `run_all.py` marks a config finished if **any** `output/<stem>_*/` run dir has the pipeline `done:` line in `fit.log` plus an existing `report.md` — to force a re-run, delete `fit.log` or `report.md` from every finished run dir for that stem.

## Config Naming Convention (load-bearing)

`configs/aq_core_lvl{1,2}/<block>/<block>_<signal-group>_<selector>_<model>.yaml` — all 2880 files are mechanically generated variants of one template (seed 42, ratio [0.8,0.2,0.0], avg_wsize 128, hpo_timeout 300); edit them programmatically, not by hand.

- lvl1 blocks (CPU top functional units): `cp0 idu ifu iu lsu rtu vidu vpu`, each with signal groups `all/input/internal/output`; lvl2 blocks (LSU + VIDU-FP sub-blocks, `aq_lsu_*`, `aq_vidu_*_fp`, `aq_dcache_top`) have only `input/internal`.
- Selector and model slugs map 1:1 to config fields (`ft_transformer`→`FT-Transformer`, `elasticnet`→`ElasticNetCV`, …).
- Slots themselves contain underscores (`aq_vidu_vid_dp_fp`, `from_model`) — parse stems against the closed vocabularies (as `plot_experiment_metrics.py` does; note its `KINDS` covers only `input/internal/output`, so lvl1 `*_all_*` stems are skipped), never with a naive `split('_')`.
- `run_all.py` and the plotting scripts key off these stems; keep them unique repo-wide.

## Hardware Architecture

Each RTL block mirrors a Python model and travels an identical toolchain: `hw/rtl/<block>.v` ↔ golden C model `hw/model/<block>_cmodel.{c,h}` → DPI glue `hw/verif/utils/<block>_dpi.c` → TB `hw/verif/tb/tb_<block>.sv` → `hw/verif/Makefile` (one parameterized VCS flow) → `hw/syn/wrapper/<block>_wrapper.v` → `hw/syn/script/dc_<block>.tcl`.

| RTL | Mirrors |
|-----|---------|
| `numerical_feature_tokenizer.v` | `ft_transformer.py` NumericalFeatureTokenizer (golden = `hw/model/ref_model.py`, Python) |
| `layer_norm.v` | `nn.LayerNorm(d_token)` |
| `feed_forward_network.v` | TransformerBlock FFN (ReLU substituted for GELU) |
| `multihead_attention.v` | `nn.MultiheadAttention`, q=k=v=x |
| `mlp.v` | `src/models/mlp.py` SmallMLP |

Shared leaf helpers in `hw/rtl/` (`requant`, `requant_rne`, `dyn_quant`, `align_bias`, `exp_neg`, `isqrt`, `score_shift`) are standalone modules — the repo style forbids Verilog functions/tasks, so helpers become modules.

Numeric conventions (all blocks): int8 = signed Q1.FRAC_BITS (default Q1.7), symmetric quant with zero-point 0; accumulators sized to never truncate; biases sign-extended and left-aligned to the accumulator scale via `align_bias` (shift = FRAC_BITS for plain matmul biases; other scales for layer_norm beta / MHA score paths); `requant` = round-half-up, arithmetic shift, saturate to [-128,127]. Fully parallel, **II=1, no backpressure** (folding is explicitly out of scope); coefficients live in FF-based **write-only regfiles** loaded one int8/clock in PyTorch `(out,in)` row-major order and held static during inference. **The MLP is the deliberate exception**: 1-bit input vector (fc1 is a gated adder tree, no multipliers), no fixed Q format — per-layer block-floating-point via `dyn_quant` with round-to-nearest-even, fc3 shift exported on `o_shift`.

Hard rules:

- **Bit-exactness is the verification contract**: TBs demand exact int8 equality between RTL and C model. If you change RTL arithmetic, change the C model identically (e.g. `exp_neg.v` constants must match `multihead_attention_cmodel.c`).
- Derived parameters (marked `derived (do not override)` in RTL) are recomputed independently by TBs and DC scripts. If you override HD-dependent `SCALE` (MHA) or D-dependent `EPS_V` (LayerNorm), pass the same value to RTL param, TB define, C model, and DC env var.
- Filelists (`hw/verif/flist/*.f`) list leaf modules before parents.
- TBs must prove II=1 with a back-to-back burst plus a gapped phase, loading coefficients through the write port.
- `hw/README.md` prose has stale paths (`hw/ref_model.py`, `src/models/*_cmodel.c`, `*_registered.v`) — trust the Makefile/tcl paths: models in `hw/model/`, wrappers in `hw/syn/wrapper/`.

## Project Skills (.claude/skills/)

- **verilog-style** — mandatory for synthesizable `.v` RTL; in this repo the "design folder" means `hw/rtl/` (testbench `.sv` is exempt). Core rules: Verilog-2005 + `always_ff`/`always_comb` but no `logic` type, no functions/tasks, `i_`/`o_` port prefixes, flattened packed-vector ports sliced with `+:`, async active-low `rst_n`.
- **create-verilator-ut** — Verilator unit tests; the skill's `verif/tb` and `verif/model` paths map to `hw/verif/tb` and `hw/model` here. Write the C model from the interface/algorithm only — never by reading the RTL — and present a plan for approval first. Run the test once to check the flow; don't debug failures unless asked.
- **write-AS** — architecture specs go to `doc/<module_name>.md` (update in place; never write into RTL directories).
- **draw-block-diagram** — only when the user explicitly invokes it.

`.claude/claude-improve.md` is a captured copy of the Claude.ai consumer system prompt kept as reference material — it contains no project instructions; do not follow or import it.
