"""X-OPM Feature Interaction & Model Fitting (spec
``doc/spec/x-opm-trainning-procedure.md`` section "Feature Interaction and Model
Fitting").

A RuleFit model per module. The spec cites the XGBoost DART and monotonic tutorials,
but the ``third_party/rulefit`` fork's tree stage is sklearn and supports neither, so
this module keeps RuleFit's *methodology* (rule generation from a tree ensemble + a
sparse non-negative linear model over ``[linear terms | rule indicators]`` + the
Friedman & Popescu H-statistic) while swapping the tree stage for XGBoost:

  * ``booster='gbtree'`` + DART dropout params (``rate_drop``/``skip_drop``/``one_drop``)
    -- the non-deprecated DART API.
  * ``monotone_constraints`` = +1 on every feature (more toggling -> more power).
  * Optuna HPO over the DART + tree hyperparameters (per the DART tutorial).
  * Rules extracted from ``booster.trees_to_dataframe()`` (every non-root node -> a
    conjunction of split conditions), each carrying the summed split ``Gain``.
  * Low-gain rules pruned (adjustable threshold), then the surviving rules + FriedScale
    linear terms are fit by a **non-negative** ``LassoCV`` / ``ElasticNetCV``
    (``allow_negative_coef=False`` -> ``positive=True``), so only features positively
    correlated with power get a coefficient.

The eight modules (cp0, idu, ifu, iu, lsu, rtu, vidu, vpu) are trained separately; test
= conv_softmax + coremark, everything else is training (plus the ``random`` case for
cp0/idu/ifu/iu/vidu). All artifacts land in ``analysis/x-opm/<YYYY-MM-DD-HH-MM>/``.
Per-module predictions are summed to reconstruct whole-core (aq_core) power and scored
against the true ``Pc(x_aq_core)`` (the one place raw ``dataset/`` is read).

CLI (interpreter ~/anaconda3/bin/python):
  python src/xopm_lib/model_regression.py --module cp0 --n-trials 30
  python src/xopm_lib/model_regression.py --all --n-trials 30
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as _sk_r2

import xgboost as xgb
import optuna

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "third_party", "rulefit"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from rulefit import RuleFitRegressor, RuleEnsemble, Rule, RuleCondition  # noqa: E402

try:
    from xopm_lib import feature_selection as fs  # noqa: E402
except ImportError:                               # run as a bare script
    import feature_selection as fs                # noqa: E402

log = logging.getLogger("xopm.model_regression")

MODULES = ("cp0", "idu", "ifu", "iu", "lsu", "rtu", "vidu", "vpu")
TEST_BENCHMARKS = ("conv_softmax", "coremark")
RAND_CASE = "random"
DATASET_PROCESSED = os.path.join(_REPO_ROOT, "dataset_processed")
ANALYSIS_ROOT = os.path.join(_REPO_ROOT, "analysis", "x-opm")

# aq_core reconstruction reference (spec: the only raw-dataset read)
RAW_DB = os.path.join(_REPO_ROOT, "dataset", "c906_db_net_1cyc_20260729")
RAW_PWR_DIR = os.path.join(RAW_DB, "pwr")
AQCORE_COL = "Pc(x_aq_core)"

POWER_SCALE = 1000.0   # dataset target is Watts; plot in mW
POWER_UNIT = "mW"
PLOT_MAX_POINTS = 50_000
INTERP_SAMPLE = 3000   # rows sampled for H-statistic / SHAP (speed)
PREDICT_CHUNK = 100_000  # rows per predict chunk (rule matrix is n x n_rules)

# Row caps that keep the rule-indicator design matrix (n_rows x n_rules) tractable.
DEFAULT_RAND_MAX_ROWS = 200_000   # cap the giant random-stimulus case before pooling
DEFAULT_HPO_ROWS = 150_000        # rows sampled for the Optuna booster search
DEFAULT_FIT_ROWS = 120_000        # rows sampled for the RuleFit linear (Lasso) stage
DEFAULT_MAX_RULES = 800           # keep at most this many rules (top by gain)


# ===========================================================================
# Metrics (ported from git f636ea9:x-opm/train.py -- masked MAPE + RMSE + R2)
# ===========================================================================

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(_sk_r2(y, yhat))


def mape_percent(y: np.ndarray, yhat: np.ndarray, eps_frac: float = 1e-3) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    eps = eps_frac * float(np.median(np.abs(y))) if y.size else 0.0
    mask = np.abs(y) > eps
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def metric_block(y, yhat) -> dict:
    return {"r2": r2_score(y, yhat), "mape": mape_percent(y, yhat), "rmse": rmse(y, yhat)}


def predict_chunked(rf, X: np.ndarray, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """Predict in row chunks -- ``predict`` materializes an n x n_rules matrix, which
    is large for the ~600k-row test benchmarks."""
    if len(X) <= chunk:
        return rf.predict(X)
    return np.concatenate([rf.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


# ===========================================================================
# Data assembly
# ===========================================================================

class Split:
    """One data split: feature matrix, target, per-row (bench, time_ns), plot slices."""

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                 slices: dict[str, tuple[int, int]]):
        self.X = X
        self.y = y
        self.meta = meta                # columns: bench, time_ns
        self.slices = slices            # bench -> (lo, hi) in concatenated order


def _slices_from_meta(meta: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Recover contiguous (lo, hi) plot slices per benchmark from a meta frame."""
    b = meta["bench"].to_numpy()
    slices, start = {}, 0
    for i in range(1, len(b) + 1):
        if i == len(b) or b[i] != b[start]:
            slices.setdefault(str(b[start]), (start, i))
            start = i
    return slices


def subset_split(split: Split, mask: np.ndarray) -> Split:
    """A row subset of a Split (mask over rows), with slices recomputed."""
    meta = split.meta[mask].reset_index(drop=True)
    return Split(split.X[mask], split.y[mask], meta, _slices_from_meta(meta))


def _reference_columns(train_dir: str, selected_cols: list[str] | None) -> list[str]:
    """Canonical feature order (control->data->config) for a module."""
    cases = fs.list_cases(train_dir)
    real = [c for c in cases if c != RAND_CASE]
    ref_case = real[0] if real else cases[0]
    X, _ = fs.load_case(train_dir, ref_case)
    cols = list(X.columns)
    if selected_cols is not None:
        sel = set(selected_cols)
        cols = [c for c in cols if c in sel]   # keep canonical order, subset
    return cols


def _block_mean(X: np.ndarray, y: np.ndarray, t: np.ndarray, win_size: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-overlapping ``win_size``-cycle block means of features + target.

    Features -> f32 block mean, target -> f64 block mean, ``time_ns`` -> the FIRST
    cycle of each block (an exact int64 label, so the aq_core reconstruction merge
    key survives windowing). The trailing partial block is truncated. ``win_size<=1``
    is a no-op; a stream shorter than one full window yields zero (correctly shaped)
    rows. Mirrors ``x-opm/train.py:_block_mean`` (git f636ea9), extended to carry
    ``time_ns``.
    """
    if win_size <= 1:
        return X, y, t
    nb = len(y) // win_size          # number of full blocks
    m = nb * win_size
    if nb == 0:                      # benchmark shorter than one full window
        return X[:0], y[:0], t[:0]
    # explicit nb (not -1): a zero-column X -- used by the aq_core truth loader --
    # has size 0, which -1 cannot resolve.
    Xw = X[:m].reshape(nb, win_size, X.shape[1]).mean(axis=1).astype(np.float32)
    yw = y[:m].reshape(nb, win_size).mean(axis=1).astype(np.float64)
    tw = t[:m].reshape(nb, win_size)[:, 0].astype(np.int64)
    return Xw, yw, tw


def _case_arrays(split_dir: str, case: str, cols: list[str], win_size: int = 1
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[cols] float32, y float64, time_ns int64) for one case, optionally
    averaged into non-overlapping ``win_size``-cycle blocks (per case, so a window
    never straddles two benchmarks)."""
    X, y = fs.load_case(split_dir, case)
    X = X.reindex(columns=cols)
    if X.isna().any().any():
        X = X.fillna(0.0)
    return _block_mean(X.to_numpy(np.float32), y.to_numpy(np.float64),
                       X.index.to_numpy(np.int64), win_size)


def build_splits(module: str, dataset_dir: str = DATASET_PROCESSED,
                 val_fraction: float = 0.2, selected_cols: list[str] | None = None,
                 rand_max_rows: int = DEFAULT_RAND_MAX_ROWS, seed: int = 0,
                 win_size: int = 1
                 ) -> tuple[dict[str, Split], list[str]]:
    """Assemble train/val/test splits for one module.

    val = contiguous tail ``val_fraction`` of each *real* training benchmark (repo
    convention); the ``random`` augmentation case goes entirely to train (no
    whole-core truth) but is capped at ``rand_max_rows`` rows so it does not swamp
    the pooled training set or the rule-indicator matrix. test = conv_softmax +
    coremark, pooled. Returns ``({'train','val','test'}: Split, feature_names)``.
    """
    train_dir = os.path.join(dataset_dir, "trainset", module)
    test_dir = os.path.join(dataset_dir, "testset", module)
    cols = _reference_columns(train_dir, selected_cols)
    rng = np.random.default_rng(seed)

    tr_X, tr_y, tr_meta = [], [], []
    va_X, va_y, va_meta = [], [], []
    tr_slices: dict[str, tuple[int, int]] = {}
    va_slices: dict[str, tuple[int, int]] = {}
    tr_pos = va_pos = 0

    for case in fs.list_cases(train_dir):
        X, y, t = _case_arrays(train_dir, case, cols, win_size)
        if case == RAND_CASE and len(y) > rand_max_rows:
            idx = np.sort(rng.choice(len(y), rand_max_rows, replace=False))
            X, y, t = X[idx], y[idx], t[idx]
        n = len(y)
        if n == 0:                      # bench shorter than one window -> no rows
            continue
        if case == RAND_CASE:
            n_val = 0                       # random: all to train
        else:
            n_val = int(round(n * val_fraction))
            n_val = min(max(n_val, 1), n - 1) if n >= 2 else 0
        n_tr = n - n_val
        tr_X.append(X[:n_tr]); tr_y.append(y[:n_tr])
        tr_meta.append(pd.DataFrame({"bench": case, "time_ns": t[:n_tr]}))
        tr_slices[case] = (tr_pos, tr_pos + n_tr); tr_pos += n_tr
        if n_val:
            va_X.append(X[n_tr:]); va_y.append(y[n_tr:])
            va_meta.append(pd.DataFrame({"bench": case, "time_ns": t[n_tr:]}))
            va_slices[case] = (va_pos, va_pos + n_val); va_pos += n_val

    te_X, te_y, te_meta = [], [], []
    te_slices: dict[str, tuple[int, int]] = {}
    te_pos = 0
    for case in fs.list_cases(test_dir):
        X, y, t = _case_arrays(test_dir, case, cols, win_size)
        if len(y) == 0:                 # bench shorter than one window -> no rows
            continue
        te_X.append(X); te_y.append(y)
        te_meta.append(pd.DataFrame({"bench": case, "time_ns": t}))
        te_slices[case] = (te_pos, te_pos + len(y)); te_pos += len(y)

    def _pack(Xs, ys, metas, slices) -> Split:
        return Split(np.concatenate(Xs), np.concatenate(ys),
                     pd.concat(metas, ignore_index=True), slices)

    splits = {
        "train": _pack(tr_X, tr_y, tr_meta, tr_slices),
        "val": _pack(va_X, va_y, va_meta, va_slices),
        "test": _pack(te_X, te_y, te_meta, te_slices),
    }
    log.info("%s splits: train=%d val=%d test=%d, features=%d", module,
             len(splits["train"].y), len(splits["val"].y), len(splits["test"].y),
             len(cols))
    return splits, cols


# ===========================================================================
# XGBoost -> RuleFit rule extraction
# ===========================================================================

def _feature_index(name: str) -> int:
    """Map an XGBoost positional feature name ('f7') to its column index."""
    return int(name[1:])


def extract_rules_from_booster(booster: xgb.Booster, feature_names: list[str]
                               ) -> tuple[list[Rule], np.ndarray, list[str]]:
    """Turn every non-root tree node into a :class:`Rule` (fork type).

    Mirrors ``extract_rules_from_tree``: each node except the root yields a rule =
    the conjunction of split conditions on the root->node path (Yes branch -> ``<=``
    Split, No branch -> ``>`` Split, matching sklearn's left/right convention).
    Duplicate rules (identical condition set, order-insensitive via ``Rule.__hash__``)
    are merged and their gains summed. Each rule's **gain** is the sum of the split
    ``Gain`` of the interior nodes on its path.

    Returns ``(rules, gains, how_built)`` aligned by index.
    """
    tdf = booster.trees_to_dataframe()
    merged: dict[int, dict] = {}   # rule hash -> {rule, gain, how}

    for tree_id, tdf_tree in tdf.groupby("Tree"):
        nodes = {row.ID: row for row in tdf_tree.itertuples(index=False)}
        # find the root of this tree (node index 0)
        root_id = f"{tree_id}-0"
        if root_id not in nodes:
            continue

        def emit(rule_conditions, gain_path, how_path):
            rule = Rule(list(rule_conditions), 0.0)
            h = hash(rule)
            if h not in merged:
                merged[h] = {"rule": rule, "gain": 0.0, "how": " & ".join(how_path)}
            merged[h]["gain"] += gain_path

        def traverse(node_id, conditions, gain_path, how_path):
            row = nodes[node_id]
            if node_id != root_id and conditions:
                emit(conditions, gain_path, how_path)
            if row.Feature == "Leaf":
                return
            fidx = _feature_index(row.Feature)
            thr = float(row.Split)
            gain_here = gain_path + float(row.Gain)
            yes, no = row.Yes, row.No
            fname = feature_names[fidx]
            cond_yes = RuleCondition(fidx, thr, "<=", 1.0, fname)
            cond_no = RuleCondition(fidx, thr, ">", 1.0, fname)
            traverse(yes, conditions + [cond_yes], gain_here,
                     how_path + [f"{fname} <= {thr:g}"])
            traverse(no, conditions + [cond_no], gain_here,
                     how_path + [f"{fname} > {thr:g}"])

        traverse(root_id, [], 0.0, [])

    rules = [m["rule"] for m in merged.values()]
    gains = np.array([m["gain"] for m in merged.values()], dtype=float)
    how = [m["how"] for m in merged.values()]
    return rules, gains, how


class XGBRuleEnsemble(RuleEnsemble):
    """A :class:`RuleEnsemble` populated from pre-built rules (no sklearn trees)."""

    def __init__(self, rules: list[Rule], feature_names: list[str]):
        self.tree_list = []
        self.feature_names = feature_names
        self.include_interior_rules = True
        self.rules = list(rules)


# ===========================================================================
# XGBRuleFit -- the bridge
# ===========================================================================

class XGBRuleFit(RuleFitRegressor):
    """RuleFit with an XGBoost (DART + monotone) tree stage.

    Reuses the fork's non-negative linear refit, FriedScale linear terms, ``get_rules``,
    ``predict`` and the Friedman H-statistic engine unchanged -- only the tree/rule
    stage is replaced. ``fit`` trains the booster with all-+1 monotone constraints,
    extracts and gain-prunes rules, then fits the sparse non-negative linear model.
    """

    def __init__(self, xgb_params: dict, num_boost_round: int,
                 gain_threshold: float = 0.0, max_rules: int = DEFAULT_MAX_RULES,
                 fit_rows: int = DEFAULT_FIT_ROWS, penalty: str = "l1",
                 l1_ratio: float = 0.5, fit_intercept: bool = True,
                 lin_trim_quantile: float = 0.025, lin_standardise: bool = True,
                 cv: int = 3, random_state: int = 0):
        super().__init__(rfmode="regress", model_type="rl", penalty=penalty,
                         l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                         allow_negative_coef=False, lin_trim_quantile=lin_trim_quantile,
                         lin_standardise=lin_standardise, cv=cv,
                         random_state=random_state)
        self.xgb_params = xgb_params
        self.num_boost_round = num_boost_round
        self.gain_threshold = gain_threshold
        self.max_rules = max_rules
        self.fit_rows = fit_rows

    def fit(self, X, y=None, feature_names=None, monotone_constraints=None):
        from rulefit import Winsorizer, FriedScale
        from sklearn.linear_model import LassoCV, ElasticNetCV

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        linear_kwargs = self._resolve_penalty()   # sets self.allow_negative_coef_
        self.feature_names = list(feature_names)
        self.n_features_in_ = X.shape[1]

        # ---- tree stage: XGBoost DART + monotone (on the full training set) --
        dtrain = xgb.DMatrix(X, label=y)           # positional f{idx} names
        params = dict(self.xgb_params)
        if monotone_constraints is not None:
            params["monotone_constraints"] = monotone_constraints
        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)

        # The linear (Lasso) stage builds an n_rows x n_rules indicator matrix, so it
        # runs on a bounded row subsample; the booster above already used every row.
        X, y = _row_sample(X, y, self.fit_rows, self.random_state)

        rules, gains, how = extract_rules_from_booster(self.booster_, self.feature_names)
        # gain-threshold pruning (+ optional max_rules cap by gain)
        keep_mask = gains >= self.gain_threshold
        order = np.argsort(-gains)
        kept_idx = [i for i in order if keep_mask[i]][: self.max_rules]
        kept_set = set(kept_idx)
        kept_rules = [rules[i] for i in kept_idx]
        kept_gains = [gains[i] for i in kept_idx]
        kept_how = [how[i] for i in kept_idx]
        self.dropped_rules_ = [
            {"rule": str(rules[i]), "how_built": how[i], "gain": float(gains[i])}
            for i in order if i not in kept_set]
        self.rule_ensemble = XGBRuleEnsemble(kept_rules, self.feature_names)
        self.rule_gains_ = np.array(kept_gains, dtype=float)
        self.rule_how_ = kept_how
        log.info("rules: %d extracted, %d kept (gain>=%.4g), %d dropped",
                 len(rules), len(kept_rules), self.gain_threshold,
                 len(self.dropped_rules_))

        # ---- linear-term preprocessing (fork convention) ------------------
        self.winsorizer = Winsorizer(trim_quantile=self.lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)
        self.winsorizer.train(X)
        winsorized_X = self.winsorizer.trim(X)
        self.stddev = np.std(winsorized_X, axis=0)
        self.mean = np.mean(winsorized_X, axis=0)
        if self.lin_standardise:
            self.friedscale.train(X)

        X_rules = self.rule_ensemble.transform(X)
        if X_rules.shape[1] > 0:
            self.rule_ensemble.update_support(X_rules)
        self.n_linear_terms_ = X.shape[1]
        self.n_rules_ = X_rules.shape[1]
        X_concat = self._design_matrix(X, X_rules=X_rules)

        # ---- sparse NON-NEGATIVE linear model (eq. 25) --------------------
        max_iter = self._resolve_max_iter()
        self.max_iter_ = max_iter
        common = dict(cv=self.cv, max_iter=max_iter, tol=self.tol, n_jobs=self.n_jobs,
                      random_state=self.random_state,
                      fit_intercept=linear_kwargs["fit_intercept"],
                      positive=linear_kwargs["positive"])
        if linear_kwargs["penalty"] == "elasticnet":
            self.lscv = ElasticNetCV(l1_ratio=linear_kwargs["l1_ratio"], **common)
        else:
            self.lscv = LassoCV(**common)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lscv.fit(X_concat, y)
        self.coef_ = self.lscv.coef_
        self.intercept_ = self.lscv.intercept_
        return self


# ===========================================================================
# HPO (Optuna over DART + tree params; objective = booster val RMSE)
# ===========================================================================

BASE_XGB = dict(booster="gbtree", objective="reg:squarederror", tree_method="hist",
                verbosity=0)


def _suggest(trial: "optuna.Trial") -> tuple[dict, int]:
    params = dict(BASE_XGB)
    params.update(
        # shallow trees keep the rule count (2*(leaves-1) per tree) tractable and
        # the interaction order low, as RuleFit intends.
        eta=trial.suggest_float("eta", 1e-3, 0.5, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        # DART dropout (non-deprecated API: gbtree + dropout params)
        rate_drop=trial.suggest_float("rate_drop", 0.0, 0.3),
        skip_drop=trial.suggest_float("skip_drop", 0.0, 0.5),
        one_drop=trial.suggest_categorical("one_drop", [0, 1]),
        sample_type=trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
        normalize_type=trial.suggest_categorical("normalize_type", ["tree", "forest"]),
    )
    nbr = trial.suggest_int("num_boost_round", 50, 300)
    return params, nbr


def _row_sample(X: np.ndarray, y: np.ndarray, n_max: int, seed: int
                ) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= n_max:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(y), n_max, replace=False))
    return X[idx], y[idx]


def run_hpo(splits: dict[str, Split], monotone: str, n_trials: int, seed: int,
            hpo_rows: int = DEFAULT_HPO_ROWS) -> tuple[dict, int, "optuna.Study"]:
    Xtr, ytr = _row_sample(splits["train"].X, splits["train"].y, hpo_rows, seed)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(splits["val"].X, label=splits["val"].y)
    yval = splits["val"].y
    log.info("HPO on %d train rows (of %d), %d trials", len(ytr),
             len(splits["train"].y), n_trials)

    def objective(trial):
        params, nbr = _suggest(trial)
        params["monotone_constraints"] = monotone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bst = xgb.train(params, dtr, num_boost_round=nbr)
        return rmse(yval, bst.predict(dval))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(BASE_XGB)
    bp = dict(study.best_params)
    nbr = bp.pop("num_boost_round")
    best.update(bp)
    best["monotone_constraints"] = monotone
    log.info("HPO best val RMSE=%.6g params=%s nbr=%d", study.best_value, bp, nbr)
    return best, nbr, study


# ===========================================================================
# Plots
# ===========================================================================

def _subsample(n: int, seed: int = 0) -> np.ndarray:
    if n <= PLOT_MAX_POINTS:
        return np.arange(n)
    idx = np.random.RandomState(seed).choice(n, PLOT_MAX_POINTS, replace=False)
    idx.sort()
    return idx


def plot_pred_vs_time(split: Split, yhat: np.ndarray, name: str, path: str,
                      win_size: int = 1) -> None:
    """True + predicted power (mW) vs cycle (or window) index, benchmark boundaries
    marked."""
    y = np.asarray(split.y, float) * POWER_SCALE
    yh = np.asarray(yhat, float) * POWER_SCALE
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, color="#1f77b4", lw=0.8, alpha=0.9, label="true", rasterized=True)
    ax.plot(x, yh, color="#ff7f0e", lw=0.8, alpha=0.9, label="predicted",
            rasterized=True)
    if len(y):
        ytop = ax.get_ylim()[1]
        for b, (lo, hi) in split.slices.items():
            ax.axvline(hi, color="grey", lw=0.4, ls=":")
            ax.text(0.5 * (lo + hi), ytop, b, fontsize=6, ha="center", va="top",
                    rotation=90, color="grey")
    unit = f"{win_size}-cycle window index" if win_size > 1 else "cycle index"
    ax.set_xlabel(f"{unit} (concatenated benchmarks)")
    ax.set_ylabel(f"power ({POWER_UNIT})")
    ax.set_title(f"Power vs time [{name}]")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_residual_panels(preds: dict[str, tuple[np.ndarray, np.ndarray]],
                         name: str, path: str) -> None:
    """3-panel residual map (train/val/test): residual (mW) vs predicted (mW)."""
    order = [s for s in ("train", "val", "test") if s in preds and len(preds[s][0])]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4.2),
                             squeeze=False)
    for ax, split in zip(axes[0], order):
        y, yh = (np.asarray(a, float) * POWER_SCALE for a in preds[split])
        resid = y - yh
        idx = _subsample(len(y))
        ax.scatter(yh[idx], resid[idx], s=2, alpha=0.3, rasterized=True)
        ax.axhline(0.0, color="r", lw=0.8, ls="--")
        ax.set_xlabel(f"predicted power ({POWER_UNIT})")
        ax.set_ylabel(f"residual ({POWER_UNIT})")
        ax.set_title(f"{split} (n={len(y)})")
    fig.suptitle(f"Residuals [{name}]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_h_bar(frame: pd.DataFrame, label_cols: list[str], title: str, path: str,
               top: int = 20) -> None:
    """Horizontal bar of Friedman H, from an interaction_* DataFrame."""
    if frame is None or not len(frame):
        _placeholder(path, f"{title}\n(no interactions -- additive model or no rules)")
        return
    frame = frame.head(top).iloc[::-1]
    labels = frame[label_cols].astype(str).agg("  x  ".join, axis=1)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.35 * len(frame) + 1)))
    ax.barh(range(len(frame)), frame["H"].values, color="#2a78d6")
    ax.set_yticks(range(len(frame))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Friedman H"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _placeholder(path: str, text: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=9)
    ax.axis("off"); fig.savefig(path, dpi=110); plt.close(fig)


def plot_shap_beeswarm(booster: xgb.Booster, X: np.ndarray, feature_names: list[str],
                       path: str, top: int = 20, seed: int = 0) -> None:
    """Hand-rolled SHAP beeswarm from XGBoost native TreeSHAP (pred_contribs)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, min(INTERP_SAMPLE, n), replace=False) if n else np.arange(0)
    Xs = X[idx]
    contribs = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]  # drop bias
    mean_abs = np.abs(contribs).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:top]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(top_idx) + 1)))
    for row, f in enumerate(top_idx[::-1]):
        sv = contribs[:, f] * POWER_SCALE
        fv = Xs[:, f].astype(float)
        rng_v = np.ptp(fv)
        color = (fv - fv.min()) / rng_v if rng_v > 0 else np.zeros_like(fv)
        jitter = (rng.random(len(sv)) - 0.5) * 0.6
        sc = ax.scatter(sv, np.full(len(sv), row) + jitter, c=color, cmap="coolwarm",
                        s=4, alpha=0.5, rasterized=True, vmin=0, vmax=1)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[f] for f in top_idx[::-1]], fontsize=6)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"SHAP value ({POWER_UNIT})")
    ax.set_title("SHAP beeswarm (XGBoost TreeSHAP)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("feature value (norm)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_shap_interactions(booster: xgb.Booster, X: np.ndarray,
                           feature_names: list[str], path: str, top: int = 20,
                           seed: int = 0) -> None:
    """Top interacting feature pairs by mean |SHAP interaction| (native TreeSHAP)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # interactions are O(F^2) per row -- sample rows AND restrict to top features.
    idx = rng.choice(n, min(800, n), replace=False) if n else np.arange(0)
    Xs = X[idx]
    contribs = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
    topf = np.argsort(-np.abs(contribs).mean(axis=0))[:min(40, len(feature_names))]
    inter = booster.predict(xgb.DMatrix(Xs), pred_interactions=True)
    inter = inter[:, :-1, :-1][:, topf][:, :, topf]
    mean_abs = np.abs(inter).mean(axis=0) * POWER_SCALE
    np.fill_diagonal(mean_abs, 0.0)
    pairs = []
    for a in range(len(topf)):
        for b in range(a + 1, len(topf)):
            pairs.append((mean_abs[a, b], topf[a], topf[b]))
    pairs.sort(reverse=True)
    pairs = pairs[:top]
    if not pairs:
        _placeholder(path, "SHAP interactions\n(no off-diagonal interaction)")
        return
    labels = [f"{feature_names[i]}  x  {feature_names[j]}" for _, i, j in pairs][::-1]
    vals = [v for v, _, _ in pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(pairs) + 1)))
    ax.barh(range(len(pairs)), vals, color="#eb6834")
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(f"mean |SHAP interaction| ({POWER_UNIT})")
    ax.set_title("Top SHAP interaction pairs")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ===========================================================================
# rule.csv + report
# ===========================================================================

def write_rule_csv(rf: XGBRuleFit, path: str) -> pd.DataFrame:
    """Kept rules (with gain + coef + importance + how-built) and dropped rules,
    sorted by gain descending. Linear terms are appended (gain = NaN)."""
    got = rf.get_rules(scaled=False)          # rule/linear rows in ensemble order
    lin = got[got["type"] == "linear"].copy()
    rul = got[got["type"] == "rule"].copy().reset_index(drop=True)

    rul["gain"] = rf.rule_gains_[: len(rul)]
    rul["how_built"] = rf.rule_how_[: len(rul)]
    rul["dropped"] = False
    rul = rul.rename(columns={"rule": "name"})

    lin["gain"] = np.nan
    lin["how_built"] = "linear term"
    lin["dropped"] = False
    lin = lin.rename(columns={"rule": "name"})

    dropped = pd.DataFrame(rf.dropped_rules_)
    if len(dropped):
        dropped = dropped.rename(columns={"rule": "name"})
        dropped["type"] = "rule"
        dropped["coef"] = np.nan
        dropped["support"] = np.nan
        dropped["importance"] = np.nan
        dropped["n_variables"] = np.nan
        dropped["dropped"] = True

    cols = ["name", "type", "how_built", "gain", "coef", "support", "importance",
            "n_variables", "dropped"]
    out = pd.concat([rul, dropped, lin], ignore_index=True)
    out = out.reindex(columns=cols)
    out = out.sort_values("gain", ascending=False, na_position="last").reset_index(
        drop=True)
    out.to_csv(path, index=False)
    return out


def write_report(module: str, out_dir: str, cfg: dict, metrics: dict,
                 dims: dict, rule_df: pd.DataFrame) -> None:
    lines = [f"# X-OPM RuleFit report -- module `{module}`", ""]
    lines += ["## Dataset", "",
              f"- Source: `dataset_processed/` (module `{module}`)",
              f"- Target power column: `{dims['target']}`",
              f"- Features: {dims['n_features']} "
              f"(pre-selection {dims['n_features_pre']})",
              f"- Rows: train {dims['n_train']} (+{dims['n_train_random_aug']} "
              f"random-stimulus augmentation used only for fitting), "
              f"val {dims['n_val']}, test {dims['n_test']}",
              f"- Train benchmarks: {', '.join(dims['train_benches'])}",
              f"- Test benchmarks: {', '.join(dims['test_benches'])}", ""]
    win = cfg.get("win_size", 1)
    win_note = ("per-cycle samples, no averaging" if win <= 1
                else f"each sample = mean of {win} consecutive cycles "
                     "(train/val/test counts are windows)")
    lines += ["## Model & parameters", "",
              "- Backend: XGBoost (gbtree + DART dropout) -> RuleFit "
              "(non-negative Lasso/ElasticNet)",
              "- Monotone constraint: +1 on every feature",
              f"- Window averaging: win_size={win} ({win_note})",
              f"- Penalty: {cfg['penalty']} (positive coefficients only)",
              f"- Gain threshold (rule prune): {cfg['gain_threshold']}",
              f"- num_boost_round: {cfg['num_boost_round']}",
              f"- HPO trials: {cfg['n_trials']}", "",
              "```json", json.dumps(cfg["xgb_params"], indent=2), "```", ""]
    lines += ["## Metrics", "",
              "| split | R2 | MAPE% | RMSE |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        m = metrics[s]
        lines.append(f"| {s} | {m['r2']:.4f} | {m['mape']:.3f} | {m['rmse']:.3e} |")
    n_rules = int((rule_df["type"] == "rule").sum())
    n_kept = int(((rule_df["type"] == "rule") & (~rule_df["dropped"])).sum())
    lines += ["", "## Rules", "",
              f"- Rules extracted: {n_rules} (kept {n_kept}, "
              f"dropped {n_rules - n_kept})",
              f"- Linear terms: {int((rule_df['type'] == 'linear').sum())}",
              "- See `rule.csv` (sorted by gain).", "",
              "## Figures", "",
              "- `residual_train_val_test.png`",
              "- `pred_vs_time_{train,val,test}.png`",
              "- `h_overall.png`, `h_pairwise.png`",
              "- `shap_beeswarm.png`, `shap_interaction_top_pairs.png`",
              "- `hpo_history.png`", ""]
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))


# ===========================================================================
# Per-module run
# ===========================================================================

def run_module(module: str, ts_dir: str, dataset_dir: str = DATASET_PROCESSED,
               n_trials: int = 30, gain_threshold: float = 0.0,
               penalty: str = "l1", val_fraction: float = 0.2, seed: int = 0,
               use_selection: bool = True, max_rules: int = DEFAULT_MAX_RULES,
               fit_rows: int = DEFAULT_FIT_ROWS, hpo_rows: int = DEFAULT_HPO_ROWS,
               rand_max_rows: int = DEFAULT_RAND_MAX_ROWS,
               win_size: int = 64) -> dict:
    out_dir = os.path.join(ts_dir, module)
    os.makedirs(out_dir, exist_ok=True)

    selected = fs.load_selected_columns(module, dataset_dir) if use_selection else None
    n_pre = len(_reference_columns(os.path.join(dataset_dir, "trainset", module), None))
    splits, feat = build_splits(module, dataset_dir, val_fraction, selected,
                                rand_max_rows=rand_max_rows, seed=seed,
                                win_size=win_size)
    monotone = "(" + ",".join(["1"] * len(feat)) + ")"

    best_params, nbr, study = run_hpo(splits, monotone, n_trials, seed, hpo_rows=hpo_rows)

    rf = XGBRuleFit(best_params, nbr, gain_threshold=gain_threshold, max_rules=max_rules,
                    fit_rows=fit_rows, penalty=penalty, random_state=seed)
    rf.fit(splits["train"].X, splits["train"].y, feature_names=feat,
           monotone_constraints=monotone)

    # The model is fit on the full train (incl. the random augmentation), but the
    # reported train metric / plots use only the real benchmarks, so train R2 is
    # comparable to val/test (random-stimulus power is not a benchmark of interest).
    tr = splits["train"]
    tr_mask = (tr.meta["bench"] != RAND_CASE).to_numpy()
    train_eval = subset_split(tr, tr_mask) if not tr_mask.all() else tr
    sp = {"train": train_eval, "val": splits["val"], "test": splits["test"]}

    preds = {s: (sp[s].y, predict_chunked(rf, sp[s].X)) for s in sp}
    metrics = {s: metric_block(*preds[s]) for s in preds}
    for s in ("train", "val", "test"):
        log.info("%s %-5s R2=%.4f MAPE=%.3f RMSE=%.3e", module, s,
                 metrics[s]["r2"], metrics[s]["mape"], metrics[s]["rmse"])

    # ---- artifacts ----
    rule_df = write_rule_csv(rf, os.path.join(out_dir, "rule.csv"))
    plot_residual_panels(preds, module, os.path.join(out_dir, "residual_train_val_test.png"))
    for s in ("train", "val", "test"):
        plot_pred_vs_time(sp[s], preds[s][1], f"{module} | {s}",
                          os.path.join(out_dir, f"pred_vs_time_{s}.png"),
                          win_size=win_size)

    # Friedman H (fork engine, on a row sample for tractability)
    Xs = splits["train"].X
    samp = Xs[np.random.RandomState(seed).choice(
        len(Xs), min(INTERP_SAMPLE, len(Xs)), replace=False)] if len(Xs) else Xs
    try:
        h_over = rf.interaction_strength(samp, top_k=25)
    except Exception as e:                                        # noqa: BLE001
        log.warning("interaction_strength failed: %s", e); h_over = None
    plot_h_bar(h_over, ["feature"], "Friedman overall H (per feature)",
               os.path.join(out_dir, "h_overall.png"))
    try:
        h_pair = rf.interaction_statistics(samp, order=2, top_k=12, max_tuples=5000)
    except Exception as e:                                        # noqa: BLE001
        log.warning("interaction_statistics failed: %s", e); h_pair = None
    plot_h_bar(h_pair, ["feature_1", "feature_2"], "Friedman pairwise H (top pairs)",
               os.path.join(out_dir, "h_pairwise.png"))

    # SHAP (native TreeSHAP)
    plot_shap_beeswarm(rf.booster_, Xs, feat,
                       os.path.join(out_dir, "shap_beeswarm.png"), seed=seed)
    plot_shap_interactions(rf.booster_, Xs, feat,
                           os.path.join(out_dir, "shap_interaction_top_pairs.png"),
                           seed=seed)

    # HPO history
    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        ax = plot_optimization_history(study)
        ax.figure.tight_layout(); ax.figure.savefig(
            os.path.join(out_dir, "hpo_history.png"), dpi=110)
        plt.close(ax.figure)
    except Exception as e:                                        # noqa: BLE001
        log.warning("hpo history plot failed: %s", e)

    # predictions for aq_core reconstruction (random already excluded from sp['train'])
    pred_frames = []
    for s in ("train", "val", "test"):
        m = sp[s].meta.copy()
        m["split"] = s
        m["y_true"] = preds[s][0]
        m["y_pred"] = preds[s][1]
        pred_frames.append(m)
    allp = pd.concat(pred_frames, ignore_index=True)
    allp.to_pickle(os.path.join(out_dir, "predictions.pkl.zst"), compression="zstd")

    dims = {"target": f"x_aq_core/Pc(x_aq_{module}_top)", "n_features": len(feat),
            "n_features_pre": n_pre, "n_train": len(train_eval.y),
            "n_train_random_aug": int((~tr_mask).sum()),
            "n_val": len(splits["val"].y), "n_test": len(splits["test"].y),
            "train_benches": sorted(sp["train"].slices),
            "test_benches": sorted(splits["test"].slices)}
    cfg = {"xgb_params": {k: v for k, v in best_params.items()
                          if k != "monotone_constraints"},
           "num_boost_round": nbr, "n_trials": n_trials, "penalty": penalty,
           "gain_threshold": gain_threshold, "monotone": "+1 all features",
           "win_size": win_size}
    write_report(module, out_dir, cfg, metrics, dims, rule_df)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"module": module, "metrics": metrics, "dims": dims,
                   "config": cfg}, fh, indent=2, default=str)
    return {"module": module, "metrics": metrics, "dims": dims}


# ===========================================================================
# aq_core reconstruction (Part C)
# ===========================================================================

def _load_true_aqcore(win_size: int = 1) -> pd.DataFrame:
    """Long frame (bench, time_ns, aqcore_true) of true whole-core power.

    When ``win_size>1`` each benchmark's per-cycle power is averaged into the same
    non-overlapping ``win_size``-cycle blocks (block first cycle as the ``time_ns``
    label) that the per-module predictions use, so the reconstruction merge on
    ``(bench, time_ns)`` stays aligned block-for-block."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW_PWR_DIR, "*_pwr.pkl.zst"))):
        bench = os.path.basename(p)[: -len("_pwr.pkl.zst")]
        df = pd.read_pickle(p)
        if AQCORE_COL not in df.columns:
            log.warning("aq_core: %s missing %s", bench, AQCORE_COL); continue
        s = df[AQCORE_COL]
        t = s.index.to_numpy(np.int64)
        v = s.to_numpy(np.float64)
        # window with the identical block scheme (empty feature stub -> only y/t used)
        _, v, t = _block_mean(np.empty((len(v), 0), np.float32), v, t, win_size)
        if len(v) == 0:
            continue
        rows.append(pd.DataFrame({"bench": bench, "time_ns": t, "aqcore_true": v}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def reconstruct_aqcore(ts_dir: str, modules=MODULES, win_size: int = 1) -> dict:
    """Sum per-module predictions -> predicted aq_core; score vs true Pc(x_aq_core).

    ``win_size`` must match the windowing the per-module predictions were trained
    with, so the raw truth is averaged into the same blocks before the merge."""
    per_mod = []
    for m in modules:
        p = os.path.join(ts_dir, m, "predictions.pkl.zst")
        if not os.path.exists(p):
            log.warning("reconstruction: %s has no predictions, skipping", m); continue
        d = pd.read_pickle(p)[["bench", "time_ns", "split", "y_pred"]].copy()
        d = d.rename(columns={"y_pred": m})
        per_mod.append(d.set_index(["bench", "time_ns", "split"]))
    if not per_mod:
        log.warning("reconstruction: no module predictions found"); return {}
    wide = pd.concat(per_mod, axis=1)
    present = [m for m in modules if m in wide.columns]
    n_missing = wide[present].isna().sum().sum()
    if n_missing:
        log.warning("reconstruction: %d missing module/cycle cells filled 0", int(n_missing))
    wide["pred_sum"] = wide[present].fillna(0.0).sum(axis=1)
    wide = wide.reset_index()

    truth = _load_true_aqcore(win_size)
    merged = wide.merge(truth, on=["bench", "time_ns"], how="inner")
    if not len(merged):
        log.warning("reconstruction: no overlap with true aq_core power"); return {}

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=False)
    for ax, split in zip(axes[0], ("train", "test")):
        sub = merged[merged["split"].isin(
            ("train", "val") if split == "train" else ("test",))]
        if not len(sub):
            ax.set_title(f"{split} (no data)"); continue
        y = sub["aqcore_true"].to_numpy() * POWER_SCALE
        yh = sub["pred_sum"].to_numpy() * POWER_SCALE
        out[split] = metric_block(sub["aqcore_true"].to_numpy(),
                                  sub["pred_sum"].to_numpy())
        idx = _subsample(len(y))
        ax.scatter(yh[idx], (y - yh)[idx], s=2, alpha=0.3, rasterized=True)
        ax.axhline(0, color="r", lw=0.8, ls="--")
        ax.set_xlabel(f"predicted aq_core ({POWER_UNIT})")
        ax.set_ylabel(f"residual ({POWER_UNIT})")
        ax.set_title(f"{split}: R2={out[split]['r2']:.3f} "
                     f"MAPE={out[split]['mape']:.2f}% n={len(y)}")
    fig.suptitle("aq_core reconstruction (sum of module predictions)")
    fig.tight_layout()
    fig.savefig(os.path.join(ts_dir, "aq_core_residual_train_test.png"), dpi=120)
    plt.close(fig)

    merged[["bench", "time_ns", "split", "pred_sum", "aqcore_true"]].to_csv(
        os.path.join(ts_dir, "aq_core_reconstruction.csv"), index=False)
    log.info("aq_core reconstruction: %s", {k: round(v["r2"], 4) for k, v in out.items()})
    return out


def write_top_report(ts_dir: str, results: list[dict], recon: dict) -> None:
    lines = ["# X-OPM RuleFit -- run summary", "",
             f"Generated: {os.path.basename(ts_dir)}", "",
             "## Per-module metrics", "",
             "| module | features | train R2 | val R2 | test R2 | test MAPE% | test RMSE |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['module']} | {r['dims']['n_features']} | "
            f"{m['train']['r2']:.4f} | {m['val']['r2']:.4f} | {m['test']['r2']:.4f} | "
            f"{m['test']['mape']:.3f} | {m['test']['rmse']:.3e} |")
    lines += ["", "## aq_core reconstruction (sum of modules vs true Pc(x_aq_core))", ""]
    if recon:
        lines += ["| split | R2 | MAPE% | RMSE |", "|---|---|---|---|"]
        for s in ("train", "test"):
            if s in recon:
                m = recon[s]
                lines.append(f"| {s} | {m['r2']:.4f} | {m['mape']:.3f} | {m['rmse']:.3e} |")
        lines += ["", "See `aq_core_residual_train_test.png` and "
                  "`aq_core_reconstruction.csv`."]
    else:
        lines.append("_reconstruction unavailable (missing module predictions)._")
    with open(os.path.join(ts_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))


# ===========================================================================
# Orchestration / CLI
# ===========================================================================

def _make_ts_dir(outdir: str | None, clean: bool = True) -> str:
    if outdir:
        ts_dir = outdir
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        ts_dir = os.path.join(ANALYSIS_ROOT, ts)
    if clean and os.path.exists(ts_dir):
        shutil.rmtree(ts_dir)               # spec: replace an existing folder
    os.makedirs(ts_dir, exist_ok=True)
    return ts_dir


def _results_from_disk(ts_dir: str, modules=MODULES) -> list[dict]:
    """Reload per-module ``metrics.json`` (for the parallel reconstruct-only pass)."""
    out = []
    for m in modules:
        p = os.path.join(ts_dir, m, "metrics.json")
        if os.path.exists(p):
            with open(p) as fh:
                out.append(json.load(fh))
    return out


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(prog="xopm-model-regression")
    ap.add_argument("--module", help="one module or a comma-separated list "
                    f"(from {', '.join(MODULES)})")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--gain-threshold", type=float, default=0.0)
    ap.add_argument("--penalty", choices=["l1", "elasticnet"], default="l1")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-rules", type=int, default=DEFAULT_MAX_RULES)
    ap.add_argument("--fit-rows", type=int, default=DEFAULT_FIT_ROWS,
                    help="row subsample for the RuleFit linear (Lasso) stage")
    ap.add_argument("--hpo-rows", type=int, default=DEFAULT_HPO_ROWS,
                    help="row subsample for the Optuna booster search")
    ap.add_argument("--rand-max-rows", type=int, default=DEFAULT_RAND_MAX_ROWS,
                    help="cap on the random-stimulus training case")
    ap.add_argument("--win-size", type=int, default=64,
                    help="average this many consecutive per-cycle samples into one "
                         "sample (non-overlapping block mean) before training; "
                         "1 = per-cycle (no averaging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", default=DATASET_PROCESSED)
    ap.add_argument("--outdir", default=None,
                    help="override analysis/x-opm/<ts> (created fresh)")
    ap.add_argument("--no-clean", action="store_true",
                    help="do not wipe --outdir (for parallel writers sharing one dir)")
    ap.add_argument("--no-selection", action="store_true",
                    help="ignore feature_selection output, use all features")
    ap.add_argument("--no-reconstruct", action="store_true")
    ap.add_argument("--reconstruct-only", action="store_true",
                    help="skip training; aggregate metrics.json in --outdir into the "
                         "aq_core reconstruction + top-level report (parallel driver)")
    args = ap.parse_args(argv)

    if args.reconstruct_only:
        if not args.outdir:
            ap.error("--reconstruct-only requires --outdir")
        results = _results_from_disk(args.outdir)
        win = next((r["config"]["win_size"] for r in results
                    if "win_size" in r.get("config", {})), args.win_size)
        recon = reconstruct_aqcore(args.outdir, [r["module"] for r in results],
                                   win_size=win)
        write_top_report(args.outdir, results, recon)
        log.info("reconstruct-only done -> %s", args.outdir)
        return

    if args.all:
        targets = list(MODULES)
    elif args.module:
        targets = [m.strip() for m in args.module.split(",") if m.strip()]
        bad = [m for m in targets if m not in MODULES]
        if bad:
            ap.error(f"unknown module(s): {bad}; choose from {', '.join(MODULES)}")
    else:
        ap.error("pass --module <name[,name...]> or --all")

    ts_dir = _make_ts_dir(args.outdir, clean=not args.no_clean)
    log.info("output dir: %s", ts_dir)
    results = []
    for m in targets:
        try:
            results.append(run_module(
                m, ts_dir, dataset_dir=args.dataset_dir, n_trials=args.n_trials,
                gain_threshold=args.gain_threshold, penalty=args.penalty,
                val_fraction=args.val_fraction, seed=args.seed,
                use_selection=not args.no_selection, max_rules=args.max_rules,
                fit_rows=args.fit_rows, hpo_rows=args.hpo_rows,
                rand_max_rows=args.rand_max_rows, win_size=args.win_size))
        except Exception as e:                                    # noqa: BLE001
            log.exception("module %s failed: %s", m, e)

    if not args.no_reconstruct:
        recon = reconstruct_aqcore(ts_dir, [r["module"] for r in results],
                                   win_size=args.win_size) if results else {}
        write_top_report(ts_dir, results, recon)
    log.info("done -> %s", ts_dir)


if __name__ == "__main__":
    main()
