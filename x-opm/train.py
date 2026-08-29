#!/usr/bin/env python
"""x-opm steps 6-8: train ONE power model on the typed dataset, restricted to a
chosen set of feature types (A/B/C/D).

Two backends:
  * ``cobit``   -- reuse cobit's XGBoost + Optuna modelling code (cobit/model.py)
                   on the x-opm typed features. No monotonicity constraint (matches
                   doc/cobit.pdf); base_score=0 (x=0 -> y=0). "Coefficients" for a
                   tree model = per-feature gain importance.
  * ``rulefit`` -- RuleFit (third_party/rulefit) with an ElasticNetCV regularizer and
                   positive coefficients; a fitted intercept captures the leakage floor.
                   A second GBDT head is additionally trained on the transformed
                   features (selected rules + linear terms) and reported under
                   <outdir>/gbdt/.

Outputs (into --outdir): metrics.json, coefficient.csv, run_config.json, and
power_{train,val,test}.png (true vs predicted power, mW) + scatter_{...}.png.

Usage:
    ~/anaconda3/bin/python x-opm/train.py --backend cobit|rulefit \
        --types AB|ABC|ABCD --outdir out/x-opm/results/<typeset>/<backend>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "out" / "x-opm" / "dataset"
REPORTS = REPO / "out" / "x-opm" / "reports"
TARGET_NAME = "x_aq_core/Pc(x_aq_cp0_top)"
PLOT_MAX_POINTS = 50_000  # subsample points for scatter plots (speed / file size)
POWER_UNIT = "mW"         # y-axis unit for the power-vs-time trace
POWER_SCALE = 1000.0      # dataset target is in Watts; multiply by this to get mW


# ----------------------------------------------------------------- metrics --
# (mirror cobit/evaluate.py: masked MAPE + R2)
def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def mape_percent(y: np.ndarray, yhat: np.ndarray, eps_frac: float = 1e-3) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    eps = eps_frac * float(np.median(np.abs(y))) if y.size else 0.0
    mask = np.abs(y) > eps
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def metric_block(y, yhat) -> dict:
    return {"r2": r2_score(y, yhat), "mape": mape_percent(y, yhat), "rmse": rmse(y, yhat)}


# -------------------------------------------------------------- data load --
def _benches() -> tuple[list[str], str]:
    summary = json.loads((REPORTS / "summary.json").read_text())
    return summary["train_benchmarks"], summary["test_benchmarks"][0]


def _test_benches() -> list[str]:
    """Every held-out test benchmark. All are POOLED into one test set."""
    summary = json.loads((REPORTS / "summary.json").read_text())
    return list(summary["test_benchmarks"])


def _cd_feature_names(types: list[str]) -> set[str]:
    """Names of every type-C / type-D feature present in ``types``.

    Read straight from the per-type pickles (the same source ``main`` uses for the
    by-type counts), so membership is the ground-truth dataset partition rather
    than a re-classification.
    """
    test_bench = _benches()[1]
    names: set[str] = set()
    for t in types:
        if t in ("C", "D"):
            cols = pd.read_pickle(DATASET / "testset" / test_bench / f"type{t}.pkl").columns
            names.update(cols)
    return names


def _load_case(split: str, case: str, types: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    d = DATASET / split / case
    parts = [pd.read_pickle(d / f"type{t}.pkl") for t in types]
    X = pd.concat(parts, axis=1)
    y = pd.read_pickle(d / "target.pkl")
    y = y.reindex(X.index)  # align (already aligned by construction)
    return X, y


def _block_mean(X: np.ndarray, y: np.ndarray, window: int):
    """Non-overlapping N-cycle block means of features and target.

    Truncates the trailing partial block. window<=1 is a no-op.
    """
    if window <= 1:
        return X, y
    m = (len(X) // window) * window
    if m == 0:  # benchmark shorter than one window -> no full block
        return X[:0], y[:0]
    Xw = X[:m].reshape(-1, window, X.shape[1]).mean(axis=1).astype(np.float32)
    yw = y[:m].reshape(-1, window).mean(axis=1).astype(np.float64)
    return Xw, yw


def _val_count(n: int, frac: float) -> int:
    if frac == 0.0:
        return 0
    n_val = int(round(n * frac))
    return min(max(n_val, 1), n - 1) if n >= 2 else 0


def build_dataset(types: list[str], val_fraction: float, window: int = 1):
    """Return (Xtr, ytr, Xval, yval, Xte, yte, feature_names, slices) as numpy.

    If window > 1, each benchmark is averaged into non-overlapping N-cycle blocks
    (features and target) BEFORE the tail validation split, so val is whole windows.
    """
    train_benches, _ = _benches()
    Xtr_parts, ytr_parts, Xval_parts, yval_parts = [], [], [], []
    tr_slices, val_slices = {}, {}
    ctr = cval = 0
    feature_names = None
    for b in train_benches:
        X, y = _load_case("trainset", b, types)
        if feature_names is None:
            feature_names = list(X.columns)
        else:
            X = X[feature_names]
        Xb, yb = _block_mean(X.to_numpy(np.float32), y.to_numpy(np.float64), window)
        n = len(Xb)
        n_val = _val_count(n, val_fraction)
        n_tr = n - n_val
        Xtr_parts.append(Xb[:n_tr]); ytr_parts.append(yb[:n_tr])
        Xval_parts.append(Xb[n_tr:]); yval_parts.append(yb[n_tr:])
        tr_slices[b] = (ctr, ctr + n_tr); ctr += n_tr
        val_slices[b] = (cval, cval + n_val); cval += n_val

    Xtr = np.concatenate(Xtr_parts); ytr = np.concatenate(ytr_parts)
    Xval = np.concatenate(Xval_parts); yval = np.concatenate(yval_parts)

    # POOL every held-out test benchmark into one test set. Window each benchmark
    # SEPARATELY (so a trailing partial block of one bench is never averaged with the
    # head of the next), then concatenate. te_slices records each bench's span so the
    # power-trace plot can still draw benchmark boundaries.
    Xte_parts, yte_parts, te_slices, cte = [], [], {}, 0
    for b in _test_benches():
        Xb_df, yb_s = _load_case("testset", b, types)
        Xb, yb = _block_mean(Xb_df[feature_names].to_numpy(np.float32),
                             yb_s.to_numpy(np.float64), window)
        Xte_parts.append(Xb); yte_parts.append(yb)
        te_slices[b] = (cte, cte + len(Xb)); cte += len(Xb)
    Xte = np.concatenate(Xte_parts); yte = np.concatenate(yte_parts)

    slices = {"train": tr_slices, "val": val_slices, "test": te_slices}
    return Xtr, ytr, Xval, yval, Xte, yte, feature_names, slices


# ----------------------------------------------------------------- plots --
def _subsample(n: int, seed: int = 0) -> np.ndarray:
    if n <= PLOT_MAX_POINTS:
        return np.arange(n)
    idx = np.random.RandomState(seed).choice(n, PLOT_MAX_POINTS, replace=False)
    idx.sort()
    return idx


def plot_power_trace(y, yhat, slices, split, backend, types, path: Path, window=1):
    """Power vs time: true and predicted power (mW) as two overlaid curves.

    x-axis is the cycle index (or N-cycle window index when window>1) over the
    concatenated benchmarks; benchmark boundaries are marked. The dataset target
    is in Watts, so both traces are scaled to mW (``POWER_SCALE``).
    """
    y = np.asarray(y, float) * POWER_SCALE
    yhat = np.asarray(yhat, float) * POWER_SCALE
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, color="#1f77b4", lw=0.8, alpha=0.9, label="true",
            rasterized=True)
    ax.plot(x, yhat, color="#ff7f0e", lw=0.8, alpha=0.9, label="predicted",
            rasterized=True)
    ytop = ax.get_ylim()[1]
    for b, (lo, hi) in slices.items():
        ax.axvline(hi, color="grey", lw=0.4, ls=":")
        ax.text(0.5 * (lo + hi), ytop, b, fontsize=6,
                ha="center", va="top", rotation=90, color="grey")
    unit = f"{window}-cycle window index" if window > 1 else "cycle index"
    ax.set_xlabel(f"{unit} (concatenated benchmarks)")
    ax.set_ylabel(f"power ({POWER_UNIT})")
    ax.set_title(f"Power vs time [{backend} | types {types} | {split}]")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_scatter(y, yhat, split, backend, types, path: Path):
    y = np.asarray(y); yhat = np.asarray(yhat)
    idx = _subsample(len(y))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y[idx], yhat[idx], s=2, alpha=0.3, rasterized=True)
    lo = float(min(y.min(), yhat.min())); hi = float(max(y.max(), yhat.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
    ax.set_xlabel("true power"); ax.set_ylabel("predicted power")
    ax.set_title(f"Pred vs true [{backend} | types {types} | {split}]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def make_plots(outdir, backend, types_str, splits, window=1):
    for split, (y, yhat, slc) in splits.items():
        plot_power_trace(y, yhat, slc, split, backend, types_str,
                         outdir / f"power_{split}.png", window=window)
        plot_scatter(y, yhat, split, backend, types_str,
                     outdir / f"scatter_{split}.png")


# --------------------------------------------------------------- backends --
def _load_cobit_model():
    spec = importlib.util.spec_from_file_location(
        "cobit_model_standalone", REPO / "cobit" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # cobit/model.py has no relative imports
    return mod


def _sanitize_names(names):
    """XGBoost forbids '[', ']', '<' in feature names; map to safe unique names."""
    safe, used, back = [], set(), {}
    for i, n in enumerate(names):
        s = n.replace("[", "_").replace("]", "").replace("<", "_")
        if s in used:
            s = f"{s}__{i}"
        used.add(s)
        safe.append(s)
        back[s] = n
    return safe, back


def run_cobit(data, feature_names, args, outdir):
    import xgboost as xgb
    import optuna
    cm = _load_cobit_model()
    Xtr, ytr, Xval, yval, Xte, yte = data
    nthr = args.nthread
    safe_names, back = _sanitize_names(feature_names)
    # No monotonicity constraint: cobit.pdf models plain XGBoost boosting with no
    # per-feature monotone constraint, so we reproduce it exactly.
    # Scale the target by a positive constant (train std) so cobit's SPACE
    # regularization ranges (gamma/lambda/min_child_weight, tuned for a larger-
    # magnitude target) stay meaningful. Scaling by a positive constant keeps
    # x=0 -> y=0, so base_score=0 remains valid. Predictions are inverted before
    # any metric/plot, so all reported numbers are in original power units.
    yscale = float(np.std(ytr)) or 1.0
    dtrain = xgb.DMatrix(Xtr, label=ytr / yscale, feature_names=safe_names, nthread=nthr or None)
    dval = xgb.DMatrix(Xval, label=yval / yscale, feature_names=safe_names, nthread=nthr or None)
    dtest = xgb.DMatrix(Xte, feature_names=safe_names, nthread=nthr or None)

    def objective(trial):
        params = cm.suggest_params(trial)
        params["base_score"] = 0.0
        booster, _, _ = cm.train_boosting(params, dtrain, dval, args.num_rounds,
                                           seed=args.seed, nthread=nthr)
        return rmse(yval, booster.predict(dval) * yscale)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["base_score"] = 0.0
    booster, rounds, _ = cm.train_boosting(best, dtrain, dval, args.num_rounds,
                                            seed=args.seed, nthread=nthr)

    preds = {"train": booster.predict(dtrain) * yscale,
             "val": booster.predict(dval) * yscale,
             "test": booster.predict(dtest) * yscale}

    # coefficients analog = per-feature gain importance (keyed by sanitized name)
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    rows = [{"feature": back[s], "gain": gain.get(s, 0.0),
             "weight": weight.get(s, 0.0), "cover": cover.get(s, 0.0)}
            for s in safe_names]
    coef_df = pd.DataFrame(rows).sort_values("gain", ascending=False)
    coef_df.to_csv(outdir / "coefficient.csv", index=False)

    extra = {"best_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in study.best_params.items()},
             "num_rounds": int(rounds), "n_trials": args.n_trials,
             "base_score": 0.0,
             "target_train_scale": yscale,
             "n_nonzero_gain_features": int((coef_df["gain"] > 0).sum())}
    booster.save_model(str(outdir / "model.xgb.json"))
    return preds, extra


def run_rulefit(data, feature_names, args, outdir):
    from rulefit import RuleFitRegressor
    Xtr, ytr, Xval, yval, Xte, yte = data

    # subsample train rows for the (slow) rule-generation fit; predict on all rows.
    if len(Xtr) > args.rulefit_max_rows:
        sel = np.random.RandomState(args.seed).choice(
            len(Xtr), args.rulefit_max_rows, replace=False)
        sel.sort()
        Xfit, yfit = Xtr[sel], ytr[sel]
    else:
        Xfit, yfit = Xtr, ytr

    # Monotonicity: sklearn's GradientBoostingRegressor (the generator rulefit
    # accepts) has no monotone-constraint support, so we enforce the achievable
    # guarantee via rulefit's positive-coefficient constraint (default) -- every
    # rule/linear term contributes non-negatively, so increasing a feature never
    # lowers predicted power.
    # Intercept: we FIT it (fit_intercept=True) rather than force 0. The windowed
    # mean-power target has a non-zero static/leakage floor (min ~2.5e-4 != 0), so
    # the doc's "x=0 -> y=0" assumption is physically wrong here; forcing intercept
    # =0 makes positive-only lasso underfit to a NEGATIVE R2. The fitted intercept
    # captures the leakage baseline; positive coefficients model activity above it.
    # Optional interaction constraint: forbid rules that combine two or more
    # type-C/type-D features. A/B features interact freely and each may pair with a
    # single C/D feature; C/D never co-occur with another C/D in a rule. C/D still
    # enter the model as standalone linear terms, so their first-order effect stays.
    rule_filter = None
    if getattr(args, "rulefit_cd_no_interact", False):
        cd_names = _cd_feature_names(list(args.types.upper()))
        is_cd = np.array([n in cd_names for n in feature_names], dtype=bool)

        def rule_filter(rule, _is_cd=is_cd):
            return sum(1 for fi in rule.defining_variables() if _is_cd[fi]) <= 1

    # Final regularizer: ElasticNetCV (penalty="elasticnet") by default; pass
    # penalty="l1" to fall back to the LassoCV path. l1_ratio only applies to enet.
    rf = RuleFitRegressor(tree_size=4, max_rules=args.max_rules,
                          exp_rand_tree_size=False, random_state=args.seed,
                          fit_intercept=True, penalty=args.rulefit_penalty,
                          l1_ratio=args.rulefit_l1_ratio)
    rf.fit(Xfit, yfit, feature_names=feature_names, rule_filter=rule_filter)

    preds = {"train": rf.predict(Xtr), "val": rf.predict(Xval), "test": rf.predict(Xte)}

    rules = rf.get_rules(exclude_zero_coef=True).copy()
    rules = rules.reindex(rules["coef"].abs().sort_values(ascending=False).index)
    rules.to_csv(outdir / "coefficient.csv", index=False)

    extra = {"intercept": float(getattr(rf, "intercept_", 0.0)),
             "n_terms_kept": int(len(rules)),
             "n_rule_terms": int((rules["type"] == "rule").sum()),
             "n_linear_terms": int((rules["type"] == "linear").sum()),
             "monotonicity": "positive-coefficient constraint (all coefs >= 0)",
             "fit_intercept": True,
             "intercept_note": "fitted (captures static/leakage power floor)",
             "fit_rows": int(len(Xfit)), "max_rules": args.max_rules,
             "penalty": args.rulefit_penalty,
             "l1_ratio": (float(getattr(rf.lscv, "l1_ratio_", args.rulefit_l1_ratio))
                          if args.rulefit_penalty == "elasticnet" else None),
             "alpha": float(getattr(rf.lscv, "alpha_", float("nan"))),
             "interaction_constraint": ("<=1 type-C/D feature per rule"
                                        if rule_filter is not None else "none"),
             "min_coef": float(rules["coef"].min()) if len(rules) else None}

    # Parallel GBDT head: fit a boosting model on rulefit's transformed features
    # (the [linear terms | rule indicators] design matrix the regularizer saw).
    gbdt_preds = _rulefit_gbdt_head(rf, data, args, outdir, extra) if args.rulefit_gbdt \
        else None
    return preds, extra, gbdt_preds


def _rulefit_gbdt_head(rf, data, args, outdir, extra):
    """Train a GBDT on rulefit's transformed features and record it in ``extra``.

    Feature set (``--rulefit-gbdt-features``): all raw linear terms plus either the
    enet-``selected`` (nonzero-coef) rules or ``all`` generated rules. Reuses cobit's
    XGBoost + Optuna machinery (no monotone constraint). Predictions (original power
    units) are returned; ``coefficient.csv`` + ``model.xgb.json`` are written into
    ``<outdir>/gbdt/`` and a scalar summary is stored under ``extra["gbdt"]``. The
    train/val/test metrics + plots for this head are produced by ``main`` (it owns the
    per-benchmark ``slices`` needed for the trace plots).
    """
    import xgboost as xgb
    import optuna
    cm = _load_cobit_model()
    Xtr, ytr, Xval, yval, Xte, yte = data
    nthr = args.nthread

    # Regenerate the exact [linear | rules] matrix the regularizer saw (rule_coefs=None
    # keeps every rule column unzeroed). rf was fit on a subsample; transform all rows.
    Xc_tr, Xc_val, Xc_te = (rf._design_matrix(X) for X in (Xtr, Xval, Xte))
    n_lin, n_rules = int(rf.n_linear_terms_), int(rf.n_rules_)
    col_names = list(rf.feature_names) + [str(r) for r in rf.rule_ensemble.rules]

    if args.rulefit_gbdt_features == "selected":
        rule_keep = np.nonzero(np.asarray(rf.coef_[n_lin:]) != 0)[0]
    else:
        rule_keep = np.arange(n_rules)
    sel_idx = np.concatenate([np.arange(n_lin), n_lin + rule_keep]).astype(int)
    if sel_idx.size == 0:
        return None  # degenerate: no linear terms and no (selected) rules
    sel_names = [col_names[i] for i in sel_idx]
    Xc_tr, Xc_val, Xc_te = Xc_tr[:, sel_idx], Xc_val[:, sel_idx], Xc_te[:, sel_idx]

    safe_g, back_g = _sanitize_names(sel_names)
    g_yscale = float(np.std(ytr)) or 1.0
    dtr = xgb.DMatrix(Xc_tr, label=ytr / g_yscale, feature_names=safe_g, nthread=nthr or None)
    dva = xgb.DMatrix(Xc_val, label=yval / g_yscale, feature_names=safe_g, nthread=nthr or None)
    dte = xgb.DMatrix(Xc_te, feature_names=safe_g, nthread=nthr or None)

    # No monotone constraint and no forced base_score (a zero rule/linear vector need
    # not map to zero power).
    def objective(trial):
        params = cm.suggest_params(trial)
        booster, _, _ = cm.train_boosting(params, dtr, dva, args.rulefit_gbdt_rounds,
                                           seed=args.seed, nthread=nthr)
        return rmse(yval, booster.predict(dva) * g_yscale)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.rulefit_gbdt_trials, show_progress_bar=False)
    booster, rounds, _ = cm.train_boosting(dict(study.best_params), dtr, dva,
                                           args.rulefit_gbdt_rounds, seed=args.seed,
                                           nthread=nthr)

    gbdt_preds = {"train": booster.predict(dtr) * g_yscale,
                  "val": booster.predict(dva) * g_yscale,
                  "test": booster.predict(dte) * g_yscale}

    gdir = outdir / "gbdt"; gdir.mkdir(parents=True, exist_ok=True)
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    grows = [{"feature": back_g[s], "gain": gain.get(s, 0.0),
              "weight": weight.get(s, 0.0), "cover": cover.get(s, 0.0)}
             for s in safe_g]
    gcoef = pd.DataFrame(grows).sort_values("gain", ascending=False)
    gcoef.to_csv(gdir / "coefficient.csv", index=False)
    booster.save_model(str(gdir / "model.xgb.json"))

    extra["gbdt"] = {"features": args.rulefit_gbdt_features,
                     "n_features": int(sel_idx.size),
                     "n_linear_terms": n_lin, "n_rules_total": n_rules,
                     "n_selected_rules": int(rule_keep.size),
                     "num_rounds": int(rounds), "n_trials": args.rulefit_gbdt_trials,
                     "best_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                                     for k, v in study.best_params.items()},
                     "target_train_scale": g_yscale,
                     "n_nonzero_gain_features": int((gcoef["gain"] > 0).sum()),
                     "outdir": "gbdt"}
    return gbdt_preds


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cobit", "rulefit"])
    ap.add_argument("--types", required=True, help="subset of ABCD, e.g. AB, ABC, ABCD")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=1,
                    help="average features+target over N-cycle non-overlapping blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nthread", type=int, default=0, help="0 = xgboost default")
    # module selection (defaults reproduce the cp0 dataset baked into the constants above)
    ap.add_argument("--dataset-dir", default=None,
                    help="override the dataset dir (holds trainset/ + testset/); "
                         "e.g. out/x-opm/<module>/dataset")
    ap.add_argument("--reports-dir", default=None,
                    help="override the reports dir (holds summary.json); "
                         "e.g. out/x-opm/<module>/reports")
    ap.add_argument("--target", default=None,
                    help="target power-column name for the metrics.json label "
                         "(the actual target is baked into target.pkl by the build)")
    # cobit HPO
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--num-rounds", type=int, default=300)
    # rulefit
    ap.add_argument("--rulefit-max-rows", type=int, default=200_000)
    ap.add_argument("--max-rules", type=int, default=500)
    ap.add_argument("--rulefit-cd-no-interact", action="store_true",
                    help="interaction constraint: each rule may contain at most one "
                         "type-C/type-D feature (A/B interact freely and each with a "
                         "single C/D feature; no C-C/C-D/D-D). C/D features still "
                         "appear as standalone linear terms.")
    ap.add_argument("--rulefit-penalty", choices=["l1", "elasticnet"],
                    default="elasticnet",
                    help="rulefit final regularizer: 'l1' (LassoCV) or 'elasticnet' "
                         "(ElasticNetCV, default)")
    ap.add_argument("--rulefit-l1-ratio", type=float, default=0.5,
                    help="ElasticNetCV l1_ratio (only used when penalty=elasticnet)")
    ap.add_argument("--rulefit-gbdt", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="also train a GBDT on rulefit's transformed features "
                         "(--no-rulefit-gbdt to disable)")
    ap.add_argument("--rulefit-gbdt-features", choices=["selected", "all"],
                    default="selected",
                    help="GBDT feature set: enet-'selected' (nonzero-coef) rules + "
                         "linear terms, or 'all' rules + linear terms")
    ap.add_argument("--rulefit-gbdt-trials", type=int, default=20,
                    help="Optuna trials for the rulefit GBDT head")
    ap.add_argument("--rulefit-gbdt-rounds", type=int, default=300,
                    help="max boosting rounds for the rulefit GBDT head")
    args = ap.parse_args()

    global DATASET, REPORTS, TARGET_NAME
    if args.dataset_dir:
        DATASET = Path(args.dataset_dir)
    if args.reports_dir:
        REPORTS = Path(args.reports_dir)
    if args.target:
        TARGET_NAME = args.target

    types = list(args.types.upper())
    assert all(t in "ABCD" for t in types), args.types
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{args.backend}|{args.types}] loading dataset ...", flush=True)

    Xtr, ytr, Xval, yval, Xte, yte, feature_names, slices = build_dataset(
        types, args.val_fraction, args.window)
    print(f"[{args.backend}|{args.types}] window={args.window} X_train={Xtr.shape} "
          f"X_val={Xval.shape} X_test={Xte.shape} n_features={len(feature_names)}",
          flush=True)

    data = (Xtr, ytr, Xval, yval, Xte, yte)
    if args.backend == "cobit":
        preds, extra = run_cobit(data, feature_names, args, outdir)
        gbdt_preds = None
    else:
        preds, extra, gbdt_preds = run_rulefit(data, feature_names, args, outdir)

    ys = {"train": ytr, "val": yval, "test": yte}
    metrics = {split: metric_block(ys[split], preds[split]) for split in ys}
    make_plots(outdir, args.backend, args.types,
               {s: (ys[s], preds[s], slices[s]) for s in ys}, window=args.window)

    # GBDT head (rulefit only): its own metrics + plots in <outdir>/gbdt/. Metrics are
    # also folded into extra["gbdt"] so the primary metrics.json carries both models.
    gbdt_dir = None
    if gbdt_preds is not None:
        gbdt_dir = outdir / "gbdt"; gbdt_dir.mkdir(parents=True, exist_ok=True)
        extra.setdefault("gbdt", {})["metrics"] = {
            split: metric_block(ys[split], gbdt_preds[split]) for split in ys}
        make_plots(gbdt_dir, "rulefit-gbdt", args.types,
                   {s: (ys[s], gbdt_preds[s], slices[s]) for s in ys}, window=args.window)

    # feature counts by type
    by_type = {}
    for t in types:
        cols = pd.read_pickle(DATASET / "testset" / _benches()[1] / f"type{t}.pkl").shape[1]
        by_type[t] = int(cols)

    out = {"backend": args.backend, "types": args.types, "window": args.window,
           "dataset": {"name": f"c906_db_net_1cyc_20260729 ({TARGET_NAME}, x-opm typed)",
                       "target": TARGET_NAME,
                       "window_cycles": args.window,
                       "n_features": len(feature_names),
                       "n_features_by_type": by_type,
                       "n_train_rows": int(len(ytr)), "n_val_rows": int(len(yval)),
                       "n_test_rows": int(len(yte)),
                       "row_unit": (f"{args.window}-cycle window" if args.window > 1
                                    else "cycle"),
                       "test_benchmarks": _test_benches()},
           "metrics": metrics, "extra": extra,
           "elapsed_sec": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(out, indent=2))
    (outdir / "run_config.json").write_text(json.dumps({
        "argv": vars(args), "feature_names_head": feature_names[:20]}, indent=2))

    if gbdt_dir is not None:
        gout = dict(out)
        gout["backend"] = "rulefit-gbdt"
        gout["metrics"] = extra["gbdt"]["metrics"]
        gout["extra"] = extra["gbdt"]
        (gbdt_dir / "metrics.json").write_text(json.dumps(gout, indent=2))
        gm = gout["metrics"]
        print(f"[rulefit-gbdt|{args.types}] head  "
              f"train_R2={gm['train']['r2']:.4f} val_R2={gm['val']['r2']:.4f} "
              f"test_R2={gm['test']['r2']:.4f}", flush=True)
    print(f"[{args.backend}|{args.types}] DONE in {out['elapsed_sec']}s  "
          f"train_R2={metrics['train']['r2']:.4f} val_R2={metrics['val']['r2']:.4f} "
          f"test_R2={metrics['test']['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
