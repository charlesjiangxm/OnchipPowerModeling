"""
Contains three stages: build_db, feature_select, fit. Run directly (no package/-m needed):
1. Build DB only (config-agnostic, but need one; cobit.yaml and nn.yaml have identical build sections):
    python src/binary_fit/run.py --build_db --config src/binary_fit/configs/cobit.yaml

2. Feature selection:
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/cobit.yaml --outdir analysis/cobit/2026-09-03-17-100proxy;
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/nn.yaml --outdir analysis/nn/2026-09-03-17-100proxy;
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/ridge.yaml --outdir analysis/ridge/2026-09-03-17-100proxy;
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/rulefit.yaml --outdir analysis/rulefit/2026-09-03-17-100proxy;

3. Regression (fit, tree, nn, ridge, rulefit) on the selected proxies:
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/cobit.yaml --outdir analysis/cobit/2026-09-03-17-100proxy --model tree;
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/nn.yaml --outdir analysis/nn/2026-09-03-17-100proxy --model nn;
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/ridge.yaml --outdir analysis/ridge/2026-09-03-17-100proxy --model ridge;
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/rulefit.yaml --outdir analysis/rulefit/2026-09-03-17-100proxy --model rulefit

Options:
1. --no-hpo to skip HPO
2. --nthread N to cap threads
3. --window-size N to average N cycles into one row (default data.window_size=32);
   use the SAME value for --feature_select and --fit, a mismatch is only a warning

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binary_fit import build_db, data, hpo, models
from binary_fit.config import Config
from binary_fit.evaluate import evaluation_report
from binary_fit.plots import (
    plot_interaction_strength,
    plot_pred_vs_time,
    plot_q_sweep,
    plot_residual_panels,
)
from binary_fit.utils import (
    load_json,
    load_proxies_csv,
    log,
    save_coefficients_csv,
    save_json,
    save_pickle_zst,
    save_proxies_csv,
    save_ridge_coefficients_csv,
    save_rulefit_terms_csv,
    setup_logging,
)


# The Stage-2 regressors, in report order. Single source of truth for the
# --model choices, the "both" expansion and the cmd_fit pre-flight check, so a
# new backend's NAME is registered in exactly one place. (Its _run_one dispatch
# arm is still a separate, deliberate edit -- see the comment there.)
MODEL_KINDS = ("tree", "nn", "ridge", "rulefit")


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


# --------------------------------------------------------------------------- #
# Stage 0: build_db
# --------------------------------------------------------------------------- #
def cmd_build_db(cfg: Config) -> int:
    written = build_db.build(cfg)
    log.info("build_db: wrote %d func files under %s/func", len(written), cfg.build.out_root)
    return 0


# --------------------------------------------------------------------------- #
# Stage 1: feature_select
# --------------------------------------------------------------------------- #
def cmd_feature_select(cfg: Config, outdir: Path) -> int:
    bundle = data.load_split(cfg)
    proxies = data.mcp_select(cfg, bundle)
    achieved = {q: r.q for q, r in sorted(proxies.items())}
    log.info("MCP achieved proxies per target Q: %s", achieved)
    full = max(proxies.values(), key=lambda pr: pr.q)
    save_json(outdir / "proxies.json",
              {"window_size": int(cfg.data.window_size),
               "proxies": {str(t): r.to_json() for t, r in proxies.items()},
               "selected": full.to_json()})
    save_proxies_csv(outdir / "proxies.csv", full.names, full.col_ids, full.weights)
    log.info("wrote %d proxies -> %s", full.q, outdir / "proxies.csv")
    return 0


# --------------------------------------------------------------------------- #
# Stage 2: fit
# --------------------------------------------------------------------------- #
def _fit_tree(cfg, qdir, label, Xtr, ytr, Xval, yval, col_ids, use_hpo):
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xval, label=yval) if Xval is not None else None
    if not use_hpo:
        booster, _, _ = models.train_boosting(
            models.NOHPO_PARAMS, dtrain, dval, num_rounds=models.NOHPO_ROUNDS,
            seed=cfg.train.base_seed, nthread=cfg.train.nthread)
        best = {"params": models.NOHPO_PARAMS, "num_rounds": models.NOHPO_ROUNDS,
                "achieved_rounds": models.NOHPO_ROUNDS, "val_mape": None,
                "leaves": int(models.count_leaves(booster))}
    else:
        if dval is None:
            raise RuntimeError("HPO needs a validation split (split.val_fraction > 0)")
        stamp = hpo.study_stamp(cfg, col_ids)
        pof: list[dict] = []
        for r in cfg.hpo.r_rgs:
            study = hpo.run_study(
                cfg, study_name=f"{label}_R{r}_{stamp}", storage_path=qdir / "optuna.db",
                dtrain=dtrain, dval=dval, y_val=yval, num_rounds=r,
                sampler=cfg.hpo.sampler, pruner=cfg.hpo.pruner,
                population_size=cfg.hpo.population_size, n_trials=cfg.hpo.n_trials,
                seed=cfg.runtime.seed)
            pts, trials = hpo.pareto_points(study)
            for p, t in zip(pts, trials):
                pof.append({"mape": float(p[0]), "leaves": int(p[1]), "params": t.params, "R": r,
                            "achieved_rounds": int(t.user_attrs.get("achieved_rounds", r))})
        if not pof:
            raise RuntimeError(f"{label}: HPO produced no completed trials")
        bt = hpo.pick_best_trial(pof, cfg.hpo.t_th)
        log.info("%s best tree: R=%d(ach %d) valMAPE=%.3f%% leaves=%d",
                 label, bt.num_rounds, bt.achieved_rounds, bt.val_mape, bt.leaves)
        X_fit = np.vstack([Xtr, Xval])
        y_fit = np.concatenate([ytr, yval])
        booster, _, _ = models.train_boosting(
            bt.params, xgb.DMatrix(X_fit, label=y_fit), None, num_rounds=bt.achieved_rounds,
            seed=cfg.train.base_seed, nthread=cfg.train.nthread)
        best = {"params": bt.params, "num_rounds": bt.num_rounds,
                "achieved_rounds": bt.achieved_rounds, "val_mape": bt.val_mape, "leaves": bt.leaves}
    booster.save_model(str(qdir / "model.json"))
    importances = models.tree_importance(booster, len(col_ids))
    leaves = int(models.count_leaves(booster))
    return best, importances, leaves, lambda X: booster.predict(xgb.DMatrix(X))


def _fit_nn(cfg, qdir, Xtr, ytr, Xval, yval, use_hpo):
    if not use_hpo:
        model, xs, ys = models.fit_scaled(Xtr, ytr, hidden=models.NOHPO_HIDDEN, seed=cfg.runtime.seed)
        best = {"hidden": models.NOHPO_HIDDEN, "alpha": 1e-4, "lr": 1e-3, "val_r2": None}
    else:
        if Xval is None or yval.size == 0:
            raise RuntimeError("HPO needs a validation split (split.val_fraction > 0)")
        best = hpo.run_nn_study(Xtr, ytr, Xval, yval,
                                cfg.hpo.nn_n_trials, cfg.hpo.nn_n_jobs, cfg.runtime.seed)
        log.info("best nn: hidden=%d alpha=%.2e lr=%.2e valR2=%.4f",
                 best["hidden"], best["alpha"], best["lr"], best["val_r2"])
        X_fit = np.vstack([Xtr, Xval])
        y_fit = np.concatenate([ytr, yval])
        model, xs, ys = models.fit_scaled(X_fit, y_fit, best["hidden"], best["alpha"],
                                          best["lr"], cfg.runtime.seed)
    joblib.dump({"model": model, "x_scaler": xs, "y_scaler": ys}, qdir / "model.joblib")
    return best, models.nn_importance(model), "", lambda X: models.predict(model, xs, ys, X)


def _cap_ridge_rows(cfg, X, y):
    """Seeded row subsample of the ridge fitting set (``ridge.max_rows``; 0 = all).

    Two costs scale with the row count, neither of them an SVD -- for ``p < n``
    and ``gcv_mode="auto"`` sklearn picks the ``"cov"`` path (a Q x Q
    eigendecomposition), never ``"svd"``:

    * memory -- the float64 standardized design is ``n_rows x Q x 8`` bytes on top
      of the float32 copy ``Union.slice`` already holds: 160 MB at 200k x 100, but
      16 GB at the ``window_size: 1`` x ``Q=1000`` corner;
    * time -- ``_solve_eigen_covariance`` recomputes two ``n x Q`` products once
      *per alpha* for the leverage diagonal, so the grid does not come free the
      way a single-decomposition path would: measured at n=120k, Q=150, RidgeCV
      takes 1.39 s over 5 alphas and 4.19 s over 25, against 1.05 s for one
      ``X'X`` + ``eigh`` (after which every alpha is a back-substitution).

    A linear model over at most a few thousand proxies is fully determined long
    before 200k rows, so the cap costs nothing measurable -- and at the default
    ``window_size: 32`` it never fires on aq_core (measured: 6895 train + 1721 val
    = 8616 fitting rows). Same seeded-subsample idiom as ``data.load_split``'s
    ``selection.max_rows``.
    """
    return _cap_rows(cfg, X, y, cfg.ridge.max_rows, "ridge.max_rows")


def _cap_rows(cfg, X, y, cap: int, what: str):
    """Seeded row subsample, shared by the ridge and rulefit fitters.

    ``cap <= 0`` and ``cap >= n`` both mean "every row" and return ``X`` and ``y``
    unchanged, by identity -- unlike ``src/xopm_lib``'s ``_row_sample``, where a
    cap of 0 samples zero rows. ``what`` names the config key that fired, because
    rulefit has three separate caps (``hpo_rows``, ``fit_rows``, ``h_rows``).
    Indices are sorted, so a benchmark's rows stay contiguous and in order.
    """
    n, cap = int(X.shape[0]), int(cap)
    if not (0 < cap < n):
        return X, y
    rng = np.random.default_rng(cfg.runtime.seed)
    rows = np.sort(rng.choice(n, size=cap, replace=False))
    log.info("%s: using a seeded %d/%d row subsample", what, cap, n)
    return X[rows], y[rows]


def _fit_ridge(cfg, qdir, Xtr, ytr, Xval, yval, names, col_ids, use_hpo):
    """L2-penalized linear fit; alpha from RidgeCV's leave-one-out generalized CV.

    Two deliberate divergences from ``_fit_tree`` / ``_fit_nn``, both consequences
    of the search living inside the estimator instead of in an optuna study on the
    validation tail:

    * The search does not consume the validation split -- RidgeCV's GCV runs
      *within* the rows it is handed -- so with HPO on, train and val go in as one
      fitting set. That is exactly the data the other two backends end up refit on
      once their hyperparameters are picked, which is why ``_run_one``'s
      "in HPO refit" panel captions are already correct here. ``best["val_r2"]``
      is ``None`` accordingly: nothing in this function scored the val split.
    * Consequently this is the one backend that runs at ``split.val_fraction: 0``
      with HPO on, where the other two raise. Do not add that guard here.
    """
    if use_hpo and Xval is not None and yval.size:
        X_fit = np.vstack([Xtr, Xval])
        y_fit = np.concatenate([ytr, yval])
    else:  # --no-hpo fits train only, like the other two, so the captions hold
        X_fit, y_fit = Xtr, ytr
    X_fit, y_fit = _cap_ridge_rows(cfg, X_fit, y_fit)

    n_fit = int(X_fit.shape[0])
    if use_hpo:
        alphas = models.ridge_alphas(cfg.ridge.alpha_rel_max, cfg.ridge.grid_decades,
                                     cfg.ridge.grid_points, n_fit)
        model, xs, ys = models.fit_ridge_scaled(
            X_fit, y_fit, alphas=alphas, fit_intercept=cfg.ridge.fit_intercept)
        # RidgeCV(scoring=None).best_score_ is NEGATIVE MSE on the standardized
        # target, not an R2. Named so it cannot be misread as one.
        best = {"alpha": float(model.alpha_), "gcv_neg_mse": float(model.best_score_),
                "alpha_grid": [float(alphas[0]), float(alphas[-1]), int(alphas.size)],
                "val_r2": None}
        log.info("best ridge: alpha=%.4e (alpha_rel=%.4e, LOO-GCV over %d rows, "
                 "grid %.1e..%.1e x%d)", best["alpha"], best["alpha"] / n_fit, n_fit,
                 alphas[0], alphas[-1], alphas.size)
    else:
        alpha = models.NOHPO_RIDGE_ALPHA_REL * n_fit
        model, xs, ys = models.fit_ridge_scaled(
            X_fit, y_fit, alpha=alpha, fit_intercept=cfg.ridge.fit_intercept)
        best = {"alpha": float(alpha), "gcv_neg_mse": None,
                "alpha_grid": None, "val_r2": None}
    coef_std, coef_watts, intercept_watts = models.ridge_coefficients(model, xs, ys)
    best["intercept_watts"] = intercept_watts
    best["n_fit_rows"] = n_fit
    # the row-relative alpha is the one comparable across runs and row counts
    best["alpha_rel"] = best["alpha"] / n_fit if n_fit else None
    joblib.dump({"model": model, "x_scaler": xs, "y_scaler": ys}, qdir / "model.joblib")
    save_ridge_coefficients_csv(qdir / "ridge_coefficients.csv", names, col_ids,
                                coef_std, coef_watts)
    return best, models.ridge_importance(model), "", lambda X: models.predict(model, xs, ys, X)


def _fit_rulefit(cfg, qdir, label, Xtr, ytr, Xval, yval, names, col_ids, use_hpo):
    """Rule ensemble + sparse linear readout; hyperparameters from an optuna study.

    Unlike ``_fit_ridge`` three functions up, this one KEEPS the tree/nn
    validation-split guard. Ridge is the deliberate exception because RidgeCV's
    GCV runs inside the rows it is handed and never scores val; this study does
    score the val tail, so without a val split there is nothing to maximize.

    Two more divergences from the other three fitters, both because nothing here
    standardizes anything:

    * ``model.joblib`` holds ``{"model": rf}`` alone, not the nn/ridge
      ``{"model", "x_scaler", "y_scaler"}`` triple. The fork's ``Winsorizer`` and
      ``FriedScale`` live on ``rf`` and rescale only the *linear* terms;
      predictions and coefficients are already in watts.
    * ``models.predict`` is not reused -- ``models.rulefit_predict`` is, because it
      also carries the row batching and the empty-split guard.
    """
    models._import_rulefit()  # fail here, not minutes deeper, if the fork is absent
    if use_hpo and (Xval is None or not yval.size):
        raise RuntimeError("HPO needs a validation split (split.val_fraction > 0)")

    if use_hpo:
        Xh, yh = _cap_rows(cfg, Xtr, ytr, cfg.rulefit.hpo_rows, "rulefit.hpo_rows")
        params = hpo.run_rulefit_study(cfg.rulefit, Xh, yh, Xval, yval, names,
                                       cfg.runtime.seed)
        log.info("best rulefit: tree_size=%d max_rules=%d memory_par=%.3g valR2=%.4f "
                 "(%d trials in %.0fs)", params["tree_size"], params["max_rules"],
                 params["memory_par"], params["val_r2"], params["n_trials"],
                 params["hpo_runtime_s"])
        # the same train+val set the tree and nn backends refit on, which is what
        # makes _run_one's "in HPO refit" panel captions correct here too
        X_fit = np.vstack([Xtr, Xval])
        y_fit = np.concatenate([ytr, yval])
        hpo_rows = int(Xh.shape[0])
    else:
        # clamp the fixed point to the configured ceiling: NOHPO_RULEFIT names
        # max_rules explicitly, so it would otherwise override a smaller cap
        params = dict(models.NOHPO_RULEFIT)
        params["max_rules"] = min(int(params["max_rules"]), int(cfg.rulefit.max_rules))
        params |= {"val_r2": None, "n_trials": None, "hpo_runtime_s": None}
        X_fit, y_fit, hpo_rows = Xtr, ytr, None
    X_fit, y_fit = _cap_rows(cfg, X_fit, y_fit, cfg.rulefit.fit_rows, "rulefit.fit_rows")

    n_fit = int(X_fit.shape[0])
    r = cfg.rulefit
    # logged BEFORE the fit: on the real data this takes minutes, and which
    # regime it is in (non-negative? intercept? trimmed?) decides how to read it
    log.info("rulefit: fitting %d rows x %d proxies | tree_size=%d max_rules=%d "
             "memory_par=%.3g exp_rand_tree_size=%s fit_intercept=%s "
             "non_negative_coef=%s lin_trim_quantile=%g n_alphas=%d cv=%d",
             n_fit, len(col_ids), params["tree_size"], params["max_rules"],
             params["memory_par"], r.exp_rand_tree_size, r.fit_intercept,
             r.non_negative_coef, r.lin_trim_quantile, r.n_alphas, r.cv)
    rf = models.fit_rulefit(
        models.make_rulefit(r, cfg.train.base_seed,
                            **{k: params[k] for k in models.RULEFIT_SPACE}),
        X_fit, y_fit, names)

    best = {k: params[k] for k in (*models.RULEFIT_SPACE, "val_r2", "n_trials",
                                   "hpo_runtime_s")}
    best |= models.rulefit_summary(rf)
    best |= {"n_fit_rows": n_fit, "hpo_rows": hpo_rows,
             "fit_intercept": bool(r.fit_intercept),
             "non_negative_coef": bool(r.non_negative_coef),
             "exp_rand_tree_size": bool(r.exp_rand_tree_size),
             "lin_trim_quantile": float(r.lin_trim_quantile)}
    if best["n_nonzero_terms"] == 0:
        # Not cosmetic: save_coefficients_csv skips normalization when the
        # importances sum to 0 and then ranks by |mcp_weight|, which would
        # republish the Stage-1 MCP order as if it were this model's.
        log.error("rulefit: every coefficient is zero -- the model predicts a "
                  "constant and rulefit_terms.csv is empty. Try "
                  "rulefit.non_negative_coef=false, rulefit.fit_intercept=true, "
                  "or a smaller rulefit.n_alphas.")

    joblib.dump({"model": rf}, qdir / "model.joblib")
    # written inside the fitter so a rule/coefficient desync fails here rather
    # than three splits later, once result.json is already on disk
    save_rulefit_terms_csv(qdir / "rulefit_terms.csv",
                           models.rulefit_terms(rf, names, col_ids))
    Xh, _ = _cap_rows(cfg, X_fit, y_fit, cfg.rulefit.h_rows, "rulefit.h_rows")

    def _draw_interaction_strength():
        # the H computation is INSIDE the guarded call, not an argument to it:
        # interaction_strength walks the rule ensemble, so it can fail on its own,
        # and this runs before result.json is written
        return plot_interaction_strength(
            models.rulefit_interactions(rf, Xh, cfg.rulefit.h_top_k),
            qdir / "interaction_strength.png", name=f"rulefit {label}")

    _try_plot(_draw_interaction_strength)
    return (best, models.rulefit_importance(rf, len(col_ids)), "",
            lambda X: models.rulefit_predict(rf, X, int(cfg.rulefit.predict_chunk)))


def _predictions_frame(preds: dict, splits: dict) -> pd.DataFrame:
    """Long frame of every split's labels and predictions, for later replotting.

    Lets a figure or a per-benchmark breakdown be redone without re-running the
    fit (an HPO fit costs minutes to hours). ``bench``/``split`` are stored as
    categoricals; rows no slice covers keep an empty benchmark name.
    """
    frames = []
    for split, (y, yhat) in preds.items():
        bench = np.full(y.size, "", dtype=object)
        for name, sl in splits[split][2].items():
            bench[sl] = name
        frames.append(pd.DataFrame(
            {"bench": bench, "split": split, "y_true": y, "y_pred": yhat}))
    out = pd.concat(frames, ignore_index=True)
    for col in ("bench", "split"):
        out[col] = out[col].astype("category")
    return out


def _try_plot(fn, *args, **kwargs):
    """Render a figure without ever letting it cost a completed fit.

    The property this relies on is that it NEVER RAISES: a render failure loses
    the figure and nothing else, where unwrapped it would abort the remaining
    ``-q`` experiments of a run whose models are already saved. That holds
    wherever it is called from, which matters because there are now two kinds of
    call site: the figures in ``_run_one`` are drawn after ``result.json`` is on
    disk, while ``_fit_rulefit``'s interaction-strength figure is drawn before it
    (only the fitter holds the fitted estimator). In both cases the fit survives.

    The broad ``except`` is safe structurally rather than by narrowing the
    exception tuple: the plot functions are unit-tested with no wrapper in the
    path, the full traceback is logged, and ``test_e2e`` asserts the exact
    artifact set, so this handler cannot go permanently active unnoticed.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - see the docstring
        log.exception("%s: figure not written (the fit result is unaffected)",
                      getattr(fn, "__name__", fn))
        return None


def _experiment_dir(mdir: Path, label: str, n_experiments: int) -> Path:
    """Where one experiment's artifacts go, under its model directory.

    A lone experiment -- the default ``-q -1`` -- writes straight into the model
    directory, so a ridge run is just ``<outdir>/ridge/`` with no single-child
    ``all/`` wrapper to descend through. Two or more experiments would overwrite
    each other there (one ``result.json``, one ``model.*``, one set of figures
    between them), so each of those keeps a ``<label>/`` subdirectory.

    ``label`` is deliberately not consulted for the choice: it stays the
    experiment's *name* -- the optuna study in ``_fit_tree``, ``result.json``'s
    ``label``, the ``report.md`` row, the figure titles -- whether or not it also
    names a directory.
    """
    return Path(mdir) if n_experiments == 1 else Path(mdir) / label


def _run_one(kind, cfg, qdir, label, names, col_ids, weights, union, use_hpo) -> dict:
    qdir = Path(qdir)
    qdir.mkdir(parents=True, exist_ok=True)
    Xtr, Xval, Xte = union.slice(col_ids)
    ytr, yval, yte = union.y_train, union.y_val, union.y_test
    log.info("[%s] %s: train %s val %s test %s", kind, label, Xtr.shape,
             None if Xval is None else Xval.shape, Xte.shape)
    # Explicit per-kind, never a bare else: an unknown kind used to fall through
    # to the nn fitter and get reported under its own name in result.json.
    if kind == "tree":
        best, importances, leaves, predict_fn = _fit_tree(
            cfg, qdir, label, Xtr, ytr, Xval, yval, col_ids, use_hpo)
    elif kind == "nn":
        best, importances, leaves, predict_fn = _fit_nn(cfg, qdir, Xtr, ytr, Xval, yval, use_hpo)
    elif kind == "ridge":
        best, importances, leaves, predict_fn = _fit_ridge(
            cfg, qdir, Xtr, ytr, Xval, yval, names, col_ids, use_hpo)
    elif kind == "rulefit":
        best, importances, leaves, predict_fn = _fit_rulefit(
            cfg, qdir, label, Xtr, ytr, Xval, yval, names, col_ids, use_hpo)
    else:
        raise ValueError(f"unknown model kind {kind!r} (expected {', '.join(MODEL_KINDS)})")

    save_coefficients_csv(qdir / "coefficients.csv", names, col_ids, weights, importances)
    window = int(cfg.data.window_size)
    splits = {}  # split -> (y, X, bench_slices)
    if ytr.size:
        splits["train"] = (ytr, Xtr, union.train_slices)
    if Xval is not None and yval.size:
        splits["val"] = (yval, Xval, union.val_slices)
    splits["test"] = (yte, Xte, union.test_slices)  # reported even when empty
    reports, preds = {}, {}
    for split, (y, X, bench_slices) in splits.items():
        # one inference per split, shared by the metrics and both figure types,
        # so what is plotted is provably what was scored
        yhat = predict_fn(X)
        reports[split] = evaluation_report(cfg, y, yhat, bench_slices, split)
        if y.size:
            preds[split] = (y, yhat)

    rec = {"method": kind, "mode": "hpo" if use_hpo else "no_hpo", "label": label,
           "q": int(len(col_ids)), "n_proxies": int(len(col_ids)), "best": best,
           "window_size": window, "final_leaves": leaves,
           "train_r2": reports.get("train", {}).get("r2"),
           "val_r2": reports.get("val", {}).get("r2"),
           "test_r2": reports["test"]["r2"], "test_mape": reports["test"]["mape"],
           # pooled test_r2 above concatenates the held-out benchmarks and scores
           # them against ONE mean, so a benchmark sitting in a different power
           # band inflates it. The per-benchmark split is each benchmark scored
           # against its own mean -- the honest per-workload number.
           "test_r2_per_benchmark": {b: m["r2"]
                                     for b, m in reports["test"]["per_benchmark"].items()},
           "reports": reports}
    save_json(qdir / "result.json", rec)
    # Figures and predictions come AFTER result.json. The model was saved back in
    # _fit_*, and _aggregate collects experiments by reading result.json back, so a
    # render failure here must not be able to erase a completed fit.
    if preds:
        save_pickle_zst(_predictions_frame(preds, splits), qdir / "predictions.pkl.zst")
        for split, (y, yhat) in preds.items():
            _try_plot(plot_pred_vs_time, y, yhat, splits[split][2],
                      qdir / f"pred_vs_time_{split}.png",
                      name=f"{kind} {label} | {split}", window_size=window)
        # the figure name records which splits existed. Only test is ever held
        # out, and with HPO not even val: the final model is refit on train+val
        # once the hyperparameters are picked, so both panels say why.
        in_sample = {"train": "in HPO refit", "val": "in HPO refit"} if use_hpo \
            else {"train": "in-sample"}
        _try_plot(plot_residual_panels, preds,
                  qdir / f"residual_{'_'.join(preds)}.png", name=f"{kind} {label}",
                  in_sample=in_sample)
    log.info("[%s] %s: test R2=%.4f MAPE=%.3f%% (train R2=%s val R2=%s)",
             kind, label, rec["test_r2"], rec["test_mape"], _fmt(rec["train_r2"]), _fmt(rec["val_r2"]))
    for bench, r2 in rec["test_r2_per_benchmark"].items():
        log.info("[%s] %s: test R2[%s]=%s", kind, label, bench, _fmt(r2))
    return rec


def _per_benchmark_test_lines(recs: list[dict]) -> list[str]:
    """Test R² broken out per held-out benchmark, one column per benchmark.

    The `test R²` column of the table above is POOLED: the held-out benchmarks
    are concatenated and scored against a single mean, which credits a model for
    merely separating two benchmarks that sit in different power bands. Each
    column here scores one benchmark against its own mean, so they are the
    numbers to read for per-workload accuracy -- and they can be much lower than
    the pooled one without anything being wrong.
    """
    # older result.json files (written before this column existed) carry the same
    # numbers under reports.test.per_benchmark, so read that as the fallback
    per = [({b: m.get("r2") for b, m in
             r.get("reports", {}).get("test", {}).get("per_benchmark", {}).items()}
            if not r.get("test_r2_per_benchmark") else r["test_r2_per_benchmark"])
           for r in recs]
    benches = list(dict.fromkeys(b for d in per for b in d))
    if not benches:
        return []
    head = " | ".join(f"test R² {b}" for b in benches)
    out = ["", "## Test R² per held-out benchmark", "",
           f"| experiment | Q | window | {head} |",
           "|---|---|---|" + "---|" * len(benches)]
    for r, d in zip(recs, per):
        cells = " | ".join(_fmt(d.get(b)) for b in benches)
        out.append(f"| {r.get('label','')} | {r.get('q','')} | "
                   f"{r.get('window_size', 1)} | {cells} |")
    out += ["", "Each column is scored against that benchmark's own mean; the "
            "`test R²`", "above pools all of them against one mean and is not "
            "their average."]
    return out


def _write_report_md(records: list[dict], path: Path, title: str, *,
                     flat: bool = False, sweep: bool = True) -> None:
    recs = sorted(records, key=lambda r: -r["q"])
    # window is per-record: an outdir reused across window sizes must not read
    # as one experiment just because the title stamps the latest run's value
    lines = [f"# {title}", "",
             "| experiment | Q | window | leaves | train R² | val R² | test R² | test MAPE% |",
             "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(
            f"| {r.get('label','')} | {r.get('q','')} | {r.get('window_size', 1)} | "
            f"{r.get('final_leaves','')} | "
            f"{_fmt(r.get('train_r2'))} | {_fmt(r.get('val_r2'))} | "
            f"{_fmt(r.get('test_r2'))} | {r.get('test_mape', float('nan')):.3f} |")
    lines += _per_benchmark_test_lines(recs)
    if recs:
        best = max(recs, key=lambda r: (r.get("test_r2") if r.get("test_r2") is not None else -9))
        # a lone experiment writes beside this report, several keep one <label>/
        # subdirectory each -- see _experiment_dir
        where = "Beside this report" if flat else "Per experiment, under `<experiment>/`"
        lines += ["", f"**Best test R²** = {_fmt(best.get('test_r2'))} at {best.get('label','')} "
                      f"(Q={best.get('q','')}, test MAPE {best.get('test_mape', float('nan')):.3f}%).",
                  "", "## Figures", "",
                  f"{where} (power plotted in mW; the",
                  "target and every metric above are in watts):", "",
                  "- `residual_train_val_test.png` — predicted vs true power, one panel",
                  "  per split, red dashed line is y = x. The name records which splits",
                  "  existed (`residual_train_test.png` at `split.val_fraction: 0`). With",
                  "  HPO the final model is refit on train+val, so those panels are",
                  "  labelled in-sample.",
                  "- `pred_vs_time_{train,val,test}.png` — label and prediction over the",
                  "  split's concatenated benchmarks, boundaries dotted and named. Full",
                  "  resolution, never decimated.",
                  "- `predictions.pkl.zst` — bench/split/y_true/y_pred, to redo either",
                  "  figure without re-running the fit."]
        # ridge-only, so a tree or nn report.md stays byte-identical to before
        if any(r.get("method") == "ridge" for r in recs):
            lines += ["- `ridge_coefficients.csv` — the signed linear fit: `coef_std` is",
                      "  comparable across bits, `coef_watts` is watts per unit of the",
                      "  feature. The matching `intercept_watts` is in `result.json`",
                      "  under `best`."]
        # rulefit-only, and a separate `if` rather than an `elif`, so a tree, nn
        # or ridge report.md stays byte-identical to before
        if any(r.get("method") == "rulefit" for r in recs):
            lines += ["- `rulefit_terms.csv` — the terms the model actually uses, one",
                      "  row per rule and per linear term, ranked by importance.",
                      "  `coef_watts` is directly comparable with",
                      "  `ridge_coefficients.csv`'s and needs no un-scaling; `col_ids`",
                      "  traces a rule back to its RTL nets. Its `importance` is the raw",
                      "  Friedman eq. (28)/(29) value, *not* normalized like",
                      "  `coefficients.csv`'s. `intercept_watts`, `n_rules` and the",
                      "  non-zero term counts are in `result.json` under `best`.",
                      "- `interaction_strength.png` — Friedman's overall H per feature, a",
                      "  ranking only (the library implements no null distribution).",
                      "- the `leaves` column above is blank on purpose: a deduplicated",
                      "  rule is not a score register. `best.n_rules` and",
                      "  `best.n_nonzero_rules` are this backend's size numbers."]
        if sweep:
            lines += ["", "Across experiments: `q_sweep.png`."]
    Path(path).write_text("\n".join(lines) + "\n")


def _aggregate(mdir: Path, title: str) -> list[dict]:
    mdir = Path(mdir)
    # An experiment is either a <label>/ subdirectory (several -q values) or mdir
    # itself (a lone one -- see _experiment_dir), and both shapes can coexist when
    # one outdir is reused across --fit runs. The scan is kept in preference to
    # reusing _run_one's return values: it is what lets the Q sweep accumulate
    # across separate invocations, and it reads result.json back from disk, which
    # is written before any figure so a render failure cannot erase an experiment.
    dirs = [d for d in sorted(mdir.iterdir())
            if d.is_dir() and (d / "result.json").exists()]
    if (mdir / "result.json").exists():
        dirs.insert(0, mdir)
    records = [load_json(d / "result.json") for d in dirs]
    if not records:
        log.warning("no result.json in %s or its subdirectories", mdir)
        return records
    # The one way the two shapes go wrong together: refitting into an outdir
    # written before the flat layout leaves the old <label>/ beside the new flat
    # result.json, and the SAME experiment is then reported twice -- two "all"
    # rows with different scores, and a q_sweep drawn across a duplicated point.
    # Different labels coexisting is legitimate (-q 50 then -q -1) and stays
    # quiet; only a repeated one is a stale directory.
    labels = [r.get("label", "") for r in records]
    dupes = sorted({lb for lb in labels if labels.count(lb) > 1})
    if dupes:
        log.warning("%s reports %s twice - a stale pre-flattening subdirectory is "
                    "sitting beside the flat result.json. Delete %s and refit, or "
                    "read report.md knowing the duplicate rows are different runs.",
                    mdir, ", ".join(dupes), ", ".join(str(mdir / lb) for lb in dupes))
    # A one-point sweep draws a single marker and no curve, so it is not written
    # -- and report.md must not advertise a figure that is not there. The prose
    # only claims the flat layout when that lone experiment IS the flat one.
    sweep = len(records) > 1
    flat = not sweep and dirs == [mdir]
    save_json(mdir / "metrics.json", {"records": records})
    _write_report_md(records, mdir / "report.md", title, flat=flat, sweep=sweep)
    if sweep:
        _try_plot(plot_q_sweep, records, mdir / "q_sweep.png")
    else:
        log.info("q_sweep: 1 experiment under %s, nothing to sweep", mdir)
    log.info("aggregate: %d experiments -> %s", len(records), mdir / "report.md")
    return records


def _build_selections(qs, names, pos, weights):
    """One (label, names, col_ids, weights) per requested q, deduped by label."""
    n = len(names)
    sels, seen = [], set()
    for q in qs:
        if q == -1 or q >= n:
            take, label = n, "all"
            if q not in (-1, n) and q > n:
                log.warning("-q %d exceeds %d available proxies; using all", q, n)
        elif q >= 1:
            take, label = q, f"q{q}"
        else:
            log.warning("ignoring invalid -q %d (want -1 or >=1)", q)
            continue
        if label in seen:
            continue
        seen.add(label)
        sels.append((label, names[:take], pos[:take], weights[:take]))
    return sels


def _warn_window_mismatch(cfg: Config, proxies_path: Path) -> None:
    """Warn when --fit aggregates rows differently than --feature_select did.

    Advisory only: a missing/unreadable ``proxies.json`` never fails the fit. A
    file with no ``window_size`` key predates the knob, so it was selected on
    per-cycle rows -- treat it as 1 rather than skipping the check, which is
    exactly the stale-proxies case worth warning about now that 32 is the default.
    """
    window = int(cfg.data.window_size)
    meta_path = Path(proxies_path).with_suffix(".json")
    if not meta_path.exists():
        return
    try:
        meta = load_json(meta_path)
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s for the window_size check (%s)", meta_path, exc)
        return
    if not isinstance(meta, dict):
        log.warning("%s is not a JSON object - skipping the window_size check", meta_path)
        return
    recorded = meta.get("window_size", 1)
    try:
        selected_with = int(recorded)
    except (TypeError, ValueError):
        log.warning("%s records a non-integer window_size %r - skipping the check",
                    meta_path, recorded)
        return
    if selected_with != window:
        how = ("recorded" if "window_size" in meta
               else "implied: this proxies.json predates data.window_size")
        log.warning("proxies were selected at window_size=%d (%s) but this fit uses %d - "
                    "the proxy ranking does not match the design matrix",
                    selected_with, how, window)


def cmd_fit(cfg: Config, outdir: Path, proxies_path: Path, qs, model_kinds, use_hpo) -> int:
    # checked before load_split/Union, which are minutes of I/O on the real
    # dataset -- a mistyped kind should not be discovered after all of that
    unknown = [k for k in model_kinds if k not in MODEL_KINDS]
    if unknown:
        raise ValueError(f"unknown model kind(s) {unknown} (expected {', '.join(MODEL_KINDS)})")
    names, col_ids, weights = load_proxies_csv(proxies_path)
    log.info("loaded %d proxies from %s", len(names), proxies_path)
    _warn_window_mismatch(cfg, proxies_path)
    bundle = data.load_split(cfg)
    name2pos = {c: i for i, c in enumerate(bundle.columns)}
    missing = [n for n in names if n not in name2pos]
    if missing:
        raise KeyError(f"{len(missing)} proxy names absent from the dataset columns, "
                       f"e.g. {missing[:3]} - proxies.csv and func_dir are out of sync")
    pos = np.array([name2pos[n] for n in names], dtype=np.int64)

    sels = _build_selections(qs, names, pos, weights)
    if not sels:
        raise ValueError(f"no valid -q values in {qs}")
    log.info("fit experiments: %s", [s[0] for s in sels])
    union = data.Union(cfg, bundle, [ids for _, _, ids, _ in sels])

    for kind in model_kinds:
        mdir = outdir / kind
        for label, snames, sids, sw in sels:
            _run_one(kind, cfg, _experiment_dir(mdir, label, len(sels)), label,
                     snames, sids, sw, union, use_hpo)
        _aggregate(mdir, title=f"binary_fit {kind} ({'hpo' if use_hpo else 'no_hpo'}, "
                              f"window={cfg.data.window_size})")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(prog="binary_fit")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build_db", action="store_true", help="materialize the single-bit dataset")
    mode.add_argument("--feature_select", action="store_true", help="MCP proxy selection -> proxies.csv")
    mode.add_argument("--fit", action="store_true", help="fit a model per -q proxy count")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None, help="output dir (feature_select / fit)")
    ap.add_argument("--proxies", default=None, help="proxies.csv path (default <outdir>/proxies.csv)")
    ap.add_argument("-q", "--q", type=int, nargs="+", default=[-1],
                    help="proxy counts to fit; -1 = all (default [-1])")
    ap.add_argument("--model", choices=[*MODEL_KINDS, "both"], default="tree",
                    help="Stage-2 regressor; 'both' fits every kind "
                         f"({', '.join(MODEL_KINDS)})")
    ap.add_argument("--no-hpo", action="store_true", help="skip HPO (fixed hyperparameters)")
    ap.add_argument("--window-size", "--window_size", type=int, default=None, dest="window_size",
                    help="cycles averaged into one row before selection/fit "
                         "(overrides data.window_size; 1 = per-cycle)")
    ap.add_argument("--nthread", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=None, help="tree HPO trials per (Q,R) study")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config, overrides=args.overrides)
    if args.nthread is not None:
        cfg.train.nthread = args.nthread
    if args.n_trials is not None:
        cfg.hpo.n_trials = args.n_trials
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    if args.window_size is not None:
        cfg.data.window_size = args.window_size
    cfg.validate()  # CLI flags bypass from_yaml's validation

    if args.build_db:
        return cmd_build_db(cfg)

    outdir = Path(args.outdir) if args.outdir else Path(cfg.runtime.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.feature_select:
        return cmd_feature_select(cfg, outdir)

    proxies_path = Path(args.proxies) if args.proxies else outdir / "proxies.csv"
    if not proxies_path.exists():
        raise FileNotFoundError(f"{proxies_path} not found - run --feature_select first")
    # "both" is inherited from when there were two backends; it means all of them
    kinds = list(MODEL_KINDS) if args.model == "both" else [args.model]
    return cmd_fit(cfg, outdir, proxies_path, args.q, kinds, use_hpo=not args.no_hpo)


if __name__ == "__main__":
    raise SystemExit(main())
