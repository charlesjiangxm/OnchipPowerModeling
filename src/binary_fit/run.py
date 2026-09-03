"""
Contains three stages: build_db, feature_select, fit. Run directly (no package/-m needed):
1. Build DB only (config-agnostic, but need one; cobit.yaml and nn.yaml have identical build sections):
    python src/binary_fit/run.py --build_db --config src/binary_fit/configs/cobit.yaml

2. Feature selection:
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/cobit.yaml --outdir analysis/cobit/2026-09-01-16-30-01
    python src/binary_fit/run.py --feature_select --config src/binary_fit/configs/nn.yaml --outdir analysis/nn/2026-09-01-16-30-01

3. Regression (fit):
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/cobit.yaml --outdir analysis/cobit/2026-09-01-16-30-01 --model tree
    python src/binary_fit/run.py --fit --config src/binary_fit/configs/nn.yaml --outdir analysis/nn/2026-09-01-16-30-01 --model nn

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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binary_fit import build_db, data, hpo, models
from binary_fit.config import Config
from binary_fit.evaluate import evaluation_report, plot_q_sweep, plot_trace
from binary_fit.utils import (
    load_json,
    load_proxies_csv,
    log,
    save_coefficients_csv,
    save_json,
    save_proxies_csv,
    setup_logging,
)


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


def _run_one(kind, cfg, mdir, label, names, col_ids, weights, union, use_hpo) -> dict:
    qdir = Path(mdir) / label
    qdir.mkdir(parents=True, exist_ok=True)
    Xtr, Xval, Xte = union.slice(col_ids)
    ytr, yval, yte = union.y_train, union.y_val, union.y_test
    log.info("[%s] %s: train %s val %s test %s", kind, label, Xtr.shape,
             None if Xval is None else Xval.shape, Xte.shape)
    if kind == "tree":
        best, importances, leaves, predict_fn = _fit_tree(
            cfg, qdir, label, Xtr, ytr, Xval, yval, col_ids, use_hpo)
    else:
        best, importances, leaves, predict_fn = _fit_nn(cfg, qdir, Xtr, ytr, Xval, yval, use_hpo)

    save_coefficients_csv(qdir / "coefficients.csv", names, col_ids, weights, importances)
    reports = {}
    if ytr.size:
        reports["train"] = evaluation_report(cfg, ytr, predict_fn(Xtr), union.train_slices, "train")
    if Xval is not None and yval.size:
        reports["val"] = evaluation_report(cfg, yval, predict_fn(Xval), union.val_slices, "val")
    yhat = predict_fn(Xte)
    reports["test"] = evaluation_report(cfg, yte, yhat, union.test_slices, "test")
    window = int(cfg.data.window_size)
    if yte.size:
        unit = "cycle" if window == 1 else f"window ({window} cycles)"
        plot_trace(yte, yhat, qdir / "trace_test.png", max_cycles=cfg.eval.trace_plot_cycles,
                   title=f"{kind} {label}: test power label vs prediction", xlabel=unit)

    rec = {"method": kind, "mode": "hpo" if use_hpo else "no_hpo", "label": label,
           "q": int(len(col_ids)), "n_proxies": int(len(col_ids)), "best": best,
           "window_size": window, "final_leaves": leaves,
           "train_r2": reports.get("train", {}).get("r2"),
           "val_r2": reports.get("val", {}).get("r2"),
           "test_r2": reports["test"]["r2"], "test_mape": reports["test"]["mape"],
           "reports": reports}
    save_json(qdir / "result.json", rec)
    log.info("[%s] %s: test R2=%.4f MAPE=%.3f%% (train R2=%s val R2=%s)",
             kind, label, rec["test_r2"], rec["test_mape"], _fmt(rec["train_r2"]), _fmt(rec["val_r2"]))
    return rec


def _write_report_md(records: list[dict], path: Path, title: str) -> None:
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
    if recs:
        best = max(recs, key=lambda r: (r.get("test_r2") if r.get("test_r2") is not None else -9))
        lines += ["", f"**Best test R²** = {_fmt(best.get('test_r2'))} at {best.get('label','')} "
                      f"(Q={best.get('q','')}, test MAPE {best.get('test_mape', float('nan')):.3f}%)."]
    Path(path).write_text("\n".join(lines) + "\n")


def _aggregate(mdir: Path, title: str) -> list[dict]:
    mdir = Path(mdir)
    records = [load_json(d / "result.json") for d in sorted(mdir.iterdir())
               if d.is_dir() and (d / "result.json").exists()]
    if not records:
        log.warning("no */result.json under %s", mdir)
        return records
    save_json(mdir / "metrics.json", {"records": records})
    plot_q_sweep(records, mdir / "figures" / "q_sweep.png")
    _write_report_md(records, mdir / "report.md", title)
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
            _run_one(kind, cfg, mdir, label, snames, sids, sw, union, use_hpo)
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
    ap.add_argument("--model", choices=["tree", "nn", "both"], default="tree")
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
    kinds = ["tree", "nn"] if args.model == "both" else [args.model]
    return cmd_fit(cfg, outdir, proxies_path, args.q, kinds, use_hpo=not args.no_hpo)


if __name__ == "__main__":
    raise SystemExit(main())
