"""Algorithm 1: two-stage power modeling for OPM design and optimization.

Stage 1 selects proxy sets for each target Q (LR-MCP lambda sweep).
Stage 2 runs, per Q, one multi-objective HPO study per boosting-round count R
with the candidate sampler+pruner pair, unions the per-R Pareto fronts,
re-filters, picks the Best Trial nearest the leaf budget T_th, retrains it on
train+val, and evaluates on the held-out test benchmarks.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import xgboost as xgb

from .config import CobitConfig
from .data.build import build_dataset
from .data.dataset import DatasetBundle, DatasetCache
from .evaluate import evaluation_report, mape_percent, plot_q_sweep, plot_trace
from .hpo import compare_sampler_pruner_pairs, pareto_points, run_study, study_stamp
from .model import train_boosting, count_leaves
from .pareto import non_dominated
from .selection import ProxyResult, select_proxies
from .utils import load_json, log, save_json


# ------------------------------------------------------------- stage 1 -----
def stage1_proxies(cfg: CobitConfig, run_dir: Path, force: bool = False) -> dict[int, ProxyResult]:
    """Select proxies for every target Q; cached in <run_dir>/proxies.json."""
    out_path = run_dir / "proxies.json"
    cache = DatasetCache(cfg)
    data_stamp = cache.data_stamp()
    if out_path.exists() and not force:
        raw = load_json(out_path)
        if (
            raw.get("registry_hash") == cache.registry.content_hash
            and raw.get("config_hash") == cfg.stage_hash("data", "split", "selection")
            and raw.get("data_stamp") == data_stamp
        ):
            log.info("proxies.json up to date - skipped")
            return {
                int(k): ProxyResult(
                    target_q=v["target_q"], q=v["q"], alpha=v["alpha"], gamma=v["gamma"],
                    col_ids=np.asarray(v["col_ids"], dtype=np.int64),
                    names=v["names"], weights=np.asarray(v["weights"]),
                )
                for k, v in raw["proxies"].items()
            }
        log.info("proxies.json stale - recomputing")

    kept = cache.kept_ids()
    X, y = cache.selection_matrix(kept, cfg.selection.max_rows, cfg.runtime.seed)
    results = select_proxies(cfg, X, y, kept, cache.registry.feature_names)
    save_json(
        out_path,
        {
            "registry_hash": cache.registry.content_hash,
            "config_hash": cfg.stage_hash("data", "split", "selection"),
            "data_stamp": data_stamp,
            "proxies": {str(t): r.to_json() for t, r in results.items()},
        },
    )
    return results


# ------------------------------------------------------------- stage 2 -----
@dataclasses.dataclass
class BestTrial:
    params: dict
    num_rounds: int  # R of the study the trial came from
    achieved_rounds: int  # rounds actually boosted (rung truncation)
    val_mape: float
    leaves: int


def pick_best_trial(pof: list[dict], t_th: int) -> BestTrial:
    """Best Trial = minimum-MAPE Pareto point whose leaf count fits T_th."""
    pts = np.array([[p["mape"], p["leaves"]] for p in pof], dtype=float)
    front_mask = non_dominated(pts)
    front = [p for p, m in zip(pof, front_mask) if m]
    feasible = [p for p in front if p["leaves"] <= t_th]
    if feasible:
        chosen = min(feasible, key=lambda p: p["mape"])
    else:
        chosen = min(front, key=lambda p: p["leaves"])
        log.warning(
            "no Pareto trial fits T_th=%d leaves; falling back to the smallest "
            "(%d leaves)", t_th, chosen["leaves"],
        )
    return BestTrial(
        params=chosen["params"],
        num_rounds=chosen["R"],
        achieved_rounds=chosen["achieved_rounds"],
        val_mape=chosen["mape"],
        leaves=chosen["leaves"],
    )


def _slice_bundle(
    union: DatasetBundle,
    union_ids: np.ndarray,
    col_ids: np.ndarray,
    dense_dtype=np.float32,
) -> DatasetBundle:
    """Dense per-Q view of the shared sparse union-column bundle."""
    pos = np.searchsorted(union_ids, col_ids)
    assert np.array_equal(union_ids[pos], col_ids), "proxy ids missing from union"

    def densify(X):
        return np.asarray(X.tocsc()[:, pos].todense(), dtype=dense_dtype)

    return dataclasses.replace(
        union,
        X_train=densify(union.X_train),
        X_val=densify(union.X_val),
        X_test=densify(union.X_test),
        col_ids=np.asarray(col_ids, dtype=np.int64),
        feature_names=[union.feature_names[i] for i in pos],
    )


def run_pipeline(cfg: CobitConfig, run_dir: Path, force: bool = False) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config_resolved.json", cfg.to_dict())
    build_dataset(cfg, force=False)  # idempotent
    proxies = stage1_proxies(cfg, run_dir, force=force)
    cache = DatasetCache(cfg)

    # one pass over the feature cache for the union of all proxy columns;
    # every Q (and the pair comparison) slices its columns from this bundle
    union_ids = np.unique(np.concatenate([pr.col_ids for pr in proxies.values()]))
    union = cache.load_split(union_ids, dense=False)
    if union.y_val.size == 0:
        raise RuntimeError(
            "HPO needs a validation split: set split.val_fraction > 0"
        )

    # optional Algorithm 2 pair comparison at a pinned (Q, R)
    dense_dtype = np.float64 if not cfg.data.bit_expand else np.float32
    if cfg.hpo.run_pair_comparison:
        q_cmp = cfg.hpo.pair_q or sorted(proxies)[0]
        if q_cmp not in proxies:
            raise ValueError(f"hpo.pair_q={q_cmp} is not one of selection.target_qs")
        b = _slice_bundle(union, union_ids, proxies[q_cmp].col_ids, dense_dtype=dense_dtype)
        dtrain = xgb.DMatrix(b.X_train, label=b.y_train)
        dval = xgb.DMatrix(b.X_val, label=b.y_val)
        compare_sampler_pruner_pairs(
            cfg, run_dir, dtrain, dval, b.y_val,
            stamp=study_stamp(cfg, proxies[q_cmp].col_ids),
        )

    records: list[dict] = []
    for target_q in sorted(proxies):
        pr = proxies[target_q]
        rec = _run_for_q(cfg, run_dir, cache, pr, union, union_ids, force=force)
        records.append(rec)

    save_json(run_dir / "metrics.json", {"records": records})
    plot_q_sweep(records, run_dir / "figures" / "q_sweep.png")
    log.info("pipeline complete: %s", run_dir / "metrics.json")
    return {"records": records}


def _run_for_q(
    cfg: CobitConfig,
    run_dir: Path,
    cache: DatasetCache,
    pr: ProxyResult,
    union: DatasetBundle,
    union_ids: np.ndarray,
    force: bool = False,
) -> dict:
    q = pr.target_q
    qdir = run_dir / f"Q{q}"
    qdir.mkdir(parents=True, exist_ok=True)
    done_path = qdir / "result.json"
    stage_hash = cfg.stage_hash("data", "split", "selection", "hpo", "train", "eval")
    data_stamp = cache.data_stamp()
    if done_path.exists() and not force:
        rec = load_json(done_path)
        if rec.get("config_hash") == stage_hash and rec.get("data_stamp") == data_stamp:
            log.info("Q=%d already complete - skipped", q)
            return rec

    dense_dtype = np.float64 if not cfg.data.bit_expand else np.float32
    bundle = _slice_bundle(union, union_ids, pr.col_ids, dense_dtype=dense_dtype)
    dtrain = xgb.DMatrix(bundle.X_train, label=bundle.y_train)
    dval = xgb.DMatrix(bundle.X_val, label=bundle.y_val)
    log.info(
        "Q=%d (actual %d proxies): train %s, val %s, test %s",
        q, pr.q, bundle.X_train.shape, bundle.X_val.shape, bundle.X_test.shape,
    )
    stamp = study_stamp(cfg, pr.col_ids)

    # inner loop over boosting rounds R; union the Pareto fronts
    pof: list[dict] = []
    for r in cfg.hpo.r_rgs:
        study = run_study(
            cfg,
            study_name=f"cp_Q{q}_R{r}_{stamp}",
            storage_path=run_dir / "optuna.db",
            dtrain=dtrain, dval=dval, y_val=bundle.y_val,
            num_rounds=r,
            sampler=cfg.hpo.sampler,
            pruner=cfg.hpo.pruner,
            population_size=cfg.hpo.population_size,
            n_trials=cfg.hpo.n_trials,
            seed=cfg.runtime.seed,
        )
        pts, trials = pareto_points(study)
        for p, t in zip(pts, trials):
            pof.append(
                {
                    "mape": float(p[0]),
                    "leaves": int(p[1]),
                    "params": t.params,
                    "R": r,
                    "achieved_rounds": int(t.user_attrs.get("achieved_rounds", r)),
                    "trial": t.number,
                }
            )
    if not pof:
        raise RuntimeError(f"Q={q}: HPO produced no completed trials")
    bt = pick_best_trial(pof, cfg.hpo.t_th)
    log.info(
        "Q=%d best trial: R=%d (achieved %d), val MAPE %.3f%%, %d leaves",
        q, bt.num_rounds, bt.achieved_rounds, bt.val_mape, bt.leaves,
    )

    # retrain the BT on train+val, evaluate on the held-out test benchmarks
    X_fit = np.vstack([bundle.X_train, bundle.X_val])
    y_fit = np.concatenate([bundle.y_train, bundle.y_val])
    dfit = xgb.DMatrix(X_fit, label=y_fit)
    booster, _, _ = train_boosting(
        bt.params, dfit, None,
        num_rounds=bt.achieved_rounds,
        seed=cfg.train.base_seed,
        nthread=cfg.train.nthread,
    )
    booster.save_model(qdir / "model.json")

    dtest = xgb.DMatrix(bundle.X_test)
    yhat = booster.predict(dtest)
    report = evaluation_report(cfg, bundle.y_test, yhat, bundle.test_slices, "test")
    val_pred = booster.predict(dval)
    val_mape_retrained, _ = mape_percent(bundle.y_val, val_pred, cfg.eval.mape_eps_frac)

    if bundle.y_test.size:
        plot_trace(
            bundle.y_test, yhat,
            qdir / "trace_test.png",
            max_cycles=cfg.eval.trace_plot_cycles,
            title=f"Q={q}: test power label vs prediction",
        )

    rec = {
        "q": q,
        "q_actual": pr.q,
        "alpha": pr.alpha,
        "best_trial": dataclasses.asdict(bt),
        "final_leaves": count_leaves(booster),
        "val_mape_retrained": val_mape_retrained,
        "test_mape": report["mape"],
        "test_r2": report["r2"],
        "report": report,
        "pof": pof,
        "config_hash": stage_hash,
        "data_stamp": data_stamp,
    }
    save_json(done_path, rec)
    return rec
