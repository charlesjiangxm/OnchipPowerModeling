"""Algorithm 2: multi-objective design-space exploration of the boosting model.

Optuna studies with two minimized objectives: (validation MAPE %, total leaf
count). Samplers: NSGA-II, NSGA-III, TPE (optuna's multi-objective TPE is the
closest available analogue of the paper's "BO with EHVI"), and Random.
Pruners: because optuna forbids ``trial.report`` in multi-objective studies,
Hyperband/Median run as rung-based early-stopping surrogates on the
boosting-round axis (:class:`cobit.model.RungPruner`); in the default
``truncate`` mode a stopped trial still returns its truncated model's real
objectives, so every trial completes and contributes to the Pareto front.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from .config import CobitConfig
from .evaluate import mape_percent
from .model import RungPruner, count_leaves, suggest_params, train_boosting
from .pareto import (
    coverage_matrix,
    hypervolume_2d,
    net_coverage,
    non_dominated,
    roi_filter,
    spacing,
)
from .utils import load_json, log, save_json, stable_hash


def study_stamp(cfg: CobitConfig, col_ids) -> str:
    """Identity of everything a study's objectives depend on.

    Folded into the study NAME so that resuming a run directory after a
    config or proxy-set change starts a fresh study instead of silently
    reusing trials whose objectives were computed on different data.
    """
    return stable_hash(
        {
            "config": cfg.stage_hash("data", "split", "selection", "hpo", "train", "eval"),
            "col_ids": np.asarray(col_ids, dtype=np.int64).tolist(),
        }
    )[:10]

SAMPLERS = ("nsga2", "nsga3", "tpe", "random")
PRUNERS = ("hyperband", "median")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sampler(kind: str, population_size: int, seed: int) -> optuna.samplers.BaseSampler:
    if kind == "nsga2":
        return optuna.samplers.NSGAIISampler(population_size=population_size, seed=seed)
    if kind == "nsga3":
        return optuna.samplers.NSGAIIISampler(population_size=population_size, seed=seed)
    if kind == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=population_size)
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler {kind!r}")


def run_study(
    cfg: CobitConfig,
    study_name: str,
    storage_path: Path,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    y_val: np.ndarray,
    num_rounds: int,
    sampler: str,
    pruner: str,
    population_size: int,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """One multi-objective HPO study at fixed (Q dataset, R rounds)."""
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )
    # only trials that actually produced (or deliberately pruned) objectives
    # consume budget; FAILED trials from crashes must not
    budget_states = {optuna.trial.TrialState.COMPLETE}
    if cfg.hpo.prune_mode == "prune":
        budget_states.add(optuna.trial.TrialState.PRUNED)
    done = sum(t.state in budget_states for t in study.trials)
    if done >= n_trials:
        log.info("study %s already has %d trials - skipped", study_name, done)
        return study
    # resume-aware seed: an identically re-seeded sampler would replay the
    # suggestions of the trials already in the study
    study.sampler = make_sampler(sampler, population_size, seed + done)

    rung_pruner = RungPruner(pruner, num_rounds) if pruner in PRUNERS else None
    rung_path = storage_path.parent / f"rungs_{study_name}.json"
    if rung_pruner is not None and rung_path.exists():
        for k, v in load_json(rung_path).items():  # keep pruning history across resumes
            if int(k) in rung_pruner.history:
                rung_pruner.history[int(k)] = [float(x) for x in v]
    hv_history: list[tuple[int, float]] = []
    eps_frac = cfg.eval.mape_eps_frac

    def objective(trial: optuna.Trial):
        params = suggest_params(trial)
        booster, achieved, pruned = train_boosting(
            params,
            dtrain,
            dval,
            num_rounds=num_rounds,
            seed=cfg.train.base_seed + trial.number,
            nthread=cfg.train.nthread,
            rung_pruner=rung_pruner,
        )
        if pruned and cfg.hpo.prune_mode == "prune":
            raise optuna.TrialPruned()
        pred = booster.predict(dval)
        mape, n_masked = mape_percent(y_val, pred, eps_frac=eps_frac)
        leaves = count_leaves(booster)
        trial.set_user_attr("achieved_rounds", achieved)
        trial.set_user_attr("rung_stopped", pruned)
        trial.set_user_attr("mape_masked_cycles", n_masked)
        return mape, leaves

    def hv_callback(study_: optuna.Study, _trial) -> None:
        pts = np.array(
            [t.values for t in study_.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        hv = hypervolume_2d(pts, tuple(cfg.hpo.hv_ref)) if pts.size else 0.0
        hv_history.append((len(study_.trials), hv))

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials - done, callbacks=[hv_callback])
    study.set_user_attr("runtime_s", study.user_attrs.get("runtime_s", 0.0) + time.time() - t0)
    if rung_pruner is not None:
        save_json(rung_path, {str(k): v for k, v in rung_pruner.history.items()})
    save_json(
        storage_path.parent / f"hv_history_{study_name}.json",
        {"study": study_name, "hv_ref": cfg.hpo.hv_ref, "history": hv_history},
    )
    return study


def pareto_points(study: optuna.Study) -> tuple[np.ndarray, list[optuna.trial.FrozenTrial]]:
    """(n, 2) objective array and trials of the study's Pareto front."""
    trials = study.best_trials
    if not trials:
        return np.zeros((0, 2)), []
    return np.array([t.values for t in trials], dtype=float), trials


def compare_sampler_pruner_pairs(
    cfg: CobitConfig,
    run_dir: Path,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    y_val: np.ndarray,
    stamp: str = "",
) -> dict:
    """Algorithm 2 main loop: rank sampler x pruner pairs, pick the CP.

    Runs at a fixed (Q, R) - the paper used Q=197, R=50 - for each population
    size; reports HV, Spacing, #Pareto, Coverage/NetCoverage (full and ROI).
    """
    r = cfg.hpo.pair_r
    storage_path = run_dir / "optuna.db"
    results: dict = {"pairs": {}, "pop_sizes": {}}
    roi = (cfg.hpo.roi_mape, float(cfg.hpo.roi_leaves))

    for pop in cfg.hpo.pair_pop_sizes:
        fronts, labels = [], []
        for s in SAMPLERS:
            for p in PRUNERS:
                name = f"pair_{s}_{p}_pop{pop}_R{r}" + (f"_{stamp}" if stamp else "")
                study = run_study(
                    cfg, name, storage_path, dtrain, dval, y_val,
                    num_rounds=r, sampler=s, pruner=p,
                    population_size=pop, n_trials=cfg.hpo.pair_n_trials,
                    seed=cfg.runtime.seed,
                )
                pts, _ = pareto_points(study)
                fronts.append(pts)
                labels.append(f"{s}+{p}")
                in_roi = pts[roi_filter(pts, *roi)] if pts.size else pts
                results["pairs"][f"{s}+{p}@pop{pop}"] = {
                    "n_pareto": int(pts.shape[0]),
                    "hv": hypervolume_2d(pts, tuple(cfg.hpo.hv_ref)),
                    "hv_roi": hypervolume_2d(in_roi, tuple(cfg.hpo.hv_ref)),
                    "spacing": spacing(pts),
                    "runtime_s": study.user_attrs.get("runtime_s"),
                    "front": pts.tolist(),
                }
        netcov = net_coverage(fronts)
        rois = [f[roi_filter(f, *roi)] if f.size else f for f in fronts]
        netcov_roi = net_coverage(rois)
        cov = coverage_matrix(fronts)
        results["pop_sizes"][str(pop)] = {
            "labels": labels,
            "coverage_matrix": cov.tolist(),
            "net_coverage": dict(zip(labels, netcov.tolist())),
            "net_coverage_roi": dict(zip(labels, netcov_roi.tolist())),
        }
        for lbl, nc, ncr in zip(labels, netcov, netcov_roi):
            results["pairs"][f"{lbl}@pop{pop}"]["netcov"] = float(nc)
            results["pairs"][f"{lbl}@pop{pop}"]["netcov_roi"] = float(ncr)

    # CP choice: best mean ROI NetCoverage across population sizes, HV_roi tiebreak
    agg: dict[str, list[float]] = {}
    for key, rec in results["pairs"].items():
        lbl = key.split("@")[0]
        agg.setdefault(lbl, []).append((rec["netcov_roi"], rec["hv_roi"]))
    ranking = sorted(
        agg.items(),
        key=lambda kv: (np.mean([x[0] for x in kv[1]]), np.mean([x[1] for x in kv[1]])),
        reverse=True,
    )
    results["ranking"] = [
        {"pair": lbl, "mean_netcov_roi": float(np.mean([x[0] for x in v])),
         "mean_hv_roi": float(np.mean([x[1] for x in v]))}
        for lbl, v in ranking
    ]
    results["candidate_pair"] = ranking[0][0]
    log.info("Algorithm 2 candidate pair: %s", results["candidate_pair"])
    save_json(run_dir / "pair_comparison.json", results)
    _plot_pair_comparison(results, run_dir)
    return results


def _plot_pair_comparison(results: dict, run_dir: Path) -> None:
    """Fig.7-style diagnostics: per-pair Pareto fronts + coverage heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    for pop, rec in results["pop_sizes"].items():
        labels = rec["labels"]
        # small multiples: one Pareto scatter per pair (single hue - identity
        # is carried by the facet, not by color)
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        for ax, lbl in zip(axes.ravel(), labels):
            pts = np.array(results["pairs"][f"{lbl}@pop{pop}"]["front"])
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], s=18, color="#2a78d6")
            ax.set_title(lbl, fontsize=10)
            ax.grid(True, color="#eeeeee", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.supxlabel("validation MAPE (%)")
        fig.supylabel("total leaves")
        fig.suptitle(f"Pareto fronts by sampler+pruner (pop={pop})")
        fig.tight_layout()
        fig.savefig(figdir / f"pareto_pairs_pop{pop}.png", dpi=150)
        plt.close(fig)

        cov = np.array(rec["coverage_matrix"])
        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.imshow(cov, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="#0b0b0b" if cov[i, j] < 0.6 else "#ffffff")
        ax.set_title(f"Coverage Cov(row, col) (pop={pop})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(figdir / f"coverage_pop{pop}.png", dpi=150)
        plt.close(fig)
