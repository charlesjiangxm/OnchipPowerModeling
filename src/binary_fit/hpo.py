"""Hyperparameter optimization for both Stage-2 backends.

* tree: multi-objective Optuna study minimizing (validation MAPE %, total leaf
  count) at a fixed boosting-round count R. NSGA-II/III, TPE or Random samplers;
  Hyperband/Median run as rung early-stopping surrogates (:class:`models.RungPruner`).
  The Algorithm-1 driver runs one study per R, unions the Pareto fronts and picks
  the Best Trial nearest the leaf budget T_th (:func:`pick_best_trial`).
* nn: single-objective Optuna study maximizing validation R2 over the MLP's
  hidden width, L2 alpha and learning rate (:func:`run_nn_study`).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from .config import Config
from .evaluate import mape_percent, r2_score
from .models import (
    RungPruner,
    count_leaves,
    fit_scaled,
    predict,
    suggest_params,
    train_boosting,
)
from .utils import load_json, log, save_json, stable_hash

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS = ("nsga2", "nsga3", "tpe", "random")
PRUNERS = ("hyperband", "median")


# --------------------------------------------------------------------------- #
# Pareto helpers
# --------------------------------------------------------------------------- #
def non_dominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows of an (n, 2) minimization array.

    Weak dominance: a dominates b iff a <= b componentwise and a < b in at least
    one objective. Duplicate points are all kept.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    n = pts.shape[0]
    order = np.lexsort((pts[:, 1], pts[:, 0]))  # by obj0 asc, then obj1 asc
    mask = np.zeros(n, dtype=bool)
    best1 = np.inf
    prev = None
    for i in order:
        x0, x1 = pts[i]
        if x1 < best1 or (prev is not None and x0 == prev[0] and x1 == prev[1]):
            mask[i] = True
            best1 = min(best1, x1)
            prev = (x0, x1)
    return mask


# --------------------------------------------------------------------------- #
# study identity
# --------------------------------------------------------------------------- #
def study_stamp(cfg: Config, col_ids) -> str:
    """Identity of everything a study's objectives depend on (folded into names)."""
    return stable_hash(
        {
            "config": cfg.stage_hash("data", "split", "selection", "hpo", "train", "eval"),
            "col_ids": np.asarray(col_ids, dtype=np.int64).tolist(),
        }
    )[:10]


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


# --------------------------------------------------------------------------- #
# tree HPO study
# --------------------------------------------------------------------------- #
def run_study(
    cfg: Config,
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
    budget_states = {optuna.trial.TrialState.COMPLETE}
    if cfg.hpo.prune_mode == "prune":
        budget_states.add(optuna.trial.TrialState.PRUNED)
    done = sum(t.state in budget_states for t in study.trials)
    if done >= n_trials:
        log.info("study %s already has %d trials - skipped", study_name, done)
        return study
    # resume-aware seed: an identically re-seeded sampler would replay suggestions
    study.sampler = make_sampler(sampler, population_size, seed + done)

    rung_pruner = RungPruner(pruner, num_rounds) if pruner in PRUNERS else None
    rung_path = storage_path.parent / f"rungs_{study_name}.json"
    if rung_pruner is not None and rung_path.exists():
        for k, v in load_json(rung_path).items():  # keep pruning history across resumes
            if int(k) in rung_pruner.history:
                rung_pruner.history[int(k)] = [float(x) for x in v]
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

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials - done)
    study.set_user_attr("runtime_s", study.user_attrs.get("runtime_s", 0.0) + time.time() - t0)
    if rung_pruner is not None:
        save_json(rung_path, {str(k): v for k, v in rung_pruner.history.items()})
    return study


def pareto_points(study: optuna.Study) -> tuple[np.ndarray, list[optuna.trial.FrozenTrial]]:
    """(n, 2) objective array and trials of the study's Pareto front."""
    trials = study.best_trials
    if not trials:
        return np.zeros((0, 2)), []
    return np.array([t.values for t in trials], dtype=float), trials


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


# --------------------------------------------------------------------------- #
# nn HPO study
# --------------------------------------------------------------------------- #
def run_nn_study(Xtr, y_train, Xval, y_val, n_trials: int, n_jobs: int, seed: int) -> dict:
    """Maximize validation R2 over (hidden, alpha, lr); returns best params dict."""
    from .models import ALPHA_RANGE, HIDDEN_CHOICES, LR_RANGE

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", HIDDEN_CHOICES)
        alpha = trial.suggest_float("alpha", *ALPHA_RANGE, log=True)
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        m, sx, sy = fit_scaled(Xtr, y_train, hidden, alpha, lr, seed)
        return r2_score(y_val, predict(m, sx, sy, Xval))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    bp = study.best_params
    return {"hidden": int(bp["hidden"]), "alpha": float(bp["alpha"]),
            "lr": float(bp["lr"]), "val_r2": float(study.best_value)}
