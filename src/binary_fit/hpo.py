"""Hyperparameter optimization for the Stage-2 backends.

* tree: multi-objective Optuna study minimizing (validation MAPE %, total leaf
  count) at a fixed boosting-round count R. NSGA-II/III, TPE or Random samplers;
  Hyperband/Median run as rung early-stopping surrogates (:class:`models.RungPruner`).
  The Algorithm-1 driver runs one study per R, unions the Pareto fronts and picks
  the Best Trial nearest the leaf budget T_th (:func:`pick_best_trial`).
* nn: single-objective Optuna study maximizing validation R2 over the MLP's
  hidden width, L2 alpha and learning rate (:func:`run_nn_study`).
* rulefit: single-objective Optuna study maximizing validation R2 over the rule
  stage's (tree_size, max_rules, memory_par) (:func:`run_rulefit_study`), with a
  hard wall-clock timeout. Unnamed and unpersisted on purpose, so
  :func:`study_stamp` is never involved -- see that function's docstring.
* ridge does not appear here at all: its alpha comes from RidgeCV's leave-one-out
  generalized CV inside the fitting rows.
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
    """Identity of everything a study's objectives depend on (folded into names).

    Note which sections are listed: a new top-level section (``ridge:``,
    ``rulefit:``) is NOT hashed, which is exactly why those backends keep their
    knobs in one. Adding a field to ``hpo`` instead renames every tree study and
    orphans the trials already in an existing ``analysis/.../tree/*/optuna.db``,
    silently -- ``load_if_exists=True`` just starts fresh.
    """
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


# --------------------------------------------------------------------------- #
# rulefit HPO study
# --------------------------------------------------------------------------- #
def run_rulefit_study(rf_cfg, Xtr, y_train, Xval, y_val, names, seed: int) -> dict:
    """Maximize validation R2 over (tree_size, max_rules, memory_par).

    The ``run_nn_study`` shape -- in-memory, unnamed, single-objective TPE -- with
    two additions the rulefit cost makes necessary:

    * ``timeout``. One full fit measured 448.5 s at 68,966 rows x 39 proxies, so
      an uncapped search runs for hours with nothing to point at.
    * the no-completed-trials guard that ``_fit_tree`` has and ``run_nn_study``
      lacks. Without it a study whose every trial errored or was cut short by the
      timeout fails in ``study.best_params`` with a bare ``ValueError``.

    The objective scores the COMPLETE RuleFit, not just its booster the way
    ``src/xopm_lib/model_regression.py`` does. That shortcut buys nothing here:
    the tree stage is ~1 s of a 448 s fit, so skipping the lasso would save
    almost none of the cost while tuning a model that is never shipped.

    Single-objective, with ``max_rules`` merely capped rather than a second
    objective, because the deployed cost is the NON-ZERO term count -- which the
    lasso already minimizes (measured 39 linear + 75 rules out of a 354-column
    design). The per-trial ``n_rules`` / ``n_nonzero_terms`` user attributes keep
    that trade-off readable; ``non_dominated`` above is the upgrade path if the
    measured curve turns out steep.
    """
    # Imported inside the function, like run_nn_study's own models import: the
    # rulefit fork must stay lazy, so it must not join the module-level list.
    from .models import (
        fit_rulefit,
        make_rulefit,
        rulefit_predict,
        suggest_rulefit_params,
    )

    def objective(trial):
        params = suggest_rulefit_params(trial, rf_cfg.max_rules)
        rf = fit_rulefit(make_rulefit(rf_cfg, seed, **params), Xtr, y_train, names)
        trial.set_user_attr("n_rules", int(rf.n_rules_))
        trial.set_user_attr("n_nonzero_terms", int((np.asarray(rf.coef_) != 0).sum()))
        return r2_score(y_val, rulefit_predict(rf, Xval, rf_cfg.predict_chunk))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    t0 = time.time()
    study.optimize(objective, n_trials=rf_cfg.hpo_n_trials,
                   n_jobs=rf_cfg.hpo_n_jobs,
                   timeout=(float(rf_cfg.hpo_timeout_s) or None))
    runtime = time.time() - t0
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError(
            f"rulefit HPO produced no completed trials in {runtime:.0f}s "
            f"(rulefit.hpo_timeout_s={rf_cfg.hpo_timeout_s} too small for a fit "
            f"this size, or every trial errored -- check the logged warnings)"
        )
    bp = study.best_params
    return {"tree_size": int(bp["tree_size"]), "max_rules": int(bp["max_rules"]),
            "memory_par": float(bp["memory_par"]),
            "val_r2": float(study.best_value), "n_trials": len(done),
            "hpo_runtime_s": float(runtime)}
