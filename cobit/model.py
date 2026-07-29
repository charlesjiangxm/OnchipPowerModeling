"""Boosting-tree power model: XGBoost mapping of the paper's Table II.

Fixed choices: squared-error objective, ``hist`` tree method with
``lossguide`` growth (leaf-wise, bounded by ``max_leaves`` - the direct
hardware cost knob). The number of boosting rounds R is NOT a searched
hyperparameter; Algorithm 1 sweeps it explicitly.

Pruning: optuna cannot prune multi-objective trials via ``trial.report``, so
Hyperband/Median are implemented as rung-based early-stopping surrogates on
the boosting-round axis (see :class:`RungPruner`). In ``truncate`` mode a
stopped trial returns the truncated model's real objectives - a truncated
boosting model is itself a valid OPM design point - so every trial stays
COMPLETE and usable for the Pareto front.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

FIXED_PARAMS: dict = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "eval_metric": "mape",
}

# Table II search space: (low, high, log-scale). Integers where noted.
SPACE = {
    "eta": (1e-8, 1.0, True),
    "gamma": (1e-8, 1.0, True),  # paper's gamma' (min split loss)
    "max_leaves": (16, 64, True),  # int
    "min_child_weight": (1e-8, 5.0, True),
    "max_depth": (4, 32, False),  # int
    "reg_alpha": (1e-8, 1.0, False),  # paper's alpha (L1 on leaf weights)
    "reg_lambda": (1e-8, 1.0, False),  # paper's lambda' (L2 on leaf weights)
    "subsample": (0.6, 1.0, False),
    "colsample_bytree": (0.6, 1.0, False),
}
_INT_PARAMS = {"max_leaves", "max_depth"}


def suggest_params(trial) -> dict:
    """Sample one Table II configuration from an optuna trial."""
    params = {}
    for name, (lo, hi, log) in SPACE.items():
        if name in _INT_PARAMS:
            params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log)
    return params


def count_leaves(booster: xgb.Booster) -> int:
    """Total leaves across all trees (= score registers in the OPM)."""
    return sum(d.count("leaf=") for d in booster.get_dump(dump_format="text"))


class RungPruner:
    """Shared-history successive-halving ('hyperband') or median pruning.

    One instance is shared by all trials of a study. ``observe`` records the
    trial's validation error at a rung and answers whether to stop boosting.
    """

    def __init__(self, kind: str, num_rounds: int, reduction: int = 3, min_history: int = 4):
        if kind not in ("hyperband", "median"):
            raise ValueError(f"unknown pruner kind {kind!r}")
        self.kind = kind
        self.reduction = reduction
        self.min_history = min_history
        if kind == "hyperband":
            # geometric rungs R/eta^k (k = deepest..1), like successive halving
            rungs = []
            r = num_rounds // reduction
            while r >= 1:
                rungs.append(r)
                r //= reduction
            self.rungs = sorted(set(rungs))
        else:
            self.rungs = sorted(
                {max(1, num_rounds // 4), max(1, num_rounds // 2), max(1, (3 * num_rounds) // 4)}
            )
        self.history: dict[int, list[float]] = {r: [] for r in self.rungs}

    def observe(self, rung: int, value: float) -> bool:
        """Record ``value`` at ``rung``; return True if the trial should stop."""
        hist = self.history[rung]
        stop = False
        if len(hist) >= self.min_history:
            if self.kind == "median":
                stop = value > float(np.median(hist))
            else:  # keep only the top-1/reduction fraction at each rung
                stop = value > float(np.quantile(hist, 1.0 / self.reduction))
        hist.append(value)
        return stop


class _RungCallback(xgb.callback.TrainingCallback):
    """Stops boosting at a rung boundary when the pruner says so."""

    def __init__(self, pruner: RungPruner, eval_name: str = "val", metric: str = "mape"):
        self.pruner = pruner
        self.eval_name = eval_name
        self.metric = metric
        self.stopped_at: int | None = None

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        rounds_done = epoch + 1
        if rounds_done in self.pruner.rungs:
            value = evals_log[self.eval_name][self.metric][-1]
            if self.pruner.observe(rounds_done, float(value)):
                self.stopped_at = rounds_done
                return True
        return False


def train_boosting(
    params: dict,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix | None,
    num_rounds: int,
    seed: int = 0,
    nthread: int = 0,
    rung_pruner: RungPruner | None = None,
) -> tuple[xgb.Booster, int, bool]:
    """Train R rounds (optionally rung-pruned); returns (booster, rounds, pruned)."""
    full = dict(FIXED_PARAMS, **params, seed=seed)
    if nthread:
        full["nthread"] = nthread
    evals = [(dval, "val")] if dval is not None else []
    callback = _RungCallback(rung_pruner) if (rung_pruner and dval is not None) else None
    booster = xgb.train(
        full,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        callbacks=[callback] if callback else None,
        verbose_eval=False,
    )
    pruned = callback is not None and callback.stopped_at is not None
    achieved = callback.stopped_at if pruned else num_rounds
    return booster, int(achieved), pruned
