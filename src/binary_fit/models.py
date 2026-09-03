"""Stage-2 regressors: gradient-boosted trees (XGBoost) and a two-layer MLP.

Both consume the same binary MCP-selected proxies and are directly comparable.

* ``tree`` -- XGBoost mapping of the paper's Table II: squared-error objective,
  ``hist`` + ``lossguide`` growth bounded by ``max_leaves`` (the hardware cost
  knob). Boosting rounds R are swept explicitly by the HPO driver, not searched.
  Multi-objective HPO cannot prune via ``trial.report``, so Hyperband/Median run
  as rung-based early-stopping surrogates on the boosting-round axis
  (:class:`RungPruner`); in ``truncate`` mode a stopped trial still returns a
  valid (truncated) design point.
* ``nn`` -- ``sklearn.MLPRegressor(hidden_layer_sizes=(h,))`` ("two-layer" =
  one hidden + one output layer) with feature/target standardization; metrics
  are reported in the original power (W) scale.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# =========================================================================== #
# tree backend (XGBoost)
# =========================================================================== #
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

# Fixed hyperparameters for the tree --no-hpo baseline (a mid-space Table II point).
NOHPO_PARAMS: dict = {
    "eta": 0.1, "gamma": 1e-8, "max_leaves": 64, "min_child_weight": 1.0,
    "max_depth": 8, "reg_alpha": 1e-8, "reg_lambda": 1.0,
    "subsample": 0.9, "colsample_bytree": 0.9,
}
NOHPO_ROUNDS = 60


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


def tree_importance(booster: xgb.Booster, n_features: int) -> np.ndarray:
    """Gain-based importance aligned with the DMatrix columns (0 if never split)."""
    gain = booster.get_score(importance_type="gain")  # {"f{j}": gain}
    return np.array([gain.get(f"f{j}", 0.0) for j in range(n_features)], dtype=float)


# =========================================================================== #
# nn backend (two-layer MLP)
# =========================================================================== #
HIDDEN_CHOICES = [8, 16, 32, 64]
ALPHA_RANGE = (1e-6, 1e-1)  # L2 penalty, log scale
LR_RANGE = (1e-4, 1e-2)  # adam initial learning rate, log scale
NOHPO_HIDDEN = 16


def make_mlp(hidden: int, alpha: float, lr: float, seed: int, max_iter: int = 500) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(int(hidden),),
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate_init=float(lr),
        batch_size=256,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        max_iter=int(max_iter),
        shuffle=True,
        random_state=int(seed),
    )


def fit_scaled(X_tr, y_tr, hidden=16, alpha=1e-4, lr=1e-3, seed=0, max_iter=500):
    """Fit standardizers on train, then a two-layer MLP. Returns (model, xs, ys)."""
    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(np.asarray(y_tr, dtype=np.float64).reshape(-1, 1))
    model = make_mlp(hidden, alpha, lr, seed, max_iter)
    model.fit(xs.transform(X_tr), ys.transform(np.asarray(y_tr).reshape(-1, 1)).ravel())
    return model, xs, ys


def predict(model, xs, ys, X) -> np.ndarray:
    z = model.predict(xs.transform(X)).reshape(-1, 1)
    return ys.inverse_transform(z).ravel()


def nn_importance(model: MLPRegressor) -> np.ndarray:
    """Per-input connection-weight importance (Olden, magnitude-only).

    ``imp = (|W1| @ |W2| @ ... @ |W_L|).sum(axis=1)`` -- each input's total
    absolute weight along all paths to the output. Inputs are standardized before
    fitting, so first-layer magnitudes are comparable across features.
    """
    prod = np.abs(np.asarray(model.coefs_[0], dtype=np.float64))  # (n_features, h1)
    for w in model.coefs_[1:]:
        prod = prod @ np.abs(np.asarray(w, dtype=np.float64))
    return prod.sum(axis=1)
