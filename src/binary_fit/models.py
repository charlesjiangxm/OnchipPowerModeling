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
* ``ridge`` -- L2-penalized linear regression, the linear reference point for
  the other two: it answers how much of the power is a weighted sum of the
  selected proxy bits, and its coefficients are per-bit watts. Ridge rather
  than OLS because the proxy set is strongly collinear (80.3% of the aq_core
  kept bits are exact copies of another bit), so ``X'X`` is rank-deficient and
  the penalty is what makes the solve well-posed at all. Alpha comes from
  ``RidgeCV``'s leave-one-out generalized CV over the fitting rows, not from an
  optuna study on the validation tail -- see :func:`fit_ridge_scaled`.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV
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


# =========================================================================== #
# ridge backend (L2-penalized linear)
# =========================================================================== #
# --no-hpo: mid-grid. ROW-RELATIVE, like the grid -- see ridge_alphas.
NOHPO_RIDGE_ALPHA_REL = 1e-2


def ridge_alphas(alpha_rel_max: float, decades: float, points: int, n_rows: int) -> np.ndarray:
    """Descending log-spaced grid of ABSOLUTE alphas from a ROW-RELATIVE spec.

    Configured relative to the row count and multiplied up here, because sklearn
    minimizes a *sum* of squared residuals, not a mean: ``alpha`` competes with
    ``||Zw||^2``, which grows with n. Standardizing on the training rows makes
    that exact -- ``diag(Z'Z) = n`` to 5e-15 (a zero-variance column gets
    ``scale_ = 1.0`` and contributes 0) -- so a unit-variance direction is
    shrunk by ``n / (n + alpha) = 1 / (1 + alpha_rel)``. One relative number
    therefore means the same amount of regularization at every n, and a fixed
    absolute one does not: an absolute grid topping out at 1e4 shrinks to 0.013
    at n=135 but only to 0.95 at n=200_000, i.e. its *heaviest* point is nearly
    unregularized exactly where the design is largest and most rank-deficient.
    Verified by row replication: tiling the rows 4x moves an absolute-alpha
    solution by 1.3e-3 and a row-relative one by 2.1e-15.

    Descending, matching the Stage-1 alpha sweep (``selection.select_proxies``).
    The floor stays bounded away from zero for the reason the penalty exists at
    all: duplicate proxy bits make ``X'X`` singular, and an unregularized solve
    on them returns coefficients four orders of magnitude too large.
    """
    grid = np.logspace(0.0, -float(decades), int(points))
    return float(alpha_rel_max) * float(n_rows) * grid


def fit_ridge_scaled(X_tr, y_tr, alphas=None, alpha=None, fit_intercept: bool = True):
    """Fit standardizers on train, then ``RidgeCV(alphas)`` or ``Ridge(alpha)``.

    Returns ``(model, xs, ys)`` -- the same triple as :func:`fit_scaled`, so
    :func:`predict` is reused verbatim and ``run._fit_ridge`` persists the same
    ``model.joblib`` payload as the nn backend. Exactly one of ``alphas`` (search)
    or ``alpha`` (fixed) must be given.

    Both X and y are standardized, which is the *opposite* of the Stage-1 choice
    documented in ``selection.py`` ("standardizing would destroy sparsity and the
    toggle semantics"). That argument is about a sparse L1 selector over a sparse
    0/1 matrix; this is a dense L2 fit whose penalty is scale-dependent, so
    without standardization alpha would mean something different per feature and
    the coefficients would not be comparable across bits. Standardizing y as well
    makes the grid dimensionless -- alpha does not have to be re-tuned when the
    target changes physical scale -- and :func:`predict` inverse-transforms back
    to watts, so every reported metric stays in the original power scale.

    The solve runs in float64. sklearn preserves its input dtype and
    ``Union.slice`` hands over float32; factorizing a rank-deficient design in
    float32 (an SVD on the RidgeCV path, a Cholesky on the fixed-alpha one) is
    not something to rest a published coefficient table on.
    """
    if (alphas is None) == (alpha is None):
        raise ValueError("fit_ridge_scaled needs exactly one of alphas= (a RidgeCV "
                         "grid) or alpha= (a fixed L2 strength)")
    y_tr = np.asarray(y_tr, dtype=np.float64).reshape(-1, 1)
    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr)
    Z = np.asarray(xs.transform(X_tr), dtype=np.float64)
    z = ys.transform(y_tr).ravel()
    if alphas is not None:
        # scoring=None selects the efficient leave-one-out generalized-CV path
        model = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64),
                        fit_intercept=fit_intercept, scoring=None)
    else:
        model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
    model.fit(Z, z)
    return model, xs, ys


def ridge_importance(model) -> np.ndarray:
    """Per-input ``|coef_|`` on the standardized scale (magnitude-only).

    The direct analogue of :func:`nn_importance`, and non-negative for the same
    reason: ``utils.save_coefficients_csv`` normalizes ``importances`` to sum to 1
    and ranks by them descending, both of which a signed vector would corrupt.
    Inputs are standardized before fitting, so magnitudes are comparable across
    features; the signed values go to ``ridge_coefficients.csv``.
    """
    return np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()


def ridge_coefficients(model, xs: StandardScaler, ys: StandardScaler):
    """The fit un-scaled: ``(coef_std, coef_watts, intercept_watts)``.

    :func:`fit_ridge_scaled` solves in standardized space, so ``model.coef_`` is
    watts-free. Since the prediction :func:`predict` computes is

        y = ys.mean_ + ys.scale_ * (b + sum_j w_j (x_j - xs.mean_[j]) / xs.scale_[j])

    the same line in the original units has slope ``w_j * ys.scale_ /
    xs.scale_[j]`` -- watts per unit of that feature, i.e. per toggle of the bit
    at ``window_size = 1`` and per unit of window density above it -- and
    intercept ``ys.mean_ + ys.scale_ * b - sum_j coef_watts[j] * xs.mean_[j]``.
    ``coef_std`` stays alongside because it is the cross-feature-comparable one.
    """
    w = np.asarray(model.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(model.intercept_).ravel()[0]) if np.size(model.intercept_) else 0.0
    y_mean, y_scale = float(ys.mean_[0]), float(ys.scale_[0])
    x_mean = np.asarray(xs.mean_, dtype=np.float64).ravel()
    x_scale = np.asarray(xs.scale_, dtype=np.float64).ravel()
    coef_watts = w * y_scale / x_scale
    intercept_watts = y_mean + y_scale * b - float(coef_watts @ x_mean)
    return w, coef_watts, intercept_watts
