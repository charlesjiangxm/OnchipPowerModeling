"""Stage-2 regressors: gradient-boosted trees, a two-layer MLP, ridge and RuleFit.

All four consume the same binary MCP-selected proxies and are directly comparable.

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
* ``rulefit`` -- a sparse linear model over ``[linear terms | rule indicators]``
  (Friedman & Popescu eq. 25), from the vendored fork in ``third_party/rulefit``.
  Its selling point here is an auditable term list, not accuracy: ridge already
  reaches test R2 0.9148 on these proxies, because at ``data.window_size > 1``
  every feature is a bit density and window-mean power is nearly additive. The
  rule stage is the fork's OWN sklearn ``GradientBoostingRegressor``, measured at
  ~1 s of a 448 s fit at 68,966 rows x 39 proxies -- so the stage an XGBoost/DART
  bridge would replace is under 1% of the cost, and going library-native keeps
  the fork's tested ``fit``/``predict``/``get_rules``/``get_feature_importance``.
"""

from __future__ import annotations

import warnings

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .utils import log

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


# =========================================================================== #
# rulefit backend (rule ensemble + sparse linear model)
# =========================================================================== #
def _import_rulefit():
    """The vendored fork in ``third_party/rulefit``, imported LAZILY.

    It is a git submodule plus an editable install, not a PyPI package, so a
    module-level import would break ``--build_db``, ``--feature_select`` and
    ``--model tree|nn|ridge`` on any checkout that never ran the install. Every
    rulefit entry point below routes through here.
    """
    try:
        from rulefit import RuleFitRegressor
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the rulefit backend needs the vendored fork: "
            "`git submodule update --init && pip install -e third_party/rulefit`. "
            "It is NOT the PyPI package of the same name, which has no "
            "RuleFitRegressor, allow_negative_coef or get_feature_importance "
            "(and `ordered-set` is the fork's own easily-missed dependency)."
        ) from exc
    return RuleFitRegressor


# Search space for the rule stage: (low, high, log-scale). Only the three knobs
# that change the rule/linear balance this backend exists to expose. Deliberately
# NOT searched: n_alphas/cv/tol/max_iter are cost knobs (a longer alpha path
# cannot score worse, so searching it makes trials incomparable);
# lin_trim_quantile/lin_standardise change what the published coefficients MEAN;
# fit_intercept/non_negative_coef are modelling decisions settled in the config;
# sample_fract stays at the paper's default (the only O(n) lever in the tree
# stage, resolving to 0.024 at n=68,966).
RULEFIT_SPACE = {
    "tree_size": (2, 8, False),  # int
    "max_rules": (100, 1200, True),  # int, log; clamped to rulefit.max_rules
    "memory_par": (1e-3, 0.3, True),
}
_RULEFIT_INT_PARAMS = {"tree_size", "max_rules"}

# Fixed rule-stage point for --no-hpo (mid-space, matching the config defaults).
NOHPO_RULEFIT: dict = {"tree_size": 4, "max_rules": 600, "memory_par": 0.01}


def suggest_rulefit_params(trial, max_rules_cap: int) -> dict:
    """Sample one rule-stage configuration, respecting ``rulefit.max_rules``.

    Mirrors :func:`suggest_params`. ``max_rules`` is clamped to the configured
    cap so a trial can never grow a bigger ensemble than the final fit is allowed
    to; ``lo = min(lo, hi)`` keeps the range from inverting when the cap sits
    below the space's own floor.
    """
    params = {}
    for name, (lo, hi, log) in RULEFIT_SPACE.items():
        if name == "max_rules":
            hi = min(int(hi), int(max_rules_cap))
            lo = min(int(lo), hi)
        if name in _RULEFIT_INT_PARAMS:
            params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log)
    return params


def make_rulefit(rf_cfg, seed: int, **overrides):
    """The SINGLE ``rulefit:`` config -> constructor mapping.

    Both the HPO trials and the final fit go through here, so they cannot drift
    apart. Three translations live only in this function:

    * ``n_jobs: 0`` -> ``None`` (sklearn's "unset"), matching how ``0`` means
      "everything" elsewhere in this package;
    * ``n_alphas: <int>`` -> ``Cs=<int>``, which the fork reads as the length of
      the alpha path. A *sequence* ``Cs`` would instead be inverted to
      ``alphas = 1/Cs``, which is meaningless for this problem -- hence the
      integer guard in ``Config._validate_rulefit``;
    * ``non_negative_coef`` -> ``allow_negative_coef=not it``, passed as an
      explicit bool. ``allow_negative_coef=None`` silently resolves to
      non-negative for regression, so leaving it unset would hide the decision.

    ``penalty`` stays ``"l1"`` rather than becoming a knob: the elastic-net ridge
    term is not scale-neutral across rules (it shrinks a rule of support ``s`` as
    ``1/s`` against L1's ``1/sqrt(s(1-s))``), which puts a support-dependent bias
    into the very eq-(28) importances the term table is ranked by.
    """
    RuleFitRegressor = _import_rulefit()
    params = {"tree_size": rf_cfg.tree_size, "max_rules": rf_cfg.max_rules,
              "memory_par": rf_cfg.memory_par}
    params.update(overrides)
    return RuleFitRegressor(
        rfmode="regress",
        model_type="rl",
        penalty="l1",
        tree_size=int(params["tree_size"]),
        max_rules=int(params["max_rules"]),
        memory_par=float(params["memory_par"]),
        exp_rand_tree_size=bool(rf_cfg.exp_rand_tree_size),
        include_interior_rules=bool(rf_cfg.include_interior_rules),
        lin_standardise=bool(rf_cfg.lin_standardise),
        lin_trim_quantile=float(rf_cfg.lin_trim_quantile),
        fit_intercept=bool(rf_cfg.fit_intercept),
        allow_negative_coef=not bool(rf_cfg.non_negative_coef),
        Cs=int(rf_cfg.n_alphas),
        cv=int(rf_cfg.cv),
        tol=float(rf_cfg.tol),
        max_iter=int(rf_cfg.max_iter),
        n_jobs=(None if int(rf_cfg.n_jobs) == 0 else int(rf_cfg.n_jobs)),
        random_state=int(seed),
    )


def fit_rulefit(rf, X, y, names):
    """Fit ``rf`` on ``(X, y)`` with ``names`` as the feature names.

    The length check comes first because ``RuleFit.fit`` stores ``feature_names``
    with no validation of its own, and a mismatch then surfaces minutes later as
    an opaque pandas error from inside ``get_feature_importance``. This fit costs
    minutes on the real data; nothing about it should fail late.

    Warnings are recorded and re-logged rather than suppressed. ``XGBRuleFit.fit``
    in ``src/xopm_lib/model_regression.py`` wraps its solve in
    ``simplefilter("ignore")``, which also hides the fork's own
    "Every coefficient of the fitted linear model is zero" warning -- the one
    warning that means the published model is a constant.
    """
    names = list(names)
    if len(names) != int(X.shape[1]):
        raise ValueError(f"rulefit: {len(names)} feature names for "
                         f"{X.shape[1]} columns")
    y = np.asarray(y, dtype=np.float64).ravel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rf.fit(X, y, feature_names=names)
    for w in caught:
        log.warning("rulefit: %s: %s", w.category.__name__, w.message)
    return rf


def rulefit_importance(rf, n_features: int) -> np.ndarray:
    """Per-input importance, Friedman & Popescu eq. (35).

    Each rule's eq-(28) importance is shared equally among the variables its
    conditions reference, plus that variable's own eq-(29) linear importance. The
    variables are read off ``Rule.defining_variables()``, never string-matched
    against the rule text -- the bug that once let ``feature_1`` collect
    ``feature_10``'s importance.

    ``_linear_importances`` alone would have been the wrong choice: it ignores
    every rule, so a bit that survives only inside conjunctions would read as
    unimportant, which is backwards for a rule model.

    The shape and sign are ASSERTED rather than trusted, because the failure is
    silent and then fatal: ``utils.save_coefficients_csv`` is called from
    ``run._run_one`` *after* ``model.joblib`` and ``rulefit_terms.csv`` are on
    disk but *before* ``result.json``, so a wrong length there kills every later
    ``-q`` and ``--model`` and leaves a directory ``_aggregate`` skips entirely.
    No ``abs()`` and no ``clip()``: an upstream change must fail a test, not be
    masked.
    """
    imp = np.asarray(rf.get_feature_importance()["importance"],
                     dtype=np.float64).ravel()
    if imp.shape != (int(n_features),):
        raise ValueError(f"rulefit importance has shape {imp.shape}, expected "
                         f"({int(n_features)},)")
    if not np.all(imp >= 0):
        raise ValueError("rulefit importance must be non-negative "
                         "(save_coefficients_csv normalizes and ranks by it)")
    return imp


def rulefit_predict(rf, X, chunk: int) -> np.ndarray:
    """Predict in row batches, in watts.

    Two reasons this is the only prediction path:

    * ``rf.predict`` on an EMPTY ``X`` raises ("Found array with 0 sample(s) ...
      required by LassoCV"), and ``run._run_one`` scores ``splits["test"]``
      unconditionally -- it is reported even when empty.
    * one call materializes ``n_rows x (Q + n_rules)`` float64 several times over
      (``Winsorizer.trim``'s copy and tiles, ``RuleEnsemble.transform``'s scatter,
      ``_design_matrix``'s concatenation): measured 5.33 GB peak for 509,062 rows
      x 604 terms, where batching was also FASTER (1.8 s against 3.3 s).

    Batching is not bit-identical to one call -- measured max abs difference
    1.11e-16 from BLAS blocking, exact only for ``chunk >= n``. Nothing here
    should be compared with ``==``.
    """
    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    chunk = int(chunk)
    if chunk <= 0 or n <= chunk:
        return np.asarray(rf.predict(X), dtype=np.float64).ravel()
    parts = [np.asarray(rf.predict(X[i:i + chunk]), dtype=np.float64).ravel()
             for i in range(0, n, chunk)]
    return np.concatenate(parts)


def rulefit_terms(rf, names, col_ids) -> list[dict]:
    """The non-zero terms of the fit, as rows for ``rulefit_terms.csv``.

    THE ONE PLACE the term -> global-column-id mapping lives, and the one that is
    easy to get wrong. ``get_rules(exclude_zero_coef=True)`` filters with
    ``.loc[]``, so the returned frame keeps its ORIGINAL, now-gapped integer index
    (e.g. ``[0, 1, 2, 3, 4, 6, 10, 11, ...]``) into the fit's
    ``[linear terms | rules]`` column order. Its ``rule`` column is a *string*, so
    ``Rule.defining_variables()`` is not reachable from a row -- the rule object
    has to come from ``rule_ensemble.rules`` by offset. ``enumerate``-ing the
    filtered frame instead would point every rule at the wrong RTL net.

    ``coef_watts`` needs no un-scaling, unlike ridge's: the lasso is fitted
    against raw ``y``, and ``linear_coef_`` already multiplies back through
    ``FriedScale`` (which only ever multiplies, never centres).
    """
    names = list(names)
    col_ids = np.asarray(col_ids).ravel()
    n_lin = int(rf.n_linear_terms_)
    rules = list(getattr(rf.rule_ensemble, "rules", []))
    rows = []
    frame = rf.get_rules(exclude_zero_coef=True, scaled=False)
    for i, row in frame.iterrows():
        i = int(i)
        if i < n_lin:
            ids = [int(col_ids[i])]
        else:
            ids = [int(col_ids[j])
                   for j in sorted(rules[i - n_lin].defining_variables())]
        rows.append({"term": str(row["rule"]), "type": str(row["type"]),
                     "coef_watts": float(row["coef"]),
                     "support": float(row["support"]),
                     "importance": float(row["importance"]),
                     "n_variables": int(row["n_variables"]),
                     "col_ids": ";".join(str(c) for c in ids)})
    # importance desc, ties by |coef| desc -- the save_coefficients_csv idiom.
    # Sorted only AFTER the ids are attached, since sorting destroys the index.
    rows.sort(key=lambda r: (-r["importance"], -abs(r["coef_watts"])))
    return rows


def rulefit_summary(rf) -> dict:
    """Size and readout numbers for ``result.json``'s ``best``.

    Lives here rather than in ``run.py`` so the invariant
    ``n_nonzero_linear + n_nonzero_rules == n_nonzero_terms`` is unit-testable.
    ``n_rules`` is the ACHIEVED count, which is below ``max_rules``: that cap is
    enforced by sizing the ensemble, and identical conjunctions are then merged
    (measured 315 rules from a 600 cap at Q=39).
    """
    coef = np.asarray(rf.coef_, dtype=np.float64).ravel()
    n_lin = int(rf.n_linear_terms_)
    nz = coef != 0
    lscv = getattr(rf, "lscv", None)
    alphas = getattr(lscv, "alphas_", None)
    return {
        "n_rules": int(rf.n_rules_),
        "n_linear_terms": n_lin,
        "n_nonzero_terms": int(nz.sum()),
        "n_nonzero_linear": int(nz[:n_lin].sum()),
        "n_nonzero_rules": int(nz[n_lin:].sum()),
        "alpha": (None if lscv is None or not hasattr(lscv, "alpha_")
                  else float(lscv.alpha_)),
        "n_alphas": (None if alphas is None else int(np.size(alphas))),
        "intercept_watts": float(np.ravel(rf.intercept_)[0])
        if np.size(rf.intercept_) else 0.0,
        "allow_negative_coef": bool(rf.allow_negative_coef_),
    }


def rulefit_interactions(rf, X, top_k: int):
    """Friedman's overall H-statistic for the ``top_k`` most important features.

    A thin pass-through to the fork's closed-form ``interaction_strength``, which
    is exact rather than sampled. Cheap and bounded by ``top_k``: measured 0.2 s
    at 100 features and 1.1 s at 1000 for ``top_k=10``, 0.3 s / 2.0 s at 20.

    The pairwise companion (``interaction_statistics(order=2)``) is deliberately
    not used: candidate pairs come only from rules that reference both features,
    and at 1000 proxies that measured ZERO pairs -- an always-empty figure.
    """
    return rf.interaction_strength(np.asarray(X), top_k=int(top_k))
