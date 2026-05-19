"""Shared utility helpers retained from the original pipeline:
- ``compute_metrics``: RMSE / MAE / MAPE / sMAPE / R^2 on a single target.
- ``parse_rule_features``: RuleFit rule-string -> referenced feature names.
- ``refit_nonneg_lasso``: in-place positive-coefficient refit of a fitted
  RuleFit's internal Lasso (used by ``non_negative_coef_only=True``).

The previous c906-specific dataset loaders (``load_c906_pair``, ``load_all``,
``validate_presim_subdir``, the ``PREFIXES``/``TARGET_COL`` constants) have
been removed: the new pipeline takes explicit pkl paths from the YAML config
and matches them in ``src/data.py``.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import r2_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Single-target RMSE / MAE / MAPE(%) / sMAPE(%) / R^2.

    All metrics are computed in whatever scale ``y_true`` and ``y_pred`` are
    passed in. The pipeline always inverse-transforms before calling this
    helper so units stay physical.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(
        np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)) * 100
    )
    smape = float(
        np.mean(
            2.0 * np.abs(y_true - y_pred)
            / (np.abs(y_true) + np.abs(y_pred) + 1e-12)
        ) * 100
    )
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2}


# ---------------------------------------------------------------------------
# Rule-string parsing for interaction / feature-level analysis
# ---------------------------------------------------------------------------

_COND_RE = re.compile(r"\s*(<=|>=|<|>|==|!=)\s*-?\d+\.?\d*(?:[eE][+-]?\d+)?\s*$")


def parse_rule_features(rule_str: str, rule_type: str, feature_names) -> list[str]:
    """Return the list of feature names referenced in a RuleFit rule.

    - linear terms -> [rule_str] (the feature itself)
    - rule (conjunction) terms -> features extracted from each conjunct
    """
    if rule_type == "linear":
        return [rule_str] if rule_str in set(feature_names) else []

    conjuncts = re.split(r"\s+&\s+|\s+and\s+", rule_str)
    feat_set = set(feature_names)
    out: list[str] = []
    for c in conjuncts:
        name = _COND_RE.sub("", c).strip()
        if name in feat_set and name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Non-negative refit of RuleFit's internal Lasso
# ---------------------------------------------------------------------------

def refit_nonneg_lasso(rf, X, y, random_state=42, n_alphas=100, cv=3):
    """Re-fit RuleFit's internal LassoCV with ``positive=True``.

    Rebuilds the same design matrix RuleFit fits internally (linear features
    optionally Friedman-standardized + rule indicators) and refits LassoCV
    with ``positive=True`` so every linear and rule coefficient is constrained
    to be >= 0 -- useful when each rule should represent an additive
    switching-event contribution to power and negative coefficients are
    physically suspect.

    Replaces ``rf.lscv``, ``rf.coef_``, ``rf.intercept_`` in place so
    ``predict()``, ``get_rules()``, and the importance scores derived from
    them all reflect the new coefficients.
    """
    from sklearn.linear_model import LassoCV

    X_arr = np.asarray(X, dtype=np.float64)
    X_concat = np.zeros((X_arr.shape[0], 0))
    if "l" in rf.model_type:
        if getattr(rf, "lin_standardise", False):
            X_regn = rf.friedscale.scale(X_arr)
        else:
            X_regn = X_arr.copy()
        X_concat = np.concatenate((X_concat, X_regn), axis=1)
    if "r" in rf.model_type and len(rf.rule_ensemble.rules) > 0:
        X_rules = rf.rule_ensemble.transform(X_arr)
        if X_rules.size > 0 and X_rules.shape[1] > 0:
            X_concat = np.concatenate((X_concat, X_rules), axis=1)

    lscv = LassoCV(
        n_alphas=n_alphas, cv=cv, random_state=random_state, positive=True,
    )
    lscv.fit(X_concat, np.asarray(y, dtype=np.float64))
    rf.lscv = lscv
    rf.coef_ = lscv.coef_
    rf.intercept_ = lscv.intercept_
    return rf
