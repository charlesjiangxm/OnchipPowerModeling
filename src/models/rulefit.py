"""RuleFit wrapper. Uses the PyPI ``rulefit`` package.

When run inside the ``src.models`` subpackage, ``from rulefit import RuleFit``
resolves to the PyPI top-level package (not this module), because absolute
imports skip the local package hierarchy. The shadowing problem that the old
top-level ``src/rulefit.py`` had is therefore avoided.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna

from .base import BaseModel, register
from ..rulefit_utils import parse_rule_features


@register("RuleFit")
class RuleFitModel(BaseModel):
    backend = "sklearn"
    supports_intercept = True
    supports_non_negative = True

    @classmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        hp: dict[str, Any] = {}
        if isinstance(fixed.get("tree_size"), int):
            hp["tree_size"] = int(fixed["tree_size"])
        else:
            hp["tree_size"] = trial.suggest_int("tree_size", 2, 5)
        if isinstance(fixed.get("max_rules"), int):
            hp["max_rules"] = int(fixed["max_rules"])
        else:
            hp["max_rules"] = trial.suggest_categorical("max_rules", [500, 1000, 2000, 4000])
        if isinstance(fixed.get("memory_par"), (int, float)):
            hp["memory_par"] = float(fixed["memory_par"])
        else:
            hp["memory_par"] = trial.suggest_float("memory_par", 1e-3, 1e-1, log=True)
        return hp

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *, hp: dict[str, Any],
    ) -> "RuleFitModel":
        from rulefit import RuleFit  # PyPI

        self.feature_names_ = list(X_train.columns)
        X_arr = X_train.to_numpy(dtype=np.float64, copy=False)
        y_arr = y_train.to_numpy(dtype=np.float64, copy=False)

        self.model_ = RuleFit(
            tree_size=int(hp["tree_size"]),
            max_rules=int(hp["max_rules"]),
            memory_par=float(hp["memory_par"]),
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_arr, y_arr, feature_names=self.feature_names_)

        # The internal LassoCV always fits an intercept and allows signed
        # coefficients. Refit only when the user asked for non-defaults.
        if (not self.intercept_on) or self.non_negative_coef_only:
            self._refit_lasso(X_arr, y_arr)
        return self

    def _refit_lasso(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
        from sklearn.linear_model import LassoCV
        rf = self.model_
        X_concat = np.zeros((X_arr.shape[0], 0))
        if "l" in rf.model_type:
            if getattr(rf, "lin_standardise", False):
                X_regn = rf.friedscale.scale(X_arr)
            else:
                X_regn = X_arr.copy()
            X_concat = np.concatenate((X_concat, X_regn), axis=1)
        if "r" in rf.model_type and len(rf.rule_ensemble.rules) > 0:
            X_rules = rf.rule_ensemble.transform(X_arr)
            if X_rules.size and X_rules.shape[1] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
        lscv = LassoCV(
            n_alphas=100, cv=3, random_state=self.seed,
            positive=self.non_negative_coef_only,
            fit_intercept=self.intercept_on,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lscv.fit(X_concat, y_arr)
        rf.lscv = lscv
        rf.coef_ = lscv.coef_
        rf.intercept_ = float(getattr(lscv, "intercept_", 0.0) or 0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy(dtype=np.float64, copy=False))

    def importance(self) -> pd.Series:
        rules = self.model_.get_rules()
        nz = rules[rules["coef"] != 0]
        scores = pd.Series(0.0, index=self.feature_names_)
        for _, r in nz.iterrows():
            feats = parse_rule_features(r["rule"], r["type"], self.feature_names_)
            if not feats:
                continue
            share = abs(float(r["coef"])) * float(r.get("support", 1.0)) / len(feats)
            for f in feats:
                scores.loc[f] += share
        return scores

    def interaction_matrix(
        self,
        X: pd.DataFrame,
        top_features: list[str],
    ) -> pd.DataFrame | None:
        if not top_features:
            return None
        K = len(top_features)
        mat = np.zeros((K, K), dtype=np.float64)
        idx = {f: i for i, f in enumerate(top_features)}
        top_set = set(top_features)
        rules = self.model_.get_rules()
        rules = rules[(rules["coef"] != 0) & (rules["type"] == "rule")]
        for _, r in rules.iterrows():
            feats = [
                f for f in parse_rule_features(r["rule"], r["type"], self.feature_names_)
                if f in top_set
            ]
            if len(feats) < 2:
                continue
            weight = abs(float(r["coef"])) * float(r.get("support", 1.0))
            for i, fi in enumerate(feats):
                for fj in feats[i + 1:]:
                    a, b = idx[fi], idx[fj]
                    mat[a, b] += weight
                    mat[b, a] += weight
        return pd.DataFrame(mat, index=top_features, columns=top_features)

    def save_extra(self, run_dir: Path) -> None:
        rules = self.model_.get_rules()
        rules.to_csv(Path(run_dir) / "rules.csv", index=False)
