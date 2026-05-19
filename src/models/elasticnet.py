"""ElasticNetCV wrapper. Replaces the old `src/ridge.py`."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import optuna

from .base import BaseModel, register


@register("ElasticNetCV")
class ElasticNetCVModel(BaseModel):
    backend = "sklearn"
    supports_intercept = True
    supports_non_negative = True

    @classmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        hp: dict[str, Any] = {}

        # n_alphas: pin if scalar in fixed, otherwise search.
        if "n_alphas" in fixed and isinstance(fixed["n_alphas"], int):
            hp["n_alphas"] = int(fixed["n_alphas"])
        else:
            hp["n_alphas"] = trial.suggest_categorical("n_alphas", [10, 20, 50])

        # l1_ratio: a list pins the ElasticNetCV's internal grid; a scalar pins
        # to one value; absence triggers Optuna search over a single value
        # (ElasticNetCV needs at least a list of 1).
        l1 = fixed.get("l1_ratio")
        if isinstance(l1, list):
            hp["l1_ratio"] = [float(v) for v in l1]
        elif isinstance(l1, (int, float)):
            hp["l1_ratio"] = [float(l1)]
        else:
            picked = trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
            hp["l1_ratio"] = [float(picked)]

        hp["cv"] = int(fixed.get("cv", 5))
        return hp

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *, hp: dict[str, Any],
    ) -> "ElasticNetCVModel":
        from sklearn.linear_model import ElasticNetCV

        self.feature_names_ = list(X_train.columns)
        self.model_ = ElasticNetCV(
            l1_ratio=hp["l1_ratio"],
            n_alphas=hp["n_alphas"],
            cv=hp["cv"],
            fit_intercept=self.intercept_on,
            positive=self.non_negative_coef_only,
            random_state=self.seed,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(
                X_train.to_numpy(dtype=np.float64, copy=False),
                y_train.to_numpy(dtype=np.float64, copy=False),
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy(dtype=np.float64, copy=False))

    def importance(self) -> pd.Series:
        return pd.Series(np.abs(self.model_.coef_), index=self.feature_names_)
