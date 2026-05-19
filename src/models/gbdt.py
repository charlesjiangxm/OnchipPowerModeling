"""GBDT (XGBoost) wrapper. Uses val for early stopping and SHAP for
feature-interaction heatmaps (when ``shap`` is installed; otherwise the
interaction matrix is omitted)."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import optuna

from .base import BaseModel, register


log = logging.getLogger(__name__)


@register("GBDT")
class GBDTModel(BaseModel):
    backend = "sklearn"
    supports_intercept = False
    supports_non_negative = False

    @classmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        hp: dict[str, Any] = dict(fixed) if fixed else {}
        if not isinstance(hp.get("n_estimators"), int):
            hp["n_estimators"] = trial.suggest_int("n_estimators", 100, 1500)
        if not isinstance(hp.get("max_depth"), int):
            hp["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        if not isinstance(hp.get("learning_rate"), (int, float)) or isinstance(hp.get("learning_rate"), bool):
            hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        if not isinstance(hp.get("subsample"), (int, float)) or isinstance(hp.get("subsample"), bool):
            hp["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        if not isinstance(hp.get("colsample_bytree"), (int, float)) or isinstance(hp.get("colsample_bytree"), bool):
            hp["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        hp.setdefault("tree_method", "hist")
        hp.setdefault("n_jobs", -1)
        hp.setdefault("early_stopping_rounds", 30)
        return hp

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *, hp: dict[str, Any],
    ) -> "GBDTModel":
        import xgboost as xgb

        self.feature_names_ = list(X_train.columns)
        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
        ytr = y_train.to_numpy(dtype=np.float64, copy=False)

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        early = int(hp.get("early_stopping_rounds", 30)) if use_val else None

        self.model_ = xgb.XGBRegressor(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            learning_rate=float(hp["learning_rate"]),
            subsample=float(hp["subsample"]),
            colsample_bytree=float(hp["colsample_bytree"]),
            tree_method=str(hp["tree_method"]),
            n_jobs=int(hp["n_jobs"]),
            random_state=self.seed,
            verbosity=0,
            early_stopping_rounds=early,
        )

        eval_set = None
        if use_val:
            eval_set = [
                (Xtr, ytr),
                (X_val.to_numpy(dtype=np.float64, copy=False),
                 y_val.to_numpy(dtype=np.float64, copy=False)),
            ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy(dtype=np.float64, copy=False))

    def importance(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.feature_names_)

    def convergence_history(self) -> dict[str, list[float]] | None:
        evals = getattr(self.model_, "evals_result_", None)
        if not evals:
            return None
        out: dict[str, list[float]] = {}
        keys = list(evals.keys())
        if keys:
            metric_tr = next(iter(evals[keys[0]].keys()))
            out["train_loss"] = list(evals[keys[0]][metric_tr])
        if len(keys) > 1:
            metric_va = next(iter(evals[keys[1]].keys()))
            out["val_loss"] = list(evals[keys[1]][metric_va])
        return out or None

    def interaction_matrix(
        self,
        X: pd.DataFrame,
        top_features: list[str],
    ) -> pd.DataFrame | None:
        try:
            import shap
        except ImportError:
            log.info("GBDT interaction heatmap: shap not installed; skipping.")
            return None
        if not top_features:
            return None
        n_sample = min(500, len(X))
        if n_sample == 0:
            return None
        rng = np.random.default_rng(self.seed)
        sample_idx = rng.choice(len(X), size=n_sample, replace=False)
        X_full = X.iloc[sample_idx].to_numpy(dtype=np.float64, copy=False)
        try:
            explainer = shap.TreeExplainer(self.model_)
            inter = explainer.shap_interaction_values(X_full)
        except Exception as exc:  # noqa: BLE001 — shap is finicky on some configs
            log.warning("shap_interaction_values failed: %s", exc)
            return None
        if isinstance(inter, list):
            inter = inter[0]
        top_idx = np.array(
            [self.feature_names_.index(f) for f in top_features], dtype=int,
        )
        inter_top = inter[:, top_idx, :][:, :, top_idx]
        mean_abs = np.abs(inter_top).mean(axis=0)
        np.fill_diagonal(mean_abs, 0.0)
        return pd.DataFrame(mean_abs, index=top_features, columns=top_features)
