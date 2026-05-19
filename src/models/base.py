"""BaseModel abstract class + registry.

Each subclass wraps one regression algorithm with a uniform interface so the
pipeline orchestrator and the Optuna driver in ``src/hpo.py`` can drive any
algorithm identically. Models receive pre-standardized X/y (z-scored on the
training split) and predict in z-scored space; the pipeline inverse-transforms
before computing physical-unit metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import optuna


MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}


def register(name: str):
    """Class decorator: add a BaseModel subclass to the registry by name."""
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls
    return _wrap


class BaseModel(ABC):
    name: ClassVar[str] = "base"
    backend: ClassVar[str] = "sklearn"   # "sklearn" or "torch" (controls Optuna n_jobs)
    supports_intercept: ClassVar[bool] = False
    supports_non_negative: ClassVar[bool] = False

    def __init__(
        self,
        *,
        intercept_on: bool = False,
        non_negative_coef_only: bool = False,
        fixed_hp: dict[str, Any] | None = None,
        seed: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ):
        if intercept_on and not self.supports_intercept:
            # honor the spec: silently ignore for models that don't support it,
            # but log it once.
            import logging
            logging.getLogger(__name__).info(
                "%s does not honor intercept_on; ignoring.", self.name,
            )
        if non_negative_coef_only and not self.supports_non_negative:
            import logging
            logging.getLogger(__name__).info(
                "%s does not honor non_negative_coef_only; ignoring.", self.name,
            )
        self.intercept_on = intercept_on
        self.non_negative_coef_only = non_negative_coef_only
        self.fixed_hp = fixed_hp or {}
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.feature_names_: list[str] | None = None

    @classmethod
    @abstractmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        """Suggest hyperparameters for one Optuna trial.

        ``fixed`` is the user's per-algorithm hyperparams dict from the YAML;
        scalar entries PIN a key (skip its search dimension). Returns the
        resolved hyperparameters as a plain dict for ``fit()``.
        """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *, hp: dict[str, Any],
    ) -> "BaseModel":
        """Fit on train (val used for early-stopping where supported)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict in z-scored space (pipeline inverse-transforms downstream)."""

    @abstractmethod
    def importance(self) -> pd.Series:
        """Per-feature non-negative importance score, indexed by feature name."""

    def interaction_matrix(
        self,
        X: pd.DataFrame,
        top_features: list[str],
    ) -> pd.DataFrame | None:
        """Return a (K, K) DataFrame of pairwise interaction strengths over
        the given ``top_features`` (in that order). Default: not applicable."""
        return None

    def convergence_history(self) -> dict[str, list[float]] | None:
        """Train/val loss-vs-iteration if the algorithm has one. Default: None."""
        return None

    def save_extra(self, run_dir: Path) -> None:
        """Hook for model-specific artifacts (e.g. RuleFit's rules.csv)."""
        return None
