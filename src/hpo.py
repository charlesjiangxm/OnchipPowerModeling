"""Optuna-based HPO driver.

Each trial fits the chosen ``BaseModel`` on the training split (val drives
early stopping where supported) and scores R^2 on the held-out val set. The
study is bounded by ``hpo_timeout`` seconds.

Optuna runs trials sequentially (``n_jobs=1``) because the inner sklearn
models already use ``n_jobs=-1`` to saturate all cores per fit, and torch
models share one GPU/MPS device. Parallel trials would oversubscribe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import r2_score

from .models.base import BaseModel


log = logging.getLogger(__name__)


def run_study(
    model_cls: type[BaseModel],
    *,
    train_x: pd.DataFrame, train_y: pd.Series,
    val_x: pd.DataFrame, val_y: pd.Series,
    fixed_hp: dict[str, Any],
    intercept_on: bool,
    non_negative_coef_only: bool,
    timeout: int,
    seed: int,
    device: str,
    hpo_dir: Path,
    verbose: bool = False,
) -> optuna.Study:
    """Run Optuna study; save plots + trials CSV; return the study."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _objective(trial: optuna.Trial) -> float:
        try:
            hp = model_cls.hpo_space(trial, fixed_hp)
            model = model_cls(
                intercept_on=intercept_on,
                non_negative_coef_only=non_negative_coef_only,
                fixed_hp=fixed_hp,
                seed=seed,
                device=device,
                verbose=False,
            )
            model.fit(train_x, train_y, val_x, val_y, hp=hp)
            y_pred = model.predict(val_x)
            score = float(r2_score(val_y.to_numpy(), y_pred))
            if verbose:
                log.info("trial %d: R^2(val)=%.4f  hp=%s", trial.number, score, hp)
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d failed: %s", trial.number, exc)
            return float("-inf")

    study.optimize(
        _objective,
        timeout=int(timeout),
        n_jobs=1,
        show_progress_bar=False,
    )

    _save_study_artifacts(study, hpo_dir)

    if not study.best_trial or study.best_value == float("-inf"):
        raise RuntimeError(
            f"All HPO trials failed (n={len(study.trials)}); check logs above."
        )

    log.info(
        "HPO done: %d trials in <=%ds; best R^2(val)=%.4f; best_params=%s",
        len(study.trials), timeout, study.best_value, study.best_params,
    )
    return study


def best_hp_from_study(
    study: optuna.Study, model_cls: type[BaseModel], fixed_hp: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the best-trial hyperparameters via a FixedTrial replay so
    we get the post-defaults dict (with PIN'd keys from ``fixed_hp`` applied)."""
    fixed_trial = optuna.trial.FixedTrial(study.best_params)
    return model_cls.hpo_space(fixed_trial, fixed_hp)


def _save_study_artifacts(study: optuna.Study, hpo_dir: Path) -> None:
    hpo_dir = Path(hpo_dir)
    hpo_dir.mkdir(parents=True, exist_ok=True)
    try:
        study.trials_dataframe().to_csv(hpo_dir / "trials.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("trials_dataframe export skipped: %s", exc)

    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_param_importances,
        )
        ax = plot_optimization_history(study)
        ax.figure.savefig(
            hpo_dir / "optimization_history.png", dpi=120, bbox_inches="tight",
        )
        ax.figure.clear()
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 2:
            ax = plot_param_importances(study)
            ax.figure.savefig(
                hpo_dir / "param_importances.png", dpi=120, bbox_inches="tight",
            )
            ax.figure.clear()
    except Exception as exc:  # noqa: BLE001
        log.info("Optuna matplotlib plots skipped: %s", exc)
