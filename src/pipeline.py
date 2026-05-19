"""End-to-end pipeline orchestrator.

Reads a YAML config, builds the train/val/test splits, runs preprocessing
and feature selection, drives Optuna HPO, refits with the best params,
emits artifacts, and writes report.md.

Outputs land under ``<repo_root>/output/<config_stem>_<timestamp>/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from . import data, hpo, plotting, preprocess, report
from .config import Config, load_config
from .feature_selectors import FeatureSelector
from .models import MODEL_REGISTRY
from .rulefit_utils import compute_metrics


def run(config_path: str | Path, output_root: str | Path | None = None) -> int:
    cfg = load_config(config_path, output_root=output_root)
    log = _setup_logging(cfg.run_dir)
    log.info("=== fit run ===")
    log.info("config:  %s", cfg.config_path)
    log.info("run dir: %s", cfg.run_dir)
    _seed_everything(int(cfg.general["seed"]))

    # 1. Load + pair
    train_bms, test_bms = data.load_pairs(
        cfg.general["trainset_x_path"],
        cfg.general["testset_x_path"],
        cfg.general["y_path"],
        cfg.general["y_label"],
    )

    # 2. Per-benchmark split
    splits = data.split_per_benchmark(
        train_bms, test_bms,
        tuple(cfg.preprocessing["train_val_test_ratio"]),
        seed=int(cfg.general["seed"]),
    )
    counts = {"loaded": _counts(splits.train_x, splits.val_x, splits.test_x)}

    # 3. Zero-variance drop
    if cfg.preprocessing["drop_zero_var"]:
        tr_x, va_x, te_x = preprocess.drop_zero_variance(
            splits.train_x, splits.val_x, splits.test_x,
        )
    else:
        tr_x, va_x, te_x = splits.train_x, splits.val_x, splits.test_x
    tr_y, va_y, te_y = splits.train_y, splits.val_y, splits.test_y

    # 4. Non-overlapped window averaging
    wsize = int(cfg.preprocessing["avg_wsize"])
    tr_x, tr_y = data.avg_window(tr_x, tr_y, wsize)
    if len(va_x):
        va_x, va_y = data.avg_window(va_x, va_y, wsize)
    if len(te_x):
        te_x, te_y = data.avg_window(te_x, te_y, wsize)

    data.assert_finite(tr_x, "train_x")
    if len(va_x):
        data.assert_finite(va_x, "val_x")
    if len(te_x):
        data.assert_finite(te_x, "test_x")

    # 5. Standardize on train, apply to val/test
    tr_xz, tr_yz, va_xz, va_yz, te_xz, te_yz, std = (
        preprocess.standardize_train_apply(tr_x, tr_y, va_x, va_y, te_x, te_y)
    )
    counts["after_preprocess"] = _counts(tr_xz, va_xz, te_xz)

    # 6. Save preprocessed splits + standardizer
    preprocess.save_splits(cfg.run_dir, tr_xz, tr_yz, va_xz, va_yz, te_xz, te_yz)
    preprocess.save_standardizer(cfg.run_dir, std)

    # 7. Feature selection
    fs_alg = cfg.feature_selection["algorithm"]
    if fs_alg:
        selector = FeatureSelector.from_config_dict(
            method=fs_alg,
            top_k=int(cfg.feature_selection["top_k"]),
            hyperparams=cfg.feature_selection.get("hyperparams", {}),
            seed=int(cfg.general["seed"]),
            verbose=True,
        )
        kept = selector.fit_select(tr_xz, tr_yz)
        log.info("feature_selection(%s) kept %d / %d cols",
                 fs_alg, len(kept), tr_xz.shape[1])
        tr_xz, va_xz, te_xz = tr_xz[kept], va_xz[kept], te_xz[kept]
        fs_dir = cfg.run_dir / "feature_selection"
        preprocess.save_splits(fs_dir, tr_xz, tr_yz, va_xz, va_yz, te_xz, te_yz)
    counts["after_feature_selection"] = _counts(tr_xz, va_xz, te_xz)

    # 8. HPO
    alg = cfg.regression["algorithm"]
    model_cls = MODEL_REGISTRY[alg]
    fixed_hp = (cfg.regression.get("hyperparams") or {}).get(alg, {}) or {}
    log.info("HPO: %s for up to %ds", alg, int(cfg.regression["hpo_timeout"]))

    study = hpo.run_study(
        model_cls,
        train_x=tr_xz, train_y=tr_yz,
        val_x=va_xz, val_y=va_yz,
        fixed_hp=fixed_hp,
        intercept_on=bool(cfg.regression["intercept_on"]),
        non_negative_coef_only=bool(cfg.regression["non_negative_coef_only"]),
        timeout=int(cfg.regression["hpo_timeout"]),
        seed=int(cfg.general["seed"]),
        device="auto",
        hpo_dir=cfg.run_dir / "hpo",
    )
    best_hp = hpo.best_hp_from_study(study, model_cls, fixed_hp)
    log.info("HPO best R^2(val)=%.4f  params=%s", study.best_value, best_hp)

    # 9. Final refit on train (val drives early-stopping where supported)
    model = model_cls(
        intercept_on=bool(cfg.regression["intercept_on"]),
        non_negative_coef_only=bool(cfg.regression["non_negative_coef_only"]),
        fixed_hp=fixed_hp,
        seed=int(cfg.general["seed"]),
        device="auto",
        verbose=True,
    )
    model.fit(tr_xz, tr_yz, va_xz, va_yz, hp=best_hp)

    # 10. Metrics in original y units
    tr_true = std.inverse_y(tr_yz.to_numpy())
    tr_pred = std.inverse_y(model.predict(tr_xz))
    metrics = {"train": compute_metrics(tr_true, tr_pred)}
    if len(va_yz):
        va_true = std.inverse_y(va_yz.to_numpy())
        va_pred = std.inverse_y(model.predict(va_xz))
        metrics["val"] = compute_metrics(va_true, va_pred)
    else:
        va_true, va_pred = np.zeros(0), np.zeros(0)
    if len(te_yz):
        te_true = std.inverse_y(te_yz.to_numpy())
        te_pred = std.inverse_y(model.predict(te_xz))
        metrics["test"] = compute_metrics(te_true, te_pred)
    else:
        te_true, te_pred = np.zeros(0), np.zeros(0)

    for split, m in metrics.items():
        log.info(
            "%s: sMAPE=%.3f%%  MAPE=%.3f%%  RMSE=%.5f  MAE=%.5f  R^2=%.4f",
            split, m["smape"], m["mape"], m["rmse"], m["mae"], m["r2"],
        )

    # 11. Artifacts
    artifacts_dir = cfg.run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    extras: dict[str, str] = {}

    plotting.plot_pred_vs_true(
        tr_true, tr_pred, artifacts_dir / "pred_vs_true_train.png",
        title=f"{alg} — pred vs true (train)",
    )
    figures["pred_vs_true_train"] = "artifacts/pred_vs_true_train.png"
    if len(te_yz):
        plotting.plot_pred_vs_true(
            te_true, te_pred, artifacts_dir / "pred_vs_true_test.png",
            title=f"{alg} — pred vs true (test)",
        )
        figures["pred_vs_true_test"] = "artifacts/pred_vs_true_test.png"

    imp = model.importance().sort_values(ascending=False)
    imp.to_csv(artifacts_dir / "feature_importance.csv")
    drawn = plotting.plot_top_features(
        imp, artifacts_dir / "top_features.png",
        top_n=20, title=f"{alg} — top 20 features",
    )
    figures["top_features"] = "artifacts/top_features.png"

    inter = model.interaction_matrix(tr_xz, list(drawn.index))
    if inter is not None and inter.shape[0] > 0:
        plotting.plot_interaction_heatmap(
            inter, artifacts_dir / "interaction_heatmap.png",
            title=f"{alg} — feature interactions (top 20)",
        )
        inter.to_csv(artifacts_dir / "interaction_heatmap.csv")
        figures["interaction_heatmap"] = "artifacts/interaction_heatmap.png"
    else:
        extras["Interaction heatmap"] = (
            f"Not produced for **{alg}** (interaction extraction not defined "
            f"or unavailable in this environment)."
        )

    history = model.convergence_history()
    if history:
        plotting.plot_convergence(history, artifacts_dir / "convergence.png")
        figures["convergence"] = "artifacts/convergence.png"

    model.save_extra(cfg.run_dir)

    hpo_dir = cfg.run_dir / "hpo"
    if (hpo_dir / "optimization_history.png").exists():
        figures["hpo_optimization_history"] = "hpo/optimization_history.png"
    if (hpo_dir / "param_importances.png").exists():
        figures["hpo_param_importances"] = "hpo/param_importances.png"

    # 12. Report
    report.write_report(
        cfg.run_dir,
        config_summary={
            "config_path": str(cfg.config_path),
            "algorithm": alg,
            "fs_alg": fs_alg or "none",
            "top_k": cfg.feature_selection.get("top_k"),
            "seed": cfg.general.get("seed"),
        },
        counts=counts,
        metrics=metrics,
        best_hp=best_hp,
        figures=figures,
        extras=extras,
    )
    log.info("done: %s", cfg.run_dir / "report.md")
    return 0


def _counts(train_x, val_x, test_x) -> dict[str, int]:
    return {
        "train_rows": int(len(train_x)),
        "val_rows": int(len(val_x)),
        "test_rows": int(len(test_x)),
        "features": int(train_x.shape[1]) if hasattr(train_x, "shape") else 0,
    }


def _setup_logging(run_dir: Path) -> logging.Logger:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    file_h = logging.FileHandler(run_dir / "fit.log", mode="w")
    file_h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    )
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_h)
    root.addHandler(stream_h)
    root.setLevel(logging.INFO)
    return logging.getLogger("pipeline")


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
