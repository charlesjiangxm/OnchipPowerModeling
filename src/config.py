"""YAML config loader, schema validation, and run-directory bookkeeping.

The runner reads a YAML config file shaped like ``configs/example.yaml`` and
this module validates it, applies defaults, and creates an output directory
under the repo root at ``output/<config_stem>_<timestamp>/`` (override with
``--output-root`` on the runner).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ -> repo root

VALID_FS_METHODS = {
    None, "pearson", "variance", "univariate", "rfe",
    "from_model", "sequential", "mcp", "deep",
}
VALID_RGR = {"RuleFit", "FT-Transformer", "GBDT", "MLP", "ElasticNetCV"}


@dataclass
class Config:
    general: dict
    preprocessing: dict
    feature_selection: dict
    regression: dict
    config_path: Path
    run_dir: Path


def load_config(config_path: str | Path, output_root: str | Path | None = None) -> Config:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: YAML root must be a mapping")

    for section in ("general", "preprocessing", "feature_selection", "regression"):
        if section not in raw:
            raise ValueError(f"{config_path}: missing section '{section}'")

    _validate_general(raw["general"], config_path)
    _validate_preprocessing(raw["preprocessing"], config_path)
    _validate_feature_selection(raw["feature_selection"], config_path)
    _validate_regression(raw["regression"], config_path)

    output_root = Path(output_root).resolve() if output_root else REPO_ROOT / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{config_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        general=raw["general"],
        preprocessing=raw["preprocessing"],
        feature_selection=raw["feature_selection"],
        regression=raw["regression"],
        config_path=config_path,
        run_dir=run_dir,
    )
    _dump_resolved(cfg)
    return cfg


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    """Resolve a config-relative path against the repo root (or `base`)."""
    p = Path(p)
    if p.is_absolute():
        return p
    return ((base or REPO_ROOT) / p).resolve()


def _dump_resolved(cfg: Config) -> None:
    out = {
        "general": cfg.general,
        "preprocessing": cfg.preprocessing,
        "feature_selection": cfg.feature_selection,
        "regression": cfg.regression,
        "_meta": {
            "config_path": str(cfg.config_path),
            "run_dir": str(cfg.run_dir),
            "repo_root": str(REPO_ROOT),
            "resolved_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(cfg.run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)


def _require(d: dict, key: str, where: str, types: type | tuple) -> Any:
    if key not in d:
        raise ValueError(f"{where}: missing required key '{key}'")
    if not isinstance(d[key], types):
        raise ValueError(
            f"{where}: '{key}' must be {types}, got {type(d[key]).__name__}"
        )
    return d[key]


def _validate_general(g: dict, cfg_path: Path) -> None:
    where = f"{cfg_path}: general"
    _require(g, "trainset_x_path", where, list)
    _require(g, "testset_x_path", where, list)
    _require(g, "y_path", where, list)
    _require(g, "y_label", where, str)
    _require(g, "seed", where, int)
    if not g["trainset_x_path"]:
        raise ValueError(f"{where}: trainset_x_path must be non-empty")
    if not g["y_path"]:
        raise ValueError(f"{where}: y_path must be non-empty")


def _validate_preprocessing(p: dict, cfg_path: Path) -> None:
    where = f"{cfg_path}: preprocessing"
    _require(p, "drop_zero_var", where, bool)
    _require(p, "train_val_test_ratio", where, (list, tuple))
    _require(p, "avg_wsize", where, int)
    r = p["train_val_test_ratio"]
    if len(r) != 3 or not all(isinstance(x, (int, float)) for x in r):
        raise ValueError(f"{where}: train_val_test_ratio must be 3 floats")
    if not all(0 <= x <= 1 for x in r):
        raise ValueError(f"{where}: each ratio must be in [0, 1]")
    if abs(sum(r) - 1.0) > 1e-6:
        raise ValueError(f"{where}: train_val_test_ratio must sum to 1, got {sum(r)}")
    if r[0] <= 0:
        raise ValueError(f"{where}: train ratio must be > 0")
    if p["avg_wsize"] < 1:
        raise ValueError(f"{where}: avg_wsize must be > 0")


def _validate_feature_selection(fs: dict, cfg_path: Path) -> None:
    where = f"{cfg_path}: feature_selection"
    if "algorithm" not in fs:
        raise ValueError(f"{where}: missing 'algorithm' (use null to bypass)")
    if fs["algorithm"] not in VALID_FS_METHODS:
        raise ValueError(
            f"{where}: algorithm={fs['algorithm']!r} not in {VALID_FS_METHODS}"
        )
    if fs["algorithm"] is not None:
        _require(fs, "top_k", where, int)
        if fs["top_k"] < 1:
            raise ValueError(f"{where}: top_k must be > 0")
    if "hyperparams" in fs and not isinstance(fs["hyperparams"], dict):
        raise ValueError(f"{where}: hyperparams must be a mapping")


def _validate_regression(r: dict, cfg_path: Path) -> None:
    where = f"{cfg_path}: regression"
    _require(r, "algorithm", where, str)
    if r["algorithm"] not in VALID_RGR:
        raise ValueError(
            f"{where}: algorithm={r['algorithm']!r} not in {VALID_RGR}"
        )
    _require(r, "intercept_on", where, bool)
    _require(r, "non_negative_coef_only", where, bool)
    _require(r, "hpo_timeout", where, int)
    if r["hpo_timeout"] < 1:
        raise ValueError(f"{where}: hpo_timeout must be > 0")
    if "hyperparams" in r and not isinstance(r["hyperparams"], dict):
        raise ValueError(f"{where}: hyperparams must be a mapping")
