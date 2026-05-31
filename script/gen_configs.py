#!/usr/bin/env python
"""Generate YAML configs for the power-modeling sweep.

Writes <feature-folder> x <fs_method> x <model> YAML files into ../configs/.
Existing YAML configs are removed first so the directory mirrors the current
db layout exactly.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DB_DIR = REPO_ROOT / "db"

BLOCK_TARGETS = {
    "cp0": "x_aq_core/Pc(x_aq_cp0_top)",
    "idu": "x_aq_core/Pc(x_aq_idu_top)",
    "ifu": "x_aq_core/Pc(x_aq_ifu_top)",
    "iu": "x_aq_core/Pc(x_aq_iu_top)",
    "lsu": "x_aq_core/Pc(x_aq_lsu_top)",
    "rtu": "x_aq_core/Pc(x_aq_rtu_top)",
    "vidu": "x_aq_core/Pc(x_aq_vidu_top)",
    "vpu": "x_aq_core/Pc(x_aq_vpu_top)",
}

KINDS = ("input", "net", "output")

BENCHMARKS = [
    "ISA_FP",
    "ISA_INT",
    "ISA_LS",
    "ISA_THEAD",
    "MMU",
    "cache",
    "conv_softmax",
    "conv_softmax_part000_c906_float16",
    "conv_softmax_part000_c906_float32",
    "conv_softmax_part000_c906_int8_sym",
    "coremark",
    "csr",
    "debug",
    "exception",
    "interrupt",
]

FS_METHODS = [
    "pearson",
    "variance",
    "univariate",
    "rfe",
    "from_model",
    "sequential",
    "mcp",
    "deep",
]

MODELS = {
    "RuleFit": "rulefit",
    "FT-Transformer": "ft_transformer",
    "GBDT": "gbdt",
    "MLP": "mlp",
    "ElasticNetCV": "elasticnet",
}

TOP_K = 20
AVG_WSIZE = 128
HPO_TIMEOUT = 300
SEED = 42
RATIO = [0.8, 0.1, 0.1]


def feature_folders() -> list[tuple[str, str, Path]]:
    """Return (block, folder_name, relative_path) entries in deterministic order."""
    folders: list[tuple[str, str, Path]] = []
    for block in BLOCK_TARGETS:
        for kind in KINDS:
            folder = f"{block}_{kind}"
            rel_path = Path("db") / block / folder
            if not (REPO_ROOT / rel_path).is_dir():
                raise FileNotFoundError(f"feature folder not found: {REPO_ROOT / rel_path}")
            folders.append((block, folder, rel_path))
    return folders


def build_config(block: str, rel_folder: Path, fs_method: str, model: str) -> dict:
    trainset = [str(rel_folder / f"{stem}_func.pkl") for stem in BENCHMARKS]
    y_paths = [f"db/pwr/{stem}_pwr.pkl" for stem in BENCHMARKS]
    return {
        "general": {
            "trainset_x_path": trainset,
            "testset_x_path": [],
            "y_path": y_paths,
            "y_label": BLOCK_TARGETS[block],
            "seed": SEED,
        },
        "preprocessing": {
            "drop_zero_var": True,
            "train_val_test_ratio": list(RATIO),
            "avg_wsize": AVG_WSIZE,
        },
        "feature_selection": {
            "algorithm": fs_method,
            "top_k": TOP_K,
            "hyperparams": {fs_method: {}},
        },
        "regression": {
            "algorithm": model,
            "intercept_on": True,
            "non_negative_coef_only": True,
            "hpo_timeout": HPO_TIMEOUT,
            "hyperparams": {model: {}},
        },
    }


def remove_existing_configs() -> int:
    count = 0
    for path in CONFIGS_DIR.glob("*.yaml"):
        path.unlink()
        count += 1
    return count


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    removed = remove_existing_configs()
    count = 0
    for block, folder, rel_folder in feature_folders():
        for fs_method in FS_METHODS:
            for model, slug in MODELS.items():
                cfg = build_config(block, rel_folder, fs_method, model)
                out = CONFIGS_DIR / f"{folder}_{fs_method}_{slug}.yaml"
                with out.open("w") as fh:
                    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
                count += 1
    print(f"removed {removed} existing configs")
    print(f"wrote {count} configs to {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
