#!/usr/bin/env python
"""Generate YAML configs for the IDU/IFU power-modeling sweep.

Writes <folder> x <fs_method> x <model> YAML files into ../configs/.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

FOLDERS = [
    "idu_input",
    "idu_net",
    "idu_output",
    "ifu_input",
    "ifu_net",
    "ifu_output",
]

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

IDU_TARGET = "/openC906/Pc(x_aq_top_0_x_aq_core_x_aq_idu_top)"
IFU_TARGET = "/openC906/Pc(x_aq_top_0_x_aq_core_x_aq_ifu_top)"

TOP_K = 20
AVG_WSIZE = 128
HPO_TIMEOUT = 300
SEED = 42
RATIO = [0.8, 0.1, 0.1]


def y_label_for(folder: str) -> str:
    if folder.startswith("idu"):
        return IDU_TARGET
    if folder.startswith("ifu"):
        return IFU_TARGET
    raise ValueError(f"unexpected folder: {folder}")


def build_config(folder: str, fs_method: str, model: str) -> dict:
    trainset = [f"db/{folder}/{stem}_func.pkl" for stem in BENCHMARKS]
    y_paths = [f"db/pwr/{stem}_pwr.pkl" for stem in BENCHMARKS]
    return {
        "general": {
            "trainset_x_path": trainset,
            "testset_x_path": [],
            "y_path": y_paths,
            "y_label": y_label_for(folder),
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


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder in FOLDERS:
        for fs_method in FS_METHODS:
            for model, slug in MODELS.items():
                cfg = build_config(folder, fs_method, model)
                out = CONFIGS_DIR / f"{folder}_{fs_method}_{slug}.yaml"
                with out.open("w") as fh:
                    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
                count += 1
    print(f"wrote {count} configs to {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
