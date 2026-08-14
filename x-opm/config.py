"""Typed configuration for the x-opm dataset build (steps 1-5)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import yaml

from schema import TypeRules


@dataclass(frozen=True)
class XopmConfig:
    # --- data locations -----------------------------------------------------
    db_root: str                      # e.g. c906_db_net_1cyc_20260729
    scope: str                        # e.g. cp0  -> aq_core/<scope>/<case>_func.pkl
    target: str                       # power column, e.g. x_aq_core/Pc(x_aq_cp0_top)
    cases: list[str]                  # all benchmark names to process
    test_benchmarks: list[str]        # subset held out as the test set
    output_dir: str                   # e.g. out/x-opm

    # --- feature engineering knobs -----------------------------------------
    rules: TypeRules = field(default_factory=TypeRules)
    # which types get the toggle->Hamming transform; the rest of the wide
    # signals (type D) are kept as raw integers scaled by 2^width-1.
    hamming_types: tuple[str, ...] = ("C",)
    # count the initial state (t=0) as a toggle-from-reset in the Hamming feature
    hamming_count_initial: bool = True
    # detect exact-duplicate raw columns (same net at multiple hierarchy depths)
    # and, if dedup=True, drop all but a canonical representative.
    detect_duplicates: bool = True
    dedup_identical: bool = False

    @property
    def train_benchmarks(self) -> list[str]:
        test = set(self.test_benchmarks)
        return [c for c in self.cases if c not in test]

    # ------------------------------------------------------------------ paths
    def func_pkl(self, case: str) -> str:
        return f"{self.db_root}/aq_core/{self.scope}/{case}_func.pkl"

    def pwr_pkl(self, case: str) -> str:
        return f"{self.db_root}/pwr/{case}_pwr.pkl"

    @property
    def manifest_path(self) -> str:
        return f"{self.output_dir}/manifest.json"

    @property
    def dataset_dir(self) -> str:
        return f"{self.output_dir}/dataset"

    @property
    def reports_dir(self) -> str:
        return f"{self.output_dir}/reports"

    def case_out_dir(self, case: str) -> str:
        split = "testset" if case in set(self.test_benchmarks) else "trainset"
        return f"{self.dataset_dir}/{split}/{case}"

    # --------------------------------------------------------------- loading
    @staticmethod
    def from_yaml(path: str) -> "XopmConfig":
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        rules_raw = raw.pop("rules", None)
        cfg = XopmConfig(**raw)
        if rules_raw:
            cfg = replace(cfg, rules=TypeRules(**{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in rules_raw.items()
            }))
        if isinstance(cfg.hamming_types, list):
            cfg = replace(cfg, hamming_types=tuple(cfg.hamming_types))
        return cfg
