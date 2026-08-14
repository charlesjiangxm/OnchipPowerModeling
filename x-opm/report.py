"""Feature report (step 5a) + run summary.

``feature_report.csv`` has one row per FINAL feature (post-transform, post-scale)
with its type, the classification rule, widths, transform, scale divisor, and both
raw (source-signal) and scaled (feature) min/max/mean over the training set.
Dropped source signals are appended with ``dropped=True`` and their reason.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import XopmConfig
from io_utils import save_csv, save_json

log = logging.getLogger("x-opm.report")


class ScaledStats:
    """Streaming min/max/sum/count of the final scaled features (training only)."""

    def __init__(self):
        self.min: dict[str, float] = {}
        self.max: dict[str, float] = {}
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def update(self, name: str, arr) -> None:
        mn = float(arr.min()); mx = float(arr.max())
        self.min[name] = mn if name not in self.min else min(self.min[name], mn)
        self.max[name] = mx if name not in self.max else max(self.max[name], mx)
        self.sum[name] = self.sum.get(name, 0.0) + float(arr.sum())
        self.count[name] = self.count.get(name, 0) + int(arr.shape[0])

    def stats(self, name: str) -> tuple[float, float, float]:
        c = self.count.get(name, 0)
        mean = self.sum[name] / c if c else float("nan")
        return self.min.get(name, float("nan")), self.max.get(name, float("nan")), mean


def _final_feature_names(source: str, plan: dict) -> list[str]:
    path, width, lo = plan["source_signal"], plan["width"], plan["lo"]
    t = plan["transform"]
    if t == "bit_split":
        if width == 1:
            return [path]
        return [f"{path}[{lo + k}]" for k in range(width)]
    if t == "toggle_hamming":
        return [f"{path}_toggle_hamming"]
    return [path if width == 1 else f"{path}[{lo + width - 1}:{lo}]"]


def build_report(cfg: XopmConfig, manifest: dict, scaled: ScaledStats) -> None:
    rows = []
    for col, plan in manifest["features"].items():
        for name in _final_feature_names(col, plan):
            smin, smax, smean = scaled.stats(name)
            rows.append({
                "feature_name": name,
                "source_signal": col,
                "type": plan["type"],
                "classification_rule": plan["classification_rule"],
                "source_width": plan["width"],
                "feature_width": 1 if plan["transform"] == "bit_split"
                                 else (_ham_width(plan["width"]) if plan["transform"] == "toggle_hamming"
                                       else plan["width"]),
                "transform": plan["transform"],
                "invert": plan["invert"],
                "scale_divisor": plan["scale_divisor"],
                "raw_min": plan["raw_min"],
                "raw_max": plan["raw_max"],
                "raw_mean": plan["raw_mean"],
                "scaled_min": smin,
                "scaled_max": smax,
                "scaled_mean": smean,
                "dropped": False,
                "drop_reason": "",
            })
    for col, info in manifest["dropped"].items():
        rows.append({
            "feature_name": col, "source_signal": col, "type": "",
            "classification_rule": "", "source_width": info["width"],
            "feature_width": "", "transform": "", "invert": "",
            "scale_divisor": "", "raw_min": info["raw_min"],
            "raw_max": info["raw_max"], "raw_mean": "", "scaled_min": "",
            "scaled_max": "", "scaled_mean": "", "dropped": True,
            "drop_reason": info["reason"],
        })
    df = pd.DataFrame(rows)
    save_csv(df, f"{cfg.reports_dir}/feature_report.csv")
    log.info("wrote feature_report.csv: %d rows (%d final features, %d dropped)",
             len(df), int((~df["dropped"]).sum()), int(df["dropped"].sum()))

    summary = {
        "scope": cfg.scope, "target": cfg.target,
        "train_benchmarks": cfg.train_benchmarks,
        "test_benchmarks": cfg.test_benchmarks,
        "n_raw_columns": manifest["n_raw_columns"],
        "n_kept_signals": manifest["n_kept_signals"],
        "n_final_features": manifest["n_final_features"],
        "n_dropped_signals": len(manifest["dropped"]),
        "count_train_rows": manifest["count_train_rows"],
        "n_duplicate_groups": len(manifest["duplicate_groups"]),
        "type_feature_counts": _type_counts(manifest),
    }
    save_json(summary, f"{cfg.reports_dir}/summary.json")
    log.info("wrote summary.json")


def _ham_width(bus_width: int) -> int:
    return max(1, (bus_width).bit_length())  # bits to store max hamming = bus_width


def _type_counts(manifest: dict) -> dict:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for plan in manifest["features"].values():
        counts[plan["type"]] += plan["n_features"]
    return counts
