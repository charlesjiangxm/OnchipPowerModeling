"""Transform stage (steps 3-4): apply the manifest to one case and write pkls.

For each surviving source signal the manifest says exactly how to transform it;
we group the resulting scaled features by classification type and write one pkl
per (case, type) plus one aligned target pkl.  Optionally accumulates scaled
feature statistics (training cases only) for the report.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from align import align_case
from config import XopmConfig
from io_utils import ensure_dir, save_pickle
from transform import (scale_bits, scale_hamming, scale_raw_int, split_bits,
                       toggle_hamming)

log = logging.getLogger("x-opm.apply")

TYPES = ("A", "B", "C", "D")
_STORE_DTYPE = np.float32  # scaled features live in [0,1]; float32 is ample


def _transform_signal(values: np.ndarray, plan: dict,
                      count_initial: bool) -> list[tuple[str, np.ndarray]]:
    """Return list of (feature_name, scaled float array) for one source signal."""
    path = plan["source_signal"]
    width, lo = plan["width"], plan["lo"]
    t = plan["transform"]
    if t == "bit_split":
        return [(name, scale_bits(bit))
                for name, bit in split_bits(values, path, width, lo, plan["invert"])]
    if t == "toggle_hamming":
        ham = toggle_hamming(values, width, count_initial)
        return [(f"{path}_toggle_hamming", scale_hamming(ham, width))]
    # raw_int: keep the original column name (with its [msb:lsb] suffix)
    return [(_raw_name(path, width, lo), scale_raw_int(values, width))]


def _raw_name(path: str, width: int, lo: int) -> str:
    return path if width == 1 else f"{path}[{lo + width - 1}:{lo}]"


def apply_case(case: str, manifest: dict, cfg: XopmConfig,
               force: bool = False) -> dict:
    out_dir = cfg.case_out_dir(case)
    done_marker = os.path.join(out_dir, "target.pkl")
    if os.path.exists(done_marker) and not force:
        log.info("apply: skip %s (exists; use --force to rebuild)", case)
        return {"case": case, "skipped": True}

    feats, y = align_case(cfg.func_pkl(case), cfg.pwr_pkl(case), cfg.target,
                          manifest["canonical_columns"])
    index = feats.index

    per_type: dict[str, dict[str, np.ndarray]] = {t: {} for t in TYPES}
    features = manifest["features"]
    for col, plan in features.items():
        vals = feats[col].to_numpy()
        for name, arr in _transform_signal(vals, plan, cfg.hamming_count_initial):
            per_type[plan["type"]][name] = arr.astype(_STORE_DTYPE)

    ensure_dir(out_dir)
    counts = {}
    for t in TYPES:
        cols = per_type[t]
        counts[t] = len(cols)
        if not cols:
            continue
        df = pd.DataFrame(cols, index=index)
        df.index.name = "time_ns"
        save_pickle(df, os.path.join(out_dir, f"type{t}.pkl"))
    save_pickle(y.astype(np.float64), done_marker)
    log.info("apply: %s -> %s  rows=%d types=%s", case, out_dir, len(index), counts)
    return {"case": case, "skipped": False, "rows": int(len(index)),
            "type_feature_counts": counts}
