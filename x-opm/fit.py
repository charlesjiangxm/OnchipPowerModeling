"""Fit stage (steps 1-2 + denominators): decide everything from TRAIN cases only.

Produces ``manifest.json`` -- the single source of truth that the transform stage
applies verbatim to every case (train and test).  Keeping all decisions here and
applying them elsewhere is what structurally prevents test leakage.
"""

from __future__ import annotations

import logging

from align import load_features
from config import XopmConfig
from io_utils import save_json
from schema import classify, parse_column, should_invert
from variance import RawStats, duplicate_groups

log = logging.getLogger("x-opm.fit")


def _plan_feature(col: str, cfg: XopmConfig) -> dict:
    """Type/rule/width/lo + transform + scale plan for one surviving column."""
    path, width, lo = parse_column(col)
    ftype, rule = classify(col, cfg.rules)
    invert = ftype == "A" and should_invert(col, cfg.rules)
    if ftype in ("A", "B"):
        transform, scale, denom, n_feat = "bit_split", "bits", 1, width
    elif ftype in cfg.hamming_types:
        transform, scale, denom, n_feat = "toggle_hamming", "hamming", width, 1
    else:  # residual wide/narrow payload kept as a raw integer
        transform, scale, denom, n_feat = "raw_int", "raw_int", (1 << width) - 1, 1
    return {
        "source_signal": path, "type": ftype, "classification_rule": rule,
        "width": width, "lo": lo, "invert": invert,
        "transform": transform, "scale": scale, "scale_divisor": denom,
        "n_features": n_feat,
    }


def run_fit(cfg: XopmConfig, force: bool = False) -> dict:
    # 1. canonical column order (assert identical column set across all cases).
    first = load_features(cfg.func_pkl(cfg.cases[0]))
    canonical = sorted(first.columns)
    widths = {c: parse_column(c)[1] for c in canonical}
    del first
    log.info("canonical columns: %d (scope=%s)", len(canonical), cfg.scope)

    # 2. stream TRAIN cases -> raw stats + constancy + dup fingerprints.
    stats = RawStats(canonical, widths, detect_duplicates=cfg.detect_duplicates)
    for case in cfg.train_benchmarks:
        df = load_features(cfg.func_pkl(case), canonical)
        stats.update(df)
        log.info("fit: folded %s (%d rows)", case, df.shape[0])
        del df
    finalized = stats.finalize()

    # 3. constancy filter -> kept vs dropped.
    kept = [c for c in canonical if not finalized[c]["constant"]]
    dropped = {c: {"reason": "zero-variance (constant across train)",
                   "raw_min": finalized[c]["raw_min"],
                   "raw_max": finalized[c]["raw_max"],
                   "width": widths[c]}
               for c in canonical if finalized[c]["constant"]}
    log.info("kept %d / %d columns (dropped %d zero-variance)",
             len(kept), len(canonical), len(dropped))

    # 4. optional exact-duplicate detection / dedup.
    dup_groups = duplicate_groups(finalized, kept) if cfg.detect_duplicates else []
    deduped_out: set[str] = set()
    if cfg.dedup_identical:
        for group in dup_groups:
            rep = group[0]
            for member in group[1:]:
                dropped[member] = {"reason": f"duplicate of {rep}",
                                   "raw_min": finalized[member]["raw_min"],
                                   "raw_max": finalized[member]["raw_max"],
                                   "width": widths[member]}
                deduped_out.add(member)
        kept = [c for c in kept if c not in deduped_out]
    log.info("duplicate groups: %d (dedup=%s)", len(dup_groups), cfg.dedup_identical)

    # 5. plan features + attach raw stats.
    features: dict[str, dict] = {}
    for col in kept:
        plan = _plan_feature(col, cfg)
        plan.update({"raw_min": finalized[col]["raw_min"],
                     "raw_max": finalized[col]["raw_max"],
                     "raw_mean": finalized[col]["raw_mean"]})
        # sanity: raw values fit their declared width.
        if plan["raw_max"] > (1 << plan["width"]) - 1:
            log.warning("%s: raw_max %d exceeds 2^%d-1", col,
                        plan["raw_max"], plan["width"])
        features[col] = plan

    manifest = {
        "config": {
            "db_root": cfg.db_root, "scope": cfg.scope, "target": cfg.target,
            "train_benchmarks": cfg.train_benchmarks,
            "test_benchmarks": cfg.test_benchmarks,
            "hamming_types": list(cfg.hamming_types),
            "hamming_count_initial": cfg.hamming_count_initial,
            "dedup_identical": cfg.dedup_identical,
        },
        "n_raw_columns": len(canonical),
        "n_kept_signals": len(kept),
        "n_final_features": sum(f["n_features"] for f in features.values()),
        "count_train_rows": stats.count,
        "canonical_columns": canonical,
        "features": features,
        "dropped": dropped,
        "duplicate_groups": dup_groups,
    }
    save_json(manifest, cfg.manifest_path)
    log.info("wrote manifest -> %s (%d final features)",
             cfg.manifest_path, manifest["n_final_features"])
    return manifest
