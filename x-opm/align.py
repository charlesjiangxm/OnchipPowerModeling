"""Load a case's features + power target and align them in time.

Feature and power DataFrames are both indexed by ``time_ns``.  The power pkl has
exactly one extra trailing cycle and a float index; we cast both indices to int64
and take the intersection (same-cycle alignment, no lag), dropping the trailing
power row.  Feature columns are reindexed to a fixed canonical order so nothing
downstream depends on stored order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_features(func_pkl: str, canonical_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_pickle(func_pkl)
    df.index = df.index.astype(np.int64)
    df.index.name = "time_ns"
    if canonical_cols is not None:
        if set(df.columns) != set(canonical_cols):
            missing = set(canonical_cols) - set(df.columns)
            extra = set(df.columns) - set(canonical_cols)
            raise RuntimeError(
                f"{func_pkl}: column set differs from canonical "
                f"(missing {len(missing)}, extra {len(extra)})"
            )
        df = df.reindex(columns=canonical_cols)
    return df


def load_target(pwr_pkl: str, target_col: str, ref_index: pd.Index) -> pd.Series:
    pwr = pd.read_pickle(pwr_pkl)
    pwr.index = pwr.index.astype(np.int64)
    if target_col not in pwr.columns:
        raise RuntimeError(f"{pwr_pkl}: target column {target_col!r} not found")
    return pwr.loc[ref_index, target_col].rename("target")


def align_case(func_pkl: str, pwr_pkl: str, target_col: str,
               canonical_cols: list[str] | None = None
               ) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features, target)`` sharing an identical int64 time_ns index."""
    feats = load_features(func_pkl, canonical_cols)
    pwr = pd.read_pickle(pwr_pkl)
    pwr.index = pwr.index.astype(np.int64)
    ref_index = feats.index.intersection(pwr.index)
    if len(ref_index) != feats.shape[0]:
        # some feature cycles have no power label -> drop them (rare; warn upstream)
        feats = feats.loc[ref_index]
    y = pwr.loc[ref_index, target_col].rename("target")
    return feats, y
