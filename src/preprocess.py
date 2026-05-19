"""Drop zero-variance columns (identified on train) and z-score X and y on
the training split. The same train-fit Standardizer is applied to val and
test, never refit on them. Reuses ``Standardizer`` from
``ft_transformer_model.py`` so the FT-Transformer path stays compatible.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .ft_transformer_model import Standardizer


log = logging.getLogger(__name__)


def drop_zero_variance(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop columns with zero variance on train; apply the same mask to val/test."""
    arr = train_x.to_numpy(dtype=np.float64, copy=False)
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(arr)
    mask = vt.get_support()
    n_dropped = int((~mask).sum())
    if n_dropped:
        log.info(
            "drop_zero_variance: removed %d / %d constant columns on train",
            n_dropped, len(mask),
        )
    keep_cols = train_x.columns[mask]
    return train_x[keep_cols], val_x[keep_cols], test_x[keep_cols]


def standardize_train_apply(
    train_x: pd.DataFrame, train_y: pd.Series,
    val_x: pd.DataFrame, val_y: pd.Series,
    test_x: pd.DataFrame, test_y: pd.Series,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    Standardizer,
]:
    """Fit Standardizer on train; transform train/val/test x and y.

    Returns standardized DataFrames/Series in the original column order plus
    the fitted Standardizer (needed for inverse-transform at metric time).
    """
    std = Standardizer.fit(
        train_x.to_numpy(dtype=np.float64, copy=False),
        train_y.to_numpy(dtype=np.float64, copy=False),
    )

    def _to_df(df_in, x_z):
        return pd.DataFrame(x_z, columns=df_in.columns)

    def _to_series(s_in, y_z):
        return pd.Series(y_z, name=s_in.name)

    train_xz = _to_df(train_x, std.transform_X(train_x))
    val_xz = _to_df(val_x, std.transform_X(val_x))
    test_xz = _to_df(test_x, std.transform_X(test_x))
    train_yz = _to_series(train_y, std.transform_y(train_y))
    val_yz = _to_series(val_y, std.transform_y(val_y))
    test_yz = _to_series(test_y, std.transform_y(test_y))
    return train_xz, train_yz, val_xz, val_yz, test_xz, test_yz, std


def save_splits(
    out_dir: Path,
    train_x: pd.DataFrame, train_y: pd.Series,
    val_x: pd.DataFrame, val_y: pd.Series,
    test_x: pd.DataFrame, test_y: pd.Series,
) -> None:
    """Write the six split pkls to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x.to_pickle(out_dir / "train_x.pkl")
    train_y.to_pickle(out_dir / "train_y.pkl")
    val_x.to_pickle(out_dir / "val_x.pkl")
    val_y.to_pickle(out_dir / "val_y.pkl")
    test_x.to_pickle(out_dir / "test_x.pkl")
    test_y.to_pickle(out_dir / "test_y.pkl")


def save_standardizer(out_dir: Path, std: Standardizer) -> None:
    """Persist the fitted Standardizer for reproducibility."""
    with open(Path(out_dir) / "standardizer.pkl", "wb") as f:
        pickle.dump(std, f)
