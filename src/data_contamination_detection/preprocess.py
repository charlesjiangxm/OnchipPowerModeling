"""Preprocessing: constant-column drop, QuantileTransformer, float32 cast.

The QuantileTransformer is essential (not optional): the raw data contains
arbitrary-precision integers up to ~180 bits (~5.76e+53), which overflow
float32. The transformer maps all values to [0,1], making float32 safe and
Euclidean distance meaningful per the spec's requirement.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from data_loader import LoadedData

logger = logging.getLogger(__name__)

N_QUANTILES = 1000
RANDOM_SEED = 42


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_test: np.ndarray
    qt: QuantileTransformer
    keep_mask: np.ndarray
    feature_names: list


def run(train_data: LoadedData, test_data: LoadedData) -> PreprocessedData:
    X_train_raw = train_data.X
    X_test_raw = test_data.X
    n_raw = X_train_raw.shape[1]
    logger.info(f"Preprocessing: train {X_train_raw.shape}, test {X_test_raw.shape}")

    # Step 1: Drop constant columns (using ptp = max - min, avoids squaring/overflow)
    col_min = X_train_raw.min(axis=0)
    col_max = X_train_raw.max(axis=0)
    col_range = col_max - col_min
    keep_mask = col_range > 0
    n_dropped = n_raw - keep_mask.sum()
    logger.info(f"Dropped {n_dropped} constant columns, {keep_mask.sum()} remain")

    X_train_kept = X_train_raw[:, keep_mask]
    X_test_kept = X_test_raw[:, keep_mask]
    del X_train_raw, X_test_raw

    kept_names = [train_data.feature_names[i] for i in range(n_raw) if keep_mask[i]]

    # Step 2: QuantileTransformer (fit on train only)
    logger.info(f"Fitting QuantileTransformer (n_quantiles={N_QUANTILES})...")
    t0 = time.time()
    qt = QuantileTransformer(
        n_quantiles=N_QUANTILES,
        output_distribution="uniform",
        random_state=RANDOM_SEED,
    )
    X_train_qt = qt.fit_transform(X_train_kept)
    logger.info(f"  fit+transform train: {time.time()-t0:.1f}s")

    t0 = time.time()
    X_test_qt = qt.transform(X_test_kept)
    logger.info(f"  transform test: {time.time()-t0:.1f}s")
    del X_train_kept, X_test_kept

    # Step 3: Cast to float32 (safe: all values in [0,1] after QT)
    X_train = X_train_qt.astype(np.float32)
    X_test = X_test_qt.astype(np.float32)
    del X_train_qt, X_test_qt

    # Sanity checks
    assert not np.isinf(X_train).any(), "Inf values in train after QT"
    assert not np.isinf(X_test).any(), "Inf values in test after QT"
    assert not np.isnan(X_train).any(), "NaN values in train after QT"
    assert not np.isnan(X_test).any(), "NaN values in test after QT"

    train_mb = X_train.nbytes / 1e6
    test_mb = X_test.nbytes / 1e6
    logger.info(f"Done: train {X_train.shape} ({train_mb:.0f}MB), test {X_test.shape} ({test_mb:.0f}MB)")

    return PreprocessedData(X_train, X_test, qt, keep_mask, kept_names)
