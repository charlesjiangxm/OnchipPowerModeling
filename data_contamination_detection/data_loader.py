"""Data loading for contamination detection.

Loads PKL feature files, converts object-dtype arbitrary-precision integers
to float64, concatenates training files in order, and tracks benchmark boundaries.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/home/jjiangan/disk/OnchipPowerModelingNew/c906_db_net_1cyc_20260729/aq_core")

TRAIN_FILES = [
    "cache_func.pkl",
    "csr_func.pkl",
    "debug_func.pkl",
    "exception_func.pkl",
    "interrupt_func.pkl",
    "ISA_FP_func.pkl",
    "ISA_INT_func.pkl",
    "ISA_LS_func.pkl",
    "ISA_THEAD_func.pkl",
    "MMU_func.pkl",
]

TEST_FILE = "coremark_func.pkl"


@dataclass
class BenchmarkBoundary:
    name: str
    start: int
    end: int

    @property
    def n_rows(self):
        return self.end - self.start


@dataclass
class LoadedData:
    X: np.ndarray
    source_ids: np.ndarray
    boundaries: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)


def _load_one_pkl(path: Path, max_rows: int | None = None) -> np.ndarray:
    df = pd.read_pickle(path)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows]
    feature_names = list(df.columns)
    arr = df.to_numpy(dtype=np.float64)
    del df
    return arr, feature_names


def load_data(data_dir: Path | None = None, smoke_test: bool = False):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    max_rows = 1000 if smoke_test else None

    train_arrays = []
    train_boundaries = []
    train_source_ids = []
    feature_names = None
    offset = 0

    for i, fname in enumerate(TRAIN_FILES):
        path = data_dir / fname
        logger.info(f"Loading train [{i+1}/{len(TRAIN_FILES)}]: {fname}")
        t0 = time.time()
        arr, names = _load_one_pkl(path, max_rows)
        train_arrays.append(arr)
        train_boundaries.append(BenchmarkBoundary(fname, offset, offset + len(arr)))
        train_source_ids.append(np.full(len(arr), i, dtype=np.int32))
        offset += len(arr)
        if feature_names is None:
            feature_names = names
        logger.info(f"  {fname}: {len(arr)} rows, {arr.shape[1]} cols, {time.time()-t0:.1f}s")

    X_train = np.vstack(train_arrays)
    del train_arrays
    source_ids_train = np.concatenate(train_source_ids)
    del train_source_ids

    logger.info(f"Loading test: {TEST_FILE}")
    t0 = time.time()
    X_test, _ = _load_one_pkl(data_dir / TEST_FILE, max_rows)
    test_boundaries = [BenchmarkBoundary(TEST_FILE, 0, len(X_test))]
    source_ids_test = np.zeros(len(X_test), dtype=np.int32)
    logger.info(f"  {TEST_FILE}: {len(X_test)} rows, {time.time()-t0:.1f}s")

    train_data = LoadedData(X_train, source_ids_train, train_boundaries, feature_names)
    test_data = LoadedData(X_test, source_ids_test, test_boundaries, feature_names)

    return train_data, test_data
