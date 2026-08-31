import numpy as np
import pandas as pd
import pytest

from cobit.data.build import build_dataset
from cobit.data.dataset import DatasetCache, aggregate_windows


def test_float_power_index_and_extra_row(synthetic_cfg):
    """The synthetic pwr pkl has a float index and one extra row by design."""
    cfg, _ = synthetic_cfg
    pwr = pd.read_pickle(f"{cfg.data.db_root}/pwr/b1_pwr.pkl")
    assert pwr.index.dtype.kind == "f" and len(pwr) == 601
    build_dataset(cfg)
    labels = DatasetCache(cfg).labels("b1")
    assert len(labels) == 600
    assert labels.index.dtype.kind == "i"


def test_split_rows_and_bundle_shapes(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    cache = DatasetCache(cfg)
    tr, va = cache.split_rows("b1")
    assert (tr.stop - tr.start) + (va.stop - va.start) == 600
    assert va.stop - va.start == 120  # 20% tail

    # val_fraction=0 means literally no validation rows
    cfg.split.val_fraction = 0.0
    tr0, va0 = DatasetCache(cfg).split_rows("b1")
    assert (tr0.stop - tr0.start, va0.stop - va0.start) == (600, 0)
    cfg.split.val_fraction = 0.2
    bundle = cache.load_split(np.arange(10), dense=True)
    assert bundle.X_train.shape == (960, 10)  # 2 train benches x 480
    assert bundle.X_val.shape == (240, 10)
    assert bundle.X_test.shape == (600, 10)
    assert bundle.train_slices["b2"] == slice(480, 960)
    # X rows and y rows always agree
    assert bundle.X_test.shape[0] == bundle.y_test.size


def test_y_matches_power_column(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    cache = DatasetCache(cfg)
    bundle = cache.load_split(np.arange(4), dense=True)
    pwr = pd.read_pickle(f"{cfg.data.db_root}/pwr/b3_pwr.pkl")
    np.testing.assert_allclose(bundle.y_test, pwr["Pc(x_aq_core)"].to_numpy()[:600])


def test_aggregate_windows_dense_and_sparse():
    from scipy import sparse

    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    slices = {"a": slice(0, 10)}
    Xw, yw, sl = aggregate_windows(X, y, slices, tau=4, mode="sum")
    assert Xw.shape == (2, 2) and yw.tolist() == [1.5, 5.5]
    np.testing.assert_allclose(Xw[0], X[:4].sum(axis=0))

    Xs = sparse.csr_matrix(X)
    Xw2, yw2, _ = aggregate_windows(Xs, y, slices, tau=4, mode="sum")
    np.testing.assert_allclose(np.asarray(Xw2.todense()), Xw)

    Xw3, _, _ = aggregate_windows(Xs, y, slices, tau=4, mode="any")
    assert set(np.asarray(Xw3.todense()).ravel()) <= {0, 1}
    # tail rows that do not fill a window are dropped
    assert yw.size == 2
