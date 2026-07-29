import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cobit.data.build import build_dataset, scope_col_range
from cobit.data.dataset import DatasetCache
from cobit.data.discovery import discover, plan_benchmarks


def test_build_and_load_roundtrip(synthetic_cfg):
    cfg, toggles = synthetic_cfg
    layout, registry = build_dataset(cfg)
    assert registry.n_features == 8 + 311 + 1 + 31 + 64 + 65 + 3 + 1  # all bits

    cache = DatasetCache(cfg)
    # reconstruct the full matrix of b1 and compare against the ground truth
    all_ids = np.arange(registry.n_features)
    X = cache.load_features("b1", all_ids)
    assert X.shape == (600, registry.n_features)
    names = registry.feature_names(all_ids)
    got = np.asarray(X.todense())
    for j, name in enumerate(names):
        expected = toggles["b1"][name]
        assert np.array_equal(got[:, j], expected), f"column {name} mismatch"


def test_label_alignment_drops_extra_power_row(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    cache = DatasetCache(cfg)
    # power pkl has n+1 rows; aligned labels must have exactly n rows
    assert len(cache.labels("b1")) == 600
    assert cache.n_rows("b1") == 600


def test_toggle_counts_match_data(synthetic_cfg):
    cfg, toggles = synthetic_cfg
    _, registry = build_dataset(cfg)
    cache = DatasetCache(cfg)
    counts = cache.toggle_counts(["b1"])
    names = registry.feature_names(np.arange(registry.n_features))
    for j, name in enumerate(names):
        assert counts[j] == toggles["b1"][name].sum()


def test_chunking_consistency(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)  # chunk_rows=256 -> 3 chunks of 600 rows
    cache = DatasetCache(cfg)
    m = cache.manifest("b2")
    assert [c["stop"] - c["start"] for c in m["chunks"]] == [256, 256, 88]
    ids = np.arange(50)
    whole = cache.load_features("b2", ids)
    part = sparse.vstack(
        [cache.load_features("b2", ids, rows=slice(0, 300)),
         cache.load_features("b2", ids, rows=slice(300, 600))]
    )
    assert (whole != part).nnz == 0


def test_missing_scope_policy(synthetic_cfg):
    cfg, _ = synthetic_cfg
    db_root = cfg.data.db_root
    # remove m2's pkl for b2: b2 must drop from training
    import os

    os.remove(f"{db_root}/aq_core/m2/b2_func.pkl")
    layout = discover(db_root)
    train, test = plan_benchmarks(layout, ["b3"])
    assert train == ["b1"] and test == ["b3"]

    # as a TEST benchmark it must be kept and zero-filled
    train2, test2 = plan_benchmarks(layout, ["b2"])
    assert "b2" in test2 and set(train2) == {"b1", "b3"}
    build_dataset(cfg, force=True)
    cfg.split.test_benchmarks = ["b2"]
    cache = DatasetCache(cfg)
    _, registry = discover(db_root), cache.registry
    lo, hi = scope_col_range(cache.registry, "m2")
    X = cache.load_features("b2", np.arange(lo, hi))
    assert X.nnz == 0  # zero-filled scope


def test_stale_manifest_rebuilds(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    # touch a source pkl -> stamp changes -> rebuild happens (no exception)
    p = f"{cfg.data.db_root}/aq_core/b1_func.pkl"
    df = pd.read_pickle(p)
    df.to_pickle(p)
    build_dataset(cfg)
    cache = DatasetCache(cfg)
    assert cache.n_rows("b1") == 600


def test_interrupted_build_self_heals(synthetic_cfg):
    """Regression: missing bit_stats.npz must not crash subsequent builds."""
    import os

    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    os.remove(f"{cfg.data.cache_dir}/bit_stats.npz")  # simulate interrupt
    build_dataset(cfg)  # must recover via per-benchmark counts.npy
    cache = DatasetCache(cfg)
    assert cache.toggle_counts(["b1"]).sum() > 0

    # a damaged counts.npy falls through to a rebuild rather than crashing
    os.remove(f"{cfg.data.cache_dir}/features/b1/counts.npy")
    build_dataset(cfg)
    assert (cache.cache / "features" / "b1" / "counts.npy").exists()


def test_data_stamp_tracks_source_content(synthetic_cfg):
    cfg, _ = synthetic_cfg
    build_dataset(cfg)
    cache = DatasetCache(cfg)
    s0 = cache.data_stamp()
    # re-dump a source pkl in place (new mtime) and rebuild
    import time

    p = f"{cfg.data.db_root}/aq_core/b1_func.pkl"
    time.sleep(1.1)  # ensure a different integer mtime
    pd.read_pickle(p).to_pickle(p)
    build_dataset(cfg)
    assert DatasetCache(cfg).data_stamp() != s0
