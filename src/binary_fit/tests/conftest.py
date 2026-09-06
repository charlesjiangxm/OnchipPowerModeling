"""Synthetic raw-state dataset fixtures for the binary_fit tests.

Builds a tiny source dataset in the same on-disk shape as the real one:
``<src>/func/<module>/<bench>_func.pkl.zst`` (object-dtype Python-int columns,
``path[hi:lo]`` names, raw per-cycle STATES) and a pre-populated
``<out>/pwr/<bench>_pwr.pkl.zst`` (single ``Pc(x_aq_core)`` column with the extra
trailing cycle). The power law depends on a few known bits so selection/fit can
be checked. Covers: a 1-bit net, a within-64 bus, a >64-bit bus (two uint64
lanes, lo=64), a globally constant-1 bit and a globally-dead bus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from binary_fit.config import Config
from binary_fit.utils import save_pickle_zst

MODULE = "m0"
BENCHES = ["b0", "b1", "b2", "tst"]
# Power law: y = base + weights . [ top/a , bit0 of top/b , stored-bit3 of top/c ].
PLANTED = ["top/a", "top/b[0]", "top/c[67]"]  # c[hi:lo]=[135:64] so stored bit k -> RTL 64+k


def _make_case(n: int, rng) -> tuple[pd.DataFrame, np.ndarray]:
    a = rng.integers(0, 2, n)
    a[0], a[1] = 0, 1  # guarantee global variation
    b = rng.integers(0, 16, n)  # 4-bit bus top/b[3:0]
    clow = rng.integers(0, 16, n)  # stored bits 0..3 of the 72-bit bus top/c[135:64]
    chi = rng.integers(0, 2, n)  # stored bit 65 (RTL 129) -> exercises lane 1
    chi[0], chi[1] = 0, 1
    c = np.array([int(clow[i]) | (int(chi[i]) << 65) for i in range(n)], dtype=object)
    func = pd.DataFrame(
        {
            "top/a": a.astype(object),
            "top/b[3:0]": b.astype(object),
            "top/c[135:64]": c,
            "top/const1": np.ones(n, dtype=object),  # globally constant-1 -> dropped
            "top/dead[1:0]": np.zeros(n, dtype=object),  # globally dead -> dropped
        },
        index=np.arange(n, dtype=np.int64),
    )
    b0 = (b & 1).astype(float)
    c3 = ((clow >> 3) & 1).astype(float)  # stored bit 3 -> top/c[67]
    y = 0.5 + 0.30 * a + 0.20 * b0 + 0.15 * c3 + rng.normal(0, 0.005, n)
    return func, y


def make_dataset(src_root, out_root, n: int = 60, seed: int = 0, window: int = 1) -> Config:
    """Write the synthetic source func pkls + target pwr pkls; return a wired Config.

    ``window`` sets ``data.window_size``; the default 1 keeps the per-cycle rows
    the older tests assert on (the package default is 32, far more than the ``n``
    cycles these fixtures write).
    """
    src_root, out_root = str(src_root), str(out_root)
    rng = np.random.default_rng(seed)
    from pathlib import Path

    fdir = Path(src_root) / "func" / MODULE
    opwr = Path(out_root) / "pwr"
    fdir.mkdir(parents=True, exist_ok=True)
    opwr.mkdir(parents=True, exist_ok=True)
    for b in BENCHES:
        func, y = _make_case(n, rng)
        save_pickle_zst(func, fdir / f"{b}_func.pkl.zst")
        # pwr has one extra trailing cycle and a float index
        yv = np.concatenate([y, [-999.0]])  # sentinel trailing row must be dropped
        pwr = pd.DataFrame({"Pc(x_aq_core)": yv}, index=np.arange(n + 1, dtype=float))
        save_pickle_zst(pwr, opwr / f"{b}_pwr.pkl.zst")

    cfg = Config()
    cfg.build.source_db_root = src_root
    cfg.build.modules = [MODULE]
    cfg.build.out_root = out_root
    cfg.data.func_dir = f"{out_root}/func"
    cfg.data.pwr_dir = f"{out_root}/pwr"
    cfg.data.window_size = window
    cfg.split.test_benchmarks = ["tst"]
    cfg.split.val_fraction = 0.25
    cfg.selection.selector = "lasso"  # deterministic, no skglm dependency in tests
    cfg.selection.target_qs = [3]
    cfg.selection.max_rows = 0  # use all rows (tiny)
    cfg.runtime.seed = 0
    return cfg


WINDOW = 8  # cycles per row in the windowed fixture
WINDOWED_N = 320  # -> 40 rows per benchmark, 30 train / 10 val at val_fraction 0.25


@pytest.fixture
def synth(tmp_path):
    """(cfg, planted_names): synthetic dataset wired into a Config, per-cycle rows."""
    cfg = make_dataset(tmp_path / "src", tmp_path / "out")
    return cfg, PLANTED


@pytest.fixture
def synth_windowed(tmp_path):
    """(cfg, planted_names): same dataset averaged to WINDOW-cycle rows.

    The planted power law is linear in the bits, so window averaging preserves it
    exactly (mean(y) = base + w . mean(bits)) -- selection and fit must still
    recover the planted bits.
    """
    cfg = make_dataset(tmp_path / "src", tmp_path / "out", n=WINDOWED_N, window=WINDOW)
    return cfg, PLANTED


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Every plot function must close its own figure.

    ``--fit`` calls them once per (-q, model) experiment, and a leaked Figure
    pins its whole data array -- gigabytes at ``window_size: 1``.
    """
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    assert plt.get_fignums() == [], "a plot function did not close its figure"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end test that trains a model")
