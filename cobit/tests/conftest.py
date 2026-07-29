"""Shared fixtures: a synthetic toggle DB in exactly the real pkl format."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# (scope, column, density) - covers 1-bit, lo>0, and >64-bit wide nets
TOP_NETS = [("clk_en", 0.30), ("bus_a[7:0]", 0.10), ("wide[310:0]", 0.02)]
M1_NETS = [("x_m1/sig_a", 0.20), ("x_m1/cnt[31:1]", 0.05), ("x_m1/w64[63:0]", 0.03)]
M2_NETS = [("x_m2/w65[64:0]", 0.03), ("x_m2/tiny[2:0]", 0.15)]

# ground-truth power model over specific bits (feature names in registry form)
TRUE_BITS = {
    "bus_a[3]": 0.030,
    "x_m1/cnt[5]": 0.020,  # mask bit 4 of cnt[31:1]
    "wide[100]": 0.015,
}
# interaction between two reasonably-active bits: a LINEAR stage-1 selector
# can only find proxies with a visible marginal effect (~w * partner density)
INTERACTION = (("bus_a[3]", "x_m2/tiny[1]"), 0.040)
BASE_POWER = 0.050
NOISE = 0.0005


def _parse(col: str) -> tuple[str, int, int]:
    import re

    m = re.match(r"^(.*?)(?:\[(\d+):(\d+)\])?$", col)
    if m.group(2) is None:
        return col, 1, 0
    a, b = int(m.group(2)), int(m.group(3))
    return m.group(1), abs(a - b) + 1, min(a, b)


def make_synthetic_db(root, n_cycles: int = 600, seed: int = 0,
                      benches: tuple[str, ...] = ("b1", "b2", "b3")) -> dict:
    """Write a small DB under ``root``; returns per-bit toggle arrays."""
    rng = np.random.default_rng(seed)
    scope_nets = {"top": TOP_NETS, "m1": M1_NETS, "m2": M2_NETS}
    bit_toggles: dict[str, dict[str, np.ndarray]] = {}

    for bench in benches:
        toggles: dict[str, np.ndarray] = {}  # feature name -> {0,1}^n
        frames: dict[str, pd.DataFrame] = {}
        for scope, nets in scope_nets.items():
            cols = {}
            for col, dens in nets:
                path, width, lo = _parse(col)
                bits = (rng.random((n_cycles, width)) < dens).astype(np.uint8)
                for k in range(width):
                    name = path if (width == 1 and "[" not in col) else f"{path}[{lo + k}]"
                    toggles[name] = bits[:, k]
                masks = [int("".join(str(b) for b in row[::-1]), 2) for row in bits]
                cols[col] = pd.Series(masks, dtype=object)
            df = pd.DataFrame(cols)
            df.index = pd.RangeIndex(n_cycles, name="time_ns")
            frames[scope] = df

        # power labels: linear + interaction ground truth, one EXTRA row,
        # float index (mirrors the real DB)
        y = np.full(n_cycles, BASE_POWER)
        for name, w in TRUE_BITS.items():
            y = y + w * toggles[name]
        (a, b), w_int = INTERACTION
        y = y + w_int * (toggles[a] & toggles[b])
        y = y + rng.normal(0.0, NOISE, n_cycles)
        pwr = pd.DataFrame(
            {"Pc(x_aq_core)": np.concatenate([y, [BASE_POWER]]),
             "x_aq_core/Pc(x_m1)": np.full(n_cycles + 1, 0.01)},
        )
        pwr.index = pd.Index(np.arange(n_cycles + 1, dtype=float), name="time_ns")

        (root / "aq_core" / "m1").mkdir(parents=True, exist_ok=True)
        (root / "aq_core" / "m2").mkdir(parents=True, exist_ok=True)
        (root / "pwr").mkdir(parents=True, exist_ok=True)
        frames["top"].to_pickle(root / "aq_core" / f"{bench}_func.pkl")
        frames["m1"].to_pickle(root / "aq_core" / "m1" / f"{bench}_func.pkl")
        frames["m2"].to_pickle(root / "aq_core" / "m2" / f"{bench}_func.pkl")
        pwr.to_pickle(root / "pwr" / f"{bench}_pwr.pkl")
        bit_toggles[bench] = toggles
    return bit_toggles


@pytest.fixture()
def synthetic_db(tmp_path):
    toggles = make_synthetic_db(tmp_path / "db")
    return tmp_path / "db", toggles


@pytest.fixture()
def synthetic_cfg(tmp_path, synthetic_db):
    from cobit.config import CobitConfig

    db_root, toggles = synthetic_db
    cfg = CobitConfig()
    cfg.data.db_root = str(db_root)
    cfg.data.cache_dir = str(tmp_path / "cache")
    cfg.data.chunk_rows = 256  # force multiple chunks
    cfg.split.test_benchmarks = ["b3"]
    cfg.split.val_fraction = 0.2
    cfg.selection.selector = "lasso"
    cfg.selection.target_qs = [6]
    cfg.selection.max_rows = 10_000
    cfg.runtime.output_dir = str(tmp_path / "runs")
    cfg.runtime.seed = 0
    return cfg, toggles
