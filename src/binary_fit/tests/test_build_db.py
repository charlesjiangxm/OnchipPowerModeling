import numpy as np
import pandas as pd

from binary_fit import build_db
from binary_fit.build_db import parse_column
from binary_fit.data import _find, _FUNC_SUFFIXES


def test_parse_column():
    assert parse_column("top/a") == ("top/a", 1, 0)
    assert parse_column("top/b[3:0]") == ("top/b", 4, 0)
    assert parse_column("top/c[135:64]") == ("top/c", 72, 64)
    assert parse_column("x/RAM[0]/clk") == ("x/RAM[0]/clk", 1, 0)  # no trailing range


def _load_out(cfg, bench):
    return pd.read_pickle(_find(cfg.data.func_dir, bench, _FUNC_SUFFIXES))


def test_build_db_roundtrip_and_schema(synth):
    cfg, _ = synth
    build_db.build(cfg)

    df0 = _load_out(cfg, "b0")
    dft = _load_out(cfg, "tst")

    # canonical column set is identical across benchmarks
    assert list(df0.columns) == list(dft.columns)
    # dtype uint8, int64 index
    assert (df0.dtypes == np.uint8).all()
    assert df0.index.dtype == np.int64

    cols = set(df0.columns)
    # 1-bit net keeps its bare name; buses expand to path[lo+k]
    assert "top/a" in cols
    assert {"top/b[0]", "top/b[1]", "top/b[2]", "top/b[3]"} <= cols
    # live bits of the 72-bit bus: stored 0..3 (RTL 64..67) and 65 (RTL 129)
    assert {"top/c[64]", "top/c[65]", "top/c[66]", "top/c[67]", "top/c[129]"} <= cols
    # globally constant-1 and globally-dead bits are dropped
    assert "top/const1" not in cols
    assert "top/dead[0]" not in cols and "top/dead[1]" not in cols
    # names unique
    assert len(df0.columns) == len(set(df0.columns))

    # bit-exact round trip vs the raw source, incl. the >64-bit lane (bit 65)
    src = pd.read_pickle(_find(f"{cfg.build.source_db_root}/func/m0", "b0", _FUNC_SUFFIXES))
    cvals = [int(v) for v in src["top/c[135:64]"].to_numpy()]
    assert np.array_equal(df0["top/c[67]"].to_numpy(), np.array([(v >> 3) & 1 for v in cvals], np.uint8))
    assert np.array_equal(df0["top/c[129]"].to_numpy(), np.array([(v >> 65) & 1 for v in cvals], np.uint8))
    bvals = [int(v) for v in src["top/b[3:0]"].to_numpy()]
    assert np.array_equal(df0["top/b[0]"].to_numpy(), np.array([v & 1 for v in bvals], np.uint8))


def test_build_db_keep_all_bits(synth):
    cfg, _ = synth
    cfg.build.drop_dead_bits = False
    build_db.build(cfg)
    df0 = _load_out(cfg, "b0")
    # with dropping off, constant/dead columns survive to disk
    assert "top/const1" in df0.columns
    assert {"top/dead[0]", "top/dead[1]"} <= set(df0.columns)
    # full width: a(1) + b(4) + c(72) + const1(1) + dead(2) = 80 columns
    assert df0.shape[1] == 80
