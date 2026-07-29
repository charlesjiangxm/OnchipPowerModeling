import numpy as np
import pytest

from cobit.data.registry import Net, Registry, parse_column


def test_parse_column():
    assert parse_column("foo") == ("foo", 1, 0)
    assert parse_column("a/b/c[7:0]") == ("a/b/c", 8, 0)
    assert parse_column("x/wb_debug[31:1]") == ("x/wb_debug", 31, 1)
    assert parse_column("y[0:0]") == ("y", 1, 0)


def _reg():
    nets = [
        Net(scope="top", column="a", path="a", width=1, lo=0, base_col=0),
        Net(scope="top", column="b[31:1]", path="b", width=31, lo=1, base_col=1),
        Net(scope="m1", column="x_m1/c[2:0]", path="x_m1/c", width=3, lo=0, base_col=32),
    ]
    return Registry(nets)


def test_feature_names_respect_lo_offset():
    reg = _reg()
    assert reg.n_features == 35
    names = reg.feature_names([0, 1, 31, 32, 34])
    # b's mask bit k names RTL bit lo+k
    assert names == ["a", "b[1]", "b[31]", "x_m1/c[0]", "x_m1/c[2]"]


def test_scalar_net_with_instance_array_brackets_keeps_bare_name():
    """Real DB has 1-bit nets like .../RAM_DIN_VEC[0]/ram_instance/PortAClk."""
    col = "x/RAM_DIN_VEC[0]/ram_instance/PortAClk"
    assert parse_column(col) == (col, 1, 0)
    reg = Registry([Net(scope="top", column=col, path=col, width=1, lo=0, base_col=0)])
    assert reg.feature_names([0]) == [col]  # no phantom [0] suffix


def test_registry_roundtrip(tmp_path):
    reg = _reg()
    reg.save(tmp_path / "nets.json")
    loaded = Registry.load(tmp_path / "nets.json")
    assert loaded.content_hash == reg.content_hash
    assert loaded.feature_names([2]) == reg.feature_names([2])


def test_registry_canonical_order_ignores_pkl_column_order(synthetic_db):
    from cobit.data.discovery import discover
    from cobit.data.registry import build_registry

    db_root, _ = synthetic_db
    reg1 = build_registry(discover(db_root))

    # shuffle the columns of one pkl on disk; canonical order must not change
    import pandas as pd

    p = db_root / "aq_core" / "b1_func.pkl"
    df = pd.read_pickle(p)
    df = df[list(df.columns)[::-1]]
    df.to_pickle(p)
    reg2 = build_registry(discover(db_root))
    assert reg1.content_hash == reg2.content_hash


def test_feature_id_out_of_range():
    reg = _reg()
    with pytest.raises(AssertionError):
        reg.feature_names([35])
    assert np.asarray([0]).dtype.kind in "iu"  # sanity
