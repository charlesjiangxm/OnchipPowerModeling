import numpy as np
import pytest

from cobit.data.bitmask import expand_net_column, expand_reference


@pytest.mark.parametrize("width", [1, 7, 31, 64, 65, 128, 311])
def test_expansion_matches_reference(width):
    rng = np.random.default_rng(width)
    values = []
    for _ in range(50):
        bits = rng.random(width) < 0.2
        values.append(sum(1 << k for k in range(width) if bits[k]))
    values += [0, (1 << width) - 1, 1, 1 << (width - 1)]  # edge masks
    vals = np.array(values, dtype=object)

    rows_out, cols_out = [], []
    base_col = 1000
    nnz = expand_net_column(vals, width, base_col, 0, rows_out, cols_out)
    got = {}
    if nnz:
        for r, c in zip(np.concatenate(rows_out), np.concatenate(cols_out)):
            got.setdefault(int(r), []).append(int(c) - base_col)
    for i, v in enumerate(values):
        assert sorted(got.get(i, [])) == expand_reference(int(v), width), (
            f"width={width} row={i}"
        )


def test_row_offset_and_all_zero():
    vals = np.array([0, 0, 5], dtype=object)  # 5 = bits 0 and 2
    rows_out, cols_out = [], []
    nnz = expand_net_column(vals, 3, 10, row_offset=100, rows_out=rows_out, cols_out=cols_out)
    assert nnz == 2
    assert np.concatenate(rows_out).tolist() == [102, 102]
    assert sorted(np.concatenate(cols_out).tolist()) == [10, 12]

    rows_out, cols_out = [], []
    assert expand_net_column(np.array([0, 0], dtype=object), 3, 0, 0, rows_out, cols_out) == 0
    assert rows_out == []


def test_both_code_paths_agree_at_boundary():
    # width 64 uses the uint64 shift path, 65 the bytes path; cross-check a
    # value that fits both widths
    v = (1 << 63) | (1 << 40) | 1
    for width in (64, 65):
        rows_out, cols_out = [], []
        expand_net_column(np.array([v], dtype=object), width, 0, 0, rows_out, cols_out)
        assert sorted(np.concatenate(cols_out).tolist()) == [0, 40, 63]


def test_float_cells_are_coerced_or_rejected():
    """The real DB stores some bitmask cells as floats (verified)."""
    # exact-integer floats coerce silently
    vals = np.array([0.0, 8589934592.0, 3], dtype=object)  # 2^33
    rows_out, cols_out = [], []
    nnz = expand_net_column(vals, 64, 0, 0, rows_out, cols_out)
    assert nnz == 3
    got = sorted(np.concatenate(cols_out).tolist())
    assert got == [0, 1, 33]

    # NaN aborts with context instead of crashing deep in numpy
    with pytest.raises(ValueError, match="corrupt bitmask"):
        expand_net_column(np.array([float("nan")], dtype=object), 4, 0, 0, [], [],
                          context="bench/scope/col")

    # fractional floats are corrupt too
    with pytest.raises(ValueError, match="corrupt bitmask"):
        expand_net_column(np.array([2.5], dtype=object), 4, 0, 0, [], [])

    # >= 2^53 warns but proceeds (upstream precision loss, e.g. 2^63 exactly)
    v63 = 9.223372036854776e18
    rows_out, cols_out = [], []
    expand_net_column(np.array([v63], dtype=object), 64, 0, 0, rows_out, cols_out)
    assert np.concatenate(cols_out).tolist() == [63]
