import numpy as np
import pandas as pd
import pytest

from binary_fit import build_db, data
from binary_fit.config import Config
from binary_fit.utils import save_pickle_zst


def test_pwr_alignment_drops_trailing_row(synth):
    """load_aligned must reindex pwr (n+1 rows) onto the func index (n rows).

    The synthetic pwr carries a -999.0 sentinel as its extra trailing cycle; if
    the off-by-one is mishandled it leaks into the labels.
    """
    cfg, _ = synth
    build_db.build(cfg)
    func, y = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, "b0", cfg.data.target)
    assert y.shape[0] == func.shape[0]
    assert not np.any(y == -999.0)  # trailing sentinel dropped
    assert np.all(y > 0)


def test_variance_kept_filter(synth):
    """The kept filter drops all-zero AND constant-1 columns (0 < ones < n)."""
    cfg, planted = synth
    cfg.build.drop_dead_bits = False  # keep const1/dead on disk so the loader must filter them
    build_db.build(cfg)
    bundle = data.load_split(cfg)

    col_of = {c: i for i, c in enumerate(bundle.columns)}
    kept = set(bundle.kept_ids.tolist())
    assert col_of["top/const1"] not in kept  # constant-1 -> excluded
    assert col_of["top/dead[0]"] not in kept  # all-zero -> excluded
    for name in planted:
        assert col_of[name] in kept  # informative bits survive

    n_train = bundle.X_train.shape[0]
    counts = np.asarray(bundle.X_train.sum(axis=0)).ravel()
    assert counts[col_of["top/const1"]] == n_train  # was indeed constant-1
    assert counts[col_of["top/dead[0]"]] == 0


def test_load_split_shapes(synth):
    cfg, _ = synth
    build_db.build(cfg)
    bundle = data.load_split(cfg)
    assert bundle.test_benches == ["tst"]
    assert set(bundle.train_benches) == {"b0", "b1", "b2"}
    assert bundle.X_train.shape[0] == bundle.y_train.size
    assert bundle.X_val.shape[0] == bundle.y_val.size
    assert bundle.X_train.shape[1] == len(bundle.columns)
    assert bundle.window_size == 1
    assert bundle.X_train.dtype == np.uint8  # per-cycle rows stay binary


# --------------------------------------------------------------------------- #
# window averaging (data.window_size)
# --------------------------------------------------------------------------- #
def test_window_average_means_and_drops_tail():
    """Non-overlapping block means over axis 0; the partial tail is dropped."""
    a = np.arange(14, dtype=np.uint8).reshape(7, 2)  # 7 rows, window 3 -> 2 rows
    out = data.window_average(a, 3)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[2.0, 3.0], [8.0, 9.0]])  # rows 0-2 and 3-5
    y = np.arange(7, dtype=np.float64)
    np.testing.assert_allclose(data.window_average(y, 3), [1.0, 4.0])
    assert data.n_windows(7, 3) == 2


def test_window_average_identity_at_one():
    a = np.arange(6, dtype=np.uint8).reshape(3, 2)
    assert data.window_average(a, 1) is a  # no copy, no cast
    with pytest.raises(ValueError):
        data.window_average(a, 0)


def test_window_one_keeps_the_raw_per_cycle_bits(synth):
    """window_size=1 must leave the design matrix the untouched per-cycle 0/1 bits."""
    cfg, _ = synth
    build_db.build(cfg)
    bundle = data.load_split(cfg)
    for b in bundle.train_benches:
        func, y = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, b, cfg.data.target)
        bits = func[bundle.columns].to_numpy(dtype=np.uint8)
        sl = bundle.train_slices[b]
        n_tr = sl.stop - sl.start
        np.testing.assert_array_equal(bundle.X_train[sl].toarray(), bits[:n_tr])
        np.testing.assert_array_equal(bundle.y_train[sl], y[:n_tr])
        vsl = bundle.val_slices[b]
        np.testing.assert_array_equal(bundle.X_val[vsl].toarray(), bits[n_tr:])
        np.testing.assert_array_equal(bundle.y_val[vsl], y[n_tr:])


def _cycles_of(cfg, bench):
    func, _ = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, bench, cfg.data.target)
    return func.shape[0]


def test_windowed_split_row_counts_and_densities(synth_windowed):
    """Rows become windows, features become densities in [0, 1], labels means."""
    cfg, _ = synth_windowed
    build_db.build(cfg)
    w = cfg.data.window_size
    bundle = data.load_split(cfg)

    n_win = _cycles_of(cfg, "b0") // w
    n_val_win = int(n_win * cfg.split.val_fraction)
    n_tr_win = n_win - n_val_win
    assert bundle.window_size == w
    assert bundle.X_train.shape[0] == 3 * n_tr_win == bundle.y_train.size
    assert bundle.X_val.shape[0] == 3 * n_val_win == bundle.y_val.size
    assert bundle.X_train.dtype == np.float32

    vals = bundle.X_train[:, bundle.kept_ids].toarray()
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert np.any((vals > 0.0) & (vals < 1.0))  # genuinely averaged, not binary
    np.testing.assert_allclose(vals * w, np.round(vals * w), atol=1e-6)  # multiples of 1/w


def test_windowed_rows_match_manual_average(synth_windowed):
    """The first training row equals the per-cycle mean of its own w cycles."""
    cfg, _ = synth_windowed
    build_db.build(cfg)
    w = cfg.data.window_size
    bundle = data.load_split(cfg)
    b0 = sorted(bundle.train_benches)[0]

    func, y = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, b0, cfg.data.target)
    bits = func[bundle.columns].to_numpy(dtype=np.float64)
    sl = bundle.train_slices[b0]
    got_x = np.asarray(bundle.X_train[sl.start].todense()).ravel()
    np.testing.assert_allclose(got_x, bits[:w].mean(axis=0), atol=1e-6)
    assert bundle.y_train[sl.start] == pytest.approx(y[:w].mean())


def test_kept_bits_stable_when_the_counted_cycles_match(synth):
    """min_toggle_count counts per-cycle bits, so the kept set is window-stable
    whenever the two splits cover the same training cycles.

    window 5 makes them match exactly: 60 cycles -> 12 windows -> 9 train windows
    = 45 cycles at val_fraction 0.25, which is also the per-cycle split's 45 train
    rows. (Stability is only up to that span: a window boundary can drop a few
    tail cycles, and a bit hot only in those would differ -- see
    test_kept_filter_counts_training_cycles_only for the rule itself.)
    """
    cfg, planted = synth
    cfg.build.drop_dead_bits = False  # const1/dead reach the loader's filter
    build_db.build(cfg)
    per_cycle = data.load_split(cfg)
    cfg.data.window_size = 5
    windowed = data.load_split(cfg)

    # 3 training benchmarks: 3*45 per-cycle rows vs 3*9 window rows, same cycles
    assert per_cycle.X_train.shape[0] == 135 and windowed.X_train.shape[0] == 27
    assert per_cycle.kept_ids.tolist() == windowed.kept_ids.tolist()
    col_of = {c: i for i, c in enumerate(windowed.columns)}
    kept = set(windowed.kept_ids.tolist())
    assert col_of["top/const1"] not in kept and col_of["top/dead[0]"] not in kept
    for name in planted:
        assert col_of[name] in kept


# --------------------------------------------------------------------------- #
# A hand-built single-bit dataset whose bits pin each windowing rule, so a
# mutation of load_split has to break a test. Written straight into the flat
# func_dir/pwr_dir layout (no --build_db needed).
#
# window 4, benchmarks of 66/67/65 cycles: each gives 16 windows -> 12 train + 4
# val, so the training cycles are exactly 0..47 and validation starts at cycle 48.
# The differing cycle counts make sum-of-floors (48 rows) != floor-of-sum (49),
# which is what catches windows formed after concatenating benchmarks.
# --------------------------------------------------------------------------- #
_PIN_W = 4
_PIN_CYCLES = {"b0": 66, "b1": 67, "b2": 65, "tst": 66}
_PIN_TRAIN_WIN, _PIN_VAL_WIN = 12, 4
_PIN_VAL_START = 48  # first validation CYCLE when the cut is window-aligned


def _pin_columns(n: int) -> dict:
    """Bits chosen so each kept-filter rule has an observable consequence."""
    c = np.arange(n)
    return {
        "top/x": ((c // _PIN_W) % 2 == 0).astype(np.uint8),      # density 1 or 0 per window
        "top/frac": ((c % 8) < 3).astype(np.uint8),              # density 0.75 or 0
        "top/half": (c % 2 == 0).astype(np.uint8),               # density 0.5 in EVERY window
        "top/rare": (c == 3).astype(np.uint8),                   # one training cycle only
        # validation-side at BOTH window 4 (val = cycles 48..63 for every
        # benchmark) and window 1 (val starts at cycle 49/50/51 depending on the
        # benchmark's length), so the same assertion holds at either window
        "top/valonly": ((c == 55) | (c == 56)).astype(np.uint8),
        "top/const1": np.ones(n, dtype=np.uint8),
        "top/dead": np.zeros(n, dtype=np.uint8),
    }


def _pin_dataset(tmp_path, window: int = _PIN_W) -> Config:
    root = tmp_path / "pin"
    (root / "func").mkdir(parents=True, exist_ok=True)
    (root / "pwr").mkdir(parents=True, exist_ok=True)
    for b, n in _PIN_CYCLES.items():
        cols = _pin_columns(n)
        save_pickle_zst(pd.DataFrame(cols, index=np.arange(n, dtype=np.int64)),
                        root / "func" / f"{b}_func.pkl.zst")
        y = (0.5 + 0.3 * cols["top/x"] + 0.1 * cols["top/frac"]
             + 0.001 * (np.arange(n) % 7)).astype(np.float64)
        save_pickle_zst(  # one surplus trailing cycle, float index, as on disk
            pd.DataFrame({"Pc(x_aq_core)": np.concatenate([y, [-999.0]])},
                         index=np.arange(n + 1, dtype=float)),
            root / "pwr" / f"{b}_pwr.pkl.zst")
    cfg = Config()
    cfg.data.func_dir = str(root / "func")
    cfg.data.pwr_dir = str(root / "pwr")
    cfg.data.window_size = window
    cfg.split.test_benchmarks = ["tst"]
    cfg.split.val_fraction = 0.25
    cfg.selection.max_rows = 0
    return cfg


def _col(bundle, name):
    return bundle.columns.index(name)


def test_train_val_cut_lands_on_a_window_boundary(tmp_path):
    """The split is taken on whole windows, not on cycles that are then averaged.

    Cutting cycles first would put validation at cycle 50 for the 67-cycle
    benchmark (int(67*0.25) = 16 held out) instead of at cycle 48.
    """
    cfg = _pin_dataset(tmp_path)
    bundle = data.load_split(cfg)
    w = _PIN_W
    checked = 0
    for b in bundle.train_benches:
        tsl, vsl = bundle.train_slices[b], bundle.val_slices[b]
        assert tsl.stop - tsl.start == _PIN_TRAIN_WIN
        assert vsl.stop - vsl.start == _PIN_VAL_WIN
        func, y = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, b, cfg.data.target)
        bits = func[bundle.columns].to_numpy(dtype=np.float64)
        s = _PIN_VAL_START
        np.testing.assert_allclose(np.asarray(bundle.X_val[vsl.start].todense()).ravel(),
                                   bits[s:s + w].mean(axis=0), atol=1e-6)
        assert bundle.y_val[vsl.start] == pytest.approx(y[s:s + w].mean())
        # ... and the last training row is the window that ends right before it
        np.testing.assert_allclose(np.asarray(bundle.X_train[tsl.stop - 1].todense()).ravel(),
                                   bits[s - w:s].mean(axis=0), atol=1e-6)
        checked += 1
    assert checked == 3  # all three training benchmarks were actually inspected


def test_windows_never_span_two_benchmarks(tmp_path):
    """Row counts are sum-of-floors, not floor-of-sum over the concatenated trace."""
    cfg = _pin_dataset(tmp_path)
    bundle = data.load_split(cfg)
    per_bench = [data.n_windows(_PIN_CYCLES[b], _PIN_W) for b in bundle.train_benches]
    assert sum(per_bench) == 48
    assert data.n_windows(sum(_PIN_CYCLES[b] for b in bundle.train_benches), _PIN_W) == 49
    assert bundle.X_train.shape[0] + bundle.X_val.shape[0] == 48
    assert bundle.X_train.shape[0] == 3 * _PIN_TRAIN_WIN


@pytest.mark.parametrize("window", [_PIN_W, 1])
def test_kept_filter_counts_training_cycles_only(tmp_path, window):
    """A bit that is 1 only in validation cycles has no training signal -> dropped.

    Counting ones over all cycles (or over the averaged rows) would keep
    top/valonly; counting the averaged densities would drop top/rare, whose one
    hot cycle averages to 0.25 < min_toggle_count. window 1 matters as much as
    window 4 here: with no averaging the post-averaging flat filter is skipped,
    so the counting range is the only thing keeping top/valonly out.
    """
    cfg = _pin_dataset(tmp_path, window=window)
    bundle = data.load_split(cfg)
    kept = set(bundle.kept_ids.tolist())
    assert _col(bundle, "top/valonly") not in kept
    assert _col(bundle, "top/rare") in kept
    assert _col(bundle, "top/const1") not in kept
    assert _col(bundle, "top/dead") not in kept
    for name in ("top/x", "top/frac"):
        assert _col(bundle, name) in kept


def test_constant_after_averaging_is_dropped_only_when_windowing(tmp_path):
    """top/half toggles every cycle, so every 4-cycle window has density 0.5."""
    cfg = _pin_dataset(tmp_path, window=_PIN_W)
    windowed = data.load_split(cfg)
    half = _col(windowed, "top/half")
    col = windowed.X_train[:, [half]].toarray().ravel()
    np.testing.assert_allclose(col, 0.5)  # genuinely constant in the averaged matrix
    assert half not in set(windowed.kept_ids.tolist())

    cfg.data.window_size = 1
    per_cycle = data.load_split(cfg)
    assert half in set(per_cycle.kept_ids.tolist())  # a normal toggling bit per cycle


def test_zero_validation_windows_leaves_an_empty_val_matrix(tmp_path):
    """A window big enough that int(n_win * val_fraction) == 0 must not crash."""
    cfg = _pin_dataset(tmp_path, window=30)  # 65..67 cycles -> 2 windows -> 0 val
    bundle = data.load_split(cfg)
    assert bundle.X_train.shape[0] == 6 and bundle.val_slices == {}
    assert bundle.X_val.shape == (0, len(bundle.columns))
    assert bundle.X_val.dtype == bundle.X_train.dtype
    assert bundle.y_val.size == 0


def test_window_longer_than_trace_is_an_error(synth):
    cfg, _ = synth
    build_db.build(cfg)
    cfg.data.window_size = 10_000  # far more cycles than the fixture writes
    with pytest.raises(RuntimeError, match="shorter than data.window_size"):
        data.load_split(cfg)


def test_featurize_test_windows_the_test_benchmarks(synth_windowed):
    cfg, _ = synth_windowed
    build_db.build(cfg)
    w = cfg.data.window_size
    bundle = data.load_split(cfg)
    n_win = _cycles_of(cfg, "tst") // w
    ids = bundle.kept_ids[:3]
    X, y, slices = data.featurize_test(cfg, bundle.columns, ids)
    assert X.shape == (n_win, ids.size) and y.size == n_win
    assert slices["tst"] == slice(0, n_win)

    func, y_cyc = data.load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, "tst", cfg.data.target)
    names = [bundle.columns[int(i)] for i in ids]
    np.testing.assert_allclose(X[0], func[names].to_numpy(dtype=np.float64)[:w].mean(axis=0),
                               atol=1e-6)
    assert y[0] == pytest.approx(y_cyc[:w].mean())
