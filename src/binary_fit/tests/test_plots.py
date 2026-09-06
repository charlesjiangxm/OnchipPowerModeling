"""Figure tests that assert on artists, not pixels.

Every test here pins a rule a plausible edit could break; the mutation each one
catches is named in its docstring. Pixel/image-hash baselines are deliberately
absent: they false-positive on every matplotlib bump and name no cause.

The fixture is built to be falsifiable. Benchmark widths are unequal and their
names are NOT in alphabetical order, so iterating ``sorted(bench_slices)`` or
using ``sl.start`` for ``sl.stop`` moves a boundary. ``yhat`` is an asymmetric
function of ``y`` with a strictly wider range, so an identity line computed from
the labels alone is too short.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from binary_fit import plots

_SLICES = {"b_wide": slice(0, 7), "b_mid": slice(7, 10), "b_thin": slice(10, 12)}
_N = 12


def _trace(n: int = _N):
    y = np.linspace(0.040, 0.062, n)
    return y, 2.0 * y - 0.030  # asymmetric: range strictly wider than y's


def _lines(ax, label):
    return [l for l in ax.lines if l.get_label() == label]


# --------------------------------------------------------------- pred_vs_time --
def test_pred_vs_time_marks_interior_boundaries_at_slice_stops(tmp_path):
    """Boundaries sit at the slice STOPS, in dict order, last one suppressed.

    Catches sl.start for sl.stop, sorted(bench_slices), and drawing the final
    boundary on the axes edge.
    """
    y, yhat = _trace()
    fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                  name="x", window_size=8)
    ax = fig.axes[0]
    xs = [l.get_xdata()[0] for l in _lines(ax, "_bench_boundary")]
    assert xs == [7, 10]  # 12 is the axes edge under margins(x=0)


def test_pred_vs_time_labels_every_benchmark_at_its_midpoint(tmp_path):
    """All benchmarks are named, rotated 90 deg, centred on their own slice.

    Catches a dropped label for narrow slices and a wrong midpoint formula.
    """
    y, yhat = _trace()
    fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                  name="x", window_size=8)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["b_wide", "b_mid", "b_thin"]
    assert [t.get_position()[0] for t in ax.texts] == [3.5, 8.5, 11.0]
    assert {t.get_rotation() for t in ax.texts} == {90}


def test_pred_vs_time_keeps_the_legend_clear_of_the_labels(tmp_path):
    """A headroom strip above the data holds the legend.

    Without it the upper-right legend covers the rightmost benchmark labels --
    which is what happens in the x-opm original. Catches a removed set_ylim and
    labels placed in the strip instead of below it.
    """
    y, yhat = _trace()
    fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                  name="x", window_size=8)
    ax = fig.axes[0]
    top = ax.get_ylim()[1]
    data_top = max(y.max(), yhat.max()) * 1000.0
    assert top > data_top  # a strip exists
    assert all(t.get_position()[1] < top for t in ax.texts)  # labels sit below it


def test_pred_vs_time_plots_mw_against_the_row_index(tmp_path):
    """y is scaled to mW and x is the row index; the label is the ylabel.

    Catches a dropped or doubled POWER_SCALE and an x/y swap.
    """
    y, yhat = _trace()
    fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                  name="x", window_size=8)
    ax = fig.axes[0]
    true, pred = _lines(ax, "true")[0], _lines(ax, "predicted")[0]
    np.testing.assert_array_equal(true.get_xdata(), np.arange(_N))
    np.testing.assert_allclose(true.get_ydata(), y * 1000.0)
    np.testing.assert_allclose(pred.get_ydata(), yhat * 1000.0)
    assert ax.get_ylabel() == "power (mW)"


@pytest.mark.parametrize("window_size,expected", [
    (1, "cycle index (concatenated benchmarks)"),
    (8, "8-cycle window index (concatenated benchmarks)"),
    (32, "32-cycle window index (concatenated benchmarks)"),
])
def test_pred_vs_time_axis_unit_follows_window_size(tmp_path, window_size, expected):
    """The x unit names window_size; catches an inverted `> 1` branch."""
    y, yhat = _trace()
    fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                  name="x", window_size=window_size)
    assert fig.axes[0].get_xlabel() == expected


def test_pred_vs_time_requires_window_size(tmp_path):
    """No default for window_size: the package default is 32, so a forgotten
    argument would silently label 32-cycle rows "cycle index"."""
    y, yhat = _trace()
    with pytest.raises(TypeError):
        plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png", name="x")


def test_pred_vs_time_blanks_non_finite_without_shifting_rows(tmp_path, caplog):
    """A NaN label is blanked in place, never dropped.

    Catches filtering the bad rows out, which would shorten the series and pull
    every benchmark boundary off its true row index.
    """
    y, yhat = _trace()
    y = y.copy(); y[4] = np.nan
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                      name="x", window_size=8)
    ax = fig.axes[0]
    true = _lines(ax, "true")[0]
    np.testing.assert_array_equal(true.get_xdata(), np.arange(_N))
    assert np.isnan(true.get_ydata()[4])
    assert [l.get_xdata()[0] for l in _lines(ax, "_bench_boundary")] == [7, 10]
    assert any("non-finite" in r.getMessage() for r in caplog.records)


def test_pred_vs_time_warns_when_slices_do_not_tile_the_split(tmp_path, caplog):
    """Passing another split's slices is otherwise invisible: numpy truncates an
    over-long slice silently. Warn, but still draw."""
    y, yhat = _trace(20)
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        fig = plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png",
                                      name="x", window_size=8)
    assert fig is not None
    assert any("cover 12 of 20" in r.getMessage() for r in caplog.records)


def test_pred_vs_time_skips_empty_slices(tmp_path):
    """A zero-width slice gets neither a boundary nor a label."""
    y, yhat = _trace()
    slices = {**_SLICES, "b_empty": slice(7, 7)}
    fig = plots.plot_pred_vs_time(y, yhat, slices, tmp_path / "t.png",
                                  name="x", window_size=8)
    ax = fig.axes[0]
    assert len(ax.texts) == 3
    assert len(_lines(ax, "_bench_boundary")) == 2


def test_pred_vs_time_skips_an_empty_split(tmp_path, caplog):
    """An empty split writes nothing and says so, rather than raising."""
    empty = np.empty(0)
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        assert plots.plot_pred_vs_time(empty, empty, {}, tmp_path / "t.png",
                                       name="x", window_size=8) is None
    assert not (tmp_path / "t.png").exists()
    assert any("empty" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------ residual panels --
def test_residual_panel_order_is_train_val_test(tmp_path):
    """Panels follow the split order, not the dict insertion order, and empty
    splits are skipped (subplots(1, 0) raises)."""
    y, yhat = _trace()
    preds = {"test": (y, yhat), "train": (y, yhat), "val": (y, yhat)}
    fig = plots.plot_residual_panels(preds, tmp_path / "r.png", name="x")
    assert [ax.get_title() for ax in fig.axes] == ["train", "val", "test"]

    preds = {"train": (y, yhat), "val": (np.empty(0), np.empty(0)), "test": (y, yhat)}
    fig = plots.plot_residual_panels(preds, tmp_path / "r2.png", name="x")
    assert [ax.get_title() for ax in fig.axes] == ["train", "test"]


def test_residual_returns_none_when_every_split_is_empty(tmp_path, caplog):
    """Guards subplots(1, 0), reachable via an empty split.test_benchmarks."""
    empty = np.empty(0)
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        assert plots.plot_residual_panels({"test": (empty, empty)},
                                          tmp_path / "r.png", name="x") is None
    assert not (tmp_path / "r.png").exists()


def test_residual_x_is_the_label_and_y_is_the_prediction(tmp_path):
    """Catches an axis swap; the asymmetric yhat makes it detectable."""
    y, yhat = _trace()
    fig = plots.plot_residual_panels({"train": (y, yhat)}, tmp_path / "r.png", name="x")
    ax = fig.axes[0]
    off = ax.collections[0].get_offsets()
    np.testing.assert_allclose(off[:, 0], y * 1000.0)
    np.testing.assert_allclose(off[:, 1], yhat * 1000.0)
    assert ax.get_xlabel() == "true power (mW)"
    assert ax.get_ylabel() == "predicted power (mW)"


def test_residual_identity_line_spans_both_series(tmp_path):
    """y = x must cover the prediction range too.

    Catches `lo, hi = y.min(), y.max()`: yhat = 2y - 0.03 overshoots both ends,
    so a labels-only line would stop short.
    """
    y, yhat = _trace()
    fig = plots.plot_residual_panels({"train": (y, yhat)}, tmp_path / "r.png", name="x")
    line = _lines(fig.axes[0], "_identity")[0]
    xd, yd = line.get_xdata(), line.get_ydata()
    np.testing.assert_allclose(xd, yd)  # it really is y = x
    assert xd[0] <= min(y.min(), yhat.min()) * 1000.0
    assert xd[-1] >= max(y.max(), yhat.max()) * 1000.0


def test_residual_identity_line_is_visible_on_a_flat_split(tmp_path):
    """A constant-power split still gets a drawable line, not a zero-length one."""
    flat = np.full(8, 0.050)
    fig = plots.plot_residual_panels({"train": (flat, flat)}, tmp_path / "r.png",
                                     name="x")
    xd = _lines(fig.axes[0], "_identity")[0].get_xdata()
    assert xd[1] > xd[0]


def test_residual_drops_non_finite_pairs(tmp_path, caplog):
    """Catches the reference's builtin min()/max(): a NaN label there makes
    lo = hi = nan and the y = x line silently disappears."""
    y, yhat = _trace()
    y = y.copy(); y[2] = np.nan
    yhat = yhat.copy(); yhat[5] = np.inf
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        fig = plots.plot_residual_panels({"train": (y, yhat)}, tmp_path / "r.png",
                                         name="x")
    ax = fig.axes[0]
    assert ax.collections[0].get_offsets().shape == (_N - 2, 2)
    assert np.isfinite(_lines(ax, "_identity")[0].get_xdata()).all()
    assert any("non-finite" in r.getMessage() for r in caplog.records)


def test_residual_marks_in_sample_panels_with_their_reason(tmp_path):
    """Which panels are in-sample depends on the mode, so the reason is printed.

    Without HPO only train is training data; with HPO the model is refit on
    train+val, so val is too. Unlabelled, that val panel reads as held-out
    evidence. Catches a hardcoded note applied to the wrong mode.
    """
    y, yhat = _trace()
    preds = {"train": (y, yhat), "val": (y, yhat), "test": (y, yhat)}
    fig = plots.plot_residual_panels(
        preds, tmp_path / "hpo.png", name="x",
        in_sample={"train": "in HPO refit", "val": "in HPO refit"})
    assert [ax.get_title() for ax in fig.axes] == [
        "train (in HPO refit)", "val (in HPO refit)", "test"]

    fig = plots.plot_residual_panels(preds, tmp_path / "nohpo.png", name="x",
                                     in_sample={"train": "in-sample"})
    assert [ax.get_title() for ax in fig.axes] == ["train (in-sample)", "val", "test"]

    fig = plots.plot_residual_panels(preds, tmp_path / "plain.png", name="x")
    assert [ax.get_title() for ax in fig.axes] == ["train", "val", "test"]


# ------------------------------------------------------------------- contracts --
@pytest.mark.parametrize("fn", ["pred_vs_time", "residual"])
def test_plot_functions_reject_mismatched_lengths(tmp_path, fn):
    """A wrong-length prediction is a bug, not something to plot around."""
    y, yhat = _trace()
    with pytest.raises(ValueError, match="differ"):
        if fn == "pred_vs_time":
            plots.plot_pred_vs_time(y, yhat[:-1], _SLICES, tmp_path / "t.png",
                                    name="x", window_size=8)
        else:
            plots.plot_residual_panels({"train": (y, yhat[:-1])},
                                       tmp_path / "r.png", name="x")


def test_figures_are_written_as_real_pngs(tmp_path):
    """The artists are only worth asserting on if the file also lands."""
    y, yhat = _trace()
    plots.plot_pred_vs_time(y, yhat, _SLICES, tmp_path / "t.png", name="x",
                            window_size=8)
    plots.plot_residual_panels({"train": (y, yhat)}, tmp_path / "r.png", name="x")
    for f in ("t.png", "r.png"):
        assert (tmp_path / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
