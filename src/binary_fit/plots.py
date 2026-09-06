"""Figures written after a fit: parity panels, prediction traces, the Q sweep.

Rendering lives apart from ``evaluate.py`` (the metric protocol) so importing
the metrics never pulls in matplotlib. Every function takes raw arrays, writes
``out_path``, closes its figure and *returns* it: the return value exists for
the tests, which assert on artists (lines, texts, offsets) rather than pixels.
``None`` means "nothing to draw, nothing written".

Units: the target and ``result.json`` are in watts, but power is *plotted* in
mW (``POWER_SCALE``) so the ticks are legible and the figures line up with
``analysis/x-opm/<ts>/<win>/<module>/``, whose layout these two figure types
are ported from (``src/xopm_lib/model_regression.py``).

No figure decimates its data. A power trace's spikes are exactly what
``eval.peak_window`` / ``report["peak"]`` score, and stride sampling deletes
them; measured, the full 2.04M-row worst case (``window_size = 1``) still
renders in ~3 s at ~56 KiB, and a sparse polyline does not even compress
smaller than the saturated one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import log

_BLUE, _ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1-2
_RED = "#c1121f"  # the y = x reference line
POWER_SCALE, POWER_UNIT = 1000.0, "mW"  # every Pc(*) target is in watts
_SPLIT_ORDER = ("train", "val", "test")
_LEGEND_HEADROOM = 0.16  # fraction of the y range kept clear above the data


def _plt():
    """``matplotlib.pyplot`` on the Agg backend, imported lazily."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axis(ax) -> None:
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _finite_pair(y, yhat, what: str):
    """``(y, yhat)`` as float64 mW plus a finite mask; a length mismatch is fatal.

    Non-finite points must be excluded explicitly. Taking the identity line
    from ``min(y.min(), yhat.min())`` -- the builtins -- returns ``nan`` when
    the nan comes first, so one NaN label silently erases the y = x line while
    the scatter still looks plausible.
    """
    y = np.asarray(y, dtype=float) * POWER_SCALE
    yhat = np.asarray(yhat, dtype=float) * POWER_SCALE
    if y.shape != yhat.shape:
        raise ValueError(f"{what}: y {y.shape} and yhat {yhat.shape} differ")
    ok = np.isfinite(y) & np.isfinite(yhat)
    n_bad = int(ok.size) - int(ok.sum())
    if n_bad:
        log.warning("%s: %d/%d non-finite point(s) excluded from the figure",
                    what, n_bad, int(ok.size))
    return y, yhat, ok


def _check_slices(bench_slices: dict[str, slice], n: int, what: str) -> None:
    """Warn (never raise) when the benchmark slices do not tile ``[0, n)``.

    ``run._run_one`` pairs three slice dicts with three label vectors, and numpy
    truncates an over-long slice silently (``np.arange(10)[slice(5, 50)]`` is
    five elements, no error), so a crossed pair would otherwise show up only as
    a subtly wrong figure.
    """
    spans = [sl.indices(n)[:2] for sl in bench_slices.values()]
    if not spans:
        return
    covered = sum(hi - lo for lo, hi in spans)
    top = max(hi for _, hi in spans)
    if covered != n or top != n:
        log.warning("%s: benchmark slices cover %d of %d row(s) (max stop %d) - "
                    "wrong split's slices?", what, covered, n, top)


def plot_pred_vs_time(
    y: np.ndarray,
    yhat: np.ndarray,
    bench_slices: dict[str, slice],
    out_path: Path,
    name: str,
    window_size: int,
):
    """Label and prediction (mW) across one split's concatenated benchmarks.

    One row is one prediction: a cycle at ``data.window_size = 1``, otherwise a
    window of that many cycles. ``window_size`` is required rather than
    defaulted -- the package default is 32, so a forgotten argument would label
    32-cycle rows "cycle index" and the figure would lie with no error.

    Benchmark boundaries are dotted and named. Only interior boundaries get a
    line: under ``margins(x=0)`` the last one falls on the axes edge, where it
    reads as a stray artifact rather than a boundary.
    """
    plt = _plt()
    what = f"pred_vs_time [{name}]"
    y, yhat, ok = _finite_pair(y, yhat, what)
    n = int(y.size)
    if not n:
        log.warning("%s: split is empty - %s not written", what, out_path)
        return None
    _check_slices(bench_slices, n, what)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        # blank the bad points rather than dropping them: dropping would shorten
        # the series and pull every boundary off its true row index
        ax.plot(x, np.where(ok, y, np.nan), color=_BLUE, linewidth=0.8, alpha=0.9,
                label="true")
        ax.plot(x, np.where(ok, yhat, np.nan), color=_ORANGE, linewidth=0.8,
                alpha=0.9, label="predicted")
        # reserve a strip above the data for the legend, so it cannot sit on top
        # of the benchmark labels of the last benchmarks (which it does in the
        # x-opm original -- see analysis/x-opm/.../pred_vs_time_train.png)
        ylo, ytop = ax.get_ylim()
        ax.set_ylim(ylo, ytop + _LEGEND_HEADROOM * (ytop - ylo))
        for bench, sl in bench_slices.items():
            lo, hi = sl.indices(n)[:2]
            if hi <= lo:
                continue
            if hi < n:
                ax.axvline(hi, color="grey", linewidth=0.4, linestyle=":",
                           label="_bench_boundary")
            ax.text(0.5 * (lo + hi), ytop, bench, fontsize=6, ha="center",
                    va="top", rotation=90, color="grey")
        unit = (f"{window_size}-cycle window index" if window_size > 1
                else "cycle index")
        ax.set_xlabel(f"{unit} (concatenated benchmarks)")
        ax.set_ylabel(f"power ({POWER_UNIT})")
        ax.set_title(f"Power vs time [{name}]", fontsize=11)
        ax.margins(x=0)
        _style_axis(ax)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)  # unconditional: a half-drawn Figure must not leak
    return fig


def plot_residual_panels(
    preds: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    name: str,
    in_sample: dict[str, str] | None = None,
):
    """Parity map (mW): predicted vs true power, one panel per split, red y = x.

    ``preds`` maps a split name to its ``(y, yhat)`` pair. Panels are ordered
    train / val / test whatever the insertion order, and empty splits are
    skipped -- ``subplots(1, 0)`` raises.

    ``in_sample`` maps a split to the *reason* its panel is not held-out
    evidence, and the reason differs by mode: without HPO only train is
    in-sample, but with HPO the final model is refit on ``vstack([Xtr, Xval])``
    (``run._fit_tree`` / ``run._fit_nn``) so val is training data too.
    Unlabelled, that val panel looks like generalization and gets quoted as
    such, which is why the reason is printed rather than implied.

    The frame stays plain (no ``_style_axis``): a parity cloud is read against a
    closed box, and a grid under a dense translucent scatter only muddies it.
    """
    plt = _plt()
    in_sample = in_sample or {}
    order = [s for s in _SPLIT_ORDER if s in preds and np.size(preds[s][0])]
    if not order:
        log.warning("residual [%s]: every split is empty - %s not written",
                    name, out_path)
        return None
    # every pair is validated BEFORE a Figure exists: _finite_pair raises on a
    # length mismatch, and a raise between subplots() and close() leaks a Figure
    panels = []
    for split in order:
        y, yhat, ok = _finite_pair(*preds[split], f"residual [{name} | {split}]")
        panels.append((split, y[ok], yhat[ok]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2),
                             squeeze=False)
    try:
        for ax, (split, y, yhat) in zip(axes[0], panels):
            ax.scatter(y, yhat, s=2, alpha=0.3, color=_BLUE)
            lo = float(min(y.min(), yhat.min())) if y.size else 0.0
            hi = float(max(y.max(), yhat.max())) if y.size else 1.0
            span = hi - lo
            # pad so the line stays visible on a split whose power is flat
            pad = 0.05 * span if span > 0 else max(abs(hi) * 0.05, 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_RED,
                    linewidth=0.9, linestyle="--", label="_identity")  # y = x
            ax.set_xlabel(f"true power ({POWER_UNIT})")
            ax.set_ylabel(f"predicted power ({POWER_UNIT})")
            note = in_sample.get(split)
            ax.set_title(f"{split} ({note})" if note else split)
        fig.suptitle(f"Predicted vs true power [{name}]", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return fig


def plot_q_sweep(records: list[dict], out_path: Path):
    """MAPE and R2 versus proxy count Q (two panels, one axis each)."""
    plt = _plt()
    recs = sorted((r for r in records if r.get("test_mape") is not None),
                  key=lambda r: r["q"])
    if not recs:
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    qs = [r["q"] for r in recs]
    mapes = [r["test_mape"] for r in recs]
    r2s = [r["test_r2"] for r in recs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    try:
        ax1.plot(qs, mapes, color=_BLUE, marker="o", markersize=5, linewidth=1.6)
        ax1.set_xlabel("number of bitwise proxies Q")
        ax1.set_ylabel("test MAPE (%)")
        ax2.plot(qs, r2s, color=_BLUE, marker="o", markersize=5, linewidth=1.6)
        ax2.set_xlabel("number of bitwise proxies Q")
        ax2.set_ylabel("test R$^2$")
        for ax in (ax1, ax2):
            _style_axis(ax)
        windows = {int(r.get("window_size", 1)) for r in recs}
        if windows == {1}:
            unit = "Per-cycle"
        elif len(windows) == 1:
            unit = f"{windows.pop()}-cycle-window"
        else:
            unit = f"Mixed-window ({sorted(windows)})"
        fig.suptitle(f"{unit} accuracy vs proxy count", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return fig
