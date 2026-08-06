"""Evaluation protocol: MAPE / R2, Peak-Power sets, APET success, multicycle.

Peak-Power set construction follows the paper: the continuous power trace of
each (test) benchmark is partitioned into windows; cycles whose power exceeds
window mean + 3 sigma are peaks. APET success rate = fraction of cycles whose
absolute percentage error falls below a threshold.

Multicycle prediction (per-cycle model): predictions and labels are averaged
over non-overlapping t-cycle windows. Documented deviation from the paper:
the paper re-measures window labels with its power tool; here only per-cycle
labels exist, so window labels are their means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CobitConfig
from .utils import log


# ---------------------------------------------------------------- metrics --
def mape_percent(
    y: np.ndarray, yhat: np.ndarray, eps_frac: float = 1e-3
) -> tuple[float, int]:
    """Masked MAPE in percent and the number of masked (near-zero) cycles."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = eps_frac * float(np.median(np.abs(y))) if y.size else 0.0
    mask = np.abs(y) > eps
    n_masked = int(y.size - mask.sum())
    if not mask.any():
        return float("nan"), n_masked
    return float(100.0 * np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask]))), n_masked


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def apet_success_rates(
    y: np.ndarray, yhat: np.ndarray, thresholds: list[float], eps_frac: float = 1e-3
) -> dict[str, float]:
    """Fraction of cycles with |y - yhat| / y <= threshold."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = eps_frac * float(np.median(np.abs(y))) if y.size else 0.0
    mask = np.abs(y) > eps
    out = {}
    for th in thresholds:
        if not mask.any():
            out[f"{100 * th:g}%"] = float("nan")
        else:
            ape = np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])
            out[f"{100 * th:g}%"] = float(np.mean(ape <= th))
    return out


def peak_indices(
    y: np.ndarray,
    bench_slices: dict[str, slice],
    window: int,
    sigma: float = 3.0,
) -> np.ndarray:
    """Peak cycles: power > window mean + sigma * std, windows per benchmark."""
    peaks: list[np.ndarray] = []
    for _, sl in bench_slices.items():
        yb = y[sl]
        n = yb.size
        if n == 0:
            continue
        w = min(window, n)
        for start in range(0, n, w):
            seg = yb[start : start + w]
            if seg.size < 2:
                continue
            mu, sd = float(seg.mean()), float(seg.std())
            if sd == 0.0:
                continue
            local = np.flatnonzero(seg > mu + sigma * sd)
            peaks.append(sl.start + start + local)
    return np.concatenate(peaks) if peaks else np.zeros(0, dtype=np.int64)


def multicycle_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    bench_slices: dict[str, slice],
    windows: list[int],
    eps_frac: float = 1e-3,
) -> dict[str, dict]:
    """Average per-cycle predictions/labels over t-cycle windows, then score."""
    out: dict[str, dict] = {}
    for t in windows:
        ys, ps = [], []
        for _, sl in bench_slices.items():
            yb, pb = y[sl], yhat[sl]
            n_win = yb.size // t
            if n_win == 0:
                continue
            ys.append(yb[: n_win * t].reshape(n_win, t).mean(axis=1))
            ps.append(pb[: n_win * t].reshape(n_win, t).mean(axis=1))
        if not ys:
            out[str(t)] = {"mape": float("nan"), "r2": float("nan"), "n_windows": 0}
            continue
        yw, pw = np.concatenate(ys), np.concatenate(ps)
        m, _ = mape_percent(yw, pw, eps_frac)
        out[str(t)] = {"mape": m, "r2": r2_score(yw, pw), "n_windows": int(yw.size)}
    return out


# ---------------------------------------------------------------- reports --
def evaluation_report(
    cfg: CobitConfig,
    y: np.ndarray,
    yhat: np.ndarray,
    bench_slices: dict[str, slice],
    split_name: str,
) -> dict:
    """Normal + Peak-Power + multicycle metrics for one split."""
    eps_frac = cfg.eval.mape_eps_frac
    mape, n_masked = mape_percent(y, yhat, eps_frac)
    report = {
        "split": split_name,
        "n_cycles": int(y.size),
        "mape": mape,
        "mape_masked_cycles": n_masked,
        "r2": r2_score(y, yhat),
        "apet_success": apet_success_rates(y, yhat, cfg.eval.apet, eps_frac),
    }
    pk = peak_indices(y, bench_slices, cfg.eval.peak_window, cfg.eval.peak_sigma)
    if pk.size:
        pmape, _ = mape_percent(y[pk], yhat[pk], eps_frac)
        report["peak"] = {
            "n_peaks": int(pk.size),
            "mape": pmape,
            "r2": r2_score(y[pk], yhat[pk]),
            "apet_success": apet_success_rates(y[pk], yhat[pk], cfg.eval.apet, eps_frac),
        }
    else:
        log.warning("%s: no power peaks found (trace too short or flat)", split_name)
        report["peak"] = None
    if cfg.eval.multicycle_windows:
        report["multicycle"] = multicycle_metrics(
            y, yhat, bench_slices, cfg.eval.multicycle_windows, eps_frac
        )
    per_bench: dict[str, dict] = {}
    for bench_name, sl in bench_slices.items():
        yb, yhatb = y[sl], yhat[sl]
        if yb.size == 0:
            continue
        b_mape, b_n_masked = mape_percent(yb, yhatb, eps_frac)
        per_bench[bench_name] = {
            "n_cycles": int(yb.size),
            "mape": b_mape,
            "mape_masked_cycles": b_n_masked,
            "r2": r2_score(yb, yhatb),
            "apet_success": apet_success_rates(yb, yhatb, cfg.eval.apet, eps_frac),
        }
    report["per_benchmark"] = per_bench
    return report


# ------------------------------------------------------------------ plots --
_BLUE, _ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1-2


def _style_axis(ax) -> None:
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_trace(
    y: np.ndarray,
    yhat: np.ndarray,
    out_path: Path,
    max_cycles: int = 12000,
    title: str = "Per-cycle power: label vs prediction",
) -> None:
    """Fig.10-style overlay of ground truth and prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(y.size, max_cycles)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(np.arange(n), y[:n], color=_BLUE, linewidth=0.9, label="power label")
    ax.plot(np.arange(n), yhat[:n], color=_ORANGE, linewidth=0.9, label="COBIT prediction")
    ax.set_xlabel("cycle")
    ax.set_ylabel("power")
    ax.set_title(title, fontsize=11)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_q_sweep(records: list[dict], out_path: Path) -> None:
    """MAPE and R2 versus proxy count Q (Fig.8-style, two panels, one axis each)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted((r for r in records if r.get("test_mape") is not None), key=lambda r: r["q"])
    if not recs:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    qs = [r["q"] for r in recs]
    mapes = [r["test_mape"] for r in recs]
    r2s = [r["test_r2"] for r in recs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(qs, mapes, color=_BLUE, marker="o", markersize=5, linewidth=1.6)
    ax1.set_xlabel("number of bitwise proxies Q")
    ax1.set_ylabel("test MAPE (%)")
    ax2.plot(qs, r2s, color=_BLUE, marker="o", markersize=5, linewidth=1.6)
    ax2.set_xlabel("number of bitwise proxies Q")
    ax2.set_ylabel("test R$^2$")
    for ax in (ax1, ax2):
        _style_axis(ax)
    fig.suptitle("COBIT per-cycle accuracy vs proxy count", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
