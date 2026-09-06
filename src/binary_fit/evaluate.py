"""Evaluation protocol: MAPE / R2, Peak-Power sets, APET success, multicycle.

Peak-Power set construction follows the paper: the continuous power trace of
each (test) benchmark is partitioned into windows; cycles whose power exceeds
window mean + 3 sigma are peaks. APET success rate = fraction of cycles whose
absolute percentage error falls below a threshold.

Multicycle prediction (per-cycle model): predictions and labels are averaged
over non-overlapping t-cycle windows. Documented deviation from the paper:
the paper re-measures window labels with its power tool; here only per-cycle
labels exist, so window labels are their means.

Units: every knob here counts ROWS of the design matrix, which is one cycle only
at ``data.window_size = 1``. With window averaging on, a row is already a
``window_size``-cycle mean, so ``eval.peak_window`` and ``eval.multicycle_windows``
are in units of those rows (t rows = t * window_size cycles).
"""

from __future__ import annotations

import numpy as np

from .config import Config
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
    cfg: Config,
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
        # n_cycles (and every other count here) counts ROWS: one row is
        # window_size cycles, recorded alongside so the numbers are unambiguous
        "window_size": int(cfg.data.window_size),
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
