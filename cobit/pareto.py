"""Pareto-front utilities for the 2-objective HPO (MAPE %, total leaves).

Pure numpy implementations of the metrics used in the paper's Algorithm 2:
non-dominated filtering, 2-D hypervolume, Coverage / NetCoverage, and the
Schott Spacing indicator. Both objectives are minimized.

Note: the paper does not state its hypervolume reference point, so absolute
HV values are not reproducible - only relative comparisons between
sampler-pruner pairs are meaningful. We normalize objectives by a
configurable reference point before computing HV.
"""

from __future__ import annotations

import numpy as np


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows of an (n, 2) minimization array.

    Weak dominance: a dominates b iff a <= b componentwise and a < b in at
    least one objective. Duplicate points are all kept.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    n = pts.shape[0]
    order = np.lexsort((pts[:, 1], pts[:, 0]))  # by obj0 asc, then obj1 asc
    mask = np.zeros(n, dtype=bool)
    best1 = np.inf
    prev = None  # (obj0, obj1) of the last kept point
    for i in order:
        x0, x1 = pts[i]
        if x1 < best1 or (prev is not None and x0 == prev[0] and x1 == prev[1]):
            # strictly better in obj1 than everything with smaller-or-equal
            # obj0, or an exact duplicate of a kept point
            mask[i] = True
            best1 = min(best1, x1)
            prev = (x0, x1)
    return mask


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """Exact hypervolume of the region dominated by ``points`` up to ``ref``.

    Objectives are normalized by the reference point, so the ideal HV is 1.0
    (a single point at the origin) and points outside the box contribute 0.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    ref_arr = np.asarray(ref, dtype=float)
    norm = pts / ref_arr
    norm = norm[np.all(norm < 1.0, axis=1)]
    if norm.size == 0:
        return 0.0
    front = norm[non_dominated(norm)]
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_x1 = 1.0
    for x0, x1 in front:
        if x1 < prev_x1:
            hv += (1.0 - x0) * (prev_x1 - x1)
            prev_x1 = x1
    return float(hv)


def coverage(front_i: np.ndarray, front_j: np.ndarray) -> float:
    """Cov(i, j): fraction of points in front j dominated by some point of i."""
    fi = np.asarray(front_i, dtype=float)
    fj = np.asarray(front_j, dtype=float)
    if fj.size == 0:
        return 0.0
    if fi.size == 0:
        return 0.0
    covered = 0
    for b in fj:
        if np.any(np.all(fi <= b, axis=1) & np.any(fi < b, axis=1)):
            covered += 1
    return covered / fj.shape[0]


def coverage_matrix(fronts: list[np.ndarray]) -> np.ndarray:
    n = len(fronts)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cov[i, j] = coverage(fronts[i], fronts[j])
    return cov


def net_coverage(fronts: list[np.ndarray]) -> np.ndarray:
    """NetCov[i] = sum_j Cov(i, j) - sum_j Cov(j, i)."""
    cov = coverage_matrix(fronts)
    return cov.sum(axis=1) - cov.sum(axis=0)


def spacing(points: np.ndarray) -> float:
    """Schott spacing: std of nearest-neighbor Manhattan distances."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    d = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def roi_filter(points: np.ndarray, mape_max: float, leaves_max: float) -> np.ndarray:
    """Mask of points inside the region of interest [mape_max, leaves_max]."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    return (pts[:, 0] <= mape_max) & (pts[:, 1] <= leaves_max)
