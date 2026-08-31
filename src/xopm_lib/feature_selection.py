"""X-OPM Feature Selection (spec ``doc/spec/x-opm-trainning-procedure.md`` section
"Feature Selection").

Two reducers, applied per module on the *training* split of ``dataset_processed/``
(never the test benchmarks, to avoid leakage):

1. **Correlation-based redundancy elimination.** Pairwise Pearson correlations across
   all features; for every pair with ``|r| > thresh`` (default 0.9) drop the member
   with the *weaker* absolute correlation to the target, keeping the more informative
   one. Zero-variance columns (which cannot correlate) are dropped first.
2. **Secondary reducer (only if many features remain).** "If features are still a lot,
   we use MCP." Implemented dependency-free:
     - ``mcp`` (default): Minimax Concave Penalty sparse regression by coordinate
       descent with non-negativity (consistent with the positive-coefficient model),
       walking a lambda path until the active set reaches the target size.
     - ``gain``: a quick monotone XGBoost fit; keep the top-K features by total gain.
   Only triggered when the surviving feature count exceeds ``--max-features``.

Output per module (consumed by ``model_regression.py``):
  ``dataset_processed/feature_selection/<module>.json``  -- kept columns + params
  ``dataset_processed/feature_selection/<module>_dropped.csv`` -- every drop + reason

CLI (interpreter ~/anaconda3/bin/python):
  python src/xopm_lib/feature_selection.py --module cp0
  python src/xopm_lib/feature_selection.py --all --thresh 0.9 --max-features 1000
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("xopm.feature_selection")

MODULES = ("cp0", "idu", "ifu", "iu", "lsu", "rtu", "vidu", "vpu")
CATEGORIES = ("control", "data", "config")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PROCESSED = os.path.join(_REPO_ROOT, "dataset_processed")


# ===========================================================================
# Load the pooled training feature matrix + target for one module
# ===========================================================================

def list_cases(split_dir: str) -> list[str]:
    """Case names in a ``dataset_processed/<split>/<module>`` dir (from targets)."""
    tgt = sorted(glob.glob(os.path.join(split_dir, "*_target.pkl.zst")))
    return [os.path.basename(p)[: -len("_target.pkl.zst")] for p in tgt]


def load_case(split_dir: str, case: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one case: concat control+data+config feature frames + the target.

    A category file is absent when the module has no feature of that category
    (``data_preprocess`` skips empty categories); such categories are skipped.
    Columns keep the concat order control -> data -> config.
    """
    frames = []
    for cat in CATEGORIES:
        path = os.path.join(split_dir, f"{case}_{cat}.pkl.zst")
        if os.path.exists(path):
            frames.append(pd.read_pickle(path))
    if not frames:
        raise RuntimeError(f"{split_dir}/{case}: no feature files found")
    X = pd.concat(frames, axis=1)
    y = pd.read_pickle(os.path.join(split_dir, f"{case}_target.pkl.zst"))
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if not X.index.equals(y.index):
        raise RuntimeError(f"{split_dir}/{case}: feature/target index mismatch")
    return X, y


def load_module_train(module: str, dataset_dir: str = DATASET_PROCESSED
                      ) -> tuple[pd.DataFrame, pd.Series]:
    """Pool every training case (real benchmarks + ``random`` where present)."""
    split_dir = os.path.join(dataset_dir, "trainset", module)
    cases = list_cases(split_dir)
    if not cases:
        raise RuntimeError(f"no training cases in {split_dir}")
    xs, ys, ref_cols = [], [], None
    for case in cases:
        X, y = load_case(split_dir, case)
        if ref_cols is None:
            ref_cols = list(X.columns)
        elif list(X.columns) != ref_cols:
            # canonical columns should match across cases; reindex defensively.
            X = X.reindex(columns=ref_cols)
        xs.append(X)
        ys.append(y)
    X = pd.concat(xs, axis=0, ignore_index=True)
    y = pd.concat(ys, axis=0, ignore_index=True)
    log.info("%s: pooled train %d rows x %d features over %d cases",
             module, X.shape[0], X.shape[1], len(cases))
    return X, y


# ===========================================================================
# Correlation utilities (float32, row-subsampled for tractability)
# ===========================================================================

def _standardize(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return column-standardized A (float32) and a boolean mask of live columns.

    Constant columns (std==0) are left as zeros and flagged dead so they read as
    uncorrelated rather than producing NaNs.
    """
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    live = std > 0
    Z = np.zeros_like(A, dtype=np.float32)
    if live.any():
        Z[:, live] = ((A[:, live] - mean[live]) / std[live]).astype(np.float32)
    return Z, live


def target_abs_corr(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson r| of each standardized feature column with the target."""
    yz = (y - y.mean())
    ystd = yz.std()
    if ystd == 0:
        return np.zeros(Z.shape[1])
    yz = (yz / ystd).astype(np.float32)
    n = Z.shape[0]
    return np.abs((Z.T @ yz) / n).astype(np.float64)


def feature_abs_corr(Z: np.ndarray) -> np.ndarray:
    """Absolute feature-feature Pearson correlation matrix (F x F, float32)."""
    n = Z.shape[0]
    C = (Z.T @ Z) / n
    np.clip(C, -1.0, 1.0, out=C)
    return np.abs(C)


def correlation_prune(X: pd.DataFrame, y: pd.Series, thresh: float = 0.9,
                      max_rows: int = 50000, seed: int = 0
                      ) -> tuple[list[str], list[dict], np.ndarray]:
    """Greedy redundancy elimination on ``|r| > thresh`` pairs.

    Returns ``(kept_columns, dropped_log, r_target)`` where ``r_target`` is the
    per-column absolute target correlation (aligned to ``X.columns``). For each
    highly-correlated surviving pair the member with the weaker target correlation
    is dropped (ties broken by dropping the higher column index).
    """
    cols = list(X.columns)
    A = X.to_numpy(dtype=np.float32, copy=False)
    if A.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(A.shape[0], max_rows, replace=False))
        A = A[idx]
        yv = y.to_numpy()[idx]
    else:
        yv = y.to_numpy()

    Z, live = _standardize(A)
    r_target = target_abs_corr(Z, yv)

    dropped: list[dict] = []
    dead: set[int] = set()

    # Drop zero-variance columns outright (cannot carry information / correlate).
    for j in np.where(~live)[0]:
        dead.add(int(j))
        dropped.append({"feature": cols[j], "partner": None,
                        "r_target": 0.0, "r_partner_target": None,
                        "abs_corr": None, "reason": "zero-variance"})

    C = feature_abs_corr(Z)
    F = len(cols)
    # Iterate upper triangle; skip already-dead columns. O(F^2) (F<=~1867).
    iu, ju = np.triu_indices(F, k=1)
    high = C[iu, ju] > thresh
    for i, j in zip(iu[high], ju[high]):
        i, j = int(i), int(j)
        if i in dead or j in dead:
            continue
        # keep the one more correlated with the target
        if r_target[i] >= r_target[j]:
            keep, drop = i, j
        else:
            keep, drop = j, i
        dead.add(drop)
        dropped.append({"feature": cols[drop], "partner": cols[keep],
                        "r_target": float(r_target[drop]),
                        "r_partner_target": float(r_target[keep]),
                        "abs_corr": float(C[i, j]),
                        "reason": f"|corr|>{thresh} with {cols[keep]}"})

    kept = [c for k, c in enumerate(cols) if k not in dead]
    log.info("correlation prune: kept %d / %d (dropped %d)",
             len(kept), F, len(dropped))
    return kept, dropped, r_target


# ===========================================================================
# Secondary reducer -- MCP (Minimax Concave Penalty) or XGBoost gain top-K
# ===========================================================================

def _mcp_threshold(z: float, lam: float, gamma: float, nonneg: bool) -> float:
    """Firm-threshold operator for MCP with a standardized column (x^T x / n = 1)."""
    if nonneg and z <= 0:
        return 0.0
    az = abs(z)
    if az <= lam:
        return 0.0
    if az <= gamma * lam:
        val = np.sign(z) * (az - lam) / (1.0 - 1.0 / gamma)
    else:
        val = z
    return max(val, 0.0) if nonneg else val


def mcp_select(X: pd.DataFrame, y: pd.Series, max_features: int,
               gamma: float = 3.0, n_lambda: int = 30, max_iter: int = 100,
               tol: float = 1e-4, max_rows: int = 10000, nonneg: bool = True,
               seed: int = 0) -> list[str]:
    """Select <= ``max_features`` columns via non-negative MCP coordinate descent.

    Walks a decreasing lambda path (warm-started) until the active set first
    reaches ``max_features``; if it overshoots, keeps the top ones by |coefficient|.
    Rows are subsampled to ``max_rows`` to bound runtime. Returns kept column names.
    """
    cols = list(X.columns)
    A = X.to_numpy(dtype=np.float64, copy=False)
    yv = y.to_numpy(dtype=np.float64)
    if A.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(A.shape[0], max_rows, replace=False))
        A, yv = A[idx], yv[idx]

    n, p = A.shape
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std[std == 0] = 1.0
    Xs = (A - mean) / std
    yc = yv - yv.mean()

    lam_max = float(np.abs(Xs.T @ yc / n).max())
    if lam_max <= 0:
        return cols[:max_features]
    lambdas = lam_max * np.logspace(0, -3, n_lambda)

    beta = np.zeros(p)
    r = yc.copy()  # residual = yc - Xs @ beta
    chosen: np.ndarray | None = None
    for lam in lambdas:
        for _ in range(max_iter):
            max_delta = 0.0
            for j in range(p):
                bj = beta[j]
                zj = float(Xs[:, j] @ r) / n + bj
                bn = _mcp_threshold(zj, lam, gamma, nonneg)
                if bn != bj:
                    r += Xs[:, j] * (bj - bn)
                    beta[j] = bn
                    max_delta = max(max_delta, abs(bn - bj))
            if max_delta < tol:
                break
        nz = np.where(beta != 0)[0]
        if len(nz) >= max_features:
            chosen = nz
            break
    if chosen is None:
        chosen = np.where(beta != 0)[0]
    if len(chosen) > max_features:
        order = np.argsort(-np.abs(beta[chosen]))
        chosen = chosen[order[:max_features]]
    log.info("MCP select: %d features (target %d, lambda-path)", len(chosen), max_features)
    return [cols[int(j)] for j in sorted(chosen)]


def gain_select(X: pd.DataFrame, y: pd.Series, max_features: int,
                seed: int = 0) -> list[str]:
    """Keep the top-``max_features`` columns by monotone-XGBoost total gain.

    XGBoost's DMatrix rejects feature names containing ``[ ] <``, which our net
    paths carry, so the matrix uses positional ``f{idx}`` names and gain scores are
    mapped back to real column names by index.
    """
    import xgboost as xgb
    cols = list(X.columns)
    dtrain = xgb.DMatrix(X.to_numpy(np.float32), label=y.to_numpy(np.float64))
    params = dict(objective="reg:squarederror", tree_method="hist", eta=0.1,
                  max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  monotone_constraints="(" + ",".join(["1"] * len(cols)) + ")")
    bst = xgb.train(params, dtrain, num_boost_round=100)
    score = bst.get_score(importance_type="total_gain")  # {'f{idx}': gain}
    gain = np.array([score.get(f"f{i}", 0.0) for i in range(len(cols))])
    order = np.argsort(-gain)[:max_features]
    kept = [cols[int(i)] for i in sorted(order)]
    log.info("gain select: kept top %d / %d by total_gain", len(kept), len(cols))
    return kept


# ===========================================================================
# Orchestration
# ===========================================================================

def select_module(module: str, thresh: float = 0.9, max_features: int = 1000,
                  secondary: str = "mcp", dataset_dir: str = DATASET_PROCESSED,
                  out_dir: str | None = None, seed: int = 0) -> dict:
    X, y = load_module_train(module, dataset_dir)
    n_all = X.shape[1]

    kept, dropped, r_target = correlation_prune(X, y, thresh=thresh, seed=seed)
    n_after_corr = len(kept)

    secondary_used = None
    if len(kept) > max_features:
        Xk = X[kept]
        if secondary == "gain":
            kept2 = gain_select(Xk, y, max_features, seed=seed)
        else:
            kept2 = mcp_select(Xk, y, max_features, seed=seed)
        secondary_used = secondary
        removed = set(kept) - set(kept2)
        rt = {c: float(r_target[i]) for i, c in enumerate(X.columns)}
        for c in sorted(removed):
            dropped.append({"feature": c, "partner": None,
                            "r_target": rt.get(c), "r_partner_target": None,
                            "abs_corr": None, "reason": f"{secondary}-secondary-reduce"})
        kept = kept2

    out_dir = out_dir or os.path.join(dataset_dir, "feature_selection")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "module": module,
        "params": {"thresh": thresh, "max_features": max_features,
                   "secondary": secondary_used, "seed": seed},
        "n_features_input": int(n_all),
        "n_after_correlation": int(n_after_corr),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "kept_columns": kept,
    }
    with open(os.path.join(out_dir, f"{module}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    pd.DataFrame(dropped).to_csv(
        os.path.join(out_dir, f"{module}_dropped.csv"), index=False)
    log.info("%s: %d -> %d features (corr %d, secondary %s); wrote %s.json",
             module, n_all, len(kept), n_after_corr, secondary_used, module)
    return result


def load_selected_columns(module: str, dataset_dir: str = DATASET_PROCESSED
                          ) -> list[str] | None:
    """Return kept columns for a module if feature selection has been run, else None."""
    path = os.path.join(dataset_dir, "feature_selection", f"{module}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)["kept_columns"]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(prog="xopm-feature-selection")
    ap.add_argument("--module", choices=MODULES)
    ap.add_argument("--all", action="store_true", help="run every module")
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--max-features", type=int, default=1000)
    ap.add_argument("--secondary", choices=["mcp", "gain"], default="mcp")
    ap.add_argument("--dataset-dir", default=DATASET_PROCESSED)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    targets = MODULES if args.all else ([args.module] if args.module else None)
    if not targets:
        ap.error("pass --module <name> or --all")
    for m in targets:
        select_module(m, thresh=args.thresh, max_features=args.max_features,
                      secondary=args.secondary, dataset_dir=args.dataset_dir,
                      seed=args.seed)


if __name__ == "__main__":
    main()
