"""Stage 1: bit-level power-proxy selection with LR-MCP.

Linear regression with the minimax concave penalty (paper Eq. 4) fitted on
the sparse binary toggle matrix; nets whose coefficients survive the penalty
become the power proxies. Sweeping the penalty strength lambda (skglm's
``alpha``) produces different proxy counts Q; a log-grid sweep plus bisection
targets the configured Q list.

Feature scaling policy: binary (bit_expand=True) features are 0/1 and left
unscaled (standardizing would destroy sparsity and the 0/1 toggle
semantics); multi-bit (bit_expand=False) features are raw toggle bitmask
integers whose magnitudes span ~90 orders of magnitude, so they are
column-wise max-abs scaled to [-1, 1] inside ``select_proxies``. The
intercept absorbs baseline power.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import sparse

from .config import CobitConfig
from .utils import log

try:  # skglm is the faithful LR-MCP implementation used by the paper
    from skglm import MCPRegression

    HAVE_SKGLM = True
except ImportError:  # pragma: no cover - exercised only without skglm
    HAVE_SKGLM = False


@dataclasses.dataclass
class ProxyResult:
    target_q: int
    q: int
    alpha: float
    gamma: float
    col_ids: np.ndarray  # GLOBAL feature ids (registry space)
    names: list[str]
    weights: np.ndarray

    def to_json(self) -> dict:
        return {
            "target_q": self.target_q,
            "q": self.q,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "col_ids": self.col_ids.tolist(),
            "names": self.names,
            "weights": self.weights.tolist(),
        }


class _McpSelector:
    def __init__(self, cfg: CobitConfig):
        self.gamma = cfg.selection.gamma
        # persistent estimator: warm starts make the descending-alpha sweep
        # dramatically cheaper on large matrices
        self._est = MCPRegression(
            alpha=1.0,
            gamma=cfg.selection.gamma,
            fit_intercept=cfg.selection.fit_intercept,
            max_iter=cfg.selection.max_iter,
            warm_start=True,
        )

    def fit_alpha(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        self._est.set_params(alpha=alpha)
        self._est.fit(X, y)
        # copy: estimators may reuse/mutate coef_ buffers on later fits
        return np.array(self._est.coef_, dtype=float).ravel()


class _LassoSelector:
    """sklearn fallback (documented deviation: L1 over-shrinks vs MCP)."""

    def __init__(self, cfg: CobitConfig):
        from sklearn.linear_model import Lasso

        self._lasso = Lasso(
            alpha=1.0,
            fit_intercept=cfg.selection.fit_intercept,
            max_iter=max(cfg.selection.max_iter * 10, 1000),
            warm_start=True,
        )
        self.gamma = float("nan")

    def fit_alpha(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        self._lasso.set_params(alpha=alpha)
        self._lasso.fit(X, y)
        # copy: warm_start reuses the coef_ buffer across fits
        return np.array(self._lasso.coef_, dtype=float).ravel()


def make_selector(cfg: CobitConfig):
    kind = cfg.selection.selector
    if kind == "mcp":
        if HAVE_SKGLM:
            return _McpSelector(cfg)
        log.warning("skglm not installed - falling back to Lasso proxy selection")
        return _LassoSelector(cfg)
    if kind == "lasso":
        return _LassoSelector(cfg)
    raise ValueError(f"unknown selector {kind!r}")


def alpha_max(X: sparse.spmatrix, y: np.ndarray) -> float:
    """Smallest alpha at which no feature enters the model."""
    resid = y - y.mean()
    grad = np.abs(X.T @ resid)
    return float(np.max(grad)) / X.shape[0]


def _maxabs_scale_columns(X: sparse.csc_matrix) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Column-wise max-abs scaling to [-1, 1]; sparsity is preserved.

    Multi-bit (``bit_expand=False``) features are raw toggle bitmask integers
    whose magnitudes span ~90 orders of magnitude (a 311-bit net's mask reaches
    ~1e+93). Without scaling, ``alpha_max`` is ~1e+84, the log alpha grid
    cannot reach the solvable range, skglm's Anderson acceleration hits
    divide-by-zero, and each fit takes tens of minutes. Scaling each column by
    its max |entry| brings every feature into [-1, 1], drops ``alpha_max`` to
    ~O(10-100), and leaves the sparsity pattern untouched (column scaling only
    rescales nonzero entries). Binary (``bit_expand=True``) features are 0/1
    and are NOT routed through here.
    """
    indptr = X.indptr
    abs_data = np.abs(X.data)
    col_max = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if e > s:
            col_max[j] = abs_data[s:e].max()
    inv = np.divide(1.0, col_max, out=np.ones_like(col_max), where=col_max > 0.0)
    Xs = X.multiply(sparse.csc_matrix(inv.reshape(1, -1))).tocsc()
    return Xs, col_max


def _canon(alpha: float) -> float:
    """Canonical alpha key (6 significant digits) to avoid float near-dupes."""
    return float(f"{alpha:.6e}")


def select_proxies(
    cfg: CobitConfig,
    X: sparse.csc_matrix,
    y: np.ndarray,
    kept_ids: np.ndarray,
    feature_names_of: callable,
) -> dict[int, ProxyResult]:
    """Run the lambda sweep and return one ProxyResult per target Q."""
    sel = make_selector(cfg)
    scfg = cfg.selection
    if not cfg.data.bit_expand:
        X, col_max = _maxabs_scale_columns(X)
        log.info(
            "multi-bit: max-abs column-scaled (col_max span %.3e .. %.3e)",
            float(col_max.min()) if col_max.size else 0.0,
            float(col_max.max()) if col_max.size else 0.0,
        )
    a_max = alpha_max(X, y)
    log.info("alpha_max = %.4e", a_max)
    if not np.isfinite(a_max) or a_max <= 0:
        if not cfg.runtime.allow_tiny:
            raise RuntimeError("degenerate labels: alpha_max is not positive")
        # 2-row smoke data: labels can be constant; rank bits by toggle count
        log.warning(
            "degenerate labels (alpha_max=%.3g) - allow_tiny fallback: ranking "
            "bits by toggle count instead of LR-MCP", a_max,
        )
        colsum = np.asarray(X.sum(axis=0)).ravel()
        order = np.argsort(-colsum)
        results = {}
        for target in scfg.target_qs:
            take = min(target, int(np.count_nonzero(colsum)))
            nz = order[:take]
            gids = kept_ids[nz]
            results[target] = ProxyResult(
                target_q=target, q=int(take), alpha=float("nan"),
                gamma=getattr(sel, "gamma", float("nan")),
                col_ids=np.asarray(gids, dtype=np.int64),
                names=feature_names_of(gids),
                weights=colsum[nz].astype(float),
            )
        return results

    cache: dict[float, np.ndarray] = {}  # canonical alpha -> coef

    def fit(alpha: float) -> float:
        key = _canon(alpha)
        if key not in cache:
            coef = sel.fit_alpha(X, y, key)
            cache[key] = coef
            log.info("  alpha=%.3e -> Q=%d", key, int(np.sum(coef != 0)))
        return key

    def q_at(key: float) -> int:
        return int(np.sum(cache[key] != 0))

    # coarse descending log grid, stopped once Q overshoots every target -
    # the small-alpha fits are by far the most expensive ones
    q_stop = max(scfg.target_qs) * (1.0 + scfg.q_tol)
    for a in a_max * np.logspace(0, -scfg.grid_decades, scfg.grid_points):
        key = fit(float(a))
        if q_at(key) > q_stop:
            break

    results: dict[int, ProxyResult] = {}
    max_possible = min(X.shape[0], X.shape[1])
    for target in scfg.target_qs:
        eff_target = min(target, max_possible)
        for _ in range(scfg.max_bisect):
            best = min(cache, key=lambda a: (abs(q_at(a) - eff_target), -a))
            if abs(q_at(best) - eff_target) <= scfg.q_tol * eff_target:
                break
            # bracket the target: Q shrinks as alpha grows
            left = [a for a in cache if q_at(a) > eff_target]  # too many proxies
            right = [a for a in cache if q_at(a) < eff_target]  # too few
            if not left:
                # even the smallest alpha selects too few: extend the grid down
                prev_q = q_at(best)
                fit(min(cache) / 10.0)
                if q_at(min(cache)) <= prev_q:
                    break  # saturation: further alpha reduction doesn't increase Q
                continue
            a_left, a_right = max(left), min(right) if right else max(cache)
            if a_left >= a_right:
                break  # non-monotone wrinkle; keep the closest evaluated point
            mid = _canon(float(np.sqrt(a_left * a_right)))
            if mid in cache:
                break  # bracket exhausted at float resolution
            fit(mid)

        best = min(cache, key=lambda a: (abs(q_at(a) - eff_target), -a))
        coef = cache[best]
        nz = np.flatnonzero(coef)
        if nz.size == 0:
            # degenerate data: fall back to the strongest-correlation bits
            log.warning(
                "selector found 0 proxies for target Q=%d - falling back to "
                "top-|X'y| ranking", target,
            )
            score = np.asarray(np.abs(X.T @ (y - y.mean()))).ravel()
            nz = np.argsort(score)[::-1][:eff_target]
            weights = score[nz]
        else:
            weights = coef[nz]
        order = np.argsort(-np.abs(np.asarray(weights)))
        nz = np.asarray(nz)[order]
        weights = np.asarray(weights)[order]
        gids = kept_ids[nz]
        results[target] = ProxyResult(
            target_q=target,
            q=int(nz.size),
            alpha=float(best),
            gamma=getattr(sel, "gamma", float("nan")),
            col_ids=np.asarray(gids, dtype=np.int64),
            names=feature_names_of(gids),
            weights=np.asarray(weights, dtype=float),
        )
        log.info(
            "proxies: target Q=%d -> Q=%d at alpha=%.3e", target, results[target].q, best
        )
    return results
