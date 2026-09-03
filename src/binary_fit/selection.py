"""Stage 1: bit-level power-proxy selection.

Two kinds of selector live here.

**Penalized (``mcp``, ``lasso``)** -- the prior-art route. Linear regression with
the minimax concave penalty (or plain L1) fitted on the sparse binary feature
matrix; features whose coefficients survive the penalty become the power proxies.
Sweeping the penalty strength lambda (skglm's ``alpha``) produces different proxy
counts Q; a log-grid sweep plus bisection targets the configured Q list. Binary
(0/1) features are left unscaled (standardizing would destroy sparsity and the
toggle semantics); the intercept absorbs baseline power.

  Passing ``selection.tol`` is NOT optional here. skglm's AndersonCD stops on an
  ABSOLUTE criterion, so its default ``tol=1e-4`` silently exceeds the entire
  gradient scale of a watt-valued target (alpha_max = 1.8e-3 on the aq_core
  single-bit matrix) and the solver returns after ~7 iterations at every alpha
  with 85% of excluded columns violating their entry condition -- reporting
  success. That pinned Q at 17 regardless of ``target_qs``, ``max_iter``,
  ``max_bisect`` or ``grid_decades``. ``kkt_residual`` is logged per fit so the
  condition is visible rather than assumed.

**Greedy (``fsr``)** -- takes Q as a direct argument, so ``target_qs`` are hit
exactly with no alpha grid, no gamma and no tolerance. Measured to transfer
better to the held-out benchmarks than the penalized route at equal Q.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve, solve_triangular

from .config import Config
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
    col_ids: np.ndarray  # GLOBAL feature ids (materialized column positions)
    names: list[str]
    weights: np.ndarray
    # Stationarity of the fit this support came from: how many EXCLUDED columns
    # violate |grad_j| <= alpha, and the worst |grad_j|/alpha. (0, <=1.0) means a
    # genuine optimum; a large count means the solver stopped early and the
    # support is not the penalized solution. -1 for the greedy selector, which
    # solves no penalized problem.
    kkt_violations: int = -1
    kkt_max_ratio: float = float("nan")

    def to_json(self) -> dict:
        return {
            "target_q": self.target_q,
            "q": self.q,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "kkt_violations": self.kkt_violations,
            "kkt_max_ratio": self.kkt_max_ratio,
            "col_ids": self.col_ids.tolist(),
            "names": self.names,
            "weights": self.weights.tolist(),
        }


class _McpSelector:
    def __init__(self, cfg: Config):
        self.gamma = cfg.selection.gamma
        self.intercept = 0.0
        # persistent estimator: warm starts make the descending-alpha sweep
        # dramatically cheaper on large matrices
        self._est = MCPRegression(
            alpha=1.0,
            gamma=cfg.selection.gamma,
            fit_intercept=cfg.selection.fit_intercept,
            max_iter=cfg.selection.max_iter,
            # skglm's AndersonCD breaks on the ABSOLUTE `stop_crit <= tol`; leaving
            # this at its 1e-4 default silently caps Q (see the module docstring).
            tol=cfg.selection.tol,
            warm_start=True,
        )

    def fit_alpha(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        self._est.set_params(alpha=alpha)
        self._est.fit(X, y)
        self.intercept = float(np.asarray(self._est.intercept_).ravel()[0]) \
            if np.size(self._est.intercept_) else 0.0
        return np.array(self._est.coef_, dtype=float).ravel()

    def fit_cold(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        """Solve one alpha with no path history (see selection.resolve_cold)."""
        p = self._est.get_params()
        p.update(alpha=alpha, warm_start=False)
        est = MCPRegression(**p)
        est.fit(X, y)
        self.intercept = float(np.asarray(est.intercept_).ravel()[0]) \
            if np.size(est.intercept_) else 0.0
        return np.array(est.coef_, dtype=float).ravel()


class _LassoSelector:
    """sklearn fallback (documented deviation: L1 over-shrinks vs MCP).

    sklearn scales its own ``tol`` by ``||y||^2 / n`` internally, so the same
    ``selection.tol`` is a stricter criterion here than in skglm -- which is why
    this fallback was reaching far more proxies than the MCP path it substitutes
    for, an artefact of the tolerance convention rather than of the penalty.
    """

    def __init__(self, cfg: Config):
        from sklearn.linear_model import Lasso

        self._lasso = Lasso(
            alpha=1.0,
            fit_intercept=cfg.selection.fit_intercept,
            max_iter=max(cfg.selection.max_iter * 10, 1000),
            tol=cfg.selection.tol,
            warm_start=True,
        )
        self.gamma = float("nan")
        self.intercept = 0.0

    def fit_alpha(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        self._lasso.set_params(alpha=alpha)
        self._lasso.fit(X, y)
        self.intercept = float(np.asarray(self._lasso.intercept_).ravel()[0]) \
            if np.size(self._lasso.intercept_) else 0.0
        return np.array(self._lasso.coef_, dtype=float).ravel()

    def fit_cold(self, X: sparse.csc_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
        """Solve one alpha with no path history (see selection.resolve_cold)."""
        from sklearn.linear_model import Lasso

        p = self._lasso.get_params()
        p.update(alpha=alpha, warm_start=False)
        est = Lasso(**p)
        est.fit(X, y)
        self.intercept = float(np.asarray(est.intercept_).ravel()[0]) \
            if np.size(est.intercept_) else 0.0
        return np.array(est.coef_, dtype=float).ravel()


class _FsrSelector:
    """Greedy forward selection (orthogonal matching pursuit): Q is the loop bound.

    No penalty, no alpha grid, no stopping tolerance -- ``select_q`` is asked for
    q columns and returns exactly q. Each step admits the column that reduces the
    residual sum of squares most, then re-solves ALL active coefficients by least
    squares, so the residual stays orthogonal to every column already chosen.
    That orthogonalization is the point: it makes the selector skip near-clones of
    what it already holds, which is the failure mode of marginal screening (whose
    top-1000 by |corr| contains only 259 distinct signals on the aq_core
    single-bit matrix, against 1000 here).

    The intercept is carried as an always-active constant column, so the residual
    is mean-zero and ``x_j' r`` needs no centering correction; a candidate's
    usable energy is its norm after that constant is projected out. The active
    normal equations are maintained by rank-1 Cholesky updates, so a step costs
    one sparse ``X.T @ r`` (the dominant term) plus O(n*k + k^2) -- about 250 s at
    q=1000 on the 6895 x 177898 matrix.

    ``weights`` are the exact least-squares coefficients on the selected set, not
    penalized ones. ``select_proxies`` ranks proxies.csv by |weight| for every
    selector, but on a strongly collinear set that ranking carries less meaning
    than the greedy entry order, so ask for the Q you want in
    ``selection.target_qs`` rather than slicing a large set with ``--fit -q``.
    """

    def __init__(self, cfg: Config):
        self.gamma = float("nan")  # no concavity parameter
        self.intercept = 0.0

    def select_q(self, X: sparse.csc_matrix, y: np.ndarray, q: int):
        n, p = X.shape
        q = int(min(q, p, max(n - 1, 1)))
        col_sum = np.asarray(X.sum(axis=0)).ravel()
        col_sq = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        # energy left in a candidate once the constant column is projected out;
        # a column that is constant over these rows has none and can never help
        cnorm2 = np.maximum(col_sq - col_sum ** 2 / n, 0.0)
        blocked = cnorm2 <= 1e-12 * max(float(col_sq.max(initial=0.0)), 1.0)

        Z = np.empty((n, q + 1), dtype=np.float64)  # [1 | X_A], intercept first
        Z[:, 0] = 1.0
        L = np.zeros((q + 1, q + 1), dtype=np.float64)  # lower Cholesky of Z'Z
        L[0, 0] = np.sqrt(n)
        rhs = np.empty(q + 1, dtype=np.float64)
        rhs[0] = float(y.sum())
        beta = np.array([float(y.mean())])
        r = y - y.mean()
        active: list[int] = []

        while len(active) < q:
            # r is orthogonal to the constant, so x_j' r is already the centered
            # correlation and (x_j' r)^2 / ||x_j~||^2 is the exact RSS reduction
            c = np.asarray(X.T @ r).ravel()
            gain = np.where(blocked, -1.0, c * c / np.maximum(cnorm2, 1e-300))
            j = int(np.argmax(gain))
            if gain[j] <= 0.0:
                log.warning("fsr: no column left that reduces RSS - stopping at Q=%d",
                            len(active))
                break
            xj = np.asarray(X[:, j].todense(), dtype=np.float64).ravel()
            k = len(active) + 1  # columns of Z currently in use
            g = Z[:, :k].T @ xj
            u = solve_triangular(L[:k, :k], g, lower=True)
            xx = float(xj @ xj)
            d2 = xx - float(u @ u)
            if d2 <= 1e-10 * max(xx, 1.0):
                # linearly dependent on the active set (an exact duplicate of an
                # active column is the common case): it cannot reduce RSS, and
                # admitting it would break the Cholesky. Drop it for good.
                blocked[j] = True
                continue
            L[k, :k] = u
            L[k, k] = np.sqrt(d2)
            Z[:, k] = xj
            rhs[k] = float(xj @ y)
            active.append(j)
            blocked[j] = True  # never reconsider an active column
            k += 1
            beta = cho_solve((L[:k, :k], True), rhs[:k])
            r = y - Z[:, :k] @ beta

        self.intercept = float(beta[0]) if beta.size else 0.0
        return (np.asarray(active, dtype=np.int64),
                np.asarray(beta[1:], dtype=float))


def make_selector(cfg: Config):
    kind = cfg.selection.selector
    if kind == "mcp":
        if HAVE_SKGLM:
            return _McpSelector(cfg)
        log.warning("skglm not installed - falling back to Lasso proxy selection")
        return _LassoSelector(cfg)
    if kind == "lasso":
        return _LassoSelector(cfg)
    if kind == "fsr":
        return _FsrSelector(cfg)
    raise ValueError(f"unknown selector {kind!r} (expected mcp, lasso or fsr)")


def alpha_max(X: sparse.spmatrix, y: np.ndarray) -> float:
    """Smallest alpha at which no feature enters the model."""
    resid = y - y.mean()
    grad = np.abs(X.T @ resid)
    return float(np.max(grad)) / X.shape[0]


def kkt_residual(X, y: np.ndarray, coef: np.ndarray, intercept: float, alpha: float):
    """(#excluded columns violating stationarity, worst |grad_j|/alpha over them).

    A penalized fit is a solution only if every EXCLUDED column satisfies
    ``|x_j'(Xw + b - y)| / n <= alpha``. skglm instead stops on an absolute
    ``stop_crit <= tol``, so with a tol above the problem's gradient scale it
    returns a support that is nowhere near stationary and reports convergence.
    ``(0, <= 1.0)`` is a genuine optimum.
    """
    resid = X @ coef + intercept - y
    grad = np.abs(np.asarray(X.T @ resid).ravel()) / X.shape[0]
    zero = coef == 0
    if not np.any(zero) or not (alpha > 0):
        return 0, float("nan")
    gz = grad[zero]
    return int(np.count_nonzero(gz >= alpha)), float(gz.max() / alpha)


def _collapse_duplicates(X, nz: np.ndarray, weights: np.ndarray):
    """Keep one column per exact-duplicate group, the first in ``nz`` order.

    ``nz`` arrives sorted by descending |weight|, so the representative kept is
    the strongest-weighted member of its group.
    """
    from hashlib import blake2b

    Xc = X if sparse.isspmatrix_csc(X) else sparse.csc_matrix(X)
    seen, keep = set(), []
    for pos, j in enumerate(nz):
        s, e = Xc.indptr[j], Xc.indptr[j + 1]
        h = blake2b(digest_size=16)
        h.update(np.ascontiguousarray(Xc.indices[s:e], dtype=np.int32).tobytes())
        h.update(np.ascontiguousarray(Xc.data[s:e], dtype=np.float64).tobytes())
        d = h.digest()
        if d not in seen:
            seen.add(d)
            keep.append(pos)
    idx = np.asarray(keep, dtype=np.int64)
    return nz[idx], weights[idx]


def _finalize(cfg: Config, X, nz, weights, target: int, eff_target: int, alpha: float,
              gamma: float, kept_ids, feature_names_of, kkt=(-1, float("nan"))) -> ProxyResult:
    """Rank by |weight|, collapse duplicates, cut to exactly target_q."""
    scfg = cfg.selection
    nz = np.asarray(nz, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(-np.abs(weights))
    nz, weights = nz[order], weights[order]
    if scfg.dedup_proxies and nz.size:
        before = nz.size
        nz, weights = _collapse_duplicates(X, nz, weights)
        if nz.size != before:
            log.info("  target Q=%d: %d of %d selected columns were exact duplicates "
                     "of a stronger one - dropped", target, before - nz.size, before)
    if scfg.exact_q and nz.size > eff_target:
        nz, weights = nz[:eff_target], weights[:eff_target]
    gids = kept_ids[nz]
    return ProxyResult(
        target_q=target, q=int(nz.size), alpha=float(alpha), gamma=float(gamma),
        col_ids=np.asarray(gids, dtype=np.int64), names=feature_names_of(gids),
        weights=weights, kkt_violations=int(kkt[0]), kkt_max_ratio=float(kkt[1]),
    )


def _warn_duplicate_supports(results: dict[int, ProxyResult]) -> None:
    """Two targets returning the SAME support are not two deliverables."""
    seen: dict[bytes, int] = {}
    for target in sorted(results):
        key = np.ascontiguousarray(results[target].col_ids).tobytes()
        first = seen.setdefault(key, target)
        if first != target:
            log.warning(
                "target Q=%d returned the SAME %d-proxy support as target Q=%d at "
                "alpha=%.3e - the sweep saturated, so these are not distinct proxy "
                "sets. Check selection.tol (skglm's default silently caps Q) and "
                "whether the alpha grid brackets the target.",
                target, results[target].q, first, results[target].alpha,
            )


def _canon(alpha: float) -> float:
    """Canonical alpha key (6 significant digits) to avoid float near-dupes."""
    return float(f"{alpha:.6e}")


def select_proxies(
    cfg: Config,
    X: sparse.csc_matrix,
    y: np.ndarray,
    kept_ids: np.ndarray,
    feature_names_of: callable,
) -> dict[int, ProxyResult]:
    """Run the lambda sweep and return one ProxyResult per target Q."""
    sel = make_selector(cfg)
    scfg = cfg.selection
    a_max = alpha_max(X, y)
    log.info("alpha_max = %.4e", a_max)
    if not np.isfinite(a_max) or a_max <= 0:
        if not cfg.runtime.allow_tiny:
            raise RuntimeError("degenerate labels: alpha_max is not positive")
        # tiny synthetic data: labels can be constant; rank bits by toggle count
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

    if hasattr(sel, "select_q"):
        # Greedy selectors take Q directly: no alpha grid, no bisection.
        results = {}
        max_possible = min(X.shape[0], X.shape[1])
        for target in scfg.target_qs:
            eff_target = min(target, max_possible)
            nz, weights = sel.select_q(X, y, eff_target)
            results[target] = _finalize(
                cfg, X, nz, weights, target, eff_target, float("nan"),
                getattr(sel, "gamma", float("nan")), kept_ids, feature_names_of,
            )
            log.info("proxies: target Q=%d -> Q=%d (%s, exact)",
                     target, results[target].q, scfg.selector)
        _warn_duplicate_supports(results)
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
        kkt = (-1, float("nan"))
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
            if scfg.resolve_cold and hasattr(sel, "fit_cold"):
                # the sweep warm-starts every fit from the previous alpha, which is
                # what makes the search affordable -- but MCP is nonconvex, so that
                # path can end on a wildly non-stationary point. Re-solve the one
                # alpha we are about to report, from scratch.
                cold = sel.fit_cold(X, y, best)
                if np.any(cold):
                    log.info("  re-solved alpha=%.3e cold: Q %d -> %d", best,
                             int(np.count_nonzero(coef)), int(np.count_nonzero(cold)))
                    coef = cold
                    nz = np.flatnonzero(coef)
                else:
                    log.warning("  cold re-solve at alpha=%.3e selected nothing - "
                                "keeping the warm-path support", best)
            weights = coef[nz]
            if scfg.kkt_report:
                # the fit is only the penalized solution if no excluded column
                # violates |grad_j| <= alpha; skglm can return long before that.
                # Scale the alarm to the size of the violation: a handful of
                # columns a few percent over is a near-optimum, while thousands
                # of columns orders of magnitude over is a different support
                # entirely. A warning that fires on every run gets ignored, which
                # is how the tol=1e-4 default survived unnoticed in the first place.
                n_zero = int(np.count_nonzero(coef == 0))
                kkt = kkt_residual(X, y, coef, getattr(sel, "intercept", 0.0), best)
                frac = kkt[0] / max(n_zero, 1)
                if kkt[0] == 0:
                    verdict = "stationary (a genuine optimum)"
                elif frac < 1e-3 and kkt[1] < 1.5:
                    verdict = "near-stationary - fine for proxy selection"
                else:
                    verdict = ("NOT an optimum - tighten selection.tol (currently "
                               "%.1e)" % scfg.tol)
                emit = log.info if kkt[0] == 0 or verdict.startswith("near") else log.warning
                emit("  KKT at alpha=%.3e: %d of %d excluded columns (%.3f%%) violate "
                     "stationarity, max|grad|/alpha=%.4f - %s",
                     best, kkt[0], n_zero, 100.0 * frac, kkt[1], verdict)
        results[target] = _finalize(
            cfg, X, nz, weights, target, eff_target, best,
            getattr(sel, "gamma", float("nan")), kept_ids, feature_names_of, kkt,
        )
        log.info(
            "proxies: target Q=%d -> Q=%d at alpha=%.3e", target, results[target].q, best
        )
    _warn_duplicate_supports(results)
    return results
