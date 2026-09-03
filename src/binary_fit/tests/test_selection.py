import numpy as np
import pytest
from scipy import sparse

from binary_fit.config import Config
from binary_fit.selection import alpha_max, make_selector, select_proxies


def _toy_problem(seed=0, n=1500, m=200):
    rng = np.random.default_rng(seed)
    X = sparse.csc_matrix(rng.random((n, m)) < 0.05, dtype=np.float64)
    true_ids = [3, 50, 120, 170]
    w = np.zeros(m)
    w[true_ids] = [0.02, 0.015, 0.03, 0.01]
    y = 0.05 + X @ w + rng.normal(0, 0.0005, n)
    return X, y, true_ids


def _cfg(selector="lasso"):
    cfg = Config()
    cfg.selection.selector = selector
    cfg.selection.target_qs = [4, 20]
    cfg.selection.q_tol = 0.25
    cfg.selection.max_bisect = 10
    cfg.selection.grid_points = 10
    cfg.selection.grid_decades = 4.0
    return cfg


def test_lasso_selector_recovers_signal_and_targets_q():
    X, y, true_ids = _toy_problem()
    cfg = _cfg("lasso")
    kept = np.arange(X.shape[1], dtype=np.int64)
    res = select_proxies(cfg, X, y, kept, lambda g: [f"b{i}" for i in g])
    assert set(true_ids) <= set(res[4].col_ids.tolist())
    assert res[4].q == 4
    assert abs(res[20].q - 20) <= 0.25 * 20
    # proxies ranked by descending |weight| (fit's top-q slice relies on this)
    w = np.abs(res[20].weights)
    assert np.all(np.diff(w) <= 1e-12)


def test_degenerate_labels_allow_tiny_fallback():
    X = sparse.csc_matrix(np.eye(4))
    y = np.ones(4)  # constant labels -> alpha_max == 0
    cfg = _cfg("lasso")
    cfg.selection.target_qs = [2]
    cfg.runtime.allow_tiny = True
    res = select_proxies(cfg, X, y, np.arange(4), lambda g: [str(i) for i in g])
    assert res[2].q == 2
    cfg.runtime.allow_tiny = False
    with pytest.raises(RuntimeError):
        select_proxies(cfg, X, y, np.arange(4), lambda g: [str(i) for i in g])


@pytest.mark.skipif(
    not pytest.importorskip("binary_fit.selection").HAVE_SKGLM, reason="skglm not installed"
)
def test_mcp_selector_recovers_signal():
    X, y, true_ids = _toy_problem()
    cfg = _cfg("mcp")
    kept = np.arange(X.shape[1], dtype=np.int64)
    res = select_proxies(cfg, X, y, kept, lambda g: [f"b{i}" for i in g])
    assert set(true_ids) <= set(res[4].col_ids.tolist())


def test_fsr_hits_exact_q_and_skips_duplicate_columns():
    """The greedy selector takes Q directly, so target_qs are exact."""
    X, y, true_ids = _toy_problem()
    Xd = X.toarray()
    Xd[:, 7] = Xd[:, true_ids[0]]  # exact duplicate of a planted column
    X = sparse.csc_matrix(Xd)
    cfg = _cfg("fsr")
    cfg.selection.target_qs = [4, 25]
    kept = np.arange(X.shape[1], dtype=np.int64)
    res = select_proxies(cfg, X, y, kept, lambda g: [f"b{i}" for i in g])
    assert res[4].q == 4 and res[25].q == 25  # exact, not "within q_tol"
    assert set(true_ids) <= set(res[25].col_ids.tolist())
    # the residual is re-orthogonalized every step, so a duplicate of an
    # already-active column can never be admitted
    assert 7 not in res[25].col_ids.tolist()
    assert np.isnan(res[4].alpha)  # no penalty path was walked
    w = np.abs(res[25].weights)
    assert np.all(np.diff(w) <= 1e-12)  # proxies.csv contract


def test_exact_q_truncates_and_dedup_collapses_copies():
    X, y, _ = _toy_problem()
    Xd = X.toarray()
    Xd[:, 1] = Xd[:, 3]  # plant exact copies of a signal-carrying column
    Xd[:, 2] = Xd[:, 3]
    X = sparse.csc_matrix(Xd)
    cfg = _cfg("lasso")
    cfg.selection.target_qs = [6]
    kept = np.arange(X.shape[1], dtype=np.int64)
    res = select_proxies(cfg, X, y, kept, lambda g: [f"b{i}" for i in g])
    assert res[6].q <= 6  # exact_q never overshoots the target
    picked = res[6].col_ids.tolist()
    assert len(set(picked)) == len(picked)
    # at most one member of the {1, 2, 3} duplicate group survives dedup
    assert len({1, 2, 3} & set(picked)) <= 1


def test_kkt_residual_flags_a_non_stationary_support():
    """A support that omits a strongly correlated column is not an optimum."""
    from binary_fit.selection import kkt_residual

    X, y, true_ids = _toy_problem()
    coef = np.zeros(X.shape[1])
    # pretend the solver returned only one of the four planted columns
    n_viol, ratio = kkt_residual(X, y, coef, float(y.mean()), alpha=1e-9)
    assert n_viol > 0 and ratio > 1.0
    # and at an alpha above alpha_max the all-zero vector IS stationary
    n_viol, ratio = kkt_residual(X, y, coef, float(y.mean()), alpha=alpha_max(X, y) * 1.01)
    assert n_viol == 0 and ratio <= 1.0
