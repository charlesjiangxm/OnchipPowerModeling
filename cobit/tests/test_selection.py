import numpy as np
import pytest
from scipy import sparse

from cobit.config import CobitConfig
from cobit.selection import alpha_max, make_selector, select_proxies


def _toy_problem(seed=0, n=1500, m=200):
    rng = np.random.default_rng(seed)
    X = sparse.csc_matrix(rng.random((n, m)) < 0.05, dtype=np.float64)
    true_ids = [3, 50, 120, 170]
    w = np.zeros(m)
    w[true_ids] = [0.02, 0.015, 0.03, 0.01]
    y = 0.05 + X @ w + rng.normal(0, 0.0005, n)
    return X, y, true_ids


def _cfg(selector="lasso"):
    cfg = CobitConfig()
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
    assert abs(res[20].q - 20) <= 0.25 * 20  # within tolerance


def test_coef_arrays_are_not_aliased():
    """Regression: warm-start estimators reuse coef_ buffers."""
    X, y, _ = _toy_problem()
    cfg = _cfg("lasso")
    sel = make_selector(cfg)
    a = alpha_max(X, y)
    c1 = sel.fit_alpha(X, y, a * 0.5)
    q1 = int(np.sum(c1 != 0))
    sel.fit_alpha(X, y, a * 1e-4)  # much denser fit afterwards
    assert int(np.sum(c1 != 0)) == q1  # first result must be unchanged


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
    not pytest.importorskip("cobit.selection").HAVE_SKGLM, reason="skglm not installed"
)
def test_mcp_selector_recovers_signal():
    X, y, true_ids = _toy_problem()
    cfg = _cfg("mcp")
    kept = np.arange(X.shape[1], dtype=np.int64)
    res = select_proxies(cfg, X, y, kept, lambda g: [f"b{i}" for i in g])
    assert set(true_ids) <= set(res[4].col_ids.tolist())
