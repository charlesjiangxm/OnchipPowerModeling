"""Unit tests for the ridge backend's estimator and its un-scaling algebra.

``fit_ridge_scaled`` solves in standardized space, so the coefficients it reports
in watts are computed, not read off the estimator. That arithmetic
(:func:`models.ridge_coefficients`) is the only genuinely new numerics in the
backend and is what ``ridge_coefficients.csv`` publishes, so it is pinned here
against the model's own predictions rather than trusted.
"""

import numpy as np
import pytest

from binary_fit import models


def _linear_case(n=400, p=6, seed=0, dtype=np.float64):
    """(X, y, w_true, intercept_true) for an exactly linear law plus tiny noise."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, p)).astype(dtype)
    w = rng.normal(size=p)
    y = np.asarray(X, dtype=np.float64) @ w + 0.42 + rng.normal(0, 1e-4, n)
    return X, y, w, 0.42


def _grid(n_rows=400):
    """The default row-relative grid, resolved for ``n_rows`` fitting rows."""
    return models.ridge_alphas(1e2, 8.0, 25, n_rows)


def test_ridge_alphas_is_descending_and_strictly_positive():
    """The floor must stay off zero: a duplicate-column Gram is singular."""
    g = _grid(n_rows=1000)
    assert g.size == 25
    assert np.all(np.diff(g) < 0)  # descending, like the Stage-1 alpha sweep
    assert g.min() > 0
    # bounds are per-row, so the absolute grid is alpha_rel * n
    assert g[0] == pytest.approx(1e2 * 1000) and g[-1] == pytest.approx(1e-6 * 1000)


def test_ridge_alphas_scales_with_the_row_count():
    """sklearn minimizes a SUM, so the same shrinkage needs alpha proportional to n.

    Pinned because a fixed ABSOLUTE grid silently changes meaning with n: its
    heaviest point shrinks a unit-variance direction to 0.013 at n=135 but only to
    0.95 at n=200_000, i.e. nearly unregularized exactly where the design is
    largest and most rank-deficient.
    """
    for n in (135, 8616, 200_000):
        g = models.ridge_alphas(1e2, 8.0, 25, n)
        assert g / n == pytest.approx(models.ridge_alphas(1e2, 8.0, 25, 1))
        # shrinkage of a unit-variance direction is n/(n+alpha) = 1/(1+alpha_rel)
        assert n / (n + g[0]) == pytest.approx(1 / (1 + 1e2))
        assert n / (n + g[-1]) == pytest.approx(1 / (1 + 1e-6))


def test_row_relative_alpha_is_invariant_to_replicating_the_rows():
    """The property an absolute alpha lacks: tiling the rows must not move the fit."""
    X, y, _, _ = _linear_case(n=300, p=5)
    base = models.fit_ridge_scaled(X, y, alpha=1e-2 * 300)[0].coef_
    Xk, yk = np.tile(X, (4, 1)), np.tile(y, 4)
    rel = models.fit_ridge_scaled(Xk, yk, alpha=1e-2 * 1200)[0].coef_
    abs_ = models.fit_ridge_scaled(Xk, yk, alpha=1e-2 * 300)[0].coef_
    assert np.abs(rel - base).max() < 1e-10
    assert np.abs(abs_ - base).max() > 1e-10  # the absolute one does move


def test_fit_ridge_scaled_recovers_a_linear_law():
    X, y, w, b = _linear_case()
    model, xs, ys = models.fit_ridge_scaled(X, y, alphas=_grid())
    _, coef_watts, intercept_watts = models.ridge_coefficients(model, xs, ys)
    assert coef_watts == pytest.approx(w, abs=5e-3)
    assert intercept_watts == pytest.approx(b, abs=5e-3)


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-6)])
def test_ridge_coefficients_reproduce_the_prediction(dtype, tol):
    """``coef_watts . x + intercept_watts`` IS the model's prediction in watts.

    The identity that makes ridge_coefficients.csv quotable. The float32 row is
    the shape ``Union.slice`` actually hands over, and its looser tolerance is
    ``StandardScaler.transform``'s own rounding, not the algebra.

    Scored as an absolute error against the *signal* scale rather than
    element-wise relative: a prediction that happens to land near zero fails any
    relative tolerance while being off by one float32 ulp of the range.
    """
    X, y, _, _ = _linear_case(dtype=dtype)
    model, xs, ys = models.fit_ridge_scaled(X, y, alphas=_grid())
    _, coef_watts, intercept_watts = models.ridge_coefficients(model, xs, ys)
    direct = np.asarray(X, dtype=np.float64) @ coef_watts + intercept_watts
    yhat = models.predict(model, xs, ys, X)
    assert np.abs(direct - yhat).max() <= tol * np.abs(yhat).max()


def test_ridge_importance_is_non_negative_and_sized():
    """utils.save_coefficients_csv normalizes and ranks by this: no negatives."""
    X, y, _, _ = _linear_case()
    model, _, _ = models.fit_ridge_scaled(X, y, alphas=_grid())
    imp = models.ridge_importance(model)
    assert imp.shape == (X.shape[1],)
    assert np.all(imp >= 0) and imp.sum() > 0


def test_fit_ridge_scaled_survives_duplicate_columns():
    """80.3% of the aq_core kept bits are exact copies of another bit.

    That makes X'X singular, so this is the case the penalty exists for: the fit
    must stay finite and bounded, and the identity above must still hold.
    """
    X, y, _, _ = _linear_case(p=5)
    Xd = np.hstack([X, X[:, :3]])  # three exact clones
    model, xs, ys = models.fit_ridge_scaled(Xd, y, alphas=_grid())
    coef_std, coef_watts, intercept_watts = models.ridge_coefficients(model, xs, ys)
    assert np.isfinite(coef_std).all() and np.abs(coef_std).max() < 1e3
    direct = Xd @ coef_watts + intercept_watts
    yhat = models.predict(model, xs, ys, Xd)
    assert np.abs(direct - yhat).max() <= 1e-12 * np.abs(yhat).max()


def test_fit_ridge_scaled_survives_a_constant_column():
    """StandardScaler sets scale_ = 1.0 for zero variance, so nothing divides by 0."""
    X, y, _, _ = _linear_case()
    X = X.copy()
    X[:, 0] = 0.5
    model, xs, ys = models.fit_ridge_scaled(X, y, alphas=_grid())
    _, coef_watts, intercept_watts = models.ridge_coefficients(model, xs, ys)
    assert xs.scale_[0] == 1.0
    assert np.isfinite(coef_watts).all() and np.isfinite(intercept_watts)


def test_fit_ridge_scaled_requires_exactly_one_of_alpha_alphas():
    X, y, _, _ = _linear_case(n=50, p=3)
    for kwargs in ({}, {"alpha": 1.0, "alphas": _grid()}):
        with pytest.raises(ValueError, match="exactly one"):
            models.fit_ridge_scaled(X, y, **kwargs)


def test_no_hpo_alpha_is_used_verbatim():
    """--no-hpo must not silently search; the fixed alpha reaches the estimator."""
    X, y, _, _ = _linear_case()
    alpha = models.NOHPO_RIDGE_ALPHA_REL * X.shape[0]
    model, _, _ = models.fit_ridge_scaled(X, y, alpha=alpha)
    assert not hasattr(model, "alpha_")  # a Ridge, not a RidgeCV
    assert model.alpha == pytest.approx(alpha)
    # and it must actually regularize: 1/(1+alpha_rel) at the package default
    assert X.shape[0] / (X.shape[0] + alpha) == pytest.approx(1 / (1 + 1e-2))
