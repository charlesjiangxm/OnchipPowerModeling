"""Unit tests for the rulefit backend's estimator wiring.

The estimator itself is the vendored fork's (134 of its own tests cover ``fit``,
``predict`` and ``get_rules``), so what is pinned here is everything
``binary_fit`` puts around it: the eq-(35) importance vector the shared
``coefficients.csv`` consumes, the batched prediction path, the term-to-column-id
mapping ``rulefit_terms.csv`` publishes, and the config translations that are
silently wrong if they drift. Each test names the mutation it kills.
"""

import numpy as np
import pytest

from binary_fit import models
from binary_fit.config import Config

pytest.importorskip("rulefit")


def _cfg(**over):
    """A small, fast rulefit config: the shipped defaults are sized for real data."""
    cfg = Config()
    cfg.rulefit.max_rules = 60
    cfg.rulefit.cv = 2
    cfg.rulefit.n_jobs = 1
    for key, value in over.items():
        setattr(cfg.rulefit, key, value)
    return cfg


def _fit(cfg, X, y, names=None, seed=0):
    names = names or [f"x_aq_cp0_top/net_{j}[3:0]" for j in range(X.shape[1])]
    rf = models.fit_rulefit(models.make_rulefit(cfg.rulefit, seed), X, y, names)
    return rf, names


def _linear_case(n=600, p=6, seed=0, dtype=np.float32):
    """Densities in [0, 1] and an exactly linear, all-positive power law."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, p)).astype(dtype)
    w = rng.random(p) * 0.01 + 0.002
    y = 0.042 + np.asarray(X, dtype=np.float64) @ w + rng.normal(0, 1e-5, n)
    return X, y, w


# --------------------------------------------------------------------------- #
# importances -- what utils.save_coefficients_csv consumes
# --------------------------------------------------------------------------- #
def test_rulefit_importance_is_non_negative_and_sized():
    """Kills returning ``rf.coef_``, which is both signed and the wrong length."""
    X, y, _ = _linear_case()
    rf, _ = _fit(_cfg(), X, y)
    imp = models.rulefit_importance(rf, X.shape[1])
    assert imp.shape == (X.shape[1],)
    assert np.all(imp >= 0) and imp.sum() > 0
    # the length is asserted inside, not merely documented
    with pytest.raises(ValueError, match="shape"):
        models.rulefit_importance(rf, X.shape[1] + 1)


def test_rulefit_importance_credits_a_feature_through_its_rules_alone():
    """THE decisive one: a feature that appears only inside rules still counts.

    Kills substituting ``_linear_importances``, which ignores every rule -- the
    single most likely wrong choice here. It would report a purely interacting
    bit as unimportant, and if every linear term were zero it would hand
    save_coefficients_csv an all-zero vector, silently republishing the Stage-1
    MCP ranking as if it were the model's own.
    """
    rng = np.random.default_rng(1)
    X = rng.random((900, 4)).astype(np.float32)
    # a pure interaction: no main effect in either driver
    y = 0.04 + 0.02 * ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)) + rng.normal(0, 1e-5, 900)
    rf, _ = _fit(_cfg(max_rules=80), X, y.astype(np.float64))
    imp = models.rulefit_importance(rf, 4)
    assert rf.n_rules_ > 0
    assert imp[0] > 0 and imp[1] > 0
    # and the credit really did come from the rules: the drivers have no linear
    # term at all, so _linear_importances alone would report both as zero
    lin = np.asarray(rf.linear_coef_, dtype=np.float64)
    assert lin[0] == 0.0 and lin[1] == 0.0
    assert np.asarray(rf.rule_coef_).any()


def test_rulefit_importance_is_column_aligned():
    """Kills a re-sorted or transposed importance frame."""
    rng = np.random.default_rng(2)
    X = rng.random((600, 5)).astype(np.float32)
    X[:, 3] = 0.0  # an untouched (constant) column
    y = 0.04 + 0.05 * X[:, 1] + rng.normal(0, 1e-5, 600)
    rf, names = _fit(_cfg(), X, y.astype(np.float64))
    imp = models.rulefit_importance(rf, 5)
    assert int(imp.argmax()) == 1
    assert imp[3] == 0.0
    assert list(rf.get_feature_importance()["feature"]) == names


def test_rulefit_importance_survives_a_collapsed_fit(tmp_path):
    """An all-zero fit must still produce a writable, correctly sized vector.

    The collapse is CONSTRUCTED, not assumed: with no intercept and no negative
    coefficients the model cannot go below 0 on a non-negative design, so an
    all-zero model wins cross-validation on a target that is centred well away
    from 0 in the wrong direction.
    """
    from binary_fit.utils import save_coefficients_csv

    rng = np.random.default_rng(3)
    X = rng.random((400, 4)).astype(np.float32)
    y = -0.5 - 0.01 * X[:, 0] + rng.normal(0, 1e-4, 400)  # strictly negative target
    cfg = _cfg(fit_intercept=False, non_negative_coef=True)
    rf, names = _fit(cfg, X, y.astype(np.float64))
    if np.any(np.asarray(rf.coef_) != 0):
        pytest.skip("this target did not collapse the fit on this sklearn build")
    imp = models.rulefit_importance(rf, 4)
    assert imp.shape == (4,) and imp.sum() == 0.0
    # save_coefficients_csv must not divide by a zero total
    path = tmp_path / "coefficients.csv"
    save_coefficients_csv(path, names, np.arange(4), np.ones(4), imp)
    assert len(path.read_text().strip().splitlines()) == 5  # header + 4 rows


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def test_rulefit_predict_chunked_matches_one_call():
    """Batched prediction is equivalent, NOT bit-identical.

    Measured max abs difference 1.11e-16 at chunk 1/7/n-1 from BLAS blocking,
    exact only at chunk >= n -- so ``np.array_equal`` would be red on day one.
    The tolerance is still tight enough to kill an off-by-one in the range() or a
    dropped final partial batch, which would shift or truncate whole rows.
    """
    X, y, _ = _linear_case(n=311)
    rf, _ = _fit(_cfg(), X, y)
    full = models.rulefit_predict(rf, X, 10 ** 9)
    scale = max(1.0, float(np.abs(full).max()))
    for chunk in (1, 7, len(X) - 1, len(X), len(X) + 1):
        got = models.rulefit_predict(rf, X, chunk)
        assert got.shape == full.shape
        np.testing.assert_allclose(got, full, rtol=0, atol=1e-12 * scale)


def test_rulefit_predict_handles_an_empty_split():
    """``rf.predict`` on an empty X RAISES; ``_run_one`` calls it unconditionally.

    The raw call fails with "Found array with 0 sample(s) ... required by
    LassoCV", and run._run_one builds splits["test"] whether or not it has rows
    ("reported even when empty") and scores it. So the guard is load-bearing, not
    defensive.
    """
    X, y, _ = _linear_case(n=200)
    rf, _ = _fit(_cfg(), X, y)
    with pytest.raises(ValueError):
        rf.predict(X[:0])
    out = models.rulefit_predict(rf, X[:0], 50_000)
    assert out.shape == (0,) and out.dtype == np.float64


# --------------------------------------------------------------------------- #
# the term table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-6)])
def test_rulefit_terms_reproduce_the_prediction(dtype, tol):
    """The identity that makes rulefit_terms.csv quotable, at lin_trim_quantile 0.

    ``intercept + x . linear_coef_ + r(x) . rule_coef_`` IS the prediction, in
    watts, against the RAW feature -- no un-scaling arithmetic, unlike ridge's.
    The float32 row is the dtype ``Union.slice`` actually hands over.
    """
    X, y, _ = _linear_case(dtype=dtype)
    rf, _ = _fit(_cfg(), X, y)
    lin = np.asarray(rf.linear_coef_, dtype=np.float64)
    rule = np.asarray(rf.rule_coef_, dtype=np.float64)
    direct = (float(np.ravel(rf.intercept_)[0])
              + np.asarray(X, dtype=np.float64) @ lin
              + rf.rule_ensemble.transform(X) @ rule)
    yhat = models.rulefit_predict(rf, X, 10 ** 9)
    assert np.abs(direct - yhat).max() <= tol * max(1.0, float(np.abs(yhat).max()))


def test_rulefit_terms_carry_index_mapped_col_ids():
    """The one that catches the worst bug: a rule pointing at the wrong RTL net.

    ``get_rules(exclude_zero_coef=True)`` filters with ``.loc[]``, so the frame
    keeps its ORIGINAL, now-gapped index into the ``[linear | rules]`` column
    order. ``enumerate``-ing it instead would offset every rule's variables. A
    NON-CONTIGUOUS col_ids array is what makes that visible -- with 0..p-1 the
    bug is invisible.
    """
    X, y, _ = _linear_case(p=4)
    col_ids = np.array([7, 19, 42, 88])
    rf, names = _fit(_cfg(), X, y)
    rows = models.rulefit_terms(rf, names, col_ids)

    assert len(rows) == int((np.asarray(rf.coef_) != 0).sum())
    allowed = set(col_ids.tolist())
    for row in rows:
        ids = [int(c) for c in row["col_ids"].split(";")]
        assert set(ids) <= allowed
        assert len(ids) == row["n_variables"]
        if row["type"] == "linear":
            assert ids == [int(col_ids[names.index(row["term"])])]
    # importance desc, which is the order save_rulefit_terms_csv ranks from
    imps = [r["importance"] for r in rows]
    assert imps == sorted(imps, reverse=True)
    # gapped indices really are in play, otherwise the test proves nothing
    assert list(rf.get_rules(exclude_zero_coef=True).index) != list(range(len(rows)))


def test_rulefit_summary_matches_the_coefficients():
    """The size numbers must agree with the fitted vector, and max_rules is SOFT.

    Kills slicing ``coef_[-n_rules_:]``, which is everything when n_rules_ == 0.
    ``n_rules <= max_rules`` never ``==``: the cap sizes the ensemble and
    identical conjunctions are then merged (measured 315 rules from a 600 cap).
    """
    X, y, _ = _linear_case()
    cfg = _cfg()
    rf, _ = _fit(cfg, X, y)
    s = models.rulefit_summary(rf)
    coef = np.asarray(rf.coef_)
    assert s["n_nonzero_terms"] == int((coef != 0).sum())
    assert s["n_nonzero_linear"] + s["n_nonzero_rules"] == s["n_nonzero_terms"]
    assert s["n_linear_terms"] == X.shape[1]
    assert s["n_rules"] == rf.n_rules_ <= cfg.rulefit.max_rules
    assert s["intercept_watts"] == pytest.approx(float(np.ravel(rf.intercept_)[0]))
    assert s["allow_negative_coef"] is False


# --------------------------------------------------------------------------- #
# the config translations and the deliberate divergences
# --------------------------------------------------------------------------- #
def test_make_rulefit_translates_the_config_sentinels():
    """``n_jobs: 0 -> None``, ``n_alphas -> Cs``, and the sign flag resolved."""
    cfg = _cfg(n_jobs=0, n_alphas=7)
    rf = models.make_rulefit(cfg.rulefit, 0)
    assert rf.n_jobs is None
    assert rf.Cs == 7 and isinstance(rf.Cs, int)  # an int Cs is a path LENGTH
    # allow_negative_coef must be an explicit bool: None silently means
    # non-negative for regression, which would hide the decision
    assert rf.allow_negative_coef is False
    assert models.make_rulefit(_cfg(non_negative_coef=False).rulefit,
                               0).allow_negative_coef is True
    assert rf.fit_intercept is True


def test_n_alphas_reaches_the_solver():
    """Behavioural, not an argument check: the alpha path really is that long."""
    X, y, _ = _linear_case(n=300)
    for n_alphas in (7, 20):
        rf, _ = _fit(_cfg(n_alphas=n_alphas), X, y)
        assert len(rf.lscv.alphas_) == n_alphas
        assert models.rulefit_summary(rf)["n_alphas"] == n_alphas


def test_non_negative_coef_costs_a_signed_law():
    """What the shipped default gives up, measured rather than assumed.

    With signed coefficients the negative slope is recovered; with the shipped
    non-negative default it cannot be, because a linear term has no complement
    (the library's own docstring says so) and it has to be rebuilt out of rules.
    Ridge reaches test R2 0.9148 on the real proxies using signed coefficients,
    so `rulefit.non_negative_coef: false` is the first thing to try if the fit
    underperforms.
    """
    rng = np.random.default_rng(5)
    X = rng.random((900, 3)).astype(np.float32)
    y = (0.5 + 0.30 * X[:, 0] + 0.20 * X[:, 1] - 0.15 * X[:, 2]
         + rng.normal(0, 1e-4, 900)).astype(np.float64)
    signed, _ = _fit(_cfg(non_negative_coef=False), X, y)
    assert np.asarray(signed.linear_coef_)[2] < 0
    nonneg, _ = _fit(_cfg(non_negative_coef=True), X, y)
    assert np.all(np.asarray(nonneg.coef_) >= 0)


def test_lin_trim_quantile_zero_keeps_a_sparse_bit_in_the_linear_block():
    """The guard on the divergence from the library's 0.025 default.

    At 0.025 a column non-zero in ~1% of rows has winsor limits [0, 0], so its
    trimmed std is 0 and FriedScale emits an identically-zero design column: its
    linear coefficient and its eq-(29) importance are forced to 0 with no
    warning. That is exactly the rare-but-high-power bit this backend is for.
    Kills anyone "restoring the library default".
    """
    rng = np.random.default_rng(6)
    n, p = 1000, 4
    X = rng.random((n, p)).astype(np.float32)
    sparse = np.zeros(n, dtype=np.float32)
    sparse[rng.choice(n, 10, replace=False)] = 1.0  # non-zero in 1% of rows
    X[:, 2] = sparse
    y = (0.04 + 0.01 * X[:, 0] + 0.05 * X[:, 2]
         + rng.normal(0, 1e-5, n)).astype(np.float64)

    trimmed, _ = _fit(_cfg(lin_trim_quantile=0.025), X, y)
    assert trimmed.stddev[2] == 0.0
    assert not trimmed.friedscale.scale(X)[:, 2].any()

    kept, _ = _fit(_cfg(lin_trim_quantile=0.0), X, y)
    assert kept.stddev[2] > 0.0
    assert kept.friedscale.scale(X)[:, 2].any()


def test_rulefit_survives_duplicate_and_constant_columns():
    """80.3% of the aq_core kept bits are exact copies of another bit."""
    rng = np.random.default_rng(7)
    X = rng.random((600, 8)).astype(np.float32)
    X[:, 4:] = X[:, :4]  # exact duplicates
    X[:, 3] = 0.7  # constant
    y = (0.04 + 0.01 * X[:, 0] + rng.normal(0, 1e-5, 600)).astype(np.float64)
    rf, _ = _fit(_cfg(), X, y)
    imp = models.rulefit_importance(rf, 8)
    assert np.all(np.isfinite(imp)) and np.all(imp >= 0)
    assert np.all(np.isfinite(rf.friedscale.scale_multipliers))
    assert np.all(np.isfinite(models.rulefit_predict(rf, X, 100)))


def test_suggest_rulefit_params_respects_the_cap():
    """A trial can never grow a bigger ensemble than the final fit is allowed."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for cap in (1200, 300, 50):  # the last is below the space's own floor of 100
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for _ in range(20):
            trial = study.ask()
            params = models.suggest_rulefit_params(trial, cap)
            assert params["max_rules"] <= cap
            assert 2 <= params["tree_size"] <= 8
            study.tell(trial, 0.0)
