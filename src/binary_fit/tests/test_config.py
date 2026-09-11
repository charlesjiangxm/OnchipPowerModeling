import numpy as np
import pytest
import yaml

from binary_fit.config import Config


def _write_cfg(tmp_path, extra=None):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(extra or {}))
    return p


def test_override_yaml_semantics(tmp_path):
    cfg = Config.from_yaml(
        _write_cfg(tmp_path),
        overrides=[
            "runtime.allow_tiny=false",
            "selection.target_qs=[8, 16]",
            "data.target=Pc(x_aq_core)",
            "build.modules=[cp0, idu]",
        ],
    )
    assert cfg.runtime.allow_tiny is False  # 'false' must not become a string
    assert cfg.selection.target_qs == [8, 16]
    assert cfg.data.target == "Pc(x_aq_core)"
    assert cfg.build.modules == ["cp0", "idu"]


def test_val_fraction_validation(tmp_path):
    p = _write_cfg(tmp_path, {"split": {"val_fraction": 1.0}})
    with pytest.raises(ValueError):
        Config.from_yaml(p)


def test_window_size_default_and_validation(tmp_path):
    assert Config.from_yaml(_write_cfg(tmp_path)).data.window_size == 32
    assert Config.from_yaml(
        _write_cfg(tmp_path), overrides=["data.window_size=64"]
    ).data.window_size == 64
    for bad in (0, -1):
        with pytest.raises(ValueError):
            Config.from_yaml(_write_cfg(tmp_path, {"data": {"window_size": bad}}))


def test_window_size_changes_the_study_stamp(tmp_path):
    """window_size reshapes the design matrix, so HPO must not reuse old trials."""
    from binary_fit.hpo import study_stamp

    cfg = Config.from_yaml(_write_cfg(tmp_path))
    s0 = study_stamp(cfg, np.array([1, 2, 3]))
    cfg.data.window_size = 64
    assert study_stamp(cfg, np.array([1, 2, 3])) != s0


def test_unknown_key_rejected(tmp_path):
    p = _write_cfg(tmp_path, {"data": {"nonexistent_knob": 1}})
    with pytest.raises(KeyError):
        Config.from_yaml(p)


def test_removed_keys_are_rejected(tmp_path):
    # multi-bit / disk-cache knobs no longer exist, nor does the trace row cap
    # (figures are never decimated now, so there is nothing left to configure)
    for bad in ({"data": {"bit_expand": True}}, {"data": {"db_root": "x"}},
                {"eval": {"trace_plot_cycles": 12000}}):
        with pytest.raises(KeyError):
            Config.from_yaml(_write_cfg(tmp_path, bad))


def test_stage_hash_tracks_seed_but_not_run_name(tmp_path):
    cfg = Config.from_yaml(_write_cfg(tmp_path))
    h0 = cfg.stage_hash("data", "split", "selection")
    cfg.runtime.run_name = "somewhere_else"
    assert cfg.stage_hash("data", "split", "selection") == h0
    cfg.runtime.seed = 1
    assert cfg.stage_hash("data", "split", "selection") != h0


def test_study_stamp_tracks_proxies_and_config(tmp_path):
    from binary_fit.hpo import study_stamp

    cfg = Config.from_yaml(_write_cfg(tmp_path))
    s0 = study_stamp(cfg, np.array([1, 2, 3]))
    assert study_stamp(cfg, np.array([1, 2, 3])) == s0
    assert study_stamp(cfg, np.array([1, 2, 4])) != s0
    cfg.split.val_fraction = 0.3
    assert study_stamp(cfg, np.array([1, 2, 3])) != s0


# --------------------------------------------------------------------------- #
# ridge section
# --------------------------------------------------------------------------- #
def test_ridge_section_loads_and_guards_its_grid(tmp_path):
    cfg = Config.from_yaml(_write_cfg(tmp_path))
    assert (cfg.ridge.alpha_rel_max, cfg.ridge.grid_points) == (1e2, 25)
    assert Config.from_yaml(
        _write_cfg(tmp_path, {"ridge": {"grid_points": 8, "max_rows": 0}})
    ).ridge.grid_points == 8
    for bad in ({"grid_points": 0}, {"alpha_rel_max": 0.0}, {"grid_decades": -1.0},
                {"max_rows": -1}):
        with pytest.raises(ValueError):
            Config.from_yaml(_write_cfg(tmp_path, {"ridge": bad}))


def test_ridge_alpha_rel_max_rejects_an_unsigned_yaml_exponent(tmp_path):
    """`alpha_rel_max: 1.0e4` is a STRING in plain YAML (the exponent needs a sign).

    Caught at load time with a message that says so, rather than surfacing much
    later as a TypeError from inside numpy.
    """
    p = tmp_path / "c.yaml"
    p.write_text("ridge:\n  alpha_rel_max: 1.0e4\n")
    with pytest.raises(ValueError, match="needs a sign"):
        Config.from_yaml(p)
    p.write_text("ridge:\n  alpha_rel_max: 1.0e+4\n")
    assert Config.from_yaml(p).ridge.alpha_rel_max == 1e4


def test_ridge_knobs_do_not_move_the_study_stamp(tmp_path):
    """Why the ridge knobs are NOT in HpoConfig.

    study_stamp hashes the whole `hpo` section into every tree study name, so
    adding fields there would rename the studies and orphan the trials already in
    an existing analysis/.../optuna.db. Do not "tidy" them into `hpo`.
    """
    from binary_fit.hpo import study_stamp

    cfg = Config.from_yaml(_write_cfg(tmp_path))
    ids = np.array([1, 2, 3])
    s0 = study_stamp(cfg, ids)
    cfg.ridge.alpha_rel_max = 1.0
    cfg.ridge.grid_points = 3
    cfg.ridge.max_rows = 0
    assert study_stamp(cfg, ids) == s0
    cfg.hpo.n_trials += 1  # a real hpo knob still does move it
    assert study_stamp(cfg, ids) != s0


# --------------------------------------------------------------------------- #
# rulefit section
# --------------------------------------------------------------------------- #
def test_rulefit_section_loads_and_guards_its_knobs(tmp_path):
    """Defaults, partial override, and ONE raises() per guard.

    One case per guard rather than a single loop over a mixed dict, so a dropped
    guard cannot hide behind a neighbour that happens to reject the same config.
    """
    cfg = Config.from_yaml(_write_cfg(tmp_path))
    r = cfg.rulefit
    assert (r.max_rules, r.n_alphas, r.hpo_n_trials) == (600, 20, 12)
    # the four divergences from the library's own defaults, pinned so nobody
    # "restores" them: see config.py for the measurement behind each
    assert r.lin_trim_quantile == 0.0
    assert r.exp_rand_tree_size is False
    assert r.fit_intercept is True
    assert r.non_negative_coef is True
    assert Config.from_yaml(
        _write_cfg(tmp_path, {"rulefit": {"max_rules": 50, "fit_rows": 0}})
    ).rulefit.max_rules == 50

    for bad in ({"max_rules": 0}, {"max_rules": 600.5}, {"n_alphas": 0},
                {"n_alphas": [10, 20]}, {"cv": 1}, {"tol": 0.0}, {"max_iter": 0},
                {"predict_chunk": 0}, {"lin_trim_quantile": 0.6},
                {"memory_par": 1.5}, {"hpo_rows": -1}, {"fit_rows": -1},
                {"hpo_n_trials": 0}, {"hpo_timeout_s": -1.0}, {"h_top_k": 0},
                {"h_rows": -1}, {"tree_size": 1}, {"fit_intercept": "yes"},
                {"tree_size": 2, "exp_rand_tree_size": True},
                {"include_interior_rules": False}):
        with pytest.raises(ValueError):
            Config.from_yaml(_write_cfg(tmp_path, {"rulefit": bad}))


def test_rulefit_include_interior_rules_may_be_off_with_signed_coefficients(tmp_path):
    """The interior-rules guard is CONDITIONAL on non_negative_coef.

    A non-negative model can express a decreasing effect only through a sibling
    ("> split") rule, because a linear term has no complement -- so dropping the
    interior rules while forcing non-negative coefficients removes the only
    mechanism it had. With signed coefficients there is no such dependency, and
    the combination must stay available.
    """
    cfg = Config.from_yaml(_write_cfg(tmp_path, {"rulefit": {
        "include_interior_rules": False, "non_negative_coef": False}}))
    assert cfg.rulefit.include_interior_rules is False


def test_rulefit_spec_literal_intercept_warns_but_loads(tmp_path, caplog):
    """`fit_intercept: false` + non-negative is the doc/spec combination.

    It has to stay reachable, so validate() WARNS rather than raising -- but it
    is also exactly the case the library's own collapse warning has a special
    message for, and it measured 8-80x slower here with a worse fit.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        caplog.clear()
        cfg = Config.from_yaml(_write_cfg(tmp_path, {"rulefit": {"fit_intercept": False}}))
    assert cfg.rulefit.fit_intercept is False
    assert any("fit_intercept=false" in m for m in caplog.messages)


def test_rulefit_tol_rejects_an_unsigned_yaml_exponent(tmp_path):
    """`tol: 1.0e-3` is fine, `tol: 1.0e3` is a STRING (the exponent needs a sign)."""
    p = tmp_path / "c.yaml"
    p.write_text("rulefit:\n  tol: 1.0e3\n")
    with pytest.raises(ValueError, match="needs a sign"):
        Config.from_yaml(p)
    p.write_text("rulefit:\n  tol: 1.0e-3\n")
    assert Config.from_yaml(p).rulefit.tol == 1e-3


def test_rulefit_knobs_do_not_move_the_study_stamp(tmp_path):
    """Why the rulefit knobs are NOT in HpoConfig -- the same stake as ridge's.

    study_stamp hashes the whole `hpo` section into every tree study name, and
    there are live analysis/binary-fit/*proxy-4cyc/cobit/tree/optuna.db files. So
    "tidying" hpo_n_trials into HpoConfig beside nn_n_trials would rename every
    tree study and orphan its trials SILENTLY -- load_if_exists=True simply starts
    a fresh study under the new name.
    """
    from binary_fit.hpo import study_stamp

    cfg = Config.from_yaml(_write_cfg(tmp_path))
    ids = np.array([1, 2, 3])
    s0 = study_stamp(cfg, ids)
    cfg.rulefit.max_rules = 111
    cfg.rulefit.n_alphas = 5
    cfg.rulefit.hpo_n_trials = 99
    cfg.rulefit.fit_rows = 7
    cfg.rulefit.non_negative_coef = False
    cfg.rulefit.lin_trim_quantile = 0.1
    assert study_stamp(cfg, ids) == s0
    cfg.hpo.n_trials += 1  # a real hpo knob still does move it
    assert study_stamp(cfg, ids) != s0
