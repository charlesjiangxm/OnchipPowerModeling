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
