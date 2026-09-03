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
    # multi-bit / disk-cache knobs no longer exist
    for bad in ({"data": {"bit_expand": True}}, {"data": {"db_root": "x"}}):
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
