import numpy as np
import pytest
import yaml

from cobit.config import CobitConfig


def _write_cfg(tmp_path, extra=None):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(extra or {}))
    return p


def test_override_yaml_semantics(tmp_path):
    p = _write_cfg(tmp_path)
    cfg = CobitConfig.from_yaml(
        p,
        overrides=[
            "runtime.allow_tiny=false",
            "hpo.run_pair_comparison=true",
            "hpo.pair_q=null",
            "selection.target_qs=[8, 16]",
            "data.target=Pc(x_aq_core)",
        ],
    )
    assert cfg.runtime.allow_tiny is False  # 'false' must not become a string
    assert cfg.hpo.run_pair_comparison is True
    assert cfg.hpo.pair_q is None
    assert cfg.selection.target_qs == [8, 16]
    assert cfg.data.target == "Pc(x_aq_core)"


def test_val_fraction_validation(tmp_path):
    p = _write_cfg(tmp_path, {"split": {"val_fraction": 1.0}})
    with pytest.raises(ValueError):
        CobitConfig.from_yaml(p)


def test_unknown_key_rejected(tmp_path):
    p = _write_cfg(tmp_path, {"data": {"nonexistent_knob": 1}})
    with pytest.raises(KeyError):
        CobitConfig.from_yaml(p)


def test_stage_hash_tracks_seed_but_not_run_name(tmp_path):
    cfg = CobitConfig.from_yaml(_write_cfg(tmp_path))
    h0 = cfg.stage_hash("data", "split", "selection")
    cfg.runtime.run_name = "somewhere_else"
    assert cfg.stage_hash("data", "split", "selection") == h0
    cfg.runtime.seed = 1
    assert cfg.stage_hash("data", "split", "selection") != h0


def test_study_stamp_tracks_proxies_and_config(tmp_path):
    from cobit.hpo import study_stamp

    cfg = CobitConfig.from_yaml(_write_cfg(tmp_path))
    s0 = study_stamp(cfg, np.array([1, 2, 3]))
    assert study_stamp(cfg, np.array([1, 2, 3])) == s0
    assert study_stamp(cfg, np.array([1, 2, 4])) != s0  # proxy set changed
    cfg.split.val_fraction = 0.3
    assert study_stamp(cfg, np.array([1, 2, 3])) != s0  # config changed
