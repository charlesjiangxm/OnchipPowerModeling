import logging

import numpy as np

from binary_fit.config import Config
from binary_fit.run import _build_selections, _warn_window_mismatch


def _names_pos_w(n=3):
    names = [f"n{i}" for i in range(n)]
    pos = np.arange(10, 10 + 10 * n, 10, dtype=np.int64)
    w = np.arange(n, 0, -1, dtype=float)
    return names, pos, w


def test_build_selections_topq_clamp_and_dedup():
    names, pos, w = _names_pos_w(3)
    sels = _build_selections([2, -1, 5], names, pos, w)
    labels = [s[0] for s in sels]
    assert labels == ["q2", "all"]  # 5 clamps to all, dedup vs -1

    q2 = sels[0]
    assert q2[1] == ["n0", "n1"] and q2[2].tolist() == [10, 20]  # top-2 by rank
    allsel = sels[1]
    assert allsel[1] == names and allsel[2].tolist() == [10, 20, 30]


def test_build_selections_invalid_q_skipped():
    names, pos, w = _names_pos_w(3)
    assert _build_selections([0], names, pos, w) == []
    assert [s[0] for s in _build_selections([1, 2], names, pos, w)] == ["q1", "q2"]


# --------------------------------------------------------------------------- #
# window_size provenance guard
# --------------------------------------------------------------------------- #
def _warn_for(tmp_path, meta_text, window, caplog):
    """Run the guard against a proxies.json holding ``meta_text``; return warnings."""
    csv = tmp_path / "proxies.csv"
    csv.write_text("rank,name,col_id,mcp_weight\n")
    if meta_text is not None:
        (tmp_path / "proxies.json").write_text(meta_text)
    cfg = Config()
    cfg.data.window_size = window
    with caplog.at_level(logging.WARNING, logger="binary_fit"):
        caplog.clear()
        _warn_window_mismatch(cfg, csv)
    return [r.getMessage() for r in caplog.records]


def test_window_mismatch_warns_on_a_recorded_mismatch(tmp_path, caplog):
    msgs = _warn_for(tmp_path, '{"window_size": 8}', 32, caplog)
    assert len(msgs) == 1 and "window_size=8 (recorded)" in msgs[0]
    assert _warn_for(tmp_path, '{"window_size": 32}', 32, caplog) == []


def test_window_mismatch_treats_a_missing_key_as_per_cycle(tmp_path, caplog):
    """A proxies.json predating the knob was selected per-cycle: warn at 32."""
    msgs = _warn_for(tmp_path, '{"proxies": {}}', 32, caplog)
    assert len(msgs) == 1 and "window_size=1" in msgs[0] and "predates" in msgs[0]
    assert _warn_for(tmp_path, '{"proxies": {}}', 1, caplog) == []


def test_window_mismatch_never_fails_the_fit(tmp_path, caplog):
    """Malformed, non-object or absent provenance is advisory, never fatal."""
    assert _warn_for(tmp_path, None, 32, caplog) == []  # no proxies.json at all
    for bad in ("{not json", "[1, 2, 3]", '{"window_size": "many"}'):
        msgs = _warn_for(tmp_path, bad, 32, caplog)
        assert len(msgs) == 1
        assert "skipping" in msgs[0] or "cannot read" in msgs[0]
