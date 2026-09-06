import logging

import numpy as np
import pytest

from binary_fit import run
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


# --------------------------------------------------------------------------- #
# --model surface
# --------------------------------------------------------------------------- #
def _main_kinds(tmp_path, monkeypatch, *argv):
    """Run main() with cmd_fit stubbed; return the model_kinds it was handed."""
    seen = {}
    monkeypatch.setattr(run, "cmd_fit",
                        lambda cfg, outdir, proxies, qs, kinds, use_hpo:
                        seen.update(kinds=kinds, use_hpo=use_hpo) or 0)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}")
    (tmp_path / "proxies.csv").write_text("rank,name,col_id,mcp_weight\n")
    run.main(["--fit", "--config", str(cfg), "--outdir", str(tmp_path), *argv])
    return seen


def test_model_both_now_expands_to_every_kind(tmp_path, monkeypatch):
    """`both` meant tree+nn when there were two backends; it means ALL of them.

    Pinned because it is the one behaviour change the ridge backend made to an
    existing command line.
    """
    seen = _main_kinds(tmp_path, monkeypatch, "--model", "both", "--no-hpo")
    assert seen["kinds"] == list(run.MODEL_KINDS) == ["tree", "nn", "ridge"]
    assert seen["use_hpo"] is False


def test_model_selects_a_single_kind(tmp_path, monkeypatch):
    for kind in run.MODEL_KINDS:
        seen = _main_kinds(tmp_path, monkeypatch, "--model", kind)
        assert seen["kinds"] == [kind] and seen["use_hpo"] is True
    assert _main_kinds(tmp_path, monkeypatch)["kinds"] == ["tree"]  # the default


# --------------------------------------------------------------------------- #
# ridge backend wiring
# --------------------------------------------------------------------------- #
def test_cap_ridge_rows_is_seeded_and_only_fires_above_the_cap():
    """Row-aligned, reproducible, and a no-op unless the cap actually bites."""
    cfg = Config()
    X = np.arange(200, dtype=np.float32).reshape(100, 2)  # row i = [2i, 2i+1]
    y = np.arange(100.0)

    cfg.ridge.max_rows = 0
    assert run._cap_ridge_rows(cfg, X, y)[0] is X  # 0 means every row
    cfg.ridge.max_rows = 500
    assert run._cap_ridge_rows(cfg, X, y)[0] is X  # cap above n: untouched

    cfg.ridge.max_rows = 20
    Xa, ya = run._cap_ridge_rows(cfg, X, y)
    Xb, yb = run._cap_ridge_rows(cfg, X, y)
    assert Xa.shape == (20, 2) and ya.shape == (20,)
    assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)  # seeded
    assert np.array_equal(Xa[:, 0], ya * 2)  # X and y still describe the same rows
    assert np.all(np.diff(ya) > 0)  # sorted, so benchmark order is preserved


@pytest.mark.parametrize("use_hpo,with_val,expected_rows",
                         [(True, True, 50), (False, True, 40), (True, False, 40)])
def test_fit_ridge_row_set_depends_on_hpo(tmp_path, use_hpo, with_val, expected_rows):
    """With HPO the val split JOINS the fit; with --no-hpo ridge fits train only.

    Ridge's alpha search never scores val (RidgeCV's GCV runs inside the rows it
    is handed), so under HPO train+val go in as one set -- the same data the tree
    and nn backends refit on -- which is what makes _run_one's "in HPO refit"
    captions correct. Under --no-hpo it fits train only, like the other two, so
    the "in-sample"/held-out captions stay honest there too. The third case is
    split.val_fraction = 0, where ridge must still complete.
    """
    cfg = Config()
    rng = np.random.default_rng(0)
    w = np.array([1.0, 2.0, 3.0])
    Xtr = rng.random((40, 3)).astype(np.float32)
    ytr = Xtr @ w + 0.1
    if with_val:
        Xval = rng.random((10, 3)).astype(np.float32)
        yval = Xval @ w + 0.1
    else:
        Xval, yval = None, np.empty(0)

    qdir = tmp_path / "q"
    qdir.mkdir()
    best, imp, leaves, predict_fn = run._fit_ridge(
        cfg, qdir, Xtr, ytr, Xval, yval, ["a", "b", "c"], np.array([0, 1, 2]), use_hpo)

    assert best["n_fit_rows"] == expected_rows
    assert best["val_r2"] is None and leaves == ""  # nothing here scored val
    assert (best["gcv_neg_mse"] is None) is (not use_hpo)
    assert imp.shape == (3,) and np.all(imp >= 0)
    assert sorted(p.name for p in qdir.iterdir()) == ["model.joblib",
                                                      "ridge_coefficients.csv"]
    assert predict_fn(Xtr).shape == (40,)
