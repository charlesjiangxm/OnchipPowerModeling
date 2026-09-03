import pytest

from binary_fit import build_db, run
from binary_fit.utils import load_json, load_proxies_csv

pytest.importorskip("xgboost")


@pytest.mark.slow
def test_end_to_end_build_select_fit(synth, tmp_path):
    """build_db -> feature_select -> fit (tree, no-hpo) recovers the planted bits."""
    cfg, planted = synth
    build_db.build(cfg)

    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)

    names, col_ids, weights = load_proxies_csv(outdir / "proxies.csv")
    assert set(planted) <= set(names)  # the signal-carrying bits are selected

    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["tree"], use_hpo=False)
    rec = load_json(outdir / "tree" / "all" / "result.json")
    assert rec["method"] == "tree" and rec["q"] == len(names)
    assert rec["test_r2"] > 0.5  # learnable synthetic signal


@pytest.mark.slow
def test_end_to_end_windowed(synth_windowed, tmp_path):
    """Same pipeline on window-averaged rows: proxies recovered, fit still learns."""
    cfg, planted = synth_windowed
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    assert load_json(outdir / "proxies.json")["window_size"] == cfg.data.window_size

    names, _, _ = load_proxies_csv(outdir / "proxies.csv")
    assert set(planted) <= set(names)  # averaging preserves the linear power law

    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["tree"], use_hpo=False)
    rec = load_json(outdir / "tree" / "all" / "result.json")
    assert rec["window_size"] == cfg.data.window_size
    # window rows are the mean of w cycles: fewer rows, far less label noise
    assert rec["reports"]["test"]["n_cycles"] == 320 // cfg.data.window_size
    assert rec["test_r2"] > 0.5


@pytest.mark.slow
def test_end_to_end_nn(synth, tmp_path):
    cfg, planted = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["nn"], use_hpo=False)
    rec = load_json(outdir / "nn" / "all" / "result.json")
    assert rec["method"] == "nn"
    assert rec["test_r2"] > 0.3
