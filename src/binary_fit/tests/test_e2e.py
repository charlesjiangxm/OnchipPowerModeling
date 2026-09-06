import pytest

from binary_fit import build_db, run
from binary_fit.config import Config
from binary_fit.utils import load_json, load_proxies_csv

pytest.importorskip("xgboost")

# One no-HPO fit writes exactly this. Asserting the whole set (rather than a
# per-file exists()) is what catches a renamed figure, a missing one, and a
# _try_plot that swallowed a real error -- and it keeps README.md honest.
_FIT_ARTIFACTS = [
    "coefficients.csv",
    "predictions.pkl.zst",
    "pred_vs_time_test.png",
    "pred_vs_time_train.png",
    "pred_vs_time_val.png",
    "residual_train_val_test.png",
    "result.json",
]
# --model ridge writes one more: the signed linear fit. coefficients.csv is
# shared by all three backends and its `value` column is the Stage-1 MCP weight,
# so the fitted coefficients need their own file.
_RIDGE_ARTIFACTS = _FIT_ARTIFACTS + ["ridge_coefficients.csv"]
# _aggregate's own output. A lone experiment IS the model directory
# (run._experiment_dir), so these land beside that experiment's artifacts rather
# than a level above it. q_sweep.png is deliberately NOT here: one experiment is
# a single point, so the sweep is skipped.
_AGGREGATE_ARTIFACTS = ["metrics.json", "report.md"]


def _assert_artifacts(d, expected, model_file, *, aggregate=True):
    """Assert the exact artifact set of a fit directory.

    ``aggregate`` folds in the _aggregate-level files, which is the flat
    single-experiment layout -- and because the set is exact, it is also what
    pins that no ``all/`` or ``figures/`` subdirectory came back.
    """
    want = expected + [model_file] + (_AGGREGATE_ARTIFACTS if aggregate else [])
    assert sorted(p.name for p in d.iterdir()) == sorted(want)
    for png in d.glob("*.png"):
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png


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
    rec = load_json(outdir / "tree" / "result.json")
    assert rec["method"] == "tree" and rec["q"] == len(names)
    assert rec["test_r2"] > 0.5  # learnable synthetic signal
    _assert_artifacts(outdir / "tree", _FIT_ARTIFACTS, "model.json")
    assert not (outdir / "tree" / "trace_test.png").exists()


# --------------------------------------------------------------------------- #
# result layout (run._experiment_dir)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_a_lone_experiment_is_the_model_directory(synth, tmp_path):
    """The default -q -1 writes flat: no all/ wrapper, no figures/ wrapper.

    _assert_artifacts already pins the exact file set everywhere else; what is
    asserted here is the *shape* -- that the model directory has no
    subdirectories at all, and that report.md does not advertise the q_sweep
    figure that a one-point sweep declines to draw.
    """
    cfg, _ = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["ridge"], use_hpo=False)

    mdir = outdir / "ridge"
    assert [p.name for p in mdir.iterdir() if p.is_dir()] == []
    # the label survives the directory it no longer names
    assert load_json(mdir / "result.json")["label"] == "all"
    assert [r["label"] for r in load_json(mdir / "metrics.json")["records"]] == ["all"]
    text = (mdir / "report.md").read_text()
    assert "| all |" in text
    assert "Beside this report" in text  # not "under `<experiment>/`"
    assert "q_sweep" not in text and "figures/" not in text
    # proxies stay at the outdir: --feature_select is model-agnostic and one
    # proxy set is shared by every backend of a --model both run
    assert (outdir / "proxies.csv").exists() and not (mdir / "proxies.csv").exists()


@pytest.mark.slow
def test_several_q_values_keep_one_subdirectory_each(synth, tmp_path):
    """Two experiments cannot share a directory, so each keeps its <label>/.

    Flattening is for a lone experiment; here the model directory holds only the
    aggregate -- and q_sweep.png IS written, because there are two points to
    join. This is also the regression guard for _aggregate's discovery: it has
    to find experiments in subdirectories AND flat, and report both.
    """
    cfg, _ = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    names, _, _ = load_proxies_csv(outdir / "proxies.csv")
    q = len(names) - 1  # a strict subset, so its label is not "all"
    assert q >= 1, "the fixture must select at least two proxies"
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[q, -1],
                model_kinds=["tree"], use_hpo=False)

    mdir = outdir / "tree"
    assert sorted(p.name for p in mdir.iterdir()) == sorted(
        ["all", f"q{q}"] + _AGGREGATE_ARTIFACTS + ["q_sweep.png"])
    for label in ("all", f"q{q}"):
        _assert_artifacts(mdir / label, _FIT_ARTIFACTS, "model.json", aggregate=False)
    recs = load_json(mdir / "metrics.json")["records"]
    assert sorted(r["label"] for r in recs) == sorted(["all", f"q{q}"])
    text = (mdir / "report.md").read_text()
    assert "| all |" in text and f"| q{q} |" in text
    assert "under `<experiment>/`" in text
    assert "`q_sweep.png`" in text and "figures/" not in text


def test_aggregate_warns_when_a_stale_label_dir_shadows_the_flat_result(tmp_path, caplog):
    """Refitting into a pre-flattening outdir leaves all/ beside result.json.

    Both are then reported, so report.md grows two "all" rows with different
    scores and q_sweep is drawn across a duplicated point. _aggregate cannot tell
    which is current -- neither is wrong on its face -- so it says so loudly.
    Distinct labels coexisting is legitimate (-q 50 once, -q -1 later) and must
    stay quiet, which the second half checks.
    """
    from binary_fit.utils import save_json

    def rec(label, q, r2):
        return {"method": "tree", "label": label, "q": q, "window_size": 32,
                "final_leaves": 10, "train_r2": 0.9, "val_r2": 0.8,
                "test_r2": r2, "test_mape": 4.5}

    stale = tmp_path / "tree"
    (stale / "all").mkdir(parents=True)
    save_json(stale / "result.json", rec("all", 100, 0.92))       # the new flat fit
    save_json(stale / "all" / "result.json", rec("all", 100, 0.55))  # the old one
    with caplog.at_level("WARNING", logger="binary_fit"):
        records = run._aggregate(stale, "t")
    assert len(records) == 2
    assert "twice" in caplog.text and str(stale / "all") in caplog.text

    caplog.clear()
    ok = tmp_path / "nn"
    (ok / "q50").mkdir(parents=True)
    save_json(ok / "result.json", rec("all", 100, 0.92))
    save_json(ok / "q50" / "result.json", rec("q50", 50, 0.88))
    with caplog.at_level("WARNING", logger="binary_fit"):
        records = run._aggregate(ok, "t")
    assert sorted(r["label"] for r in records) == ["all", "q50"]
    assert "twice" not in caplog.text


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
    rec = load_json(outdir / "tree" / "result.json")
    assert rec["window_size"] == cfg.data.window_size
    # window rows are the mean of w cycles: fewer rows, far less label noise
    assert rec["reports"]["test"]["n_cycles"] == 320 // cfg.data.window_size
    assert rec["test_r2"] > 0.5
    _assert_artifacts(outdir / "tree", _FIT_ARTIFACTS, "model.json")


@pytest.mark.slow
def test_end_to_end_nn(synth, tmp_path):
    cfg, planted = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["nn"], use_hpo=False)
    rec = load_json(outdir / "nn" / "result.json")
    assert rec["method"] == "nn"
    assert rec["test_r2"] > 0.3
    # the README invariant: both backends share the evaluation and the figures
    _assert_artifacts(outdir / "nn", _FIT_ARTIFACTS, "model.joblib")


@pytest.mark.slow
def test_end_to_end_no_validation_split(synth, tmp_path):
    """val_fraction 0 -> Union.Xval is None: the fit still completes.

    This is the path a naive preds dict would take through predict_fn(None), and
    where the residual figure would try to build a 0-column subplot grid.
    Nothing else drives cmd_fit with no validation split.
    """
    cfg, _ = synth
    cfg.split.val_fraction = 0.0
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["tree"], use_hpo=False)  # HPO needs a val split
    qdir = outdir / "tree"
    assert load_json(qdir / "result.json")["val_r2"] is None
    expected = [f for f in _FIT_ARTIFACTS
                if f not in ("pred_vs_time_val.png", "residual_train_val_test.png")]
    _assert_artifacts(qdir, expected + ["residual_train_test.png"], "model.json")


@pytest.mark.slow
def test_a_plot_failure_does_not_lose_the_fit(synth, tmp_path, monkeypatch, caplog):
    """A render failure must cost the figure and nothing else.

    result.json is written first and _aggregate reads it back from disk, so an
    unguarded plot call could erase a completed experiment from report.md.
    """
    cfg, _ = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "plot_residual_panels", _boom)
    with caplog.at_level("ERROR", logger="binary_fit"):
        run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                    model_kinds=["tree"], use_hpo=False)
    qdir = outdir / "tree"
    assert load_json(qdir / "result.json")["test_r2"] > 0.5
    assert not (qdir / "residual_train_val_test.png").exists()
    assert (qdir / "pred_vs_time_test.png").exists()  # the other figures still ran
    assert "boom" in caplog.text
    # and the experiment is still in the aggregate report
    assert "| all |" in (outdir / "tree" / "report.md").read_text()


@pytest.mark.slow
def test_predictions_pkl_reproduces_the_reported_metrics(synth, tmp_path):
    """predictions.pkl.zst holds the SCORED predictions, not a second inference:
    its per-benchmark R2 has to match result.json exactly."""
    import pandas as pd

    from binary_fit.evaluate import r2_score

    cfg, _ = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["tree"], use_hpo=False)
    qdir = outdir / "tree"
    rec = load_json(qdir / "result.json")
    frame = pd.read_pickle(qdir / "predictions.pkl.zst")
    assert list(frame.columns) == ["bench", "split", "y_true", "y_pred"]
    for split, report in rec["reports"].items():
        sub = frame[frame["split"] == split]
        assert len(sub) == report["n_cycles"]
        for bench, per in report["per_benchmark"].items():
            b = sub[sub["bench"] == bench]
            assert r2_score(b["y_true"].to_numpy(),
                            b["y_pred"].to_numpy()) == pytest.approx(per["r2"])


# --------------------------------------------------------------------------- #
# ridge backend
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_end_to_end_ridge(synth, tmp_path):
    """The fixture's power law is exactly linear in three bits, so ridge should
    all but nail it -- and its extra coefficient artifact must appear."""
    import csv

    cfg, planted = synth
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    names, _, _ = load_proxies_csv(outdir / "proxies.csv")
    # use_hpo=True so the RidgeCV alpha search is what runs, not the fixed alpha
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["ridge"], use_hpo=True)

    qdir = outdir / "ridge"
    rec = load_json(qdir / "result.json")
    assert rec["method"] == "ridge" and rec["final_leaves"] == ""
    assert rec["test_r2"] > 0.8  # tree asserts 0.5 and nn 0.3; this law is linear
    g, n = cfg.ridge, rec["best"]["n_fit_rows"]
    # the grid is row-relative, so its absolute bounds depend on the fit row count
    assert g.alpha_rel_max * n * 10 ** -g.grid_decades <= rec["best"]["alpha"] <= g.alpha_rel_max * n
    assert rec["best"]["alpha_rel"] == pytest.approx(rec["best"]["alpha"] / n)
    # nothing in _fit_ridge scores the val split, so it must not claim to
    assert rec["best"]["val_r2"] is None and rec["best"]["gcv_neg_mse"] <= 0
    _assert_artifacts(qdir, _RIDGE_ARTIFACTS, "model.joblib")

    with open(qdir / "ridge_coefficients.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [*rows[0]] == ["rank", "name", "col_id", "coef_std", "coef_watts"]
    assert len(rows) == len(names)
    assert set(planted) <= {r["name"] for r in rows}
    # ranked by |coef_std| descending
    mags = [abs(float(r["coef_std"])) for r in rows]
    assert mags == sorted(mags, reverse=True)


@pytest.mark.slow
def test_ridge_fits_without_a_validation_split(synth, tmp_path):
    """Ridge is the ONE backend that runs at val_fraction 0 with HPO on.

    RidgeCV's generalized CV happens inside the training rows, so unlike
    _fit_tree/_fit_nn there is nothing to raise about. Pinned here so the guard
    those two carry is not "helpfully" copied into _fit_ridge later.
    """
    cfg, _ = synth
    cfg.split.val_fraction = 0.0
    build_db.build(cfg)
    outdir = tmp_path / "run"
    outdir.mkdir()
    run.cmd_feature_select(cfg, outdir)
    run.cmd_fit(cfg, outdir, outdir / "proxies.csv", qs=[-1],
                model_kinds=["ridge"], use_hpo=True)
    qdir = outdir / "ridge"
    assert load_json(qdir / "result.json")["val_r2"] is None
    expected = [f for f in _RIDGE_ARTIFACTS
                if f not in ("pred_vs_time_val.png", "residual_train_val_test.png")]
    _assert_artifacts(qdir, expected + ["residual_train_test.png"], "model.joblib")


def test_cmd_fit_rejects_an_unknown_model_kind(tmp_path):
    """A mistyped kind used to fall through to the nn fitter and be reported as
    that kind. It is now rejected, and rejected BEFORE load_split/Union -- which
    on the real dataset are minutes of I/O -- so nothing here needs a dataset."""
    with pytest.raises(ValueError, match="unknown model kind"):
        run.cmd_fit(Config(), tmp_path, tmp_path / "proxies.csv", qs=[-1],
                    model_kinds=["ridge", "bogus"], use_hpo=False)
