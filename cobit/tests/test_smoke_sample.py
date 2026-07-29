"""CLI smoke test on the real 2-row sample DB (wiring, not accuracy)."""

import json
from pathlib import Path

import pytest

SAMPLE_DB = Path(__file__).resolve().parents[2] / "c906_db_net_1cyc_20260729_2rows"


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLE_DB.is_dir(), reason="sample DB not present")
def test_cli_run_on_sample(tmp_path):
    from cobit.cli import main

    config = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    rc = main(
        [
            "run",
            "--config", str(config),
            "--run-name", "pytest_smoke",
            f"data.cache_dir={tmp_path / 'cache'}",
            f"runtime.output_dir={tmp_path / 'runs'}",
        ]
    )
    assert rc == 0
    run_dir = tmp_path / "runs" / "pytest_smoke"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["records"], "no per-Q records produced"
    rec = metrics["records"][0]
    assert "test_mape" in rec and "report" in rec
    assert (run_dir / "proxies.json").exists()
    assert (run_dir / "pair_comparison.json").exists()  # smoke config enables it
    assert (run_dir / f"Q{rec['q']}" / "model.json").exists()
