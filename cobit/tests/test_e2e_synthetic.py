"""End-to-end learning check: the pipeline must recover a known power law.

The synthetic DB's power is BASE + w1*bus_a[3] + w2*x_m1/cnt[5] + w3*wide[100]
+ w4*(wide[100] AND x_m2/w65[64]) + tiny noise. With 2 training benchmarks of
600 cycles the pipeline must select the signal bits and reach a low test MAPE
(the interaction term is exactly what the boosting trees exist to capture).
"""

from pathlib import Path

import numpy as np
import pytest

from cobit.pipeline import run_pipeline
from cobit.tests.conftest import INTERACTION, TRUE_BITS


@pytest.mark.slow
def test_pipeline_learns_synthetic_power(synthetic_cfg, tmp_path):
    cfg, _ = synthetic_cfg
    cfg.selection.target_qs = [6]
    cfg.hpo.r_rgs = [30]
    cfg.hpo.n_trials = 16
    cfg.hpo.population_size = 8
    cfg.hpo.t_th = 400
    cfg.hpo.run_pair_comparison = False
    cfg.eval.peak_window = 100
    cfg.eval.multicycle_windows = [8, 16]

    run_dir = Path(cfg.runtime.output_dir) / "e2e"
    result = run_pipeline(cfg, run_dir)
    rec = result["records"][0]

    # Stage 1 must find the true signal bits among its proxies
    from cobit.utils import load_json

    proxies = load_json(run_dir / "proxies.json")["proxies"]["6"]
    signal = set(TRUE_BITS) | set(INTERACTION[0])
    assert signal <= set(proxies["names"]), proxies["names"]

    # Stage 2 must actually learn (labels span ~5-10% around the base power)
    assert rec["test_mape"] < 3.0, rec["test_mape"]
    assert rec["test_r2"] > 0.8, rec["test_r2"]
    assert (run_dir / "Q6" / "model.json").exists()
    assert (run_dir / "Q6" / "trace_test.png").exists()
    assert rec["report"]["multicycle"]["8"]["mape"] < 3.0
