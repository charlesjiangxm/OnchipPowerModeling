#!/usr/bin/env python
"""Argparse-free debug entry point for the YAML-driven fit pipeline.

This is the debugger-friendly twin of ``run_fit.py``: it hardcodes the config
instead of parsing CLI flags, so you can launch it straight from an IDE
(PyCharm: right-click -> Debug 'debug_fit') with no run-configuration args.

To step through training, set a breakpoint on the ``run(...)`` call below and
"step into" it, or drop breakpoints directly in ``src/pipeline.py`` (the stage
boundaries there -- load, split, preprocess, feature-select, HPO, refit,
metrics, artifacts -- are the natural inspection points).

Edit ``CONFIG`` / ``OUTPUT_ROOT`` to point at a different run. The config path
is made absolute and the repo root is put on ``sys.path``, so this works
regardless of the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- what to train (edit these) -------------------------------------------
CONFIG = REPO_ROOT / "configs" / "idu_input_from_model_ft_transformer.yaml"
OUTPUT_ROOT: Path | None = None  # None -> <repo_root>/output/<config_stem>_<ts>/
# --------------------------------------------------------------------------


def main() -> int:
    from src.pipeline import run

    return run(CONFIG, output_root=OUTPUT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
