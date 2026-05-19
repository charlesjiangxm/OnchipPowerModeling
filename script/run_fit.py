#!/usr/bin/env python
"""Runner for the YAML-driven regression-fitting library.

Usage::

    python script/run_fit.py --config configs/example.yaml [--output-root <path>]

Default behavior writes outputs under ``<repo_root>/output/<config_stem>_<ts>/``;
``--output-root`` overrides the parent directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a regression model from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Override the parent directory for run outputs "
             "(default: <repo_root>/output/).",
    )
    args = parser.parse_args()

    from src.pipeline import run
    return run(args.config, output_root=args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
