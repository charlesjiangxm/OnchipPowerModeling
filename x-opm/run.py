"""x-opm dataset build CLI (steps 1-5).

Usage (interpreter must be ~/anaconda3/bin/python):
    python x-opm/run.py all       --config x-opm/configs/cp0.yaml [--force]
    python x-opm/run.py fit       --config x-opm/configs/cp0.yaml
    python x-opm/run.py transform --config x-opm/configs/cp0.yaml [--force] [--cases a,b]
    python x-opm/run.py report    --config x-opm/configs/cp0.yaml

Stages:
    fit       -> out/x-opm/manifest.json               (train-only decisions)
    transform -> out/x-opm/dataset/{trainset,testset}/<case>/type{A,B,C,D}.pkl + target.pkl
    report    -> out/x-opm/reports/feature_report.csv + summary.json
    all       -> fit, transform (all cases), report
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from apply import TYPES, apply_case
from config import XopmConfig
from fit import run_fit
from io_utils import read_json
from report import ScaledStats, build_report


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_manifest(cfg: XopmConfig) -> dict:
    if not os.path.exists(cfg.manifest_path):
        raise SystemExit(f"manifest not found: {cfg.manifest_path} (run 'fit' first)")
    return read_json(cfg.manifest_path)


def _transform(cfg: XopmConfig, manifest: dict, cases: list[str], force: bool) -> None:
    # process the small cases first, giant cases last (independent + resumable).
    order = sorted(cases, key=lambda c: os.path.getsize(cfg.func_pkl(c)))
    for case in order:
        apply_case(case, manifest, cfg, force=force)


def _report_from_disk(cfg: XopmConfig, manifest: dict) -> None:
    """Recompute scaled feature stats from the written TRAIN pkls, then report."""
    scaled = ScaledStats()
    for case in cfg.train_benchmarks:
        out_dir = cfg.case_out_dir(case)
        for t in TYPES:
            p = os.path.join(out_dir, f"type{t}.pkl")
            if not os.path.exists(p):
                continue
            df = pd.read_pickle(p)
            for name in df.columns:
                scaled.update(name, df[name].to_numpy())
            del df
    build_report(cfg, manifest, scaled)


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(prog="x-opm")
    ap.add_argument("command", choices=["fit", "transform", "report", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cases", default=None, help="comma-separated subset for transform")
    args = ap.parse_args(argv)

    cfg = XopmConfig.from_yaml(args.config)
    log = logging.getLogger("x-opm")
    log.info("command=%s scope=%s train=%d test=%d", args.command, cfg.scope,
             len(cfg.train_benchmarks), len(cfg.test_benchmarks))

    if args.command == "fit":
        run_fit(cfg, force=args.force)
    elif args.command == "transform":
        manifest = _load_manifest(cfg)
        cases = args.cases.split(",") if args.cases else cfg.cases
        _transform(cfg, manifest, cases, args.force)
    elif args.command == "report":
        _report_from_disk(cfg, _load_manifest(cfg))
    elif args.command == "all":
        manifest = run_fit(cfg, force=args.force)
        _transform(cfg, manifest, cfg.cases, args.force)
        _report_from_disk(cfg, manifest)
    log.info("done: %s", args.command)


if __name__ == "__main__":
    main()
