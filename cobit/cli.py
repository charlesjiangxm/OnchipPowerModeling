"""Command-line interface: ``python -m cobit <cmd> --config <yaml> [k.ey=v ...]``.

Commands
--------
build-dataset   build/refresh the sparse feature cache from the pkl DB
select-proxies  Stage 1 only (LR-MCP proxy selection)
hpo-pairs       Algorithm 2 sampler-pruner pair comparison only
run             full Algorithm 1 pipeline (implies build + select)
run-all         alias of ``run`` (pair comparison included if configured)
inspect         print DB coverage, cache density, and artifact status
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from .config import CobitConfig
from .utils import log, setup_logging


def _run_dir(cfg: CobitConfig) -> Path:
    name = cfg.runtime.run_name
    if not name:
        stem = Path(cfg.config_path).stem or "run"
        name = f"{stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    return Path(cfg.runtime.output_dir) / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cobit", description=__doc__)
    parser.add_argument(
        "command",
        choices=["build-dataset", "select-proxies", "hpo-pairs", "run", "run-all", "inspect"],
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--force", action="store_true", help="rebuild stale/complete artifacts")
    parser.add_argument(
        "--run-name", default=None,
        help="reuse/resume a named run directory instead of a timestamped one",
    )
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="dotted config overrides, e.g. hpo.n_trials=16",
    )
    args = parser.parse_args(argv)

    setup_logging()
    cfg = CobitConfig.from_yaml(args.config, overrides=args.overrides)
    if args.run_name:
        cfg.runtime.run_name = args.run_name

    if args.command == "build-dataset":
        from .data.build import build_dataset

        build_dataset(cfg, force=args.force)
        return 0

    if args.command == "inspect":
        _inspect(cfg)
        return 0

    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run directory: %s", run_dir)

    if args.command == "select-proxies":
        from .data.build import build_dataset
        from .pipeline import stage1_proxies

        build_dataset(cfg, force=False)
        stage1_proxies(cfg, run_dir, force=args.force)
        return 0

    if args.command == "hpo-pairs":
        import xgboost as xgb

        from .data.build import build_dataset
        from .data.dataset import DatasetCache
        from .hpo import compare_sampler_pruner_pairs, study_stamp
        from .pipeline import stage1_proxies

        build_dataset(cfg, force=False)
        proxies = stage1_proxies(cfg, run_dir, force=False)
        q = cfg.hpo.pair_q or sorted(proxies)[0]
        if q not in proxies:
            raise SystemExit(f"hpo.pair_q={q} not in selection.target_qs")
        cache = DatasetCache(cfg)
        bundle = cache.load_split(proxies[q].col_ids, dense=True)
        if bundle.y_val.size == 0:
            raise SystemExit("HPO needs a validation split: set split.val_fraction > 0")
        dtrain = xgb.DMatrix(bundle.X_train, label=bundle.y_train)
        dval = xgb.DMatrix(bundle.X_val, label=bundle.y_val)
        compare_sampler_pruner_pairs(
            cfg, run_dir, dtrain, dval, bundle.y_val,
            stamp=study_stamp(cfg, proxies[q].col_ids),
        )
        return 0

    if args.command in ("run", "run-all"):
        from .pipeline import run_pipeline

        run_pipeline(cfg, run_dir, force=args.force)
        return 0

    raise AssertionError(args.command)


def _inspect(cfg: CobitConfig) -> None:
    import numpy as np

    from .data.discovery import discover
    from .utils import load_json

    layout = discover(cfg.data.db_root)
    print(f"DB root: {layout.db_root}")
    print(f"scopes ({len(layout.scopes)}): {' '.join(layout.scopes)}")
    print(f"benchmarks ({len(layout.benchmarks)}):")
    for b in layout.benchmarks:
        miss = layout.missing_scopes(b)
        print(f"  {b:<15} {'COMPLETE' if not miss else 'missing: ' + ', '.join(miss)}")

    cache = Path(cfg.data.cache_dir)
    if not (cache / "nets.json").exists():
        print("cache: not built (run build-dataset)")
        return
    reg = load_json(cache / "nets.json")
    be = reg.get("bit_expand", True)
    feat_label = "bit features" if be else "net features"
    print(f"cache: {cache} | {len(reg['nets'])} nets, {reg['n_features']} {feat_label}")
    for b in layout.benchmarks:
        mpath = cache / "features" / b / "manifest.json"
        if not mpath.exists():
            print(f"  {b:<15} NOT BUILT")
            continue
        m = load_json(mpath)
        nnz = sum(m["scope_nnz"].values())
        dens = 100.0 * nnz / max(1, m["n_rows"] * reg["n_features"])
        print(f"  {b:<15} rows={m['n_rows']:<9} nnz={nnz:<12} density={dens:.4f}%")
    stats = cache / "bit_stats.npz"
    if stats.exists():
        with np.load(stats) as z:
            total = None
            for b in z.files:
                total = z[b].astype(np.uint64) if total is None else total + z[b]
        toggling_label = "bits" if be else "nets"
        print(f"{toggling_label} toggling anywhere: {(total > 0).sum()} / {total.size}")


if __name__ == "__main__":
    sys.exit(main())
