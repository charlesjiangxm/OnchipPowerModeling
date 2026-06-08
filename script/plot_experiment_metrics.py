#!/usr/bin/env python
"""Plot per-experiment R2 and RMSE curves from completed run reports.

An experiment is one module/proxy-kind pair, for example ``idu_input``.
Within each experiment, each feature-selection/model combination is one setup,
for example ``pearson_gbdt`` or ``from_model_ft_transformer``.

Example:
python script/plot_experiment_metrics.py

"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
KINDS = ("input", "output", "internal")
FS_METHODS = (
    "from_model",
    "sequential",
    "univariate",
    "variance",
    "pearson",
    "deep",
    "mcp",
    "rfe",
    "none",
)
MODEL_SLUGS = ("ft_transformer", "elasticnet", "rulefit", "gbdt", "mlp")
SPLITS = ("train", "val", "test")
RUN_DIR_RE = re.compile(r"^(?P<stem>.+)_(?P<timestamp>\d{8}_\d{6})$")
CONFIG_RE = re.compile(r"^- Config:\s+`(?P<path>[^`]+)`\s*$")
ALGORITHM_RE = re.compile(r"^- Algorithm:\s+\*\*(?P<value>[^*]+)\*\*\s*$")
FS_RE = re.compile(r"^- Feature selection:\s+\*\*(?P<value>[^*]+)\*\*")


@dataclass(frozen=True)
class RunRecord:
    module: str
    kind: str
    setup: str
    fs_method: str
    model_slug: str
    algorithm: str
    timestamp: str
    run_dir: Path
    report_path: Path
    metrics: dict[str, dict[str, float]]

    @property
    def experiment(self) -> str:
        return f"{self.module}_{self.kind}"


def main() -> int:
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    plots_dir = resolve_repo_path(args.plots_dir)

    if not output_root.is_dir():
        raise SystemExit(f"output root not found: {output_root}")

    modules = normalize_filter(args.modules)
    kinds = normalize_filter(args.kinds)
    records, skipped = collect_records(
        output_root=output_root,
        modules=modules,
        kinds=kinds,
    )
    latest = keep_latest(records)

    if args.show_missing:
        print_missing_setups(latest, modules=modules, kinds=kinds)

    if skipped:
        print(f"skipped {len(skipped)} report(s)", file=sys.stderr)
        for message in skipped[: args.max_warnings]:
            print(f"  {message}", file=sys.stderr)
        remaining = len(skipped) - args.max_warnings
        if remaining > 0:
            print(f"  ... {remaining} more", file=sys.stderr)

    if not latest:
        raise SystemExit(f"no complete reports found under {output_root}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    written = plot_all(latest, plots_dir)
    print(f"found {len(records)} complete report(s)")
    print(f"kept {len(latest)} latest setup run(s)")
    print(f"wrote {len(written)} plot(s) to {plots_dir}")
    for path in written:
        print(path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan output/*/report.md and plot train/val/test R2 and RMSE "
            "curves for each module/proxy-kind experiment."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory containing timestamped run folders (default: output).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plot"),
        help="Directory for generated plots (default: plot).",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help="Only plot these modules, e.g. --modules cp0 idu.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=KINDS,
        default=None,
        help="Only plot these proxy kinds.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print missing feature-selection/model setups per experiment.",
    )
    parser.add_argument(
        "--max-warnings",
        type=positive_int,
        default=20,
        help="Maximum skipped-report warnings to print (default: 20).",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def normalize_filter(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.strip() for value in values if value.strip()}


def collect_records(
    *,
    output_root: Path,
    modules: set[str] | None,
    kinds: set[str] | None,
) -> tuple[list[RunRecord], list[str]]:
    records: list[RunRecord] = []
    skipped: list[str] = []
    for report_path in sorted(output_root.glob("*/report.md")):
        try:
            record = parse_report(report_path)
        except ValueError as exc:
            skipped.append(f"{display_path(report_path)}: {exc}")
            continue
        if modules is not None and record.module not in modules:
            continue
        if kinds is not None and record.kind not in kinds:
            continue
        records.append(record)
    return records, skipped


def parse_report(report_path: Path) -> RunRecord:
    run_dir = report_path.parent
    text = report_path.read_text(errors="replace")
    config_stem = parse_config_stem(text) or parse_run_dir_stem(run_dir)
    module, kind, fs_method, model_slug, setup = parse_config_name(config_stem)
    metrics = parse_metrics_table(text)
    if not all(split in metrics for split in SPLITS):
        missing = ", ".join(split for split in SPLITS if split not in metrics)
        raise ValueError(f"missing metrics for split(s): {missing}")

    return RunRecord(
        module=module,
        kind=kind,
        setup=setup,
        fs_method=fs_method,
        model_slug=model_slug,
        algorithm=parse_report_field(text, ALGORITHM_RE) or model_slug,
        timestamp=parse_run_timestamp(run_dir),
        run_dir=run_dir,
        report_path=report_path,
        metrics=metrics,
    )


def parse_config_stem(text: str) -> str | None:
    for line in text.splitlines():
        match = CONFIG_RE.match(line.strip())
        if match:
            return Path(match.group("path")).stem
    return None


def parse_report_field(text: str, pattern: re.Pattern[str]) -> str | None:
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group("value").strip()
    return None


def parse_run_dir_stem(run_dir: Path) -> str:
    match = RUN_DIR_RE.match(run_dir.name)
    if not match:
        raise ValueError("run directory does not end with _YYYYMMDD_HHMMSS")
    return match.group("stem")


def parse_run_timestamp(run_dir: Path) -> str:
    match = RUN_DIR_RE.match(run_dir.name)
    if match:
        return match.group("timestamp")
    return f"{int(run_dir.stat().st_mtime):014d}"


def parse_config_name(stem: str) -> tuple[str, str, str, str, str]:
    parts = stem.split("_")
    for idx, kind in enumerate(parts):
        if kind not in KINDS or idx == 0:
            continue
        module = "_".join(parts[:idx])
        tail = "_".join(parts[idx + 1 :])
        parsed_tail = parse_setup_tail(tail)
        if parsed_tail is None:
            continue
        fs_method, model_slug = parsed_tail
        return module, kind, fs_method, model_slug, f"{fs_method}_{model_slug}"
    raise ValueError(f"could not parse config stem: {stem}")


def parse_setup_tail(tail: str) -> tuple[str, str] | None:
    for fs_method in FS_METHODS:
        prefix = f"{fs_method}_"
        if not tail.startswith(prefix):
            continue
        model_slug = tail[len(prefix) :]
        if model_slug in MODEL_SLUGS:
            return fs_method, model_slug
    return None


def parse_metrics_table(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    in_metrics = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "## Metrics (in original y units)":
            in_metrics = True
            continue
        if in_metrics and line.startswith("## "):
            break
        if not in_metrics or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in SPLITS:
            continue
        split = cells[0]
        try:
            metrics[split] = {
                "smape": float(cells[1]),
                "mape": float(cells[2]),
                "rmse": float(cells[3]),
                "mae": float(cells[4]),
                "r2": float(cells[5]),
            }
        except ValueError as exc:
            raise ValueError(f"invalid numeric metrics in {split} row") from exc
    if not metrics:
        raise ValueError("metrics table not found")
    return metrics


def keep_latest(records: Iterable[RunRecord]) -> list[RunRecord]:
    by_setup: dict[tuple[str, str, str], RunRecord] = {}
    for record in records:
        key = (record.module, record.kind, record.setup)
        existing = by_setup.get(key)
        if existing is None or latest_key(record) > latest_key(existing):
            by_setup[key] = record
    return sorted(
        by_setup.values(),
        key=lambda record: (record.module, record.kind, record.setup),
    )


def latest_key(record: RunRecord) -> tuple[str, str]:
    return record.timestamp, record.run_dir.name


def plot_all(records: list[RunRecord], plots_dir: Path) -> list[Path]:
    by_experiment: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        by_experiment.setdefault((record.module, record.kind), []).append(record)

    written: list[Path] = []
    for (module, kind), experiment_records in sorted(by_experiment.items()):
        ordered = sorted(
            experiment_records,
            key=lambda record: (
                safe_metric(record, "test", "r2"),
                safe_metric(record, "val", "r2"),
                record.setup,
            ),
            reverse=True,
        )
        experiment = f"{module}_{kind}"
        written.append(
            plot_metric(
                ordered,
                metric="r2",
                ylabel="R2",
                title=f"{experiment}: train/val/test R2",
                out_path=plots_dir / "r2" / f"{experiment}_r2.png",
            )
        )
        written.append(
            plot_metric(
                ordered,
                metric="rmse",
                ylabel="RMSE",
                title=f"{experiment}: train/val/test RMSE",
                out_path=plots_dir / "rmse" / f"{experiment}_rmse.png",
            )
        )
    return written


def safe_metric(record: RunRecord, split: str, metric: str) -> float:
    value = record.metrics.get(split, {}).get(metric, math.nan)
    return value if math.isfinite(value) else -math.inf


def plot_metric(
    records: list[RunRecord],
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> Path:
    labels = [record.setup for record in records]
    x_values = list(range(len(records)))
    fig_width = max(9.0, 0.45 * len(records) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    for split in SPLITS:
        y_values = [record.metrics[split][metric] for record in records]
        ax.scatter(x_values, y_values, s=18, label=split)
        if split == "test":
            for x_value, y_value in zip(x_values, y_values):
                ax.annotate(
                    format_metric_label(metric, y_value),
                    xy=(x_value, y_value),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    annotation_clip=False,
                )

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for tick_label, label in zip(ax.get_xticklabels(), labels):
        if "rulefit" in label:
            tick_label.set_color("red")
        elif "elasticnet" in label:
            tick_label.set_color("grey")
        elif "gbdt" in label:
            tick_label.set_color("mediumpurple")
    ax.set_xlabel("setup (sorted by test R2 descending)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def format_metric_label(metric: str, value: float) -> str:
    return f"{value:.3g}"


def print_missing_setups(
    records: list[RunRecord],
    *,
    modules: set[str] | None,
    kinds: set[str] | None,
) -> None:
    expected_setups = {f"{fs}_{model}" for fs in FS_METHODS for model in MODEL_SLUGS}
    expected_setups = {setup for setup in expected_setups if not setup.startswith("none_")}
    seen_by_experiment: dict[str, set[str]] = {}
    for record in records:
        seen_by_experiment.setdefault(record.experiment, set()).add(record.setup)

    experiments = sorted(seen_by_experiment)
    if modules is not None or kinds is not None:
        experiments = [
            exp for exp in experiments
            if (modules is None or exp.rsplit("_", 1)[0] in modules)
            and (kinds is None or exp.rsplit("_", 1)[1] in kinds)
        ]

    for experiment in experiments:
        missing = sorted(expected_setups - seen_by_experiment[experiment])
        if missing:
            print(f"{experiment}: missing {len(missing)} setup(s): {', '.join(missing)}")
        else:
            print(f"{experiment}: all expected setups present")


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
