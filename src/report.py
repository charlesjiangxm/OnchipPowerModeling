"""Write the per-run report.md.

Includes the resolved config (summary), the sample/feature counts at each
pipeline stage, train/val/test metrics in original y units, the best HPO
trial's hyperparameters, and links to every produced figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    run_dir: Path,
    *,
    config_summary: dict[str, Any],
    counts: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    best_hp: dict[str, Any],
    figures: dict[str, str],
    extras: dict[str, str] | None = None,
) -> None:
    """Render report.md inside run_dir.

    ``figures`` maps a logical name (e.g. ``pred_vs_true_test``) to a path
    relative to ``run_dir`` (e.g. ``artifacts/pred_vs_true_test.png``).
    ``extras`` lets the caller drop in extra markdown sections keyed by title.
    """
    run_dir = Path(run_dir)
    lines: list[str] = []

    lines.append(f"# Run report — {config_summary.get('algorithm', '<model>')}")
    lines.append("")
    lines.append(f"- Config: `{config_summary.get('config_path', '?')}`")
    lines.append(f"- Output dir: `{run_dir}`")
    lines.append(f"- Algorithm: **{config_summary.get('algorithm', '?')}**")
    lines.append(f"- Feature selection: **{config_summary.get('fs_alg', 'none')}** "
                 f"(top_k={config_summary.get('top_k', '-')})")
    lines.append(f"- Seed: {config_summary.get('seed', '?')}")
    lines.append("")

    lines.append("## Dataset counts")
    lines.append("")
    lines.append("| Stage | train rows | val rows | test rows | features |")
    lines.append("|---|---:|---:|---:|---:|")
    for stage in ("loaded", "after_preprocess", "after_feature_selection"):
        if stage not in counts:
            continue
        c = counts[stage]
        lines.append(
            f"| {stage} | {c.get('train_rows', '-')} | {c.get('val_rows', '-')} "
            f"| {c.get('test_rows', '-')} | {c.get('features', '-')} |"
        )
    lines.append("")

    lines.append("## Metrics (in original y units)")
    lines.append("")
    lines.append("| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "val", "test"):
        if split not in metrics:
            continue
        m = metrics[split]
        lines.append(
            f"| {split} | {m['smape']:.3f} | {m['mape']:.3f} | "
            f"{m['rmse']:.5f} | {m['mae']:.5f} | {m['r2']:.4f} |"
        )
    lines.append("")

    if best_hp:
        lines.append("## Best HPO trial")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for k, v in best_hp.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for name, rel in figures.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"![{name}]({rel})")
            lines.append("")

    if extras:
        for title, body in extras.items():
            lines.append(f"## {title}")
            lines.append("")
            lines.append(body)
            lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines))
