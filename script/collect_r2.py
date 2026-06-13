"""Aggregate R² (train/val/test) from every run under output/ and plot."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIMESTAMP_RE = re.compile(r"^(?P<config>.+)_(?P<ts>\d{8}_\d{6})$")
METRICS_HEADING = "## Metrics"
# Row format: | split | sMAPE | MAPE | RMSE | MAE | R^2 |
ROW_RE = re.compile(
    r"^\|\s*(train|val|test)\s*\|"
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"  # sMAPE
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"  # MAPE
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"  # RMSE
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"  # MAE
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*$"  # R^2
)


def parse_report(report_path: Path) -> dict[str, dict[str, float]] | None:
    """Return {split: {'rmse': ..., 'r2': ...}} or None if unparseable."""
    text = report_path.read_text(encoding="utf-8", errors="replace")
    idx = text.find(METRICS_HEADING)
    if idx == -1:
        return None
    section = text[idx:]
    next_h = section.find("\n## ", 1)
    if next_h != -1:
        section = section[:next_h]
    found: dict[str, dict[str, float]] = {}
    for line in section.splitlines():
        m = ROW_RE.match(line)
        if m:
            found[m.group(1)] = {"rmse": float(m.group(4)), "r2": float(m.group(6))}
    if {"train", "val", "test"}.issubset(found):
        return {k: found[k] for k in ("train", "val", "test")}
    return None


def collect(output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    skipped: list[str] = []
    scanned = 0
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        scanned += 1
        m = TIMESTAMP_RE.match(entry.name)
        if not m:
            skipped.append(f"{entry.name} (no timestamp suffix)")
            continue
        report = entry / "report.md"
        if not report.exists():
            skipped.append(f"{entry.name} (no report.md)")
            continue
        metrics = parse_report(report)
        if metrics is None:
            skipped.append(f"{entry.name} (unparseable metrics table)")
            continue
        rows.append({
            "config": m.group("config"),
            "timestamp": m.group("ts"),
            "train_r2":   metrics["train"]["r2"],
            "val_r2":     metrics["val"]["r2"],
            "test_r2":    metrics["test"]["r2"],
            "train_rmse": metrics["train"]["rmse"],
            "val_rmse":   metrics["val"]["rmse"],
            "test_rmse":  metrics["test"]["rmse"],
        })

    print(f"scanned : {scanned} directories")
    print(f"parsed  : {len(rows)}")
    if skipped:
        print(f"skipped : {len(skipped)}")
        for s in skipped:
            print(f"  - {s}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["config", "timestamp"]).drop_duplicates("config", keep="last")
    print(f"unique configs after dedupe: {len(df)}")
    df = df.sort_values("test_r2", ascending=True).reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, out_png: Path,
         metric: str = "r2",
         title: str | None = None,
         log_y: bool = False) -> None:
    """metric: 'r2' or 'rmse'. df must already be sorted in display order."""
    n = len(df)
    width = max(12.0, 0.08 * n)
    fig, ax = plt.subplots(figsize=(width, 6.0))
    x = range(n)
    label_map = {"r2": "R²", "rmse": "RMSE"}
    ylabel = label_map[metric]
    ax.plot(x, df[f"train_{metric}"], label="train", marker=".", linewidth=1.0, alpha=0.85)
    ax.plot(x, df[f"val_{metric}"],   label="val",   marker=".", linewidth=1.0, alpha=0.85)
    ax.plot(x, df[f"test_{metric}"],  label="test",  marker=".", linewidth=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["config"].tolist(), rotation=90, fontsize=5)
    ax.set_ylabel(ylabel + (" (log)" if log_y else ""))
    ax.set_xlabel("test case")
    if title is None:
        title = f"{ylabel} across runs (sorted by test {ylabel} ascending)"
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    ax.legend(loc="lower right" if metric == "r2" else "upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    output_dir = repo / "output"
    if not output_dir.is_dir():
        print(f"output directory not found: {output_dir}", file=sys.stderr)
        return 1

    df = collect(output_dir)
    if df.empty:
        print("no runs parsed; nothing to plot", file=sys.stderr)
        return 1

    # df arrives sorted by test_r2 ascending from collect()
    csv_path     = output_dir / "r2_summary.csv"
    png_path_r2  = output_dir / "r2_summary.png"
    png_path_hi  = output_dir / "r2_summary_gt0.8.png"
    md_path_r2   = output_dir / "r2_ranking.md"
    png_path_rmse = output_dir / "rmse_summary.png"
    md_path_rmse  = output_dir / "rmse_ranking.md"

    df.to_csv(csv_path, index=False)
    plot(df, png_path_r2, metric="r2")
    df_hi = df[df["test_r2"] > 0.8].reset_index(drop=True)
    plot(df_hi, png_path_hi, metric="r2",
         title="R² across runs (test R² > 0.8, sorted ascending)")
    print(f"high-R² subset: {len(df_hi)} runs")

    md_path_r2.write_text("\n".join([
        "# Experiments ranked by test R² (low → high)",
        "",
        *(f"- {name}" for name in df["config"].tolist()),
        "",
    ]), encoding="utf-8")

    df_rmse = df.sort_values("test_rmse", ascending=True).reset_index(drop=True)
    plot(df_rmse, png_path_rmse, metric="rmse",
         title="RMSE across runs (sorted by test RMSE ascending — left = best)",
         log_y=True)
    md_path_rmse.write_text("\n".join([
        "# Experiments ranked by test RMSE (low → high, best first)",
        "",
        *(f"- {name}" for name in df_rmse["config"].tolist()),
        "",
    ]), encoding="utf-8")

    for p in [csv_path, png_path_r2, png_path_hi, md_path_r2, png_path_rmse, md_path_rmse]:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
