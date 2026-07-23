#!/usr/bin/env python
"""Plot average module power with the observed min-max range."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("db/pwr/coremark_pwr.pkl"),
        help="Pickled power DataFrame (default: db/pwr/coremark_pwr.pkl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plot/coremark_average_power.png"),
        help="Output image path (default: plot/coremark_average_power.png).",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def module_name(column: object) -> str:
    name = str(column)
    match = re.search(r"Pc\(([^()]*)\)$", name)
    return match.group(1) if match else name.rsplit("/", 1)[-1]


def plot_average_power(input_path: Path, output_path: Path) -> None:
    power_w = pd.read_pickle(input_path)
    if not isinstance(power_w, pd.DataFrame) or power_w.empty:
        raise ValueError(f"{input_path} does not contain a non-empty DataFrame")

    non_numeric = power_w.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"non-numeric power columns: {', '.join(map(str, non_numeric))}")

    power_mw = power_w * 1_000.0
    average = power_mw.mean()
    minimum = power_mw.min()
    maximum = power_mw.max()
    error = np.vstack(
        [
            (average - minimum).clip(lower=0).to_numpy(),
            (maximum - average).clip(lower=0).to_numpy(),
        ]
    )

    x = np.arange(len(average))
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(
        x,
        average.to_numpy(),
        yerr=error,
        width=0.75,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 3},
    )

    for bar, value in zip(bars, average):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
        )

    ax.set_ylabel("Power (mW)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [module_name(column) for column in average.index],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    lower_limit, upper_limit = ax.get_ylim()
    ax.set_ylim(lower_limit / 2, upper_limit)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    plot_average_power(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
