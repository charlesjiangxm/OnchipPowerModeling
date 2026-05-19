"""Plotting helpers consolidated from the old per-model entry scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def _shorten(name: str, max_len: int = 70) -> str:
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-(max_len - 11):]


def plot_pred_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title(f"{title}\nR^2 = {r2_score(y_true, y_pred):.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_top_features(
    feat_scores: pd.Series, out_path: Path,
    *, top_n: int = 20, title: str = "Top features",
    xlabel: str = "Importance",
) -> pd.Series:
    """Bar chart of top-N features. Returns the subseries actually drawn."""
    df = feat_scores.sort_values(ascending=False).head(top_n)
    if df.empty:
        return df
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(df) + 1))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df.values, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(str(n), 80) for n in df.index], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return df


def plot_interaction_heatmap(
    matrix: pd.DataFrame, out_path: Path, *, title: str = "Feature interactions",
) -> None:
    K = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * K + 2), max(5, 0.4 * K + 2)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    labels = [_shorten(str(n), 40) for n in matrix.index]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_convergence(history: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train", color="#1f77b4")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val", color="#d62728")
    best = history.get("best_epoch")
    if best is not None:
        ax.axvline(best, color="gray", linestyle="--", lw=1,
                   label=f"best iter={best}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training convergence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
