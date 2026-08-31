"""Reporting: JSON metrics, matplotlib plots, markdown report."""

import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def _ensure_dirs(output_dir: Path):
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)


def save_raw(output_dir: Path, method1_result=None, method2_result=None, qt=None):
    raw_dir = output_dir / "raw"
    if method1_result is not None:
        np.save(raw_dir / "d_te.npy", method1_result.d_te)
        np.save(raw_dir / "d_tr.npy", method1_result.d_tr)
        np.save(raw_dir / "I_te.npy", method1_result.I_te)
    if method2_result is not None and method2_result.stumpy_profile is not None:
        np.save(raw_dir / "stumpy_profile.npy", method2_result.stumpy_profile)
    if qt is not None:
        import pickle
        with open(raw_dir / "qt_model.pkl", "wb") as f:
            pickle.dump(qt, f)


def plot_method1(output_dir: Path, result):
    plots_dir = output_dir / "plots"

    # Distance histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    p99 = np.percentile(np.concatenate([result.d_te, result.d_tr]), 99)
    ax.hist(result.d_te, bins=100, alpha=0.6, label="test→train", density=True, range=(0, p99))
    ax.hist(result.d_tr, bins=100, alpha=0.6, label="train→train", density=True, range=(0, p99))
    ax.axvline(result.epsilon, color="red", linestyle="--", label=f"ε (1st pct)={result.epsilon:.4f}")
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density")
    ax.set_title("Method 1: Nearest-Neighbor Distance Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "method1_distance_hist.png", dpi=150)
    plt.close(fig)

    # CDF
    fig, ax = plt.subplots(figsize=(10, 5))
    d_te_sorted = np.sort(result.d_te)
    d_tr_sorted = np.sort(result.d_tr)
    ax.plot(d_te_sorted, np.linspace(0, 1, len(d_te_sorted)), label="test→train")
    ax.plot(d_tr_sorted, np.linspace(0, 1, len(d_tr_sorted)), label="train→train")
    ax.axvline(result.epsilon, color="red", linestyle="--", label=f"ε={result.epsilon:.4f}")
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("CDF")
    ax.set_title("Method 1: CDF Comparison (KS visualization)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "method1_distance_cdf.png", dpi=150)
    plt.close(fig)

    # Top matches
    fig, ax = plt.subplots(figsize=(12, 4))
    top_d = [m["distance"] for m in result.top_matches[:20]]
    top_pos = [m["test_pos"] for m in result.top_matches[:20]]
    ax.barh(range(len(top_d)), top_d)
    ax.set_yticks(range(len(top_d)))
    ax.set_yticklabels([str(p) for p in top_pos], fontsize=7)
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Test Position")
    ax.set_title("Method 1: 20 Closest Test Samples (to nearest train)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plots_dir / "method1_top_matches.png", dpi=150)
    plt.close(fig)


def plot_method2(output_dir: Path, result):
    plots_dir = output_dir / "plots"

    if result.jaccard_values:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(result.jaccard_values, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_xlabel("Jaccard Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Method 2: Jaccard Distribution of Matched Segments")
        fig.tight_layout()
        fig.savefig(plots_dir / "method2_jaccard_dist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        test_pos = [m["test_pos"] for m in result.matches_detail[:200]]
        train_pos = [m["train_pos"] for m in result.matches_detail[:200]]
        jac = [m["jaccard"] for m in result.matches_detail[:200]]
        sc = ax.scatter(test_pos, train_pos, c=jac, cmap="viridis", alpha=0.6, s=10)
        plt.colorbar(sc, label="Jaccard")
        ax.set_xlabel("Test Window Position")
        ax.set_ylabel("Train Window Position")
        ax.set_title("Method 2: Matched Segments (test vs train position)")
        fig.tight_layout()
        fig.savefig(plots_dir / "method2_matches_timeline.png", dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No matches found", ha="center", va="center", fontsize=16)
        ax.set_title("Method 2: Jaccard Distribution (no matches)")
        fig.tight_layout()
        fig.savefig(plots_dir / "method2_jaccard_dist.png", dpi=150)
        plt.close(fig)

    if result.stumpy_profile is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        finite = result.stumpy_profile[np.isfinite(result.stumpy_profile)]
        ax.hist(finite, bins=100, alpha=0.7, color="coral", edgecolor="black")
        ax.set_xlabel("Matrix Profile Distance (z-normalized)")
        ax.set_ylabel("Count")
        ax.set_title("Method 2 (stumpy): Matrix Profile Distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "method2_stumpy_profile.png", dpi=150)
        plt.close(fig)


def plot_method3(output_dir: Path, result):
    plots_dir = output_dir / "plots"

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result.fpr, result.tpr, label=f"AUC = {result.auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Method 3: Adversarial Validation ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "method3_roc.png", dpi=150)
    plt.close(fig)

    # Feature importance
    if result.top_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [f["name"][:30] for f in result.top_features[:20]]
        imps = [f["importance"] for f in result.top_features[:20]]
        ax.barh(range(len(imps)), imps, color="steelblue")
        ax.set_yticks(range(len(imps)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Method 3: Top-20 Feature Importances")
        fig.tight_layout()
        fig.savefig(plots_dir / "method3_importance.png", dpi=150)
        plt.close(fig)


def build_metrics_json(metadata, m1, m2, m3):
    metrics = {"metadata": metadata, "interpretation": {}}

    if m1 is not None:
        metrics["method1_ann"] = {
            "contamination_rate": m1.contamination_rate,
            "epsilon_1st_percentile": m1.epsilon,
            "distance_ratio_median": m1.distance_ratio,
            "ks_statistic": m1.ks_statistic,
            "ks_pvalue": m1.ks_pvalue,
            "d_te_median": m1.d_te_median,
            "d_te_mean": m1.d_te_mean,
            "d_tr_median": m1.d_tr_median,
            "d_tr_mean": m1.d_tr_mean,
            "n_suspicious_below_epsilon": m1.n_suspicious,
            "top_matches": m1.top_matches,
        }

    if m2 is not None:
        m2_data = {
            "n_train_windows": m2.n_train_windows,
            "n_test_windows": m2.n_test_windows,
            "n_matches": m2.n_matches,
            "match_rate": m2.match_rate,
            "jaccard_median": float(np.median(m2.jaccard_values)) if m2.jaccard_values else None,
            "jaccard_max": float(np.max(m2.jaccard_values)) if m2.jaccard_values else None,
            "matches_detail": m2.matches_detail[:20],
        }
        if m2.stumpy_stats is not None:
            m2_data["stumpy"] = m2.stumpy_stats
        metrics["method2_segment"] = m2_data

    if m3 is not None:
        metrics["method3_adversarial"] = {
            "auc": m3.auc,
            "n_estimators_used": m3.n_estimators_used,
            "top_features": m3.top_features[:20],
        }

    # Combined interpretation
    verdict = "healthy"
    reasons = []
    if m1 is not None:
        if m1.distance_ratio < 0.95:
            verdict = "suspected_contamination"
            reasons.append(f"distance_ratio={m1.distance_ratio:.4f} < 0.95 (test closer to train than train internal)")
        if m1.contamination_rate > 0.05:
            verdict = "suspected_contamination"
            reasons.append(f"contamination_rate={m1.contamination_rate:.4f} > 5%")
    if m3 is not None and m1 is not None:
        if m3.auc < 0.55 and m1.distance_ratio < 1.0:
            verdict = "suspected_contamination"
            reasons.append(f"AUC={m3.auc:.4f}≈0.5 + small NN distances = near-duplication signature")
        elif m3.auc > 0.7:
            if verdict != "suspected_contamination":
                verdict = "distribution_drift"
            reasons.append(f"AUC={m3.auc:.4f} > 0.7 = distribution drift (different benchmarks)")
    if not reasons:
        reasons.append("No contamination signals detected; distance ratio ≈1, low contamination rate")
    metrics["interpretation"] = {"verdict": verdict, "reasoning": "; ".join(reasons)}

    return metrics


def generate_report(output_dir: Path, metadata, m1, m2, m3):
    _ensure_dirs(output_dir)

    # Save raw arrays
    save_raw(output_dir, m1, m2, metadata.get("_qt"))
    metadata.pop("_qt", None)

    # Plots
    if m1 is not None:
        plot_method1(output_dir, m1)
    if m2 is not None:
        plot_method2(output_dir, m2)
    if m3 is not None:
        plot_method3(output_dir, m3)

    # JSON metrics
    metrics = build_metrics_json(metadata, m1, m2, m3)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Markdown report
    lines = ["# Data Contamination Detection Report", ""]
    lines.append(f"**Timestamp**: {metadata.get('timestamp', 'N/A')}  ")
    lines.append(f"**Train samples**: {metadata.get('n_train', '?')}  ")
    lines.append(f"**Test samples**: {metadata.get('n_test', '?')}  ")
    lines.append(f"**Features (raw)**: {metadata.get('n_features_raw', '?')}  ")
    lines.append(f"**Features (after drop)**: {metadata.get('n_features_after_drop', '?')}  ")
    lines.append("")

    if m1 is not None:
        lines.append("## Method 1: Sample-Level Neighbors (ANN)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Contamination rate | {m1.contamination_rate:.6f} |")
        lines.append(f"| Distance ratio (median) | {m1.distance_ratio:.4f} |")
        lines.append(f"| KS statistic | {m1.ks_statistic:.6f} |")
        lines.append(f"| KS p-value | {m1.ks_pvalue:.2e} |")
        lines.append(f"| ε (1st pct of train→train) | {m1.epsilon:.6f} |")
        lines.append(f"| Suspicious samples (<ε) | {m1.n_suspicious} |")
        lines.append("")
        lines.append(f"![Distance Histogram](plots/method1_distance_hist.png)")
        lines.append(f"![CDF](plots/method1_distance_cdf.png)")
        lines.append("")

    if m2 is not None:
        lines.append("## Method 2: Segment-Level (MinHash LSH)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Train windows | {m2.n_train_windows} |")
        lines.append(f"| Test windows | {m2.n_test_windows} |")
        lines.append(f"| Matches (Jaccard ≥ {0.5}) | {m2.n_matches} |")
        lines.append(f"| Match rate | {m2.match_rate:.6f} |")
        jac_med = float(np.median(m2.jaccard_values)) if m2.jaccard_values else None
        jac_max = float(np.max(m2.jaccard_values)) if m2.jaccard_values else None
        lines.append(f"| Jaccard median | {jac_med} |")
        lines.append(f"| Jaccard max | {jac_max} |")
        lines.append("")
        if m2.stumpy_stats:
            lines.append("### Stumpy Matrix Profile")
            lines.append(f"- m={m2.stumpy_stats['m']}, subsequences={m2.stumpy_stats['n_test_subsequences']}")
            lines.append(f"- median={m2.stumpy_stats.get('median')}, min={m2.stumpy_stats.get('min')}")
            lines.append(f"![Stumpy](plots/method2_stumpy_profile.png)")
            lines.append("")
        lines.append(f"![Jaccard](plots/method2_jaccard_dist.png)")
        lines.append(f"![Timeline](plots/method2_matches_timeline.png)")
        lines.append("")

    if m3 is not None:
        lines.append("## Method 3: Adversarial Validation")
        lines.append("")
        lines.append(f"**AUC = {m3.auc:.4f}**")
        lines.append("")
        lines.append("- AUC ≈ 0.5: distributions indistinguishable (combined with small NN distances → near-duplication)")
        lines.append("- AUC > 0.7: distribution drift (different benchmarks)")
        lines.append("")
        if m3.top_features:
            lines.append("### Top-10 Features")
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for f in m3.top_features[:10]:
                lines.append(f"| {f['name']} | {f['importance']} |")
            lines.append("")
        lines.append(f"![ROC](plots/method3_roc.png)")
        lines.append(f"![Importance](plots/method3_importance.png)")
        lines.append("")

    # Verdict
    interp = metrics.get("interpretation", {})
    lines.append("## Combined Verdict")
    lines.append(f"**{interp.get('verdict', 'unknown').upper()}**")
    lines.append("")
    lines.append(interp.get("reasoning", ""))
    lines.append("")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_dir / 'report.md'}")
    logger.info(f"Verdict: {interp.get('verdict', 'unknown')}")
