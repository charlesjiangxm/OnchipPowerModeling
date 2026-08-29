#!/usr/bin/env python
"""One grouped bar chart per aq_core module: per-benchmark Pearson vs Spearman
switching correlations, from corr_toggle_power.json.

Two flavors, each with x-axis = benchmarks, y-axis = correlation, two bars/benchmark:
  RAW    : |correlation| at each cell's best window (0-1)
           -> corr_bars_<module>.png + corr_bars_all.png
  HONEST : null-corrected (raw - null95) at the best headline target/window; can go
           slightly negative when switching does not beat chance
           -> corr_bars_honest_<module>.png + corr_bars_honest_all.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common

MODS = ["cp0", "idu", "ifu", "iu", "lsu", "rtu", "vidu", "vpu"]
BENCH = _common.BENCHMARKS
C_PEAR, C_SPEAR = "#1f77b4", "#d62728"


def load(m, honest=False):
    d = json.load(open(f"{_common.OUTROOT}/{m}_activity/corr_toggle_power.json"))
    pe, sp = {}, {}
    for e in d:
        s = e["switching"]
        if honest:   # null-corrected at the best headline target/window (matched per-W null)
            pt = s["per_target"][s["best_target"]]
            pe[e["case"]] = pt["pearson"] - pt["pearson_null95"]
            sp[e["case"]] = pt["spearman"] - pt["spearman_null95"]
        else:
            pe[e["case"]] = s["pearson_at_best"]
            sp[e["case"]] = s["headline_spearman"]
    cases = [b for b in BENCH if b in sp]
    return cases, [pe[c] for c in cases], [sp[c] for c in cases]


def draw(ax, m, honest=False):
    cases, pear, spear = load(m, honest)
    x = np.arange(len(cases)); w = 0.4
    ax.bar(x - w / 2, pear, w, label="Pearson", color=C_PEAR)
    ax.bar(x + w / 2, spear, w, label="Spearman", color=C_SPEAR)
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    if honest:
        ax.axhline(0.2, color="k", ls=":", lw=0.7); ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-0.2, 1.0); ax.set_ylabel("null-corrected correlation")
        kind = "honest (raw − null95)"
    else:
        ax.set_ylim(0, 1.0); ax.set_ylabel("|correlation|")
        kind = "best window"
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{m}  —  switching correlation vs power movement ({kind})")
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=8, loc="upper left")


def _emit(honest):
    tag = "corr_bars_honest" if honest else "corr_bars"
    for m in MODS:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        draw(ax, m, honest)
        fig.tight_layout()
        fp = f"{_common.OUTROOT}/{tag}_{m}.png"
        fig.savefig(fp, dpi=120); plt.close(fig)
        print("wrote", fp)

    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    for ax, m in zip(axes.ravel(), MODS):
        draw(ax, m, honest)
    fig.tight_layout()
    fp = f"{_common.OUTROOT}/{tag}_all.png"
    fig.savefig(fp, dpi=110); plt.close(fig)
    print("wrote", fp)


def main():
    _emit(honest=False)
    _emit(honest=True)


if __name__ == "__main__":
    main()
