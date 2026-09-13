#!/usr/bin/env python3
"""Residual (parity) maps for the Q=30 binary_fit runs, styled to match
``results/x-opm-aq-core.png``: 3 hexbin panels (train/val/test), log counts,
red y = x, R2 box top-left, power in mW.

Also writes the per-cycle source data in the same column layout as
``results/x-opm-aq-core.csv`` so the figures can be rebuilt later.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

RUN = "analysis/binary-fit/2026-09-04-18-30proxy-4cyc"
OUT = "results"
MODELS = {"ridge": "ridge/ridge", "tree": "cobit/tree",
          "tree+linear": "rulefit/rulefit", "nn": "nn/nn"}
SPLITS = ("train", "val", "test")
POWER_SCALE, POWER_UNIT = 1000.0, "mW"   # targets are Watts
CYCLE_NS = 4                             # 4-cycle windows, 1 ns/cycle


def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - yh) ** 2).sum() / ss if ss > 0 else float("nan")


def panel(ax, fig, y, yh, split):
    y, yh = y * POWER_SCALE, yh * POWER_SCALE
    lo = float(min(y.min(), yh.min())); hi = float(max(y.max(), yh.max()))
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    hb = ax.hexbin(y, yh, gridsize=60, norm=LogNorm(vmin=1),
                   extent=(lo, hi, lo, hi), cmap="viridis")
    fig.colorbar(hb, ax=ax, label="count")
    ax.plot([lo, hi], [lo, hi], color="r", lw=1, ls="--")   # y = x
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true power ({POWER_UNIT})")
    ax.set_ylabel(f"predicted power ({POWER_UNIT})")
    ax.set_title(split)
    ax.text(0.03, 0.965, f"R$^2$ = {r2(y, yh):.3f}", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.4", lw=0.8))


for name, sub in MODELS.items():
    df = pd.read_pickle(os.path.join(RUN, sub, "predictions.pkl.zst"))
    df["bench"] = df["bench"].astype(str)
    df["split"] = df["split"].astype(str)
    # per-cycle time axis: rows arrive in trace order within each benchmark
    df["time_ns"] = df.groupby("bench", sort=False).cumcount() * CYCLE_NS

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 3.9), squeeze=False)
    for ax, split in zip(axes[0], SPLITS):
        s = df[df["split"] == split]
        panel(ax, fig, s["y_true"].to_numpy(), s["y_pred"].to_numpy(), split)
    fig.tight_layout()
    png = os.path.join(OUT, f"binary-fit-q30-{name}.png")
    fig.savefig(png, dpi=110); plt.close(fig)

    csv = os.path.join(OUT, f"binary-fit-q30-{name}.csv")
    out = df.rename(columns={"y_pred": "pred_sum", "y_true": "aqcore_true"})
    out[["bench", "time_ns", "split", "pred_sum", "aqcore_true"]].to_csv(csv, index=False)
    print(name, {s: round(r2(df.loc[df.split == s, "y_true"],
                            df.loc[df.split == s, "y_pred"]), 4) for s in SPLITS},
          "->", png, csv)
