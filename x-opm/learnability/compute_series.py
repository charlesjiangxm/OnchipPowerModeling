#!/usr/bin/env python
"""Plot VPU power + per-cycle toggle waveforms for ISA_FP and coremark.

Per benchmark -> a 2-row figure sharing the cycle axis:
  row1: VPU power  Pc(x_aq_vpu_top)  [mW]
  row2: total signal toggle rate across all x_aq_vpu_top signals
Raw trace rasterized (thin) + centred rolling mean (bold) so trends read on the
598K-cycle coremark. Also a toggle-vs-power scatter per benchmark.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common

# resolved per-module in main(); module-level names kept so worker funcs can read them
VPU = PWR = PWR_COL = OUT = None
SCALE = _common.SCALE  # W -> mW

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def popcount_u64(a):
    return _POP[a.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.int64)

def _col_toggle_py(vals):
    ints, last = [], 0
    for x in vals:
        if isinstance(x, float) and x != x:
            ints.append(last); continue
        last = int(x); ints.append(last)
    out = np.empty(len(ints) - 1, dtype=np.int64)
    for i in range(len(ints) - 1):
        out[i] = (ints[i] ^ ints[i + 1]).bit_count()
    return out

def col_toggle(vals):
    try:
        a = vals.astype(np.uint64)
    except (ValueError, OverflowError, TypeError):
        return _col_toggle_py(vals)
    return popcount_u64(a[1:] ^ a[:-1])

def compute(args):
    case, mdir, pwr, pwr_col, out = args   # explicit args: workers don't inherit globals
    df = pd.read_pickle(os.path.join(mdir, case + "_func.pkl.zst"))
    n = len(df)
    per_cycle = np.zeros(n - 1, dtype=np.int64)
    for c in df.columns:
        per_cycle += col_toggle(df[c].to_numpy())
    del df
    p = pd.read_pickle(os.path.join(pwr, case + "_pwr.pkl.zst"))[pwr_col]
    p.index = p.index.astype(np.int64)
    pv = p.reindex(range(n)).to_numpy(float) * SCALE
    np.save(os.path.join(out, f"{case}_toggle.npy"), per_cycle)
    np.save(os.path.join(out, f"{case}_power_mW.npy"), pv)
    return case, n

def roll(a, w):
    if w <= 1:
        return a
    return pd.Series(a).rolling(w, center=True, min_periods=max(1, w // 4)).mean().to_numpy()

def plot_case(case):
    tog = np.load(os.path.join(OUT, f"{case}_toggle.npy"))
    pw = np.load(os.path.join(OUT, f"{case}_power_mW.npy"))
    n = len(pw)
    xp = np.arange(n)
    xt = np.arange(1, n)                 # flip entering cycle t
    w = max(1, n // 400)                 # rolling window ~0.25% of trace

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(13, 6.5))
    a1.plot(xp, pw, color="#1f77b4", lw=0.4, alpha=0.35, rasterized=True)
    a1.plot(xp, roll(pw, w), color="#08519c", lw=1.3, label=f"rolling mean (w={w})")
    a1.set_ylabel("VPU power (mW)")
    a1.set_title(f"{case}: VPU power & feature toggling  "
                 f"(n={n:,} cyc | mean {pw.mean():.2f} mW, std {pw.std():.2f} mW)")
    a1.legend(loc="upper right", fontsize=8); a1.margins(x=0); a1.grid(alpha=0.15)

    a2.plot(xt, tog, color="#ff7f0e", lw=0.4, alpha=0.35, rasterized=True)
    a2.plot(xt, roll(tog, w), color="#d94801", lw=1.3, label=f"rolling mean (w={w})")
    a2.set_ylabel("signal toggle rate")
    a2.set_xlabel("cycle index")
    a2.set_title(f"total toggling across all x_aq_vpu_top signals  "
                 f"(mean signal toggle rate {tog.mean():.1f})", fontsize=10)
    a2.legend(loc="upper right", fontsize=8); a2.margins(x=0); a2.grid(alpha=0.15)
    fig.tight_layout()
    fp = os.path.join(OUT, f"vpu_wave_{case}.png")
    fig.savefig(fp, dpi=120); plt.close(fig)

    # toggle-vs-power scatter (align: power at cycle t vs flips entering cycle t)
    fig2, ax = plt.subplots(figsize=(5, 5))
    pw_al = pw[1:]
    m = min(len(pw_al), len(tog))
    idx = np.arange(m)
    if m > 40000:
        idx = np.random.RandomState(0).choice(m, 40000, replace=False)
    ax.scatter(tog[idx], pw_al[idx], s=3, alpha=0.25, rasterized=True)
    r = np.corrcoef(tog[:m], pw_al[:m])[0, 1]
    ax.set_xlabel("signal toggle rate"); ax.set_ylabel("VPU power (mW)")
    ax.set_title(f"{case}: toggle vs power  (Pearson r={r:.3f})")
    ax.grid(alpha=0.2); fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, f"vpu_scatter_{case}.png"), dpi=120); plt.close(fig2)
    return case, fp, float(r)

def main():
    global VPU, PWR, PWR_COL, OUT
    module, cases, OUT = _common.parse(__doc__ or "")
    VPU = _common.module_dir(module)
    PWR = _common.pwr_dir()
    PWR_COL = _common.pwr_col(module)
    print(f"module={module}  cases={cases}  out={OUT}", flush=True)
    jobs = [(c, VPU, PWR, PWR_COL, OUT) for c in cases]
    with ProcessPoolExecutor(max_workers=min(len(cases), _common.CASE_WORKERS)) as ex:
        for f in as_completed({ex.submit(compute, j): j[0] for j in jobs}):
            c, n = f.result(); print("computed", c, n, flush=True)
    for c in cases:
        c, fp, r = plot_case(c)
        print("plotted", c, "-> ", fp, "corr", round(r, 3), flush=True)

if __name__ == "__main__":
    main()
