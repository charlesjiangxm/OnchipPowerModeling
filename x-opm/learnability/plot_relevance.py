#!/usr/bin/env python
"""Visualize signal power-relevance vs rarity, and covered/uncovered power steps."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common
OUT = None   # resolved in main()

def relevance_scatter(case):
    t=pd.read_csv(f"{OUT}/per_signal_{case}.csv")
    a=t[(t.n_edges>0)].dropna(subset=["lvl_corr"]).copy()
    a["abscorr"]=a.lvl_corr.abs()
    fig,ax=plt.subplots(figsize=(9,6))
    sc=ax.scatter(a.edge_rate, a.abscorr, s=8+40*(a.lift/ (a.lift.max() or 1)),
                  c=np.log10(a.n_edges), cmap="viridis", alpha=0.6, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("edge_rate  =  fraction of cycles the signal changes  (≈ its toggle activity)")
    ax.set_ylabel("|corr(signal value, power level)|   (power relevance)")
    ax.set_title(f"{case}: power relevance vs how often a signal moves\n"
                 "upper-LEFT = rarely changes but strongly tracks power  →  toggle-rate BLIND SPOT")
    cb=fig.colorbar(sc); cb.set_label("log10(n_edges)")
    # shade blind-spot region
    ax.axvspan(a.edge_rate.min()*0.5, 1e-3, color="red", alpha=0.05)
    ax.axhline(0.3, color="grey", ls=":", lw=0.8)
    ax.text(a.edge_rate.min(), 0.31, "|corr|=0.3", fontsize=7, color="grey")
    # annotate a few rare+relevant
    blind=a[(a.edge_rate<1e-3)&(a.abscorr>0.3)].sort_values("abscorr",ascending=False)
    for _,r in blind.head(4).iterrows():
        ax.annotate(r.signal.split("/")[-1], (r.edge_rate, r.abscorr),
                    fontsize=6.5, xytext=(6,3), textcoords="offset points", color="darkred")
    ax.grid(alpha=0.15, which="both")
    fig.tight_layout(); fp=f"{OUT}/relevance_{case}.png"; fig.savefig(fp,dpi=120); plt.close(fig)
    n_blind=len(blind)
    print(f"{case}: {n_blind} rare(edge_rate<1e-3)+relevant(|corr|>0.3) signals -> {fp}")
    return n_blind

def covered_timeline(case):
    from matplotlib.ticker import FuncFormatter
    p=np.load(f"{OUT}/{case}_power_mW.npy"); info=np.load(f"{OUT}/{case}_info_activity.npy")
    dP=np.diff(p); adP=np.abs(dP); thr=np.quantile(adP,0.99); big=adP>=thr
    cov=big&(info>0); unc=big&(info==0)
    x=np.arange(len(dP))
    fig,ax1=plt.subplots(figsize=(13,5))
    l_pow,=ax1.plot(np.arange(len(p)), p, color="#1f77b4", lw=0.3, alpha=0.6,
                    rasterized=True, label="power")
    ax1.set_ylabel("VPU power (mW)"); ax1.set_xlabel("simulation time (cycles)")
    ax1.margins(x=0); ax1.grid(alpha=0.15)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v/1e3:.0f}K"))
    ax2=ax1.twinx()
    l_unc=ax2.scatter(x[unc], adP[unc], s=6, c="#d62728", alpha=0.5,
                      rasterized=True, label="unexplained sample")
    l_cov=ax2.scatter(x[cov], adP[cov], s=6, c="#2ca02c", alpha=0.5,
                      rasterized=True, label="explained sample")
    ax2.set_ylabel("absolute power difference (mW)"); ax2.margins(x=0)
    ax1.legend(handles=[l_pow, l_unc, l_cov], loc="upper left", fontsize=8)
    fig.tight_layout(); fp=f"{OUT}/coverage_{case}.png"; fig.savefig(fp,dpi=120); plt.close(fig)
    print(f"{case}: covered={cov.sum()} unexplained={unc.sum()} ({100*unc.sum()/big.sum():.1f}%) -> {fp}")

def main():
    global OUT
    module, cases, OUT = _common.parse(__doc__ or "")
    print(f"module={module}  cases={cases}  out={OUT}", flush=True)
    for c in cases:
        relevance_scatter(c); covered_timeline(c)

    # refined ranking (robust: require >=10 edges so the single global-peak degeneracy is excluded)
    print("\n==== rare-but-relevant control/state signals (n_edges>=10, sorted by |lvl_corr|) ====")
    for c in cases:
        t=pd.read_csv(f"{OUT}/per_signal_{c}.csv")
        a=t[(t.n_edges>=10)].dropna(subset=["lvl_corr"]).copy(); a["abscorr"]=a.lvl_corr.abs()
        a=a[a.edge_rate<1e-3].sort_values("abscorr",ascending=False)
        print(f"\n-- {c} (rare: edge_rate<1e-3) --")
        print(a.head(6)[["signal","n_edges","edge_rate","impact_mW","lift","lvl_corr"]].to_string(index=False))


if __name__ == "__main__":
    main()
