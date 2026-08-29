#!/usr/bin/env python
"""Characterize phase lag between switching activity and power (from cache).

(1) cross-correlation curve corr(activity(t), power(t+L)) over lags L -> peak L*
    and its width = how many cycles power takes to respond / how smeared.
(2) windowed correlation at lag 0 vs block size W -> shows windowing absorbs lag.
(3) coverage of big |dP| steps vs a WIDE tolerance window (rule out large lag).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/ic/projects/A513software/jingbo.jiang/OnchipPowerModeling/out/x-opm/vpu_activity"

def load(case):
    a=np.load(f"{OUT}/{case}_toggle.npy").astype(float)   # activity, len n-1 (transition k->k+1)
    p=np.load(f"{OUT}/{case}_power_mW.npy")                # power level, len n
    act=np.zeros(len(p)); act[1:]=a                        # activity "at" cycle c
    return act, p

def xcorr_curve(act,p,lags):
    act=act-act.mean(); p=p-p.mean()
    out=[]
    n=len(p)
    for L in lags:
        if L>=0: x,y=act[:n-L], p[L:]      # activity at c vs power at c+L (power later)
        else:    x,y=act[-L:], p[:n+L]
        d=np.sqrt((x*x).sum()*(y*y).sum())
        out.append((x*y).sum()/d if d>0 else 0.0)
    return np.array(out)

def block_corr(act,p,W):
    m=(len(p)//W)*W
    if m==0: return float("nan")
    A=act[:m].reshape(-1,W).mean(1); P=p[:m].reshape(-1,W).mean(1)
    if A.std()==0 or P.std()==0: return float("nan")
    return np.corrcoef(A,P)[0,1]

def dilate(mask,w):
    out=mask.copy()
    for s in range(1,w+1):
        out[s:]|=mask[:-s]; out[:-s]|=mask[s:]
    return out

lags=np.arange(-8,65)
fig,ax=plt.subplots(figsize=(10,5))
for case,col in [("ISA_FP","#1f77b4"),("coremark","#d62728")]:
    act,p=load(case)
    c=xcorr_curve(act,p,lags)
    Lstar=lags[np.argmax(c)]
    ax.plot(lags,c,color=col,lw=1.5,label=f"{case}  (peak L*={Lstar:+d}, r={c.max():.3f})")
    ax.axvline(Lstar,color=col,ls=":",lw=0.8)
    print(f"\n===== {case} =====")
    print(f"  activity-vs-power peak lag L* = {Lstar:+d} cycles, r={c.max():.3f}  (r@lag0={c[list(lags).index(0)]:.3f})")
    print("  windowed corr(activity,power) @lag0 vs block size W:")
    for W in [1,4,8,16,32,64,128,256]:
        print(f"     W={W:4d}: r={block_corr(act,p,W):.3f}")
    # coverage vs wide window
    info=np.load(f"{OUT}/{case}_info_activity.npy"); ie=info>0
    dP=np.diff(p); adP=np.abs(dP); big=adP>=np.quantile(adP,0.99)
    covs=[f"+-{w}:{100*np.mean(big&dilate(ie,w))/np.mean(big):.0f}%" for w in [0,4,16,32,64]]
    print("  big-step coverage vs tolerance:", "  ".join(covs))
ax.axvline(0,color="grey",lw=0.6)
ax.set_xlabel("lag L (cycles): power measured L cycles AFTER activity  →")
ax.set_ylabel("corr(activity(t), power(t+L))")
ax.set_title("Phase lag: how many cycles after switching does VPU power peak?")
ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout(); fp=f"{OUT}/lag_xcorr.png"; fig.savefig(fp,dpi=120); plt.close(fig)
print("\nwrote",fp)
