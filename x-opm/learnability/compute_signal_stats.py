#!/usr/bin/env python
"""Per-signal change<->power-step coincidence (the refined 'derivative' test).

For every x_aq_vpu_top signal in a benchmark:
  edges   : cycles where the signal changes  (Hamming(state[t],state[t-1])>0)
  impact  : mean |dP| on this signal's edge cycles          (mW)
  lift    : impact / mean|dP|   -> how much bigger power steps are when it moves
  spec    : frac of its edges coincident (+-1 cyc) with a top-1% |dP| step
  lvl_corr: corr(signal value, power level)  -> catches held-constant control
            signals whose LEVEL sets a sustained power plateau
Rare + high-lift = a control signal that moves seldom but shifts power a lot
(invisible to mean toggle rate). Also computes how many big power steps are
"covered" by an informative (non-clock) signal edge = learnability check.
"""
import os, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import _common

VPU = PWR = PWR_COL = OUT = None
SCALE = _common.SCALE

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def popcount_u64(a): return _POP[a.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.int64)

def _py_hamming(vals):
    ints, last = [], 0
    for x in vals:
        if isinstance(x, float) and x != x: ints.append(last); continue
        last = int(x); ints.append(last)
    out = np.empty(len(ints)-1, dtype=np.int64)
    for i in range(len(ints)-1): out[i] = (ints[i]^ints[i+1]).bit_count()
    return out, None

def hamming_and_value(vals):
    """(hamming[n-1], value_float[n] or None)."""
    try:
        a = vals.astype(np.uint64)
    except (ValueError, OverflowError, TypeError):
        return _py_hamming(vals)
    return popcount_u64(a[1:] ^ a[:-1]), a.astype(np.float64)

def analyze(args):
    case, mdir, pwr, pwr_col, out = args   # explicit args: workers don't inherit globals
    df = pd.read_pickle(os.path.join(mdir, case + "_func.pkl.zst"))
    n  = len(df)
    p  = pd.read_pickle(os.path.join(pwr, case + "_pwr.pkl.zst"))[pwr_col]
    p.index = p.index.astype(np.int64)
    p = p.reindex(range(n)).to_numpy(float) * SCALE
    dP  = np.diff(p)                 # len n-1, step entering cyc k+1
    adP = np.abs(dP); adP_mean = float(adP.mean())
    plevel = p[1:]                   # power level aligned to edge index
    thr = np.quantile(adP, 0.99)
    big = adP >= thr                 # top-1% power steps
    big_w = big.copy()               # +-1 cycle tolerance
    big_w[1:]  |= big[:-1]; big_w[:-1] |= big[1:]

    rows = []
    info_activity = np.zeros(n-1, dtype=np.int64)   # switching from non-clock signals
    any_edge      = np.zeros(n-1, dtype=np.int64)    # edges from all non-freeclock signals
    for c in df.columns:
        h, vfloat = hamming_and_value(df[c].to_numpy())
        edges = h > 0
        ne = int(edges.sum())
        if ne == 0:                                  # constant within benchmark
            rows.append((c, 0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0)); continue
        er = ne / (n-1)
        impact = float(adP[edges].mean())
        lift   = impact / adP_mean if adP_mean else float("nan")
        spec   = float((edges & big_w).sum()) / ne
        lvl    = (float(np.corrcoef(vfloat, p)[0,1])
                  if vfloat is not None and np.std(vfloat) > 0 else float("nan"))
        rows.append((c, ne, er, impact, lift, spec, lvl, int(h.max())))
        if er < 0.99: any_edge += edges
        if er < 0.50: info_activity += h
    del df

    tab = pd.DataFrame(rows, columns=["signal","n_edges","edge_rate","impact_mW",
                                      "lift","spec","lvl_corr","max_hamming"])
    tab.to_csv(os.path.join(out, f"per_signal_{case}.csv"), index=False)
    np.save(os.path.join(out, f"{case}_info_activity.npy"), info_activity)

    nbig = int(big.sum())
    cov_info = float((big & (info_activity > 0)).sum()) / max(nbig, 1)
    cov_any  = float((big & (any_edge > 0)).sum())      / max(nbig, 1)
    summary = {
        "case": case, "n_cycles": int(n), "n_signals": int(len(tab)),
        "n_constant_signals": int((tab.n_edges == 0).sum()),
        "big_step_thr_mW": round(float(thr), 4), "n_big_steps": nbig,
        "coverage_by_informative_edge(er<0.5)": round(cov_info, 3),
        "coverage_by_any_nonclock_edge": round(cov_any, 3),
        "mean_abs_dP_mW": round(adP_mean, 4),
    }
    return summary

def main():
    global VPU, PWR, PWR_COL, OUT
    module, cases, OUT = _common.parse(__doc__ or "")
    VPU = _common.module_dir(module)
    PWR = _common.pwr_dir()
    PWR_COL = _common.pwr_col(module)
    print(f"module={module}  cases={cases}  out={OUT}", flush=True)
    summaries = []
    jobs = [(c, VPU, PWR, PWR_COL, OUT) for c in cases]
    with ProcessPoolExecutor(max_workers=min(len(cases), _common.CASE_WORKERS)) as ex:
        for f in as_completed({ex.submit(analyze, j): j[0] for j in jobs}):
            s = f.result(); summaries.append(s); print("done", s["case"], flush=True)
    json.dump(summaries, open(os.path.join(OUT, "per_signal_summary.json"), "w"), indent=2)
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
