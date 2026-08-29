#!/usr/bin/env python
"""Diagnostic B (correlation flavor): does SWITCHING track POWER MOVEMENT?  (+ a
state->level companion so a low switching score is never mistaken for "unlearnable").

A linear ridge R^2 answers "learnable?" with a *linear* model, so it under-reports
whenever the true relation is nonlinear or the power trace is fat-tailed. This script
replaces that pre-training estimate with a model-free, multi-scale CORRELATION.

TWO TRACKS, because power is driven two different ways (see the doc, section 2):

  SWITCHING track  (this file's core, the user's ask -- "toggle vs big power step"):
    at window W, per non-overlapping block, correlate
        A_W  = total bit-flips in the block                (how much switching)
      vs a POWER-MOVEMENT target
        tv     = sum|dP|      total variation in the block
        maxstep= max|dP|      the single biggest power STEP in the block
        range  = max(P)-min(P) peak-to-peak excursion
        level  = mean(P)      (reference only -- correlating with the LEVEL inflates)

  STATE->LEVEL track  (companion, reads per_signal_<case>.csv from step 2):
        the best in-scope |corr(signal VALUE, power LEVEL)| -- catches a held-state
        control signal that sets a sustained power PLATEAU (P high, dP~0). The
        switching track is *structurally blind* to these, so we report both.

Why correlation-of-CHANGE, not of level (switching track): correlating switching
with the power *level* rises "for free" as W grows (plateaus are autocorrelated);
correlating with power *movement* does not -- the circular-shift null stays ~0.

Why SPEARMAN alongside Pearson: Pearson is itself linear and leverage-sensitive
(same flaw as ridge). Spearman (rank) reports the TYPICAL-block monotone coupling;
Pearson reports whether the BIG switching events drive the BIG power events. They are
complementary: Pearson >> Spearman means the coupling is real but concentrated in a
few high-switching burst windows (NOT a mere "artifact"). We lead with Spearman for
the headline because dynamic power is monotone in switching, but print both.

Merge over windows ("multi-scale"): lag varies by signal, so no single W is right.
We sweep W and take the max; because a max over (W x targets) is optimistic, the
headline is tested against a permutation null of that SAME max statistic (circularly
shift power, recompute the whole max, repeat) -- report score, null95 and a p-value.

WHAT A LOW SWITCHING SCORE MEANS: "not learnable from unweighted datapath switching"
-- it is INCONCLUSIVE for overall learnability, because (a) a model uses per-signal,
capacitance-weighted features, not this one aggregate count, and (b) the driver may be
a control-STATE level (see the level track / Diagnostic A). Only "low switching AND no
in-scope level coupling" points to an off-camera / noise driver.

Inputs : out/.../<case>_{toggle,power_mW}.npy (compute_series.py) and, optionally,
         out/.../per_signal_<case>.csv (compute_signal_stats.py, for the level track).
Outputs: corr_toggle_power.json (+ .csv table) and corr_toggle_power_<case>.png
"""
import os, json, csv
import numpy as np
import _common
try:
    from scipy.stats import rankdata
except Exception:                       # tiny fallback so the file runs without scipy
    def rankdata(a):
        order = np.argsort(a, kind="mergesort"); r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1)
        a_sorted = a[order]; i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]: j += 1
            if j > i: r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return r

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = None   # resolved in main()
CASES = None
WS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MIN_BLOCKS = 120          # drop a window if it leaves fewer blocks than this
NSHIFT = 64               # circular-shift replicates for the null (more = stabler null95 / finer p)
TARGETS = ["tv", "maxstep", "range", "level"]
HEADLINE = ["tv", "maxstep"]     # change-based, big-step-weighted -> the switching verdict
LVL_MIN_EDGES = 10        # a signal must move >=this many times before we trust its lvl_corr
LVL_REL = 0.3             # |lvl_corr| above this = a state signal that tracks the power level
TCOLOR = {"tv": "#1f77b4", "maxstep": "#d62728", "range": "#2ca02c", "level": "#7f7f7f"}
TLABEL = {"tv": "tv = Σ|ΔP|", "maxstep": "maxstep = max|ΔP|",
          "range": "range = max−min P", "level": "level = mean P (ref)"}


def load(case):
    tog = np.load(f"{OUT}/{case}_toggle.npy").astype(np.float64)   # len n-1, transition k->k+1
    pw = np.load(f"{OUT}/{case}_power_mW.npy")                     # len n; dP[k]=P[k+1]-P[k] pairs tog[k]
    return tog, pw


def _blocks(x, W):
    m = (len(x) // W) * W
    return x[:m].reshape(-1, W)


def _target_from(adP_block, P_block, name):
    return {"tv": adP_block.sum(1), "maxstep": adP_block.max(1),
            "range": P_block.max(1) - P_block.min(1), "level": P_block.mean(1)}[name]


def targets(tog, pw, W):
    """Block-aligned (A, {target: T}) for a given window W."""
    L = len(tog)
    A = _blocks(tog, W).sum(1)
    aB = _blocks(np.abs(np.diff(pw)), W)
    Pb = _blocks(pw[:L], W)
    return A, {t: _target_from(aB, Pb, t) for t in TARGETS}


def pear(a, b):
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(abs(np.corrcoef(a, b)[0, 1]))


def spear(ra, b):
    """Spearman = Pearson on ranks; ra is a PRE-RANKED activity vector."""
    if b.std() == 0: return float("nan")
    rb = rankdata(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def switching_track(case, tog, pw):
    """Full multi-scale switching<->power-movement analysis with a max-statistic null."""
    L = len(tog); adP = np.abs(np.diff(pw)); Pl = pw[:L]
    ws = [W for W in WS if L // W >= MIN_BLOCKS]
    shifts = np.linspace(L / (NSHIFT + 1), NSHIFT * L / (NSHIFT + 1), NSHIFT).astype(int)

    curves = {t: {"W": [], "pear": [], "spear": [], "np": [], "ns": []} for t in TARGETS}
    # per (target) collect, for every shift, the shifted-Spearman at each W -> lets us
    # build BOTH the per-W null and the null of the max-over-(W,target) statistic.
    shift_spear = {t: [] for t in TARGETS}     # list over W of arrays[NSHIFT]
    for W in ws:
        A, T = targets(tog, pw, W); ra = rankdata(A)
        aB_shift = [_blocks(np.roll(adP, s), W) for s in shifts]
        Pb_shift = [_blocks(np.roll(Pl, s), W) for s in shifts]
        for t in TARGETS:
            ps = np.array([pear(A, _target_from(aB_shift[i], Pb_shift[i], t)) for i in range(NSHIFT)])
            ss = np.array([spear(ra, _target_from(aB_shift[i], Pb_shift[i], t)) for i in range(NSHIFT)])
            c = curves[t]
            c["W"].append(W); c["pear"].append(pear(A, T[t])); c["spear"].append(spear(ra, T[t]))
            c["np"].append(float(np.nanpercentile(ps, 95))); c["ns"].append(float(np.nanpercentile(ss, 95)))
            shift_spear[t].append(ss)

    # headline = max Spearman over (W, HEADLINE targets); null of that same max statistic
    def stack(sel):
        real = np.array([[s if s == s else -9 for s in curves[t]["spear"]] for t in sel])   # [nt, nW]
        null = np.array([np.nan_to_num(np.vstack(shift_spear[t]), nan=-9) for t in sel])      # [nt, nW, NSHIFT]
        return real, null
    real_h, null_h = stack(HEADLINE)
    S_max = float(real_h.max())
    S_max_null = null_h.max(axis=(0, 1))                       # [NSHIFT] draws of the max stat
    null95 = float(np.percentile(S_max_null, 95))
    pval = float((S_max_null >= S_max).mean())
    # locate the winning (target, W)
    ti, wi = np.unravel_index(int(real_h.argmax()), real_h.shape)
    best_t = HEADLINE[ti]; best_W = curves[best_t]["W"][wi]
    pear_at = curves[best_t]["pear"][wi]

    per_target = {}
    for t in TARGETS:
        c = curves[t]
        gap = [ (s - ns) if s == s else -9 for s, ns in zip(c["spear"], c["ns"]) ]
        i = int(np.argmax(gap))
        per_target[t] = {"spearman": round(c["spear"][i], 3), "spearman_null95": round(c["ns"][i], 3),
                         "pearson": round(c["pear"][i], 3), "pearson_null95": round(c["np"][i], 3),
                         "best_W": int(c["W"][i])}
    # is the coupling concentrated in a few bursts?  (Pearson strong where Spearman is weak)
    pear_max = max(max((v for v in curves[t]["pear"] if v == v), default=0.0) for t in HEADLINE)
    burst_concentrated = pear_max - S_max >= 0.2

    summary = {
        "headline_spearman": round(S_max, 3), "headline_null95": round(null95, 3),
        "headline_honest": round(S_max - null95, 3), "headline_pvalue": round(pval, 3),
        "best_target": best_t, "best_W": int(best_W), "pearson_at_best": round(pear_at, 3),
        "pearson_max_over_headline": round(pear_max, 3),
        "burst_concentrated": bool(burst_concentrated), "per_target": per_target,
    }
    return summary, curves


def level_track(case):
    """State->level companion from per_signal_<case>.csv (Diagnostic A material)."""
    fp = f"{OUT}/per_signal_{case}.csv"
    if not os.path.exists(fp):
        return {"available": False}
    best, nrel, n = 0.0, 0, 0
    with open(fp) as f:
        for row in csv.DictReader(f):
            try:
                ne = int(float(row["n_edges"])); lc = abs(float(row["lvl_corr"]))
            except (ValueError, KeyError):
                continue
            if ne < LVL_MIN_EDGES or lc != lc:
                continue
            n += 1; best = max(best, lc); nrel += (lc > LVL_REL)
    # multiple-comparison guard: max of ~n weak corrs over N samples ~ sqrt(2 ln n)/sqrt(N)
    return {"available": True, "best_lvl_corr": round(best, 3),
            "n_signals_relevant": nrel, "n_signals_tested": n}


def verdict(sw, lv):
    S = sw["headline_honest"]; strong = S >= 0.5; weak = 0.2 <= S < 0.5
    has_level = lv.get("available") and lv.get("best_lvl_corr", 0) >= LVL_REL
    burst = sw["burst_concentrated"]
    if strong:
        v = f"LEARNABLE — datapath switching drives power (set model window ≈ {sw['best_W']} cyc)."
    elif weak:
        v = ("PARTIAL — switching explains some of it"
             + (", concentrated in high-switching bursts" if burst else "") + ".")
    else:
        v = "switching does NOT track power movement (unweighted-toggle screen is negative)."
    if not strong:
        if has_level:
            v += (f"  BUT an in-scope control/STATE signal tracks the power LEVEL "
                  f"(|lvl_corr|={lv['best_lvl_corr']}, {lv['n_signals_relevant']} signals >|{LVL_REL}|) "
                  f"→ LEARNABLE from LEVEL features, not switching. Add them; see Diagnostic A/C.")
        elif lv.get("available"):
            v += ("  And no in-scope signal's LEVEL tracks power either → driver likely "
                  "OFF-CAMERA/noise: widen scope or check the power estimate.")
        else:
            v += "  (run compute_signal_stats.py for the level track before concluding.)"
    return v


def plot(case, curves):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ws = curves["tv"]["W"]
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, key, ylab in [(axp, "pear", "absolute pearson correlation"),
                          (axs, "spear", "absolute spearman correlation")]:
        for t in TARGETS:
            c = curves[t]; style = "--" if t == "level" else "-"
            ax.plot(c["W"], c[key], style, color=TCOLOR[t], lw=1.8, marker="o", ms=3,
                    label=TLABEL[t], alpha=0.9 if t != "level" else 0.6)
            ax.fill_between(c["W"], 0, c["np" if key == "pear" else "ns"], color=TCOLOR[t], alpha=0.06)
        ax.set_xscale("log", base=2)                       # even spacing, but decimal tick labels:
        ax.set_xticks(ws); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("window Size (cycles)"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.2, which="both"); ax.set_ylim(-0.02, 1.0)
    axs.axhline(0.5, color="k", ls=":", lw=0.7); axs.text(ws[0], 0.51, "strong ≥0.5", fontsize=7)
    axs.axhline(0.2, color="k", ls=":", lw=0.7); axs.text(ws[0], 0.21, "weak ≥0.2", fontsize=7)
    axp.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fp = f"{OUT}/corr_toggle_power_{case}.png"
    fig.savefig(fp, dpi=120); plt.close(fig)
    return fp


def main():
    global OUT, CASES
    module, CASES, OUT = _common.parse(__doc__ or "")
    print(f"module={module}  cases={CASES}  out={OUT}", flush=True)
    all_summ, rows = [], []
    for case in CASES:
        tog, pw = load(case)
        sw, curves = switching_track(case, tog, pw)
        lv = level_track(case)
        fp = plot(case, curves)
        v = verdict(sw, lv)
        summ = {"case": case, "n_cycles": int(len(pw)), "switching": sw, "level": lv, "verdict": v}
        all_summ.append(summ)

        print(f"\n===== {case}  (n={len(pw):,}) =====")
        print(f"  {'target':9s} {'spearman':>9s} {'sp.null':>8s} {'pearson':>8s} {'pe.null':>8s} {'bestW':>6s}")
        for t in TARGETS:
            d = sw["per_target"][t]
            print(f"  {t:9s} {d['spearman']:>9.3f} {d['spearman_null95']:>8.3f} "
                  f"{d['pearson']:>8.3f} {d['pearson_null95']:>8.3f} {d['best_W']:>6d}")
            rows.append([case, t, d['spearman'], d['spearman_null95'], d['pearson'], d['pearson_null95'], d['best_W']])
        print(f"  SWITCHING headline (max Spearman over {HEADLINE}): {sw['headline_spearman']:.3f} "
              f"[{sw['best_target']} @W={sw['best_W']}]  null95={sw['headline_null95']:.3f}  "
              f"honest={sw['headline_honest']:.3f}  p={sw['headline_pvalue']:.3f}"
              f"  (Pearson_max={sw['pearson_max_over_headline']:.3f}"
              f"{', BURST-concentrated' if sw['burst_concentrated'] else ''})")
        if lv.get("available"):
            print(f"  LEVEL track: best in-scope |lvl_corr|={lv['best_lvl_corr']:.3f} "
                  f"({lv['n_signals_relevant']} signals >|{LVL_REL}|, of {lv['n_signals_tested']} tested)")
        print(f"  >>> VERDICT: {v}")
        print(f"  wrote {fp}")

    json.dump(all_summ, open(f"{OUT}/corr_toggle_power.json", "w"), indent=2)
    with open(f"{OUT}/corr_toggle_power.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "target", "spearman", "spearman_null95",
                                       "pearson", "pearson_null95", "best_W"]); w.writerows(rows)
    print(f"\nwrote {OUT}/corr_toggle_power.json and .csv")


if __name__ == "__main__":
    main()
