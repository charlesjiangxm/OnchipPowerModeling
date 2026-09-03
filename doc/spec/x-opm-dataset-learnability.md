# How We Built a Fair "Is This Data Learnable?" Test — A Report

**Project:** x-opm on-chip power modeling (C906, VPU scope).  
**Date:** 2026-08-16.  
**Companion how-to guide:** `doc/x-opm_dataset_learnability.md` (step-by-step usage).  

---

## 1. Summary in one paragraph

Before we spend hours training a power model on a benchmark, we want a cheap, honest
answer to one question: *does the recorded power actually depend on the signals we
recorded, in a way a model could learn?* Our first tool for this — a quick linear fit
(called **ridge**) that produces a score called **R²** — was giving discouraging
numbers, and we suspected the tool, not the data, was at fault. This report records how
we replaced that tool with a **correlation-based test**, why each design choice was
made, how we stress-tested the new test with an independent review, and what it finally
told us. The short version: the new test is fairer and clearer, and it revealed that our
two example benchmarks are learnable for **completely different reasons** — one from
*activity*, one from *held state* — a distinction the old score hid.

---

## 2. The problem we started with

Power in a chip goes up for two very different reasons:

1. **Activity (switching).** Wires and gates flip bits; every flip costs energy. More
   flipping → more power. This is the "datapath" story.
2. **State (held level).** A control signal switches a whole block *on* and holds it on.
   The block then burns power steadily — a high **plateau** — even though almost nothing
   is flipping. This is the "control" story.

Our original learnability score used **ridge regression**, which draws the best
*straight line* from signals to power. Straight lines are a poor fit for the on/off,
plateau-shaped behaviour above, so ridge would often report a low R² and make a
benchmark look **unlearnable** when it simply wasn't *linear*. A low ridge R² was being
read as "give up," which is the wrong conclusion.

The request that started this work was, in plain terms:

> *"Stop using the linear fit. Instead, just measure whether the signal activity
> **correlates** with the big jumps in power — and do it at several time-window sizes and
> combine them."*

That instinct was right. The rest of this report is how we turned it into a test we can
trust.

---

## 3. What we built, in plain terms

We built a new test (script: `x-opm/learnability/correlate_toggle_power.py`) that has
**two tracks**, because power has the two causes above.

### Track 1 — the "activity" test (the original ask)
- Chop the run into equal **time windows** (e.g. 8, 16, 32 … cycles each).
- In each window, count **how much switching happened** (total bit-flips).
- In the same window, measure **how much the power moved** (for example, the size of the
  biggest power jump in that window — the "big power step").
- Ask: *across all the windows, do the busy windows line up with the big-power-move
  windows?* That "lining up" is a **correlation**, a single number from 0 (no relation)
  to 1 (perfect relation).

We do this at **many window sizes at once** and keep the best, because different parts
of the chip react with different delays — a small window catches quick bursts, a larger
one catches slow multi-cycle operations. No single window size is right for everything,
so we sweep them and **merge**.

### Track 2 — the "held state" test (added after review — see §5)
Track 1, by design, is **blind** to the plateau story: if a signal switches a unit on
and then holds still, there is no ongoing flipping for Track 1 to see, even though the
power stays high. So we added a second, simple check: **is there any recorded signal
whose steady value tracks the power level?** If yes, the data is learnable — just from a
*held value*, not from activity. This second track reuses numbers we already compute
elsewhere (per-signal value-vs-power correlation), so it is essentially free.

Reading the two tracks together gives a complete verdict instead of a misleading half of
one.

---

## 4. The four design choices, and why each matters

Each choice below removes a specific way the number could **lie to us**.

**Choice 1 — Compare *changes* with *changes*, never activity with the raw power level.**
If you correlate activity with the raw power *level*, the score creeps up automatically
as the window grows — not because of any real relationship, but because power sits on
long flat stretches that look similar to each other. Comparing *movement* with *movement*
(how much switching vs how much the power moved) does not have this false inflation.

**Choice 2 — Use a *rank* correlation (Spearman), not a plain one (Pearson).**
This is the subtle one, and it is the heart of the fix. The original complaint was that
ridge is *linear* (straight-line). But the ordinary correlation, **Pearson, is also a
straight-line measure** — swapping one for the other would not have solved anything, and
Pearson is additionally thrown off by a few extreme windows. So we lead with
**Spearman**, which only cares about **order** ("busier windows tend to be
higher-power windows"), ignores the exact shape, and shrugs off a handful of freak
windows. This is allowed here because more switching can only *raise* switching power,
never lower it — the relationship is always in one direction, which is exactly what a
rank measure needs.

We still print the Pearson number too, because the **gap** between the two is
informative: when Pearson is high but Spearman is low, it means the relationship is
**real but concentrated in a few very busy bursts**, rather than being true of a typical
window. That is a genuine finding, not noise to be discarded.

**Choice 3 — Always compare against a "could this happen by chance?" baseline.**
We deliberately **misalign** the power trace against the activity (slide it by a large
offset so the timing no longer matches) and re-measure. Whatever correlation survives
that scramble is fake. The honest score is *real minus this chance level*, and we also
report a **p-value** (the probability the result is a fluke). Because our headline is the
*best* of many windows — and "best of many" is naturally flattering — we make the
chance-baseline use that *same* "best of many" rule, so the comparison is apples to
apples.

**Choice 4 — Treat a low activity-score as *inconclusive*, not as a death sentence.**
The activity test looks at the *total* switching, but a real model gets to look at each
signal separately and weight them. So "the total activity doesn't track power" does not
prove "no model can learn this." A low score sends us to Track 2 and to the coverage
check, not to the bin.

---

## 5. How we made sure the test is right

We did not just trust our own code. We ran an independent, adversarial review:

- **Two independent reproductions.** Two separate checks re-computed every headline
  number from the raw data using different code (including a standard statistics
  library). **Every number matched** to three decimals.
- **Three "attack" reviews**, each trying to break the method from a different angle:
  statistics, machine-learning, and hardware/power.

The reviews found three things worth acting on, and we acted on all of them:

1. **"Your activity test is blind to held-state power."** Correct. This is *why* we added
   Track 2 (the held-level test) and why the final verdict combines both.
2. **"Don't call the Pearson-vs-Spearman gap a mere 'artifact'."** Correct. We reworded
   everything to say the gap means *"real, but concentrated in a few bursts."*
3. **"Your 'best of many windows' score is a bit optimistic; test it properly."** Correct.
   We made the chance-baseline use the identical "best of many" rule and report a p-value
   (Choice 3 above). We also increased the number of scramble trials so the baseline is
   stable.

The result is a test whose conclusions survived people actively trying to poke holes in
them.

---

## 6. What the test found (two example benchmarks)

We ran it on the VPU (vector unit) power of two benchmarks. The contrast is the whole
point:

| What we measured | **ISA_FP** | **coremark** | Plain reading |
|---|---|---|---|
| **Activity test** (honest rank correlation, higher = better) | **0.56** (very significant, p≈0) | **≈0** (raw 0.13, no better than chance, p≈0.11) | ISA_FP's power clearly follows its switching. coremark's does not. |
| Pearson-vs-Spearman gap | small (uniform relationship) | large (Pearson ~0.49, Spearman ~0.12) | coremark *does* have some switching-driven power, but only in a few busy bursts. |
| **Held-state test** (best signal whose value tracks power level) | 0.42 | **0.45** (over 1000 signals) | Both have a strong held-state signal — for coremark this is the **main** story. |
| Coverage (share of big power jumps with a visible cause) | 99.9% | 43.9% | Almost everything in ISA_FP is explained; over half of coremark's big jumps are not. |

**Plain conclusions:**

- **ISA_FP is learnable from activity.** More switching reliably means more power, the
  timing lines up, and nearly every power jump has a visible cause. A model should do
  well; a good time-window to use is around **64–128 cycles** (where the relationship is
  strongest).

- **coremark is a different animal — learnable, but from *held state*, not activity.** Its
  switching barely predicts its power (no better than chance). What *does* predict it is
  the **steady value** of a small number of control signals — chiefly the floating-point
  divide/square-root unit's state registers, which flip only a couple of dozen times in
  600,000 cycles yet set long power plateaus. So the fix for coremark is **not** more model
  tuning; it is to **feed those signals' held values** into the model. And because only
  ~44% of its big power jumps have any visible in-scope cause, some of the driver may sit
  in **neighbouring blocks we haven't recorded** — worth widening the capture next.

The old single R² number would have lumped both of these into one discouraging figure and
hidden the crucial difference between them.

### Figures
Two figures per benchmark tell the story (both under `out/x-opm/vpu_activity/`), each
produced by a different script:

- `corr_toggle_power_<bench>.png` — **written by `correlate_toggle_power.py`** — the
  correlation vs window-size curves (left: Pearson; right: Spearman). The coremark pair is
  the clearest picture of "high Pearson, low Spearman = a few bursts, not a general rule."
- `coverage_<bench>.png` — **written by `plot_relevance.py`** — the power trace over time
  with each big jump marked **green** (has a visible cause) or **red** (unexplained).

---

## 7. Recommendations

1. **Use the new test as the pre-training screen**, replacing the ridge-R² estimate. Read
   its two tracks together, and alongside the coverage check.
2. **A low activity-score is a signal to add features, not to quit** — specifically the
   *held values* of control/state signals, then re-measure.
3. **Keep the real trained-model R² as the final word.** The correlation test is a fast,
   honest *screen*; it is not a promise about the trained model's exact accuracy.
4. **For coremark specifically:** add the floating-point unit state signals as level
   features, and consider capturing neighbouring blocks to cover the ~56% of big power
   jumps that currently have no visible cause.

---

## 8. Where everything lives / how to run it

```bash
cd /ic/projects/A513software/jingbo.jiang/OnchipPowerModeling
PY=~/anaconda3/bin/python

# 1. build the per-cycle activity + power traces (needed once per benchmark)
$PY x-opm/learnability/compute_series.py
# 2. per-signal stats incl. the held-value correlations used by Track 2 (+ coverage)
$PY x-opm/learnability/compute_signal_stats.py
# 3. THE NEW TEST — prints the verdict, writes the corr_toggle_power figures + JSON/CSV
$PY x-opm/learnability/correlate_toggle_power.py
# 4. draws the coverage_<bench>.png figures (green=explained / red=unexplained big jumps)
$PY x-opm/learnability/plot_relevance.py
```

- **Scripts:** `x-opm/learnability/correlate_toggle_power.py` (the activity + held-state
  test), `plot_relevance.py` (the coverage figures).
- **Outputs:** `out/x-opm/vpu_activity/corr_toggle_power_<bench>.png`,
  `corr_toggle_power.json`, `corr_toggle_power.csv`, and `coverage_<bench>.png`.
- **Detailed how-to and the full set of diagnostics:** `doc/x-opm_dataset_learnability.md`

---

## Appendix — plain-word glossary

- **Toggle / switching:** a signal flipping between 0 and 1. Each flip costs energy.
- **Big power step:** a large jump in power from one cycle to the next.
- **Correlation:** a number from 0 to 1 saying how tightly two things move together.
- **Pearson vs Spearman:** Pearson measures a *straight-line* fit and is swayed by
  extremes; Spearman only checks whether the *order* matches ("busier ⇒ higher") and is
  robust. We lead with Spearman.
- **Window:** a block of consecutive cycles we average/aggregate over, to absorb timing
  delays.
- **Chance baseline (null) / p-value:** we scramble the timing and re-measure; the
  p-value is the chance the real result is just luck.
- **Held state / plateau:** a signal switches a unit on and holds it, so power stays high
  while little is flipping — invisible to the activity test, caught by the held-state test.
- **R²:** the score from a trained model saying what fraction of the power it explains
  (1 = perfect, 0 = no better than guessing the average).
