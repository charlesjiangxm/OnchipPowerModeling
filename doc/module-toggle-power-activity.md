# Per-Module Toggle Activity vs. Power Fluctuation

Which benchmarks exercise each `aq_core` sub-module — measured as feature **bit
toggle rate** and module **power fluctuation** — for the
`c906_db_net_1cyc_20260729` dataset.

Modules covered: **cp0, idu, ifu, iu, lsu, rtu, vidu, vpu**.
Benchmarks (12): cache, conv_softmax, coremark, csr, debug, exception, interrupt,
ISA_FP, ISA_INT, ISA_LS, ISA_THEAD, MMU. **Note: `lsu` and `vpu` have no
`conv_softmax` (11 benchmarks each).**

Generated 2026-08-15.

---

## Data sources

- **Features:** `c906_db_net_1cyc_20260729/aq_core/<module>/<bench>_func.pkl` —
  a pandas DataFrame indexed by `time_ns` (integer cycles), every column
  `object`-dtype holding the **raw per-cycle signal STATE value as an int** (NOT
  a toggle bitmask). Column names are full RTL hierarchy paths with an optional
  `[msb:lsb]` bit-range suffix.
- **Power:** `c906_db_net_1cyc_20260729/pwr/<bench>_pwr.pkl`, column
  `x_aq_core/Pc(x_aq_<module>_top)`. Units are **Watts**; all values below are
  reported in **mW** (×1000). Power has exactly one more row than the features;
  aligned by integer-index intersection (same-cycle, no lag), dropping the
  trailing power row.

---

## Column formulas

Let a module have columns `c = 1..M`. For column `c`, let `w_c` be its bit width
(`abs(msb-lsb)+1`, or `1` for a single-bit / bare-name net) and let `s_c[t]` be
its integer state at cycle `t`, over `N` cycles (`t = 0..N-1`).

| Column | Definition |
|---|---|
| **cycles** | `N` — number of simulated cycles in the benchmark (= feature rows). |
| **total_bits** | `B = Σ_c w_c` — sum of all signal bit widths in the module (constant across benchmarks; shown in each module header). |
| **flips/cyc** (intermediate) | `F = (1/(N-1)) · Σ_{t=1}^{N-1} Σ_c popcount(s_c[t] XOR s_c[t-1])` — mean number of feature bits that flip per cycle, summed over all columns and length-normalised. |
| **bit toggle rate** | `BTR = 100% · F / B` — fraction of all module feature bits that flip per cycle. The main, length- and size-normalised activity metric. |
| **pwr_mean** | `mean_t P[t]` in mW — average per-cycle module power. |
| **pwr_std** | `std_t P[t]` in mW — standard deviation of per-cycle power (the fluctuation metric). |
| **pwr_cv** | `pwr_std / pwr_mean` — coefficient of variation (dimensionless). |

**Verdict** column (heuristic flag, per-module relative to that module's idle
floor): `toggle-hot` if `F ≥ max(3·floor_F, floor_F+20)`; `power-hot` if
`pwr_std ≥ max(2·floor_std, floor_std+0.5 mW)`, where `floor_*` is the minimum
of that metric across the module's benchmarks. `*** BOTH` = toggle-hot AND
power-hot; else `toggle-only` / `power-only` / `idle-ish`.

### Notes / caveats

- **popcount is approximate for wide buses.** Columns whose state cannot be cast
  directly to `uint64` (contain NaN, or a data bus stored as `float64`) go through
  a `to_numeric → forward-fill(hold previous state) → uint64` path, which loses
  bits above 2^53 and caps buses at 64 bits. This slightly under-counts a few wide
  data buses but does not change the benchmark ranking.
- **`total_bits` uses nominal widths** (up to 128 b for the widest buses), so for
  those few >64 b buses `BTR` is a mild under-estimate.
- **Power range (max−min) was intentionally dropped**: it is dominated by a
  shared reset/warmup spike present in every benchmark and is therefore nearly
  constant across benchmarks (non-discriminative). Use `pwr_std` / `pwr_cv`.
- **Toggle activity vs. power responsiveness are two different axes.** A benchmark
  can toggle heavily while the module power barely moves — see cp0 and rtu below.

---

## Module summary: does power respond to workload?

`std_max/min` = ratio of the largest to smallest `pwr_std` across a module's
benchmarks (how strongly benchmark choice changes power fluctuation).

| Module | total_bits | pwr_mean (mW) | pwr_std range (mW) | std_max/min | BTR range | Power responsiveness |
|---|---|---|---|---|---|---|
| **lsu** | 50362 | ~24–34 | 4.9 – 18.2 | 3.7× | 0.01 – 2.82% | **Strong** (largest absolute swings) |
| **ifu** | 17244 | ~2.8–15 | 2.5 – 7.1 | 2.9× | 0.08 – 8.05% | **Strong** (most sensitive) |
| **idu** | 18731 | ~2.9–5.9 | 0.99 – 2.10 | 2.1× | 0.06 – 4.91% | Moderate |
| **iu** | 18995 | ~1.0–2.1 | 0.24 – 0.85 | 3.6× | 0.03 – 2.29% | Relative yes, small absolute (<1 mW) |
| **vidu** | 13716 | ~1.5–2.1 | 0.40 – 0.83 | 2.1× | 0.03 – 1.87% | Relative yes, small absolute |
| **rtu** | 6802 | ~0.4–1.1 | 0.21 – 0.55 | 2.6× | 0.04 – 3.69% | Tiny absolute; weakly discriminative |
| **vpu** | 104567 | ~4.5–4.9 | 1.04 – 1.66 | 1.6× | 0.002 – 0.186% | Weak — only `ISA_FP` lifts power; else near idle |
| **cp0** | 18873 | ~1.16 | 0.25 – 0.28 | **1.1×** | 0.01 – 0.43% | **Flat** (power ~fixed regardless of benchmark) |

**Bottom line:** toggle activity is universally led by `coremark`, `conv_softmax`
and the `ISA_*` micro-benchmarks, but power only tracks the workload for **lsu**
and **ifu** (strong), **idu** (moderate) and **iu / vidu** (small absolute).
For **vpu**, only `ISA_FP` (the FP / vector datapath) meaningfully raises toggling
and power; every other benchmark leaves the VPU near idle and even ISA_FP's power
lift is modest. **cp0** and **rtu** per-cycle power is effectively fixed — no
benchmark produces a meaningful power swing there.

---

## Per-module tables

Sorted by bit toggle rate (descending).

### lsu — `total_bits = 50362`, `Pc(x_aq_lsu_top)` (no conv_softmax)

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 2.822% | 34.118 | 18.235 | 0.534 | *** BOTH |
| ISA_THEAD | 21143 | 0.351% | 24.888 | 10.254 | 0.412 | *** BOTH |
| ISA_INT | 30030 | 0.285% | 24.386 | 8.029 | 0.329 | toggle-only |
| ISA_LS | 83386 | 0.248% | 23.745 | 7.716 | 0.325 | toggle-only |
| exception | 12169 | 0.214% | 30.464 | 15.723 | 0.516 | *** BOTH |
| ISA_FP | 71076 | 0.201% | 23.375 | 5.063 | 0.217 | toggle-only |
| interrupt | 3563 | 0.166% | 24.631 | 7.385 | 0.300 | toggle-only |
| cache | 8313 | 0.150% | 24.303 | 5.180 | 0.213 | toggle-only |
| MMU | 4233 | 0.138% | 24.767 | 8.180 | 0.330 | toggle-only |
| csr | 2736 | 0.089% | 24.209 | 5.335 | 0.220 | toggle-only |
| debug | 39228 | 0.013% | 23.595 | 4.913 | 0.208 | idle-ish |

### ifu — `total_bits = 17244`, `Pc(x_aq_ifu_top)`

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 8.047% | 15.242 | 5.323 | 0.349 | *** BOTH |
| conv_softmax | 1438221 | 4.006% | 8.645 | 7.093 | 0.820 | *** BOTH |
| ISA_FP | 71076 | 1.647% | 5.646 | 6.180 | 1.095 | *** BOTH |
| ISA_THEAD | 21143 | 1.194% | 5.049 | 5.626 | 1.114 | *** BOTH |
| ISA_INT | 30030 | 1.154% | 4.995 | 5.664 | 1.134 | *** BOTH |
| ISA_LS | 83386 | 0.980% | 4.226 | 5.069 | 1.200 | *** BOTH |
| exception | 12169 | 0.791% | 4.327 | 4.564 | 1.055 | toggle-only |
| MMU | 4233 | 0.697% | 5.232 | 4.018 | 0.768 | toggle-only |
| interrupt | 3563 | 0.625% | 5.213 | 3.861 | 0.741 | toggle-only |
| csr | 2736 | 0.514% | 5.649 | 3.511 | 0.621 | toggle-only |
| cache | 8313 | 0.400% | 4.184 | 3.474 | 0.830 | toggle-only |
| debug | 39228 | 0.076% | 2.835 | 2.486 | 0.877 | idle-ish |

### idu — `total_bits = 18731`, `Pc(x_aq_idu_top)`

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 4.911% | 5.915 | 1.392 | 0.235 | toggle-only |
| conv_softmax | 1438221 | 2.539% | 4.313 | 2.101 | 0.487 | *** BOTH |
| ISA_FP | 71076 | 1.088% | 4.526 | 1.345 | 0.297 | toggle-only |
| ISA_THEAD | 21143 | 1.000% | 4.323 | 1.369 | 0.317 | toggle-only |
| ISA_INT | 30030 | 0.960% | 4.287 | 1.419 | 0.331 | toggle-only |
| ISA_LS | 83386 | 0.759% | 4.322 | 1.257 | 0.291 | toggle-only |
| exception | 12169 | 0.374% | 3.800 | 1.353 | 0.356 | toggle-only |
| MMU | 4233 | 0.282% | 3.606 | 1.341 | 0.372 | toggle-only |
| interrupt | 3563 | 0.262% | 3.652 | 1.333 | 0.365 | toggle-only |
| csr | 2736 | 0.207% | 3.508 | 1.315 | 0.375 | toggle-only |
| cache | 8313 | 0.117% | 2.914 | 1.224 | 0.420 | idle-ish |
| debug | 39228 | 0.063% | 4.145 | 0.990 | 0.239 | idle-ish |

### iu — `total_bits = 18995`, `Pc(x_aq_iu_top)`

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 2.293% | 2.096 | 0.847 | 0.404 | *** BOTH |
| conv_softmax | 1438221 | 1.127% | 1.489 | 0.735 | 0.494 | toggle-only |
| ISA_FP | 71076 | 0.520% | 1.132 | 0.475 | 0.420 | toggle-only |
| ISA_THEAD | 21143 | 0.468% | 1.132 | 0.536 | 0.474 | toggle-only |
| ISA_INT | 30030 | 0.467% | 1.218 | 0.688 | 0.565 | toggle-only |
| ISA_LS | 83386 | 0.348% | 1.087 | 0.442 | 0.407 | toggle-only |
| exception | 12169 | 0.193% | 1.030 | 0.305 | 0.296 | toggle-only |
| interrupt | 3563 | 0.142% | 1.013 | 0.307 | 0.303 | toggle-only |
| MMU | 4233 | 0.136% | 1.014 | 0.302 | 0.298 | toggle-only |
| csr | 2736 | 0.104% | 1.006 | 0.307 | 0.305 | idle-ish |
| cache | 8313 | 0.058% | 0.993 | 0.244 | 0.245 | idle-ish |
| debug | 39228 | 0.026% | 0.997 | 0.237 | 0.238 | idle-ish |

### vidu — `total_bits = 13716`, `Pc(x_aq_vidu_top)`

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| conv_softmax | 1438221 | 1.869% | 2.054 | 0.827 | 0.403 | toggle-only |
| coremark | 598030 | 0.884% | 1.951 | 0.458 | 0.235 | toggle-only |
| ISA_FP | 71076 | 0.518% | 1.647 | 0.515 | 0.313 | toggle-only |
| ISA_INT | 30030 | 0.163% | 1.582 | 0.439 | 0.277 | idle-ish |
| ISA_THEAD | 21143 | 0.156% | 1.581 | 0.442 | 0.279 | idle-ish |
| ISA_LS | 83386 | 0.107% | 1.568 | 0.427 | 0.272 | idle-ish |
| exception | 12169 | 0.092% | 1.550 | 0.419 | 0.270 | idle-ish |
| MMU | 4233 | 0.063% | 1.539 | 0.408 | 0.265 | idle-ish |
| interrupt | 3563 | 0.061% | 1.539 | 0.411 | 0.267 | idle-ish |
| csr | 2736 | 0.048% | 1.532 | 0.406 | 0.265 | idle-ish |
| debug | 39228 | 0.032% | 1.528 | 0.401 | 0.263 | idle-ish |
| cache | 8313 | 0.029% | 1.525 | 0.401 | 0.263 | idle-ish |

### rtu — `total_bits = 6802`, `Pc(x_aq_rtu_top)` (absolute power <0.6 mW; weak discrimination)

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 3.693% | 1.139 | 0.408 | 0.358 | toggle-only |
| conv_softmax | 1438221 | 1.609% | 0.845 | 0.445 | 0.527 | toggle-only |
| ISA_FP | 71076 | 0.730% | 0.532 | 0.383 | 0.719 | toggle-only |
| ISA_THEAD | 21143 | 0.700% | 0.544 | 0.404 | 0.743 | toggle-only |
| ISA_INT | 30030 | 0.677% | 0.550 | 0.392 | 0.712 | toggle-only |
| ISA_LS | 83386 | 0.557% | 0.508 | 0.365 | 0.718 | toggle-only |
| exception | 12169 | 0.293% | 0.718 | 0.448 | 0.623 | idle-ish |
| MMU | 4233 | 0.247% | 0.738 | 0.507 | 0.687 | idle-ish |
| interrupt | 3563 | 0.238% | 0.737 | 0.522 | 0.708 | idle-ish |
| csr | 2736 | 0.182% | 0.791 | 0.549 | 0.694 | idle-ish |
| cache | 8313 | 0.096% | 0.938 | 0.459 | 0.490 | idle-ish |
| debug | 39228 | 0.043% | 0.399 | 0.207 | 0.519 | idle-ish |

> rtu `pwr_cv` looks high (0.5–0.74) only because `pwr_mean` is tiny; the absolute
> `pwr_std` stays below ~0.55 mW and active vs. idle benchmarks overlap, so no
> benchmark produces a genuinely significant rtu power swing.

### vpu — `total_bits = 104567`, `Pc(x_aq_vpu_top)` (no conv_softmax; only ISA_FP active)

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| ISA_FP | 71076 | 0.186% | 4.858 | 1.656 | 0.341 | toggle-only |
| coremark | 598030 | 0.066% | 4.608 | 1.197 | 0.260 | toggle-only |
| ISA_THEAD | 21143 | 0.027% | 4.492 | 1.106 | 0.246 | toggle-only |
| ISA_INT | 30030 | 0.023% | 4.474 | 1.086 | 0.243 | toggle-only |
| ISA_LS | 83386 | 0.008% | 4.473 | 1.043 | 0.233 | idle-ish |
| exception | 12169 | 0.008% | 4.481 | 1.087 | 0.243 | idle-ish |
| interrupt | 3563 | 0.007% | 4.500 | 1.205 | 0.268 | idle-ish |
| MMU | 4233 | 0.007% | 4.496 | 1.178 | 0.262 | idle-ish |
| csr | 2736 | 0.006% | 4.508 | 1.251 | 0.278 | idle-ish |
| cache | 8313 | 0.003% | 4.483 | 1.110 | 0.248 | idle-ish |
| debug | 39228 | 0.002% | 4.469 | 1.054 | 0.236 | idle-ish |

> VPU has by far the largest `total_bits` (104567, wide vector datapaths), so its
> bit toggle rate is very low even when active — `ISA_FP` tops out at only 0.186%.
> `ISA_FP` is the one benchmark that truly loads the VPU FP/vector datapath (195
> flips/cyc, ~3× the next); `coremark` is a distant second. All other benchmarks
> leave the VPU near idle (≈ clock + control baseline), and even ISA_FP raises
> `pwr_std` only to 1.66 mW vs. the ~1.05 mW noise floor (`std_max/min = 1.6×`).

### cp0 — `total_bits = 18873`, `Pc(x_aq_cp0_top)` (per-cycle power is flat)

| bench | cycles | bit toggle rate | pwr_mean | pwr_std | pwr_cv | verdict |
|---|---|---|---|---|---|---|
| coremark | 598030 | 0.429% | 1.162 | 0.248 | 0.214 | toggle-only |
| ISA_FP | 71076 | 0.224% | 1.166 | 0.256 | 0.219 | toggle-only |
| conv_softmax | 1438221 | 0.217% | 1.165 | 0.249 | 0.214 | toggle-only |
| ISA_INT | 30030 | 0.085% | 1.159 | 0.254 | 0.219 | idle-ish |
| ISA_THEAD | 21143 | 0.084% | 1.159 | 0.250 | 0.215 | idle-ish |
| exception | 12169 | 0.070% | 1.166 | 0.268 | 0.230 | idle-ish |
| ISA_LS | 83386 | 0.065% | 1.153 | 0.249 | 0.216 | idle-ish |
| MMU | 4233 | 0.055% | 1.170 | 0.260 | 0.222 | idle-ish |
| csr | 2736 | 0.051% | 1.180 | 0.283 | 0.240 | idle-ish |
| interrupt | 3563 | 0.050% | 1.170 | 0.259 | 0.221 | idle-ish |
| cache | 8313 | 0.041% | 1.170 | 0.261 | 0.223 | idle-ish |
| debug | 39228 | 0.012% | 1.144 | 0.252 | 0.220 | idle-ish |

> cp0 `pwr_std` is ~0.25 mW for **every** benchmark (`std_max/min = 1.1×`): raising
> the toggle rate does not move cp0 power. Consistent with the earlier finding that
> the per-cycle `Pc(x_aq_cp0_top)` target is effectively unlearnable.
