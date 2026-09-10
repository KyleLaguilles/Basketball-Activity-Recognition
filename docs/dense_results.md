# Dense Per-Sample Prediction — Results, Seeds 1–3

Kyle Laguilles · ARCS Lab · 2026-09-09 · read-only (no run directory or source file modified)

Results for the dense intervention pre-registered at [`docs/prereg_dense.md`](prereg_dense.md).
Every number below was produced in this session by the repo's own analysis scripts run read-only
against the three dense run directories in §1. Section references of the form "prereg §N" point at
that document; its §5 anchors and §6 bars are reproduced here only where a result is compared
against them.

The pre-registration was committed at `3fdc8ed`, 2026-09-04 09:46 UTC. The earliest dense run
started 2026-09-04 16:04 UTC. No result below was available when the bars in prereg §6 and the
directional predictions in prereg §7 were written.

**Commands.**

```
python analysis/sample_level_f1.py --dense \
    --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-04-41_dense_b10_seed1 \
                  logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-33-27_dense_b10_seed2 \
                  logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-58-51_dense_b10_seed3

python analysis/boracle_duration_bias.py --dense --results_dir <one of the three dirs>   # x3
```

---

## 1. Run directories and certification

All three runs share `dense=true`, `dense_seq_len=500`, `dense_overlap=0.5`, `dense_min_seg=25`,
`sw_length=1.0`, `sw_overlap=50`, `epochs=40`, `batch_size=100`, `sqrt_inverse`, `bidirectional`,
`loso_subjects=[b512,a0da,4d70,ce9d,9bd4]`, differing only in `seed` and `name`.

| seed | dir under `logs/subset_specific/loso_G/inceptioncontext/` | npz naming |
|---|---|---|
| 1 | `2026-09-04_16-04-41_dense_b10_seed1` | `preds_<fold>_1.0_seed1.npz` |
| 2 | `2026-09-04_16-33-27_dense_b10_seed2` | `preds_<fold>_1.0_seed2.npz` |
| 3 | `2026-09-04_16-58-51_dense_b10_seed3` | `preds_<fold>_1.0_seed3.npz` |

Per-fold non-padding sample counts, identical across all three seeds and equal to the recon §3.3
table that prereg §4.1 requires as the verification condition:

| fold | 4d70 | 9bd4 | a0da | b512 | ce9d | total |
|---|---|---|---|---|---|---|
| samples | 126839 | 121401 | 122130 | 117586 | 126777 | 614733 |
| coverage | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

Coverage is 1.00000 on every fold: at `dense_min_seg=25` no segment is discarded, so the dense
path scores the whole stream. Ground truth is certified sample for sample against the raw label
stream rebuilt from `data/seam_map.json`, which is stricter than the windowed last-sample check.
Fold → subject resolution was correct from the filename token on all 15 fold-runs, but was
certified independently in each case rather than taken from the token.

---

## 2. Sample-level per-class F1 — prereg §4.1 primary metric

Pooled over 5 LOSO folds, 614,733 samples, scored against `labels_export.csv.gz → label_id`.
The baseline column is the windowed seed-1 anchor from prereg §5.

| class | seed 1 | seed 2 | seed 3 | 3-seed mean | baseline (seed 1) | Δ mean vs baseline |
|---|---|---|---|---|---|---|
| dribbling | 0.7261 | 0.7248 | 0.7141 | 0.7217 | 0.6612 | +0.0605 |
| shot | 0.3951 | 0.2722 | 0.2774 | 0.3149 | 0.2244 | +0.0905 |
| pass | 0.4421 | 0.4358 | 0.4393 | 0.4391 | 0.2676 | +0.1715 |
| **rebound** | **0.2346** | **0.2885** | **0.2194** | **0.2475** | **0.1092** | **+0.1383** |
| layup | 0.2952 | 0.2415 | 0.2009 | 0.2459 | 0.2372 | +0.0087 |
| walking | 0.8456 | 0.8521 | 0.8605 | 0.8527 | 0.7676 | +0.0851 |
| running | 0.8283 | 0.8280 | 0.8322 | 0.8295 | 0.7847 | +0.0448 |
| standing | 0.1958 | 0.2060 | 0.2177 | 0.2065 | 0.2128 | −0.0063 |
| sitting | 0.7929 | 0.8046 | 0.8304 | 0.8093 | 0.8602 | −0.0509 |
| **macro** | **0.5284** | **0.5171** | **0.5102** | **0.5186** | **0.4583** | **+0.0603** |

Against the 3-seed baseline means quoted in prereg §5 (rebound 0.1061, macro 0.4513): rebound
+0.1414, macro +0.0673.

The two denominators are not identical — the windowed path drops a trailing tail per fold, the
dense path drops segments shorter than `min_segment_len` (zero of them here) — so these columns
are read side by side rather than differenced automatically. Dense coverage is 1.00000 and the
windowed baseline's is within a few samples per fold, so the comparison is sound at this precision.

---

## 3. Sample-level precision and recall — prereg §4.2

| class | support | share | seed 1 P / R | seed 2 P / R | seed 3 P / R | baseline P / R |
|---|---|---|---|---|---|---|
| dribbling | 24543 | 0.0399 | 0.7064 / 0.7469 | 0.7191 / 0.7306 | 0.7223 / 0.7061 | 0.6270 / 0.6993 |
| shot | 2579 | 0.0042 | 0.4097 / 0.3815 | 0.3646 / 0.2171 | 0.4580 / 0.1989 | 0.2087 / 0.2427 |
| pass | 9708 | 0.0158 | 0.3638 / 0.5632 | 0.3667 / 0.5372 | 0.3844 / 0.5124 | 0.2371 / 0.3071 |
| **rebound** | 5381 | 0.0088 | **0.2808 / 0.2014** | **0.2736 / 0.3051** | **0.2775 / 0.1814** | **0.1882 / 0.0769** |
| layup | 4426 | 0.0072 | 0.5024 / 0.2090 | 0.5548 / 0.1543 | 0.4648 / 0.1281 | 0.3028 / 0.1950 |
| walking | 314864 | 0.5122 | 0.8710 / 0.8216 | 0.8766 / 0.8290 | 0.8798 / 0.8421 | 0.8497 / 0.6999 |
| running | 169355 | 0.2755 | 0.8250 / 0.8317 | 0.8184 / 0.8379 | 0.8146 / 0.8505 | 0.7964 / 0.7734 |
| standing | 27080 | 0.0441 | 0.1684 / 0.2338 | 0.1896 / 0.2256 | 0.2082 / 0.2280 | 0.1450 / 0.3994 |
| sitting | 56797 | 0.0924 | 0.7700 / 0.8172 | 0.7625 / 0.8517 | 0.7830 / 0.8838 | 0.7876 / 0.9475 |

Rebound recall is the quantity prereg §7 prediction 1 named explicitly ("baseline: 0.077"). It
rises to 0.2014 / 0.3051 / 0.1814, a 2.36–3.97x increase, and precision rises alongside it
(0.1882 → ~0.277), so the gain is not recall bought with false positives.

---

## 4. Per-class duration bias — prereg §4.2, BOracle interface

`est_sec` is a straight count of per-sample predictions; there is no expansion step between it and
the raw 50 Hz timeline. True durations are identical across seeds. Total covered duration
12294.66 s (3h24m55s); conservation is exact on all three seeds (est total − true total = +0.00).

| class | true_s | seed 1 est / ratio | seed 2 est / ratio | seed 3 est / ratio | mean ratio |
|---|---|---|---|---|---|
| dribbling | 490.86 | 518.98 / 1.0573 | 498.76 / 1.0161 | 479.80 / 0.9775 | 1.0170 |
| shot | 51.58 | 48.04 / 0.9314 | 30.72 / 0.5956 | 22.40 / 0.4343 | 0.6538 |
| pass | 194.16 | 300.62 / 1.5483 | 284.46 / 1.4651 | 258.76 / 1.3327 | 1.4487 |
| **rebound** | **107.62** | **77.20 / 0.7173** | **120.02 / 1.1152** | **70.34 / 0.6536** | **0.8287** |
| layup | 88.52 | 36.82 / 0.4160 | 24.62 / 0.2781 | 24.40 / 0.2756 | 0.3232 |
| walking | 6297.28 | 5940.48 / 0.9433 | 5955.26 / 0.9457 | 6027.02 / 0.9571 | 0.9487 |
| running | 3387.10 | 3414.70 / 1.0081 | 3467.62 / 1.0238 | 3536.56 / 1.0441 | 1.0253 |
| standing | 541.60 | 752.18 / 1.3888 | 644.36 / 1.1897 | 593.08 / 1.0951 | 1.2245 |
| sitting | 1135.94 | 1205.64 / 1.0614 | 1268.84 / 1.1170 | 1282.30 / 1.1288 | 1.1024 |

Rebound duration ratio against the prereg §5 BOracle anchor of 0.4088:

| seed | ratio | bias | direction | folds backing the pooled sign |
|---|---|---|---|---|
| 1 | 0.7173 | −28.27% | UNDER | 3/5 |
| 2 | 1.1152 | +11.52% | OVER | 3/5 |
| 3 | 0.6536 | −34.64% | UNDER | 4/5 |
| **mean** | **0.8287** | | | |

Explosive ball actions (shot, rebound, layup), the classes prereg §1 identifies as the
injury-relevant ones: under-estimated 3 of 3 on seeds 1 and 3, 2 of 3 on seed 2 (rebound flips to
over-estimated). Layup remains the worst-biased class on every seed (ratio 0.28–0.42).

---

## 5. Success bar evaluation — prereg §6

| bar | condition | observed | result |
|---|---|---|---|
| §6.1 primary | seed-1 sample-level rebound F1 ≥ 0.1392 | **0.2346** | **CLEARED** (+0.0954, 1.69x the bar) |
| §6.2 confirmation | 3-seed mean rebound F1 > 0.1361 | **0.2475** | **CLEARED** (+0.1114) |
| §6.3 decision date | a seed-1 result exists by 2026-09-22 | 2026-09-04 | met, 18 days early |

Both bars are cleared, so by prereg §6.2 the result is confirmed and prereg §8.1's failure branches
(Path B / TCN as the motivated next experiment) are not triggered. Dense prediction enters the
paper as a successful tested intervention.

---

## 6. Directional prediction scorecard — prereg §7

These were committed sign-only before any result was read.

| # | prediction | outcome | evidence |
|---|---|---|---|
| 1 | Rebound F1 improves; recall increases substantially | **HELD** | F1 0.1092 → 0.2475 mean; recall 0.0769 → 0.2293 mean |
| 2 | Layup and shot F1 move in the same direction as rebound | **HELD** | shot 0.2244 → 0.3149, layup 0.2372 → 0.2459, both up as rebound is |
| 3 | Running and walking F1 each change < 0.02 | **FALSIFIED** | walking +0.0851, running +0.0448 (mean vs baseline); both exceed the 0.02 band, walking by 4x |
| 4 | Rebound duration ratio improves toward 1.0 | **HELD** | 0.4088 → 0.8287 mean; \|1 − ratio\| falls from 0.591 to 0.171 |
| 5 | Macro F1 improves | **HELD** | 0.4583 → 0.5186 mean |

**Prediction 3 is the informative failure.** It was the control: prereg §7 reasoned that walking and
running are "already well-separated at sample level (F1 > 0.76)" and that "dense prediction changes
nothing about their signal availability." Both improved well outside the pre-registered band, and
walking's gain (+0.085) is larger than dribbling's, running's or layup's, and on a par
with shot's (+0.091) despite shot being one of the ball actions the mechanism was supposed to
single out. That means dense
prediction is not acting only on the diagnosed mechanism — the window's exclusion of the run-in
transition, which prereg §1 argues is specific to transition-heavy ball actions. Some of the gain is
a general effect of per-sample labeling and the added temporal context, and it lands on classes the
pre-registration expected to be inert.

This does not affect the §6 bars, which are rebound-only. It affects the *attribution*: the paper
cannot claim the improvement isolates the windowing artifact without accounting for why the control
classes moved. The `--no_bilstm` ablation described at the end of prereg §12 is the direct test —
it separates dense labeling from the added BiLSTM capacity, which is prereg §9.5's named confound
and the most likely explanation for a broad, non-class-specific gain. Four such ablation runs exist
in `logs/` (`2026-09-08_*_ablation_no_bilstm_seed{1,2,3}`) and are not analysed here.

---

## 7. Observations not pre-registered

These fell out of the runs and are recorded because they qualify how §4 and §5 should be read.
None of them is a pre-registered outcome.

**Rebound's duration bias is not sign-stable.** It flips across seeds (UNDER, OVER, UNDER) and
within every seed: only 3/5, 3/5 and 4/5 folds back the pooled direction. The mean ratio of 0.8287
is therefore a pooled average over a quantity whose per-subject sign is not consistent, and it
should not be quoted as "dense under-estimates rebound by 17%". `boracle_duration_bias.py` flags
this itself with `!` markers. The improvement over the 0.4088 baseline anchor is real in magnitude;
its direction is not a property of the model.

**The explosive classes are seed-sensitive.** Shot swings 0.3951 / 0.2722 / 0.2774 and layup
declines monotonically 0.2952 / 0.2415 / 0.2009. At supports of 2,579 and 4,426 samples (0.42% and
0.72% of the stream) this spread is expected, but it means seed-1 figures for these two classes
overstate the typical run — seed 1 is the best seed for both. Quote 3-seed means for shot and layup.

**Two classes regressed.** Standing (−0.0063) and sitting (−0.0509) are slightly worse than the
windowed baseline. Standing was already the weakest class in both paths (F1 ≈ 0.21). Sitting's
decline comes from precision holding while recall drops from 0.9475 to ~0.85.

**The prereg §4.3 window-level bridge metric was not computed.** That section asks for dense
predictions to be re-windowed by majority vote and scored against window-level truth, as a bridge
to the existing evidence chain. `sample_level_f1.py --dense` does not implement it — it reports
"[3]/[4] WINDOW-LEVEL TABLES -- not applicable to a dense run" and treats table [2] as the whole
result. The window-level anchors in prereg §5 (rebound 0.1969 mean, walking 0.7741, macro 0.5273)
therefore have no dense counterpart yet. §4.3 states the bridge metric "is not used for any
decision", so no bar depends on it, but it remains outstanding if the paper wants that comparison.

---

Provenance: `analysis/sample_level_f1.py` and `analysis/boracle_duration_bias.py`, both `--dense`,
run read-only against the three directories in §1 on 2026-09-09. Baseline anchors quoted from
prereg §5, which sources them from `docs/recon_dense_report.md` §5.4. This document contains
results only; the pre-registration it reports against is unmodified apart from a pointer line.
