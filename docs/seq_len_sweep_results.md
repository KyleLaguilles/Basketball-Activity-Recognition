# Dense Sequence-Length Sweep — Results, seq_len ∈ {250, 500, 1000}, seed 1

Kyle Laguilles · ARCS Lab · 2026-09-10 · read-only (no run directory or source file modified)

Housekeeping pass over the two sequence-length sweep runs under `logs/subset_specific/loso_G/`,
run under the same protocol as the seeds 1–3 dense housekeeping reported at
[`docs/dense_results.md`](dense_results.md): both analysis scripts run read-only with `--dense`,
folds certified sample-for-sample, results reported against the existing anchors.

**This sweep has no pre-registration.** `docs/prereg_sequence_length_sweep.md` and
`scripts/run_seq_len_sweep.sh` both exist as 0-byte files. There are therefore no pre-committed
success bars and no directional predictions for this experiment, and nothing below is scored
against one. Every comparison here is post-hoc against the seq_len=500 dense runs, which were
themselves pre-registered at [`docs/prereg_dense.md`](prereg_dense.md). Read §5 before quoting any
of it.

**Commands.**

```
python analysis/sample_level_f1.py --dense \
    --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-09-09_23-33-00_sweep_seqlen250_seed1 \
                  logs/subset_specific/loso_G/inceptioncontext/2026-09-10_00-09-30_sweep_seqlen1000_seed1

python analysis/boracle_duration_bias.py --dense --results_dir <one of the two dirs>   # x2
```

---

## 1. Run directories and certification

| dense_seq_len | dir under `logs/subset_specific/loso_G/inceptioncontext/` | window equivalent | step |
|---|---|---|---|
| 250 | `2026-09-09_23-33-00_sweep_seqlen250_seed1` | 5.0 s | 125 |
| 500 (anchor) | `2026-09-04_16-04-41_dense_b10_seed1` | 10.0 s | 250 |
| 1000 | `2026-09-10_00-09-30_sweep_seqlen1000_seed1` | 20.0 s | 500 |

The two sweep `cfg.txt` files are byte-identical to the seq_len=500 seed-1 `cfg.txt` on every field
except `name` and `dense_seq_len`, plus a `no_bilstm: false` key the 500 run predates (the flag was
added at `c895bd4`; its default is the 500 run's behaviour). So `seed=1`, `dense=true`,
`dense_overlap=0.5`, `dense_min_seg=25`, `sw_length=1.0`, `sw_overlap=50`, `epochs=40`,
`batch_size=10`, `sqrt_inverse`, `bidirectional`, `loso_subjects=[b512,a0da,4d70,ce9d,9bd4]` are held
fixed across all three configs. Sequence length is the only manipulated variable.

Per-fold non-padding sample counts, identical across all three configs and equal to the recon §3.3
table that prereg §4.1 requires as the verification condition:

| fold | 4d70 | 9bd4 | a0da | b512 | ce9d | total |
|---|---|---|---|---|---|---|
| samples | 126839 | 121401 | 122130 | 117586 | 126777 | 614733 |
| coverage | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

Coverage is 1.00000 on every fold of both runs: at `dense_min_seg=25` no segment is discarded at any
sequence length, so all three configs score the identical 614,733-sample stream on a shared
denominator. Ground truth is certified sample for sample against the raw label stream rebuilt from
`data/seam_map.json`. Fold → subject resolution was correct from the filename token on all 10
fold-runs, but was certified independently in each case. Both scripts exited 0 with no warnings; both
`log.txt` files are clean of errors, NaNs and warnings.

---

## 2. Sample-level per-class F1

Pooled over 5 LOSO folds, 614,733 samples, scored against `labels_export.csv.gz → label_id`.
All three columns are seed 1 and share one denominator, so the deltas are exact.

| class | 250 (5.0 s) | **500 (10.0 s)** | 1000 (20.0 s) | Δ 250 vs 500 | Δ 1000 vs 500 |
|---|---|---|---|---|---|
| dribbling | 0.6976 | 0.7261 | 0.7175 | −0.0285 | −0.0086 |
| shot | 0.4535 | 0.3951 | 0.2082 | +0.0584 | −0.1869 |
| pass | 0.4366 | 0.4421 | 0.4360 | −0.0055 | −0.0061 |
| **rebound** | **0.3169** | **0.2346** | **0.2179** | **+0.0823** | **−0.0167** |
| layup | 0.2422 | 0.2952 | 0.2233 | −0.0530 | −0.0719 |
| walking | 0.8631 | 0.8456 | 0.8212 | +0.0175 | −0.0244 |
| running | 0.8289 | 0.8283 | 0.8284 | +0.0006 | +0.0001 |
| standing | 0.3097 | 0.1958 | 0.1928 | +0.1139 | −0.0030 |
| sitting | 0.8111 | 0.7929 | 0.7656 | +0.0182 | −0.0273 |
| **macro** | **0.5511** | **0.5284** | **0.4901** | **+0.0227** | **−0.0383** |

Direction is monotone on macro F1 across the three lengths: 0.5511 → 0.5284 → 0.4901. Shorter
sequences are better, and 250 is the best of the three on 6 of 9 classes and on macro.

**Read these against seed spread, not as point estimates.** Only seq_len=500 has three seeds; the
sweep runs are seed 1 only. Using the seed-1/2/3 range at 500 from `dense_results.md` §2 as the
noise band:

| class | 500 3-seed range | 250 | verdict | 1000 | verdict |
|---|---|---|---|---|---|
| shot | 0.2722 – 0.3951 | 0.4535 | above band | 0.2082 | below band |
| **rebound** | 0.2194 – 0.2885 | **0.3169** | **above band** | 0.2179 | marginally below (−0.0015) |
| layup | 0.2009 – 0.2952 | 0.2422 | inside band | 0.2233 | inside band |
| standing | 0.1958 – 0.2177 | 0.3097 | above band, 1.42x the top | 0.1928 | marginally below |
| walking | 0.8456 – 0.8605 | 0.8631 | just above | 0.8212 | below band |
| **macro** | 0.5102 – 0.5284 | **0.5511** | **above band** | 0.4901 | below band |

Rebound and macro at 250 sit outside the three-seed spread observed at 500, which is the strongest
statement a single seed supports. Layup's change is inside seed noise at both lengths and should not
be read as an effect. Shot and layup were flagged in `dense_results.md` §7 as the seed-sensitive
classes; their single-seed values here carry the least weight.

---

## 3. Sample-level precision and recall

| class | support | 250 P / R | 500 P / R | 1000 P / R |
|---|---|---|---|---|
| dribbling | 24543 | 0.6831 / 0.7127 | 0.7064 / 0.7469 | 0.7239 / 0.7112 |
| shot | 2579 | 0.5452 / 0.3881 | 0.4097 / 0.3815 | 0.4027 / 0.1404 |
| pass | 9708 | 0.3614 / 0.5514 | 0.3638 / 0.5632 | 0.3688 / 0.5331 |
| **rebound** | 5381 | **0.3332 / 0.3022** | **0.2808 / 0.2014** | **0.2687 / 0.1832** |
| layup | 4426 | 0.5072 / 0.1591 | 0.5024 / 0.2090 | 0.4288 / 0.1509 |
| walking | 314864 | 0.8718 / 0.8546 | 0.8710 / 0.8216 | 0.8641 / 0.7823 |
| running | 169355 | 0.8154 / 0.8429 | 0.8250 / 0.8317 | 0.8342 / 0.8226 |
| standing | 27080 | 0.3125 / 0.3069 | 0.1684 / 0.2338 | 0.1450 / 0.2873 |
| sitting | 56797 | 0.8175 / 0.8048 | 0.7700 / 0.8172 | 0.7187 / 0.8191 |

Rebound at 250 gains on both axes (precision 0.2808 → 0.3332, recall 0.2014 → 0.3022), so its F1
gain is not recall bought with false positives — the same pattern the dense intervention itself
showed against the windowed baseline. Standing's gain at 250 is likewise two-sided (precision
0.1684 → 0.3125). Shot at 1000 loses almost all of its recall (0.3815 → 0.1404) at roughly unchanged
precision: the long sequence suppresses the class rather than confusing it.

---

## 4. Per-class duration bias — BOracle interface

`est_sec` is a straight count of per-sample predictions. True durations are identical across all
three configs. Total covered duration 12294.66 s (3h24m55s); conservation is exact on both sweep runs
(est total − true total = +0.00 per fold and pooled).

| class | true_s | 250 est / ratio | 500 est / ratio | 1000 est / ratio |
|---|---|---|---|---|
| dribbling | 490.86 | 512.14 / 1.0434 | 518.98 / 1.0573 | 482.24 / 0.9824 |
| shot | 51.58 | 36.72 / 0.7119 | 48.04 / 0.9314 | 17.98 / 0.3486 |
| pass | 194.16 | 296.26 / 1.5259 | 300.62 / 1.5483 | 280.64 / 1.4454 |
| **rebound** | **107.62** | **97.60 / 0.9069** | **77.20 / 0.7173** | **73.40 / 0.6820** |
| layup | 88.52 | 27.76 / 0.3136 | 36.82 / 0.4160 | 31.16 / 0.3520 |
| walking | 6297.28 | 6173.02 / 0.9803 | 5940.48 / 0.9433 | 5701.78 / 0.9054 |
| running | 3387.10 | 3501.04 / 1.0336 | 3414.70 / 1.0081 | 3339.94 / 0.9861 |
| standing | 541.60 | 531.94 / 0.9822 | 752.18 / 1.3888 | 1072.90 / 1.9810 |
| sitting | 1135.94 | 1118.18 / 0.9844 | 1205.64 / 1.0614 | 1294.62 / 1.1397 |

Rebound's distance from unity, the prereg §7 prediction-4 quantity:

| seq_len | ratio | \|1 − ratio\| | direction | folds backing the pooled sign |
|---|---|---|---|---|
| 250 | 0.9069 | **0.093** | UNDER | 3/5 (flips) |
| 500 | 0.7173 | 0.283 | UNDER | 3/5 (flips) |
| 1000 | 0.6820 | 0.318 | UNDER | **5/5 (stable)** |

250 gives the closest rebound exposure estimate of any dense config measured so far — a 3.0x
reduction in absolute duration error against the seq_len=500 seed-1 anchor, and 6.4x against the
windowed baseline anchor of 0.4088 (prereg §5). Five of nine classes land within ±5% of true duration
at 250 (dribbling, walking, running, standing, sitting), with rebound the next closest at −9.3%;
at 500 only running is inside ±5%. Pass remains the worst-inflated class at every length
(+45% to +55%) and layup the worst-deflated (−58% to −69%); neither is a function of sequence length.

Two seq_len-specific observations on the sign-stability question that `dense_results.md` §7 raised:

- **At 1000, rebound's duration direction stops flipping.** All 5 folds under-estimate. This is the
  first dense config where rebound's bias sign is a property of the model rather than a pooled
  average — and it stabilises on the dangerous direction, at the config with the worst rebound
  exposure error. Sign stability here is not good news.
- **At 250, standing's bias collapses.** 1.3888 → 0.9822, alongside its F1 jump in §2. At 1000 it
  runs the other way to 1.9810, a 98% over-estimate: the long sequence is dumping the low-support
  static classes into standing.

Explosive ball actions (shot, rebound, layup), the classes prereg §1 identifies as injury-relevant,
are under-estimated 3 of 3 at both swept lengths — same as at 500. Sequence length changes the
magnitude of that bias, not its direction.

---

## 5. How far this goes

**No bars, no predictions.** The sweep's pre-registration file is empty, so nothing here is a
confirmed or falsified pre-committed result. The pattern in §2 and §4 — 250 better, 1000 worse,
monotone on macro — is a post-hoc observation on one seed. It is a reason to pre-register and run
seeds 2–3 at seq_len=250, not a result to quote as one.

**The seq_len=1000 run is plausibly under-trained, which confounds its numbers.** Best epochs are
39, 36, 39, 40, 39 out of a 40-epoch cap: four of five folds peak in the last two epochs and b512
peaks on the final epoch, so validation macro F1 was still climbing when training stopped. Compare
250 (36, 29, 36, 37, 33) and 500 (39, 25, 35, 35, 37), both of which have folds turning over well
inside the cap. Some part of the 1000 config's −0.0383 macro deficit is an epoch-budget artifact
rather than a sequence-length effect, and the honest read is that 1000 is not cleanly measured at
`epochs=40`. The 250-vs-500 comparison does not carry this problem.

**Best-epoch macro F1 ordering matches the sample-level ordering.** Mean over folds: 250 = 0.520,
500 = 0.484, 1000 = 0.457. That is the model's own windowed validation metric agreeing with the
sample-level pooled score on the direction, which is weak independent support that the 250 gain is
real and not an artifact of the scoring path.

**One correction to `dense_results.md` §1, unrelated to this sweep.** It records the seeds 1–3 dense
runs as `batch_size=100`; all three `cfg.txt` files say `batch_size=10`, consistent with the runs'
own `dense_b10` names. The sweep runs also use 10, so the comparison in this document is unaffected.

---

Provenance: `analysis/sample_level_f1.py` and `analysis/boracle_duration_bias.py`, both `--dense`,
run read-only against the two sweep directories in §1 on 2026-09-10. The seq_len=500 columns are the
seed-1 figures from `docs/dense_results.md` §2–4, and the baseline anchors are quoted from prereg §5.
This document contains results only; no run directory, source file or pre-registration was modified.
