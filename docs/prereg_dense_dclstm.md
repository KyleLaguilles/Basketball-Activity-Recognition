# Pre-Registration — Dense Per-Sample Labeling on DeepConvLSTM

Kyle Laguilles · ARCS Lab · 2026-10-02

Commit this file before any 24-fold run of the configuration it describes is launched. No
result may be read before it is committed and pushed.

This document is scoped to DeepConvLSTM. `docs/prereg_dense.md` (2026-09-04) pre-registered
dense prediction on InceptionContext only — its §2 names `InceptionContext.py` line by line
and its folds are the 5-subject placeholder set — so it does not cover this experiment and
is not amended by this one.

---

## 1. Question

Does dense per-sample labeling lift rebound F1 on a second backbone, with no added
parameters?

The InceptionContext result is confounded on two axes at once. Its dense head replaced the
windowing abstraction *and* appended a new BiLSTM (`prereg_dense.md` §9.5 names this as a
confound). So "dense labeling works" and "the extra recurrent capacity works" are not
separable from that run. DeepConvLSTM separates them: its dense path adds no layer and no
parameter. If rebound still lifts, the gain is attributable to the labeling scheme and the
longer input, not to head capacity, and it transfers across feature extractors.

---

## 2. Anchor — DeepConvLSTM windowed, 24-fold

Three finished seeds, scored 2026-10-02 with `analysis/sample_level_f1.py` (window
predictions expanded onto the raw 50 Hz timeline, last-window-wins), 55,048 windows →
1,376,800 covered samples per seed, coverage 0.99975:

| seed | rebound F1 | macro F1 |
|---|---|---|
| 1 | 0.0928 | 0.3941 |
| 2 | 0.0834 | 0.4013 |
| 3 | 0.0792 | 0.3975 |
| **mean** | **0.0851** | **0.3976** |

Run directories, under `logs/subset_specific/loso_G/deepconvlstm/`:
`2026-09-23_16-09-31_24fold_deepconvlstm_seed1`,
`2026-09-23_16-47-05_24fold_deepconvlstm_seed2`,
`2026-09-23_17-25-32_24fold_deepconvlstm_seed3`. Launched by
`scripts/run_24fold_deepconvlstm.sh`.

Full per-class anchor, seed 1 / 2 / 3: dribbling 0.3955 / 0.4015 / 0.4193, shot 0.1376 /
0.1297 / 0.1389, pass 0.1883 / 0.1887 / 0.1886, rebound 0.0928 / 0.0834 / 0.0792, layup
0.1868 / 0.1877 / 0.1943, walking 0.8337 / 0.8360 / 0.8337, running 0.7480 / 0.7481 /
0.7476, standing 0.2264 / 0.2657 / 0.2240, sitting 0.7376 / 0.7707 / 0.7517.

For orientation, the same 24-fold comparison on InceptionContext (same scorer, same day):
windowed rebound 0.1304 / 0.1222 / 0.1418 (mean 0.1315), macro mean 0.4580; dense rebound
0.2450 / 0.2558 / 0.2577 (mean 0.2528), macro mean 0.5346. That is **Δrebound +0.1213** and
**Δmacro +0.0766** from windowing to dense on that backbone, and it is the reference the
secondary prediction in §6 is stated against.

DeepConvLSTM is the weaker backbone at every level: its windowed rebound (0.0851) sits
below InceptionContext's windowed rebound (0.1315), and its windowed macro (0.3976) below
that one's (0.4580). The bar in §5 is set against DeepConvLSTM's own anchor, not against
InceptionContext's.

---

## 3. Intervention

`--dense` on `--network deepconvlstm`, launched by `scripts/run_24fold_dense_dclstm.sh`
(committed as `1dffa4f`), against the dense head committed as `e919720`.

What changes in the model (`src/model/DeepConvLSTM.py`):

1. **Same-padding, dense branch only.** The four `Conv2d((filter_width, 1))` layers are
   unpadded on the windowed path and shrink the time axis by `4 × (filter_width − 1)` = 40
   samples at `filter_width 11`, which would turn a 500-sample sequence into 460. Under
   `dense=True` they get centered padding `((k−1)//2, 0)`. An even kernel cannot be
   centered and is rejected in the constructor. The forward pass raises if T is not
   preserved.
2. **Per-timestep classifier.** The windowed path reduces the LSTM output to one vector with
   `x[-1]` and classifies that. The dense branch applies the same dropout and the same
   `Linear(128, 9)` at every timestep, returning `(batch, classes, T)` — the layout
   `src/model/train.py` already expects for the dense loss and argmax.

What does **not** change: the LSTM is the windowed baseline's own — **unidirectional, 128
units, 1 layer**. No layer is added. **Both modes have exactly 302,153 trainable parameters
and identical `state_dict` keys.** This is the property that makes the experiment a clean
test of the labeling scheme: unlike the InceptionContext dense head, nothing here can be
explained by added capacity.

Also held fixed, from the windowed anchor's own launch script: 24 participant folds, 40
epochs, Adam at lr 1e-4 with `step_lr` (step 10, decay 0.9), `weight_decay` 1e-6,
`xavier_normal` init, `cross_entropy` with `sqrt_inverse` class weights, no augmentation, no
subsampling, no normalization, `use_deterministic_algorithms(True)`.

**Two keys differ from the windowed anchor**, plus `name`:

| key | windowed | dense |
|---|---|---|
| `dense` | `false` | `true` |
| `batch_size` | `100` | `10` |

Verified by replaying both launch scripts' argv through `main.py`'s own argparse and dumping
`vars(args)` exactly as `main.py:157-159` writes `cfg.txt`; the replayed windowed cfg is
byte-identical to the finished seed-1 run's real `cfg.txt`, so this diff is exact and
nothing else moves.

`batch_size` 10 is the same change the InceptionContext windowed→dense pair made, so the two
backbones' windowed→dense comparisons differ by the same keys. It is not cosmetic: the one
`--no_bilstm` run left at `batch_size 100` (`2026-09-08_21-34-40`) scored rebound 0.1428 /
macro 0.3708 against 0.2657 / 0.4708 for the otherwise identical `batch_size 10` run. Dense
mode is strongly batch-size sensitive, and 10 is the setting every dense run to date used.

**This is a confound and is declared as one.** Rebound could move because of the labeling,
because of the 10-second input, or because of the smaller batch. The design rationale in §4
and the bar in §5 do not pretend otherwise; §8 says what would isolate it.

---

## 4. Design rationale — why a unidirectional head

The dense head keeps the windowed baseline's LSTM direction. Each timestep's prediction
therefore sees only its own past, plus ±20 samples of lookahead from the centered conv
receptive field (4 layers × kernel 11 → 41 samples wide). The InceptionContext dense head is
bidirectional by construction (`InceptionContext.py:199-206`), so this is *not* an
architecturally matched replication of that result.

The evidence that directionality is not required for the rebound effect is the
InceptionContext `--no_bilstm` ablation, which classifies each timestep straight off the
unidirectional GRU (`InceptionContext.py:168` applied at `:221`, head at `:197`) and is
therefore causal in the same sense, under the same stitching rule:

| 5-fold, old keying | rebound F1 | macro F1 |
|---|---|---|
| windowed anchor (`prereg_dense.md` §5) | 0.1061 | 0.4513 |
| dense + BiLSTM | 0.2475 | 0.5186 |
| **dense, `--no_bilstm` (causal)** | **0.2602** | **0.4680** |

Per-seed `--no_bilstm`: rebound 0.2657 / 0.2691 / 0.2459, macro 0.4708 / 0.4729 / 0.4601.

**These are 5-fold numbers and are not comparable to the 24-fold anchors in §2.** They come
from `2026-09-08_22-05-05_ablation_no_bilstm_seed1`, `..._22-26-53_..._seed2` and
`..._22-44-19_..._seed3`, which predate the `<id>_<eu|na>` re-keying: their five folds are
bare 4-hex ids, which merge EU and NA recordings of different people. `sample_level_f1.py
--dense` refuses them for that reason, so the figures above were pooled from the npz arrays
directly and their `y_true` was not re-certified against the current `labels_export.csv.gz`
(it was certified against the loader's own per-sample stream at run time,
`validation.py:425-429`). They are cited as a *direction-of-effect* argument only, never as
an anchor. `docs/dense_results.md:176-179` records that these runs exist and were left
unanalysed.

Read that way, the causal head gave up macro F1 relative to the bidirectional one (−0.0506,
concentrated in layup and dribbling) while rebound did not suffer at all. Rebound's
discriminative signal is the run-in that *precedes* the event, so a causal head has it; the
classes that lost ground are the ones using right-context smoothing.

---

## 5. Success bar

Committed before any result is read. Both conditions must hold:

1. **3-seed mean sample-level rebound F1 ≥ 0.135.**
2. **Every individual seed above 0.0928** — the best of the three windowed seeds, so no seed
   may merely overlap the anchor's range.

The first is a +0.050 absolute, +59% relative move on the 0.0851 mean. The second exists
because the anchor's own seed spread is 0.0792–0.0928 — −7% to +9% around its mean — and rebound is
0.88% of samples: a mean that clears the bar on the strength of one lucky seed would not be
evidence of anything.

Scoring: `analysis/sample_level_f1.py --dense`, pooled over all 24 folds, against
`labels_export.csv.gz → label`. Primary metric is sample-level per-class F1. No window-level
bridge metric is used for the decision.

Seeds 1-3 are launched together; this is a confirmation of a mechanism already observed on
another backbone, not a screen, so there is no seed-1 gate.

---

## 6. Secondary prediction

**Macro F1 will gain less than InceptionContext's dense run did: Δmacro < +0.0766** (§2).

Reason: a causal head has no right-context smoothing. The classes that gained most from
InceptionContext's bidirectional dense head are the sustained locomotion and ball-handling
classes, where both sides of a timestep inform its label; the `--no_bilstm` ablation in §4
cost −0.0506 macro relative to the bidirectional dense head while leaving rebound intact.
DeepConvLSTM's dense head is causal by the same argument, so the macro gain should be the
smaller part of the effect even if rebound moves fully.

Directional, sign only. Macro F1 is not a success condition.

---

## 7. Known confound — stitching position

Overlapping sequences predict the same sample more than once, and
`stitch_dense_predictions` (`validation.py:120-164`) resolves this last-sequence-wins:
sequences are assigned in emission order, so the last one covering a sample sets it. At
`seq_len 500` / `overlap 0.5` (step 250), the winning offset inside the sequence *is* the
amount of history a causal head had for that prediction.

Replaying the cutting plan and that cursor walk over all 24 folds (1,377,145 samples):

| within-sequence position | share of scored samples | history available |
|---|---|---|
| 0–49 | **19.43 %** | under 1 second |
| 50–249 | 73.16 % | 1–5 seconds |
| 250–499 | 7.41 % | over 5 seconds |

Median winning position 131 samples (2.6 s); 92.6 % of predictions have under 5 s of
history. The share above 250 tracks recording fragmentation — 1.1–3.8 % on the long EU
recordings, 10–24 % on the chopped-up NA ones — because positions ≥ 250 win only in a
segment's right-aligned tail sequence and in padded short segments.

So **roughly a fifth of all scored predictions are made with less history than the 1-second
windowed baseline gets**, and the stitch systematically hands each prediction more future
than past — which a causal head cannot use. This penalizes the configuration under test
relative to a bidirectional one and relative to what the architecture could do with
center-cropped scoring. It is declared, not corrected: the stitching rule is the convention
every dense run and `sample_level_f1.py` already share, and changing it here would break
comparability with the InceptionContext dense results. If this configuration clears the bar
*despite* the penalty, the conclusion is only strengthened.

---

## 8. If the bar is missed

**Launch exactly one additional run before interpreting the result: seed 1, bidirectional
dense LSTM, everything else identical.**

The reason is §7. A miss has two incompatible readings — "dense labeling does not transfer
to this backbone" and "a causal head cannot exploit a stitching rule that gives 19% of its
predictions under a second of history" — and the 3-seed unidirectional result cannot tell
them apart. One bidirectional seed does: if rebound lifts with direction as the only change,
the labeling scheme transferred and the causal head was the limitation; if it stays flat,
the backbone is implicated and the InceptionContext gain was partly its added BiLSTM
capacity after all. Either way the finding is reportable, and either way it is a different
claim from the one a bare miss would support.

That run requires a code change (the head is unidirectional by construction) and is **not**
pre-registered as primary. It is a disambiguation step, not a second attempt at the bar: its
result does not retroactively satisfy §5.

No other follow-up is launched on a miss before this one is read.

---

## 9. Implementation status

The dense head is committed as **`e919720`**, the launch script as **`1dffa4f`**. Verified
before this file was written, one fold (`05d8_eu` — LabelEncoder code 0, so the single
global `seed_torch` at `main.py:167` leaves the same RNG state as a 24-fold run's first
fold), seed 1:

- **Windowed regression.** DeepConvLSTM windowed reproduces
  `2026-09-23_16-09-31_24fold_deepconvlstm_seed1` exactly: `y_pred` and `y_true`
  byte-identical over 2,469 windows, `best_epoch` 30 and `best_macro_f1`
  0.369059078710778 to the last digit.
- **InceptionContext dense regression.** Reproduces
  `2026-09-11_12-06-12_24fold_dense_seed1` exactly: byte-identical over 61,755 samples,
  `best_epoch` 39, `best_macro_f1` 0.49206360849123076.
- **Dense DeepConvLSTM smoke** (2 epochs). Logits `(batch, 9, 500)`; npz holds 1-D int64
  arrays of 61,755 samples, exactly the loader's kept rows for the fold, each covered once,
  labels in 0–8 with no `ignore_index` leakage; `sample_level_f1.py --dense` scores it at
  coverage 1.00000, certified sample-for-sample against the raw label stream.
- **Parameters:** windowed 302,153, dense 302,153.

The sequence loader, per-epoch stitching, class-weight derivation (per-sample counts via
`train.py:389-390`), npz format and the analysis chain are unchanged and network-agnostic.
`--no_bilstm` is hard-rejected for this network: it ablates the InceptionContext head's
BiLSTM and has no meaning for a head that reuses the windowed path's own LSTM.

**Decision date: _to be set by the author before launch._**

---

## 10. Files

New: `docs/prereg_dense_dclstm.md` (this file); `scripts/run_24fold_dense_dclstm.sh`
(`1dffa4f`).

Modified by `e919720`: `src/model/DeepConvLSTM.py` (dense branch), `src/model/validation.py`
(passes `dense` at the LOSO construction site), `src/main.py` (`DENSE_NETWORKS`, the
`--no_bilstm` gate).

Not modified: `src/data_processing/sequence_loader.py`, `src/data_processing/sliding_window.py`,
`src/model/train.py`, `src/model/InceptionContext.py`, `analysis/*`.

Provenance: anchors scored 2026-10-02 from the run directories named in §2 and §4;
InceptionContext 24-fold figures from `2026-09-11_18-13-22/19-25-31/20-37-56_24fold_windowed_seed{1,2,3}`
and `2026-09-11_12-06-12/14-07-46/16-09-07_24fold_dense_seed{1,2,3}`. All scoring read-only.
