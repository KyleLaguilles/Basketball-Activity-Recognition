Pre-Registration — Dense Per-Sample Prediction (Path A)

Kyle Laguilles · ARCS Lab · 2026-09-04

Commit this file at docs/prereg_dense.md before any training run is launched. No result may be read before this file is committed and pushed.

1. Motivation

The six-analysis diagnostic chain (v6 §1) concluded that rebound F1 ≈ 0.20 (window-level) is a signal-availability ceiling at 1-second windows. The discriminative signal — the run-in transition preceding a rebound event — lives in the portion the window excludes. The window-length sweep (v6 §7) returned a decisive negative: uniform lengthening degrades rebound monotonically because (a) longer windows dilute purity and (b) the last-sample labeling rule penalizes short events inside long windows.

Dense per-sample prediction removes both mechanisms. Instead of window → single label, the model predicts a class for every timestep in a long sequence, trained against the raw per-sample label stream. There is no window label to dilute, no last-sample rule, no purity problem, and the receptive field of each prediction spans the run-in transition natively. Sample-level classification is duration estimation, so this directly optimizes the metric BOracle consumes.

This experiment tests one variable: the windowing abstraction. The feature extractor (InceptionContext) and the training protocol (folds, loss, weighting, optimizer) are held fixed.

2. Architecture
2.1 What changes

InceptionContext-dense (Path A). The existing InceptionContext stem (four parallel Conv2d branches with kernels 1, 3, 5, 11; same-padding; channel-last reshape; intra-window GRU) already preserves the time dimension T through all layers up to InceptionContext.py:180, producing (B, T, 128). Two lines at :181-182 — attention-weighted pooling — collapse T to a single vector. The dense path branches at :180, skips the pooling, and appends:

A new BiLSTM on the time axis: nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True) → (B, T, 256).
A per-timestep linear head: nn.Linear(256, 9) → (B, T, 9), permuted to (B, 9, T) for nn.CrossEntropyLoss.

The old attention pooling (:181-182) and batch-as-sequence context_lstm (:154-160, :185-186) remain in the file but are bypassed when --dense is set.

2.2 What is held fixed
Component	Status	Citation
InceptionContext branches (kernels 1/3/5/11, same-padding)	Unchanged	InceptionContext.py:46-139
Intra-window GRU (128 units)	Unchanged (now intra-sequence GRU)	InceptionContext.py:143-147, :180
Fold structure (loso_G, 5 subjects)	Unchanged	validation.py:115-122
Loss function	nn.CrossEntropyLoss with sqrt_inverse weights	train.py:210, :413-432
Optimizer / scheduler	Adam, same LR schedule	train.py:224-268
Epochs	40	main.py default
Seed-1 screen	Same protocol	—
Augmentation	None	—
Deterministic training	use_deterministic_algorithms(True)	torchutils.py:15-33
2.3 Rationale for Path A before TCN (Path B)

Path A isolates the diagnosed variable (windowing) by changing one thing: per-window label → per-timestep label. Same features, same loss, same folds. If it works, the paper says "the fixed-window abstraction was the problem." If it fails, the feature extractor is implicated and a TCN (new architecture, new receptive-field knob) becomes the natural next step — but the decomposition is necessary for the paper either way. A reviewer will ask whether the windowing or the features were the bottleneck; Path A answers that.

3. Data handling
3.1 Target array

The raw per-sample label stream: train[:, -1] as returned by load_dataset('subset_specific', 'loso_G'), dtype uint8, classes 0–8 (rebound = 3). This is byte-identical to labels_export.csv.gz → label_id, which sample_level_f1.py:353 and boracle_duration_bias.py:159 already use as ground truth. It removes the 10.61 % (65,190-sample) last-sample labeling artifact measured in the recon (§5.2).

3.2 Seam-aware sequence cutting

The train array contains 749 unmarked discontinuities (739 intra-recording splices from row deletion at data_creation.py:46,69 + 10 EU/US recording joins), producing 763 contiguous segments. 615 of these are shorter than 1500 samples (30 s). Raw timestamps confirm 50 Hz holds perfectly; all gaps are created by row deletion, not sensor dropout.

Seam map. A one-time replay of data_creation.py's filters on data/raw/*.csv produces a mapping (subject_code, segment_idx) → (start_row, end_row) in train-array coordinates. This is computed once, serialized to data/seam_map.json, and version-controlled. Verification: total sample count = 1,377,145; segment count = 763; per-subject sample totals match §2.1 of the recon.

Sequence cutting rule.

For segments ≥ 500 samples (10 s): cut into fixed-length sequences of 500 samples with 50 % overlap (step = 250). The last sequence in a segment is right-aligned (may overlap more than 50 % with the penultimate).
For segments 50–499 samples: keep as-is, zero-pad to 500 on the right, mark padded positions with label = −100 (ignore_index for nn.CrossEntropyLoss).
For segments < 50 samples (1 s): discard. These carry almost no interior context and are dominated by boundary artifacts. (Recon §2: shortest segment = 31 samples; discarding segments < 50 removes ≤ ~1 % of total samples — exact count to be verified at implementation.)

No sequence crosses a seam. The 1-s baseline already straddles seams silently; the dense path is cleaner.

3.3 Estimated sequence counts per fold

Naive estimate from the recon (10 s / 50 % overlap on the full subject streams, ignoring seams): ~1650 training sequences, ~165 validation sequences per fold. Seam-aware cutting on segments ≥ 500 will produce fewer sequences from fragmented US recordings but more from long EU segments. Exact counts to be reported after the seam map is built — if any fold has < 100 validation sequences, the estimate is wrong enough to revisit the sequence length.

3.4 Class weights

sqrt_inverse weights computed on flattened per-sample training labels ((N_train_samples,)), using the same compute_class_weight → sqrt → mean-normalize path at train.py:413-421. Padding samples (label = −100) are excluded from the count. The recon (§6.2) showed sample-level and window-level class shares agree within 1 %, so the weight shift is negligible — but weights are recomputed from the correct denominator, not carried over.

3.5 Normalization

None. Raw accelerometer g values are fed directly, exactly as the windowed path does. No scaling or standardization exists anywhere in the current pipeline (recon §4.1). The InceptionContext.py:14-15 docstring claiming z-scored inputs is false and should not be relied on.

4. Evaluation
4.1 Primary metric

Sample-level F1 per class, pooled over 5 folds, scored against the raw per-sample label stream (labels_export.csv.gz → label_id). This is what sample_level_f1.py:353 already uses.

For the dense path, no expansion step is needed: the model's output is already per-sample. The y_pred / y_true arrays in the npz are 1-D, length = number of non-padding validation samples for that fold, in raw-stream order. A --dense flag in sample_level_f1.py skips resolve_fold certification (which requires window counts) and expand_last_window_wins, scoring the arrays directly. Verification: the non-padding sample count per fold must equal the validation sample counts in the recon §3.3 table (126839 / 121401 / 122130 / 117586 / 126777).

4.2 Secondary metrics
Per-class precision and recall at sample level (same script).
BOracle duration bias via boracle_duration_bias.py, adapted for dense predictions: the "window_proxy" legs become undefined; the script scores est_sec = (y_pred == c).sum() / 50 directly against true_sec = (y_true == c).sum() / 50 per game span. Rebound duration ratio est/true is the target (baseline: 0.4088, i.e. −59.12 %).
Boundary-tolerant F1 (new, secondary): ignore ±5 samples (~100 ms) around each label transition in the raw stream. Annotation boundaries carry the most label noise; this metric isolates interior performance. Report alongside the primary metric but do not use for the success bar.
4.3 Window-level diagnostic (not a success metric)

For comparability with the six-analysis chain, also report window-level F1 by applying the standard 1-s / 50 %-overlap windowing to the dense predictions (majority vote within each window) and scoring against the window-level ground truth. This is strictly a bridge metric for interpreting the result against the existing evidence chain. It is not used for any decision.

5. Baseline anchors

From the recon (§5.4), seed 1, pooled 5 folds, 614,700 covered samples:

class	sample-level F1	precision	recall
dribbling	0.6612	0.6270	0.6993
shot	0.2244	0.2087	0.2427
pass	0.2676	0.2371	0.3071
rebound	0.1092	0.1882	0.0769
layup	0.2372	0.3028	0.1950
walking	0.7676	0.8497	0.6999
running	0.7847	0.7964	0.7734
standing	0.2128	0.1450	0.3994
sitting	0.8602	0.7876	0.9475
macro	0.4583		

3-seed means: rebound 0.1061, macro 0.4513.

Window-level anchors (for the bridge metric only): rebound 0.2053 / 0.1879 / 0.1975 (mean 0.1969), walking 0.7926 / 0.7610 / 0.7686 (mean 0.7741), macro 0.5273 / 0.5143 / 0.5129.

BOracle anchor: rebound duration ratio est/true = 0.4088 (−59.12 %, UNDER in 5/5 folds).

6. Success bar
6.1 Primary bar

Seed-1 sample-level rebound F1 ≥ 0.1392 (= anchor 0.1092 + 0.03).

This is a 27 % relative improvement over the baseline's sample-level performance, which reflects the severity of the diagnosed ceiling: the baseline achieves 0.077 recall on rebound at sample level, meaning it correctly identifies fewer than 1 in 12 rebound samples.

6.2 Seed protocol
Seed 1 only for the initial run (5 folds × 40 epochs).
Seeds 2–3 launched only if seed 1 clears the bar.
The result is confirmed if the 3-seed mean also exceeds 0.1061 + 0.03 = 0.1361.
6.3 Decision date

September 22, 2026. If no seed-1 result exists by this date, dense prediction is written as future work and the paper ships on the six-analysis diagnostic chain (Option A). This date allows ~2.5 weeks for implementation + training + one iteration, with buffer before the end-of-September code-complete target.

7. Directional predictions

These are committed before any result is read. Each is directional (sign only), not quantitative.

Rebound F1 improves. The diagnosed mechanism (window excludes run-in transition) is removed. The model now sees the full preceding context at every timestep. Rebound recall should increase substantially (baseline: 0.077).
Layup and shot F1 move in the same direction as rebound. The onset finding (v5 §5) was consistent across all three ball-action classes, and running is the dominant confusion target for all three (running confusion share: rebound 44.9 %, shot 28.2 %, layup 42.5 %).
Running and walking F1 each change < 0.02. These classes are already well-separated at sample level (F1 > 0.76); dense prediction changes nothing about their signal availability.
Rebound duration bias (est/true) improves toward 1.0. The baseline ratio is 0.41. If rebound recall improves, the model identifies more rebound samples, and BOracle's per-game duration estimate moves toward truth.
Macro F1 improves. The windowed baseline's macro F1 is dragged down by ball-action classes; if predictions 1–3 hold, macro rises.
8. Failure interpretation (committed before launch)
8.1 If seed-1 rebound F1 < 0.1392 (bar not met)

The windowing abstraction was not the sole bottleneck. Two sub-cases:

Rebound F1 improves but misses the bar (0.1092 < F1 < 0.1392): the diagnosis is partially confirmed — removing windowing helps, but the InceptionContext feature extractor does not capture the run-in transition well enough. This motivates Path B (TCN with an explicit receptive-field knob) as the next experiment: the question becomes "how much context does rebound need?" with the receptive field as the pre-registrable variable.
Rebound F1 is flat or declines (F1 ≤ 0.1092): the run-in signal is not recoverable from wrist IMU at this label granularity by this feature extractor, even with full temporal context. This is a stronger diagnostic claim than v6's, and it upgrades the paper: "the signal-availability ceiling persists even when the diagnosed mechanism is removed." Path B (TCN) is still the natural next step — the feature extractor changes alongside the prediction scheme — but the paper can ship on the diagnostic chain without it.

In either sub-case, dense prediction goes into the paper as a tested intervention, not future work. The negative result is informative and publishable.

8.2 If implementation is not complete by September 22

Dense prediction is described as future work. The paper ships on the six-analysis chain plus the window-sweep negative. The architecture is described in the future-work section with enough detail to reproduce.

9. Confounds
9.1 Sequence length ↔ context

Same shape as the batch-size confound in the window sweep (v6 §7.4): a 500-sample sequence gives the BiLSTM 10 s of context per direction, vs. the windowed baseline's ~50 s (100 windows × 0.5 s step). If dense prediction wins, a sequence-length ablation (250 / 500 / 1000 / 1500 samples) is the conditional follow-up — not a gate.

9.2 Annotation boundary fuzz

The raw labels have hard transitions at annotation boundaries. Under windowing, each window spans many boundaries and the last-sample rule selects one; the 10.6 % artifact (recon §5.2) smooths some boundary noise by accident. Dense prediction exposes every boundary directly. The boundary-tolerant secondary metric (§4.2, ±5 samples) quantifies this effect. If the primary metric is flat but the boundary-tolerant metric improves, boundary noise is the explanation — worth reporting but not a reason to reject the approach.

9.3 Sample-level base rate

Rebound is 0.88 % of samples (5381 / 614700), making per-class F1 noisier than at window level (0.87 %, nearly identical). This is not a new confound — it is the same base rate the windowed path faces, just measured without the expansion step.

9.4 Seam handling

The 1-s baseline straddles seams; the dense path does not. This is a confound only in the sense that the dense path is cleaner — any improvement could partially reflect seam avoidance rather than the dense prediction itself. The size of this effect can be bounded post hoc by scoring only on samples > 500 away from any seam (interior samples), where neither path is affected.

9.5 New BiLSTM capacity

The dense head adds a new BiLSTM (128 × 2 = 256 hidden) that the windowed path does not have. The windowed path's Stage 2 context_lstm (128 × 2 bidirectional) operates across windows in a batch, not within a sequence — it is not an equivalent. If dense prediction wins, a control that replaces the new BiLSTM with a simple Linear(128, 9) (no temporal smoothing) isolates the contribution of the added capacity. This is a conditional follow-up, not a gate.

10. Conditional follow-ups (not pre-registered as primary)

Launched only if seed 1 clears the bar:

Seeds 2–3. Same config, different seed. Confirms the result is not a seed artifact.
Receptive-field ablation. Vary the new BiLSTM's hidden size or replace it with a 1D conv stack at different kernel sizes to test how much temporal smoothing the head needs. This is the mechanistic follow-up analogous to the k-NN re-run in the window sweep.
Sequence-length ablation. 250 / 500 / 1000 / 1500 samples, holding everything else fixed. Tests whether the context confound (§9.1) matters.
No-BiLSTM control. Linear(128, 9) directly on the GRU output — tests whether the InceptionContext features are already dense-capable without additional temporal smoothing.
BOracle recalibration. If rebound duration bias improves, re-run the full BOracle pipeline with the new predictions and report downstream injury-risk metric changes.
Path B (TCN). If Path A fails (§8.1), the TCN experiment becomes the next primary, with the receptive field as the pre-registrable variable and the Path A result as the ablation baseline.
11. Implementation checklist (verify before launch)
 data/seam_map.json committed. Total samples = 1,377,145; segments = 763.
 Sequence loader verified: non-padding sample count per fold = 126839 / 121401 / 122130 / 117586 / 126777 (recon §3.3).
 Discarded samples (segments < 50) counted and reported: must be < 1 % of total.
 Forward pass verified: (2, 500, 3) input → (2, 9, 500) output.
 --dense flag plumbed end-to-end: grep-verify from main.py argparse through model constructor, train loop (argmax axis at :533/579), npz dump, cfg.txt, and W&B config.
 sample_level_f1.py --dense branch: skips resolve_fold, scores y_pred / y_true directly, reports per-class F1 and macro.
 boracle_duration_bias.py dense branch: skips window_proxy legs, computes est_sec = (y_pred == c).sum() / 50 directly.
 cfg.txt diff against baseline: exactly the keys that should differ (dense, name, sequence-related params) and nothing else.
 Launch script committed at scripts/run_dense.sh with all flags explicit.
 This file committed and pushed before any run is launched.
 args.subjects order verified: the fold-token lookup at validation.py:117 is order-coincidence-dependent (recon §3.4); the dense loader must use the same expression.
 No uncommitted diffs on validation.py that remove args.pre_aug_train_counts (recon blocking surprise 5).
12. Files created or modified

New files:

docs/prereg_dense.md — this document
data/seam_map.json — segment boundaries
scripts/run_dense.sh — launch script
src/data_processing/sequence_loader.py — seam-aware sequence cutter (new)

Modified files (dense branch only, --dense flag):

main.py — --dense argparse flag
src/model/InceptionContext.py — dense head (bypass pooling at :181-182, add BiLSTM + linear)
validation.py — call sequence loader instead of apply_sliding_window when --dense
train.py — argmax axis, target flatten, npz dump format
analysis/sample_level_f1.py — --dense branch
analysis/boracle_duration_bias.py — --dense branch

Not modified:

src/data_processing/sliding_window.py — windowed path untouched
src/data_processing/preprocess_data.py — data loading untouched
train.py loss / weight / optimizer / scheduler — reused unchanged

Provenance: anchors from docs/recon_dense_report.md (2026-09-04), all reproduced against baseline seeds 1–3 in logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 (seed 1), 2026-07-17_11-15-00 (seed 2), 2026-07-17_11-15-03 (seed 3). Evaluation scripts analysis/sample_level_f1.py and analysis/boracle_duration_bias.py run read-only against these directories. This document committed at docs/prereg_dense.md before any dense training run is launched.

Hypothesis: the dense labeling, not the BiLSTM capacity, is the primary driver of rebound F1 improvement.
Success bar: seed-1 rebound F1 ≥ some threshold with --no_bilstm (you pick the number — even "above the windowed baseline of 0.1114" is a meaningful bar).
Failure interpretation: if rebound F1 drops back toward ~0.11, the BiLSTM's temporal smoothing is essential and the dense labeling alone is insufficient.
Results: seeds 1–3 dense results are reported separately at docs/dense_results.md (2026-09-09). This pre-registration is unmodified apart from this line.
