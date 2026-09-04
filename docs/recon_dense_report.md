# Recon Report — Dense Per-Sample Prediction Path

Kyle Laguilles · ARCS Lab · 2026-09-04 · inspect-only (no source file modified; this report is the only file written)

Every count below was reproduced in this session by running the repo's own loader
(`src/data_processing/preprocess_data.py:load_dataset`, `sliding_window.py:apply_sliding_window`)
and the repo's own analysis scripts (`analysis/sample_level_f1.py`, `analysis/boracle_duration_bias.py`)
read-only, against the three baseline seed directories identified in §9. All §10 anchors reproduced
exactly; one checklist premise (§6, "share undercount") did not, and is explained there.

Baseline run directories (from `cfg.txt`, all: `seed=N sw=1.0/50 epochs=40 batch=100 sqrt_inverse bidirectional loso_subjects=[b512,a0da,4d70,ce9d,9bd4] subsample_fraction=1.0 no augmentation`):

| seed | dir under `logs/subset_specific/loso_G/inceptioncontext/` | npz naming |
|---|---|---|
| 1 | `2026-07-17_07-56-30` (name `lc_frac1.0`) | `preds_<fold>_1.0.npz` |
| 2 | `2026-07-17_11-15-00` (name `lc_frac1.0_seed2`) | `preds_<fold>_1.0.npz` |
| 3 | `2026-07-17_11-15-03` (name `lc_frac1.0_seed3`) | `preds_<fold>_1.0.npz` |
| 1 (re-run) | `2026-08-14_06-21-45_lc_frac1.0_ckpt` | `preds_<fold>_1.0_seed1.npz` — y_pred identical to seed-1 baseline on all 5 folds (agreement 1.0) |

---

## §1–§3 CHECKPOINT (read this first)

**The continuous per-sample label stream exists and survives up to the windowing call.** `load_dataset` returns
`train` of shape `(1377145, 5)` float32 = `[subject_code, acc_x, acc_y, acc_z, label]`, one row per 50 Hz
sample, and `cross_participant_cv` passes `train_data[:, :-1]` / `train_data[:, -1]` straight into
`apply_sliding_window` (`validation.py:126-138`). Nothing between the CSV and the windowing call aggregates,
resamples, normalizes, or regroups labels. The dense path can take `train_data` as-is.

The rest of the checklist therefore does not change, and §4–§9 follow. Three surprises do change the
*plan*, not the checklist — see the final "Blocking surprises" block:

1. **The stream is not contiguous.** Rows dropped at CSV-creation time (`not_labeled` / `jumping`, `data_creation.py:46,69`) leave **739 intra-recording splices** in the game data, and 10 subjects have an EU and a US recording concatenated under one subject code (**+10 seams**). 615 of the 763 resulting contiguous segments are shorter than 30 s (1500 samples). A 30-s sequence cutter keyed only on subject code will straddle seams constantly. No timestamps survive into the CSV, so seam positions must be recovered from `data/raw/*.csv`.
2. **There is no normalization anywhere in the pipeline** (raw g values go straight into the model), so §4's "which alternative" question is moot — but `InceptionContext.py:14-15` claims inputs are z-scored upstream, which is false.
3. **Fold tokens are matched against the unsorted `args.subjects` array** (`validation.py:117`), which only coincides with the LabelEncoder order because the drill CSV happens to be written in sorted filename order. It is correct today; the dense path must use the same lookup or the same tokens will pick different subjects.

---

## 1. The continuous label stream (before windowing)

**1.1 Objects at the last point before windowing.**
- `preprocess_data.py:35-38` loads the three CSVs with `header=None`; `:98` `X_g, y_g = preprocess_data(data_game, ...)`; `:112` `train = np.concatenate((X_g, np.expand_dims(y_g, axis=1).astype(np.uint8)), axis=1)`; `:113` `valid = None`.
- `preprocess_data.py:152` `X, y = data.iloc[:, :-1], adjust_labels(data.iloc[:, -1]).astype(int)`; `:160` `return X.astype(np.float32), y.astype(np.uint8)`.
- Reproduced: `train.shape == (1377145, 5)`, dtype `float32` (the uint8 label is upcast by the concatenate). `valid is None`.
- Example row `train[0] = [0., 1.83008, -1.0603, -0.04761, 6.]` → subject code 0 (`05d8`), acc_x/y/z in g, label 6 = running. Corresponding raw CSV row: `eu,expert,male,05d8,1.83008,-1.0603,-0.04761,running`.
- Windowing call: `validation.py:126-131` `X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1], sliding_window_size=args.sw_length, unit=args.sw_unit, sampling_rate=args.sampling_rate, sliding_window_overlap=args.sw_overlap)`; same for val at `:133-138`.

**1.2 Layout.** One concatenated 2-D array; recording membership is carried only by column 0 (subject code). Rows are in CSV order, which is raw-file order (`data_creation.py:26` `filenames = sorted(...)`, `:81-84` `pd.concat`). Verified: 14 contiguous subject blocks in row order, and 24 contiguous `(location, subject)` blocks: `eu_05d8, eu_0846, us_0846, eu_10f0, us_10f0, eu_2dd9, us_2dd9, eu_4991, eu_4d70, us_4d70, eu_9bd4, us_9bd4, eu_a0da, us_a0da, eu_ac59, us_ac59, eu_b512, us_b512, us_c6f3, eu_ce9d, us_ce9d, eu_e90f, eu_f2ad, us_f2ad`. **No index array, ID column, or list marks the EU/US boundary inside a subject** — location (column 0 of the CSV) is dropped at `preprocess_data.py:97` `data_game = data_game.iloc[:, 3:]`.

**1.3 Columns.** After `iloc[:, 3:]`: col 0 = subject code (metadata, int stored as float32), cols 1–3 = `acc_x, acc_y, acc_z` (sensor), col 4 = label. Column 0 is stripped after windowing at `validation.py:206` `X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]`, giving `nb_channels = 3` (`validation.py:216`). CSV columns 0–2 (location, skill, gender) are dropped at `:97` and never reach the model. There is **no timestamp column** anywhere after `data_creation.py:52` `sbj_data.drop(['timestamp', 'in/out'], axis=1)`.

**1.4 Label encoding.** `preprocess_data.py:175-184`: `void→0, dribbling→1, shot→2, pass→3, rebound→4, layup→5, walking→6, running→7, standing→8, sitting→9`; then `:155-156` `if has_void and not include_void: y -= 1` (always true for loso_G: `has_void = True` at `:32`, `include_void=False` default `main.py:37`). Effective 9-class map: `class_names = ['dribbling','shot','pass','rebound','layup','walking','running','standing','sitting']` (`:33`), **rebound = 3**, layup = 4. Same map is restated in `analysis/_loso_common.py:33-41` and certified against the npz by exact label-sequence match (`sample_level_f1.py:263-289`).

**1.5 Regrouping order.** UNCONFIRMED what "v4 label regrouping" refers to — no regrouping/merge/class_map logic exists in `src/` (grep for `regroup|relabel|merge_class|class_map|majority|mode(` finds only analysis scripts). The only label merging in the whole pipeline is at CSV-creation time, before any of `src/` runs: `data_creation.py:66-67` `mask = sbj_data["basketball"].eq("not_labeled"); sbj_data.loc[mask, "basketball"] = sbj_data.loc[mask, "locomotion"]` (locomotion fills unlabeled basketball), then `:69` drops `not_labeled` and `jumping`. This happens **before windowing** and is already baked into `hangtime_game_data.csv`, so the dense path inherits it with no re-application. If "v4 regrouping" is something else, it is not in this repo.

**1.6 Null / unlabeled samples.** None reach windowing. `preprocess_data.py:150` drops `void_class` rows, but the game CSV contains **0** such rows (label vocabulary in `data/hangtime_game_data.csv` col 8: exactly the 9 classes; counts dribbling 47583, shot 7568, pass 21511, rebound 10337, layup 8242, walking 724405, running 351798, standing 57595, sitting 148106 = 1,377,145). Unlabeled samples were removed at `data_creation.py:46` (`coarse != 'not_labeled'`) and `:69`, which is what creates the splices in §2. Windowing drops only the tail: `sliding_window.py:35` `while curr < len(data) - win_len:` (strict `<`) leaves 14/1/5/11/2 uncovered samples on the five validation subjects. **No `ignore_index` is needed for a dense loss on this stream**; one may still be wanted for padding if variable-length segments are batched (§2.3).

**1.7 Label aggregation.** `sliding_window.py:117` `output_y = [[i[-1]] for i in output_y]` is the only aggregation in `src/`. Majority/mode logic exists only in `analysis/_loso_common.py:188-205` (`window_majority`, `majority_vote_label`) and in `analysis/duration_mae.py:171` (`mode_filter` on predictions), none of which feed training.

## 2. Recording structure

**2.1 Recordings.** 24 raw files = 24 (subject, location) recordings covering 14 subject codes; 10 subjects have both EU and US recordings. Session/game ID: none survives (`data_creation.py:52,73`); the `coarse` column is used only to select `game` rows. Per-recording table (game rows only, reproduced from `data/raw/*.csv` by replicating `data_creation.py:46,66-69,73`; `n_game` matched the CSV block count for all 24):

| recording | subj | loc | samples | seconds | raw span (s) | splices (>25 ms) | max gap (s) | segments | min seg (samples) | segs < 1500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 05d8_eu | 05d8 | eu | 61755 | 1235.1 | 1433.3 | 6 | 195.5 | 7 | 48 | 1 |
| 0846_eu | 0846 | eu | 18684 | 373.7 | 379.3 | 3 | 4.0 | 4 | 830 | 1 |
| 0846_na | 0846 | us | 50885 | 1017.7 | 1334.5 | 97 | 20.6 | 98 | 63 | 92 |
| 10f0_eu | 10f0 | eu | 46161 | 923.2 | 1466.5 | 8 | 359.3 | 9 | 1022 | 3 |
| 10f0_na | 10f0 | us | 65399 | 1308.0 | 1334.7 | 19 | 22.1 | 20 | 72 | 13 |
| 2dd9_eu | 2dd9 | eu | 33465 | 669.3 | 670.9 | 4 | 0.8 | 5 | 768 | 2 |
| 2dd9_na | 2dd9 | us | 60820 | 1216.4 | 1329.3 | 41 | 13.4 | 42 | 75 | 35 |
| 4991_eu | 4991 | eu | 67411 | 1348.2 | 1502.8 | 2 | 154.5 | 3 | 1139 | 1 |
| 4d70_eu | 4d70 | eu | 61539 | 1230.8 | 1464.6 | 10 | 224.5 | 11 | 39 | 4 |
| 4d70_na | 4d70 | us | 65300 | 1306.0 | 1331.6 | 28 | 3.9 | 29 | 66 | 23 |
| 9bd4_eu | 9bd4 | eu | 62717 | 1254.3 | 1449.3 | 4 | 191.8 | 5 | 76 | 1 |
| 9bd4_na | 9bd4 | us | 58684 | 1173.7 | 1441.6 | 145 | 16.0 | 146 | 50 | 143 |
| a0da_eu | a0da | eu | 62570 | 1251.4 | 1436.5 | 6 | 177.2 | 7 | 1317 | 1 |
| a0da_na | a0da | us | 59560 | 1191.2 | 1343.4 | 77 | 11.3 | 78 | 59 | 66 |
| ac59_eu | ac59 | eu | 63109 | 1262.2 | 1473.2 | 5 | 202.0 | 6 | 1292 | 1 |
| ac59_na | ac59 | us | 55082 | 1101.6 | 1343.4 | 74 | 17.7 | 75 | 31 | 68 |
| b512_eu | b512 | eu | 50572 | 1011.4 | 1240.5 | 5 | 174.5 | 6 | 432 | 1 |
| b512_na | b512 | us | 67014 | 1340.3 | 1343.4 | 5 | 2.2 | 6 | 37 | 3 |
| c6f3_na | c6f3 | us | 55213 | 1104.3 | 1329.0 | 91 | 14.4 | 92 | 36 | 82 |
| ce9d_eu | ce9d | eu | 62697 | 1253.9 | 1446.0 | 9 | 182.1 | 10 | 407 | 2 |
| ce9d_na | ce9d | us | 64080 | 1281.6 | 1333.8 | 40 | 9.4 | 41 | 45 | 27 |
| e90f_eu | e90f | eu | 58667 | 1173.3 | 1359.8 | 6 | 183.8 | 7 | 66 | 2 |
| f2ad_eu | f2ad | eu | 62122 | 1242.4 | 1450.8 | 4 | 206.1 | 5 | 1298 | 1 |
| f2ad_na | f2ad | us | 63639 | 1272.8 | 1332.4 | 50 | 6.5 | 51 | 40 | 42 |
| **total** | | | **1377145** | 27543 | | **739** | | **763** | **31** | **615** |

Subject-stream lengths (what `apply_sliding_window` actually groups on, `sliding_window.py:102-103` `for i, subject in enumerate(np.unique(full_data[:, 0])): subject_data = full_data[full_data[:, 0] == subject]`): 05d8 61755, 0846 69569, 10f0 111560, 2dd9 94285, 4991 67411, 4d70 126839, 9bd4 121401, a0da 122130, ac59 118191, b512 117586, c6f3 55213, ce9d 126777, e90f 58667, f2ad 125761.

**2.2 Sampling rate.** `preprocess_data.py:31` `sampling_rate = 50` is the only definition in `src/` (assigned to `args.sampling_rate` at `main.py:176`, *after* the cfg dump at `:158-160`, so it never reaches `cfg.txt`). Raw files do carry timestamps (`data/raw/05d8_eu.csv` header `timestamp,subject,acc_x,acc_y,acc_z,coarse,basketball,locomotion,in/out`, 20 ms resolution). **50 Hz holds exactly**: within every recording's game span the raw timestamp stream has median Δt = 20 ms and **zero** gaps > 25 ms (`raw_ts_gaps_gt25 = 0` for all 24 files) — no drift, no dropped packets. Every gap in the table above is created by row deletion, not by the sensor.

**2.3 Contiguity.** Not contiguous. Two seam types in each subject stream:
- *Intra-recording splices* (739 total): rows removed by `data_creation.py:46` (coarse `not_labeled`) and `:69` (basketball/locomotion `not_labeled` or `jumping`). Gaps range from 40 ms (one dropped sample) to 359 s (`10f0_eu`, a break inside the game span). US recordings are far more fragmented (e.g. `9bd4_na` 145 splices, 143 segments < 30 s) than EU ones.
- *EU/US join* (10 subjects): the two recordings are separate days/venues concatenated back-to-back with no marker.
Total: **749 seams across the 14 subject streams**, none marked in `train`. The current 1-s windows already straddle them (this is `boracle_duration_bias.py:67-70`'s "time SEAM" caveat), so the baseline is not seam-clean either; but at 30 s the exposure is ~30× larger. Seam positions can be recovered only from `data/raw/*.csv` by replaying `data_creation.py`'s filters (done in this session; the row counts match the CSV block-for-block).

**2.4 Shortest recording.** `0846_eu` at 373.7 s (18684 samples) is the shortest *recording*; every recording is > 30 s. But the shortest *contiguous segment* is 31 samples (`ac59_na`), and 615/763 segments are < 1500 samples, so a "≥ 30 s contiguous" rule would discard most segments in the US recordings. The sequence-cutting rule needs a fallback (pad + `ignore_index`, or allow seam-crossing with a seam-mask channel, or shorter sequences).

## 3. Fold definition (loso_G)

**3.1 Where and on what key.** `validation.py:115` `fold_subjects = np.unique(data[:, 0])` — the key is **subject code** (column 0 of `train`, the LabelEncoder integer from `preprocess_data.py:49-55`), i.e. one fold = one subject = both of that subject's recordings. `:116-117` `if args.loso_subjects: fold_subjects = np.array([sbj for sbj in fold_subjects if str(args.subjects[int(sbj)]) in args.loso_subjects])`. `:121-122` `train_data = data[data[:, 0] != sbj]; val_data = data[data[:, 0] == sbj]`.

**3.2 Before windowing.** The split at `:121-122` is on the sample array; windowing follows at `:126-138`. Folds do not depend on any window attribute.

**3.3 Membership (reproduced).** `args.subjects` = `data.iloc[:, 3].unique()` (`preprocess_data.py:47`, order of first appearance in drill+warmup+game) = `['05d8','0846','10f0','2dd9','4991','4d70','9bd4','a0da','ac59','b512','c6f3','ce9d','e90f','f2ad']`, which is identical to the sorted LabelEncoder order. So the `--loso_subjects b512,a0da,4d70,ce9d,9bd4` tokens resolve to codes 9, 7, 5, 11, 6 and to the *same* subjects. Fold loop order is by code: 4d70 (5), 9bd4 (6), a0da (7), b512 (9), ce9d (11).

| fold (val subject) | code | val samples | val windows | train samples | train windows |
|---|---|---|---|---|---|
| 4d70 | 5 | 126839 | **5072** | 1250306 | 49994 |
| 9bd4 | 6 | 121401 | **4855** | 1255744 | 50211 |
| a0da | 7 | 122130 | **4884** | 1255015 | 50182 |
| b512 | 9 | 117586 | **4702** | 1259559 | 50364 |
| ce9d | 11 | 126777 | **5070** | 1250368 | 49996 |
| total | | 614733 | **24583** | | |

Matches v6 §7.2 exactly; rebound / layup validation windows = **214 / 174** (per fold rebound 70/38/39/16/51, layup 30/40/20/42/42). Training-side subjects per fold = the other 13 (including the 9 never used as validation: 05d8, 0846, 10f0, 2dd9, 4991, ac59, c6f3, e90f, f2ad).

**3.4 Determinism.** The split is a pure function of subject code and the CSV; no RNG, no window count enters. `seed_torch` (`torchutils.py:15-33`) seeds torch/cuda and sets `use_deterministic_algorithms(True)`; the DataLoader is unshuffled (`train.py:470` `shuffle=config["shuffling"]`, default `False` at `main.py:114`). Evidence: the 2026-08-14 seed-1 re-run reproduces the seed-1 baseline `y_pred` bit-for-bit on all five folds. **Caveat**: the token→subject lookup at `validation.py:117` indexes the *unsorted* `args.subjects` with the *sorted* code; it is correct only because the two orders coincide (`_loso_common.py:337-341` and `sample_level_f1.py:45-53` document the same hazard). The dense path should reuse `fold_subjects` from this exact expression, or resolve tokens through `sorted(args.subjects)`.

## 4. Normalization and leakage

**4.1 There is no scaling or standardization anywhere.** grep over `src/` for `scaler|normaliz|StandardScaler|MinMax|.std(|detrend|gravity|magnitude` finds nothing on the data path: `preprocess_data.py:160` returns raw `X.astype(np.float32)`; `validation.py` only windows and strips column 0; `train.py:455-456` wraps the arrays in `TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))`; `InceptionContext.forward` (`:167-193`) starts with the optional `ChannelAffine` then convolutions. Raw values are accelerometer g (e.g. `1.83008, -1.0603, -0.04761`). There is therefore **no train/val leakage from normalization and nothing per-window to reproduce**.

**4.2 Consequence for the dense path.** No choice to make — feed raw g exactly as the windowed path does. Note the stale docstring at `InceptionContext.py:14-15`: "Inputs are expected to already be z-scored upstream" — they are not; `--use_channel_affine` (default off, `main.py:236`) is the only learnable input rescaling.

**4.3 Other window-level preprocessing.** None. Augmentation (`augmentation.py`) and subsampling (`subsampling.py`) operate on training windows only, after windowing, and are off in the baseline (`augment_classes=None`, `subsample_fraction=1.0`). Neither has a sequence-level equivalent in the repo.

## 5. `sample_level_f1.py` ground truth — the anchor

**5.1 Reconstruction.** Ground truth is the **raw stream**, not expanded window labels: `sample_level_f1.py:353` `y_true_sample = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy()` from `labels_export.csv.gz` (columns `subject,label`, 1,377,145 rows, identical vocabulary and order to the game CSV; loaded by `window_purity.py:81-103`). Only the *predictions* are expanded: `:370` `sample_pred = expand_last_window_wins(y_pred_win, n_samples, win_len, step)`. Window `y_true` is used solely to certify the fold→subject mapping (`:360-368`: `idx_last = arange * step + win_len - 1`, must equal `y_true_win` exactly).

**5.2 Size of the last-sample labeling artifact.** Computed here as: expand the npz `y_true` (last-sample rule) with last-window-wins, compare to the raw stream on covered samples, seed-1 validation subjects (the artifact is seed-independent):

| fold | covered samples | disagreeing | % | rebound raw | rebound expanded |
|---|---|---|---|---|---|
| 4d70 | 126825 | 15313 | 12.07 | 1720 | 1750 |
| 9bd4 | 121400 | 12504 | 10.30 | 1031 | 950 |
| a0da | 122125 | 10248 | 8.39 | 967 | 975 |
| b512 | 117575 | 8807 | 7.49 | 405 | 400 |
| ce9d | 126775 | 18318 | 14.45 | 1258 | 1275 |
| **total** | **614700** | **65190** | **10.61** | 5381 | 5350 |

10.6 % of validation samples carry a different label under the window rule than in the raw stream. It is dominated by walking↔running (16158 + 16144), but for rebound: 2912 raw-rebound samples are relabeled (mostly walking 1159 / running 722 / pass 429 / dribbling 357) and 2881 non-rebound samples become rebound (walking 1456 / running 1130). Net class totals nearly cancel (rebound −31), which is why the window-proxy *durations* in §6 look unbiased while per-sample *labels* are 10 % wrong. This is the artifact a dense target removes.

**5.3 Recommended target = eval target: the raw per-sample stream** (`train[:, -1]` in `src/`, equivalently `labels_export.csv.gz` / `label_id` in `analysis/`). Reasons: (a) it is what `sample_level_f1.py` and `boracle_duration_bias.py` already score against (`:353`, `boracle:159`), so no evaluator changes; (b) it has no window-length dependence, so 1-s baseline and dense runs share one denominator (`sample_level_f1.py:6-14`); (c) the expanded-window alternative embeds the 10.6 % artifact above into the training signal. The two arrays must be byte-identical: `train[:, -1]` for subject code c equals `labels_df.label_id` for that subject (verified by the certification at `:364`, which passed on all 15 fold×seed npz files this session).

**5.4 Sample-level baseline anchor** (pooled over 5 folds, covered samples, `sample_level_f1.py` output this session):

| class | support (samples) | seed 1 F1 | seed 2 F1 | seed 3 F1 | 3-seed mean | seed-1 prec / rec |
|---|---|---|---|---|---|---|
| dribbling | 24543 | 0.6612 | 0.6679 | 0.6577 | 0.6623 | 0.6270 / 0.6993 |
| shot | 2579 | 0.2244 | 0.2219 | 0.2112 | 0.2192 | 0.2087 / 0.2427 |
| pass | 9708 | 0.2676 | 0.2533 | 0.2702 | 0.2637 | 0.2371 / 0.3071 |
| **rebound** | 5381 | **0.1092** | 0.0729 | 0.1363 | **0.1061** | 0.1882 / 0.0769 |
| layup | 4426 | 0.2372 | 0.2479 | 0.2294 | 0.2382 | 0.3028 / 0.1950 |
| walking | 314831 | 0.7676 | 0.7378 | 0.7432 | 0.7495 | 0.8497 / 0.6999 |
| running | 169355 | 0.7847 | 0.7895 | 0.7787 | 0.7843 | 0.7964 / 0.7734 |
| standing | 27080 | 0.2128 | 0.1889 | 0.1709 | 0.1909 | 0.1450 / 0.3994 |
| sitting | 56797 | 0.8602 | 0.8361 | 0.8457 | 0.8473 | 0.7876 / 0.9475 |
| **macro** | 614700 | **0.4583** | 0.4462 | 0.4493 | **0.4513** | |

Window-level, same runs (diagnostic): rebound 0.2053 / 0.1879 / 0.1975 (mean **0.1969**), walking 0.7926 / 0.7610 / 0.7686 (mean **0.7741**), macro 0.5273 / 0.5143 / 0.5129.

**5.5 Overlap rule.** `sample_level_f1.py:296-311` `expand_last_window_wins`: "Window i covers [i*step, i*step + win_len). Assigning in ascending i makes the LAST window covering a sample the one that sets it." Samples past the last window stay `UNCOVERED = -1` (`:93`) and are excluded (`:371`, `:386-387`). Coverage ≥ 0.99989 per fold.

## 6. Sample-level class counts

**6.1 Per fold** (reproduced from `train`; windows from `apply_sliding_window` at 1 s / 50 %):

Validation subject — samples (share %) | windows (share %):

| class | 4d70 | 9bd4 | a0da | b512 | ce9d |
|---|---|---|---|---|---|
| dribbling | 6891 (5.43) / 276 (5.44) | 2637 (2.17) / 106 (2.18) | 1731 (1.42) / 71 (1.45) | 8940 (7.60) / 354 (7.53) | 4344 (3.43) / 174 (3.43) |
| shot | 189 (0.15) / 7 (0.14) | 548 (0.45) / 23 (0.47) | 145 (0.12) / 5 (0.10) | 308 (0.26) / 13 (0.28) | 1389 (1.10) / 55 (1.08) |
| pass | 3413 (2.69) / 139 (2.74) | 1725 (1.42) / 71 (1.46) | 1053 (0.86) / 40 (0.82) | 1560 (1.33) / 65 (1.38) | 1957 (1.54) / 77 (1.52) |
| rebound | 1720 (1.36) / 70 (1.38) | 1031 (0.85) / 38 (0.78) | 967 (0.79) / 39 (0.80) | 405 (0.34) / 16 (0.34) | 1258 (0.99) / 51 (1.01) |
| layup | 809 (0.64) / 30 (0.59) | 1035 (0.85) / 40 (0.82) | 499 (0.41) / 20 (0.41) | 996 (0.85) / 42 (0.89) | 1087 (0.86) / 42 (0.83) |
| walking | 63176 (49.8) / 2525 (49.8) | 45509 (37.5) / 1823 (37.5) | 59908 (49.1) / 2398 (49.1) | 81127 (69.0) / 3243 (69.0) | 65144 (51.4) / 2609 (51.5) |
| running | 35579 (28.1) / 1423 (28.1) | 43171 (35.6) / 1725 (35.5) | 32603 (26.7) / 1302 (26.7) | 12635 (10.7) / 508 (10.8) | 45367 (35.8) / 1815 (35.8) |
| standing | 5298 (4.18) / 211 (4.16) | 3749 (3.09) / 151 (3.11) | 9319 (7.63) / 373 (7.64) | 2483 (2.11) / 96 (2.04) | 6231 (4.91) / 247 (4.87) |
| sitting | 9764 (7.70) / 391 (7.71) | 21996 (18.1) / 878 (18.1) | 15905 (13.0) / 636 (13.0) | 9132 (7.77) / 365 (7.76) | 0 / 0 |

Training side, samples | windows:

| class | 4d70 fold | 9bd4 fold | a0da fold | b512 fold | ce9d fold |
|---|---|---|---|---|---|
| dribbling | 40692 / 1626 | 44946 / 1796 | 45852 / 1831 | 38643 / 1548 | 43239 / 1728 |
| shot | 7379 / 298 | 7020 / 282 | 7423 / 300 | 7260 / 292 | 6179 / 250 |
| pass | 18098 / 724 | 19786 / 792 | 20458 / 823 | 19951 / 798 | 19554 / 786 |
| rebound | 8617 / 339 | 9306 / 371 | 9370 / 370 | 9932 / 393 | 9079 / 358 |
| layup | 7433 / 296 | 7207 / 286 | 7743 / 306 | 7246 / 284 | 7155 / 284 |
| walking | 661229 / 26424 | 678896 / 27126 | 664497 / 26551 | 643278 / 25706 | 659261 / 26340 |
| running | 316219 / 12669 | 308627 / 12367 | 319195 / 12790 | 339163 / 13584 | 306431 / 12277 |
| standing | 52297 / 2089 | 53846 / 2149 | 48276 / 1927 | 55112 / 2204 | 51364 / 2053 |
| sitting | 138342 / 5529 | 126110 / 5042 | 132201 / 5284 | 138974 / 5555 | 148106 / 5920 |

Note `ce9d` has **no sitting** in validation (0 samples); `sklearn` metrics on that fold yield a degenerate sitting column (the `RuntimeWarning: invalid value encountered in divide` at `train.py:529` in the logs).

**6.2 Rebound share, sample vs window — STOP-AND-REPORT item.** The shares **agree**: pooled validation rebound share is 0.875 % of samples (5381/614700) vs 0.870 % of windows (214/24583); per fold 1.36/1.38, 0.85/0.78, 0.79/0.80, 0.34/0.34, 0.99/1.01 %. `boracle_duration_bias.py` (run this session on seed 1) confirms the window-proxy truth differs from raw-sample truth by only **+0.58 %** for rebound (107.62 s vs 107.00 s; table "raw_sample_s / window_proxy_s"). **The ~59 % figure is the model's own duration bias, not a labeling undercount**: `[1] PER-CLASS TOTAL DURATION` → `rebound 107.62 true_sec, 44.00 est_sec, −63.62 bias_sec, −59.12 %, ratio 0.4088, UNDER 5/5 folds`. So the checklist premise ("sample-level share vs window-level share should reproduce the 59 % undercount") does not hold; the 59 % is reproduced, but as `est/true`, and it is what the dense path is meant to fix (rebound sample-level recall is 0.0769, §5.4). Nothing about the dense plan changes; only the attribution does.

**6.3 sqrt_inverse weights.** `train.py:413-432`: `weight_labels = pinned_weight_labels(train_labels, config)` (`:415`), `balanced_weights = compute_class_weight("balanced", classes=np.unique(weight_labels + 1), y=weight_labels + 1)` (`:416-418`), `derived_weights = np.sqrt(balanced_weights); derived_weights = derived_weights / derived_weights.mean()` (`:420-421`), `loss.weight = all_class_weights.to(device)` (`:432`). It counts **`train_labels`**, the `(N_windows,)` array passed from `validation.py:287`. `pinned_weight_labels` (`:285-333`) returns that array unless `pin_weights_to_pre_augmentation` is set, in which case it rebuilds a synthetic label vector from `config["pre_aug_train_counts"]` (`:312`). For a dense path the same function works unchanged if handed the flattened per-sample training labels (`(N_samples,)`); the ratio of weights would shift slightly because sample and window shares differ by < 1 % per class (§6.1). **Note**: the uncommitted working-tree diff removes the `args.pre_aug_train_counts` assignment from `validation.py` (lines 165-169 in HEAD) while `train.py:312` still reads it — pinning would silently become a no-op if that diff is committed.

## 7. Event statistics

Runs of a label within each subject stream (seams not marked, so a run may span a splice; 50 Hz):

| class | events (all) | per val fold 4d70/9bd4/a0da/b512/ce9d | median len | IQR | max | min | events < 50 samples |
|---|---|---|---|---|---|---|---|
| rebound | 161 | 25 / 14 / 14 / 6 / 20 | 61 (1.22 s) | 51–75 | 147 | 21 | 36 |
| layup | 81 | 9 / 9 / 6 / 10 / 10 | 100 (2.0 s) | 87–116 | 156 | 57 | 0 |
| shot | 78 | 2 / 5 / 2 / 3 / 15 | 92 (1.84 s) | 74–105 | 236 | 47 | 2 |

Train-side events per fold = total − val (rebound 136/147/147/155/141). All-class event counts: dribbling 241, pass 358, walking 1436, running 1326, standing 300, sitting 13.

**Class preceding each rebound event** (161): walking 79, running 64, standing 14, layup 3, dribbling 1. Preceding-segment run length: median 210 samples (4.2 s), IQR 98–431, max 2265; preceding-*running* segments (n = 64): median 204.5, IQR 96–311. So "running-dominated" per the onset test is **not** what the stream shows — walking precedes rebound slightly more often than running; consistency with `onset_test.py`'s 34-sample median is UNCONFIRMED because that script is not in the repo (only referenced from `knn_adjacency.py:6-8,36,86`) and its 34-sample figure describes the lead-in *inside error windows*, a different quantity from the full preceding segment measured here. Layup is preceded by dribbling 54/81; shot by dribbling 23, running 20, walking 17, standing 10.

**Gaps between consecutive ball-action events** (interior non-ball segments, per subject; 641 total): median 1114 samples (22 s), p90 3821 (76 s), p99 12339 (247 s), max 36576 (**731 s**, `10f0`). 457/641 (71 %) exceed 10 s and 346 exceed 20 s. So ±10 s of context around a rebound usually contains no other ball action — context is unambiguous for locating the event, but a 30-s sequence will frequently hold zero ball actions at all.

## 8. Training loop coupling

**8.1 Call chain.** `main.py:172-173` `load_dataset(...)` → `:206` `cross_participant_cv(train, args, log_dir, run)` → `validation.py:126-138` windowing → `:206` channel strip → `:215-216` `args.window_size = X_train.shape[1]; args.nb_channels = X_train.shape[2]` → `:259-270` model ctor → `:287-290` `train(X_train, y_train, X_val, y_val, network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler, config=vars(args), run=run, name='sbj_' + str(int(sbj)))`. Readers of `args.window_size` downstream: only the model constructors (`validation.py:220,222,226,229,239,255,260`). Nothing in `train.py` reads it. The `X.shape[1]` coupling means a dense loader that hands `(N_seq, T_seq, C)` arrays would set `window_size = T_seq` automatically.

**8.2 Model.** `InceptionContext.py:143-147`: `dummy = torch.zeros(1, 1, window_size, channels)` → `gru_input_dim = dummy_cat.shape[1] * dummy_cat.shape[3]` (= sum_filters × C, independent of T because branches use 'same' padding, `:46`). Stage 1 keeps T through the branches (`:175` `(B, sum_filters, T, C)`), the reshape (`:178` `(B, T, F*C)`) and the GRU (`:180` `(B, T, nb_units_gru_ic)`). **The module that collapses T is the attention pooling at `:181-182`**: `attn_weights = torch.softmax(self.gru_attention(x), dim=1)` / `x = torch.sum(attn_weights * x, dim=1)` → `(B, nb_units_gru_ic)`. Everything before it is already dense-capable; a dense head would branch off the `(B, T, 128)` GRU output at `:180`.

**8.3 Stage 2.** `:185-186` `x = x.unsqueeze(1)` → `(B, 1, 128)`; `self.context_lstm` is built with `batch_first=False` (`:154-160`), so it reads that tensor as `(seq_len=B, batch=1, 128)`: **the minibatch of windows is the sequence**. `augmentation.py:12-33` documents the same. Because the input is one pooled vector per window, `context_lstm` cannot see within-window time; for within-sequence context it would have to be re-instantiated on the `(B, T, 128)` stream with `batch_first=True` (or transposed). It cannot be reused as-is.

**8.4 Loss.** `train.py:210` `criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)`; weights injected as `loss.weight = all_class_weights.to(device)` (`:398/411/432/443`). Called as `train_loss = criterion(train_output, targets.long())` (`:525`, val `:573`). `nn.CrossEntropyLoss` accepts `(B, C, T)` logits with `(B, T)` targets natively, so the loss object and weight injection reuse unchanged; the only caller-side change is the logit layout.

**8.5 Metrics.** No torchmetrics — `sklearn.metrics` throughout: `train.py:13` import; per-epoch `f1_score(val_gt, val_preds, average=None, zero_division=1, labels=labels)` (`:598`), early-stop/best metric `f1_score(..., average="macro")` (`:636`); predictions via `np.argmax(train_output..., axis=-1)` (`:533`, val `:579` after `softmax(val_output, dim=1)` at `:575`) and `targets...flatten()`. With `(B, C, T)` logits, `argmax(axis=-1)` would take the wrong axis and `softmax(dim=1)` would be right by accident — the argmax/flatten lines need an axis change; the sklearn calls accept flattened per-sample arrays unchanged.

**8.6 cfg.txt.** `main.py:158-160` `json.dump(vars(args), fid, indent=2)` — wholesale, before `args.subjects/sampling_rate/nb_classes/class_names/has_void` are set at `:175-179` (so those never appear, as `sample_level_f1.py:87-90` notes). Any new argparse flag appears automatically; the config-diff rule holds.

**8.7 W&B / checkpoints.** Run name `main.py:145` `f"{args.test_type}_{args.test_case}_{args.network}"`; config `vars(args)` (`:146`). Checkpoint `validation.py:295-297` `checkpoint_best_{subject}_{name}.pth`; predictions csv `:303-306`; npz `:318-320` `preds_{subject}_{subsample_fraction}_seed{seed}{aug_suffix}.npz`. Per-epoch logging `train.py:617` keys `{name}/train_loss, val_loss, val/acc_macro, ...`. **Nothing is keyed to window count or `sw_length`**; `sw_length` only reaches `cfg.txt` and W&B config. (Consequence, already exploited by the window sweep: a dense run at a different `--name` is indistinguishable by filename — `discover_fold_npz` at `sample_level_f1.py:204-238` relies on one npz per fold per dir.)

**8.8 Prediction dump.** `validation.py:320` `np.savez(npz_name, y_pred=val_output[:, 0].astype(int), y_true=val_output[:, 1].astype(int))` where `val_output = np.vstack((best_val_preds, val_gt)).T` (`train.py:692`): two 1-D int arrays of length = number of validation *windows*, in window order. Consumers: `sample_level_f1.py:241-260` (`load_npz`, asserts 1-D int in [0, 8], equal shapes), then `:263-289` certifies `y_true` against the raw last-sample sequence and `:370` expands; `boracle_duration_bias.py:148-149` reuses both; `learning_curve_plot.py:205-206,269-270` read `y_pred/y_true` directly; older scripts (`duration_mae.py:100-101`, `_loso_common.py:278-280`) read `predictions_best_*.csv` col 0 = y_pred, col 1 = y_true. A dense run that writes `y_pred/y_true` of length = *samples* (in raw order, with `-1` for padding) would pass `load_npz` but **fail `resolve_fold`** (window-count match) — both scripts need a "dense" branch that skips certification/expansion and scores the arrays directly; `boracle`'s `window_count_est` / `window_proxy_true` legs become undefined.

## 9. Compute baseline

- Hardware (W&B `wandb-metadata.json`, run `gm924rui` = seed-1 baseline): NVIDIA L40 (46 GB), 24 CPUs.
- Seed-1 baseline (`2026-07-17_07-56-30`): W&B `_runtime` = 6376 s (1 h 46 min) for 5 folds × 40 epochs; per-fold from npz mtimes 3:10 / 9:22 / 30:47 / 31:33 / 31:22 — **contaminated** by a concurrent run (`2026-07-17_08-01-44`, started 5 min in, on the same GPU); seeds 2 and 3 ran simultaneously with each other (~30 min/fold each).
- Clean measurement: the identical-config seed-1 re-run `2026-08-14_06-21-45_lc_frac1.0_ckpt` (alone on the GPU, bit-identical predictions) took 176 / 175 / 175 / 175 s per fold from checkpoint mtimes → **≈ 4.4 s per epoch** (≈ 500 train batches of 100 × 50 × 3 plus 50 val batches), ≈ 15 min for all five folds.
- Peak GPU memory: UNCONFIRMED — not in `wandb-summary.json`, and `train.py` never logs it; the W&B system-metrics panel for run `gm924rui` would resolve it. The model is small (parameter table at `train.py:368`; L40 headroom is not a concern at 1 s windows).
- Training samples per fold: 1,250,306 / 1,255,744 / 1,255,015 / 1,259,559 / 1,250,368 (≈ 25,000 s each). At 30 s / 50 % overlap (1500 samples, step 750), cut naively per subject with the strict-`<` rule: **1646 / 1654 / 1653 / 1659 / 1646 training sequences and 168 / 160 / 161 / 155 / 168 validation sequences** per fold (vs ≈ 50,000 / 5,000 windows). Same number of samples per epoch; each sequence carries 30× the timesteps of a window, so per-batch memory scales with `batch_size × 1500 × F`. Cutting only on seam-free segments (§2.3) would reduce these counts substantially, especially on US recordings.

---

## Final block

### Recommended dense target array
`train[:, -1]` as returned by `load_dataset('subset_specific', 'loso_G')` — the raw per-sample label stream (uint8 class ids 0–8, rebound = 3), sliced per subject code exactly as `validation.py:121-122` does. It is byte-identical to `labels_export.csv.gz` → `label_id`, which is what `sample_level_f1.py:353` and `boracle_duration_bias.py:159` already use as truth. Using it as both training target and evaluation target removes the 10.61 % (65,190-sample) last-sample labeling artifact measured in §5.2 and keeps the evaluation denominator unchanged.

### Sample-level baseline anchor table (pre-registration bar)
Seed 1, pooled 5 folds, 614,700 covered samples (see §5.4 for seeds 2–3): rebound F1 **0.1092** (prec 0.1882 / rec 0.0769), layup 0.2372, shot 0.2244, walking 0.7676, running 0.7847, macro **0.4583**. 3-seed means: rebound 0.1061, macro 0.4513. Rebound duration ratio est/true **0.4088** (−59.12 %, UNDER in 5/5 folds).

### Must add vs. reuse unchanged

Add:
- **Sequence loader** — nothing in `src/` produces `(N_seq, T_seq, C)` with `(N_seq, T_seq)` labels; `apply_sliding_window` collapses labels at `sliding_window.py:117` and groups only by subject code (`:102-103`) with no seam awareness. Must also carry seam positions recovered from `data/raw/*.csv` (§2.3) since `train` has no timestamp/location column (`data_creation.py:52`, `preprocess_data.py:97`).
- **Dense head** — branch at `InceptionContext.py:180` (`(B, T, 128)` GRU output) instead of the pooling at `:181-182`; Stage 2 `context_lstm` (`:154-160`, `batch_first=False`, fed `(B, 1, 128)` at `:185-186`) must be re-instantiated on the time axis.
- **Logit-layout plumbing in `train.py`** — `argmax(axis=-1)` at `:533/579` and `.flatten()` on targets assume `(B, C)` / `(B,)`; with `(B, C, T)` the argmax axis changes (sklearn calls at `:598/636` then work on flattened arrays).
- **Weight input** — `train.py:413-421` counts whatever label array it is given; pass flattened per-sample training labels (and adapt `pinned_weight_labels` `:285-333` only if pinning is wanted).
- **Dense npz branch in `sample_level_f1.py` / `boracle_duration_bias.py`** — `resolve_fold` (`:263-289`) and `expand_last_window_wins` (`:296-311`) assume window-count arrays; a per-sample dump needs a bypass that scores directly and skips the window-proxy legs.
- **Padding / `ignore_index`** — not needed for the label vocabulary (§1.6) but needed if variable-length seam-free segments are batched.

Reuse unchanged:
- Folds: `validation.py:115-122` (subject-code split, pre-windowing, deterministic; §3).
- Loss object and class-weight injection: `train.py:210`, `:398-443` (`CrossEntropyLoss` takes `(B,C,T)`/`(B,T)` natively).
- `cfg.txt` writer: `main.py:158-160` (wholesale `vars(args)`).
- W&B run naming / per-epoch logging: `main.py:145-147`, `train.py:617`; checkpoint/npz naming: `validation.py:295-320` (nothing keyed to `sw_length` or window count).
- Ground-truth files: `labels_export.csv.gz` and `window_purity.load_labels` (`:81-103`).
- Optimizer, scheduler, seeding: `train.py:224-268`, `torchutils.py:15-33`.

### UNCONFIRMED
- **"v4 label regrouping"** — no such logic in `src/`; only the CSV-time locomotion→basketball fill at `data_creation.py:66-69`. Resolve by pointing at the document/commit that defines v4.
- **`onset_test.py` 34-sample median and the "running-dominated" preceding class** — script absent from the repo (referenced at `knn_adjacency.py:6-8,36,86`); its pooled 109 count *is* reproduced here (35 + 39 + 35 pure-rebound windows predicted running, §10), but the 34-sample lead-in statistic and its consistency with §7 need the script or its output.
- **Peak GPU memory** — not logged locally; check W&B system metrics for run `gm924rui`.
- **Number of game sessions per recording** — `coarse` distinguishes only game/warmup/drill; the 150–360 s in-span gaps in EU recordings may be halves/breaks or separate games. Resolve from raw timestamps + study protocol.
- **Units of acc columns** — assumed g from the value range and the git history note ("g unit added to the dataset characteristics section"); no unit is stated in code.

### Blocking surprises
1. **Seams.** 749 unmarked discontinuities in the 14 subject streams (739 row-drop splices + 10 EU/US joins); 615 of 763 contiguous segments are < 30 s, shortest 31 samples (§2). The 30-s / 50 %-overlap cutting rule in the plan cannot be applied to the `train` array alone; it needs seam positions replayed from `data/raw/` and a fallback for short segments. (The 1-s baseline already straddles these seams, so this is a pre-existing confound, not a regression.)
2. **§6 premise inverted.** Sample-level and window-level rebound shares agree to < 1 %; the ~59 % is the model's duration under-estimate (`est/true = 0.4088`), not a labeling undercount (§6.2). The plan's motivation survives (rebound sample-level recall is 0.077), but the attribution in the plan text should be corrected.
3. **No normalization exists** (§4) — removes a design question rather than adding one, but the `InceptionContext.py:14-15` docstring is wrong and should not be relied on.
4. **Fold token lookup is order-coincidence-dependent** (`validation.py:117`, §3.4) — safe now, fragile if the dense loader builds `args.subjects` any other way.
5. **Working-tree diff** removes `args.pre_aug_train_counts` from `validation.py` while `train.py:312` still reads it — unrelated to dense work but will silently disable weight pinning if committed.

### Anchor reproduction (§10)

| anchor | expected | reproduced | source this session |
|---|---|---|---|
| Seed-1 validation windows (1 s) | 24,583 | 24,583 | `apply_sliding_window` replay; `sample_level_f1.py` |
| Per-fold windows | 5072 / 4855 / 4884 / 4702 / 5070 | identical | same |
| Rebound / layup validation windows | 214 / 174 | 214 / 174 | same |
| 3-seed-mean rebound window F1 | 0.1969 | (0.2053 + 0.1879 + 0.1975)/3 = 0.1969 | npz seeds 1–3 |
| Walking window F1 (3-seed mean) | 0.7741 | (0.7926 + 0.7610 + 0.7686)/3 = 0.7741 | same |
| Rebound window errors, 3-seed pooled | 109 (35 + 39 + 35) | 35 + 39 + 35 = 109 pure-rebound windows predicted running (66 pure windows per seed) | npz + raw stream |
| Rebound duration undercount | ~59 % | −59.12 % (est 44.00 s vs true 107.62 s) | `boracle_duration_bias.py` |

Note: total rebound-labeled windows mispredicted (any class) is 183 per seed (per fold seed 1: 58/34/31/15/45) — a different, larger quantity than the 109 onset anchor.
