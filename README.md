# Basketball Activity Recognition

## What this is

Human activity recognition for basketball from wrist-worn IMU data (50 Hz), trained and
evaluated on the Hang-Time HAR dataset with leave-one-subject-out (LOSO) cross-validation
over its 24 participants. Models predict nine classes — dribbling, shot, pass, rebound,
layup, walking, running, standing, sitting — either per window (1 s windows, 50 % overlap,
last-sample labeling) or densely, one prediction per raw sample (`--dense`).

## Upstream and license

This repository is a fork of [hangtime_har](https://github.com/ahoelzemann/hangtime_har),
the code released with the Hang-Time HAR dataset:

> Hoelzemann et al., *Sensors* 23(13):5879, 2023.

The code is under the MIT License (see [`LICENSE`](LICENSE)):

    Copyright (c) 2023 Alexander Hoelzemann

The dataset is distributed separately on Zenodo, under the terms stated on its record:
<https://zenodo.org/records/7920485>.

## Setup

> **TODO (author):** exact Python and PyTorch versions used to produce the results in
> `docs/`. `requirements.txt` pins `torch==2.14.0` but does not pin Python; the setup notes
> record Python 3.11 for this fork and Python 3.10 as the upstream recommendation.

Environment, as recorded in [`setups/SETUP_AND_MODIFICATIONS.md`](setups/SETUP_AND_MODIFICATIONS.md) §4:

```bash
conda create -n hangtime_har python=3.11
conda activate hangtime_har
conda install pytorch pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt
```

Package versions are pinned in [`requirements.txt`](requirements.txt).

**Dataset.** The dataset is not in this repository. Download it from
<https://zenodo.org/records/7920485> and place the raw recordings in `data/raw/`
(`data/raw/*.csv` plus `data/raw/meta.txt`). Then build the processed CSVs:

```bash
python src/data_processing/data_creation.py
```

This writes `data/hangtime_drill_data.csv`, `data/hangtime_game_data.csv` and
`data/hangtime_warmup_data.csv`. `data/` is git-ignored. The setup notes (§1–2) record
edits made to `meta.txt` and to the data-processing code to handle the `_eu`/`_na`
filename suffixes and mixed-type subject IDs.

## Data files in the repo

**`labels_export.csv.gz`** — committed. The per-sample ground-truth label stream for the
`loso_G` game data, columns `subject,label`, void-filtered, in original row order
([`analysis/window_purity.py`](analysis/window_purity.py) `load_labels`). Per
[`docs/recon_dense_report.md`](docs/recon_dense_report.md) §5.1 and §5.3, it has
1,377,145 rows in the same order as `data/hangtime_game_data.csv`, and its `label_id`
column is byte-identical to `train[:, -1]` from
`load_dataset('subset_specific', 'loso_G')`. It is the default `--labels` input of the
sample-level, duration and dense analysis scripts, and the `scripts/run_24fold_*.sh`
launchers read it to check their participant list.
**No script that generates this file is committed.**

**`data/seam_map.json`** — **not committed** (it is excepted in `.gitignore` but absent).
It maps the contiguous, time-continuous segments of the `loso_G` training array, needed
by `--dense` training and by the dense/session analysis scripts. Build it after
`data_creation.py` has run:

```bash
python scripts/build_seam_map.py --out data/seam_map.json
```

## Training

Entry point: `src/main.py`. Use the committed launch scripts rather than retyping flags;
each states every flag explicitly in its header and body.

**Current protocol — full 24-fold LOSO, seeds 1–3:**

```bash
# windowed (InceptionContext, 1 s windows)
DRY_RUN=1 bash scripts/run_24fold_windowed.sh    # print the commands, run nothing
nohup bash scripts/run_24fold_windowed.sh > logs/24fold_windowed.out 2>&1 &

# dense per-sample prediction (requires data/seam_map.json)
nohup bash scripts/run_24fold_dense.sh > logs/24fold_dense.out 2>&1 &

# a subset of seeds
SEEDS="2 3" bash scripts/run_24fold_dense.sh
```

Other 24-fold launchers: `run_24fold_deepconvlstm.sh`, `run_24fold_shallow.sh`,
`run_24fold_deepconvcontext_uni.sh`, `run_24fold_deepconvcontext_bi.sh`,
`run_24fold_dense_dclstm.sh`.

**Earlier protocol — 5-fold placeholder LOSO** (`b512_na, a0da_eu, 4d70_eu, ce9d_eu,
9bd4_na`), one seed per call:

```bash
bash scripts/run_baseline_ckpt.sh <seed> <name>   # windowed
bash scripts/run_dense.sh <seed> <name>           # dense
```

Related 5-fold scripts: `run_ablation_no_bilstm.sh`, `run_pinned_weight.sh`,
`run_seq_len_sweep.sh`. Runs write to `logs/<test_type>/<test_case>/<network>/<timestamp>_<name>/`.
`job_scripts/` holds older batch command lists.

## Analysis

Run from the repo root. Each script's module docstring gives its full usage and flags.

### Current

| Script | What it computes | Run |
|---|---|---|
| `sample_level_f1.py` | Sample-level F1 on the raw 50 Hz timeline, comparable across window lengths and dense runs | `python analysis/sample_level_f1.py --results_dir <run_dir> [--dense]` |
| `boracle_duration_bias.py` | Signed per-class duration bias when predictions are used as exposure estimates | `python analysis/boracle_duration_bias.py --results_dir <run_dir> [--dense]` |
| `calibration_factors.py` | Per-class duration correction factors with a leave-one-fold-out estimate of their value | `python analysis/calibration_factors.py --results_dir <run_dir> --dense --out_csv calibration_factors.csv` |
| `session_duration_error.py` | Per-session (unpooled) duration error, windowed vs dense | `python analysis/session_duration_error.py --windowed_dir <dir> --dense_dir <dir>` |
| `session_rank_stability.py` | Whether sessions' true duration ranking is preserved (Spearman), windowed vs dense | `python analysis/session_rank_stability.py --windowed_dir <dir> --dense_dir <dir>` |
| `risk_proxy_sensitivity.py` | Whether duration bias moves an illustrative injury-risk proxy; implements `docs/prereg_risk_proxy.md` | `python analysis/risk_proxy_sensitivity.py --windowed_dir <dir> --dense_dir <dir> --dense_all_dir <dir> [--calibrate]` |
| `dense_confusion_matrix.py` | Per-sample confusion matrix of dense predictions, row-normalized to recall | `python analysis/dense_confusion_matrix.py --results_dir <dense_dir>` |
| `dense_pass_overcount.py` | What is truly present when the dense model predicts pass | `python analysis/dense_pass_overcount.py --results_dir <dense_dir>` |
| `event_rebound_recall.py` | Per-event rebound recall: boundary trimming vs whole-event misses | `python analysis/event_rebound_recall.py --results_dir <dense_dir>` |
| `window_purity.py` | Whether rebound's low F1 is a last-sample-labeling artifact (last-sample purity) | `python analysis/window_purity.py --log_root <dir> --labels labels_export.csv.gz --out_csv window_purity_records.csv` |
| `window_purity_corrected.py` | Per-window majority-class purity; replaces `purity.py` | `python analysis/window_purity_corrected.py <hangtime_csv_path>` |
| `knn_adjacency.py` | Whether pure rebound windows sit inside the running cluster in Stage 1 feature space | `python analysis/knn_adjacency.py --ckpt_dir <dir> --baseline_dir <dir> --labels_path labels_export.csv.gz --data_dir data` |
| `pooled_per_class.py` | Pooled per-class precision / recall / F1 across LOSO held-out subjects | `python analysis/pooled_per_class.py <log_dir>` |
| `MAE.py` | Per-class duration MAE, predicted vs true time-per-class; writes summary and per-session CSVs | Edit `RUN_DIRS` in the script, then `python analysis/MAE.py` |
| `duration_mae.py` | Duration MAE and F1 across loss configurations and median-filter widths | `python analysis/duration_mae.py <run_dir_1> <run_dir_2> [...]` |
| `count_windows.py` | Per-class window counts per LOSO fold (or drill totals) | `python analysis/count_windows.py [--mode drill]` |
| `learning_curve_plot.py` | F1 vs training fraction for the rebound/layup augmentation go/no-go | `python analysis/learning_curve_plot.py` |
| `relabel_majority_vote.py` | Re-scores LOSO predictions against majority-vote window labels | `python analysis/relabel_majority_vote.py <log_dir>` |
| `purity_stratified_scoring.py` | Whether errors concentrate in impure windows; writes `purity_per_window.csv` | `python analysis/purity_stratified_scoring.py <log_dir>` |
| `per_subject_error_breakdown.py` | Per-subject walking↔standing confusion (reads `purity_per_window.csv`) | `python analysis/per_subject_error_breakdown.py <log_dir>` |
| `subject_ranking_persistence.py` | Whether subject difficulty ordering persists across two runs | `python analysis/subject_ranking_persistence.py <run_a_dir> <run_b_dir>` |
| `walking_signal_energy.py` | Whether walking→standing confusion tracks low accelerometer energy | `python analysis/walking_signal_energy.py <log_dir>` |
| `shot_confusion.py` | Where true-shot windows are predicted to, and what predicted-shot windows are, across runs | `python analysis/shot_confusion.py <run_dir_1> <run_dir_2> [...]` |

Helper modules, not runnable: `_loso_common.py` (windowed reconstruction and alignment),
`_dense_common.py` (dense loader), `_session_common.py` (per-session loader).

### Superseded / diagnostic

| Script | Status |
|---|---|
| `purity.py` | Quarantined. `window_purity_corrected.py`'s docstring states it replaces this script's containment-denominator computation and that the "88"/"40" numbers it produced must not be reused. |
| `duration_mae_before_phase_2.py` | Earlier version of `duration_mae.py`. Its docstring is identical to `duration_mae.py`'s and does not state why it was superseded. |

## Pre-registrations and results

| File | Date | Status as stated in the file's header |
|---|---|---|
| [`docs/prereg_dense.md`](docs/prereg_dense.md) — Pre-registration, dense per-sample prediction (Path A) | 2026-09-04 | "Commit this file at docs/prereg_dense.md before any training run is launched." |
| [`docs/prereg_dense_dclstm.md`](docs/prereg_dense_dclstm.md) — Pre-registration, dense per-sample labeling on DeepConvLSTM | 2026-10-02 | "Commit this file before any 24-fold run of the configuration it describes is launched." |
| [`docs/prereg_risk_proxy.md`](docs/prereg_risk_proxy.md) — Pre-registration, risk-proxy sensitivity analysis | 2026-09-09 | Draft — pending advisor review |
| [`docs/prereg_sequence_length_sweep.md`](docs/prereg_sequence_length_sweep.md) — Pre-registration, dense sequence-length sweep | 2026-09-09 | Draft — commit before launching any sweep runs |
| [`docs/recon_dense_report.md`](docs/recon_dense_report.md) — Recon report for the dense prediction path | 2026-09-04 | inspect-only |
| [`docs/dense_results.md`](docs/dense_results.md) — Dense per-sample prediction results, seeds 1–3 | 2026-09-09 | read-only |
| [`docs/seq_len_sweep_results.md`](docs/seq_len_sweep_results.md) — Dense sequence-length sweep results, seed 1 | 2026-09-10 | read-only |

## Repo layout

```
analysis/      post-hoc analysis scripts (see above)
docs/          pre-registrations, recon report, results
figures/       learning-curve plots
job_scripts/   older batch command lists
scratchpad/    analysis_out/: per-seed analysis outputs and two MAE variants
scripts/       launch scripts and build_seam_map.py
setups/        setup and modification notes
src/           main.py, data_processing/, model/, misc/, features/, visualization/
tests/         standalone checks: python tests/test_augmentation.py, python tests/test_sequence_loader.py
labels_export.csv.gz, requirements.txt, LICENSE
```

## Known issues

These are recorded as found and have not been fixed.

- `docs/seq_len_sweep_results.md` states that `docs/prereg_sequence_length_sweep.md` and
  `scripts/run_seq_len_sweep.sh` are 0-byte files. Both now have content; they were filled
  in by a later commit.
- `scripts/run_seq_len_sweep.sh` refers to itself as `scripts/run_seqlen_sweep.sh` in its
  header and usage lines.
- `analysis/purity.py`'s usage line runs `analysis/window_purity.py`.
- `analysis/duration_mae_before_phase_2.py`'s usage line runs `analysis/duration_mae.py`.
- `analysis/MAE.py` has no usage line; its inputs are set by the `RUN_DIRS` constant in the
  script.
- Python and PyTorch versions are inconsistent across `requirements.txt` and the setup
  notes (see the TODO under Setup).

## AI assistance

Claude (Claude Code and the Claude chat interface) was used to implement analysis scripts
and parts of the pipeline from author-written specifications; all such code was reviewed
and verified by the author.
