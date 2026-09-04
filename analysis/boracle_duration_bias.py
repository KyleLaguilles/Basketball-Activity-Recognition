#!/usr/bin/env python3
"""
Duration bias at the BOracle interface: when this classifier's predictions are
consumed as per-class EXPOSURE estimates, how wrong are the durations?

BOracle ingests per-class time-on-task as an injury-risk exposure term. F1 does not
answer the question that pipeline asks -- a class can hold a mediocre F1 while its
total duration comes out very close (errors cancel), or hold a respectable F1 while
its duration is badly off (errors are one-directional). This script measures the
duration directly, and reports the SIGN, because the two directions mean opposite
things downstream:

    over-counted   -> exposure inflated  -> false high risk
    under-counted  -> exposure deflated  -> MISSED risk

The second is the dangerous one, and it is where the explosive ball actions
(rebound, layup, shot -- the jump-and-land events an injury model cares about) are
expected to sit: they are low-support classes that the model under-predicts in
volume, so their estimated exposure is short.

WHAT COUNTS AS A DURATION. Two independent choices, and this script is explicit
about both because the literature is not:

  ESTIMATE RULE
    last-window-wins   expand each window's prediction over the samples it covers,
                       later windows overwriting earlier (sample_level_f1.py's rule)
    window-count       duration = window_count(class) * stride_seconds, no expansion
                       (analysis/duration_mae.py:15,172-173's rule)

  TRUTH DEFINITION
    raw-sample         count raw 50 Hz samples carrying the label, / SAMPLING_RATE
    window-proxy       count WINDOWS whose last-sample label is the class,
                       * stride_seconds (duration_mae.py's rule)

All four combinations are reported. They are not equally consequential, and the
point of printing the 2x2 is to show which axis matters:

  - The ESTIMATE axis is inert by construction. Under last-window-wins every window
    survives on exactly `step` samples except the final one, which keeps its full
    `win_len` tail, so the two rules differ by exactly (win_len - step) samples per
    fold, landing on a single class -- 0.5 s per fold, 2.5 s over five folds at 1s.
  - The TRUTH axis is a sampling effect. The window-proxy reads the timeline once
    every `step` samples under the last-sample rule, so it inherits that rule's bias.
    That bias is roughly unbiased in expectation and its magnitude is an empirical
    question, largest in relative terms for the short classes.

The script does NOT assert which axis dominates -- it measures both and prints them
against the model's own duration bias, which is the quantity of interest. If both
measurement axes are small next to that, the conclusion is robust to the choice, and
saying so is worth more than picking a rule and hoping.

CONSERVATION. Each MATCHED pair conserves exactly -- both sides reduce to the same
sample count -- and that is a hard fail, not a tolerance:
    sum(last-window-wins) == sum(raw-sample truth)   == n_covered
    sum(window-count)     == sum(window-proxy truth) == n_windows * step
The two CROSSED pairs differ by exactly n_folds * (win_len - step), a shortfall for
window-count against raw-sample truth and a surplus the other way. That is closed
form, so it is asserted too rather than waved through.

PER-GAME BREAKDOWN IS NOT POSSIBLE, and (c) is per-SUBJECT instead. The 'coarse'
column that distinguishes game / warmup / drill is dropped at
data_creation.py:73 in the same expression that selects the game rows, and
hangtime_game_data.csv is written with only
['location','skill','gender','subject','acc_x','acc_y','acc_z','basketball'].
labels_export.csv.gz carries only [subject, label]. No session identifier survives
anywhere in the pipeline, so one subject == one contiguous block == the finest
granularity available. Note also that data_creation.py:73 selects game rows with a
boolean mask: if a subject's recording interleaves game with drill/warmup, those
rows concatenate across a time SEAM, and windows straddling a seam span a real
discontinuity. 'Duration' here is duration-of-labeled-game-samples, not wall clock.

n = 5 folds. Per-class sign agreement is reported as a count ("over-counted in 5/5
folds") with the per-fold magnitudes beside it. No significance test is computed:
a p-value on five folds would be noise dressed as inference. A class whose sign
FLIPS across folds is called out -- that is a finding, not a rounding artifact.

Baseline-only for now: a cfg with sw_length != 1.0 is a hard fail.

Read-only and CPU-only -- a deliberate departure from analysis/duration_mae.py,
which writes duration_mae.csv into every run folder it is pointed at. This script
writes nothing, loads no checkpoints, and imports nothing from src/.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/boracle_duration_bias.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --labels labels_export.csv.gz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_purity as wp          # noqa: E402  -- module-level defs, __main__-guarded
import sample_level_f1 as slf       # noqa: E402  -- certified reconstruction, reused wholesale

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
HardFail = wp.HardFail
fail = slf.fail

# Hardcoded at preprocess_data.py:31, assigned to args.sampling_rate at main.py:176 --
# after the cfg dump at main.py:158 -- so it never reaches cfg.txt. Reused from
# sample_level_f1 rather than restated, so the two scripts cannot drift apart.
SAMPLING_RATE = slf.SAMPLING_RATE

EXPECTED_FOLD_COUNT = 5
REQUIRED_SW_LENGTH = 1.0

# The five basketball classes. rebound/layup/shot are the jump-and-land events an
# injury model is built around, so their bias direction is called out separately.
BALL_ACTIONS = ("dribbling", "shot", "pass", "rebound", "layup")
EXPLOSIVE = ("shot", "rebound", "layup")

# (estimate rule, truth definition) keys, in report order.
RULES = ("expansion", "window_count")
TRUTHS = ("raw_sample", "window_proxy")
RULE_LABEL = {"expansion": "last-window-wins", "window_count": "window-count x stride"}
TRUTH_LABEL = {"raw_sample": "raw-sample", "window_proxy": "window-proxy"}
MATCHED = (("expansion", "raw_sample"), ("window_count", "window_proxy"))
CROSSED = (("expansion", "window_proxy"), ("window_count", "raw_sample"))


def secs(samples):
    """Sample count -> seconds. n_samples / SAMPLING_RATE, the only conversion used."""
    return np.asarray(samples, dtype=float) / SAMPLING_RATE


def hms(seconds):
    s = int(round(float(seconds)))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# --------------------------------------------------------------------------- #
# per-fold measurement
# --------------------------------------------------------------------------- #

def measure_fold(fold, npz_path, candidates, labels_df, results_dir, win_len, step):
    """
    All four count vectors for one fold, in SAMPLES (seconds conversion is deferred).

    The reconstruction is sample_level_f1's, unchanged: brute-force fold -> subject
    resolution accepted only on window count + exact last-sample-label equality, then
    the same certification restated against the raw timeline.
    """
    y_pred_win, y_true_win = slf.load_npz(npz_path)
    subj = slf.resolve_fold(fold, y_true_win, candidates, results_dir, win_len, step)

    cand = candidates[subj]
    n_samples = cand["n_samples"]

    n_expected = slf.expected_window_count(n_samples, win_len, step)
    if len(y_true_win) != n_expected:
        fail(f"{results_dir} fold {fold} -> {subj}: npz holds {len(y_true_win)} windows, the raw "
             f"{n_samples} samples give {n_expected} at win_len={win_len} step={step}")

    y_true_sample = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy().astype(np.int64)
    if len(y_true_sample) != n_samples:
        fail(f"fold {fold} -> {subj}: raw label slice is {len(y_true_sample)} rows, "
             f"candidate table says {n_samples}")

    idx_last = np.arange(len(y_true_win), dtype=np.int64) * step + win_len - 1
    if not np.array_equal(y_true_sample[idx_last], y_true_win):
        n_bad = int((y_true_sample[idx_last] != y_true_win).sum())
        fail(f"{results_dir} fold {fold} -> {subj}: reconstructed last-sample labels differ from "
             f"the npz y_true at {n_bad} of {len(y_true_win)} windows. Reconstruction not certified.")

    sample_pred = slf.expand_last_window_wins(y_pred_win, n_samples, win_len, step)
    covered = sample_pred != slf.UNCOVERED
    n_cov = int(covered.sum())

    return {
        "fold": fold, "subject": subj,
        "n_samples": n_samples, "n_covered": n_cov, "n_windows": len(y_true_win),
        "npz": os.path.basename(npz_path),
        # truth, in samples
        "raw_sample_true": np.bincount(y_true_sample[covered], minlength=N_CLASSES).astype(np.int64),
        "window_proxy_true": np.bincount(y_true_win, minlength=N_CLASSES).astype(np.int64) * step,
        "raw_full_true": np.bincount(y_true_sample, minlength=N_CLASSES).astype(np.int64),
        # estimates, in samples
        "expansion_est": np.bincount(sample_pred[covered], minlength=N_CLASSES).astype(np.int64),
        "window_count_est": np.bincount(y_pred_win, minlength=N_CLASSES).astype(np.int64) * step,
    }


def measure_fold_dense(fold, npz_path, segments, label_ids, results_dir, min_segment_len):
    """
    One dense fold, in SAMPLES. No expansion and no window proxy: the npz already carries one
    prediction per raw sample, so the estimate is a straight bincount of it and the truth is a
    straight bincount of the npz's y_true -- itself certified sample for sample against the raw
    label stream before either is counted.
    """
    y_pred, y_true = slf.load_npz(npz_path)

    expected = slf.dense_expected_truth(fold, segments, label_ids, min_segment_len)
    if len(y_true) != len(expected):
        fail(f"{results_dir} fold {fold}: npz holds {len(y_true)} samples, the raw stream gives "
             f"{len(expected)} kept samples at min_segment_len={min_segment_len}")
    if not np.array_equal(y_true, expected):
        n_bad = int((y_true != expected).sum())
        fail(f"{results_dir} fold {fold}: npz y_true differs from the raw per-sample labels at "
             f"{n_bad} of {len(expected)} samples. Reconstruction not certified.")

    raw_total = sum(s["length"] for s in segments if s["subject_name"] == fold)
    full = np.concatenate([label_ids[s["start_row"]:s["end_row"]] for s in segments
                           if s["subject_name"] == fold])

    zeros = np.zeros(N_CLASSES, dtype=np.int64)
    return {
        "fold": fold, "subject": fold,
        "n_samples": int(raw_total), "n_covered": int(len(y_true)), "n_windows": 0,
        "npz": os.path.basename(npz_path),
        "raw_sample_true": np.bincount(y_true, minlength=N_CLASSES).astype(np.int64),
        "window_proxy_true": zeros.copy(),
        "raw_full_true": np.bincount(full, minlength=N_CLASSES).astype(np.int64),
        "expansion_est": np.bincount(y_pred, minlength=N_CLASSES).astype(np.int64),
        "window_count_est": zeros.copy(),
    }


def process_dir_dense(results_dir, labels_df, seam, npz_pattern, allow_partial):
    """Dense counterpart of process_dir: no windowing to derive, no candidate table to build."""
    if not os.path.isdir(results_dir):
        fail(f"--results_dir does not exist or is not a directory: {results_dir}")

    cfg = slf.load_cfg(results_dir)
    if not cfg.get("dense", False):
        fail(f"{results_dir}: --dense was passed but this run's cfg.txt has "
             f"dense={cfg.get('dense')!r}. Refusing to read a windowed npz as per-sample.")

    min_segment_len = int(cfg.get("dense_min_seg", 25))
    seq_len = int(cfg.get("dense_seq_len", 500))
    overlap = float(cfg.get("dense_overlap", 0.5))
    segments = seam["segments"]

    label_ids = labels_df["label_id"].to_numpy().astype(np.int64)
    if len(label_ids) != seam["total_samples"]:
        fail(f"labels file holds {len(label_ids)} samples, seam map describes "
             f"{seam['total_samples']}")

    folds = cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)
    expect = len(folds) if allow_partial else EXPECTED_FOLD_COUNT
    if not allow_partial and len(folds) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: cfg loso_subjects lists {len(folds)} folds, expected "
             f"{EXPECTED_FOLD_COUNT}: {folds}. Pass --allow_partial_folds for a smoke test.")
    npz_paths = slf.discover_fold_npz(results_dir, folds, npz_pattern, expect_count=expect)

    fold_rows = [measure_fold_dense(f, npz_paths[f], segments, label_ids, results_dir,
                                    min_segment_len)
                 for f in sorted(npz_paths)]

    totals = {k: np.sum([r[k] for r in fold_rows], axis=0)
              for k in ("raw_sample_true", "window_proxy_true", "raw_full_true",
                        "expansion_est", "window_count_est")}

    return {
        "dir": results_dir, "name": slf.display_name(results_dir), "dense": True,
        "sw_length": seq_len / SAMPLING_RATE, "sw_overlap": int(round(overlap * 100)),
        "win_len": seq_len, "step": int(seq_len * (1 - overlap)),
        "min_segment_len": min_segment_len,
        "stride_sec": 1.0 / SAMPLING_RATE,
        "folds": fold_rows, "totals": totals,
        "n_samples": int(sum(r["n_samples"] for r in fold_rows)),
        "n_covered": int(sum(r["n_covered"] for r in fold_rows)),
        "n_windows": 0,
    }


def process_dir(results_dir, labels_df, npz_pattern):
    if not os.path.isdir(results_dir):
        fail(f"--results_dir does not exist or is not a directory: {results_dir}")

    cfg = slf.load_cfg(results_dir)
    sw_length, sw_overlap, win_len, step = slf.derive_windowing(cfg, results_dir)

    if abs(sw_length - REQUIRED_SW_LENGTH) > 1e-9:
        fail(f"{results_dir}: sw_length is {sw_length}, but this analysis is baseline-only and "
             f"requires {REQUIRED_SW_LENGTH}. The duration arithmetic generalizes, but the BOracle "
             "framing and the recorded anchors do not; re-run against a 1s baseline dir.")

    folds = cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)
    if len(folds) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: cfg loso_subjects lists {len(folds)} folds, expected "
             f"{EXPECTED_FOLD_COUNT}: {folds}")

    npz_paths = slf.discover_fold_npz(results_dir, folds, npz_pattern)
    if len(npz_paths) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: discovered {len(npz_paths)} folds, expected {EXPECTED_FOLD_COUNT}")

    candidates = slf.build_candidates(labels_df, win_len, step)
    fold_rows = [measure_fold(f, npz_paths[f], candidates, labels_df, results_dir, win_len, step)
                 for f in sorted(npz_paths)]

    subjects = [r["subject"] for r in fold_rows]
    if len(set(subjects)) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: {EXPECTED_FOLD_COUNT} folds resolved to {len(set(subjects))} distinct "
             f"subjects {sorted(subjects)} -- the resolution is not a bijection")

    totals = {k: np.sum([r[k] for r in fold_rows], axis=0)
              for k in ("raw_sample_true", "window_proxy_true", "raw_full_true",
                        "expansion_est", "window_count_est")}

    return {
        "dir": results_dir, "name": slf.display_name(results_dir),
        "sw_length": sw_length, "sw_overlap": sw_overlap, "win_len": win_len, "step": step,
        "stride_sec": step / SAMPLING_RATE,
        "folds": fold_rows, "totals": totals,
        "n_samples": int(sum(r["n_samples"] for r in fold_rows)),
        "n_covered": int(sum(r["n_covered"] for r in fold_rows)),
        "n_windows": int(sum(r["n_windows"] for r in fold_rows)),
    }


# --------------------------------------------------------------------------- #
# bias arithmetic
# --------------------------------------------------------------------------- #

def bias_of(est_samples, true_samples):
    """(true_sec, est_sec, bias_sec, bias_pct, ratio). pct/ratio are NaN at zero truth."""
    true_sec, est_sec = secs(true_samples), secs(est_samples)
    bias_sec = est_sec - true_sec
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(true_sec > 0, bias_sec / true_sec * 100.0, np.nan)
        ratio = np.where(true_sec > 0, est_sec / true_sec, np.nan)
    return true_sec, est_sec, bias_sec, pct, ratio


def direction(bias_sec, true_sec, eps=1e-9):
    if not np.isfinite(bias_sec) or true_sec <= 0:
        return "n/a"
    if abs(bias_sec) < eps:
        return "exact"
    return "OVER" if bias_sec > 0 else "UNDER"


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_header(rec):
    print("=" * 108)
    print("BORACLE DURATION BIAS -- per-class exposure estimates vs ground truth")
    print("=" * 108)
    print(f"\n  run            : {rec['name']}   [{rec['dir']}]")
    if rec.get("dense"):
        print(f"  sequencing     : DENSE, seq_len={rec['win_len']} samples "
              f"({rec['sw_length']}s), overlap={rec['sw_overlap']}%, "
              f"min_segment_len={rec['min_segment_len']}")
        print("                   one prediction per raw sample; no expansion, no window proxy")
    else:
        print(f"  windowing      : sw_length={rec['sw_length']}s  sw_overlap={rec['sw_overlap']}%  ->  "
              f"win_len={rec['win_len']} samples, step={rec['step']} samples, "
              f"stride={rec['stride_sec']:.4f}s")
    print(f"  sampling rate  : {SAMPLING_RATE} Hz (preprocess_data.py:31; never in cfg.txt)")
    print(f"  duration       : n_samples / {SAMPLING_RATE}")
    print(f"\n  {'fold':<6} {'subject':<8} {'windows':>8} {'samples':>9} {'covered':>9} "
          f"{'covered_dur':>12}  npz")
    for f in rec["folds"]:
        print(f"  {f['fold']:<6} {f['subject']:<8} {f['n_windows']:>8} {f['n_samples']:>9} "
              f"{f['n_covered']:>9} {hms(secs(f['n_covered'])):>12}  {f['npz']}")
    print(f"  {'TOTAL':<6} {'':<8} {rec['n_windows']:>8} {rec['n_samples']:>9} "
          f"{rec['n_covered']:>9} {hms(secs(rec['n_covered'])):>12}")

    gap = rec["n_samples"] - rec["n_covered"]
    if rec.get("dense"):
        print(f"\n  Uncovered samples: {gap} ({secs(gap):.2f}s) -- segments shorter than "
              f"min_segment_len={rec['min_segment_len']}, which the loader discards. "
              "0 at the default 25.")
    else:
        print(f"\n  Uncovered trailing samples: {gap} ({secs(gap):.2f}s over {EXPECTED_FOLD_COUNT} folds, "
              f"at most `step`={rec['step']} per fold).")
    print("  Every table below scores COVERED samples only, so bias is attributable to")
    print("  misclassification rather than to coverage. The full-raw column in [1] shows what")
    print("  the tail costs -- it is under a second per fold.")


def report_totals(rec):
    """(a) + (b): the headline table, last-window-wins against raw-sample truth."""
    t = rec["totals"]
    true_sec, est_sec, bias_sec, pct, ratio = bias_of(t["expansion_est"], t["raw_sample_true"])
    full_sec = secs(t["raw_full_true"])

    print("\n" + "=" * 108)
    if rec.get("dense"):
        print("[1] PER-CLASS TOTAL DURATION -- per-sample estimate vs raw-sample truth")
        print("=" * 108)
        print("\n  Dense: the estimate is a straight count of the per-sample predictions, with no")
        print("  expansion step between it and the raw 50 Hz timeline. bias = est - true.\n")
    else:
        print("[1] PER-CLASS TOTAL DURATION -- last-window-wins estimate vs raw-sample truth")
        print("=" * 108)
        print("\n  This is the headline pairing: the estimate BOracle would actually consume, against")
        print("  the raw 50 Hz timeline. bias = est - true. ratio = est / true.\n")

    hdr = (f"  {'class':<12} {'true_sec':>10} {'est_sec':>10} {'bias_sec':>10} {'bias_pct':>9} "
           f"{'ratio':>7} {'dir':>6} {'share':>7} {'full_raw_s':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c, name in enumerate(CLASS_NAMES):
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {true_sec[c]:>10.2f} {est_sec[c]:>10.2f} {bias_sec[c]:>+10.2f} "
              f"{pct[c]:>+8.2f}% {ratio[c]:>7.4f} {direction(bias_sec[c], true_sec[c]):>6} "
              f"{true_sec[c] / true_sec.sum():>7.4f} {full_sec[c]:>11.2f}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<12} {true_sec.sum():>10.2f} {est_sec.sum():>10.2f} "
          f"{est_sec.sum() - true_sec.sum():>+10.2f} {'':>9} {'':>7} {'':>6} {1.0:>7.4f} "
          f"{full_sec.sum():>11.2f}")
    print(f"\n  (* = ball action)   total covered duration = {hms(true_sec.sum())}")


def report_matrix(rec):
    """(e): the 2x2, and which axis actually moves the numbers."""
    if rec.get("dense"):
        print("\n" + "=" * 108)
        print("[2] THE 2x2 -- not applicable to a dense run")
        print("=" * 108)
        print("\n  Both axes of the 2x2 are artifacts of windowing. The estimate axis compares")
        print("  last-window-wins against window-count x stride; the truth axis compares the raw")
        print("  timeline against a window-proxy that samples it once per step under the")
        print("  last-sample rule. A dense run has neither: its estimate IS per-sample and its")
        print("  truth IS the raw timeline, so [1] carries no measurement-choice caveat at all.")
        return
    t = rec["totals"]
    print("\n" + "=" * 108)
    print("[2] THE 2x2 -- {estimate rule} x {truth definition}, per-class bias in seconds")
    print("=" * 108)
    print("\n  Columns are the four combinations. The two rightmost columns isolate each axis:")
    print("    rule_delta  = expansion_est - window_count_est   (estimate axis, truth held fixed)")
    print("    truth_delta = raw_sample_true - window_proxy_true (truth axis, estimate held fixed)")
    print("  Compare their magnitudes -- that comparison is the point of this table.\n")

    combos = [(r, tr) for r in RULES for tr in TRUTHS]
    cols = {(r, tr): bias_of(t[f"{r}_est"], t[f"{tr}_true"])[2] for r, tr in combos}
    rule_delta = secs(t["expansion_est"]) - secs(t["window_count_est"])
    truth_delta = secs(t["raw_sample_true"]) - secs(t["window_proxy_true"])

    head = [f"{RULE_LABEL[r].split()[0][:4]}/{TRUTH_LABEL[tr].split('-')[0][:5]}" for r, tr in combos]
    hdr = f"  {'class':<12}" + "".join(f"{h:>14}" for h in head) + f"{'rule_delta':>13}{'truth_delta':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}" + "".join(f"{cols[k][c]:>+14.2f}" for k in combos)
              + f"{rule_delta[c]:>+13.2f}{truth_delta[c]:>+13.2f}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<12}" + "".join(f"{cols[k].sum():>+14.2f}" for k in combos)
          + f"{rule_delta.sum():>+13.2f}{truth_delta.sum():>+13.2f}")
    print(f"  {'max |.|':<12}" + "".join(f"{np.abs(cols[k]).max():>14.2f}" for k in combos)
          + f"{np.abs(rule_delta).max():>13.2f}{np.abs(truth_delta).max():>13.2f}")

    print("\n  legend:")
    for h, (r, tr) in zip(head, combos):
        print(f"    {h:<14} {RULE_LABEL[r]} estimate vs {TRUTH_LABEL[tr]} truth")

    n_rule = int((np.abs(rule_delta) > 1e-9).sum())
    n_truth = int((np.abs(truth_delta) > 1e-9).sum())
    max_rule = float(np.abs(rule_delta).max())
    max_truth = float(np.abs(truth_delta).max())
    headline = np.abs(bias_of(t["expansion_est"], t["raw_sample_true"])[2])
    max_model = float(headline.max())

    print(f"\n  estimate axis : touches {n_rule} of {N_CLASSES} classes, max |delta| = {max_rule:.2f}s")
    print(f"                  every window survives on exactly step={rec['step']} samples except the")
    print(f"                  last of each fold, which keeps its full win_len={rec['win_len']} tail. That is")
    print(f"                  (win_len - step) = {rec['win_len'] - rec['step']} samples = "
          f"{secs(rec['win_len'] - rec['step']):.2f}s per fold, on one class per fold.")
    print(f"  truth axis    : touches {n_truth} of {N_CLASSES} classes, max |delta| = {max_truth:.2f}s")
    print("                  the window-proxy reads the timeline once every `step` samples under the")
    print("                  last-sample rule, so it inherits that rule's sampling bias.")

    bigger = "truth" if max_truth > max_rule else "estimate"
    ratio_txt = (f"{max(max_truth, max_rule) / max(min(max_truth, max_rule), 1e-12):.2f}x"
                 if min(max_truth, max_rule) > 0 else "n/a")
    print(f"\n  Larger measurement axis: {bigger} ({ratio_txt} the other).")
    print(f"  Both are small next to the model's own duration bias, which reaches "
          f"{max_model:.2f}s ({CLASS_NAMES[int(headline.argmax())]}):")
    print(f"    measurement choice  <= {max(max_rule, max_truth):>9.2f}s")
    print(f"    model bias          =  {max_model:>9.2f}s  "
          f"({max_model / max(max_rule, max_truth, 1e-12):.0f}x larger)")
    print("  So the headline direction in [1] does not depend on which rule or truth definition")
    print("  is chosen -- the measurement-choice debate does not reach the conclusion. Both axes")
    print("  also sum to the same small total; they differ only in how they redistribute it.\n")

    print(f"  {'class':<12} {'raw_sample_s':>13} {'window_proxy_s':>15} {'truth_delta_s':>14} {'delta_pct':>10}")
    print("  " + "-" * 66)
    raw_s, proxy_s = secs(t["raw_sample_true"]), secs(t["window_proxy_true"])
    with np.errstate(divide="ignore", invalid="ignore"):
        dpct = np.where(raw_s > 0, truth_delta / raw_s * 100.0, np.nan)
    for c, name in enumerate(CLASS_NAMES):
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {raw_s[c]:>13.2f} {proxy_s[c]:>15.2f} "
              f"{truth_delta[c]:>+14.2f} {dpct[c]:>+9.2f}%")


def report_per_fold(rec):
    """(c): per-subject bias, sign agreement, magnitude spread."""
    t = rec["totals"]
    print("\n" + "=" * 108)
    print("[3] PER-SUBJECT BIAS -- does the direction hold, or does it average out?")
    print("=" * 108)
    print("\n  One column per fold (== per subject; there is no session axis in this dataset --")
    print("  see the module docstring). Values are bias_pct = (est - true) / true * 100 under the")
    print(f"  headline pairing. n = {EXPECTED_FOLD_COUNT}: sign agreement is reported as a count, and")
    print("  no significance test is computed -- a p-value on five folds would be noise.\n")

    folds = rec["folds"]
    per_fold_pct = np.full((N_CLASSES, len(folds)), np.nan)
    for j, f in enumerate(folds):
        _, _, _, pct, _ = bias_of(f["expansion_est"], f["raw_sample_true"])
        per_fold_pct[:, j] = pct

    labels = [f"{f['subject']}" for f in folds]
    hdr = (f"  {'class':<12}" + "".join(f"{l:>11}" for l in labels)
           + f"{'pooled':>10}{'sign':>10}{'spread':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    _, _, pooled_bias, pooled_pct, _ = bias_of(t["expansion_est"], t["raw_sample_true"])
    flips = []
    for c, name in enumerate(CLASS_NAMES):
        vals = per_fold_pct[c]
        usable = vals[np.isfinite(vals)]
        n_over = int((usable > 0).sum())
        n_under = int((usable < 0).sum())
        n_used = len(usable)
        if n_used == 0:
            sign = "n/a"
        elif n_over == n_used:
            sign = f"OVER {n_over}/{n_used}"
        elif n_under == n_used:
            sign = f"UNDER {n_under}/{n_used}"
        else:
            sign = f"FLIPS {n_over}/{n_used}"
            flips.append((name, n_over, n_under, n_used))
        spread = (usable.max() - usable.min()) if n_used else np.nan
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag}" + "".join(f"{v:>+11.2f}" if np.isfinite(v) else f"{'n/a':>11}"
                                             for v in vals)
              + f"{pooled_pct[c]:>+10.2f}{sign:>10}{spread:>10.2f}")

    print(f"\n  'sign' counts folds with the same bias direction; 'spread' is max - min bias_pct")
    print("  across folds, so a large spread with a consistent sign means the direction is robust")
    print("  but the magnitude is not.")

    if flips:
        print("\n  CLASSES WHOSE BIAS DIRECTION FLIPS ACROSS FOLDS -- the pooled number for these is")
        print("  an average of opposing per-subject biases, not a stable property of the model:")
        for name, n_over, n_under, n_used in flips:
            print(f"    {name:<12} over-counted in {n_over}/{n_used} folds, under-counted in "
                  f"{n_under}/{n_used}")
    else:
        print(f"\n  No class flips sign: every class biases the same direction in all "
              f"{EXPECTED_FOLD_COUNT} folds, so")
        print("  the pooled direction in [1] is not an artifact of averaging opposing subjects.")

    n_zero = int((~np.isfinite(per_fold_pct)).sum())
    if n_zero:
        print(f"\n  {n_zero} (class, fold) cell(s) have zero true duration and are excluded from the")
        print("  sign counts rather than counted as agreement.")


def report_conservation(rec):
    """(d): exact within matched pairs, closed form across crossed pairs."""
    t = rec["totals"]
    print("\n" + "=" * 108)
    print("[4] CONSERVATION -- every covered sample carries exactly one prediction")
    print("=" * 108)

    if rec.get("dense"):
        est, true = int(t["expansion_est"].sum()), int(t["raw_sample_true"].sum())
        print("\n  Dense: the estimate and the truth are bincounts of two arrays of equal length,")
        print("  so conservation is exact by construction and is asserted rather than derived.\n")
        print(f"    sum(per-sample estimate) = {est} samples = {secs(est):.2f}s")
        print(f"    sum(raw-sample truth)    = {true} samples = {secs(true):.2f}s")
        print(f"    n_covered                = {rec['n_covered']} samples")
        if not (est == true == rec["n_covered"]):
            fail(f"dense conservation failed: est={est} true={true} n_covered={rec['n_covered']}")
        print("\n  status: MATCHED, exact")
        print(f"\n  Per-fold:")
        print(f"    {'fold':<6} {'subject':<8} {'est_sec':>12} {'true_sec':>12} {'delta':>12}")
        for f in rec["folds"]:
            e, g = secs(f["expansion_est"]).sum(), secs(f["raw_sample_true"]).sum()
            if not np.isclose(e, g, rtol=0, atol=1e-9):
                fail(f"fold {f['fold']}: conservation failed, {e:.6f}s vs {g:.6f}s")
            print(f"    {f['fold']:<6} {f['subject']:<8} {e:>12.4f} {g:>12.4f} {e - g:>+12.6f}")
        return

    gap_samples = EXPECTED_FOLD_COUNT * (rec["win_len"] - rec["step"])
    print(f"\n  n_covered           = {rec['n_covered']} samples = {secs(rec['n_covered']):.2f}s")
    print(f"  n_windows * step    = {rec['n_windows'] * rec['step']} samples = "
          f"{secs(rec['n_windows'] * rec['step']):.2f}s")
    print(f"  difference          = {gap_samples} samples = {secs(gap_samples):.2f}s "
          f"= n_folds * (win_len - step)")

    print(f"\n  {'pairing':<46} {'est_sec':>12} {'truth_sec':>12} {'delta_sec':>12}  status")
    print("  " + "-" * 96)
    problems = []

    for r, tr in MATCHED:
        e, g = secs(t[f"{r}_est"]).sum(), secs(t[f"{tr}_true"]).sum()
        ok = np.isclose(e, g, rtol=0, atol=1e-9)
        print(f"  {RULE_LABEL[r] + ' vs ' + TRUTH_LABEL[tr] + ' truth':<46} {e:>12.4f} {g:>12.4f} "
              f"{e - g:>+12.6f}  {'MATCHED, exact' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{RULE_LABEL[r]} vs {TRUTH_LABEL[tr]} truth: {e:.6f}s vs {g:.6f}s "
                            f"(delta {e - g:+.6f}s); these must be identical by construction")

    for r, tr in CROSSED:
        e, g = secs(t[f"{r}_est"]).sum(), secs(t[f"{tr}_true"]).sum()
        expected = secs(gap_samples) if r == "expansion" else -secs(gap_samples)
        ok = np.isclose(e - g, expected, rtol=0, atol=1e-9)
        word = "surplus" if expected > 0 else "shortfall"
        print(f"  {RULE_LABEL[r] + ' vs ' + TRUTH_LABEL[tr] + ' truth':<46} {e:>12.4f} {g:>12.4f} "
              f"{e - g:>+12.6f}  CROSSED, expected {expected:+.4f} ({word})")
        if not ok:
            problems.append(f"{RULE_LABEL[r]} vs {TRUTH_LABEL[tr]} truth: delta {e - g:+.6f}s, "
                            f"closed form says {expected:+.6f}s")

    if problems:
        fail("conservation failed -- the duration arithmetic does not close:\n    "
             + "\n    ".join(problems))

    print("\n  All four hold. The matched pairs are exact because both sides reduce to the same")
    print("  sample count; the crossed pairs differ by exactly n_folds * (win_len - step), a")
    print("  shortfall for window-count against raw-sample truth because that rule attributes")
    print("  `step` samples to every window including the last, which actually covers `win_len`.")

    print(f"\n  Per-fold, matched pair (last-window-wins vs raw-sample truth):")
    print(f"    {'fold':<6} {'subject':<8} {'est_sec':>12} {'true_sec':>12} {'delta':>12}")
    for f in rec["folds"]:
        e, g = secs(f["expansion_est"]).sum(), secs(f["raw_sample_true"]).sum()
        if not np.isclose(e, g, rtol=0, atol=1e-9):
            fail(f"fold {f['fold']} -> {f['subject']}: covered-sample conservation failed, "
                 f"{e:.6f}s estimated vs {g:.6f}s true")
        print(f"    {f['fold']:<6} {f['subject']:<8} {e:>12.4f} {g:>12.4f} {e - g:>+12.6f}")


def report_boracle(rec):
    """The column that drops into the BOracle interface discussion."""
    t = rec["totals"]
    true_sec, est_sec, bias_sec, pct, ratio = bias_of(t["expansion_est"], t["raw_sample_true"])

    folds = rec["folds"]
    per_fold_pct = np.full((N_CLASSES, len(folds)), np.nan)
    for j, f in enumerate(folds):
        per_fold_pct[:, j] = bias_of(f["expansion_est"], f["raw_sample_true"])[3]

    print("\n" + "=" * 108)
    print("[5] BORACLE INTERFACE -- signed exposure bias per class")
    print("=" * 108)
    print("\n  BOracle consumes these as exposure. OVER inflates estimated exposure (false high")
    print("  risk); UNDER deflates it (MISSED risk). Under-estimation is the dangerous direction.\n")

    contested = []
    hdr = (f"  {'class':<12} {'ratio':>7} {'bias_pct':>9} {'direction':>10} {'folds':>8} "
           f"{'ball':>5} {'exposure effect'}")
    print(hdr)
    print("  " + "-" * (len(hdr) + 12))
    for c, name in enumerate(CLASS_NAMES):
        d = direction(bias_sec[c], true_sec[c])
        vals = per_fold_pct[c][np.isfinite(per_fold_pct[c])]
        agree = int((vals > 0).sum()) if d == "OVER" else int((vals < 0).sum())
        effect = {"OVER": "exposure inflated -> false high risk",
                  "UNDER": "exposure deflated -> MISSED risk",
                  "exact": "unbiased",
                  "n/a": "no true support"}[d]
        # '!' marks a class whose folds do not unanimously back the pooled direction, and
        # '!!' one where most folds actively contradict it -- there the pooled sign is an
        # average of opposing subjects, not a property BOracle can rely on.
        if len(vals) == 0 or agree == len(vals):
            mark = ""
        elif agree * 2 <= len(vals):
            mark = "!!"
            contested.append((name, agree, len(vals)))
        else:
            mark = "!"
            contested.append((name, agree, len(vals)))
        print(f"  {name:<12} {ratio[c]:>7.4f} {pct[c]:>+8.2f}% {d:>10} "
              f"{f'{agree}/{len(vals)}{mark}':>8} {'yes' if name in BALL_ACTIONS else '':>5} {effect}")

    if contested:
        print("\n  ! = folds do not unanimously back the pooled direction;  "
              "!! = most folds contradict it.")
        for name, agree, n in contested:
            note = "  <- pooled sign is an averaging artifact" if agree * 2 <= n else ""
            print(f"    {name:<12} only {agree}/{n} folds agree with the pooled direction{note}")
        print("  Treat these classes' pooled bias as unstable; see [3] for the per-fold values.")

    under = [CLASS_NAMES[c] for c in range(N_CLASSES)
             if direction(bias_sec[c], true_sec[c]) == "UNDER"]
    expl_under = [n for n in under if n in EXPLOSIVE]

    print("\n  Ball actions, by direction:")
    for name in BALL_ACTIONS:
        c = CLASS_NAMES.index(name)
        mark = "  <- explosive (jump/land)" if name in EXPLOSIVE else ""
        print(f"    {name:<12} {direction(bias_sec[c], true_sec[c]):<6} "
              f"{pct[c]:>+8.2f}%  est {est_sec[c]:>8.2f}s vs true {true_sec[c]:>8.2f}s{mark}")

    if expl_under:
        print(f"\n  {len(expl_under)} of {len(EXPLOSIVE)} explosive ball action(s) are UNDER-estimated: "
              f"{', '.join(expl_under)}.")
        print("  These are the injury-relevant events, and under-estimation understates exposure to")
        print("  them -- the direction an injury pipeline should not be wrong in. Report this as a")
        print("  known bias of the interface, not as a metric to be tuned away.")
    else:
        print(f"\n  No explosive ball action ({', '.join(EXPLOSIVE)}) is under-estimated in this run.")

    print("\n  Note the asymmetry between duration bias and F1: they answer different questions.")
    print("  A class can hold a poor F1 with near-zero duration bias (errors cancel) or a decent")
    print("  F1 with large duration bias (errors are one-directional). Do not substitute one for")
    print("  the other in the BOracle discussion.")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True, metavar="DIR",
                        help="Baseline run dir holding cfg.txt and the 5 per-fold npz files. "
                             "sw_length must be 1.0.")
    parser.add_argument("--labels", default="labels_export.csv.gz",
                        help="Exported per-sample labels (columns: subject, label), original row order.")
    parser.add_argument("--npz_pattern", default=None,
                        help="Explicit glob containing '{fold}', overriding npz auto-detection.")
    parser.add_argument("--dense", action="store_true",
                        help="Read a --dense run: durations come straight from the per-sample "
                             "predictions, and the window-proxy / window-count legs (table [2]) "
                             "are skipped because neither exists for a dense run.")
    parser.add_argument("--seam_map", default="data/seam_map.json",
                        help="Seam map used by --dense to rebuild each fold's kept-sample stream.")
    parser.add_argument("--allow_partial_folds", action="store_true",
                        help="Read a run with fewer than 5 folds (a smoke test).")
    args = parser.parse_args()

    labels_df = wp.load_labels(args.labels)
    if args.dense:
        seam = slf.load_seam_map(args.seam_map)
        rec = process_dir_dense(args.results_dir, labels_df, seam, args.npz_pattern,
                                args.allow_partial_folds)
    else:
        rec = process_dir(args.results_dir, labels_df, args.npz_pattern)

    report_header(rec)
    report_totals(rec)
    report_matrix(rec)
    report_per_fold(rec)
    report_conservation(rec)
    report_boracle(rec)

    print("\n" + "=" * 108)
    print("Read [1] for the headline bias, [2] for which measurement choice it depends on,")
    print("[3] for whether it is a per-subject property or an averaging artifact, [5] for the")
    print("direction that matters downstream. Nothing was written; this script is read-only.")
    print("=" * 108)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
