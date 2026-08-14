#!/usr/bin/env python3
"""
Duration-MAE and F1 comparison across loss configurations and median-filter
smoothing widths, to inform whether cheap post-hoc smoothing captures most of
the hypothesized temporal-action-localization (TAL) gain.

Reads predictions_best_*.csv the same way analysis/pooled_per_class.py does
(row order within a subject's file is temporal order -- see this script's
introduction message for the evidence trail) plus each run's cfg.txt (written
automatically by main.py) for that run's actual sw_length/sw_unit/sw_overlap.
sampling_rate is hardcoded at 50 Hz, matching preprocess_data.py -- cfg.txt is
written before load_dataset() runs and never contains sampling_rate.

Definitions:
    duration(subject, class) = window_count(subject, class) * stride_seconds
    duration MAE(class) = mean over subjects with nonzero TRUE support for
        that class of |pred_duration - true_duration|, in seconds
    rel_error(class) = (total_pred_duration - total_true_duration) / total_true_duration,
        POOLED across all subjects (sums, not per-subject average) -- signed, so it
        shows under- vs over-prediction direction, unlike MAE. NaN when the pooled
        true duration for that class is 0 (never inf/0-masquerading-as-real, and the
        affected classes are printed, not silently produced).
    smoothing = per-subject temporal mode filter on the PREDICTED sequence
        only (never on y_true), widths 3/5/7, ties broken by keeping the
        center (original) label, asymmetric/shrinking window at sequence
        boundaries (no padding/wraparound -- an explicit choice, not spec'd).

Usage (from repo root, hangtime_har env):
    python analysis/duration_mae.py <run_dir_1> <run_dir_2> [<run_dir_3> ...] \
        [--labels weighted unweighted sqrt]

Writes duration_mae.csv into every run folder, plus a combined
duration_mae_summary_grid.csv into the last folder passed.
"""

import argparse
import glob
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from _loso_common import CLASS_NAMES, N_CLASSES, EXPECTED_SUBJECTS, parse_subjects

SAMPLING_RATE = 50  # hardcoded pipeline-wide (preprocess_data.py:31); never in cfg.txt
SMOOTHING_WIDTHS = [3, 5, 7]
CONDITIONS = ["raw"] + [f"smooth_{w}" for w in SMOOTHING_WIDTHS]
WALKING_IDX = CLASS_NAMES.index("walking")


def read_run_config(run_dir):
    path = os.path.join(run_dir, "cfg.txt")
    if not os.path.isfile(path):
        sys.exit(f"Error: {path!r} not found -- cannot determine this run's window "
                 "stride without it. Not guessing.")
    with open(path) as f:
        cfg = json.load(f)
    required = ["sw_length", "sw_unit", "sw_overlap"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Error: {path!r} is missing required key(s): {missing}")
    if cfg["sw_unit"] not in ("seconds", "units"):
        sys.exit(f"Error: {path!r} has unrecognized sw_unit {cfg['sw_unit']!r}.")
    return cfg


def stride_seconds_from_cfg(cfg):
    """Mirrors sliding_window.py's win_len/step arithmetic exactly."""
    if cfg["sw_unit"] == "seconds":
        win_len = int(cfg["sw_length"] * SAMPLING_RATE)
    else:  # 'units' -- sw_length is already a sample count
        win_len = int(cfg["sw_length"])
    step = win_len - int((cfg["sw_overlap"] / 100) * win_len)
    return step / SAMPLING_RATE, win_len, step


def load_run_predictions(run_dir):
    """Returns dict: subject -> (y_true, y_pred) arrays, in file row order
    (== temporal order per subject, see module docstring)."""
    pattern = os.path.join(run_dir, "predictions_best_*.csv")
    paths = sorted(
        p for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("predictions_last_")
    )
    if len(paths) != EXPECTED_SUBJECTS:
        sys.exit(f"Error: {run_dir!r} has {len(paths)} predictions_best_*.csv file(s), "
                 f"expected {EXPECTED_SUBJECTS}. Pooled totals computed from a partial "
                 "subject set would corrupt duration totals and relative error. Not proceeding.")
    subjects = parse_subjects(paths)
    if subjects is None:
        sys.exit(f"Error: could not unambiguously parse subject IDs from filenames in {run_dir!r}.")

    out = {}
    for path, subj in zip(paths, subjects):
        df = pd.read_csv(path, index_col=0)
        y_pred = df.iloc[:, 0].to_numpy().astype(int)
        y_true = df.iloc[:, 1].to_numpy().astype(int)
        out[subj] = (y_true, y_pred)
    return out


def mode_filter(seq, width):
    """Per-position temporal mode filter. Ties -> keep the center (original)
    label. Boundaries: asymmetric/shrinking window, no padding."""
    n = len(seq)
    half = (width - 1) // 2
    out = seq.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        counts = Counter(seq[lo:hi].tolist())
        max_count = max(counts.values())
        tied = [lbl for lbl, c in counts.items() if c == max_count]
        out[i] = tied[0] if len(tied) == 1 else seq[i]
    return out


def verify_cross_run_alignment(run_data):
    """run_data: list of (label, {subject: (y_true, y_pred)}). Asserts every
    run has the identical subject set and identical y_true sequence (in order)
    per subject. Hard-fails with specifics otherwise."""
    label0, preds0 = run_data[0]
    set0 = set(preds0.keys())
    problems = []
    for label, preds in run_data[1:]:
        set_cur = set(preds.keys())
        if set_cur != set0:
            problems.append(f"  subject sets differ: {label0!r} has {sorted(set0 - set_cur)} "
                             f"not in {label!r}; {label!r} has {sorted(set_cur - set0)} not in {label0!r}")
            continue
        for subj in set0:
            yt0, _ = preds0[subj]
            ytc, _ = preds[subj]
            if len(yt0) != len(ytc):
                problems.append(f"  subject {subj!r}: {label0!r} has {len(yt0)} windows, "
                                 f"{label!r} has {len(ytc)} windows")
            elif not np.array_equal(yt0, ytc):
                n_diff = int((yt0 != ytc).sum())
                problems.append(f"  subject {subj!r}: y_true differs at {n_diff} window(s) "
                                 f"between {label0!r} and {label!r}")
    if problems:
        sys.exit("Error: runs do not share the exact same test windows/labels -- this is a "
                 "data bug, not a modeling result. Stopping before scoring.\n" + "\n".join(problems))
    print(f"Cross-run alignment check passed: all {len(run_data)} runs share identical "
          f"subject sets and identical y_true sequences.")


def pooled_f1(y_true_all, y_pred_all):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=list(range(N_CLASSES)), zero_division=np.nan,
    )
    return prec, rec, f1


def build_condition_table(preds, condition, stride_seconds):
    """preds: dict subject -> (y_true, y_pred_raw). Returns (per_class_df, macro_f1,
    y_true_all, y_pred_cond_all)."""
    subj_list = list(preds.keys())
    true_durations = np.zeros((len(preds), N_CLASSES))
    pred_durations = np.zeros((len(preds), N_CLASSES))
    y_true_all, y_pred_all = [], []
    for i, (subj, (y_true, y_pred)) in enumerate(preds.items()):
        if condition == "raw":
            y_pred_cond = y_pred
        else:
            width = int(condition.split("_")[1])
            y_pred_cond = mode_filter(y_pred, width)
        true_durations[i] = np.bincount(y_true, minlength=N_CLASSES) * stride_seconds
        pred_durations[i] = np.bincount(y_pred_cond, minlength=N_CLASSES) * stride_seconds
        y_true_all.append(y_true)
        y_pred_all.append(y_pred_cond)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    prec, rec, f1 = pooled_f1(y_true_all, y_pred_all)

    total_true = true_durations.sum(axis=0)
    total_pred = pred_durations.sum(axis=0)
    n_subjects_used = (true_durations > 0).sum(axis=0)
    abs_err = np.abs(pred_durations - true_durations)
    abs_err_masked = np.where(true_durations > 0, abs_err, np.nan)
    with warnings.catch_warnings():
        # classes with zero true support anywhere are legitimately all-NaN columns;
        # nanmean's "Mean of empty slice" warning is expected here, not a bug
        warnings.simplefilter("ignore", category=RuntimeWarning)
        duration_mae = np.nanmean(abs_err_masked, axis=0)

    # pooled signed relative error: (total_pred - total_true) / total_true, per class.
    # NaN (not inf/0) whenever total_true[c] == 0, reported below rather than silently produced.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = (total_pred - total_true) / total_true
    rel_error = np.where(total_true > 0, rel_error, np.nan)

    zero_true_classes = [CLASS_NAMES[c] for c in range(N_CLASSES) if total_true[c] == 0]
    if zero_true_classes:
        print(f"  NOTE: {len(zero_true_classes)} class(es) with zero pooled true duration "
              f"(rel_error=NaN): {zero_true_classes}")

    # per (condition, class): which subject IDs were excluded from MAE/rel_error
    # (zero true duration for that subject/class), not just the count
    for c in range(N_CLASSES):
        excluded = [subj_list[i] for i in range(len(subj_list)) if true_durations[i, c] == 0]
        if excluded:
            print(f"  excluded from {CLASS_NAMES[c]} (zero true duration): {excluded}")

    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "true_duration_sec": total_true,
        "pred_duration_sec": total_pred,
        "duration_mae_sec": duration_mae,
        "n_subjects_used": n_subjects_used,
        "precision": prec, "recall": rec, "f1": f1,
        "rel_error": rel_error,
    })
    macro_f1 = float(np.nanmean(f1))
    return df, macro_f1, y_true_all, y_pred_all


def print_condition_table(run_label, condition, df, macro_f1):
    print(f"\n[{run_label}] condition={condition}")
    name_w = max(len(c) for c in CLASS_NAMES)
    header = (f"{'class':<{name_w}}  {'true_dur_s':>10}  {'pred_dur_s':>10}  "
              f"{'dur_mae_s':>9}  {'n_subj':>6}  {'f1':>6}  {'rel_err':>8}")
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        f1_str = f"{row['f1']:.4f}" if pd.notna(row["f1"]) else "nan"
        mae_str = f"{row['duration_mae_sec']:.3f}" if pd.notna(row["duration_mae_sec"]) else "nan"
        rel_str = f"{row['rel_error']:+.3f}" if pd.notna(row["rel_error"]) else "nan"
        print(f"{row['class']:<{name_w}}  {row['true_duration_sec']:>10.2f}  "
              f"{row['pred_duration_sec']:>10.2f}  {mae_str:>9}  {int(row['n_subjects_used']):>6}  "
              f"{f1_str:>6}  {rel_str:>8}")
    print(f"Macro-F1 (nanmean): {macro_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Duration-MAE and F1 across runs and median-filter smoothing widths."
    )
    parser.add_argument("run_dirs", nargs="+", help="Two or more run folders")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Friendly labels for each run_dir, same count/order (default: folder basenames)")
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("at least two run_dirs are required.")
    for d in args.run_dirs:
        if not os.path.isdir(d):
            sys.exit(f"Error: {d!r} is not a directory.")

    if args.labels is not None:
        if len(args.labels) != len(args.run_dirs):
            parser.error(f"--labels count ({len(args.labels)}) must match run_dirs count ({len(args.run_dirs)}).")
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
        if len(set(labels)) != len(labels):
            parser.error(f"folder basenames are not unique ({labels}); pass --labels to disambiguate.")

    run_data = []
    run_strides = {}
    for label, d in zip(labels, args.run_dirs):
        cfg = read_run_config(d)
        stride_seconds, win_len, step = stride_seconds_from_cfg(cfg)
        print(f"[{label}] cfg.txt: sw_length={cfg['sw_length']} sw_unit={cfg['sw_unit']!r} "
              f"sw_overlap={cfg['sw_overlap']} -> win_len={win_len} samples, step={step} samples, "
              f"stride={stride_seconds:.4f}s (sampling_rate={SAMPLING_RATE}Hz, hardcoded)")
        run_strides[label] = stride_seconds
        preds = load_run_predictions(d)
        print(f"[{label}] loaded {len(preds)} subjects from {d!r}")
        run_data.append((label, preds))

    verify_cross_run_alignment(run_data)

    summary_rows = []
    for (label, preds), run_dir in zip(run_data, args.run_dirs):
        stride_seconds = run_strides[label]
        rows_for_csv = []
        print("\n" + "=" * 100)
        print(f"RUN: {label}  (stride={stride_seconds:.4f}s)")
        print("=" * 100)
        for condition in CONDITIONS:
            df, macro_f1, _, _ = build_condition_table(preds, condition, stride_seconds)
            print_condition_table(label, condition, df, macro_f1)

            df_csv = df.copy()
            df_csv.insert(0, "condition", condition)
            rows_for_csv.append(df_csv)

            classes_with_mae = df["duration_mae_sec"].dropna()
            macro_duration_mae = float(classes_with_mae.mean()) if len(classes_with_mae) else float("nan")
            walking_row = df[df["class"] == "walking"].iloc[0]
            rebound_row = df[df["class"] == "rebound"].iloc[0]
            summary_rows.append({
                "run": label, "condition": condition,
                "macro_duration_mae_sec": macro_duration_mae,
                "walking_duration_mae_sec": walking_row["duration_mae_sec"],
                "macro_f1": macro_f1,
                "rebound_rel_error": rebound_row["rel_error"],
                "walking_rel_error": walking_row["rel_error"],
            })

        try:
            out_df = pd.concat(rows_for_csv, ignore_index=True)
            out_path = os.path.join(run_dir, "duration_mae.csv")
            out_df.to_csv(out_path, index=False)
            print(f"\nWrote duration_mae.csv to {run_dir}")
        except Exception as e:
            print(f"\n(non-fatal) failed to write duration_mae.csv for {label}: {e}")

    print("\n" + "=" * 100)
    print("SUMMARY GRID -- macro duration MAE, walking duration MAE, macro-F1, per run x condition")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_rows)
    lab_w = max(len(l) for l in labels)
    header = (f"{'run':<{lab_w}}  {'condition':<10}  {'macro_dur_mae_s':>15}  "
              f"{'walk_dur_mae_s':>14}  {'macro_f1':>8}  {'rebound_rel_err':>16}  {'walk_rel_err':>13}")
    print(header)
    print("-" * len(header))
    for _, row in summary_df.iterrows():
        rebound_rel_str = f"{row['rebound_rel_error']:+.3f}" if pd.notna(row["rebound_rel_error"]) else "nan"
        walk_rel_str = f"{row['walking_rel_error']:+.3f}" if pd.notna(row["walking_rel_error"]) else "nan"
        print(f"{row['run']:<{lab_w}}  {row['condition']:<10}  {row['macro_duration_mae_sec']:>15.3f}  "
              f"{row['walking_duration_mae_sec']:>14.3f}  {row['macro_f1']:>8.4f}  "
              f"{rebound_rel_str:>16}  {walk_rel_str:>13}")

    last_dir = args.run_dirs[-1]
    try:
        out_path = os.path.join(last_dir, "duration_mae_summary_grid.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"\nWrote duration_mae_summary_grid.csv to {last_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write duration_mae_summary_grid.csv: {e}")


if __name__ == "__main__":
    main()
