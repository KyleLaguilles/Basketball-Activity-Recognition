#!/usr/bin/env python3
"""
Pooled per-class precision / recall / F1 across all LOSO held-out subjects.

Usage (from repo root, hangtime_har env):
    python analysis/pooled_per_class.py <log_dir>

Reads predictions_best_*.csv files written by validation.py --save_predictions.
Column layout per file (after index_col=0):
    iloc[:, 0]  ->  y_pred   (argmax predictions)
    iloc[:, 1]  ->  y_true   (ground-truth labels)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = [
    "dribbling", "shot", "pass", "rebound", "layup",
    "walking", "running", "standing", "sitting",
]
N_CLASSES = len(CLASS_NAMES)
EXPECTED_SUBJECTS = 14


def parse_subjects(paths):
    """
    Recover subject IDs from filenames: predictions_best_{subject}_{runname}.csv
    All files in one log dir share the same run name, so the longest common
    _-delimited suffix across all bare names is the run name; the rest is the subject.
    Returns None when parsing is ambiguous (empty IDs or duplicates).
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    bare = [s[len("predictions_best_"):] for s in stems]
    parts_list = [b.split("_") for b in bare]
    n_min = min(len(p) for p in parts_list)

    suffix_len = 0
    for i in range(1, n_min):           # probe 1, 2, ... trailing tokens
        tail = "_".join(parts_list[0][-i:])
        if all("_".join(p[-i:]) == tail for p in parts_list):
            suffix_len = i              # still common — extend
        else:
            break                       # mismatch; can't go further

    if suffix_len == 0:
        subjects = ["_".join(p) for p in parts_list]
    else:
        subjects = ["_".join(p[:-suffix_len]) for p in parts_list]

    if any(s == "" for s in subjects) or len(set(subjects)) != len(subjects):
        return None
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Compute pooled per-class metrics from LOSO prediction CSVs."
    )
    parser.add_argument("log_dir", help="Directory containing predictions_best_*.csv files")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        sys.exit(f"Error: {log_dir!r} is not a directory.")

    # Strict: best only, never last
    pattern = os.path.join(log_dir, "predictions_best_*.csv")
    paths = sorted(
        p for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("predictions_last_")
    )

    print(f"Found {len(paths)} predictions_best_*.csv file(s) in: {log_dir}")
    if len(paths) != EXPECTED_SUBJECTS:
        print(
            f"WARNING: expected {EXPECTED_SUBJECTS} files (one per subject), "
            f"got {len(paths)}. Check that the run is complete and you are "
            "pointing at the correct directory."
        )
    print()

    y_true_parts = []
    y_pred_parts = []

    for p in paths:
        df = pd.read_csv(p, index_col=0)
        # np.vstack((val_preds, val_gt)).T  ->  col 0 = y_pred, col 1 = y_true
        y_pred_parts.append(df.iloc[:, 0].to_numpy().astype(int))
        y_true_parts.append(df.iloc[:, 1].to_numpy().astype(int))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    # Per-class metrics on pooled arrays
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        zero_division=np.nan,
    )
    support = np.bincount(y_true, minlength=N_CLASSES)

    zero_mask = support == 0
    n_excluded = int(zero_mask.sum())
    macro_f1 = float(np.nanmean(f1))

    # Table
    name_w = max(len(n) for n in CLASS_NAMES)
    sep = "-" * (3 + 2 + name_w + 2 + 7 + 2 + 9 + 2 + 6 + 2 + 6 + 14)
    print(f"{'idx':>3}  {'class':<{name_w}}  {'support':>7}  {'precision':>9}  {'recall':>6}  {'F1':>6}")
    print(sep)
    for i in range(N_CLASSES):
        tag = "  [no support]" if zero_mask[i] else ""
        p_str = f"{prec[i]:.4f}" if not np.isnan(prec[i]) else "     nan"
        r_str = f"{rec[i]:.4f}" if not np.isnan(rec[i]) else "     nan"
        f_str = f"{f1[i]:.4f}" if not np.isnan(f1[i]) else "     nan"
        print(f"{i:>3}  {CLASS_NAMES[i]:<{name_w}}  {support[i]:>7}  {p_str:>9}  {r_str:>6}  {f_str:>6}{tag}")
    print(sep)
    print(f"\nMacro-F1 (nanmean over per-class F1): {macro_f1:.4f}")
    if n_excluded:
        print(f"  {n_excluded} class(es) with zero pooled support excluded from macro-F1")
    else:
        print("  All 9 classes had pooled support — no exclusions.")

    # Best-effort: per-subject support CSV
    try:
        subjects = parse_subjects(paths)
        if subjects is None:
            raise ValueError("ambiguous subject IDs")
        rows = []
        for subj, yt in zip(subjects, y_true_parts):
            counts = np.bincount(yt, minlength=N_CLASSES)
            rows.append([subj] + counts.tolist())
        subj_df = (
            pd.DataFrame(rows, columns=["subject"] + CLASS_NAMES)
            .set_index("subject")
        )
        out_path = os.path.join(log_dir, "support_by_subject.csv")
        subj_df.to_csv(out_path)
        print(f"\nPer-subject support written to: {out_path}")
    except Exception:
        pass  # silently skip — pooled table is the deliverable


if __name__ == "__main__":
    main()
