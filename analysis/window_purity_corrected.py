#!/usr/bin/env python3
"""
Corrected per-window label purity, replacing analysis/purity.py's containment-
denominator computation (quarantined -- see the Phase 1 investigation that
preceded this script; the "88"/"40" numbers it produced must not be reused).

Purity(window) = majority_class_sample_count / window_length, using the SAME
window_majority()/is_pure() helpers from _loso_common.py that every other
purity-consuming script in this repo already uses (purity_stratified_scoring.py,
per_subject_error_breakdown.py, walking_signal_energy.py, shot_confusion.py) --
this is not a third purity implementation, just a new caller of the existing one.

"Class" for the per-class breakdown = the window's own majority-assigned label
(via window_majority()) -- analogous to purity.py's ASSIGNMENT table's concept
of "windows assigned to class c", but majority-based rather than last-sample-
based (there is no y_true here to group by; nothing is predicted, this is a
raw-label/windowing-only analysis). No containment-style denominator anywhere.

Windowing: win_len/step from --sw-length/--sw-overlap/--sampling-rate (defaults
1.0s/50%/50Hz -- confirmed identical to purity.py's own SW_LENGTH/SW_UNIT/
SW_OVERLAP/SAMPLING_RATE constants), grouped by the TRUE subject column (column
3 of the raw CSV, via _loso_common.load_game_labels' existing subject
extraction) -- deliberately NOT purity.py's own subject-grouping, which
factorized column 0 ("location", only 'us'/'eu' -- see data_creation.py's
column selection) instead of column 3 ("subject"). Replicating that grouping
would merge multiple real subjects' time series into two giant pseudo-subjects
and produce nonsense windows at real-subject boundaries; using the correct
column is a second, previously-unflagged deviation from purity.py beyond the
denominator fix -- flagged here rather than silently carried over.

Usage (from repo root, hangtime_har env):
    python analysis/window_purity_corrected.py <hangtime_csv_path> \
        [--sw-length 1.0] [--sw-overlap 50] [--sampling-rate 50] \
        [--label drill] [--output-dir .]
"""

import argparse
import os
import sys

import pandas as pd

from _loso_common import (
    CLASS_NAMES, N_CLASSES, PURITY_EPS,
    load_game_labels, make_windows, window_majority, is_pure,
)


def main():
    parser = argparse.ArgumentParser(
        description="Corrected (per-window-sample-fraction) purity distribution, replacing purity.py."
    )
    parser.add_argument("csv_path", help="Path to a hangtime_*_data.csv file (drill, warmup, or game)")
    parser.add_argument("--sw-length", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--sw-overlap", type=float, default=50, help="Overlap percentage")
    parser.add_argument("--sampling-rate", type=int, default=50, help="Sampling rate in Hz")
    parser.add_argument("--label", default=None, help="Friendly label for output (default: csv basename)")
    parser.add_argument("--output-dir", default=".", help="Directory to write CSVs to (default: .)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        sys.exit(f"Error: {args.csv_path!r} not found.")
    label = args.label or os.path.splitext(os.path.basename(args.csv_path))[0]

    win_len = int(args.sw_length * args.sampling_rate)
    step = win_len - int((args.sw_overlap / 100) * win_len)
    print(f"[{label}] windowing: sw_length={args.sw_length}s sw_overlap={args.sw_overlap}% "
          f"sampling_rate={args.sampling_rate}Hz -> win_len={win_len} samples, step={step} samples "
          f"({step / args.sampling_rate:.4f}s stride)")

    df = load_game_labels(args.csv_path)
    subjects = sorted(df["subject"].unique().tolist())
    print(f"[{label}] loaded {len(df)} samples, {len(subjects)} subjects from {args.csv_path!r} "
          f"(grouped by column 3 / true subject id, NOT purity.py's column-0 'location' grouping)")

    rows = []
    for subj in subjects:
        labels = df.loc[df["subject"] == subj, "label"].to_numpy()
        windows = make_windows(labels, win_len, step)
        for w in windows:
            maj_label, maj_count = window_majority(w)
            purity = maj_count / win_len
            rows.append({"subject": subj, "majority_label": maj_label, "purity": purity})

    win_df = pd.DataFrame(rows)
    n_windows_total = len(win_df)
    print(f"[{label}] reconstructed {n_windows_total} windows total")

    # Phase 3 sanity checks
    assert win_df["purity"].between(0, 1, inclusive="both").all(), \
        "Internal error: found a purity value outside [0,1]."

    summary_rows = []
    for c in range(N_CLASSES):
        sub = win_df[win_df["majority_label"] == c]
        n = len(sub)
        if n == 0:
            summary_rows.append({"class": CLASS_NAMES[c], "n_windows": 0,
                                  "mean_purity": float("nan"), "pct_100_pure": float("nan"),
                                  "pct_below_0.5": float("nan")})
            continue
        mean_purity = float(sub["purity"].mean())
        pct_pure = 100.0 * int(sub["purity"].apply(is_pure).sum()) / n
        pct_below_half = 100.0 * int((sub["purity"] < 0.5 - PURITY_EPS).sum()) / n
        summary_rows.append({"class": CLASS_NAMES[c], "n_windows": n,
                              "mean_purity": mean_purity, "pct_100_pure": pct_pure,
                              "pct_below_0.5": pct_below_half})
    summary_df = pd.DataFrame(summary_rows)

    assert int(summary_df["n_windows"].sum()) == n_windows_total, \
        "Internal error: per-class window counts don't sum to the total window count."

    print(f"\n[{label}] PURITY DISTRIBUTION (class = window's own majority-assigned label)")
    name_w = max(len(c) for c in CLASS_NAMES)
    header = (f"{'class':<{name_w}}  {'n_windows':>9}  {'mean_purity':>11}  "
              f"{'pct_100_pure':>12}  {'pct_below_0.5':>13}")
    print(header)
    print("-" * len(header))
    for _, row in summary_df.iterrows():
        mp = f"{row['mean_purity']:.4f}" if pd.notna(row["mean_purity"]) else "nan"
        pp = f"{row['pct_100_pure']:.2f}%" if pd.notna(row["pct_100_pure"]) else "nan"
        pb = f"{row['pct_below_0.5']:.2f}%" if pd.notna(row["pct_below_0.5"]) else "nan"
        print(f"{row['class']:<{name_w}}  {row['n_windows']:>9}  {mp:>11}  {pp:>12}  {pb:>13}")

    try:
        summary_path = os.path.join(args.output_dir, f"window_purity_corrected_{label}.csv")
        per_window_path = os.path.join(args.output_dir, f"window_purity_corrected_{label}_per_window.csv")
        summary_df.to_csv(summary_path, index=False)
        win_df.to_csv(per_window_path, index=False)
        print(f"\nWrote {summary_path} and {per_window_path}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write CSV outputs: {e}")

    return summary_df, win_df, n_windows_total


if __name__ == "__main__":
    main()
