#!/usr/bin/env python3
"""
Test whether subject "difficulty" ordering persists across two runs (e.g. two
loss configurations), using per_subject_error_breakdown.csv (written by
analysis/per_subject_error_breakdown.py) from each run's folder.

"Difficulty rank" convention (rank 1 = hardest subject), applied consistently
so the two metrics are comparable in the printed tables:
    - walking->standing rate (w2s_rate_pct): higher = worse -> rank 1 = highest value
    - walking recall:                        lower  = worse -> rank 1 = lowest value
Ties use standard competition ranking (method='min').

The Spearman correlations themselves are computed on the raw per-subject
values directly (not on the display ranks above), so the rank-direction
convention above has no effect on the correlation numbers -- it only affects
how "rank" is displayed in the printed tables.

Usage (from repo root, hangtime_har env):
    python analysis/subject_ranking_persistence.py <run_a_dir> <run_b_dir> \
        [--input-csv per_subject_error_breakdown.csv]

Output CSV is always written into run_b_dir (the second folder passed),
consistent with how the other analysis/ scripts write into the run folder
they're most "about".
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXPECTED_SUBJECTS = 24
RANK_MOVE_FLAG_THRESHOLD = 3

# metric -> (column name, True if higher value = harder/worse)
METRICS = {
    "walking->standing rate": ("w2s_rate_pct", True),
    "walking recall": ("walking_recall", False),
}


def load_breakdown(run_dir, input_csv):
    path = os.path.join(run_dir, input_csv)
    if not os.path.isfile(path):
        sys.exit(f"Error: {path!r} not found. Run analysis/per_subject_error_breakdown.py "
                 "on this folder first.")
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        sys.exit(f"Error: {path!r} has no 'subject' column.")
    if df["subject"].duplicated().any():
        dupes = sorted(df.loc[df["subject"].duplicated(), "subject"].unique().tolist())
        sys.exit(f"Error: {path!r} has duplicate subject rows: {dupes}")
    return df, path


def difficulty_rank(series, higher_is_worse):
    return series.rank(ascending=not higher_is_worse, method="min")


def print_ranking_table(merged, metric_label, col_a, col_b, rank_a_col, rank_b_col, delta_col):
    name_w = max(len(str(s)) for s in merged["subject"])
    header = (f"{'subject':<{name_w}}  {'val_A':>10}  {'rank_A':>6}  "
              f"{'val_B':>10}  {'rank_B':>6}  {'delta':>6}")
    print(f"\n{metric_label} -- subject, value/rank in run A, value/rank in run B, rank delta")
    print("sorted by |rank delta| descending")
    print(header)
    print("-" * len(header))
    ordered = merged.reindex(merged[delta_col].abs().sort_values(ascending=False).index)
    for _, row in ordered.iterrows():
        print(f"{str(row['subject']):<{name_w}}  {row[col_a]:>10.4f}  {int(row[rank_a_col]):>6}  "
              f"{row[col_b]:>10.4f}  {int(row[rank_b_col]):>6}  {int(row[delta_col]):>+6}")

    flagged = ordered[ordered[delta_col].abs() > RANK_MOVE_FLAG_THRESHOLD]
    if len(flagged):
        print(f"\nFLAGGED (rank moved by more than {RANK_MOVE_FLAG_THRESHOLD} positions):")
        for _, row in flagged.iterrows():
            print(f"  {row['subject']}: rank {int(row[rank_a_col])} -> {int(row[rank_b_col])} "
                  f"(delta={int(row[delta_col]):+d})")
    else:
        print(f"\nNo subject moved by more than {RANK_MOVE_FLAG_THRESHOLD} positions.")


def main():
    parser = argparse.ArgumentParser(
        description="Spearman rank-persistence of per-subject error rates between two runs."
    )
    parser.add_argument("run_a", help="First run folder (contains per_subject_error_breakdown.csv)")
    parser.add_argument("run_b", help="Second run folder -- output CSV is written here")
    parser.add_argument("--input-csv", default="per_subject_error_breakdown.csv",
                         help="Filename to read from each run folder (default: per_subject_error_breakdown.csv)")
    args = parser.parse_args()

    for d in (args.run_a, args.run_b):
        if not os.path.isdir(d):
            sys.exit(f"Error: {d!r} is not a directory.")

    df_a, path_a = load_breakdown(args.run_a, args.input_csv)
    df_b, path_b = load_breakdown(args.run_b, args.input_csv)

    # --- hard-fail on any subject-set mismatch; never silently drop ---
    set_a, set_b = set(df_a["subject"]), set(df_b["subject"])
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    if only_a or only_b or len(set_a) != EXPECTED_SUBJECTS or len(set_b) != EXPECTED_SUBJECTS:
        lines = ["Error: subject ID sets are not identical "
                 f"{EXPECTED_SUBJECTS}-subject sets between the two runs."]
        lines.append(f"  run A ({path_a}): {len(set_a)} subjects: {sorted(set_a)}")
        lines.append(f"  run B ({path_b}): {len(set_b)} subjects: {sorted(set_b)}")
        if only_a:
            lines.append(f"  only in run A: {only_a}")
        if only_b:
            lines.append(f"  only in run B: {only_b}")
        sys.exit("\n".join(lines))
    print(f"Subject ID sets match: {EXPECTED_SUBJECTS} identical subjects in both runs.")

    # --- column availability check (don't fabricate) ---
    if "w2s_rate_pct" not in df_a.columns or "w2s_rate_pct" not in df_b.columns:
        sys.exit("Error: 'w2s_rate_pct' column missing from one or both CSVs -- "
                 "cannot compute the walking->standing rate correlation.")
    compute_recall = "walking_recall" in df_a.columns and "walking_recall" in df_b.columns
    if not compute_recall:
        print("\nWARNING: 'walking_recall' column missing from one or both CSVs -- "
              "skipping the walking-recall correlation (not fabricating it).")

    metrics_to_run = {"walking->standing rate": METRICS["walking->standing rate"]}
    if compute_recall:
        metrics_to_run["walking recall"] = METRICS["walking recall"]

    cols_a = ["subject"] + [col for col, _ in metrics_to_run.values()]
    cols_b = cols_a
    merged = df_a[cols_a].merge(df_b[cols_b], on="subject", how="inner", suffixes=("_a", "_b"))
    if len(merged) != EXPECTED_SUBJECTS:
        sys.exit(f"Error: inner merge produced {len(merged)} rows, expected {EXPECTED_SUBJECTS}. "
                 "This should be unreachable given the subject-set check above -- stopping.")

    spearman_results = {}
    for label, (col, higher_is_worse) in metrics_to_run.items():
        col_a, col_b = f"{col}_a", f"{col}_b"
        rank_a_col, rank_b_col = f"{col}_rank_a", f"{col}_rank_b"
        delta_col = f"{col}_rank_delta"
        merged[rank_a_col] = difficulty_rank(merged[col_a], higher_is_worse)
        merged[rank_b_col] = difficulty_rank(merged[col_b], higher_is_worse)
        merged[delta_col] = merged[rank_b_col] - merged[rank_a_col]

        rho, pval = spearmanr(merged[col_a], merged[col_b])
        spearman_results[label] = (rho, pval)

        print_ranking_table(merged, label, col_a, col_b, rank_a_col, rank_b_col, delta_col)

    print("\n" + "=" * 100)
    print("SPEARMAN RANK CORRELATION BETWEEN RUN A AND RUN B")
    print("=" * 100)
    for label, (rho, pval) in spearman_results.items():
        print(f"{label:<24s} rho={rho:.4f}  p-value={pval:.4f}  (n={len(merged)} subjects)")

    # --- write merged table to run B's folder ---
    out_path = os.path.join(args.run_b, "subject_ranking_persistence.csv")
    try:
        merged.to_csv(out_path, index=False)
        print(f"\nWrote subject_ranking_persistence.csv to {args.run_b}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write {out_path!r}: {e}")


if __name__ == "__main__":
    main()
