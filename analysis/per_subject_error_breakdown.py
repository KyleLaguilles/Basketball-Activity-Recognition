#!/usr/bin/env python3
"""
Per-subject breakdown of the walking <-> standing confusion, to check whether
it is uniform across the 14 LOSO subjects or concentrated in a few.

Reads purity_per_window.csv (written by analysis/purity_stratified_scoring.py)
-- no window reconstruction needed, everything required is already in that
file: subject, window_index, y_true, y_pred, purity, majority_label.

Usage (from repo root, hangtime_har env):
    python analysis/per_subject_error_breakdown.py <log_dir> \
        [--input-csv purity_per_window.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _loso_common import CLASS_NAMES, N_CLASSES, is_pure

WALKING_IDX = CLASS_NAMES.index("walking")
STANDING_IDX = CLASS_NAMES.index("standing")
SITTING_IDX = CLASS_NAMES.index("sitting")

REQUIRED_COLS = ["subject", "window_index", "y_true", "y_pred", "purity", "majority_label"]

TOP_N_CONCENTRATION = 3
SINGLE_SUBJECT_FLAG_THRESHOLD = 30.0  # percent


def load_per_window_csv(path):
    if not os.path.isfile(path):
        sys.exit(f"Error: input CSV not found at {path!r}. Run "
                 "analysis/purity_stratified_scoring.py on this log_dir first.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Error: {path!r} is missing required column(s): {missing}")
    return df


def misclass_profile(sub_df, true_idx, other_named_idx):
    """
    Given windows whose y_true == true_idx, return counts + percentages of
    y_pred landing on each of other_named_idx (dict name -> class idx) plus
    an 'other' bucket for everything else.
    """
    n = len(sub_df)
    result = {"n": n}
    accounted = 0
    for name, idx in other_named_idx.items():
        cnt = int((sub_df["y_pred"] == idx).sum())
        accounted += cnt
        result[f"pred_{name}_n"] = cnt
        result[f"pred_{name}_pct"] = 100.0 * cnt / n if n else float("nan")
    other_cnt = n - accounted
    result["pred_other_n"] = other_cnt
    result["pred_other_pct"] = 100.0 * other_cnt / n if n else float("nan")
    return result


def build_subject_table(df):
    subjects = sorted(df["subject"].unique().tolist())
    rows = []
    for subj in subjects:
        sub_df = df[df["subject"] == subj]

        true_walking = sub_df[sub_df["y_true"] == WALKING_IDX]
        walk_profile = misclass_profile(
            true_walking, WALKING_IDX,
            {"walking": WALKING_IDX, "standing": STANDING_IDX, "sitting": SITTING_IDX},
        )

        true_standing = sub_df[sub_df["y_true"] == STANDING_IDX]
        stand_profile = misclass_profile(
            true_standing, STANDING_IDX,
            {"standing": STANDING_IDX, "walking": WALKING_IDX, "sitting": SITTING_IDX},
        )

        pred_standing_total = int((sub_df["y_pred"] == STANDING_IDX).sum())
        standing_tp = stand_profile["pred_standing_n"]
        standing_precision = standing_tp / pred_standing_total if pred_standing_total else float("nan")
        walking_recall = walk_profile["pred_walking_n"] / walk_profile["n"] if walk_profile["n"] else float("nan")

        support = np.bincount(sub_df["y_true"].to_numpy(), minlength=N_CLASSES)

        row = {
            "subject": subj,
            "n_true_walking": walk_profile["n"],
            "walk_pred_walking_n": walk_profile["pred_walking_n"],
            "walk_pred_walking_pct": walk_profile["pred_walking_pct"],
            "walk_pred_standing_n": walk_profile["pred_standing_n"],
            "walk_pred_standing_pct": walk_profile["pred_standing_pct"],
            "walk_pred_sitting_n": walk_profile["pred_sitting_n"],
            "walk_pred_sitting_pct": walk_profile["pred_sitting_pct"],
            "walk_pred_other_n": walk_profile["pred_other_n"],
            "walk_pred_other_pct": walk_profile["pred_other_pct"],
            "n_true_standing": stand_profile["n"],
            "stand_pred_standing_n": stand_profile["pred_standing_n"],
            "stand_pred_standing_pct": stand_profile["pred_standing_pct"],
            "stand_pred_walking_n": stand_profile["pred_walking_n"],
            "stand_pred_walking_pct": stand_profile["pred_walking_pct"],
            "stand_pred_sitting_n": stand_profile["pred_sitting_n"],
            "stand_pred_sitting_pct": stand_profile["pred_sitting_pct"],
            "stand_pred_other_n": stand_profile["pred_other_n"],
            "stand_pred_other_pct": stand_profile["pred_other_pct"],
            "walking_recall": walking_recall,
            "standing_precision": standing_precision,
            # walking -> standing rate/count, pulled out for sorting/summary
            "w2s_rate_pct": walk_profile["pred_standing_pct"],
            "w2s_count": walk_profile["pred_standing_n"],
        }
        for i in range(N_CLASSES):
            row[f"support_{CLASS_NAMES[i]}"] = int(support[i])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("w2s_rate_pct", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    return table


def build_purity_stratified_table(df, subject_order):
    rows = []
    for subj in subject_order:
        sub_df = df[df["subject"] == subj]
        true_walking = sub_df[sub_df["y_true"] == WALKING_IDX]
        pure_walking = true_walking[true_walking["purity"].apply(is_pure)]
        mixed_walking = true_walking[~true_walking["purity"].apply(is_pure)]

        def rate(d):
            n = len(d)
            if n == 0:
                return n, float("nan")
            cnt = int((d["y_pred"] == STANDING_IDX).sum())
            return n, 100.0 * cnt / n

        n_pure, rate_pure = rate(pure_walking)
        n_mixed, rate_mixed = rate(mixed_walking)
        rows.append({
            "subject": subj,
            "n_walking_pure": n_pure, "w2s_rate_pure_pct": rate_pure,
            "n_walking_mixed": n_mixed, "w2s_rate_mixed_pct": rate_mixed,
        })
    return pd.DataFrame(rows)


def print_table(df, cols, title, float_cols=()):
    print(f"\n{title}")
    print("-" * len(title))
    widths = {}
    for c in cols:
        header_label = c
        widths[c] = max(len(header_label), 8)
    header = "  ".join(f"{c:>{widths[c]}}" if c != "subject" else f"{c:<{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        parts = []
        for c in cols:
            v = r[c]
            if c == "subject":
                parts.append(f"{str(v):<{widths[c]}}")
            elif c in float_cols:
                parts.append(f"{v:>{widths[c]}.2f}" if pd.notna(v) else f"{'nan':>{widths[c]}}")
            else:
                parts.append(f"{v:>{widths[c]}}")
        print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Per-subject breakdown of walking<->standing confusion from purity_per_window.csv."
    )
    parser.add_argument("log_dir", help="Directory containing purity_per_window.csv")
    parser.add_argument("--input-csv", default="purity_per_window.csv",
                         help="Filename to read from log_dir (default: purity_per_window.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        sys.exit(f"Error: {args.log_dir!r} is not a directory.")

    input_path = os.path.join(args.log_dir, args.input_csv)
    df = load_per_window_csv(input_path)
    if len(df) == 0:
        sys.exit(f"Error: {input_path!r} has no rows. Re-run "
                 "analysis/purity_stratified_scoring.py on this log_dir first.")
    n_subjects = df["subject"].nunique()
    print(f"Loaded {len(df)} windows across {n_subjects} subject(s) from: {input_path}")
    if n_subjects != 14:
        print(f"WARNING: expected 14 subjects, got {n_subjects}.")

    table = build_subject_table(df)

    # --- (a) main per-subject table, sorted by w2s_rate_pct descending ---
    print("\n" + "=" * 100)
    print("PER-SUBJECT WALKING MISCLASSIFICATION PROFILE (true walking -> predicted X)")
    print("sorted by walking->standing rate, descending")
    print("=" * 100)
    print_table(
        table,
        ["subject", "n_true_walking", "walk_pred_walking_pct", "walk_pred_standing_pct",
         "walk_pred_sitting_pct", "walk_pred_other_pct"],
        "walking misclassification profile (%)",
        float_cols={"walk_pred_walking_pct", "walk_pred_standing_pct",
                    "walk_pred_sitting_pct", "walk_pred_other_pct"},
    )

    print("\n" + "=" * 100)
    print("PER-SUBJECT STANDING MISCLASSIFICATION PROFILE (true standing -> predicted X)")
    print("=" * 100)
    print_table(
        table,
        ["subject", "n_true_standing", "stand_pred_standing_pct", "stand_pred_walking_pct",
         "stand_pred_sitting_pct", "stand_pred_other_pct"],
        "standing misclassification profile (%)",
        float_cols={"stand_pred_standing_pct", "stand_pred_walking_pct",
                    "stand_pred_sitting_pct", "stand_pred_other_pct"},
    )

    print("\n" + "=" * 100)
    print("PER-SUBJECT WALKING RECALL / STANDING PRECISION")
    print("=" * 100)
    print_table(
        table,
        ["subject", "walking_recall", "standing_precision"],
        "recall / precision",
        float_cols={"walking_recall", "standing_precision"},
    )

    print("\n" + "=" * 100)
    print("PER-SUBJECT CLASS SUPPORT (true-label counts)")
    print("=" * 100)
    support_cols = ["subject"] + [f"support_{c}" for c in CLASS_NAMES]
    print_table(table, support_cols, "class support")

    # --- (b) summary stats across subjects for the walking->standing rate ---
    rates = table["w2s_rate_pct"].dropna().to_numpy()
    counts = table[["subject", "w2s_count"]].copy()
    total_w2s_errors = int(counts["w2s_count"].sum())
    counts_sorted = counts.sort_values("w2s_count", ascending=False).reset_index(drop=True)
    top_n = counts_sorted.head(TOP_N_CONCENTRATION)
    top_n_share = 100.0 * top_n["w2s_count"].sum() / total_w2s_errors if total_w2s_errors else float("nan")
    top1_subject = counts_sorted.iloc[0]["subject"] if len(counts_sorted) else None
    top1_count = int(counts_sorted.iloc[0]["w2s_count"]) if len(counts_sorted) else 0
    top1_share = 100.0 * top1_count / total_w2s_errors if total_w2s_errors else float("nan")

    print("\n" + "=" * 100)
    print("SUMMARY: walking->standing rate across subjects")
    print("=" * 100)
    print(f"mean={np.mean(rates):.2f}%  std={np.std(rates):.2f}%  "
          f"min={np.min(rates):.2f}%  max={np.max(rates):.2f}%  (n_subjects={len(rates)})")
    print(f"\nTotal walking->standing errors (raw count, all subjects): {total_w2s_errors}")
    print(f"Top-{TOP_N_CONCENTRATION} subjects by raw walking->standing error count: "
          + ", ".join(f"{r.subject} (n={int(r.w2s_count)})" for r in top_n.itertuples()))
    print(f"Top-{TOP_N_CONCENTRATION} share of total walking->standing errors: {top_n_share:.2f}%")
    print(f"Top-1 subject ({top1_subject}) share of total walking->standing errors: {top1_share:.2f}%")

    # --- (c) flag line ---
    flag = top1_share is not None and not np.isnan(top1_share) and top1_share > SINGLE_SUBJECT_FLAG_THRESHOLD
    print(f"\nFLAG: single subject contributes >{SINGLE_SUBJECT_FLAG_THRESHOLD:.0f}% of total "
          f"walking->standing errors? {'YES' if flag else 'NO'}"
          + (f" ({top1_subject}: {top1_share:.2f}%)" if flag else ""))

    # --- purity-stratified walking->standing rate per subject ---
    purity_table = build_purity_stratified_table(df, table["subject"].tolist())
    print("\n" + "=" * 100)
    print("PURITY-STRATIFIED walking->standing rate per subject (pure-only vs mixed-only)")
    print("=" * 100)
    print_table(
        purity_table,
        ["subject", "n_walking_pure", "w2s_rate_pure_pct", "n_walking_mixed", "w2s_rate_mixed_pct"],
        "pure vs mixed walking->standing rate (%)",
        float_cols={"w2s_rate_pure_pct", "w2s_rate_mixed_pct"},
    )

    # --- write CSVs ---
    try:
        table.to_csv(os.path.join(args.log_dir, "per_subject_error_breakdown.csv"), index=False)
        purity_table.to_csv(
            os.path.join(args.log_dir, "per_subject_error_breakdown_purity_stratified.csv"), index=False)

        summary_df = pd.DataFrame([{
            "mean_w2s_rate_pct": np.mean(rates), "std_w2s_rate_pct": np.std(rates),
            "min_w2s_rate_pct": np.min(rates), "max_w2s_rate_pct": np.max(rates),
            "n_subjects": len(rates), "total_w2s_errors": total_w2s_errors,
            "top1_subject": top1_subject, "top1_share_pct": top1_share,
            f"top{TOP_N_CONCENTRATION}_share_pct": top_n_share,
            "flag_single_subject_over_threshold": flag,
            "flag_threshold_pct": SINGLE_SUBJECT_FLAG_THRESHOLD,
        }])
        summary_df.to_csv(os.path.join(args.log_dir, "per_subject_error_breakdown_summary.csv"), index=False)

        print(f"\nWrote per_subject_error_breakdown.csv, "
              f"per_subject_error_breakdown_purity_stratified.csv, "
              f"per_subject_error_breakdown_summary.csv to {args.log_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write CSV outputs: {e}")


if __name__ == "__main__":
    main()
