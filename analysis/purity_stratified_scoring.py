#!/usr/bin/env python3
"""
Test whether model errors concentrate in impure (mixed-label) windows,
regardless of which single-label rule (last-sample vs majority-vote) is used.

Reads predictions_best_*.csv files, reconstructs the exact sliding windows
the pipeline built for each held-out subject directly from
hangtime_game_data.csv (same reconstruction + alignment as
analysis/relabel_majority_vote.py -- see analysis/_loso_common.py), computes
a purity score per window (fraction of the window's 50 samples belonging to
the window's majority class), and stratifies existing y_pred vs y_true
scoring by that purity.

Usage (from repo root, hangtime_har env):
    python analysis/purity_stratified_scoring.py <log_dir> \
        [--game-csv data/hangtime_game_data.csv] \
        [--sw-length 1.0] [--sw-overlap 50] [--sampling-rate 50]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from _loso_common import (
    CLASS_NAMES, N_CLASSES,
    load_and_align_predictions, window_majority, print_label_matrix,
)

# purity bins, checked high-to-low; boundaries match the task's spec exactly
PURITY_BIN_ORDER = ["1.0", "[0.9,1.0)", "[0.7,0.9)", "[0.5,0.7)", "<0.5"]
_EPS = 1e-9


def purity_bin(p):
    if p >= 1.0 - _EPS:
        return "1.0"
    if p >= 0.9 - _EPS:
        return "[0.9,1.0)"
    if p >= 0.7 - _EPS:
        return "[0.7,0.9)"
    if p >= 0.5 - _EPS:
        return "[0.5,0.7)"
    return "<0.5"


# the "five basketball classes" for the dose-response groupings, per this
# script's task spec -- note this INCLUDES dribbling, unlike
# relabel_majority_vote.py's 4-class BASKETBALL_IDX (which excludes it for
# its own pre-registered basketball<->locomotion hypothesis). Kept local and
# separately named on purpose to avoid conflating the two groupings.
BASKETBALL5_IDX = [0, 1, 2, 3, 4]  # dribbling, shot, pass, rebound, layup
DOSE_RESPONSE_GROUPS = {
    "overall": list(range(N_CLASSES)),
    "walking": [5],
    "running": [6],
    "standing": [7],
    "basketball(5)": BASKETBALL5_IDX,
}

STANDING_IDX = 7


def per_class_metrics_subset(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=np.nan,
    )
    support = np.bincount(y_true, minlength=N_CLASSES)
    return prec, rec, f1, support


def main():
    parser = argparse.ArgumentParser(
        description="Stratify LOSO prediction scoring by per-window label purity."
    )
    parser.add_argument("log_dir", help="Directory containing predictions_best_*.csv files")
    parser.add_argument("--game-csv", default=os.path.join("data", "hangtime_game_data.csv"),
                         help="Path to hangtime_game_data.csv (default: data/hangtime_game_data.csv)")
    parser.add_argument("--sw-length", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--sw-overlap", type=float, default=50, help="Overlap percentage")
    parser.add_argument("--sampling-rate", type=int, default=50, help="Sampling rate in Hz")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        sys.exit(f"Error: {args.log_dir!r} is not a directory.")

    resolutions, _, _, win_len, _ = load_and_align_predictions(
        args.log_dir, args.game_csv, args.sw_length, args.sw_overlap, args.sampling_rate,
    )

    # --- build the per-window table ---
    rows = []
    for r in resolutions:
        subject = r["resolved_subject"]
        for w_idx, (window, y_t, y_p) in enumerate(zip(r["windows"], r["y_true"], r["y_pred"])):
            maj_label, maj_count = window_majority(window)
            purity = maj_count / win_len
            rows.append({
                "subject": subject, "window_index": w_idx,
                "y_true": int(y_t), "y_pred": int(y_p),
                "purity": purity, "purity_bin": purity_bin(purity),
                "majority_label": maj_label,
            })
    win_df = pd.DataFrame(rows)
    n_total = len(win_df)
    print(f"\nReconstructed {n_total} windows across {len(resolutions)} subjects.")

    # sanity: majority_label should equal y_true for pure windows (purity==1.0)
    pure_check = win_df[win_df["purity_bin"] == "1.0"]
    assert (pure_check["majority_label"] == pure_check["y_true"]).all(), \
        "Internal error: a purity==1.0 window's majority label disagrees with y_true."

    # --- purity distribution: overall + per class (grouped by y_true) ---
    print("\n" + "=" * 100)
    print("PURITY DISTRIBUTION")
    print("=" * 100)

    def dist_row(sub_df):
        n = len(sub_df)
        mean_p = sub_df["purity"].mean() if n else float("nan")
        bin_counts = sub_df["purity_bin"].value_counts()
        return n, mean_p, [int(bin_counts.get(b, 0)) for b in PURITY_BIN_ORDER]

    header = (f"{'group':<12}  {'n':>6}  {'mean':>6}  " +
              "  ".join(f"{b:>10}" for b in PURITY_BIN_ORDER))
    print(header)
    print("-" * len(header))
    n, mean_p, bins = dist_row(win_df)
    print(f"{'overall':<12}  {n:>6}  {mean_p:>6.3f}  " + "  ".join(f"{c:>10}" for c in bins))
    dist_rows_for_csv = [{"group": "overall", "n": n, "mean_purity": mean_p,
                           **dict(zip(PURITY_BIN_ORDER, bins))}]
    for cls_idx in range(N_CLASSES):
        sub = win_df[win_df["y_true"] == cls_idx]
        n, mean_p, bins = dist_row(sub)
        print(f"{CLASS_NAMES[cls_idx]:<12}  {n:>6}  {mean_p:>6.3f}  " + "  ".join(f"{c:>10}" for c in bins))
        dist_rows_for_csv.append({"group": CLASS_NAMES[cls_idx], "n": n, "mean_purity": mean_p,
                                   **dict(zip(PURITY_BIN_ORDER, bins))})

    # --- (a) binary split: pure vs mixed ---
    print("\n" + "=" * 100)
    print("PURE (purity==1.0) vs MIXED (purity<1.0): accuracy + per-class F1/support")
    print("=" * 100)
    pure_mask = win_df["purity_bin"] == "1.0"
    mixed_mask = ~pure_mask

    def acc(sub_df):
        return (sub_df["y_true"] == sub_df["y_pred"]).mean() if len(sub_df) else float("nan")

    pure_df, mixed_df = win_df[pure_mask], win_df[mixed_mask]
    print(f"n_pure={len(pure_df)}  accuracy_pure={acc(pure_df):.4f}   |   "
          f"n_mixed={len(mixed_df)}  accuracy_mixed={acc(mixed_df):.4f}")

    pure_prec, pure_rec, pure_f1, pure_support = per_class_metrics_subset(
        pure_df["y_true"].to_numpy(), pure_df["y_pred"].to_numpy())
    mixed_prec, mixed_rec, mixed_f1, mixed_support = per_class_metrics_subset(
        mixed_df["y_true"].to_numpy(), mixed_df["y_pred"].to_numpy())

    name_w = max(len(n) for n in CLASS_NAMES)
    header2 = (f"{'idx':>3}  {'class':<{name_w}}  {'supp_pure':>9}  {'F1_pure':>8}  "
               f"{'supp_mixed':>10}  {'F1_mixed':>9}")
    print(header2)
    print("-" * len(header2))
    for i in range(N_CLASSES):
        print(f"{i:>3}  {CLASS_NAMES[i]:<{name_w}}  {pure_support[i]:>9}  {pure_f1[i]:>8.4f}  "
              f"{mixed_support[i]:>10}  {mixed_f1[i]:>9.4f}")
    print("-" * len(header2))
    print(f"Macro-F1 (nanmean): pure={np.nanmean(pure_f1):.4f}  mixed={np.nanmean(mixed_f1):.4f}")

    # --- (4a) HEADLINE: per-class F1 on PURE windows only ---
    print("\n" + "=" * 100)
    print("HEADLINE: per-class F1 on PURE windows only (game data)")
    print("=" * 100)
    header3 = f"{'idx':>3}  {'class':<{name_w}}  {'support':>7}  {'precision':>9}  {'recall':>6}  {'F1':>6}"
    print(header3)
    print("-" * len(header3))
    for i in range(N_CLASSES):
        print(f"{i:>3}  {CLASS_NAMES[i]:<{name_w}}  {pure_support[i]:>7}  "
              f"{pure_prec[i]:>9.4f}  {pure_rec[i]:>6.4f}  {pure_f1[i]:>6.4f}")
    print(f"\nMacro-F1 on pure windows (nanmean): {np.nanmean(pure_f1):.4f}")

    # --- (4b) confusion matrix restricted to pure windows ---
    print("\n" + "=" * 100)
    print("CONFUSION MATRIX -- PURE windows only (rows=y_true, cols=y_pred)")
    print("=" * 100)
    pure_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(pure_cm, (pure_df["y_true"].to_numpy(), pure_df["y_pred"].to_numpy()), 1)
    print_label_matrix(pure_cm)

    # --- (b) dose-response: accuracy as a function of purity bin ---
    print("\n" + "=" * 100)
    print("DOSE-RESPONSE: accuracy(%) by purity bin, per group  [format: acc% (n)]")
    print("=" * 100)
    group_names = list(DOSE_RESPONSE_GROUPS.keys())
    gcol_w = 18
    header4 = f"{'purity_bin':<12}  " + "".join(f"{g:>{gcol_w}}" for g in group_names)
    print(header4)
    print("-" * len(header4))
    dose_rows_for_csv = []
    for b in PURITY_BIN_ORDER:
        bin_df = win_df[win_df["purity_bin"] == b]
        cells = []
        csv_row = {"purity_bin": b}
        for g, idxs in DOSE_RESPONSE_GROUPS.items():
            sub = bin_df[bin_df["y_true"].isin(idxs)]
            n = len(sub)
            a = acc(sub) * 100 if n else float("nan")
            cells.append(f"{a:>6.2f}% ({n})" if n else f"{'--':>6}  (0)")
            csv_row[f"{g}_acc"] = a
            csv_row[f"{g}_n"] = n
        print(f"{b:<12}  " + "".join(f"{c:>{gcol_w}}" for c in cells))
        dose_rows_for_csv.append(csv_row)

    # --- (4c) standing: predicted-standing windows, true-label breakdown, pure vs mixed ---
    print("\n" + "=" * 100)
    print("STANDING DIAGNOSTIC: windows PREDICTED as 'standing' -- true-label breakdown, "
          "pure vs mixed")
    print("=" * 100)
    pred_standing = win_df[win_df["y_pred"] == STANDING_IDX]
    n_pred_standing = len(pred_standing)
    print(f"Total windows predicted as 'standing': {n_pred_standing}")
    header5 = (f"{'true_class':<{name_w}}  {'pure_n':>7}  {'pure_%':>7}  "
               f"{'mixed_n':>8}  {'mixed_%':>8}  {'total_n':>8}  {'total_%':>8}")
    print(header5)
    print("-" * len(header5))
    standing_rows_for_csv = []
    for i in range(N_CLASSES):
        cls_sub = pred_standing[pred_standing["y_true"] == i]
        n_pure_i = int((cls_sub["purity_bin"] == "1.0").sum())
        n_mixed_i = len(cls_sub) - n_pure_i
        n_total_i = len(cls_sub)
        pct_pure = 100.0 * n_pure_i / n_pred_standing if n_pred_standing else 0.0
        pct_mixed = 100.0 * n_mixed_i / n_pred_standing if n_pred_standing else 0.0
        pct_total = 100.0 * n_total_i / n_pred_standing if n_pred_standing else 0.0
        print(f"{CLASS_NAMES[i]:<{name_w}}  {n_pure_i:>7}  {pct_pure:>6.2f}%  "
              f"{n_mixed_i:>8}  {pct_mixed:>7.2f}%  {n_total_i:>8}  {pct_total:>7.2f}%")
        standing_rows_for_csv.append({
            "true_class": CLASS_NAMES[i], "pure_n": n_pure_i, "pure_pct": pct_pure,
            "mixed_n": n_mixed_i, "mixed_pct": pct_mixed,
            "total_n": n_total_i, "total_pct": pct_total,
        })
    correct_standing = n_pred_standing and pred_standing["y_true"].eq(STANDING_IDX).sum() or 0
    precision_standing = correct_standing / n_pred_standing if n_pred_standing else float("nan")
    print(f"\n(precision on 'standing' predictions = {precision_standing:.4f}; "
          "cross-check against your reported 0.085 if this is the same run)")

    # --- write per-window CSV for downstream joins (e.g. smoothing analysis) ---
    try:
        win_out_cols = ["subject", "window_index", "y_true", "y_pred", "purity", "majority_label"]
        win_df[win_out_cols].to_csv(
            os.path.join(args.log_dir, "purity_per_window.csv"), index=False)

        pd.DataFrame(dist_rows_for_csv).to_csv(
            os.path.join(args.log_dir, "purity_distribution.csv"), index=False)

        pure_mixed_df = pd.DataFrame({
            "class": CLASS_NAMES,
            "support_pure": pure_support, "precision_pure": pure_prec,
            "recall_pure": pure_rec, "f1_pure": pure_f1,
            "support_mixed": mixed_support, "precision_mixed": mixed_prec,
            "recall_mixed": mixed_rec, "f1_mixed": mixed_f1,
        })
        pure_mixed_df.to_csv(
            os.path.join(args.log_dir, "purity_pure_vs_mixed_per_class.csv"), index=False)

        pd.DataFrame(dose_rows_for_csv).to_csv(
            os.path.join(args.log_dir, "purity_dose_response.csv"), index=False)

        pd.DataFrame(pure_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(args.log_dir, "purity_confusion_matrix_pure.csv"))

        pd.DataFrame(standing_rows_for_csv).to_csv(
            os.path.join(args.log_dir, "purity_standing_breakdown.csv"), index=False)

        print(f"\nWrote purity_per_window.csv, purity_distribution.csv, "
              f"purity_pure_vs_mixed_per_class.csv, purity_dose_response.csv, "
              f"purity_confusion_matrix_pure.csv, purity_standing_breakdown.csv "
              f"to {args.log_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write CSV outputs: {e}")


if __name__ == "__main__":
    main()
