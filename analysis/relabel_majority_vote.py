#!/usr/bin/env python3
"""
Re-score existing LOSO predictions against majority-vote window labels
instead of last-sample window labels.

Reads predictions_best_*.csv files written by validation.py's
cross_participant_cv (--save_predictions), reconstructs the exact sliding
windows the pipeline built for each held-out subject directly from
hangtime_game_data.csv, verifies the reconstruction lines up with the
saved y_true column, computes a majority-vote label per window, and
re-scores the existing y_pred column against those majority-vote labels.

See analysis/_loso_common.py for the reconstruction/alignment logic and
the exact pipeline lines it is based on.

Usage (from repo root, hangtime_har env):
    python analysis/relabel_majority_vote.py <log_dir> \
        [--game-csv data/hangtime_game_data.csv] \
        [--sw-length 1.0] [--sw-overlap 50] [--sampling-rate 50]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _loso_common import (
    CLASS_NAMES, N_CLASSES,
    load_and_align_predictions, majority_vote_label, per_class_metrics,
    print_label_matrix,
)

# shot / pass / rebound / layup, per the pre-registered prediction (dribbling excluded)
BASKETBALL_IDX = sorted({1, 2, 3, 4})
# walking / running / standing / sitting -- standing is the suspected collapse class
LOCOMOTION_IDX = sorted({5, 6, 7, 8})


def main():
    parser = argparse.ArgumentParser(
        description="Re-score LOSO predictions against majority-vote window labels."
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

    resolutions, _, _, _, _ = load_and_align_predictions(
        args.log_dir, args.game_csv, args.sw_length, args.sw_overlap, args.sampling_rate,
    )

    # --- majority vote + pooling ---
    y_true_all, y_pred_all, mv_all = [], [], []
    for r in resolutions:
        mv_labels = np.array([majority_vote_label(w) for w in r["windows"]], dtype=int)
        y_true_all.append(r["y_true"])
        y_pred_all.append(r["y_pred"])
        mv_all.append(mv_labels)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    mv_all = np.concatenate(mv_all)

    # --- side-by-side per-class comparison ---
    old_prec, old_rec, old_f1, old_support = per_class_metrics(y_true_all, y_pred_all)
    new_prec, new_rec, new_f1, new_support = per_class_metrics(mv_all, y_pred_all)

    print("\n" + "=" * 100)
    print("PER-CLASS COMPARISON: last-sample label vs. majority-vote label (y_pred unchanged)")
    print("=" * 100)
    name_w2 = max(len(n) for n in CLASS_NAMES)
    header = (f"{'idx':>3}  {'class':<{name_w2}}  {'supp_old':>8}  {'supp_new':>8}  "
              f"{'F1_old':>7}  {'F1_new':>7}  {'dF1':>7}")
    print(header)
    print("-" * len(header))
    for i in range(N_CLASSES):
        d_f1 = new_f1[i] - old_f1[i] if not (np.isnan(new_f1[i]) or np.isnan(old_f1[i])) else np.nan
        print(f"{i:>3}  {CLASS_NAMES[i]:<{name_w2}}  {old_support[i]:>8}  {new_support[i]:>8}  "
              f"{old_f1[i]:>7.4f}  {new_f1[i]:>7.4f}  {d_f1:>+7.4f}")
    print("-" * len(header))
    print(f"Macro-F1 (nanmean): old={np.nanmean(old_f1):.4f}  new={np.nanmean(new_f1):.4f}")

    # --- full 9x9 transition matrix (old=last-sample, new=majority-vote) ---
    trans = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(trans, (y_true_all, mv_all), 1)

    n_total = len(y_true_all)
    n_changed = int((y_true_all != mv_all).sum())
    pct_changed = 100.0 * n_changed / n_total

    print("\n" + "=" * 100)
    print(f"LABEL CHANGE SUMMARY: {n_changed} / {n_total} windows changed label "
          f"({pct_changed:.2f}%)")
    print("=" * 100)

    print("\nFull 9x9 transition matrix (rows=old/last-sample, cols=new/majority-vote):")
    print_label_matrix(trans)

    print("\nTop transitions (old -> new), off-diagonal only, sorted by count:")
    transition_rows = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j or trans[i, j] == 0:
                continue
            transition_rows.append((CLASS_NAMES[i], CLASS_NAMES[j], int(trans[i, j])))
    transition_rows.sort(key=lambda t: t[2], reverse=True)
    print(f"{'old_label':<{name_w2}}  ->  {'new_label':<{name_w2}}  {'count':>7}  {'% of changed':>12}")
    for old, new, cnt in transition_rows:
        pct = 100.0 * cnt / n_changed if n_changed else 0.0
        print(f"{old:<{name_w2}}      {new:<{name_w2}}  {cnt:>7}  {pct:>11.2f}%")

    # --- basketball <-> locomotion, derived from the same matrix ---
    print("\n" + "=" * 100)
    print("BASKETBALL <-> LOCOMOTION (derived from transition matrix submatrices)")
    print(f"  basketball = {[CLASS_NAMES[i] for i in BASKETBALL_IDX]}")
    print(f"  locomotion = {[CLASS_NAMES[i] for i in LOCOMOTION_IDX]}")
    print("=" * 100)

    bball_to_loco = trans[np.ix_(BASKETBALL_IDX, LOCOMOTION_IDX)].sum()
    loco_to_bball = trans[np.ix_(LOCOMOTION_IDX, BASKETBALL_IDX)].sum()
    print(f"basketball -> locomotion: {int(bball_to_loco)} windows "
          f"({100.0 * bball_to_loco / n_total:.2f}% of all windows)")
    print(f"locomotion -> basketball: {int(loco_to_bball)} windows "
          f"({100.0 * loco_to_bball / n_total:.2f}% of all windows)")

    print("\nbasketball -> locomotion, by specific class pair:")
    pairs = []
    for i in BASKETBALL_IDX:
        for j in LOCOMOTION_IDX:
            if trans[i, j] > 0:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], int(trans[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    for old, new, cnt in pairs:
        print(f"  {old:<{name_w2}}  ->  {new:<{name_w2}}  {cnt:>7}")

    print("\nlocomotion -> basketball, by specific class pair:")
    pairs = []
    for i in LOCOMOTION_IDX:
        for j in BASKETBALL_IDX:
            if trans[i, j] > 0:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], int(trans[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    for old, new, cnt in pairs:
        print(f"  {old:<{name_w2}}  ->  {new:<{name_w2}}  {cnt:>7}")

    bball_support_old = int(old_support[BASKETBALL_IDX].sum())
    bball_support_new = int(new_support[BASKETBALL_IDX].sum())
    print(f"\nTotal basketball-class support: old={bball_support_old}  new={bball_support_new}  "
          f"(delta={bball_support_new - bball_support_old:+d}, "
          f"{100.0 * (bball_support_new - bball_support_old) / bball_support_old:+.2f}%)")

    # --- best-effort CSV outputs ---
    try:
        mapping_df = pd.DataFrame([
            {"file": os.path.basename(r["file"]), "filename_subject": r["filename_subject"],
             "resolved_subject": r["resolved_subject"],
             "mismatch": r["filename_subject"] != r["resolved_subject"]}
            for r in resolutions
        ])
        mapping_df.to_csv(os.path.join(args.log_dir, "majority_vote_subject_resolution.csv"), index=False)

        comparison_df = pd.DataFrame({
            "class": CLASS_NAMES,
            "support_old": old_support, "support_new": new_support,
            "precision_old": old_prec, "precision_new": new_prec,
            "recall_old": old_rec, "recall_new": new_rec,
            "f1_old": old_f1, "f1_new": new_f1,
        })
        comparison_df.to_csv(os.path.join(args.log_dir, "majority_vote_per_class_comparison.csv"), index=False)

        trans_df = pd.DataFrame(trans, index=CLASS_NAMES, columns=CLASS_NAMES)
        trans_df.to_csv(os.path.join(args.log_dir, "majority_vote_transition_matrix.csv"))

        print(f"\nWrote majority_vote_subject_resolution.csv, "
              f"majority_vote_per_class_comparison.csv, "
              f"majority_vote_transition_matrix.csv to {args.log_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write CSV outputs: {e}")


if __name__ == "__main__":
    main()
