#!/usr/bin/env python3
"""
Test whether the walking->standing confusion tracks low accelerometer energy,
i.e. whether "standing" is stealing low-intensity walking (slow shuffling,
near-stationary) rather than being a purely random/boundary artifact.

Reuses the reconstruction + alignment machinery from analysis/_loso_common.py
(same as relabel_majority_vote.py / purity_stratified_scoring.py) to get
correctly-resolved per-subject windows, then additionally loads the raw
accelerometer (acc_x, acc_y, acc_z) samples for those same windows to compute
per-window signal magnitude statistics.

Signal magnitude per sample = L2 norm sqrt(acc_x^2 + acc_y^2 + acc_z^2).
Per window: mean_magnitude = mean over the window's 50 samples (this is what
we report as "signal energy"), std_magnitude = std over the same samples.

Depends on per_subject_error_breakdown.csv (written by
analysis/per_subject_error_breakdown.py) in the same log_dir for each
subject's walking->standing rate -- run that script on this log_dir first.

Usage (from repo root, hangtime_har env):
    python analysis/walking_signal_energy.py <log_dir> \
        [--game-csv data/hangtime_game_data.csv] \
        [--error-breakdown-csv per_subject_error_breakdown.csv] \
        [--sw-length 1.0] [--sw-overlap 50] [--sampling-rate 50]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from _loso_common import (
    CLASS_NAMES, load_and_align_predictions, load_game_features,
    make_windows, is_pure, window_majority,
)

WALKING_IDX = CLASS_NAMES.index("walking")
STANDING_IDX = CLASS_NAMES.index("standing")


def window_magnitude_stats(accel_window):
    """accel_window: (win_len, 3) array of acc_x/y/z samples."""
    magnitude = np.linalg.norm(accel_window, axis=1)
    return float(magnitude.mean()), float(magnitude.std())


def main():
    parser = argparse.ArgumentParser(
        description="Correlate walking-window accelerometer energy with the walking->standing error rate."
    )
    parser.add_argument("log_dir", help="Directory containing predictions_best_*.csv files")
    parser.add_argument("--game-csv", default=os.path.join("data", "hangtime_game_data.csv"),
                         help="Path to hangtime_game_data.csv (default: data/hangtime_game_data.csv)")
    parser.add_argument("--error-breakdown-csv", default="per_subject_error_breakdown.csv",
                         help="Filename to read from log_dir for w2s_rate_pct per subject "
                              "(default: per_subject_error_breakdown.csv)")
    parser.add_argument("--sw-length", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--sw-overlap", type=float, default=50, help="Overlap percentage")
    parser.add_argument("--sampling-rate", type=int, default=50, help="Sampling rate in Hz")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        sys.exit(f"Error: {args.log_dir!r} is not a directory.")

    breakdown_path = os.path.join(args.log_dir, args.error_breakdown_csv)
    if not os.path.isfile(breakdown_path):
        sys.exit(f"Error: {breakdown_path!r} not found. Run "
                 "analysis/per_subject_error_breakdown.py on this log_dir first.")
    breakdown_df = pd.read_csv(breakdown_path)
    if "subject" not in breakdown_df.columns or "w2s_rate_pct" not in breakdown_df.columns:
        sys.exit(f"Error: {breakdown_path!r} is missing 'subject' and/or 'w2s_rate_pct' columns.")
    w2s_rate_by_subject = dict(zip(breakdown_df["subject"], breakdown_df["w2s_rate_pct"]))

    # --- reconstruction + alignment (labels only) ---
    resolutions, _, _, win_len, step = load_and_align_predictions(
        args.log_dir, args.game_csv, args.sw_length, args.sw_overlap, args.sampling_rate,
    )

    # --- load raw accelerometer samples, same void-filter/row-order as labels ---
    print(f"\nLoading accelerometer features from: {args.game_csv}")
    feat_df = load_game_features(args.game_csv)

    # --- build per-window table: subject, y_true, y_pred, purity, mean/std magnitude ---
    rows = []
    for r in resolutions:
        subj = r["resolved_subject"]
        accel_arr = feat_df.loc[feat_df["subject"] == subj, ["acc_x", "acc_y", "acc_z"]].to_numpy()
        accel_windows = make_windows(accel_arr, win_len, step)

        if len(accel_windows) != len(r["windows"]):
            sys.exit(
                f"Error: accelerometer window count ({len(accel_windows)}) for subject "
                f"{subj!r} does not match label window count ({len(r['windows'])}). "
                "load_game_labels and load_game_features disagree on this subject's row "
                "count/order -- stopping rather than silently misaligning sensor data with labels."
            )

        for w_idx, (label_window, accel_window, y_t, y_p) in enumerate(
            zip(r["windows"], accel_windows, r["y_true"], r["y_pred"])
        ):
            mean_mag, std_mag = window_magnitude_stats(accel_window)
            _, maj_count = window_majority(label_window)
            purity = maj_count / win_len
            rows.append({
                "subject": subj, "window_index": w_idx,
                "y_true": int(y_t), "y_pred": int(y_p),
                "purity": purity, "mean_magnitude": mean_mag, "std_magnitude": std_mag,
            })
    win_df = pd.DataFrame(rows)
    print(f"Reconstructed {len(win_df)} windows with accelerometer stats across {len(resolutions)} subjects.")

    win_df["is_pure"] = win_df["purity"].apply(is_pure)
    true_walking = win_df[win_df["y_true"] == WALKING_IDX]
    pure_walking = true_walking[true_walking["is_pure"]]

    # --- (1) per-subject mean signal energy for TRUE-WALKING PURE windows ---
    print("\n" + "=" * 100)
    print("PER-SUBJECT MEAN SIGNAL ENERGY -- true-walking PURE windows only")
    print("=" * 100)
    energy_rows = []
    subjects_sorted = sorted(win_df["subject"].unique().tolist())
    for subj in subjects_sorted:
        sub_pure_walk = pure_walking[pure_walking["subject"] == subj]
        n = len(sub_pure_walk)
        mean_energy = sub_pure_walk["mean_magnitude"].mean() if n else float("nan")
        std_energy = sub_pure_walk["mean_magnitude"].std() if n else float("nan")
        w2s_rate = w2s_rate_by_subject.get(subj, float("nan"))
        energy_rows.append({
            "subject": subj, "n_pure_walking_windows": n,
            "mean_signal_energy": mean_energy, "std_signal_energy_across_windows": std_energy,
            "w2s_rate_pct": w2s_rate,
        })
    energy_df = pd.DataFrame(energy_rows)

    name_w = max(len(str(s)) for s in subjects_sorted)
    header = f"{'subject':<{name_w}}  {'n_pure_walk':>11}  {'mean_energy':>12}  {'w2s_rate_pct':>13}"
    print(header)
    print("-" * len(header))
    for _, row in energy_df.iterrows():
        me = f"{row['mean_signal_energy']:.4f}" if pd.notna(row["mean_signal_energy"]) else "nan"
        wr = f"{row['w2s_rate_pct']:.2f}" if pd.notna(row["w2s_rate_pct"]) else "nan"
        print(f"{str(row['subject']):<{name_w}}  {row['n_pure_walking_windows']:>11}  {me:>12}  {wr:>13}")

    # --- (2) Spearman correlation across the 24 subjects ---
    valid = energy_df.dropna(subset=["mean_signal_energy", "w2s_rate_pct"])
    print("\n" + "=" * 100)
    print("SPEARMAN CORRELATION: per-subject mean signal energy (pure walking) vs. walking->standing rate")
    print("=" * 100)
    if len(valid) < 3:
        print(f"Not enough subjects with both values (n={len(valid)}) to compute a meaningful correlation.")
        rho, pval, n_corr = float("nan"), float("nan"), len(valid)
    else:
        rho, pval = spearmanr(valid["mean_signal_energy"], valid["w2s_rate_pct"])
        n_corr = len(valid)
        print(f"Spearman rho = {rho:.4f}  p-value = {pval:.4f}  (n={n_corr} subjects)")
        if rho < 0:
            print("Negative correlation: lower walking-window signal energy associates with "
                  "a higher walking->standing error rate (consistent with low-intensity "
                  "walking being mistaken for standing).")
        else:
            print("Non-negative correlation: no support for low signal energy driving the "
                  "walking->standing error rate.")
    if len(energy_df) != 24:
        print(f"\nWARNING: expected 24 subjects, got {len(energy_df)}.")

    # --- (3) predicted-as-standing vs predicted-as-walking energy distributions, per subject ---
    def pred_comparison_table(pool_df, label):
        print("\n" + "=" * 100)
        print(f"PREDICTED-AS-STANDING vs PREDICTED-AS-WALKING energy, per subject -- {label}")
        print("=" * 100)
        header2 = (f"{'subject':<{name_w}}  {'n_pred_walk':>11}  {'energy_walk':>11}  "
                   f"{'n_pred_stand':>12}  {'energy_stand':>12}  {'delta':>8}  {'mannwhitney_p':>13}")
        print(header2)
        print("-" * len(header2))
        rows_out = []
        for subj in subjects_sorted:
            sub = pool_df[pool_df["subject"] == subj]
            walk_energy = sub.loc[sub["y_pred"] == WALKING_IDX, "mean_magnitude"]
            stand_energy = sub.loc[sub["y_pred"] == STANDING_IDX, "mean_magnitude"]
            n_w, n_s = len(walk_energy), len(stand_energy)
            m_w = walk_energy.mean() if n_w else float("nan")
            m_s = stand_energy.mean() if n_s else float("nan")
            delta = m_s - m_w if (n_w and n_s) else float("nan")
            if n_w >= 2 and n_s >= 2:
                _, p = mannwhitneyu(walk_energy, stand_energy, alternative="two-sided")
            else:
                p = float("nan")
            print(f"{str(subj):<{name_w}}  {n_w:>11}  {m_w:>11.4f}  {n_s:>12}  {m_s:>12.4f}  "
                  f"{delta:>+8.4f}  {p if pd.isna(p) else round(p, 4):>13}")
            rows_out.append({
                "subject": subj, "n_pred_walking": n_w, "energy_pred_walking": m_w,
                "n_pred_standing": n_s, "energy_pred_standing": m_s,
                "delta_stand_minus_walk": delta, "mannwhitney_p": p,
            })
        return pd.DataFrame(rows_out)

    pred_cmp_pure = pred_comparison_table(pure_walking, "true-walking PURE windows only")
    pred_cmp_all = pred_comparison_table(true_walking, "all true-walking windows (pure + mixed)")

    # --- write CSVs ---
    try:
        win_df.to_csv(os.path.join(args.log_dir, "walking_signal_energy_per_window.csv"), index=False)
        energy_df.to_csv(os.path.join(args.log_dir, "walking_signal_energy_per_subject.csv"), index=False)
        pd.DataFrame([{"spearman_rho": rho, "p_value": pval, "n_subjects": n_corr}]).to_csv(
            os.path.join(args.log_dir, "walking_signal_energy_correlation.csv"), index=False)
        pred_cmp_pure.to_csv(
            os.path.join(args.log_dir, "walking_signal_energy_pred_comparison_pure.csv"), index=False)
        pred_cmp_all.to_csv(
            os.path.join(args.log_dir, "walking_signal_energy_pred_comparison_all.csv"), index=False)
        print(f"\nWrote walking_signal_energy_per_window.csv, walking_signal_energy_per_subject.csv, "
              f"walking_signal_energy_correlation.csv, walking_signal_energy_pred_comparison_pure.csv, "
              f"walking_signal_energy_pred_comparison_all.csv to {args.log_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write CSV outputs: {e}")


if __name__ == "__main__":
    main()
