"""
Determinism check for the last-epoch reruns. Read-only, CPU-only, run from the repo root.

The last-epoch run should follow the same training trajectory as the original run, so any
fold whose ORIGINAL best epoch was 40 must have byte-identical predictions in the new run.
If one of those folds differs, the reruns did not reproduce the originals, and the
last-epoch numbers cannot be read as "the same model, scored at a different epoch".

Usage:
    python analysis/check_lastepoch_determinism.py \
        --original logs/subset_specific/loso_G/inceptioncontext/2026-09-11_12-06-12_24fold_dense_seed1 \
        --last     logs/subset_specific/loso_G/inceptioncontext/<timestamp>_24fold_dense_last_seed1
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

EPOCHS = 40


def npz_by_fold(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "preds_*_seed*.npz")))
    out = {}
    for f in files:
        # preds_<id>_<eu|na>_<fraction>_seed<N>.npz -> fold key "<id>_<eu|na>"
        parts = os.path.basename(f).split("_")
        out[f"{parts[1]}_{parts[2]}"] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--last", required=True)
    args = ap.parse_args()

    orig, last = npz_by_fold(args.original), npz_by_fold(args.last)
    if len(orig) != 24 or set(orig) != set(last):
        sys.exit(f"HARD FAIL: fold sets differ or are not 24 (original {len(orig)}, last {len(last)})")

    summary = pd.read_csv(os.path.join(args.original, "subject_best_summary.csv"))
    best = dict(zip(summary["subject"].astype(str), summary["best_epoch"].astype(int)))
    if set(best) != set(orig):
        sys.exit("HARD FAIL: subject_best_summary.csv folds do not match the npz folds")

    checked = mismatched = 0
    for fold in sorted(orig):
        a, b = np.load(orig[fold]), np.load(last[fold])
        if not np.array_equal(a["y_true"], b["y_true"]):
            sys.exit(f"HARD FAIL: y_true differs for {fold}; the runs did not score the same samples")
        same = np.array_equal(a["y_pred"], b["y_pred"])
        if best[fold] == EPOCHS:
            checked += 1
            if not same:
                mismatched += 1
                print(f"MISMATCH  {fold}: original best epoch was {EPOCHS}, but predictions differ")
        print(f"{fold:10s} original best epoch {best[fold]:2d}   predictions identical: {same}")

    print(f"\nFolds whose original best epoch was {EPOCHS}: {checked}; mismatched: {mismatched}")
    if checked == 0:
        print("No fold peaked at the last epoch, so this run cannot certify determinism by itself.")
    elif mismatched:
        sys.exit("HARD FAIL: the rerun did not reproduce the original trajectory")
    else:
        print("OK: every fold that peaked at the last epoch reproduces exactly.")


if __name__ == "__main__":
    main()
