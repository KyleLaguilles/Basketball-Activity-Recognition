#!/usr/bin/env python3
"""
Per-sample confusion matrix over the dense predictions, row-normalized to recall.

WHY PER-SAMPLE AND NOT PER-WINDOW. A windowed confusion matrix is built over windows
whose label is their LAST sample (sliding_window.py:117), so its rows are a
subsample of the timeline taken once every `step` samples and its off-diagonal mass
inherits that rule's bias. The dense npz already holds one prediction per raw 50 Hz
sample, so the matrix here is over the timeline itself: every row sums to that class's
true sample count, and no labeling artifact sits between the model and the score.

ROW NORMALIZATION = RECALL. Cell (i, j) is P(pred = j | true = i), so row i sums to 1
and the diagonal is per-class recall. That is the orientation the rebound question
needs -- "when the truth is rebound, where does the model actually put it" -- and it
is stated explicitly because the transpose (precision) answers a different question
and is printed separately for the flagged class.

THE WINDOWED ANCHOR IS PRINTED, NEVER ASSERTED. Under windowed predictions rebound's
dominant confusion target was running at 44.9%. A dense run is a different model on a
different population, so that number is a reference point for reading the result, not
a property this script checks. It is reported alongside the measured value with the
delta spelled out.

USAGE
    python analysis/dense_confusion_matrix.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/<ts>_dense_b10_seed1

    python analysis/dense_confusion_matrix.py --results_dir <dir> --flag_class layup
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dense_common as dc          # noqa: E402
from _dense_common import fail      # noqa: E402

HardFail = dc.HardFail
CLASS_NAMES = dc.CLASS_NAMES
N_CLASSES = dc.N_CLASSES

# Windowed-prediction reference: rebound's dominant off-diagonal target and its share.
# Printed for comparison only -- a swept/dense config is not the baseline, so this is
# never asserted here.
ANCHOR_WINDOWED_REBOUND = ("running", 0.449)

CW = 10          # per-class column width; longest class name is 'dribbling' (9)


def confusion(y_true, y_pred):
    """Counts matrix, rows = truth, cols = prediction. Built by bincount, then certified."""
    flat = y_true * N_CLASSES + y_pred
    cm = np.bincount(flat, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)

    if cm.sum() != len(y_true):
        fail(f"confusion matrix totals {cm.sum()} but there are {len(y_true)} samples.")
    support = np.bincount(y_true, minlength=N_CLASSES)
    if not np.array_equal(cm.sum(axis=1), support):
        fail("confusion matrix row sums do not equal the per-class true support; "
             "the matrix is not a partition of the timeline.")
    predicted = np.bincount(y_pred, minlength=N_CLASSES)
    if not np.array_equal(cm.sum(axis=0), predicted):
        fail("confusion matrix column sums do not equal the per-class predicted counts.")
    return cm


def print_matrix(cm, normalize, title, note):
    print()
    print(title)
    print(note)
    print("-" * (12 + CW * N_CLASSES + 12))
    head = "".join(f"{n[:CW - 1]:>{CW}}" for n in CLASS_NAMES)
    corner = "true \\ pred"
    print(f"{corner:<12}{head}{'support':>12}")
    for i, name in enumerate(CLASS_NAMES):
        support = cm[i].sum()
        if normalize:
            row = cm[i] / support if support else np.zeros(N_CLASSES)
            cells = "".join(f"{100 * v:>{CW}.1f}" for v in row)
        else:
            cells = "".join(f"{int(v):>{CW}d}" for v in cm[i])
        print(f"{name:<12}{cells}{int(support):>12d}")
    print("-" * (12 + CW * N_CLASSES + 12))


def print_diagonal(cm):
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm)
    recall = np.divide(diag, support, out=np.zeros(N_CLASSES), where=support > 0)
    prec = np.divide(diag, predicted, out=np.zeros(N_CLASSES), where=predicted > 0)
    denom = recall + prec
    f1 = np.divide(2 * recall * prec, denom, out=np.zeros(N_CLASSES), where=denom > 0)

    print()
    print("PER-CLASS (from the same matrix)")
    print("-" * 78)
    print(f"  {'class':<12}{'recall':>10}{'precision':>12}{'f1':>10}{'support':>12}{'predicted':>12}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}{recall[i]:>10.4f}{prec[i]:>12.4f}{f1[i]:>10.4f}"
              f"{int(support[i]):>12d}{int(predicted[i]):>12d}")
    print(f"  {'macro':<12}{recall.mean():>10.4f}{prec.mean():>12.4f}{f1.mean():>10.4f}"
          f"{int(support.sum()):>12d}{int(predicted.sum()):>12d}")
    return recall, prec


def flag_class(cm, name):
    """The headline question: where does one class's mass actually go?"""
    cid = dc.class_id(name)
    support = cm[cid].sum()
    if support == 0:
        fail(f"class {name!r} has zero true samples in this run; nothing to flag.")

    row = cm[cid] / support
    off = [(j, row[j]) for j in range(N_CLASSES) if j != cid]
    off.sort(key=lambda t: -t[1])
    top_j, top_v = off[0]

    print()
    print(f"TOP CONFUSION FOR {name.upper()}")
    print("-" * 78)
    print(f"  true {name} samples          : {int(support)}")
    print(f"  recall (predicted {name}) : {row[cid]:.4f}")
    print()
    print(f"  >> dominant confusion target : {CLASS_NAMES[top_j]}  "
          f"({top_v:.1%} of all true {name} samples)")
    print()
    print("  full off-diagonal ranking:")
    for j, v in off:
        if v > 0:
            print(f"    {CLASS_NAMES[j]:<12}{v:>8.1%}   ({int(cm[cid, j])} samples)")

    if name == "rebound":
        anchor_name, anchor_v = ANCHOR_WINDOWED_REBOUND
        print()
        print(f"  windowed-prediction reference: {anchor_name} at {anchor_v:.1%} "
              "(printed for comparison, never asserted)")
        measured = row[CLASS_NAMES.index(anchor_name)]
        delta_pp = 100 * (measured - anchor_v)
        print(f"  dense measurement            : {anchor_name} at {measured:.1%}  "
              f"({delta_pp:+.1f} pp vs the windowed reference)")
        if top_j == CLASS_NAMES.index(anchor_name):
            print(f"  -> {anchor_name} REMAINS the dominant misclassification target.")
        else:
            print(f"  -> the dominant target has MOVED from {anchor_name} to "
                  f"{CLASS_NAMES[top_j]}.")

    # Precision side: what the model calls `name` but is not.
    col_total = cm[:, cid].sum()
    if col_total:
        col = cm[:, cid] / col_total
        src = [(i, col[i]) for i in range(N_CLASSES) if i != cid]
        src.sort(key=lambda t: -t[1])
        print()
        print(f"  transpose view -- when the model PREDICTS {name} ({int(col_total)} samples), "
              "the truth is:")
        print(f"    {name:<12}{col[cid]:>8.1%}   (correct)")
        for i, v in src[:3]:
            if v > 0:
                print(f"    {CLASS_NAMES[i]:<12}{v:>8.1%}   ({int(cm[i, cid])} samples)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    dc.add_common_args(parser)
    parser.add_argument("--flag_class", default="rebound",
                        help="Class whose confusion is called out in detail (default: rebound).")
    parser.add_argument("--counts", action="store_true",
                        help="Also print the raw count matrix above the normalized one.")
    parser.add_argument("--per_fold", action="store_true",
                        help="Print the flagged class's recall and top target per fold.")
    args = parser.parse_args()

    run = dc.load_dense_run(args.results_dir, args.labels, args.seam_map,
                            args.npz_pattern, args.allow_partial_folds)
    dc.run_header(run)

    cm = confusion(run["y_true"], run["y_pred"])

    if args.counts:
        print_matrix(cm, False, "CONFUSION MATRIX -- raw sample counts",
                     "rows = ground truth, cols = prediction")
    print_matrix(cm, True, "CONFUSION MATRIX -- row-normalized (%, recall)",
                 "rows = ground truth, cols = prediction; row i sums to 100, diagonal = recall")
    print_diagonal(cm)
    flag_class(cm, args.flag_class)

    if args.per_fold:
        cid = dc.class_id(args.flag_class)
        print()
        print(f"PER-FOLD -- {args.flag_class}")
        print("-" * 78)
        print(f"  {'fold':<8}{'support':>10}{'recall':>10}   top target")
        for fold in run["folds"]:
            fcm = confusion(fold["y_true"], fold["y_pred"])
            sup = fcm[cid].sum()
            if not sup:
                print(f"  {fold['fold']:<8}{0:>10}{'--':>10}   (no true samples)")
                continue
            row = fcm[cid] / sup
            off = [(j, row[j]) for j in range(N_CLASSES) if j != cid]
            j, v = max(off, key=lambda t: t[1])
            print(f"  {fold['fold']:<8}{int(sup):>10}{row[cid]:>10.4f}   "
                  f"{CLASS_NAMES[j]} ({v:.1%})")


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
