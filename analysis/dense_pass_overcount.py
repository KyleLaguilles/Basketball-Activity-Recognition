#!/usr/bin/env python3
"""
Where the dense pass over-count comes from: what is true when the model predicts pass?

Dense pass duration runs well over ground truth (+54.8% on dense_b10_seed1). Duration
bias is one-directional by construction -- it is precision failing, not recall -- so
the diagnostic question is not "how much" but "over WHAT": every over-counted second
is a false-positive pass sample sitting on top of some other class's timeline.

TWO BREAKDOWNS, because they separate two different mechanisms:

  (1) BY TRUE CLASS. Of every sample the model predicts pass, what was actually there?
      A flat spread over the whole timeline is a threshold/prior problem. Mass
      concentrated on one class is a discriminability problem between that class and
      pass.

  (2) BY DISTANCE TO A TRUE PASS. For each false positive, the distance in samples to
      the nearest genuine pass sample. This is what separates BOUNDARY BLEED --
      the model firing pass slightly early or late around a real pass, which inflates
      duration without being a detection error -- from FREE-STANDING false positives
      out in unrelated activity, which are.

      A false positive at distance d <= --near_n is "near"; the default 25 samples is
      0.5 s at 50 Hz. Because a near FP by definition abuts a true pass, its true class
      is the class the model is bleeding INTO the pass from -- which is where a
      dribbling->pass transition would show up, and the script reports the leading /
      trailing split so an early-fire and a late-release are distinguishable.

DISTANCES DO NOT CROSS A SEAM. A dense npz is the subject's kept seam-map segments
concatenated with no marker at the joins; a seam is a >25 ms timestamp gap
(build_seam_map.py:68). Measuring "12 samples away" across one would be measuring
across a hole in time, so every distance is computed inside its own segment and a
false positive in a segment holding no true pass at all is reported separately rather
than being assigned a fake large distance.

USAGE
    python analysis/dense_pass_overcount.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/<ts>_dense_b10_seed1

    python analysis/dense_pass_overcount.py --results_dir <dir> --near_n 50
    python analysis/dense_pass_overcount.py --results_dir <dir> --class_name layup
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

DEFAULT_NEAR_N = 25          # 0.5 s at 50 Hz
FAR = np.iinfo(np.int64).max  # sentinel: no true pass anywhere in this segment


def signed_distance_to_class(y_true, blocks, cls):
    """
    Per-sample signed distance to the nearest true `cls` sample, computed within blocks.

    Returns (dist, side) where dist is the absolute sample distance (0 on a true `cls`
    sample, FAR when the enclosing block holds none) and side is -1 if the nearest true
    `cls` lies AFTER the sample (so the sample leads the pass -- an early fire), +1 if
    it lies BEFORE (the sample trails it -- a late release), 0 on a tie or on-class.
    """
    n = len(y_true)
    dist = np.full(n, FAR, dtype=np.int64)
    side = np.zeros(n, dtype=np.int8)

    for b0, b1 in blocks:
        idx = np.flatnonzero(y_true[b0:b1] == cls)
        if idx.size == 0:
            continue                      # stays FAR: no true `cls` in this segment
        pos = np.arange(b1 - b0)
        j = np.searchsorted(idx, pos)
        prev_d = np.where(j > 0, pos - idx[np.clip(j - 1, 0, idx.size - 1)], FAR)
        next_d = np.where(j < idx.size, idx[np.clip(j, 0, idx.size - 1)] - pos, FAR)
        d = np.minimum(prev_d, next_d)
        s = np.where(next_d < prev_d, -1, np.where(prev_d < next_d, 1, 0)).astype(np.int8)
        dist[b0:b1] = d
        side[b0:b1] = s
    return dist, side


def collect(run, cls):
    """Concatenate per-fold distances, keeping each fold's segment blocks separate."""
    dists, sides = [], []
    for fold in run["folds"]:
        blocks = dc.blocks_of(fold, span_seams=False)
        d, s = signed_distance_to_class(fold["y_true"], blocks, cls)
        dists.append(d)
        sides.append(s)
    return np.concatenate(dists), np.concatenate(sides)


def report_duration(y_true, y_pred, cls, name):
    true_n = int((y_true == cls).sum())
    pred_n = int((y_pred == cls).sum())
    if true_n == 0:
        fail(f"class {name!r} has zero true samples in this run; there is no duration to bias.")
    bias = (pred_n - true_n) / true_n

    print()
    print(f"DURATION BIAS -- {name}")
    print("-" * 78)
    print(f"  true       : {true_n:8d} samples  ({true_n / dc.SAMPLING_RATE / 60:8.2f} min)")
    print(f"  predicted  : {pred_n:8d} samples  ({pred_n / dc.SAMPLING_RATE / 60:8.2f} min)")
    print(f"  bias       : {pred_n - true_n:+8d} samples  ({bias:+.1%})   "
          f"-> {'OVER' if bias > 0 else 'UNDER'}-counted")
    return true_n, pred_n


def report_by_true_class(y_true, y_pred, cls, name):
    """Breakdown (1): of every predicted-`cls` sample, what was actually there?"""
    sel = y_pred == cls
    total = int(sel.sum())
    if total == 0:
        fail(f"the model never predicts {name!r} in this run; there is nothing to break down.")
    counts = np.bincount(y_true[sel], minlength=N_CLASSES)
    if counts.sum() != total:
        fail(f"true-class breakdown totals {counts.sum()} but {total} samples were predicted "
             f"{name}. The breakdown is not a partition.")

    tp = int(counts[dc.class_id(name)])
    fp = total - tp
    print()
    print(f"(1) WHEN THE MODEL PREDICTS {name.upper()} ({total} samples), THE TRUTH IS")
    print("-" * 78)
    print(f"  {'true class':<12}{'samples':>9}{'of preds':>12}{'of FPs':>10}")
    for i in np.argsort(-counts):
        if counts[i] == 0:
            continue
        if i == cls:
            print(f"  {CLASS_NAMES[i]:<12}{int(counts[i]):>9d}{counts[i] / total:>12.1%}"
                  f"{'--':>10}   <- correct")
        else:
            print(f"  {CLASS_NAMES[i]:<12}{int(counts[i]):>9d}{counts[i] / total:>12.1%}"
                  f"{counts[i] / max(fp, 1):>10.1%}")
    print("-" * 78)
    print(f"  true positives : {tp:8d}   ({tp / total:.1%} of predictions)  "
          f"precision = {tp / total:.4f}")
    print(f"  false positives: {fp:8d}   ({fp / total:.1%} of predictions)")

    fp_counts = counts.copy()
    fp_counts[cls] = 0
    if fp:
        top = int(np.argmax(fp_counts))
        print(f"  >> dominant false-positive source: {CLASS_NAMES[top]} "
              f"({fp_counts[top] / fp:.1%} of all false positives, {int(fp_counts[top])} samples)")
    return sel, tp, fp, fp_counts


def report_by_distance(y_true, y_pred, sel, dist, side, cls, name, near_n, fp, fp_counts):
    """Breakdown (2): boundary bleed vs free-standing false positives."""
    fp_mask = sel & (y_true != cls)
    d = dist[fp_mask]
    s = side[fp_mask]
    t = y_true[fp_mask]

    orphan = d == FAR                       # segment contains no true pass at all
    near = (~orphan) & (d <= near_n)
    far = (~orphan) & (d > near_n)
    if int(near.sum() + far.sum() + orphan.sum()) != fp:
        fail("near/far/orphan split does not partition the false positives.")

    print()
    print(f"(2) FALSE-POSITIVE {name.upper()} BY DISTANCE TO THE NEAREST TRUE {name.upper()}")
    print(f"    near = within {near_n} samples ({near_n / dc.SAMPLING_RATE:.2f} s); "
          "distances never cross a seam")
    print("-" * 78)
    for label, m in (("near  (boundary bleed)", near),
                     ("far   (free-standing)", far),
                     (f"no true {name} in segment", orphan)):
        n = int(m.sum())
        print(f"  {label:<28}{n:>9d}   {n / fp:>7.1%} of false positives   "
              f"({n / dc.SAMPLING_RATE / 60:6.2f} min)")
    print("-" * 78)
    share = near.sum() / fp
    print(f"  >> {share:.1%} of the over-count is boundary bleed around genuine passes; "
          f"{1 - share:.1%} is not.")
    verdict = ("BOUNDARY-BLEED-DOMINATED" if share > 0.5 else "FREE-STANDING-DOMINATED")
    print(f"  -> {verdict}")

    if near.sum():
        lead = int((s[near] == -1).sum())
        trail = int((s[near] == 1).sum())
        tie = int((s[near] == 0).sum())
        print()
        print(f"  of the {int(near.sum())} near false positives:")
        print(f"    {lead:>8d}  ({lead / near.sum():>6.1%}) LEAD the true pass  "
              "(model fires early)")
        print(f"    {trail:>8d}  ({trail / near.sum():>6.1%}) TRAIL the true pass "
              "(model releases late)")
        if tie:
            print(f"    {tie:>8d}  ({tie / near.sum():>6.1%}) equidistant")
        med = int(np.median(d[near]))
        print(f"    median distance: {med} samples ({med / dc.SAMPLING_RATE * 1000:.0f} ms)")

    # The cross-tab: true class x near/far. This is where dribbling->pass shows up.
    print()
    print(f"  CROSS-TAB -- true class x proximity (rows sum to that class's FP total)")
    print("-" * 78)
    print(f"  {'true class':<12}{'FP total':>10}{'near':>10}{'far':>10}{'no-pass seg':>13}"
          f"{'near %':>9}{'lead/trail':>13}")
    order = np.argsort(-fp_counts)
    for i in order:
        if fp_counts[i] == 0:
            continue
        mi = t == i
        n_near = int((near & mi).sum())
        n_far = int((far & mi).sum())
        n_orph = int((orphan & mi).sum())
        tot = n_near + n_far + n_orph
        lt = "--"
        if n_near:
            l = int((s[near & mi] == -1).sum())
            lt = f"{l}/{n_near - l}"
        print(f"  {CLASS_NAMES[i]:<12}{tot:>10d}{n_near:>10d}{n_far:>10d}{n_orph:>13d}"
              f"{n_near / tot:>9.1%}{lt:>13}")
    print("-" * 78)
    print("  lead/trail counts the near FPs that fire before vs after the true pass.")
    return int(near.sum())


def report_transitions(run, cls, name, near_n, n_near_expected):
    """
    Which class the model is bleeding in from, at each true-`cls` event edge.

    For every true `cls` event, look at the --near_n samples immediately before its start
    and after its end, and count the ones the model already/still calls `cls`, keyed by
    what the truth says is there. A dribbling->pass hypothesis predicts dribbling to
    dominate the leading side.

    EACH FALSE POSITIVE IS COUNTED ONCE. Two true events less than 2*near_n apart have
    overlapping windows, and a sample in the overlap would otherwise be counted on both
    -- inflating this table above the near total in (2) with no way to see it had
    happened. Samples are claimed by the first event that reaches them, and the total is
    asserted against (2)'s near count, so the two sections cannot disagree.
    """
    lead_counts = np.zeros(N_CLASSES, dtype=np.int64)
    trail_counts = np.zeros(N_CLASSES, dtype=np.int64)
    n_events = 0

    for fold in run["folds"]:
        y_true, y_pred = fold["y_true"], fold["y_pred"]
        claimed = np.zeros(len(y_true), dtype=bool)
        for b0, b1 in dc.blocks_of(fold, span_seams=False):
            starts, ends = dc.contiguous_runs(y_true[b0:b1] == cls)
            for st, en in zip(starts, ends):
                n_events += 1
                lo, hi = b0 + int(st), b0 + int(en)
                for k in range(max(b0, lo - near_n), lo):
                    if y_pred[k] == cls and y_true[k] != cls and not claimed[k]:
                        claimed[k] = True
                        lead_counts[y_true[k]] += 1
                for k in range(hi, min(b1, hi + near_n)):
                    if y_pred[k] == cls and y_true[k] != cls and not claimed[k]:
                        claimed[k] = True
                        trail_counts[y_true[k]] += 1

    total_edge = int(lead_counts.sum() + trail_counts.sum())
    if total_edge != n_near_expected:
        fail(f"edge bleed totals {total_edge} samples but section (2) found "
             f"{n_near_expected} near false positives. Every near false positive lies "
             "within near_n of some event edge by construction, so these must agree.")

    print()
    print(f"(3) BLEED AT TRUE {name.upper()} EVENT EDGES  "
          f"({n_events} events, +/-{near_n} samples)")
    print("-" * 78)
    print(f"  {'true class':<12}{'before event':>14}{'after event':>14}{'total':>10}")
    tot = lead_counts + trail_counts
    for i in np.argsort(-tot):
        if tot[i] == 0:
            continue
        print(f"  {CLASS_NAMES[i]:<12}{int(lead_counts[i]):>14d}{int(trail_counts[i]):>14d}"
              f"{int(tot[i]):>10d}")
    print("-" * 78)
    print(f"  {'TOTAL':<12}{int(lead_counts.sum()):>14d}{int(trail_counts.sum()):>14d}"
          f"{int(tot.sum()):>10d}")
    if lead_counts.sum():
        top = int(np.argmax(lead_counts))
        print(f"  >> leading edge is dominated by {CLASS_NAMES[top]} "
              f"({lead_counts[top] / lead_counts.sum():.1%} of pre-event bleed) -- "
              f"i.e. {CLASS_NAMES[top]}->{name} transitions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    dc.add_common_args(parser)
    parser.add_argument("--class_name", default="pass",
                        help="Over-counted class to diagnose (default: pass).")
    parser.add_argument("--near_n", type=int, default=DEFAULT_NEAR_N,
                        help=f"A false positive within this many samples of a true "
                             f"{'pass'} counts as boundary bleed (default {DEFAULT_NEAR_N} "
                             "= 0.5 s at 50 Hz).")
    args = parser.parse_args()

    if args.near_n < 0:
        fail(f"--near_n must be >= 0 (got {args.near_n})")

    run = dc.load_dense_run(args.results_dir, args.labels, args.seam_map,
                            args.npz_pattern, args.allow_partial_folds)
    cls = dc.class_id(args.class_name)
    dc.run_header(run, extra=[
        f"class          : {args.class_name} (id {cls})",
        f"near_n         : {args.near_n} samples "
        f"({args.near_n / dc.SAMPLING_RATE:.2f} s)",
    ])

    y_true, y_pred = run["y_true"], run["y_pred"]
    report_duration(y_true, y_pred, cls, args.class_name)
    sel, tp, fp, fp_counts = report_by_true_class(y_true, y_pred, cls, args.class_name)

    if fp == 0:
        print("\nno false positives; nothing further to break down.")
        return

    dist, side = collect(run, cls)
    if len(dist) != len(y_true):
        fail(f"distance array is {len(dist)} long but the timeline is {len(y_true)}.")
    on_class = dist[y_true == cls]
    if on_class.size and on_class.max() != 0:
        fail("a true-class sample has nonzero distance to itself; the distance transform "
             "is wrong.")

    n_near = report_by_distance(y_true, y_pred, sel, dist, side, cls, args.class_name,
                                args.near_n, fp, fp_counts)
    report_transitions(run, cls, args.class_name, args.near_n, n_near)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
