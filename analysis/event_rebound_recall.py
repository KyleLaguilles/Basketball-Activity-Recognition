#!/usr/bin/env python3
"""
Per-EVENT rebound recall: is the residual error boundary trimming, or whole-event misses?

Sample-level rebound F1 cannot tell those two apart. A model that finds every rebound
but clips 40% of each one's samples off the edges, and a model that nails 60% of the
rebounds and never sees the rest, land on the same per-sample recall. They call for
opposite fixes -- the first is a boundary/smoothing problem, the second is a detection
problem -- so this script scores the EVENT as the unit.

DEFINITION. An event is a maximal contiguous run of the rebound label in the certified
per-sample ground truth. Its recall is the fraction of its own samples the model
predicted rebound: mean(y_pred[start:end] == rebound). No matching, no tolerance
window, no IoU threshold -- the denominator is the event's own extent, so the number
is directly readable as "how much of this rebound did the model see".

CONTIGUITY STOPS AT A SEAM. A dense npz is the subject's kept seam-map segments
concatenated with no marker at the joins, and a seam is a >25 ms timestamp gap
(build_seam_map.py:68) -- at least one deleted sample. Two rebound samples either side
of one are not contiguous in time, so by default an event does not span a seam.

    seam-aware (default) : 162 events across all 14 subjects
    --span_seams         : 161 events -- the recorded total

The single event responsible for the difference is in ac59: ac59_eu segment 3 ends on
rebound at row 834528 and segment 4 begins on rebound at that same row. The rows are
adjacent but a timestamp gap separates them, so it is one rebound interrupted by
dropped samples. Both readings are defensible; the script prints the reconciliation and
asserts whichever total the chosen rule implies, so neither can drift unnoticed.

BOUNDARY vs WHOLE-EVENT is then read off the trim decomposition. For each detected
event the missed samples are split into a leading trim (before the first predicted
rebound), a trailing trim (after the last), and interior gaps. An error budget that is
mostly trim is a boundary problem; one dominated by zero-recall events is not.

USAGE
    python analysis/event_rebound_recall.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/<ts>_dense_b10_seed1

    # reproduce the recorded 161-event total (needs the all-14-subject run)
    python analysis/event_rebound_recall.py --results_dir <all14_dir> --span_seams

    # save the histogram as a figure as well as printing it
    python analysis/event_rebound_recall.py --results_dir <dir> --save_fig figures/x.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dense_common as dc          # noqa: E402
from _dense_common import fail      # noqa: E402

HardFail = dc.HardFail

# Event totals over all 14 subjects, verified against the certified ground truth at
# dense_min_seg=25. Printed for comparison on any smaller fold set; ASSERTED when the
# run covers all 14, because then it is a closed property of the dataset, not a result.
ANCHOR_EVENTS_ALL14 = {False: 162, True: 161}   # keyed by span_seams

HIT_THRESHOLD = 0.5     # "hit" == the model saw more than half the event


def build_events(run, cls, span_seams):
    """One record per contiguous ground-truth run of `cls`, with its prediction trims."""
    events = []
    for fold in run["folds"]:
        y_true, y_pred = fold["y_true"], fold["y_pred"]
        for b0, b1 in dc.blocks_of(fold, span_seams):
            starts, ends = dc.contiguous_runs(y_true[b0:b1] == cls)
            for s, e in zip(starts, ends):
                lo, hi = b0 + int(s), b0 + int(e)
                hit = (y_pred[lo:hi] == cls)
                n = hi - lo
                n_hit = int(hit.sum())
                if n_hit:
                    first, last = int(np.argmax(hit)), n - 1 - int(np.argmax(hit[::-1]))
                    lead, trail = first, n - 1 - last
                    interior = n - n_hit - lead - trail
                else:
                    lead = trail = 0
                    interior = 0
                events.append({
                    "fold": fold["fold"], "start": lo, "end": hi, "length": n,
                    "n_hit": n_hit, "recall": n_hit / n,
                    "lead_trim": lead, "trail_trim": trail, "interior_gap": interior,
                })
    return events


def certify_total(events, run, span_seams):
    """The event count is a property of the ground truth, so it is checked, not trusted."""
    n = len(events)
    rule = "spanning seams" if span_seams else "seam-aware"
    if run["all_subjects"]:
        want = ANCHOR_EVENTS_ALL14[span_seams]
        if n != want:
            fail(f"found {n} rebound events across all {run['n_folds']} subjects under the "
                 f"{rule} rule, but the certified ground truth holds {want}. Either the seam "
                 "map, the label export, or the event definition has changed; nothing below "
                 "is valid.")
        other = ANCHOR_EVENTS_ALL14[not span_seams]
        print(f"event total    : {n}  ({rule} rule) -- matches the certified dataset total.")
        print(f"                 the {'seam-aware' if span_seams else 'spanning-seams'} rule "
              f"gives {other}; the one differing event is ac59_eu seg3/seg4, a rebound "
              "interrupted by dropped samples at row 834528.")
    else:
        print(f"event total    : {n}  ({rule} rule) over {run['n_folds']} of "
              f"{dc.EXPECTED_FOLD_COUNT_ALL} subjects.")
        print(f"                 the recorded {ANCHOR_EVENTS_ALL14[span_seams]}-event total is "
              "the ALL-SUBJECT figure and is not reproducible from this fold set; run the "
              "all-14 dense run to compare against it.")


def print_histogram(events, bins=10, width=46):
    """Text histogram of per-event recall, with exact-0 and exact-1 broken out."""
    rec = np.array([e["recall"] for e in events])
    edges = np.linspace(0.0, 1.0, bins + 1)
    # right-closed except the first bin, so a 0.0 event cannot land with the 0-10% partials
    idx = np.clip(np.ceil(rec * bins).astype(int) - 1, 0, bins - 1)
    idx[rec == 0.0] = 0
    counts = np.bincount(idx, minlength=bins)
    n_zero = int((rec == 0.0).sum())
    n_full = int((rec == 1.0).sum())
    peak = max(counts.max(), 1)

    print()
    print("PER-EVENT REBOUND RECALL -- distribution")
    print("-" * 78)
    for b in range(bins):
        lo, hi = 100 * edges[b], 100 * edges[b + 1]
        bar = "#" * int(round(width * counts[b] / peak))
        note = ""
        if b == 0 and n_zero:
            note = f"  (of which {n_zero} exactly 0%)"
        if b == bins - 1 and n_full:
            note = f"  (of which {n_full} exactly 100%)"
        print(f"  [{lo:3.0f}%, {hi:3.0f}%]{'*' if b == 0 else ' '} {counts[b]:4d} |{bar:<{width}}|{note}")
    print("-" * 78)
    print("  * the first bin is closed on the left so a 0% event never mixes with partials")
    return counts


def print_summary(events):
    rec = np.array([e["recall"] for e in events])
    length = np.array([e["length"] for e in events])
    n = len(events)
    n_zero = int((rec == 0.0).sum())
    n_hit = int((rec > HIT_THRESHOLD).sum())
    n_any = int((rec > 0.0).sum())

    print()
    print("SUMMARY")
    print("-" * 78)
    print(f"  events                        : {n}")
    print(f"  recall > {HIT_THRESHOLD:.0%} ('hit')            : {n_hit:4d}  ({n_hit / n:6.1%})")
    print(f"  recall > 0%  (touched at all) : {n_any:4d}  ({n_any / n:6.1%})")
    print(f"  recall = 0%  (MISSED WHOLLY)  : {n_zero:4d}  ({n_zero / n:6.1%})")
    print()
    print(f"  mean per-event recall         : {rec.mean():.4f}   (each event weighted equally)")
    print(f"  median per-event recall       : {np.median(rec):.4f}")
    print(f"  sample-weighted recall        : {(rec * length).sum() / length.sum():.4f}   "
          "(== the per-sample rebound recall)")
    print()
    print(f"  event length (samples)        : min={length.min()} median={int(np.median(length))} "
          f"max={length.max()}  mean={length.mean():.1f}")
    print(f"  event duration (s)            : min={length.min() / dc.SAMPLING_RATE:.2f} "
          f"median={np.median(length) / dc.SAMPLING_RATE:.2f} "
          f"max={length.max() / dc.SAMPLING_RATE:.2f}")


def print_error_budget(events):
    """Where the missed samples actually go: trimmed edges, interior gaps, or missed events."""
    total = sum(e["length"] for e in events)
    hit = sum(e["n_hit"] for e in events)
    missed = total - hit

    detected = [e for e in events if e["n_hit"] > 0]
    zero = [e for e in events if e["n_hit"] == 0]
    lead = sum(e["lead_trim"] for e in detected)
    trail = sum(e["trail_trim"] for e in detected)
    interior = sum(e["interior_gap"] for e in detected)
    whole = sum(e["length"] for e in zero)

    if lead + trail + interior + whole != missed:
        fail(f"error budget does not close: lead {lead} + trail {trail} + interior {interior} "
             f"+ whole-event {whole} != missed {missed}. The trim decomposition is wrong.")

    print()
    print("ERROR BUDGET -- where the missed rebound samples go")
    print("-" * 78)
    print(f"  rebound samples in ground truth : {total}")
    print(f"  predicted rebound (correctly)   : {hit}   ({hit / total:.1%})")
    print(f"  missed                          : {missed}   ({missed / total:.1%})")
    print()
    if missed:
        for label, val in (("leading-edge trim", lead), ("trailing-edge trim", trail),
                           ("interior gaps", interior), ("wholly-missed events", whole)):
            print(f"    {label:<26}: {val:6d}   {val / missed:6.1%} of all missed samples")
        print()
        boundary = (lead + trail) / missed
        print(f"  boundary trimming accounts for {boundary:.1%} of the error, "
              f"whole-event misses for {whole / missed:.1%}.")
        verdict = ("BOUNDARY-DOMINATED" if boundary > 0.5 else
                   "WHOLE-EVENT-DOMINATED" if whole / missed > 0.5 else "MIXED")
        print(f"  -> {verdict}")
    print()
    print(f"  mean trim on detected events    : {lead / max(len(detected), 1):.1f} samples leading, "
          f"{trail / max(len(detected), 1):.1f} trailing "
          f"({lead / max(len(detected), 1) / dc.SAMPLING_RATE * 1000:.0f} / "
          f"{trail / max(len(detected), 1) / dc.SAMPLING_RATE * 1000:.0f} ms)")


def print_per_fold(events):
    print()
    print("PER-FOLD")
    print("-" * 78)
    print(f"  {'fold':<8}{'events':>8}{'hit>50%':>9}{'missed':>8}{'mean rec':>10}{'samp rec':>10}")
    for fold in sorted({e["fold"] for e in events}):
        sub = [e for e in events if e["fold"] == fold]
        rec = np.array([e["recall"] for e in sub])
        ln = np.array([e["length"] for e in sub])
        print(f"  {fold:<8}{len(sub):>8}{int((rec > HIT_THRESHOLD).sum()):>9}"
              f"{int((rec == 0).sum()):>8}{rec.mean():>10.4f}"
              f"{(rec * ln).sum() / ln.sum():>10.4f}")


def save_figure(events, counts, path, run_name):
    import matplotlib
    matplotlib.use("Agg")            # CPU-only, no display
    import matplotlib.pyplot as plt

    rec = np.array([e["recall"] for e in events])
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(counts)) * 10 + 5, counts, width=9.2,
           color="#4477aa", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("per-event rebound recall (%)")
    ax.set_ylabel("events")
    ax.set_title(f"Per-event rebound recall -- {run_name}\n"
                 f"{len(events)} events, {int((rec == 0).sum())} wholly missed, "
                 f"{int((rec > HIT_THRESHOLD).sum())} above 50%")
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nfigure written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    dc.add_common_args(parser)
    parser.add_argument("--span_seams", action="store_true",
                        help="Let an event span a seam-map segment boundary (the rule that "
                             "reproduces the recorded 161-event total; default is seam-aware, 162).")
    parser.add_argument("--class_name", default="rebound",
                        help="Class to score events for (default: rebound).")
    parser.add_argument("--bins", type=int, default=10, help="Histogram bins (default 10).")
    parser.add_argument("--save_fig", default=None, metavar="PATH",
                        help="Also write the histogram as a PNG here.")
    args = parser.parse_args()

    if args.bins < 1:
        fail(f"--bins must be >= 1 (got {args.bins})")

    run = dc.load_dense_run(args.results_dir, args.labels, args.seam_map,
                            args.npz_pattern, args.allow_partial_folds)
    cls = dc.class_id(args.class_name)

    dc.run_header(run, extra=[
        f"event rule     : {'spanning seams' if args.span_seams else 'seam-aware (default)'}",
        f"class          : {args.class_name} (id {cls})",
    ])

    events = build_events(run, cls, args.span_seams)
    if not events:
        fail(f"no {args.class_name} events found in the certified ground truth for this fold set.")
    certify_total(events, run, args.span_seams)

    counts = print_histogram(events, args.bins)
    print_summary(events)
    print_error_budget(events)
    print_per_fold(events)

    if args.save_fig:
        save_figure(events, counts, args.save_fig, run["name"])


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
