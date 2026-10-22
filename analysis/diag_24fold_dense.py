#!/usr/bin/env python3
"""
Screening-protocol rebound diagnostics (a) and dense (b), over several dense seeds at once.

A driver, not a reimplementation: each run is loaded and certified by
_dense_common.load_dense_run, events come from event_rebound_recall.build_events and are
checked against the recorded totals by event_rebound_recall.certify_total, and the
confusion matrix is dense_confusion_matrix.confusion. This script only loops over runs
and adds the per-seed table plus mean +/- sample std (ddof=1).

    (a) per-event rebound recall: % of events wholly missed, and the split of missed
        rebound samples into wholly-missed events vs trimmed edges (leading + trailing;
        interior gaps are printed separately and belong to neither)
    (b) % of true rebound samples predicted running

Events use --span_seams by default, the rule behind the recorded 161-event screening
figure; --seam_aware switches to the 162-event rule. Either way certify_total asserts
the dataset total when the run covers every participant.

Read-only and CPU-only.

Usage (from repo root):
    python analysis/diag_24fold_dense.py --results_dir \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_12-06-12_24fold_dense_seed1 \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_14-07-46_24fold_dense_seed2 \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_16-09-07_24fold_dense_seed3
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dense_common as dc                  # noqa: E402
import dense_confusion_matrix as dcm        # noqa: E402
import event_rebound_recall as err          # noqa: E402

REBOUND = dc.class_id("rebound")
RUNNING = dc.class_id("running")


def measure(results_dir, args):
    run = dc.load_dense_run(results_dir, args.labels, args.seam_map, args.npz_pattern,
                            args.allow_partial_folds)
    events = err.build_events(run, REBOUND, not args.seam_aware)
    # certify_total prints its verdict and hard-fails on a mismatch; keep the verdict line.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        err.certify_total(events, run, not args.seam_aware)

    total = sum(e["length"] for e in events)
    hit = sum(e["n_hit"] for e in events)
    missed = total - hit
    detected = [e for e in events if e["n_hit"] > 0]
    lead = sum(e["lead_trim"] for e in detected)
    trail = sum(e["trail_trim"] for e in detected)
    interior = sum(e["interior_gap"] for e in detected)
    whole = sum(e["length"] for e in events if e["n_hit"] == 0)
    if lead + trail + interior + whole != missed:
        dc.fail(f"{results_dir}: error budget does not close")

    cm = dcm.confusion(run["y_true"], run["y_pred"])
    reb_support = int(cm[REBOUND].sum())
    reb_pred = int(cm[:, REBOUND].sum())
    return {
        "name": run["name"], "seed": run["cfg"].get("seed"), "n_folds": run["n_folds"],
        "certify": buf.getvalue().strip().splitlines()[0],
        "n_events": len(events),
        "n_zero": sum(1 for e in events if e["n_hit"] == 0),
        "total": total, "missed": missed,
        "lead": lead, "trail": trail, "interior": interior, "whole": whole,
        "reb_support": reb_support,
        "reb_to_running": int(cm[REBOUND, RUNNING]),
        "recall": cm[REBOUND, REBOUND] / reb_support,
        "precision": cm[REBOUND, REBOUND] / reb_pred if reb_pred else float("nan"),
    }


def metrics(r):
    return {
        "events wholly missed (%)": 100 * r["n_zero"] / r["n_events"],
        "missed samples from missed events (%)": 100 * r["whole"] / r["missed"],
        "missed samples from trimmed edges (%)": 100 * (r["lead"] + r["trail"]) / r["missed"],
        "  of which leading edge (%)": 100 * r["lead"] / r["missed"],
        "  of which trailing edge (%)": 100 * r["trail"] / r["missed"],
        "missed samples from interior gaps (%)": 100 * r["interior"] / r["missed"],
        "rebound samples predicted running (%)": 100 * r["reb_to_running"] / r["reb_support"],
        "rebound recall": r["recall"],
        "rebound precision": r["precision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", nargs="+", required=True, metavar="DIR",
                        help="One or more --dense run dirs, one per seed.")
    parser.add_argument("--labels", default="labels_export.csv.gz")
    parser.add_argument("--seam_map", default="data/seam_map.json")
    parser.add_argument("--npz_pattern", default=None)
    parser.add_argument("--allow_partial_folds", action="store_true")
    parser.add_argument("--seam_aware", action="store_true",
                        help="Stop events at seam-map segment boundaries (162-event rule) instead "
                             "of the default --span_seams rule (161 events).")
    args = parser.parse_args()

    recs = [measure(d, args) for d in args.results_dir]

    rule = "seam-aware" if args.seam_aware else "spanning seams"
    print("=" * 100)
    print(f"(a) + dense (b) -- InceptionContext dense, event rule: {rule}")
    print("=" * 100)
    for d, r in zip(args.results_dir, recs):
        print(f"  seed {r['seed']}  {r['n_folds']} folds  {d}")
        print(f"          {r['certify']}")
        print(f"          events={r['n_events']}  wholly missed={r['n_zero']}  "
              f"rebound samples={r['total']}  missed={r['missed']}  "
              f"(lead {r['lead']} / trail {r['trail']} / interior {r['interior']} / "
              f"whole-event {r['whole']})  rebound->running={r['reb_to_running']}")

    per = [metrics(r) for r in recs]
    seeds = [f"seed {r['seed']}" for r in recs]
    print(f"\n  {'metric':<40}" + "".join(f"{s:>10}" for s in seeds) + f"{'mean':>10}{'std':>8}")
    print("  " + "-" * (40 + 10 * len(seeds) + 18))
    for key in per[0]:
        vals = np.array([p[key] for p in per], dtype=float)
        std = vals.std(ddof=1) if len(vals) > 1 else float("nan")
        fmt = ".4f" if not key.endswith("(%)") else ".1f"
        print(f"  {key:<40}" + "".join(f"{v:>10{fmt}}" for v in vals)
              + f"{vals.mean():>10{fmt}}{std:>8{fmt}}")
    print("\n  std is the sample std (ddof=1) across seeds. Edge-trim + whole-event + interior = 100%.")


if __name__ == "__main__":
    try:
        main()
    except dc.HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
