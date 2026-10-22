#!/usr/bin/env python3
"""
Screening-protocol rebound diagnostics (c) and windowed (b), for any set of windowed runs.

window_purity.py's reconstruction is reused unchanged -- load_labels ->
build_candidate_windows -> resolve_and_certify -> build_window_table ->
attach_predictions -- but its run discovery is not: discover_npz hardcodes the three
July baseline dirs (BASELINE_DIRS) and a `preds_<fold>_1.0.npz` name. Here the run dirs
come from the command line, the folds from each run's cfg.txt, and the npz from
sample_level_f1.discover_fold_npz, which accepts the seed-tagged names.

KEYING SHIM. resolve_and_certify reads its input as npz_paths[BASELINE_DIRS[0]], and
attach_predictions names its column pred_seed<SEED_OF_DIR[key]>. Both use the key only
as a label, so each run's {fold: npz} dict is handed in under BASELINE_DIRS[seed - 1].
That is sound because resolution only needs y_true, and attach_predictions re-asserts
every run's y_true against the certified reconstruction, so runs whose windows differ
cannot be joined silently. Seeds must therefore be distinct and in 1..len(BASELINE_DIRS).

Reported per seed and pooled over seeds (pooled = summed counts; the seeds are
correlated replicates over the same windows, not independent samples):
    - % of rebound windows predicted running                         (windowed b)
    - pure-rebound share among rebound windows predicted running     (c, the 109/301 rule)
    - pure-rebound share among ALL misclassified rebound windows     (c, the 189/549 rule)
"pure" is window_purity's purity == 1.0: all 50 samples carry the last-sample label.

Read-only and CPU-only.

Usage (from repo root):
    python analysis/diag_24fold_windowed.py --results_dir \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_18-13-22_24fold_windowed_seed1 \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_19-25-31_24fold_windowed_seed2 \
        logs/subset_specific/loso_G/inceptioncontext/2026-09-11_20-37-56_24fold_windowed_seed3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_level_f1 as slf       # noqa: E402
import window_purity as wp          # noqa: E402

REBOUND = wp.CLASS_NAMES.index("rebound")
RUNNING = wp.CLASS_NAMES.index("running")


def discover_run(results_dir, npz_pattern):
    """(seed, {fold: npz}) for one windowed run, with window_purity's windowing asserted."""
    cfg = slf.load_cfg(results_dir)
    _sw, _ov, win_len, step = slf.derive_windowing(cfg, results_dir)
    if (win_len, step) != (wp.WIN_LEN, wp.STEP):
        raise wp.HardFail(f"{results_dir}: win_len/step {win_len}/{step} != window_purity's "
                          f"pinned {wp.WIN_LEN}/{wp.STEP}")
    folds = cfg.get("loso_subjects")
    if not folds:
        raise wp.HardFail(f"{results_dir}/cfg.txt has loso_subjects={folds!r}; refusing to guess folds")
    seed = int(cfg["seed"])
    return seed, slf.discover_fold_npz(results_dir, folds, npz_pattern, expect_count=len(folds))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", nargs="+", required=True, metavar="DIR",
                        help="Windowed run dirs, one per seed, all over the same folds.")
    parser.add_argument("--labels", default="labels_export.csv.gz")
    parser.add_argument("--npz_pattern", default=None)
    args = parser.parse_args()

    runs = {}
    for d in args.results_dir:
        seed, paths = discover_run(d, args.npz_pattern)
        if seed in runs:
            raise wp.HardFail(f"seed {seed} given twice ({runs[seed][0]} and {d})")
        if not 1 <= seed <= len(wp.BASELINE_DIRS):
            raise wp.HardFail(f"{d}: seed {seed} has no window_purity column key "
                              f"(supported 1..{len(wp.BASELINE_DIRS)})")
        runs[seed] = (d, paths)
    fold_sets = {tuple(sorted(p)) for _d, p in runs.values()}
    if len(fold_sets) != 1:
        raise wp.HardFail(f"runs disagree on their fold sets: {fold_sets}")

    labels_df = wp.load_labels(args.labels)
    candidates = wp.build_candidate_windows(labels_df)
    first = min(runs)
    resolved = wp.resolve_and_certify({wp.BASELINE_DIRS[0]: runs[first][1]}, candidates)
    table = wp.build_window_table(resolved)
    table = wp.attach_predictions(
        table, {wp.BASELINE_DIRS[s - 1]: p for s, (_d, p) in runs.items()}, resolved)

    n_wrong_token = sum(1 for f, v in resolved.items() if v[0] != f)
    reb = table[table["last_label"] == REBOUND]
    pure = reb["purity"].to_numpy() == 1.0

    print("=" * 100)
    print("(c) + windowed (b) -- InceptionContext windowed, rebound windows (last-sample label)")
    print("=" * 100)
    for s in sorted(runs):
        print(f"  seed {s}  {runs[s][0]}")
    print(f"\n  folds certified: {len(resolved)} (window count + byte-identical last-sample labels; "
          f"{n_wrong_token} fold token(s) resolved to a different subject)")
    print(f"  val windows: {len(table)}   rebound-labeled: {len(reb)}   of which pure: {int(pure.sum())}"
          f"   (seed-invariant)")

    rows, tot = [], np.zeros(5, dtype=int)
    for s in sorted(runs):
        pred = reb[f"pred_seed{s}"].to_numpy()
        to_run = pred == RUNNING
        wrong = pred != REBOUND
        c = np.array([len(reb), to_run.sum(), (to_run & pure).sum(), wrong.sum(), (wrong & pure).sum()])
        tot += c
        rows.append((f"seed {s}", c))
    rows.append(("pooled", tot))

    print(f"\n  {'':<8}{'reb win':>9}{'->running':>11}{'%':>7}{'pure|run':>10}{'%':>7}"
          f"{'wrong':>8}{'pure|wrong':>12}{'%':>7}")
    for tag, (n, r, pr, w, pw) in rows:
        print(f"  {tag:<8}{n:>9}{r:>11}{100 * r / n:>7.1f}{pr:>10}{100 * pr / r:>7.1f}"
              f"{w:>8}{pw:>12}{100 * pw / w:>7.1f}")

    per = np.array([[100 * c[1] / c[0], 100 * c[2] / c[1], 100 * c[4] / c[3]] for _t, c in rows[:-1]])
    if len(per) > 1:
        m, sd = per.mean(axis=0), per.std(axis=0, ddof=1)
        print(f"\n  per-seed mean +/- std (ddof=1):  ->running {m[0]:.1f} +/- {sd[0]:.1f}%   "
              f"pure|running {m[1]:.1f} +/- {sd[1]:.1f}%   pure|wrong {m[2]:.1f} +/- {sd[2]:.1f}%")
    print("\n  ->running = rebound windows predicted running; pure|run = pure share of those "
          "(old 109/301 rule);\n  wrong = rebound windows predicted anything but rebound; "
          "pure|wrong = pure share of those (old 189/549 rule).")


if __name__ == "__main__":
    try:
        main()
    except wp.HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
