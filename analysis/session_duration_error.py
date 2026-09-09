#!/usr/bin/env python3
"""
Per-SESSION duration error: does the dense head estimate each recording session's
per-class time-on-task closer to truth than the windowed baseline does?

boracle_duration_bias.py answers the same question POOLED over folds, and pooling is
exactly what hides the failure mode that matters downstream. A pooled duration can be
close because one session's over-count cancels another's under-count; BOracle does not
consume the pool, it consumes a session. This script never pools: every number is one
session's estimate against that same session's truth.

WHAT A SESSION IS. `recording` from data/seam_map.json -- one participant's continuous
wristband stream within one capture event (subject x capture-day). 24 in the full
dataset, 10 under the 5-subject LOSO baseline. See _session_common's module docstring
for the timestamp evidence and for why the two capture days themselves are not the
unit. Note this contradicts boracle_duration_bias.py:63-70, which predates the seam
map and states no session identifier survives; the seam map recovers it.

PAIRING. Windowed and dense do not cover the same samples -- the windowed path leaves
up to `step` trailing samples per subject uncovered, the dense path drops segments
shorter than dense_min_seg. Every paired number here is computed on the INTERSECTION
of the two covered masks, and the two independently reconstructed truths are asserted
equal there first. Without that, dense would be credited for a difference that is
bookkeeping rather than accuracy.

WHAT IS REPORTED
  [1] inventory: per session, covered samples and the mask disagreement
  [2] per-session per-class relative error, windowed and dense side by side
  [3] absolute error in seconds, and MAE per session (across classes) and per class
      (across sessions)
  [4] the improvement table -- |rel_err_win| - |rel_err_dense| per class x session,
      positive meaning dense is closer to truth; win/loss counts per class
  [5] summary across sessions: mean and std of the per-session errors
  [6] dense-only panel over all sessions the dense run covers (n=24 for an all-14 run),
      which has no windowed counterpart and is reported separately rather than mixed in

ZERO-TRUTH CELLS. Relative error is undefined where a session has no samples of a
class -- `sitting` is absent from 6 of the 10 baseline sessions, `shot` from 2. Those
cells are carried as NaN and EXCLUDED from means, std and win/loss counts rather than
counted as agreement, following boracle_duration_bias.py's convention. Every table
prints the n it actually used.

Read-only and CPU-only. Writes nothing, loads no checkpoints, imports nothing from src/.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/session_duration_error.py \
        --windowed_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --dense_dir    logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-04-41_dense_b10_seed1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _session_common as sc        # noqa: E402

CLASS_NAMES = sc.CLASS_NAMES
N_CLASSES = sc.N_CLASSES
HardFail = sc.HardFail
fail = sc.fail
secs = sc.secs
BALL_ACTIONS = sc.BALL_ACTIONS


def nanmean_std(values):
    """(mean, std, n) over the finite entries only; (nan, nan, 0) if there are none."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=0)), int(v.size)


def fmt(x, width=9, prec=1, suffix=""):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>+{width}.{prec}f}{suffix}"


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_inventory(paired, win_meta, den_meta):
    print("=" * 118)
    print("SESSION DURATION ERROR -- per-recording per-class duration, windowed vs dense")
    print("=" * 118)
    print(f"\n  windowed run   : {win_meta['name']}   [{win_meta['results_dir']}]")
    print(f"                   sw_length={win_meta['sw_length']}s sw_overlap={win_meta['sw_overlap']}% "
          f"-> win_len={win_meta['win_len']} step={win_meta['step']} samples")
    print(f"  dense run      : {den_meta['name']}   [{den_meta['results_dir']}]")
    print(f"                   dense_seq_len={den_meta['seq_len']} "
          f"dense_min_seg={den_meta['min_segment_len']}")
    if den_meta["cfg"].get("no_bilstm"):
        print("                   NOTE: dense run is the --no_bilstm ablation.")
    print(f"  sampling rate  : {sc.SAMPLING_RATE} Hz (preprocess_data.py:31; never in cfg.txt)")
    print()
    sc.session_header(paired)

    print("\n" + "=" * 118)
    print("[1] SESSION INVENTORY -- the common covered mask every paired number below uses")
    print("=" * 118)
    print("\n  win_only  = samples the windowed path covered and dense did not")
    print("  den_only  = samples dense covered and the windowed path did not (its trailing tail)\n")
    hdr = (f"  {'session':<12} {'subject':<8} {'raw':>9} {'common':>9} {'win_only':>9} "
           f"{'den_only':>9} {'common_dur':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rec in sorted(paired):
        r = paired[rec]
        print(f"  {rec:<12} {r['subject']:<8} {r['n_raw']:>9} {r['n_common']:>9} "
              f"{r['n_win_only']:>9} {r['n_den_only']:>9} {sc.hms(secs(r['n_common'])):>12}")
    tot = sum(r["n_common"] for r in paired.values())
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<12} {'':<8} {sum(r['n_raw'] for r in paired.values()):>9} {tot:>9} "
          f"{sum(r['n_win_only'] for r in paired.values()):>9} "
          f"{sum(r['n_den_only'] for r in paired.values()):>9} {sc.hms(secs(tot)):>12}")


def report_relative_error(paired):
    """[2]: per-session per-class relative error, both paths."""
    recs = sorted(paired)
    print("\n" + "=" * 118)
    print("[2] PER-SESSION RELATIVE ERROR -- (est - true) / true * 100, per class")
    print("=" * 118)
    print("\n  W = windowed, D = dense. 'n/a' = this session has no samples of that class, so")
    print("  relative error is undefined; those cells are excluded everywhere, never zeroed.\n")

    for c, name in enumerate(CLASS_NAMES):
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name}{tag}")
        hdr = (f"    {'session':<12} {'true_s':>9} {'win_s':>9} {'den_s':>9} "
               f"{'W_rel%':>10} {'D_rel%':>10} {'closer':>8}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for rec in recs:
            r = paired[rec]
            t = secs(r["true"][c])
            w, d = secs(r["win_est"][c]), secs(r["den_est"][c])
            wr = sc.rel_err_pct(r["win_est"][c], r["true"][c])
            dr = sc.rel_err_pct(r["den_est"][c], r["true"][c])
            if np.isfinite(wr) and np.isfinite(dr):
                closer = "dense" if abs(dr) < abs(wr) else ("win" if abs(wr) < abs(dr) else "tie")
            else:
                closer = "n/a"
            print(f"    {rec:<12} {t:>9.2f} {w:>9.2f} {d:>9.2f} "
                  f"{fmt(wr, 10)} {fmt(dr, 10)} {closer:>8}")
        print()


def report_absolute_error(paired):
    """[3]: absolute error in seconds, plus both MAE margins."""
    recs = sorted(paired)
    print("=" * 118)
    print("[3] ABSOLUTE ERROR AND MAE -- |est - true| in seconds")
    print("=" * 118)
    print("\n  MAE_session = mean over the 9 classes of |est - true| for that session.")
    print("  MAE_class   = mean over sessions of |est - true| for that class.")
    print("  Seconds, not percent, so every cell is defined -- a zero-truth class contributes")
    print("  its absolute over-count here even though [2] cannot express it as a ratio.\n")

    win_abs = np.array([[abs(secs(paired[r]["win_est"][c]) - secs(paired[r]["true"][c]))
                         for r in recs] for c in range(N_CLASSES)])
    den_abs = np.array([[abs(secs(paired[r]["den_est"][c]) - secs(paired[r]["true"][c]))
                         for r in recs] for c in range(N_CLASSES)])

    hdr = (f"  {'session':<12} {'W_MAE_s':>10} {'D_MAE_s':>10} {'delta_s':>10} "
           f"{'W_total_s':>11} {'D_total_s':>11} {'better':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for j, rec in enumerate(recs):
        wm, dm = win_abs[:, j].mean(), den_abs[:, j].mean()
        print(f"  {rec:<12} {wm:>10.3f} {dm:>10.3f} {wm - dm:>+10.3f} "
              f"{win_abs[:, j].sum():>11.2f} {den_abs[:, j].sum():>11.2f} "
              f"{('dense' if dm < wm else 'win'):>8}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'MEAN':<12} {win_abs.mean():>10.3f} {den_abs.mean():>10.3f} "
          f"{win_abs.mean() - den_abs.mean():>+10.3f}")
    n_dense_better = int((den_abs.mean(axis=0) < win_abs.mean(axis=0)).sum())
    print(f"\n  Dense has the lower MAE in {n_dense_better} of {len(recs)} sessions.")

    print(f"\n  Per class, MAE across the {len(recs)} sessions:")
    hdr = (f"    {'class':<12} {'W_MAE_s':>10} {'D_MAE_s':>10} {'delta_s':>10} "
           f"{'W_worst_s':>11} {'D_worst_s':>11} {'better':>8}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for c, name in enumerate(CLASS_NAMES):
        wm, dm = win_abs[c].mean(), den_abs[c].mean()
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"    {name:<11}{tag} {wm:>10.3f} {dm:>10.3f} {wm - dm:>+10.3f} "
              f"{win_abs[c].max():>11.2f} {den_abs[c].max():>11.2f} "
              f"{('dense' if dm < wm else 'win'):>8}")


def report_improvement(paired):
    """[4]: the headline -- how much closer is dense, per class, broken out by session."""
    recs = sorted(paired)
    print("\n" + "=" * 118)
    print("[4] IMPROVEMENT -- |rel_err_windowed| - |rel_err_dense|, in percentage points")
    print("=" * 118)
    print("\n  POSITIVE = dense is closer to truth for that session and class.")
    print("  NEGATIVE = the windowed baseline is closer. 'n/a' = zero-truth cell, excluded.")
    print("  One column per session; 'wins' counts sessions where dense is strictly closer.\n")

    hdr = (f"  {'class':<12}" + "".join(f"{r.replace('_', ''):>10}" for r in recs)
           + f"{'mean':>10}{'median':>10}{'wins':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    per_class = {}
    misleading = []
    for c, name in enumerate(CLASS_NAMES):
        row = []
        for rec in recs:
            r = paired[rec]
            wr = sc.rel_err_pct(r["win_est"][c], r["true"][c])
            dr = sc.rel_err_pct(r["den_est"][c], r["true"][c])
            row.append(abs(wr) - abs(dr) if np.isfinite(wr) and np.isfinite(dr) else np.nan)
        row = np.array(row)
        per_class[name] = row
        usable = row[np.isfinite(row)]
        wins = int((usable > 0).sum())
        mean = usable.mean() if usable.size else np.nan
        med = np.median(usable) if usable.size else np.nan
        tag = "*" if name in BALL_ACTIONS else " "

        # The mean is misleading when it points the other way from the sessions
        # themselves: it disagrees in sign with the median, or it claims an
        # improvement a minority of sessions actually saw. Either way one or two
        # outlier sessions are carrying it.
        if usable.size and (np.sign(mean) != np.sign(med)
                            or (mean > 0 and wins * 2 < usable.size)
                            or (mean < 0 and wins * 2 > usable.size)):
            misleading.append((name, mean, med, wins, usable.size,
                               recs[int(np.nanargmin(np.where(np.isfinite(row), row, np.nan)))]))
            flag = " <-"
        else:
            flag = ""
        print(f"  {name:<11}{tag}"
              + "".join(fmt(v, 10) for v in row)
              + fmt(mean, 10) + fmt(med, 10) + f"{f'{wins}/{usable.size}':>9}{flag}")

    print("  " + "-" * (len(hdr) - 2))
    print("\n  Reading this table: a class whose row is uniformly positive is one where dense")
    print("  helps every session. A row that mixes large positives with large negatives is a")
    print("  class where dense trades a systematic error for a scattered one -- the mean can")
    print("  look like an improvement while no individual session became reliable.")

    if misleading:
        print("\n  '<-' MARKS CLASSES WHOSE MEAN CONTRADICTS THE SESSIONS. For these the mean and")
        print("  the median disagree in sign, or the mean claims a gain a minority of sessions")
        print("  saw. Quote the median and the win count for these, not the mean:")
        for name, mean, med, wins, n, worst in misleading:
            print(f"    {name:<12} mean {mean:>+8.1f} pp but median {med:>+8.1f} pp and "
                  f"{wins}/{n} sessions improved; worst session {worst}")
    return per_class


def report_summary(paired, improvement):
    """[5]: mean/std of the per-session errors, which is the stat the write-up quotes."""
    recs = sorted(paired)
    print("\n" + "=" * 118)
    print("[5] SUMMARY ACROSS SESSIONS -- mean and std of the per-session relative errors")
    print("=" * 118)
    print(f"\n  n = {len(recs)} sessions. std is the population std (ddof=0) over the sessions")
    print("  with defined relative error; n_used is printed per class because it varies.")
    print("  A LARGER std with a SMALLER mean is the signature of trading systematic bias for")
    print("  scatter -- which is worse, not better, for a per-session exposure estimate.\n")

    hdr = (f"  {'class':<12} {'W_mean%':>10} {'W_std%':>9} {'D_mean%':>10} {'D_std%':>9} "
           f"{'|mean| gain':>12} {'std gain':>10} {'n_used':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c, name in enumerate(CLASS_NAMES):
        wr = np.array([sc.rel_err_pct(paired[r]["win_est"][c], paired[r]["true"][c]) for r in recs])
        dr = np.array([sc.rel_err_pct(paired[r]["den_est"][c], paired[r]["true"][c]) for r in recs])
        both = np.isfinite(wr) & np.isfinite(dr)
        wm, ws, n = nanmean_std(wr[both])
        dm, ds, _ = nanmean_std(dr[both])
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {fmt(wm, 10)} {ws:>9.1f} {fmt(dm, 10)} {ds:>9.1f} "
              f"{fmt(abs(wm) - abs(dm), 12)} {fmt(ws - ds, 10)} {n:>7}")
    print("  " + "-" * (len(hdr) - 2))
    print("\n  '|mean| gain' > 0: dense has the smaller mean bias.  'std gain' > 0: dense has the")
    print("  smaller spread. A class positive on the first and negative on the second has moved")
    print("  its error from bias into variance.")

    print("\n  Improvement (|rel_err_win| - |rel_err_dense|) summarised across sessions:")
    hdr = (f"    {'class':<12} {'mean_pp':>10} {'std_pp':>9} {'min_pp':>10} {'max_pp':>10} {'n':>5}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in CLASS_NAMES:
        row = improvement[name]
        m, s, n = nanmean_std(row)
        u = row[np.isfinite(row)]
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"    {name:<11}{tag} {fmt(m, 10)} {s if np.isfinite(s) else float('nan'):>9.1f} "
              f"{fmt(u.min() if u.size else np.nan, 10)} {fmt(u.max() if u.size else np.nan, 10)} "
              f"{n:>5}")


def report_dense_only(den_sessions, paired):
    """[6]: the sessions dense covers that have no windowed counterpart."""
    extra = sorted(set(den_sessions) - set(paired))
    print("\n" + "=" * 118)
    print("[6] DENSE-ONLY PANEL -- sessions with no windowed counterpart")
    print("=" * 118)
    if not extra:
        print(f"\n  None. The dense run covers the same {len(paired)} sessions as the windowed run,")
        print("  so every session is already scored in the paired tables above.")
        return
    print(f"\n  The dense run covers {len(den_sessions)} sessions, the windowed run "
          f"{len(paired)}. No windowed")
    print(f"  all-subject run exists in logs/, so these {len(extra)} sessions cannot be paired.")
    print("  They are scored on the dense path's own covered mask and are NOT mixed into any")
    print("  table above -- comparing a 24-session dense mean against a 10-session windowed one")
    print("  would confound the method with the session set.\n")

    print("  'pooled%' is computed on the summed durations; 'mean_rel%' averages the per-session")
    print("  relative errors. They differ whenever the error is not proportional to session size,")
    print("  and a large gap between the two is itself the finding: it means the classes' error")
    print("  is concentrated in the sessions that hold least of it.\n")

    hdr = (f"  {'class':<12} {'true_s':>10} {'den_s':>10} {'pooled%':>9} {'mean_rel%':>10} "
           f"{'std%':>9} {'MAE_s':>9} {'n_used':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c, name in enumerate(CLASS_NAMES):
        t = np.array([den_sessions[r]["true"][c] for r in extra])
        e = np.array([den_sessions[r]["est"][c] for r in extra])
        pooled = sc.rel_err_pct(e.sum(), t.sum())
        m, s, n = nanmean_std(sc.rel_err_pct(e, t))
        mae = float(np.mean(np.abs(secs(e) - secs(t))))
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {secs(t).sum():>10.2f} {secs(e).sum():>10.2f} "
              f"{fmt(float(pooled), 9)} {fmt(m, 10)} {s:>9.1f} {mae:>9.3f} {n:>7}")
    print(f"\n  ({len(extra)} dense-only sessions: {', '.join(extra)})")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windowed_dir", required=True, metavar="DIR",
                        help="Windowed baseline run dir (cfg.txt must NOT have dense=true).")
    parser.add_argument("--dense_dir", required=True, metavar="DIR",
                        help="Dense run dir (cfg.txt must have dense=true).")
    parser.add_argument("--windowed_npz_pattern", default=None,
                        help="Glob containing '{fold}' for the windowed run, if ambiguous.")
    parser.add_argument("--dense_npz_pattern", default=None,
                        help="Glob containing '{fold}' for the dense run, if ambiguous.")
    sc.add_session_args(parser)
    args = parser.parse_args()

    labels_df, segments = sc.load_common(args)

    win_arrays, win_meta = sc.load_path(args.windowed_dir, labels_df, segments, args.labels,
                                        args.seam_map, dense=False,
                                        npz_pattern=args.windowed_npz_pattern,
                                        allow_partial=args.allow_partial_folds)
    den_arrays, den_meta = sc.load_path(args.dense_dir, labels_df, segments, args.labels,
                                        args.seam_map, dense=True,
                                        npz_pattern=args.dense_npz_pattern,
                                        allow_partial=args.allow_partial_folds)

    if not set(win_arrays) <= set(den_arrays):
        missing = sorted(set(win_arrays) - set(den_arrays))
        fail(f"the windowed run covers subject(s) {missing} that the dense run does not. "
             "Every windowed subject must have a dense counterpart to be paired.")

    paired = sc.paired_sessions(win_arrays, den_arrays, segments)
    sc.check_session_count(paired, args.allow_partial_folds, "paired: ")
    den_sessions = sc.sessions_one_path(den_arrays, segments)

    report_inventory(paired, win_meta, den_meta)
    report_relative_error(paired)
    report_absolute_error(paired)
    improvement = report_improvement(paired)
    report_summary(paired, improvement)
    report_dense_only(den_sessions, paired)

    print("\n" + "=" * 118)
    print("Read [4] for the per-class per-session improvement, [5] for whether that improvement")
    print("is a smaller bias or merely a smaller mean over a wider spread. Nothing was written;")
    print("this script is read-only.")
    print("=" * 118)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
