#!/usr/bin/env python3
"""
Do dense predictions preserve the TRUE RANKING of sessions by activity duration better
than windowed predictions do?

THE CLAIM UNDER TEST. A downstream exposure model does not always need the absolute
seconds right; often it needs the ORDER right -- which session carried the most
rebounding, which the least. A biased-but-monotone estimator is usable (one global
correction factor fixes it, see analysis/calibration_factors.py); a scattered estimator
is not, however small its mean error. So: rank the sessions by predicted duration under
each method, rank them by ground truth, and correlate.

    rho_windowed = spearman(windowed_pred_duration, true_duration)  over sessions
    rho_dense    = spearman(dense_pred_duration,    true_duration)  over sessions

The claim is rho_dense > rho_windowed. This script MEASURES that; it does not assume
it. Where the data contradicts the claim the table says so, and the summary names the
classes where it fails.

WHAT A SESSION IS. `recording` from data/seam_map.json -- one participant's continuous
wristband stream within one capture event (subject x capture-day). See
_session_common's module docstring for the timestamp evidence. Rebound is the headline
class but all nine are computed; a claim tested on one class chosen after the fact is
not a test.

PAIRING. Windowed and dense do not cover the same samples, so every paired number is
computed on the INTERSECTION of the two covered masks, against one truth asserted equal
across both reconstructions. See _session_common.

n IS SMALL AND THE SCRIPT SAYS SO. The 5-subject LOSO baseline gives n=10 sessions; a
Spearman rho on 10 points has a wide confidence interval and its p-value is not a
licence to call a difference real. Two rho values from the same 10 sessions are
correlated with each other, so their DIFFERENCE has an even wider interval than either
alone -- this script does not test that difference for significance, because with n=10
no such test would be informative. It reports both rho values, both p-values, and the
n actually used, and leaves the inference explicit rather than implied.

DEGENERATE CLASSES ARE FLAGGED, NOT HIDDEN. `sitting` is absent from 6 of the 10
baseline sessions, so its truth vector is four values plus a six-way tie at zero. A
Spearman over that is not interpretable as a ranking, and midrank tie handling will
still return a confident-looking rho. Any class whose truth has fewer than
MIN_DISTINCT_TRUTH distinct values, or fewer than MIN_NONZERO_SESSIONS non-zero
sessions, is marked DEGENERATE and excluded from the summary verdict.

Read-only and CPU-only. Writes nothing, loads no checkpoints, imports nothing from src/.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/session_rank_stability.py \
        --windowed_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --dense_dir    logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-04-41_dense_b10_seed1
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _session_common as sc        # noqa: E402

CLASS_NAMES = sc.CLASS_NAMES
N_CLASSES = sc.N_CLASSES
HardFail = sc.HardFail
fail = sc.fail
secs = sc.secs
BALL_ACTIONS = sc.BALL_ACTIONS

HEADLINE_CLASS = "rebound"

# A truth vector with fewer distinct values than this is a tie structure, not a ranking.
MIN_DISTINCT_TRUTH = 5
MIN_NONZERO_SESSIONS = 5

# Reported as a reference line only. n=10 is small; this is not a decision threshold.
ALPHA = 0.05


def ranks(values):
    """Competition ranks, 1 = largest. Ties share the smaller rank, as in
    subject_ranking_persistence.difficulty_rank(method='min')."""
    v = np.asarray(values, float)
    order = (-v).argsort(kind="stable")
    out = np.empty(len(v), float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        out[order[i:j + 1]] = i + 1
        i = j + 1
    return out


def safe_spearman(a, b):
    """(rho, p) with NaN rather than a warning when either side is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, np.nan
    res = spearmanr(a, b)
    return float(res.statistic), float(res.pvalue)


def degeneracy(truth):
    """('' | reason) for why this class's truth cannot carry a ranking."""
    t = np.asarray(truth, float)
    n_distinct = len(np.unique(t))
    n_nonzero = int((t > 0).sum())
    if n_nonzero < MIN_NONZERO_SESSIONS:
        return f"only {n_nonzero} session(s) have any of this class"
    if n_distinct < MIN_DISTINCT_TRUTH:
        return f"truth takes only {n_distinct} distinct value(s)"
    return ""


def fmt_rho(x, width=8):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>+{width}.3f}"


def fmt_p(x, width=8):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>{width}.4f}"


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #

def rank_table(sessions, est_keys, truth_key="true"):
    """
    class -> {"truth": vec, key: vec, ...} of per-session durations in SAMPLES,
    session order fixed and returned alongside.
    """
    recs = sorted(sessions)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        entry = {"truth": np.array([sessions[r][truth_key][c] for r in recs], float)}
        for k in est_keys:
            entry[k] = np.array([sessions[r][k][c] for r in recs], float)
        out[name] = entry
    return recs, out


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_header(paired, win_meta, den_meta):
    print("=" * 112)
    print("SESSION RANK STABILITY -- does dense preserve the true ordering of sessions?")
    print("=" * 112)
    print(f"\n  windowed run   : {win_meta['name']}   [{win_meta['results_dir']}]")
    print(f"                   sw_length={win_meta['sw_length']}s sw_overlap={win_meta['sw_overlap']}% "
          f"-> win_len={win_meta['win_len']} step={win_meta['step']} samples")
    print(f"  dense run      : {den_meta['name']}   [{den_meta['results_dir']}]")
    print(f"                   dense_seq_len={den_meta['seq_len']} "
          f"dense_min_seg={den_meta['min_segment_len']}")
    if den_meta["cfg"].get("no_bilstm"):
        print("                   NOTE: dense run is the --no_bilstm ablation.")
    print()
    sc.session_header(paired)
    print(f"\n  n = {len(paired)} sessions. Every duration below is scored on the common covered")
    print("  mask of the two runs, against one truth asserted equal across both reconstructions.")


def report_headline_ranking(recs, table, cls):
    """The full ranking, spelled out for the headline class."""
    print("\n" + "=" * 112)
    print(f"[1] THE RANKING, SPELLED OUT -- {cls} (the headline class)")
    print("=" * 112)
    print("\n  rank 1 = longest predicted/true duration. This is the table the correlations")
    print("  below compress into two numbers; read it first so those numbers mean something.\n")

    e = table[cls]
    t, w, d = e["truth"], e["win_est"], e["den_est"]
    rt, rw, rd = ranks(t), ranks(w), ranks(d)

    hdr = (f"  {'session':<12} {'true_s':>9} {'rank':>5}  {'win_s':>9} {'rank':>5} {'d':>5}  "
           f"{'den_s':>9} {'rank':>5} {'d':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for j in np.argsort(rt):
        print(f"  {recs[j]:<12} {secs(t[j]):>9.2f} {int(rt[j]):>5}  "
              f"{secs(w[j]):>9.2f} {int(rw[j]):>5} {int(rw[j] - rt[j]):>+5}  "
              f"{secs(d[j]):>9.2f} {int(rd[j]):>5} {int(rd[j] - rt[j]):>+5}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'mean |rank move|':<12} {'':>9} {'':>5}  {'':>9} {'':>5} "
          f"{np.abs(rw - rt).mean():>5.2f}  {'':>9} {'':>5} {np.abs(rd - rt).mean():>5.2f}")
    print(f"  {'max  |rank move|':<12} {'':>9} {'':>5}  {'':>9} {'':>5} "
          f"{np.abs(rw - rt).max():>5.0f}  {'':>9} {'':>5} {np.abs(rd - rt).max():>5.0f}")
    print("\n  'd' is signed rank displacement from truth: negative = ranked too high (the")
    print("  method thinks this session had more of the class than it did).")


def report_spearman(recs, table, n, headline):
    """[2]: the headline table -- per-class rho and p, both methods."""
    print("\n" + "=" * 112)
    print("[2] SPEARMAN RANK CORRELATION WITH TRUTH -- all nine classes")
    print("=" * 112)
    print(f"\n  rho close to +1 means the method orders the {n} sessions the way truth does.")
    print("  'gain' = rho_dense - rho_windowed; POSITIVE supports the claim that dense")
    print("  preserves the true ranking better. p-values are two-sided and uncorrected;")
    print(f"  with n={n} they are weak evidence, and the DIFFERENCE of two rho values from the")
    print("  same sessions is not tested here because no such test is informative at this n.\n")

    hdr = (f"  {'class':<12} {'win_rho':>9} {'win_p':>9} {'den_rho':>9} {'den_p':>9} "
           f"{'gain':>9} {'supports':>10} {'nonzero':>8}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) + 18))

    rows = {}
    for name in CLASS_NAMES:
        e = table[name]
        t = e["truth"]
        rw, pw = safe_spearman(e["win_est"], t)
        rd, pd = safe_spearman(e["den_est"], t)
        deg = degeneracy(t)
        gain = rd - rw if np.isfinite(rd) and np.isfinite(rw) else np.nan
        if deg:
            supports = "DEGEN"
        elif not np.isfinite(gain):
            supports = "n/a"
        else:
            supports = "yes" if gain > 0 else ("no" if gain < 0 else "tie")
        rows[name] = {"win_rho": rw, "win_p": pw, "den_rho": rd, "den_p": pd,
                      "gain": gain, "degenerate": deg, "supports": supports}
        tag = "*" if name in BALL_ACTIONS else " "
        mark = "  <<< headline" if name == headline else ""
        print(f"  {name:<11}{tag} {fmt_rho(rw, 9)} {fmt_p(pw, 9)} {fmt_rho(rd, 9)} "
              f"{fmt_p(pd, 9)} {fmt_rho(gain, 9)} {supports:>10} "
              f"{int((t > 0).sum()):>8}  {deg}{mark}")
    print("  " + "-" * (len(hdr) + 18))
    print(f"\n  '*' = ball action.  'nonzero' = sessions with any of that class (of {n}).")
    print("  DEGEN = the truth vector cannot carry a ranking; see [3].")
    return rows


def report_degenerate(recs, table, rows):
    """[3]: why the flagged classes are excluded, with the numbers that make the case."""
    deg = {n: r for n, r in rows.items() if r["degenerate"]}
    print("\n" + "=" * 112)
    print("[3] DEGENERATE CLASSES -- excluded from the verdict")
    print("=" * 112)
    if not deg:
        print(f"\n  None. Every class's truth has at least {MIN_NONZERO_SESSIONS} non-zero sessions")
        print(f"  and at least {MIN_DISTINCT_TRUTH} distinct values, so every rho in [2] is a "
              "ranking.")
        return
    print("\n  A Spearman rho is only a statement about ordering if the truth actually orders the")
    print("  sessions. These classes do not, and scipy will still return a confident-looking rho")
    print("  from its midrank tie handling. They are reported in [2] and ignored in [4].\n")
    for name, r in deg.items():
        t = table[name]["truth"]
        print(f"  {name:<12} {r['degenerate']}")
        print(f"  {'':<12} truth per session (s): "
              + ", ".join(f"{recs[j]}={secs(t[j]):.1f}" for j in np.argsort(-t)))
        print(f"  {'':<12} rho would read win={fmt_rho(r['win_rho']).strip()} "
              f"den={fmt_rho(r['den_rho']).strip()} -- do not quote these.\n")


def report_verdict(rows, n, headline):
    """[4]: the answer to the question the script was written to ask."""
    usable = {k: v for k, v in rows.items()
              if not v["degenerate"] and np.isfinite(v["gain"])}
    print("=" * 112)
    print("[4] VERDICT -- does dense preserve the true session ranking better?")
    print("=" * 112)

    if not usable:
        print("\n  No class has an interpretable ranking; the question cannot be answered from")
        print("  this run pair.")
        return

    better = sorted([k for k, v in usable.items() if v["gain"] > 0],
                    key=lambda k: -usable[k]["gain"])
    worse = sorted([k for k, v in usable.items() if v["gain"] < 0],
                   key=lambda k: usable[k]["gain"])

    print(f"\n  Of {len(usable)} interpretable classes, dense ranks sessions better in "
          f"{len(better)} and worse in {len(worse)}.\n")
    if better:
        print("    dense BETTER : " + ", ".join(f"{k} ({usable[k]['gain']:+.3f})" for k in better))
    if worse:
        print("    dense WORSE  : " + ", ".join(f"{k} ({usable[k]['gain']:+.3f})" for k in worse))

    h = rows[headline]
    print(f"\n  HEADLINE CLASS -- {headline}:")
    if h["degenerate"]:
        print(f"    DEGENERATE ({h['degenerate']}); no ranking claim can be made.")
        return
    print(f"    windowed  rho = {h['win_rho']:+.3f}  (p = {h['win_p']:.4f})")
    print(f"    dense     rho = {h['den_rho']:+.3f}  (p = {h['den_p']:.4f})")
    print(f"    gain          = {h['gain']:+.3f}")

    sig_w = np.isfinite(h["win_p"]) and h["win_p"] < ALPHA
    sig_d = np.isfinite(h["den_p"]) and h["den_p"] < ALPHA
    if h["gain"] > 0:
        print(f"\n    The claim HOLDS for {headline}: dense tracks the true ordering more")
        print("    closely than the windowed baseline.")
    else:
        print(f"\n    The claim DOES NOT HOLD for {headline}: the windowed baseline tracks")
        print("    the true ordering more closely than dense does.")

    if not (sig_w or sig_d):
        print(f"    NEITHER rho reaches p < {ALPHA} at n={n}, so the honest reading is 'no evidence")
        print("    either way for this class', not a claim in the opposite direction. Report the")
        print("    direction with the n and the p-values attached, or not at all.")
    elif sig_w and not sig_d:
        print(f"    Only the WINDOWED rho reaches p < {ALPHA}; the dense ordering is consistent")
        print("    with chance at this n.")
    elif sig_d and not sig_w:
        print(f"    Only the DENSE rho reaches p < {ALPHA}; the windowed ordering is consistent")
        print("    with chance at this n.")
    else:
        print(f"    Both rho values reach p < {ALPHA}; the gap between them is still not tested,")
        print("    for the reason given in [2].")

    print("\n  Note what a rank result can and cannot say. Rank correlation is invariant to any")
    print("  monotone rescaling, so a method can rank perfectly while being badly biased in")
    print("  seconds -- that is precisely the case a single correction factor repairs")
    print("  (analysis/calibration_factors.py). A method that ranks poorly cannot be repaired")
    print("  that way, whatever its mean error. Read this table against [5] of")
    print("  analysis/session_duration_error.py, not instead of it.")


def report_dense_only(den_sessions, paired, n_paired, headline):
    """[5]: dense-only Spearman over every session the dense run covers."""
    extra = sorted(set(den_sessions) - set(paired))
    print("\n" + "=" * 112)
    print("[5] DENSE-ONLY PANEL -- rho over every session the dense run covers")
    print("=" * 112)
    if not extra:
        print(f"\n  None. The dense run covers the same {n_paired} sessions as the windowed run,")
        print("  so [2] already uses every session available.")
        return

    recs = sorted(den_sessions)
    print(f"\n  The dense run covers {len(recs)} sessions against the windowed run's {n_paired}.")
    print("  No windowed all-subject run exists in logs/, so there is no counterpart column here.")
    print(f"  This panel is dense-against-truth ONLY, at n={len(recs)}; it is not comparable to")
    print("  the paired rho in [2], which is computed on a different and smaller session set.\n")

    hdr = (f"  {'class':<12} {'den_rho':>9} {'den_p':>9} {'nonzero':>8} {'distinct':>9}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) + 18))
    for c, name in enumerate(CLASS_NAMES):
        t = np.array([den_sessions[r]["true"][c] for r in recs], float)
        e = np.array([den_sessions[r]["est"][c] for r in recs], float)
        rho, p = safe_spearman(e, t)
        deg = degeneracy(t)
        tag = "*" if name in BALL_ACTIONS else " "
        mark = "  <<< headline" if name == headline else ""
        print(f"  {name:<11}{tag} {fmt_rho(rho, 9)} {fmt_p(p, 9)} {int((t > 0).sum()):>8} "
              f"{len(np.unique(t)):>9}  {deg}{mark}")
    print(f"\n  ({len(recs)} sessions, of which {len(extra)} have no windowed counterpart.)")


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
    parser.add_argument("--headline_class", default=HEADLINE_CLASS,
                        help=f"Class spelled out in [1] and [4] (default: {HEADLINE_CLASS}).")
    sc.add_session_args(parser)
    args = parser.parse_args()

    headline = args.headline_class
    if headline not in CLASS_NAMES:
        fail(f"--headline_class {headline!r} is not a known class; known are {CLASS_NAMES}")

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
    n = sc.check_session_count(paired, args.allow_partial_folds, "paired: ")
    if n < 3:
        fail(f"a Spearman correlation needs at least 3 sessions; this run pair gives {n}.")
    den_sessions = sc.sessions_one_path(den_arrays, segments)

    recs, table = rank_table(paired, ("win_est", "den_est"))

    report_header(paired, win_meta, den_meta)
    report_headline_ranking(recs, table, headline)
    rows = report_spearman(recs, table, n, headline)
    report_degenerate(recs, table, rows)
    report_verdict(rows, n, headline)
    report_dense_only(den_sessions, paired, n, headline)

    print("\n" + "=" * 112)
    print("Read [2] for the per-class correlations, [4] for whether the claim survives them.")
    print("Nothing was written; this script is read-only.")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
