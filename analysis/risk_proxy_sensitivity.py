#!/usr/bin/env python3
"""
Does per-class duration bias propagate into INJURY-RISK STRATIFICATION, and does the
choice of HAR method (windowed vs dense) change the answer?

Pre-registered at docs/prereg_risk_proxy.md. This script implements that document and
nothing else; where it departs from it, the departure is printed as an AMENDMENT line
rather than made silently.

WHAT THE RISK PROXY IS, AND WHAT IT IS NOT. The score below is ILLUSTRATIVE. It does
not predict injuries, no injury labels exist in this dataset, and BOracle does not yet
exist as code. The proxy is a sensitivity vehicle: a monotone function of per-class
time-on-task, built so that "which activities carry risk" is a free parameter. The only
claim it can support is that classifier duration bias moves the output of such a
function, and by how much under each plausible weighting.

    R_c(s) = sum_k  w_{tier(k)} * f_{c,k}(s)

f is a FRACTION of session time, not seconds, so session length divides out and a long
session cannot outrank a short one merely by being long. Because f is a fraction, the
sample/second distinction is irrelevant here -- the 50 Hz conversion cancels -- and the
counts coming out of _session_common are used as-is.

TIERS ARE ORDINAL AND THE ORDERING IS THE HYPOTHESIS. Every weight vector satisfies
w3 > w2 > w1 >= 0, and that is asserted at import time, before anything is loaded: a
vector that violated it would not be a landing-outweighs-locomotion-outweighs-rest
weighting at all, and the robustness claim in section 6 of the pre-reg is quantified
over exactly this family. TIER_MAP is asserted to cover CLASS_NAMES exactly, so a
misspelled class cannot silently drop out of every score.

CALIBRATION IS FITTED PER CONDITION, NEVER SHARED. The pre-reg names one ship-ready set
(pass, walking, standing) but the correction FACTORS differ sharply between the two
prediction paths -- standing is k=0.36 windowed and k=0.78 dense on these runs -- so a
shared table would correct one condition with the other's bias. Each condition gets its
own factors, and they are estimated LEAVE-ONE-SUBJECT-OUT (the pooling rule reused
wholesale from calibration_factors.loo_factors: ratio of sums over the held-in folds,
never the mean of per-fold ratios). A factor fitted on the session it corrects would
remove the bias by construction and prove nothing. The per-condition stability verdict
is printed alongside, so a factor the sibling script would refuse to ship is visible
rather than assumed.

CORRECTION BREAKS THE PARTITION, AND THE SCORE NEEDS IT BACK. Scaling three classes by
three different constants destroys the property that the nine durations sum to the
session -- calibration_factors.py [3] measures that drift and leaves the resolution to
the consumer. This script IS that consumer, and a weighted sum of fractions is only
comparable across conditions if the fractions sum to one, so the corrected counts are
RENORMALIZED. That reintroduces some of the bias the correction removed. The drift
absorbed by renormalization is reported in [7b] rather than hidden, because it bounds
how much of the calibrated result is the correction and how much is the renormalizer.

n IS SMALL. The paired panel is 10 sessions and the dense-only panel is 24. Spearman
p-values at n=10 are weak evidence, tertiles hold 4/3/3 sessions, and the DIFFERENCE
between two rho values from the same sessions is not tested -- no such test is
informative here. Five weight vectors over the same 10 sessions are five views of one
dataset, not five independent experiments; the pre-reg's "holds under all five" bar is
a robustness check against weighting choice, not a multiple-comparison correction.

ANCHORS. The two panels reproduce rebound's duration-rank correlations from
analysis/session_rank_stability.py on the same run pair, and a mismatch is a hard fail:
the risk scores are built from the same per-session counts, so if those correlations
have moved, the loading path has changed underneath and no score below is comparable to
anything previously reported.

Read-only and CPU-only. Writes nothing, loads no checkpoints, imports nothing from src/.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/risk_proxy_sensitivity.py \
        --windowed_dir  logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --dense_dir     logs/subset_specific/loso_G/inceptioncontext/2026-09-04_16-04-41_dense_b10_seed1 \
        --dense_all_dir logs/subset_specific/loso_G/inceptioncontext/2026-09-04_17-48-10_dense_b10_all14_seed1 \
        --calibrate
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _session_common as sc            # noqa: E402
import calibration_factors as cf        # noqa: E402  -- LOO pooling rule, reused wholesale

CLASS_NAMES = sc.CLASS_NAMES
N_CLASSES = sc.N_CLASSES
HardFail = sc.HardFail
fail = sc.fail
secs = sc.secs
BALL_ACTIONS = sc.BALL_ACTIONS

RULE = "=" * 112

# --------------------------------------------------------------------------- #
# pre-registered configuration -- docs/prereg_risk_proxy.md section 2
# --------------------------------------------------------------------------- #

TIER_MAP = {
    "rebound": 3, "layup": 3, "shot": 3,
    "running": 2, "dribbling": 2, "pass": 2,
    "walking": 1, "standing": 1, "sitting": 1,
}

WEIGHT_VECTORS = {
    "A_linear":     {3: 3,  2: 2, 1: 1},
    "B_moderate":   {3: 5,  2: 2, 1: 1},
    "C_strong":     {3: 10, 2: 3, 1: 1},
    "D_compressed": {3: 5,  2: 3, 1: 1},
    "E_zero_rest":  {3: 3,  2: 2, 1: 0},
}

TIER_LABEL = {3: "landing/explosive", 2: "locomotion/load", 1: "rest/low"}

# Pre-reg section 3: the ship-ready LOO corrections, applied to these classes only.
# The FACTORS are recomputed per condition (see the module docstring); only the class
# SET is taken from the pre-registration.
CALIBRATED_CLASSES = ("pass", "walking", "standing")

# Recorded anchors, verbatim from analysis/session_rank_stability.py run on the run
# pair in this script's usage block. These are duration-rank correlations for rebound,
# not risk scores -- they certify the per-session counts the scores are built from.
ANCHOR_PAIRED_REBOUND = {"win_rho": 0.553, "den_rho": 0.115}   # n=10, windowed + dense_b10_seed1
ANCHOR_DENSE_ALL_REBOUND = {"rho": 0.544, "p": 0.0059}         # n=24, dense_b10_all14_seed1
ANCHOR_RHO_TOL = 1e-3
ANCHOR_P_TOL = 5e-4

# Tertile count is fixed by the pre-reg's "low / medium / high".
N_TERTILES = 3


# --------------------------------------------------------------------------- #
# configuration checks -- run before anything is loaded
# --------------------------------------------------------------------------- #

def check_tier_map():
    """
    TIER_MAP must cover CLASS_NAMES exactly, with tiers drawn from {1, 2, 3}.

    A class missing from the map would contribute nothing to every score under every
    weight vector, and the totals would still look plausible -- so this is asserted
    rather than left to a KeyError that a .get() default would swallow.
    """
    mapped, known = set(TIER_MAP), set(CLASS_NAMES)
    if mapped != known:
        missing, extra = sorted(known - mapped), sorted(mapped - known)
        fail(f"TIER_MAP does not cover the class list. missing={missing} unknown={extra}. "
             f"Known classes are {CLASS_NAMES}.")
    bad = {k: v for k, v in TIER_MAP.items() if v not in (1, 2, 3)}
    if bad:
        fail(f"TIER_MAP assigns tiers outside {{1, 2, 3}}: {bad}.")


def check_weight_vectors():
    """
    Every weight vector must satisfy the strict tier ordering w3 > w2 > w1 >= 0.

    This is the family the pre-reg's robustness claim is quantified over. A vector
    outside it is not a weaker version of the hypothesis, it is a different one, and
    reporting it under the same banner would misstate what was tested.
    """
    if not WEIGHT_VECTORS:
        fail("WEIGHT_VECTORS is empty; there is nothing to test robustness over.")
    for label, wv in WEIGHT_VECTORS.items():
        if set(wv) != {1, 2, 3}:
            fail(f"weight vector {label} defines tiers {sorted(wv)}, expected exactly "
                 "{1, 2, 3}.")
        w1, w2, w3 = float(wv[1]), float(wv[2]), float(wv[3])
        if not np.isfinite([w1, w2, w3]).all():
            fail(f"weight vector {label} has a non-finite weight: {wv}.")
        if not (w3 > w2 > w1 >= 0):
            fail(f"weight vector {label} violates w3 > w2 > w1 >= 0: "
                 f"w3={w3:g}, w2={w2:g}, w1={w1:g}. Every vector in WEIGHT_VECTORS must "
                 "preserve the strict tier ordering (docs/prereg_risk_proxy.md 2.3).")


def class_weights(wv):
    """Tier weights -> one weight per class, in CLASS_NAMES order."""
    return np.array([float(wv[TIER_MAP[name]]) for name in CLASS_NAMES], dtype=float)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def fractions(counts):
    """
    Per-class counts -> fraction of session time. Hard-fails on an empty session.

    A session with no covered samples cannot be scored at all, and _session_common
    already refuses to build one, so reaching here means the mask changed underneath.
    """
    v = np.asarray(counts, dtype=float)
    total = v.sum()
    if not np.isfinite(total) or total <= 0:
        fail(f"a session's class counts sum to {total}; a fraction is undefined. "
             "The covered mask is empty or non-finite.")
    return v / total


def score_sessions(sessions, key, w, recs=None):
    """(recs, scores) -- the risk score of every session under one weight vector."""
    recs = sorted(sessions) if recs is None else recs
    return recs, np.array([float(np.dot(w, fractions(sessions[r][key]))) for r in recs])


def counts_matrix(sessions, key, recs):
    """recs x N_CLASSES of raw per-class counts, session order fixed by `recs`."""
    return np.array([np.asarray(sessions[r][key], dtype=float) for r in recs])


def safe_spearman(a, b):
    """(rho, p), NaN rather than a warning when either side is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, np.nan
    res = spearmanr(a, b)
    return float(res.statistic), float(res.pvalue)


def tertiles(scores):
    """
    Session -> tertile 1 (highest risk) .. 3 (lowest), assigned by descending score.

    Sizes come from np.array_split, so n=10 gives 4/3/3 and n=24 gives 8/8/8 -- fixed
    and deterministic, as the pre-reg requires. Returns the boundary ties alongside:
    two sessions with identical scores either side of a cut are assigned to different
    tertiles by sort order alone, which is arbitrary, so it is reported rather than
    resolved.
    """
    s = np.asarray(scores, float)
    order = np.argsort(-s, kind="stable")
    out = np.empty(len(s), dtype=int)
    ties, cursor = [], 0
    for t, chunk in enumerate(np.array_split(order, N_TERTILES), start=1):
        out[chunk] = t
        cursor += len(chunk)
        if 0 < cursor < len(order) and s[order[cursor - 1]] == s[order[cursor]]:
            ties.append((t, t + 1, float(s[order[cursor]])))
    return out, ties


def masd(est, truth):
    """Mean absolute score deviation -- pre-reg 5.1(c)."""
    return float(np.mean(np.abs(np.asarray(est, float) - np.asarray(truth, float))))


def mrsd(est, truth):
    """
    Mean relative score deviation -- pre-reg 5.1(d). Returns (value, n_used, n_skipped).

    Under E_zero_rest a session spent entirely in rest classes scores R_truth = 0 and
    the ratio is undefined. Following sc.rel_err_pct, that is NaN and excluded, never a
    silent inf -- and the excluded count is returned so the mean is read against the n
    it was actually computed on.
    """
    e, t = np.asarray(est, float), np.asarray(truth, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(t > 0, np.abs(e - t) / t, np.nan)
    ok = np.isfinite(ratio)
    return (float(ratio[ok].mean()) if ok.any() else np.nan), int(ok.sum()), int((~ok).sum())


def fmt_rho(x, width=8):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>+{width}.3f}"


def fmt_p(x, width=8):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>{width}.4f}"


def fmt(x, width=9, prec=4):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>{width}.{prec}f}"


# --------------------------------------------------------------------------- #
# calibration -- per condition, leave-one-subject-out
# --------------------------------------------------------------------------- #

def subject_counts(sessions, est_key):
    """
    subject -> {"true", "pred"} summed over that subject's sessions, in SAMPLES.

    Shaped for cf.loo_factors, which needs exactly these two keys. Built from the
    sessions themselves rather than from cf.fold_counts so that the factors are fitted
    on the SAME mask they will correct -- cf.fold_counts uses each path's own covered
    mask, while a paired session is scored on the intersection of the two.
    """
    out = {}
    for rec in sorted(sessions):
        r = sessions[rec]
        acc = out.setdefault(r["subject"], {"true": np.zeros(N_CLASSES, np.int64),
                                            "pred": np.zeros(N_CLASSES, np.int64)})
        acc["true"] += np.asarray(r["true"], np.int64)
        acc["pred"] += np.asarray(r[est_key], np.int64)
    return out


def condition_factors(sessions, est_key):
    """
    (loo, k_all, stability) for one condition.

    loo[subject] is the per-class factor estimated on every OTHER subject, so a session
    is corrected by a factor that never saw it. k_all and the stability verdict are the
    all-subject factor and cf's own STABLE/loose/UNSTABLE reading of the LOO spread,
    reported so a factor the sibling script would refuse to ship is visible here too.
    """
    counts = subject_counts(sessions, est_key)
    if len(counts) < 2:
        fail(f"leave-one-subject-out calibration needs at least 2 subjects; this panel "
             f"has {len(counts)}.")
    loo, _heldin = cf.loo_factors(counts)

    true = np.sum([counts[s]["true"] for s in counts], axis=0)
    pred = np.sum([counts[s]["pred"] for s in counts], axis=0)
    k_all = np.array([cf.ratio_of_sums(true[c], pred[c]) for c in range(N_CLASSES)])

    stability = {}
    for c, name in enumerate(CLASS_NAMES):
        vals = np.array([loo[s][c] for s in sorted(loo)])
        u = vals[np.isfinite(vals)]
        if u.size and np.isfinite(k_all[c]) and k_all[c] > 0:
            rel = (u.max() - u.min()) / k_all[c]
            stability[name] = ("STABLE" if rel < 0.25 else
                               ("loose" if rel < 0.50 else "UNSTABLE"), rel, u.min(), u.max())
        else:
            stability[name] = ("n/a", np.nan, np.nan, np.nan)
    return loo, k_all, stability


def calibrated_sessions(sessions, est_key, loo, cls_idx):
    """
    sessions with `est_key` replaced by its corrected counts, plus the renormalization
    drift per session.

    Only the pre-registered classes are scaled; the rest pass through untouched. The
    drift returned is (corrected_total / raw_total - 1): the fraction of the session
    that renormalization has to absorb to make the nine fractions sum to one again.
    """
    out, drift = {}, {}
    for rec in sorted(sessions):
        r = sessions[rec]
        raw = np.asarray(r[est_key], dtype=float)
        k = loo[r["subject"]]
        corrected = raw.copy()
        for c in cls_idx:
            if not np.isfinite(k[c]):
                fail(f"session {rec}: no calibration factor exists for "
                     f"{CLASS_NAMES[c]} (the held-in subjects predict none of it). "
                     "Refusing to correct a class with an undefined factor.")
            corrected[c] = raw[c] * k[c]
        raw_total, cor_total = raw.sum(), corrected.sum()
        if cor_total <= 0:
            fail(f"session {rec}: corrected counts sum to {cor_total}; the score is undefined.")
        out[rec] = dict(r)
        out[rec][est_key] = corrected
        drift[rec] = cor_total / raw_total - 1.0 if raw_total > 0 else np.nan
    return out, drift


# --------------------------------------------------------------------------- #
# metric assembly
# --------------------------------------------------------------------------- #

def panel_metrics(sessions, est_keys, recs=None):
    """
    label -> per-weight-vector metrics for every condition in `est_keys`.

    Returns {wv_label: {"truth": scores, key: {...}}}, one entry per estimate key with
    rho, p, MASD, MRSD, tertile flips and the tertile vectors behind them.
    """
    recs = sorted(sessions) if recs is None else recs
    out = {}
    for label, wv in WEIGHT_VECTORS.items():
        w = class_weights(wv)
        _, truth = score_sessions(sessions, "true", w, recs)
        t_tert, t_ties = tertiles(truth)
        entry = {"truth": truth, "truth_tertiles": t_tert, "truth_ties": t_ties, "w": w}
        for key in est_keys:
            _, est = score_sessions(sessions, key, w, recs)
            rho, p = safe_spearman(est, truth)
            m = masd(est, truth)
            rel, n_used, n_skip = mrsd(est, truth)
            e_tert, e_ties = tertiles(est)
            entry[key] = {"scores": est, "rho": rho, "p": p, "masd": m,
                          "mrsd": rel, "mrsd_n": n_used, "mrsd_skipped": n_skip,
                          "tertiles": e_tert, "tertile_ties": e_ties,
                          "flips": int((e_tert != t_tert).sum())}
        out[label] = entry
    return out


def dense_wins(entry, win_key, den_key):
    """
    The pre-reg section 6 criteria, per weight vector: (rho_ok, flips_ok, masd_ok).

    "Higher rho (or both high)" and "fewer flips (or both zero)" are read as ties
    counting toward the claim, so the comparison is >= and <=; MASD is read strictly
    ("Lower MASD"), as written. Reading them any other way would need a threshold the
    pre-registration does not supply.
    """
    w, d = entry[win_key], entry[den_key]
    rho_ok = (np.isfinite(w["rho"]) and np.isfinite(d["rho"]) and d["rho"] >= w["rho"])
    return rho_ok, d["flips"] <= w["flips"], d["masd"] < w["masd"]


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_header(paired, win_meta, den_meta, den_all_meta, den_all_sessions):
    print(RULE)
    print("RISK-PROXY SENSITIVITY -- does duration bias move injury-risk stratification?")
    print(RULE)
    if win_meta is not None:
        print(f"\n  windowed run   : {win_meta['name']}   [{win_meta['results_dir']}]")
        print(f"                   sw_length={win_meta['sw_length']}s "
              f"sw_overlap={win_meta['sw_overlap']}% -> win_len={win_meta['win_len']} "
              f"step={win_meta['step']} samples")
        print(f"  dense run      : {den_meta['name']}   [{den_meta['results_dir']}]")
        print(f"                   dense_seq_len={den_meta['seq_len']} "
              f"dense_min_seg={den_meta['min_segment_len']}")
        for tag, meta in (("windowed", win_meta), ("dense", den_meta)):
            if meta["cfg"].get("no_bilstm"):
                print(f"  NOTE           : the {tag} run is the --no_bilstm ablation.")
    else:
        print()
    if den_all_meta is not None:
        print(f"  dense all-14   : {den_all_meta['name']}   [{den_all_meta['results_dir']}]")
        if den_all_meta["cfg"].get("no_bilstm"):
            print("  NOTE           : the all-subject dense run is the --no_bilstm ablation.")
        print(f"                   dense_seq_len={den_all_meta['seq_len']} "
              f"dense_min_seg={den_all_meta['min_segment_len']}   "
              f"{len(den_all_sessions)} sessions")

    print("\n  THE PROXY IS ILLUSTRATIVE. It does not predict injuries; no injury labels exist")
    print("  in this dataset. It is a monotone function of per-class time-on-task, used to ask")
    print("  whether duration bias would change a downstream risk ORDERING.\n")

    if paired is not None:
        sc.session_header(paired)
        print(f"\n  n = {len(paired)} paired sessions. Every duration is scored on the common")
        print("  covered mask of the two runs, against one truth asserted equal across both.")


def report_proxy_definition():
    """[1]: the scoring rule, spelled out before any number depends on it."""
    print("\n" + RULE)
    print("[1] THE RISK PROXY -- tiers, weights, and the assertion that guards them")
    print(RULE)
    print("\n  R_c(s) = sum_k w_{tier(k)} * f_{c,k}(s),  f = fraction of session s in class k")
    print("  under condition c. Fractions, not seconds: session length divides out.\n")

    for tier in (3, 2, 1):
        members = [n for n in CLASS_NAMES if TIER_MAP[n] == tier]
        print(f"  tier {tier} ({TIER_LABEL[tier]:<17}) : {', '.join(members)}")

    hdr = f"\n  {'vector':<14} {'w3':>6} {'w2':>6} {'w1':>6}   ordering"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for label, wv in WEIGHT_VECTORS.items():
        w1, w2, w3 = float(wv[1]), float(wv[2]), float(wv[3])
        print(f"  {label:<14} {w3:>6g} {w2:>6g} {w1:>6g}   "
              f"{w3:g} > {w2:g} > {w1:g} >= 0  OK")
    print("  " + "-" * (len(hdr) - 3))
    print(f"\n  All {len(WEIGHT_VECTORS)} vectors satisfy w3 > w2 > w1 >= 0 and TIER_MAP covers all")
    print(f"  {N_CLASSES} classes exactly; both were asserted before any run was loaded.")


def report_anchors(paired, den_all_sessions):
    """[2]: reproduce the sibling script's rebound rank correlations, or stop."""
    print("\n" + RULE)
    print("[2] ANCHOR REPRODUCTION -- rebound duration-rank rho, vs session_rank_stability.py")
    print(RULE)
    print("\n  The risk scores are built from the same per-session counts these correlations")
    print("  use. If they have moved, the loading path has changed and nothing below is")
    print("  comparable to previously reported numbers.\n")

    c = CLASS_NAMES.index("rebound")
    checks = []
    if paired is not None:
        recs = sorted(paired)
        truth = np.array([paired[r]["true"][c] for r in recs], float)
        for key, anchor_key in (("win_est", "win_rho"), ("den_est", "den_rho")):
            est = np.array([paired[r][key][c] for r in recs], float)
            rho, _ = safe_spearman(est, truth)
            checks.append((f"paired n={len(recs)} {key:<8}", rho,
                           ANCHOR_PAIRED_REBOUND[anchor_key], ANCHOR_RHO_TOL))

    if den_all_sessions is not None:
        recs_a = sorted(den_all_sessions)
        t_a = np.array([den_all_sessions[r]["true"][c] for r in recs_a], float)
        e_a = np.array([den_all_sessions[r]["est"][c] for r in recs_a], float)
        rho_a, p_a = safe_spearman(e_a, t_a)
        checks.append((f"dense-only n={len(recs_a)} rho ", rho_a,
                       ANCHOR_DENSE_ALL_REBOUND["rho"], ANCHOR_RHO_TOL))
        checks.append((f"dense-only n={len(recs_a)} p   ", p_a,
                       ANCHOR_DENSE_ALL_REBOUND["p"], ANCHOR_P_TOL))

    if not checks:
        fail("no anchor could be checked: neither panel was loaded.")

    hdr = f"  {'quantity':<24} {'observed':>10} {'anchor':>10} {'delta':>10}   verdict"
    print(hdr)
    print("  " + "-" * (len(hdr) + 2))
    bad = []
    for name, observed, anchor, tol in checks:
        delta = observed - anchor if np.isfinite(observed) else np.nan
        ok = np.isfinite(delta) and abs(delta) <= tol
        if not ok:
            bad.append((name, observed, anchor, tol))
        print(f"  {name:<24} {fmt(observed, 10, 4)} {anchor:>10.4f} "
              f"{fmt(delta, 10, 4)}   {'MATCH' if ok else 'MOVED'}")
    print("  " + "-" * (len(hdr) + 2))

    if bad:
        lines = "; ".join(f"{n.strip()} observed {o:.4f} vs anchor {a:.4f} (tol {t:g})"
                          for n, o, a, t in bad)
        fail("anchor reproduction failed: " + lines + ". The per-session counts no longer "
             "match analysis/session_rank_stability.py on this run pair -- either a "
             "different run dir was passed, or the loading path has changed. Refusing to "
             "report risk scores built on counts that cannot be reconciled.")
    print("\n  All anchors reproduce; the per-session counts are the ones previously reported.")


def report_scores(metrics, recs, est_keys, key_label, title):
    """[3]: the scores themselves, for the vector that spells the proxy out most plainly."""
    label = "A_linear"
    entry = metrics[label]
    print("\n" + RULE)
    print(f"[3] {title} -- weight vector {label}, spelled out")
    print(RULE)
    print("\n  Read this table first: the correlations below compress it into single numbers.")
    print("  'tert' is the risk tertile, 1 = highest risk. Sessions are listed worst-first")
    print("  by truth score.\n")

    truth, t_tert = entry["truth"], entry["truth_tertiles"]
    cols = "".join(f"{key_label[k] + '_R':>12}{'d':>9}{'tert':>6}" for k in est_keys)
    hdr = f"  {'session':<12} {'true_R':>9} {'tert':>6}{cols}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for j in np.argsort(-truth):
        row = f"  {recs[j]:<12} {truth[j]:>9.4f} {t_tert[j]:>6d}"
        for k in est_keys:
            e = entry[k]
            flip = "" if e["tertiles"][j] == t_tert[j] else "*"
            row += (f"{e['scores'][j]:>12.4f}{e['scores'][j] - truth[j]:>+9.4f}"
                    f"{str(e['tertiles'][j]) + flip:>6}")
        print(row)
    print("  " + "-" * (len(hdr) - 2))
    print("\n  '*' marks a session whose tertile differs from truth's. 'd' is the signed score")
    print("  deviation: positive = this condition overstates the session's risk.")
    if entry["truth_ties"]:
        for lo, hi, val in entry["truth_ties"]:
            print(f"\n  BOUNDARY TIE: truth scores are equal ({val:.6f}) across the tertile "
                  f"{lo}/{hi} cut;\n  the split between them is sort order, not risk.")


def report_primary(metrics, est_keys, key_label, n, title, tag):
    """[4]/[7d]: rho, MASD, MRSD and tertile flips, per weight vector."""
    print("\n" + RULE)
    print(f"[{tag}] {title}")
    print(RULE)
    print(f"\n  rho close to +1 means the condition orders the {n} sessions the way truth does.")
    print("  MASD is the mean absolute score deviation, MRSD the same relative to truth's")
    print(f"  score. 'flips' counts sessions landing in a different tertile than truth. At")
    print(f"  n={n} the p-values are weak evidence and the difference between two rho values")
    print("  from the same sessions is not tested.\n")

    per_key = []
    for k in est_keys:
        hdr = (f"  {key_label[k]:<14} {'rho':>9} {'p':>9} {'MASD':>9} {'MRSD':>9} "
               f"{'flips':>7} {'of':>4}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        skipped = 0
        for label in WEIGHT_VECTORS:
            e = metrics[label][k]
            skipped = max(skipped, e["mrsd_skipped"])
            print(f"  {label:<14} {fmt_rho(e['rho'], 9)} {fmt_p(e['p'], 9)} "
                  f"{fmt(e['masd'], 9, 4)} {fmt(e['mrsd'], 9, 4)} {e['flips']:>7} {n:>4}")
        print("  " + "-" * (len(hdr) - 2))
        if skipped:
            print(f"  MRSD excludes {skipped} session(s) scoring R_truth = 0 (all rest classes "
                  "under a w1 = 0 vector).")
        print()
        per_key.append(k)
    return per_key


def report_robustness(metrics, win_key, den_key, key_label):
    """[5]: the pre-reg section 6 bar, vector by vector."""
    print(RULE)
    print("[5] ROBUSTNESS -- does dense beat windowed under every weight vector?")
    print(RULE)
    print("\n  Pre-reg section 6: the claim is robust only if, under ALL five vectors, dense")
    print("  gives rho at least as high, tertile flips no more numerous, and MASD strictly")
    print("  lower. Ties count toward the claim on the first two ('or both high', 'or both")
    print("  zero'); MASD is read strictly, as written.\n")

    hdr = (f"  {'vector':<14} {'rho_win':>9} {'rho_den':>9} {'flip_w':>7} {'flip_d':>7} "
           f"{'MASD_win':>10} {'MASD_den':>10}  {'rho':>4} {'flip':>5} {'MASD':>5}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    passing = []
    for label in WEIGHT_VECTORS:
        e = metrics[label]
        w, d = e[win_key], e[den_key]
        rho_ok, flips_ok, masd_ok = dense_wins(e, win_key, den_key)
        won = rho_ok and flips_ok and masd_ok
        passing.append((label, won, rho_ok, flips_ok, masd_ok))
        mark = lambda ok: "ok" if ok else "NO"          # noqa: E731 -- table-local
        print(f"  {label:<14} {fmt_rho(w['rho'], 9)} {fmt_rho(d['rho'], 9)} "
              f"{w['flips']:>7} {d['flips']:>7} {w['masd']:>10.4f} {d['masd']:>10.4f}  "
              f"{mark(rho_ok):>4} {mark(flips_ok):>5} {mark(masd_ok):>5}  "
              f"{'dense wins' if won else 'no'}")
    print("  " + "-" * (len(hdr) - 2))
    return passing


def report_decomposition(metrics, sessions, recs, win_key, den_key, key_label):
    """[7]: which classes' duration errors drive the score disagreement."""
    contrast = max(WEIGHT_VECTORS,
                   key=lambda lb: abs(metrics[lb][win_key]["masd"] - metrics[lb][den_key]["masd"]))
    w = metrics[contrast]["w"]
    gap = abs(metrics[contrast][win_key]["masd"] - metrics[contrast][den_key]["masd"])

    print("\n" + RULE)
    print(f"[6] CLASS DECOMPOSITION -- weight vector {contrast} (largest windowed/dense MASD gap)")
    print(RULE)
    print(f"\n  {contrast} separates the two conditions most: |MASD_win - MASD_den| = {gap:.4f}.")
    print("  Each row is that class's mean SIGNED contribution to the score deviation,")
    print("  w_k * mean_s(f_est,k(s) - f_true,k(s)). The rows sum to the mean signed score")
    print("  deviation; a class with a large positive contribution is inflating risk.\n")

    truth_f = np.array([fractions(sessions[r]["true"]) for r in recs])
    hdr = (f"  {'class':<12} {'tier':>5} {'w_k':>6} {'true_frac':>10} "
           f"{key_label[win_key] + '_contrib':>18} {key_label[den_key] + '_contrib':>18}  driver")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    contribs = {}
    for k in (win_key, den_key):
        est_f = np.array([fractions(sessions[r][k]) for r in recs])
        contribs[k] = w * (est_f - truth_f).mean(axis=0)
    worst = int(np.argmax(np.abs(contribs[win_key] - contribs[den_key])))
    for c, name in enumerate(CLASS_NAMES):
        tag = "*" if name in BALL_ACTIONS else " "
        mark = "  <<< largest win/den gap" if c == worst else ""
        print(f"  {name:<11}{tag} {TIER_MAP[name]:>5} {w[c]:>6g} {truth_f[:, c].mean():>10.4f} "
              f"{contribs[win_key][c]:>+18.5f} {contribs[den_key][c]:>+18.5f} {mark}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<12} {'':>5} {'':>6} {truth_f.sum(axis=1).mean():>10.4f} "
          f"{contribs[win_key].sum():>+18.5f} {contribs[den_key].sum():>+18.5f}")
    print("\n  The true_frac column sums to 1.0 by construction. The contribution totals are")
    print("  the mean SIGNED deviations, so they can be small while MASD is large: over- and")
    print("  under-counted classes cancel in the total and do not cancel in MASD.")
    return contrast


def report_calibration_factors(cond_factors, key_label, cls_idx):
    """[6a]: the factors actually applied, per condition, with their stability."""
    print("\n" + RULE)
    print("[7a] CALIBRATION FACTORS -- fitted PER CONDITION, leave-one-subject-out")
    print(RULE)
    print("\n  k = sum(true) / sum(pred) over the held-in subjects, the pooling rule reused")
    print("  from calibration_factors.loo_factors. Only the pre-registered classes are")
    print("  corrected: " + ", ".join(CALIBRATED_CLASSES) + ".")
    print("  The two conditions get SEPARATE factors -- a shared table would correct one")
    print("  path with the other path's bias.\n")

    warn = []
    hdr = (f"  {'condition':<14} {'class':<12} {'k_all':>9} {'loo_min':>9} {'loo_max':>9} "
           f"{'spread/k':>9} {'stability':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, (loo, k_all, stability) in cond_factors.items():
        for c in cls_idx:
            name = CLASS_NAMES[c]
            stab, rel, lo, hi = stability[name]
            if stab in ("UNSTABLE", "loose"):
                warn.append((key_label[key], name, stab, k_all[c], lo, hi))
            print(f"  {key_label[key]:<14} {name:<12} {fmt(k_all[c], 9)} {fmt(lo, 9)} "
                  f"{fmt(hi, 9)} {fmt(rel, 9)} {stab:>11}")
    print("  " + "-" * (len(hdr) - 2))
    print("\n  spread/k = (loo_max - loo_min) / k_all: STABLE < 0.25, loose < 0.50,")
    print("  UNSTABLE >= 0.50 -- the same thresholds calibration_factors.py [2] uses.")
    if warn:
        print("\n  NOT A CONSTANT -- these factors move with the held-in subjects, so the")
        print("  calibrated results below inherit that instability:")
        for cond, name, stab, k, lo, hi in warn:
            print(f"    {cond:<10} {name:<10} {stab:<9} k={k:.4f} LOO range [{lo:.4f}, {hi:.4f}]")


def report_drift(drifts, key_label):
    """[6b]: how much of the correction renormalization gives back."""
    print("\n" + RULE)
    print("[7b] RENORMALIZATION DRIFT -- what correction breaks and the score needs back")
    print(RULE)
    print("\n  Scaling three classes by three constants destroys the timeline partition: the")
    print("  nine corrected durations no longer sum to the session. A weighted sum of")
    print("  FRACTIONS needs them to, so they are renormalized. This is the size of what")
    print("  renormalization absorbs -- an upper bound on how much of [7d] is the renormalizer")
    print("  rather than the correction.\n")

    hdr = f"  {'condition':<14} {'mean':>10} {'min':>10} {'max':>10} {'mean|drift|':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, d in drifts.items():
        v = np.array([d[r] for r in sorted(d)], float)
        v = v[np.isfinite(v)]
        print(f"  {key_label[key]:<14} {v.mean():>+10.4f} {v.min():>+10.4f} {v.max():>+10.4f} "
              f"{np.abs(v).mean():>13.4f}")
    print("  " + "-" * (len(hdr) - 2))
    print("\n  Read as a fraction of session length: +0.05 means correction added 5% of the")
    print("  session's duration, all of which renormalization then removes proportionally.")


def report_calibration_effect(base, cal, est_keys, key_label):
    """[6c]: did calibration help the risk score?"""
    print("\n" + RULE)
    print("[7c] CALIBRATION EFFECT -- pre-reg 5.2(f)")
    print(RULE)
    print("\n  Negative deltas are improvements. Calibration is fitted to repair pooled")
    print("  DURATION bias; whether that helps a per-session risk SCORE is a different")
    print("  question, and this table is the only place it is answered.\n")

    hdr = (f"  {'condition':<12} {'vector':<14} {'MASD':>9} {'MASD_cal':>10} {'dMASD':>9} "
           f"{'MRSD':>9} {'MRSD_cal':>10} {'dMRSD':>9} {'flip':>6} {'flip_cal':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    improved = {k: 0 for k in est_keys}
    for k in est_keys:
        for label in WEIGHT_VECTORS:
            b, c = base[label][k], cal[label][k]
            dm = c["masd"] - b["masd"]
            dr = c["mrsd"] - b["mrsd"]
            improved[k] += int(dm < 0)
            print(f"  {key_label[k]:<12} {label:<14} {b['masd']:>9.4f} {c['masd']:>10.4f} "
                  f"{dm:>+9.4f} {fmt(b['mrsd'], 9)} {fmt(c['mrsd'], 10)} {fmt_signed_or_na(dr)} "
                  f"{b['flips']:>6} {c['flips']:>9}")
        print("  " + "-" * (len(hdr) - 2))
    n = len(WEIGHT_VECTORS)
    print()
    for k in est_keys:
        print(f"  {key_label[k]:<10} calibration lowers MASD under {improved[k]} of {n} "
              "weight vectors.")
    return improved


def fmt_signed_or_na(x, width=9, prec=4):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>+{width}.{prec}f}"


def report_dense_only(metrics, n, paired_n=None):
    """[8]: pre-reg 5.2(g) -- dense against truth at the larger sample size."""
    print("\n" + RULE)
    print(f"[8] DENSE-ONLY PANEL (n={n}) -- pre-reg 5.2(g)")
    print(RULE)
    print(f"\n  Dense against truth over all {n} sessions. There is no windowed all-subject run")
    print("  in logs/, so there is no counterpart column here.")
    if paired_n:
        print(f"  This panel is NOT comparable to the paired rho above: that is computed on a")
        print(f"  different and smaller session set (n={paired_n}).")
    print()

    hdr = (f"  {'vector':<14} {'rho':>9} {'p':>9} {'MASD':>9} {'MRSD':>9} {'flips':>7} {'of':>4}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label in WEIGHT_VECTORS:
        e = metrics[label]["est"]
        print(f"  {label:<14} {fmt_rho(e['rho'], 9)} {fmt_p(e['p'], 9)} {e['masd']:>9.4f} "
              f"{fmt(e['mrsd'], 9)} {e['flips']:>7} {n:>4}")
    print("  " + "-" * (len(hdr) - 2))


def report_verdict(passing, metrics, win_key, den_key, contrast, n):
    """[9]: the pre-reg section 6/7 outcome, named."""
    print("\n" + RULE)
    print("[9] VERDICT -- pre-reg sections 6 and 7")
    print(RULE)

    won = [lb for lb, w, *_ in passing if w]
    lost = [lb for lb, w, *_ in passing if not w]
    total = len(passing)
    print(f"\n  Dense beats windowed on all three criteria under {len(won)} of {total} weight "
          "vectors.")
    if won:
        print("    dense wins   : " + ", ".join(won))
    if lost:
        print("    dense fails  : " + ", ".join(lost))

    masd_gap = {lb: metrics[lb][den_key]["masd"] - metrics[lb][win_key]["masd"]
                for lb in WEIGHT_VECTORS}
    print(f"\n  MASD difference (dense - windowed), negative favours dense:")
    for lb in WEIGHT_VECTORS:
        print(f"    {lb:<14} {masd_gap[lb]:>+9.4f}")

    print()
    if len(won) == total:
        print("  ROBUST CLAIM (section 6, primary). Under every weighting where landing actions")
        print("  outweigh locomotion outweigh rest, dense preserves risk stratification better")
        print("  than the windowed baseline.")
    elif len(won) >= 3:
        print("  CONDITIONAL CLAIM (section 6, partial). The claim holds under "
              f"{len(won)} of {total} vectors")
        print(f"  and fails under {', '.join(lost)}. Report the weight-vector dependence and")
        print("  examine which tier gap drives the failure.")
    elif len(won) == 0 and all(
            metrics[lb][win_key]["masd"] <= metrics[lb][den_key]["masd"] for lb in WEIGHT_VECTORS):
        print("  WINDOWED WINS (section 7, row 4). This is the unexpected outcome. Before")
        print("  reporting it, investigate whether the windowed path's overcount of pass and")
        print(f"  walking accidentally improves the tier 1/2 terms -- [6] decomposes {contrast}")
        print("  for exactly that question. It may indicate a proxy design issue.")
    else:
        print("  WEAK / NULL RESULT (section 6, null). The duration errors, while real, do not")
        print("  propagate cleanly into this proxy. The measurement-validity story (duration")
        print("  bias, rank stability) stands; the risk-propagation section becomes a brief")
        print("  negative result rather than a main finding.")

    print(f"\n  n = {n}. Five weight vectors over the same {n} sessions are five views of one")
    print("  dataset, not five independent experiments. The 'all five' bar is a robustness")
    print("  check against weighting choice, not a multiple-comparison correction.")


# --------------------------------------------------------------------------- #

def main():
    check_tier_map()
    check_weight_vectors()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windowed_dir", metavar="DIR",
                        help="Windowed baseline run dir (cfg.txt must NOT have dense=true). "
                             "Required unless --dense_only.")
    parser.add_argument("--dense_dir", metavar="DIR",
                        help="Dense run dir for the paired panel (cfg.txt must have "
                             "dense=true). Required unless --dense_only.")
    parser.add_argument("--dense_all_dir", metavar="DIR",
                        help="All-subject dense run dir for the n=24 panel. Required for "
                             "--dense_only; optional otherwise.")
    parser.add_argument("--windowed_npz_pattern", default=None,
                        help="Glob containing '{fold}' for the windowed run, if ambiguous.")
    parser.add_argument("--dense_npz_pattern", default=None,
                        help="Glob containing '{fold}' for the dense run, if ambiguous.")
    parser.add_argument("--dense_all_npz_pattern", default=None,
                        help="Glob containing '{fold}' for the all-subject dense run.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Also report the calibrated variant (pre-reg 5.2(f)): "
                             "per-condition leave-one-subject-out corrections applied to "
                             + ", ".join(CALIBRATED_CLASSES) + " before scoring.")
    parser.add_argument("--dense_only", action="store_true",
                        help="Run the n=24 dense-only panel only (pre-reg 5.2(g)).")
    sc.add_session_args(parser)
    args = parser.parse_args()

    if args.dense_only:
        if not args.dense_all_dir:
            fail("--dense_only needs --dense_all_dir (the all-subject dense run).")
    else:
        missing = [f"--{n}" for n in ("windowed_dir", "dense_dir") if not getattr(args, n)]
        if missing:
            fail(f"the paired panel needs {' and '.join(missing)}; pass --dense_only to run "
                 "the n=24 panel alone.")

    labels_df, segments = sc.load_common(args)
    cls_idx = [CLASS_NAMES.index(n) for n in CALIBRATED_CLASSES]

    den_all_sessions = den_all_meta = None
    if args.dense_all_dir:
        den_all_arrays, den_all_meta = sc.load_path(
            args.dense_all_dir, labels_df, segments, args.labels, args.seam_map, dense=True,
            npz_pattern=args.dense_all_npz_pattern, allow_partial=args.allow_partial_folds)
        den_all_sessions = sc.sessions_one_path(den_all_arrays, segments)
        sc.check_session_count(den_all_sessions, args.allow_partial_folds, "dense-only: ")

    if args.dense_only:
        print("\n  AMENDMENT: pre-reg 8 lists --dense-only as running 'metrics f, g only', but")
        print("  (f) is defined at n=10 paired and cannot be computed without a windowed run.")
        print("  This flag runs (g) alone. Pass --dense_all_dir alongside the paired dirs to")
        print("  get both panels in one invocation.\n")
        report_header(None, None, None, den_all_meta, den_all_sessions)
        report_proxy_definition()
        report_anchors(None, den_all_sessions)
        n_all = len(den_all_sessions)
        m_all = panel_metrics(den_all_sessions, ("est",))
        report_dense_only(m_all, n_all)
        print("\n" + RULE)
        print("Nothing was written; this script is read-only.")
        print(RULE)
        return

    win_arrays, win_meta = sc.load_path(
        args.windowed_dir, labels_df, segments, args.labels, args.seam_map, dense=False,
        npz_pattern=args.windowed_npz_pattern, allow_partial=args.allow_partial_folds)
    den_arrays, den_meta = sc.load_path(
        args.dense_dir, labels_df, segments, args.labels, args.seam_map, dense=True,
        npz_pattern=args.dense_npz_pattern, allow_partial=args.allow_partial_folds)

    if not set(win_arrays) <= set(den_arrays):
        missing = sorted(set(win_arrays) - set(den_arrays))
        fail(f"the windowed run covers subject(s) {missing} that the dense run does not. "
             "Every windowed subject must have a dense counterpart to be paired.")

    paired = sc.paired_sessions(win_arrays, den_arrays, segments)
    n = sc.check_session_count(paired, args.allow_partial_folds, "paired: ")
    if n < N_TERTILES:
        fail(f"tertiles need at least {N_TERTILES} sessions; this run pair gives {n}.")
    if n < 3:
        fail(f"a Spearman correlation needs at least 3 sessions; this run pair gives {n}.")

    recs = sorted(paired)
    win_key, den_key = "win_est", "den_est"
    est_keys = (win_key, den_key)
    key_label = {win_key: "windowed", den_key: "dense", "est": "dense"}

    report_header(paired, win_meta, den_meta, den_all_meta, den_all_sessions)
    report_proxy_definition()
    report_anchors(paired, den_all_sessions)

    base = panel_metrics(paired, est_keys, recs)
    report_scores(base, recs, est_keys, key_label, "THE SCORES")
    report_primary(base, est_keys, key_label, n,
                   "PRIMARY METRICS (n=%d paired) -- pre-reg 5.1(a)-(d)" % n, "4")
    passing = report_robustness(base, win_key, den_key, key_label)
    contrast = report_decomposition(base, paired, recs, win_key, den_key, key_label)

    if args.calibrate:
        cond_factors = {k: condition_factors(paired, k) for k in est_keys}
        report_calibration_factors(cond_factors, key_label, cls_idx)

        cal_sessions, drifts = {}, {}
        merged = {r: dict(paired[r]) for r in recs}
        for k in est_keys:
            loo = cond_factors[k][0]
            corrected, drift = calibrated_sessions(paired, k, loo, cls_idx)
            drifts[k] = drift
            for r in recs:
                merged[r][k] = corrected[r][k]
            cal_sessions[k] = corrected
        report_drift(drifts, key_label)

        cal = panel_metrics(merged, est_keys, recs)
        report_calibration_effect(base, cal, est_keys, key_label)
        report_primary(cal, est_keys, key_label, n,
                       "CALIBRATED VARIANT (n=%d paired) -- pre-reg 5.2(f)" % n, "7d")

    if den_all_sessions is not None:
        n_all = len(den_all_sessions)
        m_all = panel_metrics(den_all_sessions, ("est",))
        report_dense_only(m_all, n_all, paired_n=n)

    report_verdict(passing, base, win_key, den_key, contrast, n)

    print("\n" + RULE)
    print("Read [5] for the robustness bar, [6] for which classes drive the gap, [9] for the")
    print("pre-registered outcome. The proxy is illustrative; it does not predict injuries.")
    print("Nothing was written; this script is read-only.")
    print(RULE)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
