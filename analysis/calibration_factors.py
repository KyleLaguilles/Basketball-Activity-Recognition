#!/usr/bin/env python3
"""
Per-class duration CORRECTION FACTORS, and an honest leave-one-fold-out estimate of
what they are worth on a fold the factor was not fitted on.

THE DELIVERABLE. boracle_duration_bias.py establishes that this classifier's per-class
durations are biased, and that the bias is largely one-directional per class. A
one-directional bias is the repairable kind: multiply the predicted duration by a
constant and the bias is gone. That constant is

    k_c = (total TRUE duration of class c) / (total PREDICTED duration of class c)

so the corrected estimate is `predicted_seconds * k_c`. k_c > 1 means the model
UNDER-predicts that class and the correction scales it up. This is the table the
downstream pipeline can apply today, and it is written to CSV by --out_csv.

WHY LEAVE-ONE-FOLD-OUT. A factor fitted on all folds and then evaluated on those same
folds removes the bias by construction -- the residual would be exactly zero and the
table would prove nothing. So the factor is estimated on all-but-one fold and applied
to the held-out fold, once per fold (LOSO over subjects, matching how the model itself
was validated). The residual on the held-out fold is what a NEW subject should expect,
and it is the only number here that is honest about generalization.

POOLING RULE: RATIO OF SUMS.

    k_c^(-f) = sum_{g != f} true_gc / sum_{g != f} pred_gc

not the mean of the per-fold ratios. The two differ substantially for the low-support
classes, and the mean-of-ratios is the wrong one: a single fold predicting near-zero
seconds of a class produces an enormous ratio that dominates the mean, so the factor
ends up set by the fold where the model failed hardest. The ratio of sums weights each
fold by its duration, which is the quantity actually being corrected.

WHAT A CORRECTION FACTOR CANNOT DO, stated because the residual table will otherwise be
over-read: k_c is a single scalar per class. It removes the POOLED bias and it cannot
touch the SPREAD -- if the model over-predicts a class on one subject and under-predicts
it on the next, no scalar fixes both, and the corrected per-fold errors will still be
large in both directions. Table [4] reports that spread explicitly, and table [5]
reports the classes where correction makes the typical fold WORSE despite improving the
pooled total. Read analysis/session_rank_stability.py alongside this: a class whose
sessions cannot be ranked is a class no scalar correction repairs.

CONSERVATION IS BROKEN BY CORRECTION, deliberately. The raw predictions conserve --
every covered sample carries exactly one class, so the predicted durations sum to the
total. Multiplying each class by a different k_c destroys that: the corrected durations
no longer sum to the recording length. Table [3] quantifies the drift, using the
LEAVE-ONE-OUT factors rather than the all-fold one: the all-fold factor satisfies
k_c * pred_c == true_c exactly, so its drift is identically zero and measuring it would
say nothing. If the consumer needs a partition of the timeline rather than nine
independent exposure terms, it must renormalize, and doing so reintroduces bias. That
trade-off is the consumer's to make, so it is measured here rather than resolved.

Read-only and CPU-only apart from --out_csv. Loads no checkpoints, imports nothing
from src/.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/calibration_factors.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-09-04_17-48-10_dense_b10_all14_seed1 \
        --dense --out_csv calibration_factors.csv

    python analysis/calibration_factors.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30
"""

import argparse
import csv
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
EXPLOSIVE = sc.EXPLOSIVE

# A factor estimated from less than this much held-in predicted duration is reported
# but marked unreliable: the denominator is too small for the ratio to be stable.
MIN_HELDIN_PRED_SEC = 5.0


def fmt(x, width=9, prec=2):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>{width}.{prec}f}"


def fmt_signed(x, width=9, prec=1):
    return f"{'n/a':>{width}}" if not np.isfinite(x) else f"{x:>+{width}.{prec}f}"


def ratio_of_sums(true_samples, pred_samples):
    """
    sum(true) / sum(pred), NaN where nothing was predicted.

    NaN rather than inf: a class the model never predicts at all has no factor that
    could scale it up, and a downstream consumer must be told that rather than handed
    an infinity to propagate.
    """
    t, p = float(np.sum(true_samples)), float(np.sum(pred_samples))
    return t / p if p > 0 else np.nan


# --------------------------------------------------------------------------- #
# per-fold counts
# --------------------------------------------------------------------------- #

def fold_counts(subject_arrays):
    """
    fold -> {"true": vec, "pred": vec, "n_covered"} in SAMPLES, on that fold's own
    covered mask. Conservation is asserted per fold: on the covered mask every sample
    carries exactly one truth and one prediction, so both vectors must sum to n_covered.
    """
    out = {}
    for subj in sorted(subject_arrays):
        a = subject_arrays[subj]
        cov = (a["pred"] != sc.UNCOVERED) & (a["true"] != sc.UNCOVERED)
        n_cov = int(cov.sum())
        true = np.bincount(a["true"][cov], minlength=N_CLASSES).astype(np.int64)
        pred = np.bincount(a["pred"][cov], minlength=N_CLASSES).astype(np.int64)
        for label, vec in (("true", true), ("pred", pred)):
            if int(vec.sum()) != n_cov:
                fail(f"fold {subj}: {label} sums to {int(vec.sum())} samples but the covered "
                     f"mask holds {n_cov}. The fold bincounts do not conserve.")
        out[subj] = {"true": true, "pred": pred, "n_covered": n_cov,
                     "npz": a["npz"], "fold": a["fold"]}
    return out


def loo_factors(counts):
    """
    fold -> per-class factor estimated on ALL OTHER folds, ratio of sums.

    Also returns the held-in predicted duration behind each factor, so a factor resting
    on a second of predicted time can be flagged rather than quoted.
    """
    folds = sorted(counts)
    factors, heldin_pred = {}, {}
    for f in folds:
        others = [g for g in folds if g != f]
        if not others:
            fail("leave-one-fold-out needs at least 2 folds; this run has 1.")
        t = np.sum([counts[g]["true"] for g in others], axis=0)
        p = np.sum([counts[g]["pred"] for g in others], axis=0)
        factors[f] = np.array([ratio_of_sums(t[c], p[c]) for c in range(N_CLASSES)])
        heldin_pred[f] = secs(p)
    return factors, heldin_pred


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_header(counts, meta):
    print("=" * 116)
    print("CALIBRATION FACTORS -- per-class duration correction, with leave-one-fold-out residuals")
    print("=" * 116)
    print(f"\n  run            : {meta['name']}   [{meta['results_dir']}]")
    if meta["dense"]:
        print(f"  sequencing     : DENSE, dense_seq_len={meta['seq_len']}, "
              f"dense_min_seg={meta['min_segment_len']}")
        print("                   one prediction per raw 50 Hz sample")
    else:
        print(f"  windowing      : sw_length={meta['sw_length']}s sw_overlap={meta['sw_overlap']}% "
              f"-> win_len={meta['win_len']} step={meta['step']} samples")
        print("                   durations expanded last-window-wins onto the raw timeline")
    if meta["cfg"].get("no_bilstm"):
        print("  NOTE           : this run is the --no_bilstm ablation.")
    print(f"  sampling rate  : {sc.SAMPLING_RATE} Hz (preprocess_data.py:31; never in cfg.txt)")
    print("  pooling rule   : ratio of sums, sum(true) / sum(pred) over the held-in folds")

    print(f"\n  {'fold':<8} {'covered':>10} {'duration':>12}  npz")
    for f in sorted(counts):
        r = counts[f]
        print(f"  {f:<8} {r['n_covered']:>10} {sc.hms(secs(r['n_covered'])):>12}  {r['npz']}")
    tot = sum(r["n_covered"] for r in counts.values())
    print(f"  {'TOTAL':<8} {tot:>10} {sc.hms(secs(tot)):>12}   ({len(counts)} folds)")


def report_uncorrected(counts):
    """[1]: the bias the factors exist to remove."""
    folds = sorted(counts)
    true = np.sum([counts[f]["true"] for f in folds], axis=0)
    pred = np.sum([counts[f]["pred"] for f in folds], axis=0)

    print("\n" + "=" * 116)
    print("[1] UNCORRECTED BIAS -- pooled over all folds, the quantity being corrected")
    print("=" * 116)
    print("\n  bias_pct = (pred - true) / true * 100. UNDER means the model reports less exposure")
    print("  than occurred, which is the dangerous direction for an injury pipeline.\n")

    hdr = (f"  {'class':<12} {'true_sec':>11} {'pred_sec':>11} {'bias_sec':>11} {'bias_pct':>10} "
           f"{'dir':>6} {'ball':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c, name in enumerate(CLASS_NAMES):
        t, p = secs(true[c]), secs(pred[c])
        pct = sc.rel_err_pct(pred[c], true[c])
        d = "n/a" if t <= 0 else ("OVER" if p > t else ("UNDER" if p < t else "exact"))
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {t:>11.2f} {p:>11.2f} {p - t:>+11.2f} {fmt_signed(float(pct), 9)}% "
              f"{d:>6} {'yes' if name in BALL_ACTIONS else '':>5}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<12} {secs(true).sum():>11.2f} {secs(pred).sum():>11.2f} "
          f"{secs(pred).sum() - secs(true).sum():>+11.2f}")
    print("\n  The total is exact by construction: on the covered mask every sample carries one")
    print("  truth and one prediction, so the two columns sum to the same duration.")
    return true, pred


def report_factor_table(counts, true, pred, factors, heldin_pred):
    """[2]: the deliverable."""
    folds = sorted(counts)
    full = np.array([ratio_of_sums(true[c], pred[c]) for c in range(N_CLASSES)])

    print("\n" + "=" * 116)
    print("[2] CORRECTION FACTORS -- multiply predicted seconds by k to get corrected seconds")
    print("=" * 116)
    print("\n  k = sum(true) / sum(pred), pooled over all folds. k > 1 scales an under-predicted")
    print("  class up; k < 1 scales an over-predicted class down. The LOO columns show how much")
    print("  k moves when one fold is withheld -- a wide [min, max] means the factor is set by")
    print("  whichever subjects happen to be in the training set, and will not transfer.\n")

    hdr = (f"  {'class':<12} {'k_all':>9} {'loo_min':>9} {'loo_max':>9} {'loo_std':>9} "
           f"{'spread/k':>9} {'stability':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    unstable = []
    for c, name in enumerate(CLASS_NAMES):
        vals = np.array([factors[f][c] for f in folds])
        u = vals[np.isfinite(vals)]
        lo, hi = (u.min(), u.max()) if u.size else (np.nan, np.nan)
        std = u.std(ddof=0) if u.size else np.nan
        spread = hi - lo if u.size else np.nan
        # A factor whose LOO range spans more than half its own value is not a constant.
        # The column shows the RELATIVE spread, which is what the verdict is based on --
        # an absolute spread is not comparable across classes with different k.
        if np.isfinite(spread) and np.isfinite(full[c]) and full[c] > 0:
            rel_spread = spread / full[c]
            stab = "STABLE" if rel_spread < 0.25 else ("loose" if rel_spread < 0.50 else "UNSTABLE")
            if stab == "UNSTABLE":
                unstable.append((name, full[c], lo, hi))
        else:
            rel_spread = np.nan
            stab = "n/a"
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {fmt(full[c], 9, 4)} {fmt(lo, 9, 4)} {fmt(hi, 9, 4)} "
              f"{fmt(std, 9, 4)} {fmt(rel_spread, 9, 4)} {stab:>11}")
    print("  " + "-" * (len(hdr) - 2))
    print("\n  spread/k = (loo_max - loo_min) / k_all, the LOO range as a fraction of the factor")
    print("  itself: STABLE < 0.25, loose < 0.50, UNSTABLE >= 0.50.")

    thin = [(CLASS_NAMES[c], f, heldin_pred[f][c]) for f in folds for c in range(N_CLASSES)
            if np.isfinite(factors[f][c]) and heldin_pred[f][c] < MIN_HELDIN_PRED_SEC]
    if thin:
        print(f"\n  Factors resting on under {MIN_HELDIN_PRED_SEC:.0f}s of held-in predicted "
              "duration (denominator too small to be stable):")
        for name, f, s in thin:
            print(f"    {name:<12} fold {f}: held-in predicted duration {s:.2f}s")
    if unstable:
        print("\n  UNSTABLE FACTORS -- do not ship these as constants:")
        for name, k, lo, hi in unstable:
            print(f"    {name:<12} k={k:.4f} but LOO range [{lo:.4f}, {hi:.4f}]")
    return full


def report_conservation_drift(counts, factors):
    """[3]: what applying a per-class factor costs the partition property, out of sample."""
    folds = sorted(counts)
    print("\n" + "=" * 116)
    print("[3] CONSERVATION DRIFT -- correction breaks the timeline partition")
    print("=" * 116)
    print("\n  Raw predictions partition the covered timeline: every sample carries exactly one")
    print("  class, so they sum to the recording length. Scaling each class by a different k")
    print("  destroys that. The drift is measured with the LEAVE-ONE-OUT factors, applied to the")
    print("  held-out fold -- the all-fold factor is excluded because it reproduces each class's")
    print("  true duration by construction (k_c * pred_c == true_c), so its drift is exactly zero")
    print("  and would say nothing about a new subject.\n")

    hdr = (f"  {'fold':<8} {'true_s':>11} {'raw_pred_s':>12} {'corrected_s':>12} "
           f"{'drift_s':>10} {'drift_pct':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    drifts = []
    for f in folds:
        r = counts[f]
        k = factors[f]
        corr = float(np.nansum(secs(r["pred"]) * np.where(np.isfinite(k), k, 1.0)))
        t = secs(r["true"]).sum()
        pct = (corr - t) / t * 100.0 if t > 0 else np.nan
        drifts.append(pct)
        print(f"  {f:<8} {t:>11.2f} {secs(r['pred']).sum():>12.2f} {corr:>12.2f} "
              f"{corr - t:>+10.2f} {fmt_signed(pct, 9)}%")
    print("  " + "-" * (len(hdr) - 2))
    d = np.array(drifts, float)
    d = d[np.isfinite(d)]
    if d.size:
        print(f"  {'mean':<8} {'':>11} {'':>12} {'':>12} {'':>10} {d.mean():>+9.2f}%")
        print(f"  {'max |.|':<8} {'':>11} {'':>12} {'':>12} {'':>10} {np.abs(d).max():>9.2f}%")
        print(f"\n  Out of sample the corrected durations miss the recording length by "
              f"{np.abs(d).mean():.2f}% on")
        print("  average. A consumer needing a partition of the timeline must renormalize by that,")
        print("  which spreads the residual back across every class in proportion. A consumer")
        print("  needing nine independent exposure terms need not, and should not.")


def report_loo_residuals(counts, factors, heldin_pred):
    """[4]: the honest number -- residual on a fold the factor was not fitted on."""
    folds = sorted(counts)
    print("\n" + "=" * 116)
    print("[4] LEAVE-ONE-FOLD-OUT RESIDUALS -- factor fitted on the other folds, applied here")
    print("=" * 116)
    print("\n  For each fold: k is estimated from the OTHER folds only, then applied to this")
    print("  fold's predictions. 'before' is the uncorrected relative error, 'after' the")
    print("  residual. This is what a new subject should expect, not the in-sample fit.\n")

    before = np.full((N_CLASSES, len(folds)), np.nan)
    after = np.full((N_CLASSES, len(folds)), np.nan)
    for j, f in enumerate(folds):
        r = counts[f]
        for c in range(N_CLASSES):
            before[c, j] = sc.rel_err_pct(r["pred"][c], r["true"][c])
            k = factors[f][c]
            if np.isfinite(k):
                after[c, j] = sc.rel_err_pct(r["pred"][c] * k, r["true"][c])

    for c, name in enumerate(CLASS_NAMES):
        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name}{tag}")
        hdr = (f"    {'fold':<8} {'true_s':>9} {'pred_s':>9} {'k_loo':>8} {'corr_s':>9} "
               f"{'before%':>10} {'after%':>10} {'better':>8}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for j, f in enumerate(folds):
            r = counts[f]
            k = factors[f][c]
            corr = secs(r["pred"][c]) * k if np.isfinite(k) else np.nan
            b, a = before[c, j], after[c, j]
            if np.isfinite(b) and np.isfinite(a):
                better = "yes" if abs(a) < abs(b) else ("no" if abs(a) > abs(b) else "tie")
            else:
                better = "n/a"
            print(f"    {f:<8} {secs(r['true'][c]):>9.2f} {secs(r['pred'][c]):>9.2f} "
                  f"{fmt(k, 8, 4)} {fmt(corr, 9)} {fmt_signed(b, 10)} {fmt_signed(a, 10)} "
                  f"{better:>8}")
        ub = before[c][np.isfinite(before[c])]
        ua = after[c][np.isfinite(after[c])]
        both = np.isfinite(before[c]) & np.isfinite(after[c])
        wins = int((np.abs(after[c][both]) < np.abs(before[c][both])).sum())
        print(f"    {'MAE':<8} {'':>9} {'':>9} {'':>8} {'':>9} "
              f"{np.abs(ub).mean() if ub.size else np.nan:>10.1f} "
              f"{np.abs(ua).mean() if ua.size else np.nan:>10.1f} "
              f"{f'{wins}/{int(both.sum())}':>8}")
        print()
    return before, after


def pooled_loo_residual(counts, factors):
    """
    Per class: relative error of the summed LEAVE-ONE-OUT corrected durations.

    Not the all-fold residual, which is identically zero because k_c is defined to make
    it so. This is the pooled number a consumer would actually see on unseen subjects.
    """
    folds = sorted(counts)
    true = np.sum([counts[f]["true"] for f in folds], axis=0)
    corrected = np.zeros(N_CLASSES, float)
    defined = np.ones(N_CLASSES, bool)
    for f in folds:
        k = factors[f]
        corrected += secs(counts[f]["pred"]) * np.where(np.isfinite(k), k, 0.0)
        defined &= np.isfinite(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((secs(true) > 0) & defined,
                       (corrected - secs(true)) / np.maximum(secs(true), 1e-12) * 100.0, np.nan)
    return out


def report_verdict(counts, before, after, full, factors):
    """[5]: where correction actually helps, and where it does not."""
    folds = sorted(counts)
    print("=" * 116)
    print("[5] DOES THE CORRECTION HELP? -- pooled gain vs typical-fold gain, both out of sample")
    print("=" * 116)
    print("\n  'pooled' compares the summed durations, 'MAE' the mean |relative error| across")
    print("  folds -- both computed with the LEAVE-ONE-OUT factors, so neither is an in-sample")
    print("  fit. (The all-fold factor's pooled residual is identically zero by construction and")
    print("  is not shown.) A class that improves pooled but not MAE has had its BIAS removed and")
    print("  its SPREAD left intact: the pipeline's total gets better, any single subject does not.\n")

    true = np.sum([counts[f]["true"] for f in folds], axis=0)
    pred = np.sum([counts[f]["pred"] for f in folds], axis=0)
    pooled_after = pooled_loo_residual(counts, factors)

    hdr = (f"  {'class':<12} {'pooled_before%':>15} {'pooled_loo%':>13} {'MAE_before%':>12} "
           f"{'MAE_loo%':>10} {'folds_helped':>13} {'verdict':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    no_help, mixed = [], []
    for c, name in enumerate(CLASS_NAMES):
        pb = float(sc.rel_err_pct(pred[c], true[c]))
        pa = float(pooled_after[c])
        both = np.isfinite(before[c]) & np.isfinite(after[c])
        mb = np.abs(before[c][both]).mean() if both.any() else np.nan
        ma = np.abs(after[c][both]).mean() if both.any() else np.nan
        wins = int((np.abs(after[c][both]) < np.abs(before[c][both])).sum())
        n = int(both.sum())

        # The mean and the fold count can disagree -- most folds improving while the mean
        # worsens means one fold blew up. That is 'mixed', not 'HELPS' and not 'NO HELP'.
        if not np.isfinite(ma) or not np.isfinite(mb) or n == 0:
            verdict = "n/a"
        elif ma < mb and wins * 2 > n:
            verdict = "HELPS"
        elif ma >= mb and wins * 2 <= n:
            verdict = "NO HELP"
            no_help.append((name, mb, ma, wins, n))
        else:
            verdict = "mixed"
            mixed.append((name, mb, ma, wins, n))

        tag = "*" if name in BALL_ACTIONS else " "
        print(f"  {name:<11}{tag} {fmt_signed(pb, 14)}% {fmt_signed(pa, 12)}% "
              f"{fmt(mb, 12, 1)} {fmt(ma, 10, 1)} {f'{wins}/{n}':>13} {verdict:>9}")
    print("  " + "-" * (len(hdr) - 2))

    if mixed:
        print("\n  MIXED -- the fold count and the mean disagree, so one or two folds carry the mean:")
        for name, mb, ma, wins, n in mixed:
            direction = ("most folds improved but the mean worsened"
                         if wins * 2 > n else "the mean improved but most folds did not")
            print(f"    {name:<12} MAE {mb:.1f}% -> {ma:.1f}%, helped {wins}/{n} folds "
                  f"-- {direction}")
    if no_help:
        print("\n  NO HELP -- correction makes the typical fold worse:")
        for name, mb, ma, wins, n in no_help:
            print(f"    {name:<12} MAE {mb:.1f}% -> {ma:.1f}%, helped {wins}/{n} folds")
    if no_help or mixed:
        print("\n  For these the bias is not one-directional across subjects, so no scalar repairs")
        print("  it. Applying the factor anyway trades a known pooled bias for a per-subject one.")

    helps = [CLASS_NAMES[c] for c in range(N_CLASSES)
             if CLASS_NAMES[c] not in [x[0] for x in no_help + mixed]
             and np.isfinite(full[c])]
    print(f"\n  SHIP-READY (helps the typical fold, factor stable in [2]): "
          f"{', '.join(helps) if helps else 'none'}")
    helped_explosive = [n for n in EXPLOSIVE if n in helps]
    print(f"  Explosive ball actions ({', '.join(EXPLOSIVE)}): correction helps "
          f"{len(helped_explosive)} of {len(EXPLOSIVE)}"
          + (f" ({', '.join(helped_explosive)})." if helped_explosive else "."))
    return pooled_after


def write_csv(path, counts, full, factors, before, after, pooled_after, meta):
    folds = sorted(counts)
    true = np.sum([counts[f]["true"] for f in folds], axis=0)
    pred = np.sum([counts[f]["pred"] for f in folds], axis=0)

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# run", meta["name"], "dir", meta["results_dir"],
                    "dense", meta["dense"], "n_folds", len(folds),
                    "pooling", "ratio_of_sums"])
        w.writerow(["class", "true_sec", "pred_sec", "factor_k",
                    "loo_k_min", "loo_k_max", "loo_k_std",
                    "uncorrected_bias_pct", "loo_pooled_residual_pct",
                    "loo_residual_mae_pct", "loo_folds_helped", "n_folds"])
        for c, name in enumerate(CLASS_NAMES):
            vals = np.array([factors[f][c] for f in folds])
            u = vals[np.isfinite(vals)]
            both = np.isfinite(before[c]) & np.isfinite(after[c])
            wins = int((np.abs(after[c][both]) < np.abs(before[c][both])).sum())
            w.writerow([
                name,
                f"{secs(true[c]):.4f}", f"{secs(pred[c]):.4f}",
                "" if not np.isfinite(full[c]) else f"{full[c]:.6f}",
                "" if not u.size else f"{u.min():.6f}",
                "" if not u.size else f"{u.max():.6f}",
                "" if not u.size else f"{u.std(ddof=0):.6f}",
                "" if not np.isfinite(sc.rel_err_pct(pred[c], true[c]))
                else f"{float(sc.rel_err_pct(pred[c], true[c])):.4f}",
                "" if not np.isfinite(pooled_after[c]) else f"{float(pooled_after[c]):.4f}",
                "" if not both.any() else f"{np.abs(after[c][both]).mean():.4f}",
                wins, int(both.sum()),
            ])
    print(f"\n  Wrote {path}  ({N_CLASSES} classes, factor_k is the column to apply).")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True, metavar="DIR",
                        help="Run dir holding cfg.txt and the per-fold npz files.")
    parser.add_argument("--dense", action="store_true",
                        help="Read a --dense run (cfg.txt must have dense=true).")
    parser.add_argument("--npz_pattern", default=None,
                        help="Glob containing '{fold}', if a fold resolves to several npz.")
    parser.add_argument("--out_csv", default=None, metavar="PATH",
                        help="Write the correction table to this CSV. The only thing this "
                             "script ever writes; omit it and the run is read-only.")
    sc.add_session_args(parser)
    args = parser.parse_args()

    labels_df, segments = sc.load_common(args)
    subject_arrays, meta = sc.load_path(args.results_dir, labels_df, segments, args.labels,
                                        args.seam_map, dense=args.dense,
                                        npz_pattern=args.npz_pattern,
                                        allow_partial=args.allow_partial_folds)

    counts = fold_counts(subject_arrays)
    if len(counts) < 2:
        fail(f"leave-one-fold-out needs at least 2 folds; {args.results_dir} has {len(counts)}.")
    factors, heldin_pred = loo_factors(counts)

    report_header(counts, meta)
    true, pred = report_uncorrected(counts)
    full = report_factor_table(counts, true, pred, factors, heldin_pred)
    report_conservation_drift(counts, factors)
    before, after = report_loo_residuals(counts, factors, heldin_pred)
    pooled_after = report_verdict(counts, before, after, full, factors)

    if args.out_csv:
        write_csv(args.out_csv, counts, full, factors, before, after, pooled_after, meta)

    print("\n" + "=" * 116)
    print("Ship [2] as the correction table; quote [4]/[5] as what it is worth out of sample.")
    print("A class marked UNSTABLE in [2] or NO HELP in [5] should not be corrected by a scalar.")
    print("=" * 116)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
