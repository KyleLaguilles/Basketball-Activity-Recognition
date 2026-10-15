#!/usr/bin/env python3
"""
Participant-level paired bootstrap CIs for the 24-fold LOSO results.

WHAT IS RESAMPLED. The unit is the participant (24 of them, keyed <id>_<eu|na>), since
LOSO's error is dominated by between-subject variation and samples within a subject are
nowhere near independent. Each replicate draws 24 participant ids with replacement and
uses that SAME draw for every run, every architecture and every seed. That is what makes
the dense-minus-windowed and IC-minus-DCLSTM differences paired: both sides of each
difference see the same participants.

WHAT IS POOLED. Per run and per participant, the per-class TP / FP / FN / support are
taken from sample_level_f1's own certified reconstruction (slf.process_dir for windowed
runs: last-window-wins expansion onto the raw timeline; slf.process_dir_dense for dense
runs: per-sample predictions certified against the seam map). Nothing about loading,
stitching or class mapping is re-implemented here. A replicate sums those counts over the
drawn participants (a participant drawn twice counts twice) and computes

    per-class F1   = 2TP / (2TP + FP + FN), 0 where the denominator is 0
    macro F1       = unweighted mean over all N_CLASSES classes
    duration bias  = (TP + FP) / (TP + FN) - 1        (rebound; = R/P - 1)

which is exactly sklearn's f1_score(labels=all, zero_division=0) on the pooled samples,
the definition sample_level_f1.py reports. That equivalence is asserted on the full set
for every run before any resampling, not assumed.

Seed handling: each seed is pooled on its own, then the 3 per-seed values are averaged.
Headline differences are seed-mean(dense) - seed-mean(windowed), which equals the mean of
the per-seed differences, so no dense/windowed seed pairing is implied by it. The
per-seed table does pair seed s with seed s; that pairing is by label only.

CIs are percentile 95% intervals over the replicates. The point estimate is the full
(non-resampled) 24-participant set, not the bootstrap mean.

Read-only and CPU-only: reads the saved npz predictions, trains and loads nothing, and
writes only --out.

Usage (from repo root, hangtime_har env):
    python analysis/bootstrap_ci.py --out logs/bootstrap_ci.txt
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_purity as wp            # noqa: E402  -- module-level defs, __main__-guarded
import sample_level_f1 as slf         # noqa: E402  -- certified reconstruction, reused wholesale
import boracle_duration_bias as bdb   # noqa: E402  -- duration totals, used as a cross-check

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
REB = CLASS_NAMES.index("rebound")
HardFail = wp.HardFail
fail = slf.fail

N_PARTICIPANTS = slf.ALL_SUBJECT_FOLD_COUNT     # 24
SEEDS = (1, 2, 3)

_G = "logs/subset_specific/loso_G"
RUNS = {
    ("ic", "windowed"): [f"{_G}/inceptioncontext/2026-09-11_18-13-22_24fold_windowed_seed1",
                         f"{_G}/inceptioncontext/2026-09-11_19-25-31_24fold_windowed_seed2",
                         f"{_G}/inceptioncontext/2026-09-11_20-37-56_24fold_windowed_seed3"],
    ("ic", "dense"):    [f"{_G}/inceptioncontext/2026-09-11_12-06-12_24fold_dense_seed1",
                         f"{_G}/inceptioncontext/2026-09-11_14-07-46_24fold_dense_seed2",
                         f"{_G}/inceptioncontext/2026-09-11_16-09-07_24fold_dense_seed3"],
    ("dcl", "windowed"): [f"{_G}/deepconvlstm/2026-09-23_16-09-31_24fold_deepconvlstm_seed1",
                          f"{_G}/deepconvlstm/2026-09-23_16-47-05_24fold_deepconvlstm_seed2",
                          f"{_G}/deepconvlstm/2026-09-23_17-25-32_24fold_deepconvlstm_seed3"],
    ("dcl", "dense"):   [f"{_G}/deepconvlstm/2026-10-02_19-06-24_24fold_dense_dclstm_seed1",
                         f"{_G}/deepconvlstm/2026-10-02_19-06-24_24fold_dense_dclstm_seed2",
                         f"{_G}/deepconvlstm/2026-10-02_19-06-24_24fold_dense_dclstm_seed3"],
}
NETWORK = {"ic": "inceptioncontext", "dcl": "deepconvlstm"}
ARCH_LABEL = {"ic": "InceptionContext", "dcl": "DeepConvLSTM"}

# Paper numbers the full-set reconstruction must reproduce (3 dp unless noted).
EXPECT_REB_PER_SEED = {("ic", "windowed"): (0.130, 0.122, 0.142),
                       ("ic", "dense"):    (0.245, 0.256, 0.258),
                       ("dcl", "dense"):   (0.255, 0.246, 0.242)}
EXPECT_REB_MEAN_APPROX = {("dcl", "windowed"): 0.085}
EXPECT_MACRO_MEAN = {("ic", "windowed"): 0.458, ("ic", "dense"): 0.535,
                     ("dcl", "windowed"): 0.398, ("dcl", "dense"): 0.493}
EXPECT_IC_DENSE_BIAS_PCT = -17.8
TOL_3DP = 0.0005 + 1e-9         # "reproduces to 3 dp"
TOL_APPROX = 0.001              # "about .085"
TOL_BIAS_PP = 0.1               # "about -17.8%", in percentage points
TOL_EXACT = 1e-12

COUNT_KEYS = ("tp", "fp", "fn", "support")


# --------------------------------------------------------------------------- #
# metrics from counts -- vectorised over any leading axes, classes last
# --------------------------------------------------------------------------- #

def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * tp / denom, 0.0)


def bias_from_counts(tp, fp, fn):
    est, true = tp + fp, tp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(true > 0, est / true - 1.0, np.nan)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_run(key, seed, results_dir, labels_df, seam, candidate_cache):
    arch, mode = key
    cfg = slf.load_cfg(results_dir)
    want = {"network": NETWORK[arch], "seed": seed, "dense": mode == "dense"}
    got = {k: cfg.get(k) for k in want}
    if got != want:
        fail(f"{results_dir}: cfg says {got}, expected {want}")
    if mode == "dense":
        rec = slf.process_dir_dense(results_dir, labels_df, seam, None, False)
    else:
        rec = slf.process_dir(results_dir, labels_df, candidate_cache, None)
    if len(rec["folds"]) != N_PARTICIPANTS:
        fail(f"{results_dir}: {len(rec['folds'])} folds, expected {N_PARTICIPANTS}")
    rec["counts_by_subject"] = {f["subject"]: f["counts"] for f in rec["folds"]}
    return rec


def stack_counts(rec, subjects):
    """(4, n_participants, n_classes) int array in COUNT_KEYS order, rows in `subjects` order."""
    return np.stack([np.stack([rec["counts_by_subject"][s][k] for s in subjects])
                     for k in COUNT_KEYS]).astype(np.int64)


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #

class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def ci(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return lo, hi


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels_export.csv.gz")
    parser.add_argument("--seam_map", default="data/seam_map.json")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--rng_seed", type=int, default=20261015)
    parser.add_argument("--out", default="logs/bootstrap_ci.txt")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow --out to replace an existing file.")
    args = parser.parse_args()

    if args.out:
        slf.check_csv_path(args.out, args.overwrite)

    out = Report()
    out("=" * 100)
    out("PARTICIPANT-LEVEL PAIRED BOOTSTRAP -- 24-fold LOSO, sample-level pooled F1")
    out("=" * 100)
    out(f"\n  replicates : {args.n_boot}    rng seed : {args.rng_seed} (np.random.default_rng)")
    out(f"  labels     : {args.labels}    seam map : {args.seam_map}")
    out("\n  runs (seed 1, 2, 3):")
    for (arch, mode), dirs in RUNS.items():
        out(f"    {ARCH_LABEL[arch]:<17} {mode:<9}")
        for d in dirs:
            out(f"      {d}")

    labels_df = wp.load_labels(args.labels)
    seam = slf.load_seam_map(args.seam_map)
    candidate_cache = {}

    recs = {}
    for key, dirs in RUNS.items():
        for seed, d in zip(SEEDS, dirs):
            recs[key + (seed,)] = load_run(key, seed, d, labels_df, seam, candidate_cache)

    subject_sets = {tuple(sorted(r["counts_by_subject"])) for r in recs.values()}
    if len(subject_sets) != 1:
        fail(f"runs resolve to different participant sets: {subject_sets}")
    subjects = list(subject_sets.pop())
    out(f"\n  all {len(recs)} runs resolve to the same {len(subjects)} participants")

    run_keys = list(recs)
    C = {k: stack_counts(recs[k], subjects) for k in run_keys}       # (4, P, K)

    # ------------------------------------------------------------------ #
    # sanity checks on the full set
    # ------------------------------------------------------------------ #
    out("\n" + "=" * 100)
    out("[0] SANITY CHECKS -- full (non-resampled) 24-participant set")
    out("=" * 100)
    problems = []

    def check(ok, msg):
        out(f"  {'ok  ' if ok else 'FAIL'}  {msg}")
        if not ok:
            problems.append(msg)

    full = {}
    for k in run_keys:
        tp, fp, fn, sup = C[k].sum(axis=1)
        f1 = f1_from_counts(tp, fp, fn)
        full[k] = {"f1": f1, "macro": float(f1.mean()), "reb_bias": float(bias_from_counts(tp, fp, fn)[REB])}
        r = recs[k]
        same = (np.array_equal(sup, r["sample_support"])
                and np.allclose(f1, r["sample_f1"], rtol=0, atol=TOL_EXACT)
                and abs(full[k]["macro"] - r["sample_macro"]) <= TOL_EXACT)
        check(same, f"{r['name']:<28} count-pooled F1/macro/support == sample_level_f1's sklearn values")

    out("\n  rebound F1 per seed (seed 1 / 2 / 3):")
    for key, want in EXPECT_REB_PER_SEED.items():
        got = [full[key + (s,)]["f1"][REB] for s in SEEDS]
        ok = all(abs(g - w) <= TOL_3DP for g, w in zip(got, want))
        check(ok, f"{ARCH_LABEL[key[0]]:<17} {key[1]:<9} "
                  + " / ".join(f"{g:.4f}" for g in got)
                  + "   expect " + " / ".join(f"{w:.3f}" for w in want))
    for key, want in EXPECT_REB_MEAN_APPROX.items():
        got = [full[key + (s,)]["f1"][REB] for s in SEEDS]
        m = float(np.mean(got))
        check(abs(m - want) <= TOL_APPROX,
              f"{ARCH_LABEL[key[0]]:<17} {key[1]:<9} "
              + " / ".join(f"{g:.4f}" for g in got) + f"   mean {m:.4f}, expect ~{want:.3f}")

    out("\n  macro F1, seed mean:")
    for key, want in EXPECT_MACRO_MEAN.items():
        got = [full[key + (s,)]["macro"] for s in SEEDS]
        m = float(np.mean(got))
        check(abs(m - want) <= TOL_3DP,
              f"{ARCH_LABEL[key[0]]:<17} {key[1]:<9} mean {m:.4f}   expect {want:.3f}   "
              f"(seeds " + " / ".join(f"{g:.4f}" for g in got) + ")")

    out("\n  rebound duration bias, dense runs:")
    for arch in ("ic", "dcl"):
        for s in SEEDS:
            k = (arch, "dense", s)
            r = recs[k]
            b = full[k]["reb_bias"]
            rp = r["sample_rec"][REB] / r["sample_prec"][REB] - 1.0
            check(abs(b - rp) <= TOL_EXACT,
                  f"{r['name']:<28} (TP+FP)/(TP+FN)-1 = {100 * b:+.4f}%   R/P-1 from slf = {100 * rp:+.4f}%")
            # boracle_duration_bias's own dense totals: estimate and truth in samples.
            tot = bdb.process_dir_dense(r["dir"], labels_df, seam, None, False)["totals"]
            tp, fp, fn, _ = C[k].sum(axis=1)
            same = (np.array_equal(tot["expansion_est"], tp + fp)
                    and np.array_equal(tot["raw_sample_true"], tp + fn))
            check(same, f"{r['name']:<28} TP+FP / TP+FN == boracle_duration_bias est / true totals")
    ic_bias = float(np.mean([full[("ic", "dense", s)]["reb_bias"] for s in SEEDS]))
    check(abs(100 * ic_bias - EXPECT_IC_DENSE_BIAS_PCT) <= TOL_BIAS_PP,
          f"IC dense seed-mean rebound bias {100 * ic_bias:+.2f}%   expect ~{EXPECT_IC_DENSE_BIAS_PCT:+.1f}%")

    if problems:
        out(f"\n  {len(problems)} sanity check(s) FAILED -- stopping before the bootstrap. "
            f"Nothing written to {args.out}.")
        sys.exit(1)
    out("\n  all sanity checks passed")

    # ------------------------------------------------------------------ #
    # bootstrap
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(args.rng_seed)
    P = len(subjects)
    draws = rng.integers(0, P, size=(args.n_boot, P))
    W = np.zeros((args.n_boot, P), dtype=np.int64)
    np.add.at(W, (np.arange(args.n_boot)[:, None], draws), 1)     # multiplicity per participant
    W_all = np.vstack([np.ones((1, P), dtype=np.int64), W])        # row 0 = full set

    reb, macro, bias, bias_all = {}, {}, {}, {}
    n_zero_denom = 0
    for k in run_keys:
        tp, fp, fn, _ = (W_all @ C[k][i] for i in range(4))         # (1 + B, K)
        denom = 2 * tp + fp + fn
        n_zero_denom += int((denom[1:] == 0).sum())
        f1 = f1_from_counts(tp, fp, fn)
        reb[k], macro[k] = f1[:, REB], f1.mean(axis=1)
        bias_all[k] = bias_from_counts(tp, fp, fn)                 # (1 + B, K)
        bias[k] = bias_all[k][:, REB]

    def seed_mean(d, arch, mode):
        return np.mean([d[(arch, mode, s)] for s in SEEDS], axis=0)

    def row(label, x, delta=False, pct=False):
        scale, unit = (100.0, "%") if pct else (1.0, "")
        pt = scale * x[0]
        lo, hi = (scale * v for v in ci(x[1:]))
        sign = "+" if (pct or delta) else ""
        prec = ".2f" if pct else ".4f"
        tail = f"   P(<=0) = {np.mean(x[1:] <= 0):.4f}" if delta else ""
        out(f"  {label:<44} {pt:>{sign}9{prec}}{unit:<1}   "
            f"[{lo:{sign}{prec}}{unit}, {hi:{sign}{prec}}{unit}]{tail}")

    out("\n" + "=" * 100)
    out("[1] SUMMARY -- seed-averaged, point = full set, 95% percentile CI over "
        f"{args.n_boot} participant resamples")
    out("=" * 100)
    out("  Paired: every replicate applies one participant draw to all 12 runs.")
    out("  P(<=0) = share of replicates where the difference is <= 0 (descriptive, not a test).\n")
    out(f"  {'quantity':<44} {'point':>10}   95% CI")
    out("  " + "-" * 96)
    for arch in ("ic", "dcl"):
        a = ARCH_LABEL[arch]
        w_r, d_r = seed_mean(reb, arch, "windowed"), seed_mean(reb, arch, "dense")
        w_m, d_m = seed_mean(macro, arch, "windowed"), seed_mean(macro, arch, "dense")
        row(f"{a} rebound F1, windowed", w_r)
        row(f"{a} rebound F1, dense", d_r)
        row(f"{a} rebound F1, dense - windowed", d_r - w_r, delta=True)
        row(f"{a} macro F1, windowed", w_m)
        row(f"{a} macro F1, dense", d_m)
        row(f"{a} macro F1, dense - windowed", d_m - w_m, delta=True)
        out("  " + "-" * 96)
    b_ic, b_dcl = seed_mean(bias, "ic", "dense"), seed_mean(bias, "dcl", "dense")
    row("rebound duration bias, IC dense", b_ic, pct=True)
    row("rebound duration bias, DCLSTM dense", b_dcl, pct=True)
    row("rebound bias, IC dense - DCLSTM dense (pp)", b_ic - b_dcl, pct=True)
    out(f"\n  bias = (TP+FP)/(TP+FN) - 1 on pooled counts; IC - DCLSTM is in percentage points.")

    out("\n" + "=" * 100)
    out("[2] PER-SEED -- each seed pooled on its own; differences pair dense seed s with windowed seed s")
    out("=" * 100)
    for s in SEEDS:
        out(f"\n  seed {s}")
        out(f"  {'quantity':<44} {'point':>10}   95% CI")
        out("  " + "-" * 96)
        for arch in ("ic", "dcl"):
            a = ARCH_LABEL[arch]
            w, d = (arch, "windowed", s), (arch, "dense", s)
            row(f"{a} rebound F1, windowed", reb[w])
            row(f"{a} rebound F1, dense", reb[d])
            row(f"{a} rebound F1, dense - windowed", reb[d] - reb[w], delta=True)
            row(f"{a} macro F1, windowed", macro[w])
            row(f"{a} macro F1, dense", macro[d])
            row(f"{a} macro F1, dense - windowed", macro[d] - macro[w], delta=True)
        row("rebound duration bias, IC dense", bias[("ic", "dense", s)], pct=True)
        row("rebound duration bias, DCLSTM dense", bias[("dcl", "dense", s)], pct=True)
        row("rebound bias, IC dense - DCLSTM dense (pp)",
            bias[("ic", "dense", s)] - bias[("dcl", "dense", s)], pct=True)

    n_nan = sum(int(np.isnan(bias[k][1:]).sum()) for k in run_keys)
    out("\n" + "=" * 100)
    out("[3] NOTES")
    out("=" * 100)
    out(f"  (run, class, replicate) cells with 2TP+FP+FN = 0 (F1 set to 0, as sklearn's "
        f"zero_division=0): {n_zero_denom}")
    out(f"  (run, replicate) cells with zero rebound support (bias undefined, excluded): {n_nan}")
    out("  Windowed runs are scored on covered samples (trailing tail past the last window")
    out("  dropped); dense runs on kept seam-map segments -- both exactly as sample_level_f1.py.")

    # Sections [4]-[5] reuse the same draws (W) as [1]-[3]; nothing above depends on them.
    out("\n" + "=" * 100)
    out("[4] DENSE vs DENSE -- InceptionContext minus DeepConvLSTM, same paired resamples")
    out("=" * 100)
    out("  Seed-averaged: seed-mean(IC dense) - seed-mean(DCLSTM dense). Per-seed rows pair IC seed s")
    out("  with DCLSTM seed s, by label only.\n")
    out(f"  {'quantity':<44} {'point':>10}   95% CI")
    out("  " + "-" * 96)
    row("rebound F1, IC dense - DCLSTM dense",
        seed_mean(reb, "ic", "dense") - seed_mean(reb, "dcl", "dense"), delta=True)
    row("macro F1, IC dense - DCLSTM dense",
        seed_mean(macro, "ic", "dense") - seed_mean(macro, "dcl", "dense"), delta=True)
    for s in SEEDS:
        out("  " + "-" * 96)
        row(f"seed {s}: rebound F1, IC - DCLSTM",
            reb[("ic", "dense", s)] - reb[("dcl", "dense", s)], delta=True)
        row(f"seed {s}: macro F1, IC - DCLSTM",
            macro[("ic", "dense", s)] - macro[("dcl", "dense", s)], delta=True)

    b_ic_all = seed_mean(bias_all, "ic", "dense")          # (1 + B, K)
    b_dcl_all = seed_mean(bias_all, "dcl", "dense")
    out("\n" + "=" * 100)
    out("[5] PER-CLASS DURATION BIAS -- dense runs, seed-averaged, (TP+FP)/(TP+FN) - 1 on pooled counts")
    out("=" * 100)
    out("  Point = full set; brackets = 95% percentile CI. IC - DCLSTM is in percentage points.\n")
    cell = lambda x: f"{100 * x[0]:>+8.2f} [{100 * ci(x[1:])[0]:>+7.2f}, {100 * ci(x[1:])[1]:>+7.2f}]"
    out(f"  {'class':<11} {'IC dense (%)':>27} {'DCLSTM dense (%)':>28} {'IC - DCLSTM (pp)':>28}")
    out("  " + "-" * 96)
    for c, name in enumerate(CLASS_NAMES):
        out(f"  {name:<11} {cell(b_ic_all[:, c]):>27} {cell(b_dcl_all[:, c]):>28} "
            f"{cell(b_ic_all[:, c] - b_dcl_all[:, c]):>28}")
    n_nan_all = sum(int(np.isnan(bias_all[k][1:]).sum()) for k in run_keys if k[1] == "dense")
    out(f"\n  (dense run, class, replicate) cells with zero support (bias undefined, excluded): {n_nan_all}")
    out("  The rebound row equals the bias rows of [1].")

    with open(args.out, "w") as fid:
        fid.write("\n".join(out.lines) + "\n")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
