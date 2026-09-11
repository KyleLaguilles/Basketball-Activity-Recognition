#!/usr/bin/env python3
"""
Is rebound's low F1 a label-noise artifact of last-sample windowing?

Last-sample labeling ([sliding_window.py] output_y = [[i[-1]] ...]) assigns a
window the label of its final sample. A window whose last sample lands in a
rebound is labeled rebound even if the preceding 49 samples are run-in. This
script measures how often that happens and whether it tracks the model's errors.

    purity(window) = fraction of the window's 50 samples whose true per-sample
                     label equals the window's LAST-sample label

NOTE this is NOT the same purity as analysis/window_purity_corrected.py, which
uses majority_class_count / win_len. Both are defensible; they differ whenever
the last-sample label is not the modal label, which is exactly the population of
interest here. Do not cross-compare the two scripts' numbers.

SEED INVARIANCE. Purity, majority label, last-sample label and the per-sample
histogram are functions of y_true and the raw per-sample labels ONLY. y_true is
byte-identical across the three seed dirs, so the per-window table is computed
ONCE over all val windows and the three seeds' predictions are joined onto it.
Consequences, both surfaced in the output:
  - base-rate purity distributions are a property of the windowing, not of any
    model, and are reported once rather than per seed;
  - the three seeds are correlated replicates over the SAME windows, not 3x the
    sample size. With ~200 rebound windows total, seed agreement is not
    independent evidence.

Fold -> subject resolution is brute-forced over all candidate subjects and
accepted only on window-count + byte-identical last-sample-label equality,
never on the filename token (analysis/_loso_common.py:337-341 documents that the
token can be wrong). That resolution IS the reconstruction certification.

Read-only. Touches nothing under src/ and writes only the --out_csv it is told to.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/window_purity.py \
        --log_root logs/subset_specific/loso_G/inceptioncontext \
        --labels labels_export.csv.gz \
        --out_csv window_purity_records.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CLASS_NAMES = [
    "dribbling", "shot", "pass", "rebound", "layup",
    "walking", "running", "standing", "sitting",
]
N_CLASSES = len(CLASS_NAMES)
RAW_LABEL_TO_ADJUSTED = {n: i for i, n in enumerate(CLASS_NAMES)}
VOID_RAW_LABEL = "void_class"

# Windowing. sw_length/sw_overlap come from the runs' cfg.txt; sampling_rate does
# NOT -- it is hardcoded at preprocess_data.py:31 and assigned to args after the
# cfg dump, so it never reaches the file. Values below are the confirmed ones.
WIN_LEN = 50            # int(sw_length=1.0 * sampling_rate=50)
STEP = 25               # win_len - int((sw_overlap=50 / 100) * win_len)

BASELINE_DIRS = ("2026-07-17_07-56-30", "2026-07-17_11-15-00", "2026-07-17_11-15-03")
SEED_OF_DIR = {"2026-07-17_07-56-30": 1, "2026-07-17_11-15-00": 2, "2026-07-17_11-15-03": 3}
# Placeholder baseline participants (<id>_<eu|na>), matching the run scripts' --loso_subjects.
EXPECTED_FOLDS = ("4d70_eu", "9bd4_na", "a0da_eu", "b512_na", "ce9d_eu")

FOCUS_CLASSES = ["rebound", "layup", "shot", "walking", "running"]
DECILE_EDGES = np.arange(0.0, 1.01, 0.1)


class HardFail(RuntimeError):
    """Raised on any condition that voids the analysis. Never downgraded to a warning."""


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #

def load_labels(path):
    """
    Read the exported per-sample labels, preserving row order.

    The export is already void-filtered and carries a header with named columns,
    so _loso_common.load_game_labels (header=None, subject at col 3, label at the
    last col) does not apply. Validated rather than assumed: an unexpected label
    string or any surviving void row is a hard fail, not a silent mis-map.
    """
    df = pd.read_csv(path)
    missing = {"subject", "label"} - set(df.columns)
    if missing:
        raise HardFail(f"{path}: missing column(s) {sorted(missing)}; got {list(df.columns)}")

    if (df["label"] == VOID_RAW_LABEL).any():
        n = int((df["label"] == VOID_RAW_LABEL).sum())
        raise HardFail(f"{path}: {n} rows still carry {VOID_RAW_LABEL!r}; expected a void-filtered export")

    unknown = sorted(set(df["label"].unique()) - set(RAW_LABEL_TO_ADJUSTED))
    if unknown:
        raise HardFail(f"{path}: unrecognized label string(s) {unknown}; refusing to guess a mapping")

    df["label_id"] = df["label"].map(RAW_LABEL_TO_ADJUSTED).astype(int)
    return df


def window_subject(label_ids):
    """
    Replicate sliding_window_seconds' index generation for one subject.

    Strict '<' boundary: a window landing exactly on the last sample is dropped,
    as is any tail shorter than one window. Window i spans [i*STEP, i*STEP+WIN_LEN).
    Returns (starts, per-window label matrix of shape (n_windows, WIN_LEN)).
    """
    n = len(label_ids)
    starts = [c for c in range(0, n, STEP) if c < n - WIN_LEN]
    if not starts:
        return np.array([], dtype=int), np.empty((0, WIN_LEN), dtype=int)
    mat = np.stack([label_ids[c:c + WIN_LEN] for c in starts])
    return np.array(starts, dtype=int), mat


def build_candidate_windows(labels_df):
    """Window every candidate subject once; returns {subject: (starts, matrix)}."""
    out = {}
    for subj in sorted(labels_df["subject"].unique()):
        ids = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy()
        out[subj] = window_subject(ids)
    return out


def discover_npz(log_root):
    """
    Locate the three baseline run dirs and their five folds each.

    Only the named BASELINE_DIRS are considered -- augmentation runs and other
    fractions live under the same log_root and must not be swept in.
    """
    found = {}
    for d in BASELINE_DIRS:
        run_dir = os.path.join(log_root, d)
        if not os.path.isdir(run_dir):
            raise HardFail(f"baseline run dir not found: {run_dir}")
        folds = {}
        for fold in EXPECTED_FOLDS:
            p = os.path.join(run_dir, f"preds_{fold}_1.0.npz")
            if not os.path.isfile(p):
                raise HardFail(f"missing baseline file: {p}")
            folds[fold] = p
        found[d] = folds
    return found


def resolve_and_certify(npz_paths, candidates):
    """
    Resolve each fold to its raw-data subject and certify the reconstruction.

    For each fold, brute-force every candidate subject and accept only on
    (window count == len(y_true)) AND byte-identical last-sample labels,
    including dtype. The filename token is tried first purely as a fast path;
    it is never trusted. A fold matching zero or more than one candidate is a
    hard fail -- both mean the reconstruction is not established.

    Returns {fold: (subject, starts, window_matrix, y_true)}.
    """
    seed1 = npz_paths[BASELINE_DIRS[0]]
    resolved = {}

    for fold, path in seed1.items():
        with np.load(path) as z:
            y_true = z["y_true"]

        matches = []
        order = [fold] + [s for s in candidates if s != fold]
        for subj in order:
            starts, mat = candidates[subj]
            if mat.shape[0] != len(y_true):
                continue
            last = mat[:, -1].astype(y_true.dtype)
            if last.dtype != y_true.dtype or not np.array_equal(last, y_true):
                continue
            matches.append(subj)

        if len(matches) != 1:
            raise HardFail(
                f"fold {fold}: expected exactly 1 candidate subject matching on window count + "
                f"byte-identical last-sample labels, found {len(matches)}: {matches}. "
                "Reconstruction not certified; nothing downstream is valid."
            )

        subj = matches[0]
        starts, mat = candidates[subj]
        resolved[fold] = (subj, starts, mat, y_true)

    return resolved


def build_window_table(resolved):
    """
    The seed-invariant per-window table: one row per val window across all folds.

    Columns: fold, subject, window_index, start, end, last_label, majority_label,
    purity, hist_<class> x9. Computed once; predictions are joined on later.
    """
    rows = []
    for fold in sorted(resolved):
        subj, starts, mat, y_true = resolved[fold]
        last = mat[:, -1]

        # per-sample class histogram per window, shape (n_windows, N_CLASSES)
        hist = np.stack([np.bincount(w, minlength=N_CLASSES) for w in mat])

        # purity is w.r.t. the LAST-sample label, per the definition above
        purity = hist[np.arange(len(last)), last] / WIN_LEN

        # majority label; ties broken by later-sample-wins, matching
        # _loso_common.window_majority so the two agree on tied windows
        majority = np.empty(len(last), dtype=int)
        for i, w in enumerate(mat):
            counts = hist[i]
            top = counts.max()
            tied = np.flatnonzero(counts == top)
            if len(tied) == 1:
                majority[i] = tied[0]
            else:
                last_idx = {int(l): j for j, l in enumerate(w)}
                majority[i] = max(tied, key=lambda l: last_idx[int(l)])

        df = pd.DataFrame({
            "fold": fold,
            "subject": subj,
            "window_index": np.arange(len(last)),
            "start": starts,
            "end": starts + WIN_LEN,
            "last_label": last,
            "majority_label": majority,
            "purity": purity,
        })
        for c in range(N_CLASSES):
            df[f"hist_{CLASS_NAMES[c]}"] = hist[:, c]
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def attach_predictions(table, npz_paths, resolved):
    """
    Join each seed's y_pred onto the seed-invariant table as pred_seed<N>.

    Hard-fails on any per-fold count mismatch or any y_true disagreement with the
    certified reconstruction -- the certification is re-asserted per seed rather
    than assumed to carry over from seed 1.
    """
    for run_dir, folds in npz_paths.items():
        seed = SEED_OF_DIR[run_dir]
        col = np.empty(len(table), dtype=int)

        for fold, path in folds.items():
            with np.load(path) as z:
                y_pred, y_true = z["y_pred"], z["y_true"]

            mask = (table["fold"] == fold).to_numpy()
            n_expected = int(mask.sum())
            if len(y_pred) != n_expected:
                raise HardFail(
                    f"{path}: y_pred length {len(y_pred)} != reconstructed window count {n_expected} for fold {fold}"
                )

            certified_last = table.loc[mask, "last_label"].to_numpy()
            if y_true.shape != certified_last.shape or not np.array_equal(y_true, certified_last):
                raise HardFail(
                    f"{path}: y_true is not byte-identical to the reconstructed last-sample labels "
                    f"for fold {fold}. Reconstruction not certified for seed {seed}."
                )
            col[mask] = y_pred

        table[f"pred_seed{seed}"] = col

    return table


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #

def iqr_line(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return "n=0"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"n={v.size:>5}  median={med:.3f}  IQR=[{q1:.3f}, {q3:.3f}]  mean={v.mean():.3f}"


def decile_bar(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return "  (empty)"
    # np.histogram's final bin is already right-closed, so purity == 1.0 lands
    # in [0.9, 1.0] without any manual adjustment.
    counts, _ = np.histogram(v, bins=DECILE_EDGES)
    lines = []
    peak = max(counts.max(), 1)
    for i, c in enumerate(counts):
        lo, hi = DECILE_EDGES[i], DECILE_EDGES[i + 1]
        bar = "#" * int(round(40 * c / peak))
        lines.append(f"    [{lo:.1f},{hi:.1f}{']' if i == len(counts) - 1 else ')'}  {c:>6}  {bar}")
    return "\n".join(lines)


def effect_size(a, b):
    """
    Mann-Whitney U with effect sizes.

    Returns (n_a, n_b, U, p, cles, rank_biserial). cles = P(A > B) with ties at
    half weight -- the probability a randomly drawn A exceeds a randomly drawn B.
    rank_biserial = 2*cles - 1, in [-1, 1]. Both are reported because at these
    group sizes a small p is easy to over-read.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return a.size, b.size, np.nan, np.nan, np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    cles = u / (a.size * b.size)
    return a.size, b.size, u, p, cles, 2 * cles - 1


def split_by_prediction(sub, pred_col, cls_id, rival_id):
    """Split a class's windows into predicted-as-itself / predicted-as-rival / other."""
    pred = sub[pred_col].to_numpy()
    return (
        sub.loc[pred == cls_id, "purity"].to_numpy(),
        sub.loc[pred == rival_id, "purity"].to_numpy(),
        sub.loc[(pred != cls_id) & (pred != rival_id), "purity"].to_numpy(),
    )


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def report_base_rates(table):
    print("\n" + "=" * 100)
    print("[1] PURITY BASE RATES BY CLASS -- seed-invariant, computed once over all val windows")
    print("=" * 100)
    print("These depend only on y_true and the raw per-sample labels. They are a property of the")
    print("windowing, not of any model, and are deliberately NOT broken out per seed.\n")
    print(f"  purity = fraction of a window's {WIN_LEN} samples matching its last-sample label")

    for c, name in enumerate(CLASS_NAMES):
        v = table.loc[table["last_label"] == c, "purity"].to_numpy()
        print(f"\n  {name}  {iqr_line(v)}")
        print(decile_bar(v))


def report_split_by_prediction(table, cls_name, rival_name, seed_cols):
    cls_id = CLASS_NAMES.index(cls_name)
    rival_id = CLASS_NAMES.index(rival_name)
    sub = table[table["last_label"] == cls_id]

    print(f"\n  --- {cls_name} windows split by prediction (rival = {rival_name}) ---")
    print(f"  total {cls_name}-labeled val windows: {len(sub)} "
          f"({len(sub) / len(table) * 100:.2f}% of all {len(table)} windows)")

    for col in seed_cols:
        as_self, as_rival, other = split_by_prediction(sub, col, cls_id, rival_id)
        print(f"\n    {col}")
        print(f"      predicted {cls_name:<9} {iqr_line(as_self)}")
        print(f"      predicted {rival_name:<9} {iqr_line(as_rival)}")
        print(f"      predicted other     {iqr_line(other)}")
        n_a, n_b, u, p, cles, rb = effect_size(as_self, as_rival)
        if np.isnan(p):
            print(f"      Mann-Whitney: not computable (n={n_a} vs {n_b})")
        else:
            print(f"      Mann-Whitney U={u:.1f} p={p:.4g}  |  n={n_a} vs {n_b}  "
                  f"CLES={cles:.3f}  rank-biserial={rb:+.3f}")

    pooled_self = np.concatenate([split_by_prediction(sub, c, cls_id, rival_id)[0] for c in seed_cols])
    pooled_rival = np.concatenate([split_by_prediction(sub, c, cls_id, rival_id)[1] for c in seed_cols])
    print(f"\n    pooled across seeds (CORRELATED replicates over the same {len(sub)} windows -- "
          f"not {3 * len(sub)} independent observations)")
    print(f"      predicted {cls_name:<9} {iqr_line(pooled_self)}")
    print(f"      predicted {rival_name:<9} {iqr_line(pooled_rival)}")
    n_a, n_b, u, p, cles, rb = effect_size(pooled_self, pooled_rival)
    if not np.isnan(p):
        print(f"      Mann-Whitney U={u:.1f} p={p:.4g}  |  n={n_a} vs {n_b}  "
              f"CLES={cles:.3f}  rank-biserial={rb:+.3f}")
        print("      (p is inflated by the replicate structure; read the effect size, not the p)")


def report_composition(table, cls_name, rival_name, seed_cols):
    """The key number: what are these windows actually made of?"""
    cls_id = CLASS_NAMES.index(cls_name)
    rival_id = CLASS_NAMES.index(rival_name)
    sub = table[table["last_label"] == cls_id]
    hist_cols = [f"hist_{n}" for n in CLASS_NAMES]

    print(f"\n  --- {cls_name}-labeled windows predicted {rival_name}: what are they made of? ---")
    for col in seed_cols:
        sel = sub[sub[col] == rival_id]
        if len(sel) == 0:
            print(f"    {col}: none")
            continue
        totals = sel[hist_cols].to_numpy().sum(axis=0)
        modal = int(np.argmax(totals))
        frac = sel[hist_cols].to_numpy() / WIN_LEN
        mean_rival = frac[:, rival_id].mean()
        mean_self = frac[:, cls_id].mean()
        print(f"    {col}  n={len(sel):>4}  modal per-sample true class = {CLASS_NAMES[modal].upper()} "
              f"({totals[modal] / totals.sum() * 100:.1f}% of all samples)")
        print(f"           mean sample fraction: {rival_name}={mean_rival:.3f}  {cls_name}={mean_self:.3f}")
        top = np.argsort(totals)[::-1][:4]
        print("           composition: " + ", ".join(
            f"{CLASS_NAMES[t]}={totals[t] / totals.sum():.3f}" for t in top))


def report_label_rule(table, seed_cols, focus_classes):
    print("\n" + "=" * 100)
    print("[4] LABEL-RULE AGREEMENT -- does y_pred track majority vote or last-sample?")
    print("=" * 100)

    last = table["last_label"].to_numpy()
    maj = table["majority_label"].to_numpy()
    disagree = last != maj
    print(f"\n  windows where the two rules disagree: {disagree.sum()} of {len(table)} "
          f"({disagree.mean() * 100:.2f}%)")

    for col in seed_cols:
        pred = table[col].to_numpy()
        print(f"\n  {col}   overall: agrees with last-sample {np.mean(pred == last) * 100:5.2f}%   "
              f"agrees with majority {np.mean(pred == maj) * 100:5.2f}%")
        d = disagree
        if d.sum():
            print(f"           on the {d.sum()} disagreeing windows: last-sample "
                  f"{np.mean(pred[d] == last[d]) * 100:5.2f}%   majority {np.mean(pred[d] == maj[d]) * 100:5.2f}%")
        for name in focus_classes:
            cid = CLASS_NAMES.index(name)
            m = last == cid
            print(f"           {name:<10} (n={m.sum():>5}): last-sample {np.mean(pred[m] == last[m]) * 100:5.2f}%   "
                  f"majority {np.mean(pred[m] == maj[m]) * 100:5.2f}%")

    print("\n  --- windows that CHANGE class under majority vote ---")
    for name in focus_classes:
        cid = CLASS_NAMES.index(name)
        m = (last == cid) & disagree
        n_changed = int(m.sum())
        n_total = int((last == cid).sum())
        print(f"\n    {name}: {n_changed} of {n_total} would change "
              f"({n_changed / n_total * 100:.1f}%)" if n_total else f"\n    {name}: no windows")
        if n_changed:
            dest = pd.Series(maj[m]).value_counts()
            for lbl, cnt in dest.items():
                print(f"      -> {CLASS_NAMES[lbl]:<10} {cnt:>5} ({cnt / n_changed * 100:5.1f}%)")


def report_support_trade(table):
    """
    The cost side: cleaner labels bought with less data.

    Majority vote strips windows whose last sample lands in a short event but
    whose bulk is something else. For a data-limited class that is a direct loss
    of support, and the trade has to be read as one table.
    """
    print("\n" + "=" * 100)
    print("[4b] SUPPORT TRADE -- per-class window support, last-sample vs majority-vote labeling")
    print("=" * 100)
    print("\n  If a class's support collapses under majority vote, relabeling is not a free fix:")
    print("  it buys purity by discarding the very windows that carry the class.\n")

    last_counts = np.bincount(table["last_label"].to_numpy(), minlength=N_CLASSES)
    maj_counts = np.bincount(table["majority_label"].to_numpy(), minlength=N_CLASSES)

    print(f"  {'class':<12} {'last-sample':>12} {'majority':>12} {'delta':>8} {'change':>9}")
    for c, name in enumerate(CLASS_NAMES):
        lo, mo = int(last_counts[c]), int(maj_counts[c])
        pct = (mo - lo) / lo * 100 if lo else float("nan")
        print(f"  {name:<12} {lo:>12} {mo:>12} {mo - lo:>+8} {pct:>8.1f}%")
    print(f"  {'TOTAL':<12} {int(last_counts.sum()):>12} {int(maj_counts.sum()):>12}")


def report_reconciliation(table, tal_impurity):
    print("\n" + "=" * 100)
    print("[5] RECONCILIATION with the TAL-demotion window-impurity measurement")
    print("=" * 100)

    overall = 1.0 - table["purity"].mean()
    reb = table.loc[table["last_label"] == CLASS_NAMES.index("rebound"), "purity"]
    reb_impurity = 1.0 - reb.mean()
    reb_share = len(reb) / len(table) * 100

    print(f"\n  aggregate mean window impurity (all classes) : {overall:.4f}")
    print(f"  rebound-specific mean window impurity        : {reb_impurity:.4f}  "
          f"(rebound = {len(reb)} windows, {reb_share:.2f}% of {len(table)})")
    if tal_impurity is not None:
        print(f"  previously measured (TAL demotion)           : {tal_impurity:.4f}")
        print("\n  These are compatible, not contradictory: a small aggregate impurity and a high")
        print(f"  rebound-specific impurity coexist because rebound is only {reb_share:.2f}% of windows,")
        print("  so it contributes almost nothing to the aggregate. Neither number overturns the other.")
    else:
        print("\n  previously measured (TAL demotion)           : NOT SUPPLIED (--tal_impurity unset)")
        print("  Pass --tal_impurity <value> to print the side-by-side reconciliation.")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_root", default=os.path.join("logs", "subset_specific", "loso_G", "inceptioncontext"))
    parser.add_argument("--labels", default="labels_export.csv.gz",
                        help="Exported per-sample labels (columns: subject, label), original row order.")
    parser.add_argument("--out_csv", default="window_purity_records.csv",
                        help="Per-window records, one row per val window, seeds as pred_seed<N> columns.")
    parser.add_argument("--tal_impurity", default=None, type=float,
                        help="Previously measured aggregate window impurity from the TAL-demotion "
                             "analysis, for the side-by-side reconciliation line.")
    args = parser.parse_args()

    labels_df = load_labels(args.labels)
    candidates = build_candidate_windows(labels_df)
    npz_paths = discover_npz(args.log_root)
    resolved = resolve_and_certify(npz_paths, candidates)

    table = build_window_table(resolved)

    bad = (table["purity"] < 0.0) | (table["purity"] > 1.0)
    if bad.any():
        raise HardFail(f"{int(bad.sum())} window(s) have purity outside [0, 1]")

    mism = table["last_label"].to_numpy() != table[[f"hist_{n}" for n in CLASS_NAMES]].to_numpy().argmax(axis=1)
    table = attach_predictions(table, npz_paths, resolved)
    seed_cols = [f"pred_seed{SEED_OF_DIR[d]}" for d in BASELINE_DIRS]

    print("=" * 100)
    print("WINDOW PURITY UNDER LAST-SAMPLE LABELING -- loso_G, 5-fold val subset, baseline seeds 1/2/3")
    print("=" * 100)
    for fold in sorted(resolved):
        subj, starts, mat, y_true = resolved[fold]
        print(f"  fold {fold} -> subject {subj}   windows={mat.shape[0]}   "
              f"certified (count + byte-identical last-sample labels, unique match)")
    print(f"\n  total val windows: {len(table)}   window={WIN_LEN} samples, stride={STEP}")
    print(f"  windows where last-sample != modal label: {int(mism.sum())}")

    report_base_rates(table)

    print("\n" + "=" * 100)
    print("[2][3] SPLIT BY PREDICTION AND COMPOSITION -- rebound, then controls")
    print("=" * 100)

    # rival = the class each focus class is most often mispredicted as, derived
    # from seed 1 rather than assumed, so the controls are constructed the same
    # way rebound's split is rather than by hand-picking a comparison class.
    rivals = {}
    for name in FOCUS_CLASSES:
        cid = CLASS_NAMES.index(name)
        sub = table[table["last_label"] == cid]
        pred = sub[seed_cols[0]].to_numpy()
        counts = np.bincount(pred[pred != cid], minlength=N_CLASSES)
        rivals[name] = CLASS_NAMES[int(np.argmax(counts))]

    for name in FOCUS_CLASSES:
        tag = "" if name == "rebound" else "   [control]"
        print(f"\n{'-' * 100}\n{name.upper()}{tag}")
        report_split_by_prediction(table, name, rivals[name], seed_cols)
        report_composition(table, name, rivals[name], seed_cols)

    report_label_rule(table, seed_cols, FOCUS_CLASSES)
    report_support_trade(table)
    report_reconciliation(table, args.tal_impurity)

    table.to_csv(args.out_csv, index=False)
    print(f"\n\nWrote {len(table)} per-window records to {args.out_csv}")
    print("  purity/majority/histogram columns are seed-invariant; predictions are pred_seed1/2/3.")


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
