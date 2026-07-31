#!/usr/bin/env python3
"""
Diagnose the shot-class F1 anomaly by showing, per run, where true-shot
windows get predicted to (recall side) and what predicted-shot windows
actually are (precision side) -- both overall and split by window purity --
then compares those distributions across two or more runs (e.g. weighted vs
unweighted vs sqrt_inverse loss weighting).

Reads purity_per_window.csv (written by analysis/purity_stratified_scoring.py)
from each run folder -- same artifact analysis/per_subject_error_breakdown.py
and analysis/walking_signal_energy.py consume, for consistency. Columns used:
subject, window_index, y_true, y_pred, purity (0-8 ints matching CLASS_NAMES,
same list/order as analysis/pooled_per_class.py's CLASS_NAMES).

Usage (from repo root, hangtime_har env):
    python analysis/shot_confusion.py <run_dir_1> <run_dir_2> [<run_dir_3> ...] \
        [--labels weighted unweighted sqrt] \
        [--input-csv purity_per_window.csv]

Writes shot_confusion.csv into every run folder, plus a combined
shot_confusion_comparison.csv into the last folder passed.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _loso_common import CLASS_NAMES, N_CLASSES, is_pure

TARGET_CLASS = "shot"
TARGET_IDX = CLASS_NAMES.index(TARGET_CLASS)

REQUIRED_COLS = ["subject", "window_index", "y_true", "y_pred", "purity"]


def load_run(run_dir, input_csv):
    path = os.path.join(run_dir, input_csv)
    if not os.path.isfile(path):
        sys.exit(f"Error: {path!r} not found. Run analysis/purity_stratified_scoring.py "
                 "on this folder first.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Error: {path!r} is missing required column(s): {missing}")
    return df, path


def class_distribution(sub_df, label_col):
    n = len(sub_df)
    counts = np.bincount(sub_df[label_col].to_numpy(dtype=int), minlength=N_CLASSES)
    pct = 100.0 * counts / n if n else np.full(N_CLASSES, np.nan)
    return counts, pct


def print_distribution(title, counts, pct, n):
    print(f"\n{title} (n={n})")
    name_w = max(len(n) for n in CLASS_NAMES)
    header = f"{'idx':>3}  {'class':<{name_w}}  {'count':>7}  {'pct':>7}"
    print(header)
    print("-" * len(header))
    order = np.argsort(-counts)
    for i in order:
        p = f"{pct[i]:.2f}%" if n else "nan"
        print(f"{i:>3}  {CLASS_NAMES[i]:<{name_w}}  {counts[i]:>7}  {p:>7}")


def compute_side(df, mask_true_or_pred, label_col_for_dist):
    """mask_true_or_pred: boolean mask selecting the rows of interest (true-shot
    or predicted-shot). label_col_for_dist: which column's distribution to report
    ('y_pred' for the recall side, 'y_true' for the precision side)."""
    sub = df[mask_true_or_pred]
    counts, pct = class_distribution(sub, label_col_for_dist)
    return sub, counts, pct


def build_run_table(run_label, df):
    """Returns (rows_for_csv, headline_dict) for one run."""
    rows = []

    true_shot = df[df["y_true"] == TARGET_IDX]
    pred_shot = df[df["y_pred"] == TARGET_IDX]
    n_true_shot = len(true_shot)
    n_pred_shot = len(pred_shot)

    tp = int(((df["y_true"] == TARGET_IDX) & (df["y_pred"] == TARGET_IDX)).sum())
    recall = tp / n_true_shot if n_true_shot else float("nan")
    precision = tp / n_pred_shot if n_pred_shot else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (n_true_shot and n_pred_shot and (precision + recall) > 0) else float("nan"))

    print("\n" + "=" * 100)
    print(f"RUN: {run_label}  ({TARGET_CLASS} support={n_true_shot}, predicted-{TARGET_CLASS} volume={n_pred_shot})")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print("=" * 100)

    strata = [("all", df), ("pure", df[df["purity"].apply(is_pure)]), ("mixed", df[~df["purity"].apply(is_pure)])]

    for stratum_name, stratum_df in strata:
        ts = stratum_df[stratum_df["y_true"] == TARGET_IDX]
        counts, pct = class_distribution(ts, "y_pred")
        print_distribution(f"[{stratum_name}] RECALL SIDE -- true-{TARGET_CLASS} windows, predicted-class distribution",
                            counts, pct, len(ts))
        for i in range(N_CLASSES):
            rows.append({"side": "recall", "stratum": stratum_name, "class": CLASS_NAMES[i],
                         "count": int(counts[i]) if len(ts) else np.nan,
                         "pct": pct[i], "value": np.nan})

        ps = stratum_df[stratum_df["y_pred"] == TARGET_IDX]
        counts, pct = class_distribution(ps, "y_true")
        print_distribution(f"[{stratum_name}] PRECISION SIDE -- predicted-{TARGET_CLASS} windows, true-class distribution",
                            counts, pct, len(ps))
        for i in range(N_CLASSES):
            rows.append({"side": "precision", "stratum": stratum_name, "class": CLASS_NAMES[i],
                         "count": int(counts[i]) if len(ps) else np.nan,
                         "pct": pct[i], "value": np.nan})

    for metric, val in [("precision", precision), ("recall", recall), ("f1", f1),
                         ("support_true_shot", n_true_shot), ("volume_pred_shot", n_pred_shot)]:
        rows.append({"side": "summary", "stratum": "all", "class": metric,
                     "count": np.nan, "pct": np.nan, "value": val})

    headline = {"run": run_label, "support_true_shot": n_true_shot, "volume_pred_shot": n_pred_shot,
                "precision": precision, "recall": recall, "f1": f1}
    return rows, headline


def verify_cross_run_alignment(run_data):
    """run_data: list of (label, df). Asserts every run has the exact same set
    of (subject, window_index) keys with the exact same y_true -- same held-out
    test windows, only predictions differ. Hard-fails with specifics otherwise."""
    label0, df0 = run_data[0]
    ref = df0.set_index(["subject", "window_index"])["y_true"]
    problems = []
    for label, df in run_data[1:]:
        cur = df.set_index(["subject", "window_index"])["y_true"]
        only_ref = ref.index.difference(cur.index)
        only_cur = cur.index.difference(ref.index)
        if len(only_ref):
            problems.append(f"  windows in {label0!r} but missing from {label!r}: {len(only_ref)} "
                             f"(e.g. {list(only_ref[:5])})")
        if len(only_cur):
            problems.append(f"  windows in {label!r} but missing from {label0!r}: {len(only_cur)} "
                             f"(e.g. {list(only_cur[:5])})")
        common = ref.index.intersection(cur.index)
        mismatched = common[ref.loc[common].to_numpy() != cur.loc[common].to_numpy()]
        if len(mismatched):
            problems.append(f"  {len(mismatched)} window(s) common to {label0!r} and {label!r} but with "
                             f"different y_true (e.g. {list(mismatched[:5])})")
    if problems:
        sys.exit("Error: runs do not share the exact same test windows/labels -- this is a data bug, "
                 "not a modeling result. Stopping before scoring.\n" + "\n".join(problems))
    print(f"Cross-run alignment check passed: all {len(run_data)} runs share the identical "
          f"{len(ref)} (subject, window_index) -> y_true mapping.")


def print_comparison(side, run_labels, comp_df):
    print(f"\n{'='*100}\nCOMPARISON -- {side.upper()} SIDE across runs (true-{TARGET_CLASS}->class %, "
          f"sorted by max-min range descending)\n{'='*100}")
    name_w = max(len(c) for c in CLASS_NAMES)
    lab_w = max(max(len(l) for l in run_labels), 8)
    header = f"{'class':<{name_w}}  " + "  ".join(f"{l:>{lab_w}}" for l in run_labels) + f"  {'range':>8}"
    print(header)
    print("-" * len(header))
    for _, row in comp_df.iterrows():
        vals = "  ".join(f"{row[l]:>{lab_w}.2f}" for l in run_labels)
        print(f"{row['class']:<{name_w}}  {vals}  {row['range']:>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description=f"Diagnose {TARGET_CLASS}-class confusion across two or more runs."
    )
    parser.add_argument("run_dirs", nargs="+", help="Two or more run folders (each with purity_per_window.csv)")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Friendly labels for each run_dir, same count/order (default: folder basenames)")
    parser.add_argument("--input-csv", default="purity_per_window.csv",
                         help="Filename to read from each run folder (default: purity_per_window.csv)")
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("at least two run_dirs are required to compare.")
    for d in args.run_dirs:
        if not os.path.isdir(d):
            sys.exit(f"Error: {d!r} is not a directory.")

    if args.labels is not None:
        if len(args.labels) != len(args.run_dirs):
            parser.error(f"--labels count ({len(args.labels)}) must match run_dirs count ({len(args.run_dirs)}).")
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
        if len(set(labels)) != len(labels):
            parser.error(f"folder basenames are not unique ({labels}); pass --labels to disambiguate.")

    run_data = []
    for label, d in zip(labels, args.run_dirs):
        df, path = load_run(d, args.input_csv)
        print(f"Loaded {len(df)} windows for run {label!r} from: {path}")
        run_data.append((label, df))

    verify_cross_run_alignment(run_data)

    headlines = []
    recall_tables = {}
    precision_tables = {}
    for (label, df), run_dir in zip(run_data, args.run_dirs):
        rows, headline = build_run_table(label, df)
        headlines.append(headline)
        run_df = pd.DataFrame(rows)
        recall_tables[label] = run_df[(run_df["side"] == "recall") & (run_df["stratum"] == "all")].set_index("class")["pct"]
        precision_tables[label] = run_df[(run_df["side"] == "precision") & (run_df["stratum"] == "all")].set_index("class")["pct"]
        try:
            out_path = os.path.join(run_dir, "shot_confusion.csv")
            run_df.to_csv(out_path, index=False)
            print(f"\nWrote shot_confusion.csv to {run_dir}")
        except Exception as e:
            print(f"\n(non-fatal) failed to write shot_confusion.csv for {label}: {e}")

    print("\n" + "=" * 100)
    print("HEADLINE PRECISION/RECALL/F1 ACROSS RUNS")
    print("=" * 100)
    hdf = pd.DataFrame(headlines)
    print(hdf.to_string(index=False))

    def build_comparison(tables):
        comp = pd.DataFrame({label: series for label, series in tables.items()})
        comp = comp.reindex(CLASS_NAMES)
        comp["range"] = comp.max(axis=1) - comp.min(axis=1)
        comp = comp.sort_values("range", ascending=False)
        comp.index.name = "class"
        return comp.reset_index()

    recall_comp = build_comparison(recall_tables)
    precision_comp = build_comparison(precision_tables)

    print_comparison("recall", labels, recall_comp)
    print_comparison("precision", labels, precision_comp)

    last_dir = args.run_dirs[-1]
    try:
        recall_comp.insert(0, "side", "recall")
        precision_comp.insert(0, "side", "precision")
        combined = pd.concat([recall_comp, precision_comp], ignore_index=True)
        combined.to_csv(os.path.join(last_dir, "shot_confusion_comparison.csv"), index=False)
        print(f"\nWrote shot_confusion_comparison.csv to {last_dir}")
    except Exception as e:
        print(f"\n(non-fatal) failed to write shot_confusion_comparison.csv: {e}")


if __name__ == "__main__":
    main()
