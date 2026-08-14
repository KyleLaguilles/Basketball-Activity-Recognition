#!/usr/bin/env python3
"""
Shared reconstruction + alignment code for LOSO prediction re-analysis scripts
(analysis/relabel_majority_vote.py, analysis/purity_stratified_scoring.py).

Not a standalone script -- import from it.

Reconstruction is based on (test_case=loso_G, i.e. subset_specific LOSO
trained/evaluated on game data only):
    - data_processing/sliding_window.py: sliding_window_seconds,
      apply_sliding_window (window=50 samples, step=25 samples @ 1s/50Hz/50%,
      last-sample labeling, per-subject grouping in original row order,
      strict '<' boundary that drops any final window landing exactly on
      the last sample as well as any leftover tail shorter than one window)
    - data_processing/preprocess_data.py: preprocess_data (drops rows with
      raw label == 'void_class' *before* windowing), adjust_labels (string
      -> int mapping, shifted by -1 since has_void=True/include_void=False)

Column layout of predictions_best_*.csv (after index_col=0), same
convention as analysis/pooled_per_class.py:
    iloc[:, 0]  ->  y_pred
    iloc[:, 1]  ->  y_true
"""

import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

CLASS_NAMES = [
    "dribbling", "shot", "pass", "rebound", "layup",
    "walking", "running", "standing", "sitting",
]
N_CLASSES = len(CLASS_NAMES)
EXPECTED_SUBJECTS = 14

VOID_RAW_LABEL = "void_class"
RAW_LABEL_TO_ADJUSTED = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PURITY_EPS = 1e-9


def is_pure(purity):
    """A window is 'pure' if its majority class covers the whole window (purity == 1.0)."""
    return purity >= 1.0 - PURITY_EPS


def parse_subjects(paths):
    """
    Recover subject IDs from filenames: predictions_best_{subject}_{runname}.csv
    (identical logic to analysis/pooled_per_class.py's parse_subjects.)
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    bare = [s[len("predictions_best_"):] for s in stems]
    parts_list = [b.split("_") for b in bare]
    n_min = min(len(p) for p in parts_list)

    suffix_len = 0
    for i in range(1, n_min):
        tail = "_".join(parts_list[0][-i:])
        if all("_".join(p[-i:]) == tail for p in parts_list):
            suffix_len = i
        else:
            break

    if suffix_len == 0:
        subjects = ["_".join(p) for p in parts_list]
    else:
        subjects = ["_".join(p[:-suffix_len]) for p in parts_list]

    if any(s == "" for s in subjects) or len(set(subjects)) != len(subjects):
        return None
    return subjects


def load_game_labels(game_csv_path):
    """
    Load subject id + adjusted int label columns from hangtime_game_data.csv,
    preserving original row order, replicating preprocess_data.preprocess_data's
    void-row drop (has_void=True, include_void=False) and adjust_labels mapping.

    :return: pandas DataFrame with columns ['subject', 'label'] (label is the
        adjusted 0-8 int matching CLASS_NAMES / predictions_best_*.csv y_true)
    """
    if not os.path.isfile(game_csv_path):
        sys.exit(f"Error: game data CSV not found at {game_csv_path!r}")

    ncols = pd.read_csv(game_csv_path, header=None, nrows=1).shape[1]
    subject_col, label_col = 3, ncols - 1

    df = pd.read_csv(
        game_csv_path, header=None, index_col=None,
        usecols=[subject_col, label_col],
        dtype={subject_col: str, label_col: str},
    )
    df = df.rename(columns={subject_col: "subject", label_col: "label"})

    df = df[df["label"] != VOID_RAW_LABEL].reset_index(drop=True)

    unknown = sorted(set(df["label"].unique()) - set(RAW_LABEL_TO_ADJUSTED))
    if unknown:
        sys.exit(
            "Error: found raw label string(s) in game CSV not recognized by "
            f"adjust_labels' mapping (after dropping {VOID_RAW_LABEL!r} rows): "
            f"{unknown}. Refusing to guess -- update RAW_LABEL_TO_ADJUSTED if "
            "these are legitimate classes."
        )

    df["label"] = df["label"].map(RAW_LABEL_TO_ADJUSTED).astype(int)
    return df


def load_game_features(game_csv_path):
    """
    Like load_game_labels, but also returns the raw accelerometer feature
    columns sitting between the subject column and the label column.

    Column layout confirmed from data_processing/data_creation.py's final
    column selection for hangtime_*_data.csv:
        ['location', 'skill', 'gender', 'subject', 'acc_x', 'acc_y', 'acc_z', 'basketball']
    i.e. exactly 3 feature columns (acc_x, acc_y, acc_z) between subject (col 3)
    and the label (last column).

    :return: pandas DataFrame with columns ['subject', 'label', 'acc_x', 'acc_y', 'acc_z']
        (same row order/void-filtering/label-adjustment as load_game_labels)
    """
    if not os.path.isfile(game_csv_path):
        sys.exit(f"Error: game data CSV not found at {game_csv_path!r}")

    ncols = pd.read_csv(game_csv_path, header=None, nrows=1).shape[1]
    subject_col, label_col = 3, ncols - 1
    feature_cols = list(range(subject_col + 1, label_col))
    if len(feature_cols) != 3:
        sys.exit(
            f"Error: expected exactly 3 feature columns (acc_x, acc_y, acc_z) between "
            f"the subject column ({subject_col}) and label column ({label_col}), found "
            f"{len(feature_cols)}: {feature_cols}. Column layout doesn't match "
            "data_processing/data_creation.py's expected "
            "['location','skill','gender','subject','acc_x','acc_y','acc_z','basketball']."
        )

    usecols = [subject_col] + feature_cols + [label_col]
    df = pd.read_csv(
        game_csv_path, header=None, index_col=None,
        usecols=usecols,
        dtype={subject_col: str, label_col: str},
    )
    df = df.rename(columns={
        subject_col: "subject", label_col: "label",
        feature_cols[0]: "acc_x", feature_cols[1]: "acc_y", feature_cols[2]: "acc_z",
    })

    df = df[df["label"] != VOID_RAW_LABEL].reset_index(drop=True)

    unknown = sorted(set(df["label"].unique()) - set(RAW_LABEL_TO_ADJUSTED))
    if unknown:
        sys.exit(
            "Error: found raw label string(s) in game CSV not recognized by "
            f"adjust_labels' mapping (after dropping {VOID_RAW_LABEL!r} rows): "
            f"{unknown}. Refusing to guess -- update RAW_LABEL_TO_ADJUSTED if "
            "these are legitimate classes."
        )

    df["label"] = df["label"].map(RAW_LABEL_TO_ADJUSTED).astype(int)
    for c in ("acc_x", "acc_y", "acc_z"):
        df[c] = df[c].astype(float)
    return df


def make_windows(labels, win_len, step):
    """
    Replicates sliding_window.sliding_window_seconds' index generation exactly,
    including its strict '<' boundary (drops a window landing exactly on the
    last sample, and any leftover tail shorter than one window).
    """
    n = len(labels)
    windows = []
    curr = 0
    while curr < n - win_len:
        windows.append(labels[curr:curr + win_len])
        curr += step
    return windows


def window_majority(window):
    """Most frequent label and its count; ties broken by later-sample-wins."""
    counts = Counter()
    last_idx = {}
    for idx, lbl in enumerate(window):
        lbl = int(lbl)
        counts[lbl] += 1
        last_idx[lbl] = idx
    max_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == max_count]
    label = tied[0] if len(tied) == 1 else max(tied, key=lambda l: last_idx[l])
    return label, max_count


def majority_vote_label(window):
    """Most frequent label in the window; ties broken by later-sample-wins."""
    label, _ = window_majority(window)
    return label


def print_label_matrix(matrix, class_names=CLASS_NAMES):
    """Pretty-print a square int matrix with class-name row/col headers."""
    name_w = max(len(n) for n in class_names)
    col_w = max(name_w, 6)
    print(" " * (name_w + 2) + "".join(f"{class_names[j]:>{col_w + 2}}" for j in range(len(class_names))))
    for i in range(len(class_names)):
        row_str = "".join(f"{matrix[i, j]:>{col_w + 2}}" for j in range(len(class_names)))
        print(f"{class_names[i]:<{name_w}}  {row_str}")


def load_and_align_predictions(log_dir, game_csv, sw_length=1.0, sw_overlap=50, sampling_rate=50):
    """
    Full alignment pipeline shared by all LOSO re-analysis scripts:
      1. find predictions_best_*.csv in log_dir, parse filename subjects
      2. load + void-filter + adjust game data labels
      3. build per-candidate-subject windows (win_len/step from sw args)
      4. resolve each file to its true subject: filename-first, brute-force
         fallback across all candidates, accepted only if window count AND
         every row's last-sample label match y_true exactly
      5. verify the resolution is a bijection over all candidate subjects
      6. print a status report

    Exits the process (sys.exit(1)) if any file fails alignment or the
    bijection check fails -- callers should not attempt to score after that.

    :return: (resolutions, candidate_windows, candidate_subjects, win_len, step)
        resolutions: list of dicts with keys
            file, filename_subject, resolved_subject, y_pred, y_true, windows
            (windows is the ordered list of length-win_len label arrays for
            the resolved subject; windows[i] corresponds to row i of y_true/y_pred)
        candidate_windows: dict subject -> list of label-window arrays
        candidate_subjects: sorted list of subject ids found in game_csv
    """
    win_len = int(sw_length * sampling_rate)
    step = win_len - int((sw_overlap / 100) * win_len)

    pattern = os.path.join(log_dir, "predictions_best_*.csv")
    paths = sorted(
        p for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("predictions_last_")
    )
    print(f"Found {len(paths)} predictions_best_*.csv file(s) in: {log_dir}")
    if len(paths) != EXPECTED_SUBJECTS:
        print(f"WARNING: expected {EXPECTED_SUBJECTS} files, got {len(paths)}.")

    filename_subjects = parse_subjects(paths)
    if filename_subjects is None:
        sys.exit("Error: could not unambiguously parse subject IDs from filenames.")

    print(f"\nLoading game data labels from: {game_csv}")
    game_df = load_game_labels(game_csv)
    candidate_subjects = sorted(game_df["subject"].unique().tolist())
    print(f"Found {len(candidate_subjects)} candidate subject(s) in game data: {candidate_subjects}")
    if len(candidate_subjects) != EXPECTED_SUBJECTS:
        print(f"WARNING: expected {EXPECTED_SUBJECTS} candidate subjects, got {len(candidate_subjects)}.")

    candidate_windows = {}
    candidate_last_labels = {}
    for subj in candidate_subjects:
        labels = game_df.loc[game_df["subject"] == subj, "label"].to_numpy()
        windows = make_windows(labels, win_len, step)
        candidate_windows[subj] = windows
        candidate_last_labels[subj] = np.array([w[-1] for w in windows], dtype=int)

    print("\nResolving prediction files to raw-data subjects (window count + "
          "row-wise last-sample-label match against y_true)...")
    resolutions = []
    failures = []

    for path, fname_subj in zip(paths, filename_subjects):
        df_pred = pd.read_csv(path, index_col=0)
        y_pred = df_pred.iloc[:, 0].to_numpy().astype(int)
        y_true = df_pred.iloc[:, 1].to_numpy().astype(int)

        ordered_candidates = [fname_subj] + [s for s in candidate_subjects if s != fname_subj]
        resolved = None
        for cand in ordered_candidates:
            if cand not in candidate_windows:
                continue
            last_labels = candidate_last_labels[cand]
            if len(last_labels) != len(y_true):
                continue
            if not np.array_equal(last_labels, y_true):
                continue
            resolved = cand
            break

        if resolved is None:
            failures.append({"file": path, "filename_subject": fname_subj,
                              "n_pred_rows": len(y_true)})
            continue

        resolutions.append({
            "file": path, "filename_subject": fname_subj, "resolved_subject": resolved,
            "y_pred": y_pred, "y_true": y_true, "windows": candidate_windows[resolved],
        })

    name_w = max(len(os.path.basename(p)) for p in paths)
    subj_w = max(len(s) for s in candidate_subjects)
    print(f"\n{'file':<{name_w}}  {'filename_subj':<{subj_w}}  {'resolved_subj':<{subj_w}}  status")
    print("-" * (name_w + subj_w * 2 + 30))
    for r in resolutions:
        mismatch = " (MISMATCH -- filename subject was wrong)" if r["filename_subject"] != r["resolved_subject"] else ""
        print(f"{os.path.basename(r['file']):<{name_w}}  {r['filename_subject']:<{subj_w}}  "
              f"{r['resolved_subject']:<{subj_w}}  OK{mismatch}")
    for f in failures:
        print(f"{os.path.basename(f['file']):<{name_w}}  {f['filename_subject']:<{subj_w}}  "
              f"{'--':<{subj_w}}  FAILED (n_pred_rows={f['n_pred_rows']}, no candidate subject "
              f"reconstructed matching window count + last-sample labels)")

    if failures:
        print(f"\n{len(failures)} file(s) failed alignment. Stopping -- not scoring.")
        sys.exit(1)

    resolved_list = [r["resolved_subject"] for r in resolutions]
    counts = Counter(resolved_list)
    dupes = sorted(s for s, c in counts.items() if c > 1)
    missing = sorted(set(candidate_subjects) - set(resolved_list))
    if dupes or missing or len(resolved_list) != len(candidate_subjects):
        print("\nError: resolved subject mapping is not a bijection over candidate subjects.")
        if dupes:
            print(f"  Duplicated resolutions (used by >1 file): {dupes}")
        if missing:
            print(f"  Candidate subjects never resolved by any file: {missing}")
        print("Stopping -- not scoring.")
        sys.exit(1)
    print(f"\nBijection check passed: {len(resolutions)} files <-> {len(candidate_subjects)} "
          "candidate subjects, one-to-one.")

    n_mismatched = sum(1 for r in resolutions if r["filename_subject"] != r["resolved_subject"])
    if n_mismatched:
        print(f"NOTE: {n_mismatched} file(s) had a filename subject token that did not match "
              "its correct raw-data subject (see MISMATCH rows above) -- this reflects the "
              "args.subjects / LabelEncoder ordering mismatch in validation.py, not a data error.")

    return resolutions, candidate_windows, candidate_subjects, win_len, step


def per_class_metrics(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=np.nan,
    )
    support = np.bincount(y_true, minlength=N_CLASSES)
    return prec, rec, f1, support
