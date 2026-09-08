#!/usr/bin/env python3
"""
Sample-level F1 at any window length -- the primary cross-config metric for the
window-length sweep.

WHY NOT WINDOW-LEVEL F1. A window-level score is computed over a population whose
size and label purity are both functions of the window length: at 1s the five LOSO
folds yield 24583 windows, at 3s only 8189, and the last-sample labeling rule
([sliding_window.py:117] output_y = [[i[-1]] ...]) makes a long window's label an
ever-thinner summary of what the window actually contains. Two window-level F1s
computed at different sw_length are therefore not the same measurement, and their
difference is not an effect. Sample-level F1 puts every config back on one fixed
denominator -- the raw 50 Hz timeline -- by expanding each window's prediction over
the timesteps it covers and scoring against the original per-sample ground truth.

Window-level F1 is still printed, per config, from the same npz files. It is
DIAGNOSTIC ONLY: read it against its own config's sample-level number to see how far
the two rules diverge as windows lengthen, never across configs as a result.

EXPANSION RULE -- last-window-wins. Overlapping windows disagree; later windows
overwrite earlier ones, so each sample takes the prediction of the last window that
covers it. This is the rule that agrees with last-sample labeling: a window's
prediction is a statement about its final sample, so it should carry furthest at its
own right edge. Trailing samples past the final window (at most `step` of them, by
construction) are uncovered and excluded from scoring; the coverage fraction is
reported per config and is >= 0.9997 at every swept length.

WINDOWING IS DERIVED FROM cfg.txt, NEVER FROM window_purity.py. That module pins
WIN_LEN=50/STEP=25 as literals (window_purity.py:62-63) and this script must not
inherit them. Both constants are recomputed per results_dir:
    win_len = int(sw_length * SAMPLING_RATE)
    step    = win_len - int((sw_overlap / 100) * win_len)
The step formula is sliding_window_seconds' own (sliding_window.py:31,38), NOT
win_len // 2. They differ: at sw_length=1.5, win_len=75 and int(0.5 * 75) = 37, so
step = 38, not 37. A step of 37 desynchronizes every window at 1.5s.

WINDOW COUNT. sliding_window_seconds' boundary is a strict '<' (sliding_window.py:35),
which drops a window landing exactly on the last sample. The count is therefore
    ceil((n_samples - win_len) / step)
not (n_samples - win_len) // step + 1; the two differ by one whenever step divides
(n_samples - win_len) exactly. The closed form is cross-checked against
_loso_common.make_windows -- the literal replication of the pipeline's index
generation -- for every candidate subject, and a disagreement is a hard fail.

CERTIFICATION. Folds are resolved to raw-data subjects by brute force over all
candidates and accepted only on window-count + exact last-sample-label equality,
never on the filename token: preprocess_data.py:47-52 fits the LabelEncoder on a
sorted subject list while validation.py:325 names the npz via the UNSORTED
args.subjects array, so the token can disagree with the true subject
(_loso_common.py:337-341). That resolution IS the label-encoding check -- the raw CSV
carries strings mapped through RAW_LABEL_TO_ADJUSTED while the npz carries the
pipeline's ints, and an encoding mismatch cannot survive an exact label-sequence
match over thousands of windows.

Read-only and CPU-only. Imports nothing from src/, loads no checkpoints, writes no
files.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/sample_level_f1.py \
        --results_dir logs/.../2026-07-17_07-56-30 \
        --results_dir logs/.../<ts>_window_1.5s_seed1 \
        --labels labels_export.csv.gz

    # equivalently, one flag with many values
    python analysis/sample_level_f1.py --results_dir dirA dirB dirC
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_purity as wp      # noqa: E402  -- module-level defs, __main__-guarded
import _loso_common as lc       # noqa: E402

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
ALL_LABELS = list(range(N_CLASSES))
HardFail = wp.HardFail

# Hardcoded at preprocess_data.py:31 and assigned to args.sampling_rate at main.py:176,
# i.e. AFTER the cfg dump at main.py:158 -- so it never reaches cfg.txt and cannot be
# read from it. Confirmed absent from every run's cfg.
SAMPLING_RATE = 50

EXPECTED_FOLD_COUNT = 5
ALL_SUBJECT_FOLD_COUNT = 14      # a --dense run with --loso_subjects unset covers every subject
UNCOVERED = -1

# Recorded window-level anchors, verbatim from grid_read.py:61-62. Printed for
# comparison only -- never asserted here, since a swept config is not the baseline.
ANCHOR_WINDOW_SEED1 = {"rebound": 0.2053, "layup": 0.3958, "walking": 0.7926, "macro": 0.5273}
ANCHOR_WINDOW_MEAN123 = {"rebound": 0.1969, "walking": 0.7741}

BASELINE_SW_LENGTH = 1.0

TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?")


def fail(msg):
    raise HardFail(msg)


# --------------------------------------------------------------------------- #
# cfg + windowing
# --------------------------------------------------------------------------- #

def load_cfg(results_dir):
    """cfg.txt is json.dump(vars(args), indent=2) -- main.py:158-160. Flat JSON."""
    path = os.path.join(results_dir, "cfg.txt")
    if not os.path.isfile(path):
        fail(f"cfg.txt not found in {results_dir}. Cannot derive the windowing; refusing to assume it.")
    with open(path) as fid:
        return json.load(fid)


def derive_windowing(cfg, results_dir):
    """
    Recompute win_len/step exactly as apply_sliding_window does for this run.

    sw_unit is asserted rather than assumed: win_len = int(sw_length * sampling_rate)
    is only the 'seconds' branch (sliding_window.py:29). Under 'units', sw_length IS
    the window in samples and the sampling rate does not enter, so silently applying
    the seconds formula would scale every window by 50.
    """
    for key in ("sw_length", "sw_unit", "sw_overlap"):
        if key not in cfg:
            fail(f"{results_dir}/cfg.txt has no {key!r} key; got {sorted(cfg)}")

    if cfg["sw_unit"] != "seconds":
        fail(f"{results_dir}: sw_unit is {cfg['sw_unit']!r}, expected 'seconds'. "
             "win_len = int(sw_length * sampling_rate) does not apply to any other unit.")

    sw_length = float(cfg["sw_length"])
    sw_overlap = int(cfg["sw_overlap"])

    win_len = int(sw_length * SAMPLING_RATE)
    # sliding_window_seconds:31,38 -- NOT win_len // 2. int() truncates, so an odd
    # win_len (75 at 1.5s) steps by 38, not 37.
    step = win_len - int((sw_overlap / 100) * win_len)

    if win_len <= 0:
        fail(f"{results_dir}: sw_length={sw_length} gives win_len={win_len}")
    if not 0 < step <= win_len:
        fail(f"{results_dir}: sw_overlap={sw_overlap} gives step={step}, which is not in (0, {win_len}]")

    return sw_length, sw_overlap, win_len, step


def expected_window_count(n_samples, win_len, step):
    """
    Window count under sliding_window_seconds' strict '<' boundary.

    `while curr < len(data) - win_len` (sliding_window.py:35) admits every start
    i*step with i*step < n - win_len, so the count is ceil((n - win_len) / step) --
    one fewer than the inclusive (n - win_len) // step + 1 whenever step divides
    (n - win_len) exactly.
    """
    if n_samples <= win_len:
        return 0
    return -(-(n_samples - win_len) // step)


def build_candidates(labels_df, win_len, step):
    """
    Window every candidate subject at this (win_len, step); returns
    {subject: {n_samples, n_windows, last}}.

    make_windows is _loso_common's literal replication of the pipeline's index
    generation, so its count is the reference. The closed form above is checked
    against it per subject -- that is what pins the strict-'<' off-by-one rather than
    trusting either derivation alone.
    """
    out = {}
    for subj in sorted(labels_df["subject"].unique()):
        ids = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy()
        windows = lc.make_windows(ids, win_len, step)

        n_closed = expected_window_count(len(ids), win_len, step)
        if len(windows) != n_closed:
            fail(f"subject {subj}: _loso_common.make_windows produced {len(windows)} windows at "
                 f"win_len={win_len} step={step}, closed form ceil((n - win_len)/step) says "
                 f"{n_closed} (n_samples={len(ids)}). The window-count derivation is wrong; "
                 "nothing downstream is valid.")

        if windows:
            last = np.array([w[-1] for w in windows], dtype=ids.dtype)
        else:
            last = np.empty(0, dtype=ids.dtype)

        out[subj] = {"n_samples": int(len(ids)), "n_windows": len(windows), "last": last}
    return out


# --------------------------------------------------------------------------- #
# npz discovery + certification
# --------------------------------------------------------------------------- #

def discover_fold_npz(results_dir, folds, pattern=None, expect_count=EXPECTED_FOLD_COUNT):
    """
    Map fold token -> npz path, tolerating both of validation.py's naming schemes.

    The three baseline dirs predate the seed tag (git 63be22f) and are named
    `preds_<fold>_<fraction>.npz`; everything saved after it, the window sweep
    included, is `preds_<fold>_<fraction>_seed<N>.npz` (validation.py:324-325).
    args.name never enters the npz filename -- it only names the log dir
    (main.py:152) -- so --name window_1.5s_seed1 changes the DIRECTORY, not these
    files. Augmentation runs carry a further `_aug<mult><recipe>` suffix and are
    excluded: they are a different model and sweeping one in would mix conditions
    silently.

    A fold resolving to zero or several files is a hard fail; --npz_pattern (which
    must contain '{fold}') resolves the ambiguity explicitly rather than by guess.
    """
    if pattern and "{fold}" not in pattern:
        fail(f"--npz_pattern must contain '{{fold}}', got {pattern!r}")

    found = {}
    for fold in folds:
        if pattern:
            hits = sorted(glob.glob(os.path.join(results_dir, pattern.format(fold=fold))))
        else:
            hits = sorted(glob.glob(os.path.join(results_dir, f"preds_{fold}_*.npz")))
            hits = [h for h in hits if "_aug" not in os.path.basename(h)]
        if len(hits) != 1:
            fail(f"{results_dir} fold {fold}: expected exactly 1 npz, found {len(hits)}: "
                 f"{[os.path.basename(h) for h in hits]}. Pass --npz_pattern to disambiguate.")
        found[fold] = hits[0]

    if len(found) != expect_count:
        fail(f"{results_dir}: resolved {len(found)} folds, expected exactly {expect_count}: "
             f"{sorted(found)}")
    return found


def load_npz(path):
    with np.load(path) as z:
        for key in ("y_pred", "y_true"):
            if key not in z:
                fail(f"{path}: missing array {key!r}; has {sorted(z.files)}")
        y_pred = z["y_pred"]
        y_true = z["y_true"]

    for name, arr in (("y_pred", y_pred), ("y_true", y_true)):
        if arr.ndim != 1:
            fail(f"{path}: {name} has shape {arr.shape}, expected 1-D")
        if arr.dtype.kind not in "iu":
            fail(f"{path}: {name} has dtype {arr.dtype}, expected an integer dtype")
        if arr.size and (arr.min() < 0 or arr.max() >= N_CLASSES):
            fail(f"{path}: {name} ranges [{arr.min()}, {arr.max()}], outside the "
                 f"{N_CLASSES}-class id space [0, {N_CLASSES - 1}]")

    if y_pred.shape != y_true.shape:
        fail(f"{path}: y_pred {y_pred.shape} != y_true {y_true.shape}")
    return y_pred.astype(np.int64), y_true.astype(np.int64)


def resolve_fold(fold, y_true, candidates, results_dir, win_len, step):
    """
    Resolve one fold to its raw-data subject and certify the reconstruction.

    Accepts a candidate only on (window count == len(y_true)) AND exact equality of
    the reconstructed last-sample label sequence with y_true. The filename token is
    tried first purely as a fast path and is never trusted. Zero or several matches
    is a hard fail -- both mean the reconstruction is not established, and with it
    neither the subject's raw labels nor the shared label encoding.
    """
    order = [fold] + [s for s in candidates if s != fold]
    matches = []
    for subj in order:
        cand = candidates[subj]
        if cand["n_windows"] != len(y_true):
            continue
        if not np.array_equal(cand["last"].astype(np.int64), y_true):
            continue
        matches.append(subj)

    if len(matches) != 1:
        near = sorted(s for s in candidates if candidates[s]["n_windows"] == len(y_true))
        fail(f"{results_dir} fold {fold} (win_len={win_len}, step={step}, "
             f"{len(y_true)} windows): expected exactly 1 candidate subject matching on window "
             f"count + exact last-sample labels, found {len(matches)}: {matches}. "
             f"Subjects matching on count alone: {near}. Reconstruction not certified.")
    return matches[0]


# --------------------------------------------------------------------------- #
# expansion
# --------------------------------------------------------------------------- #

def expand_last_window_wins(y_pred_win, n_samples, win_len, step):
    """
    Expand window predictions onto the raw timeline; later windows overwrite earlier.

    Window i covers [i*step, i*step + win_len). Assigning in ascending i makes the
    LAST window covering a sample the one that sets it. Samples past the final
    window's right edge stay at UNCOVERED and are dropped before scoring; since
    step <= win_len the covered region is a contiguous prefix, and its complement is
    (n_samples - win_len) mod step samples, or exactly `step` when that is 0 -- never
    more than one step, whatever the window length.
    """
    sample_pred = np.full(n_samples, UNCOVERED, dtype=np.int64)
    for i in range(len(y_pred_win)):
        start = i * step
        sample_pred[start:start + win_len] = y_pred_win[i]
    return sample_pred


# --------------------------------------------------------------------------- #
# per-directory processing
# --------------------------------------------------------------------------- #

def process_dir(results_dir, labels_df, candidate_cache, npz_pattern):
    if not os.path.isdir(results_dir):
        fail(f"--results_dir does not exist or is not a directory: {results_dir}")

    cfg = load_cfg(results_dir)
    sw_length, sw_overlap, win_len, step = derive_windowing(cfg, results_dir)

    key = (win_len, step)
    if key not in candidate_cache:
        candidate_cache[key] = build_candidates(labels_df, win_len, step)
    candidates = candidate_cache[key]

    folds = cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)
    if len(folds) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: cfg loso_subjects lists {len(folds)} folds, expected "
             f"{EXPECTED_FOLD_COUNT}: {folds}")

    npz_paths = discover_fold_npz(results_dir, folds, npz_pattern)

    fold_rows = []
    pred_parts, true_parts = [], []
    win_pred_parts, win_true_parts = [], []

    for fold in sorted(npz_paths):
        y_pred_win, y_true_win = load_npz(npz_paths[fold])
        subj = resolve_fold(fold, y_true_win, candidates, results_dir, win_len, step)

        cand = candidates[subj]
        n_samples = cand["n_samples"]

        n_expected = expected_window_count(n_samples, win_len, step)
        if len(y_true_win) != n_expected:
            fail(f"{results_dir} fold {fold} -> {subj}: npz holds {len(y_true_win)} windows, "
                 f"the raw {n_samples} samples give {n_expected} at win_len={win_len} step={step}")

        y_true_sample = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy().astype(np.int64)
        if len(y_true_sample) != n_samples:
            fail(f"fold {fold} -> {subj}: raw label slice is {len(y_true_sample)} rows, "
                 f"candidate table says {n_samples}")

        # The certification, restated directly against the raw timeline: the last
        # sample of window i is y_true_sample[i*step + win_len - 1].
        idx_last = np.arange(len(y_true_win), dtype=np.int64) * step + win_len - 1
        if idx_last[-1] >= n_samples:
            fail(f"fold {fold} -> {subj}: window {len(y_true_win) - 1} ends at sample "
                 f"{idx_last[-1]}, past the subject's {n_samples} samples")
        if not np.array_equal(y_true_sample[idx_last], y_true_win):
            n_bad = int((y_true_sample[idx_last] != y_true_win).sum())
            fail(f"{results_dir} fold {fold} -> {subj}: reconstructed last-sample labels differ "
                 f"from the npz y_true at {n_bad} of {len(y_true_win)} windows "
                 f"(win_len={win_len}, step={step}). Reconstruction not certified.")

        sample_pred = expand_last_window_wins(y_pred_win, n_samples, win_len, step)
        covered = sample_pred != UNCOVERED
        n_cov = int(covered.sum())

        expected_cov = (len(y_pred_win) - 1) * step + win_len
        if n_cov != expected_cov:
            fail(f"fold {fold} -> {subj}: covered {n_cov} samples, expansion geometry says "
                 f"{expected_cov}")

        fold_rows.append({
            "fold": fold, "subject": subj, "npz": os.path.basename(npz_paths[fold]),
            "n_windows": len(y_true_win), "n_samples": n_samples,
            "n_covered": n_cov, "coverage": n_cov / n_samples,
            "token_ok": fold == subj,
        })

        pred_parts.append(sample_pred[covered])
        true_parts.append(y_true_sample[covered])
        win_pred_parts.append(y_pred_win)
        win_true_parts.append(y_true_win)

    subjects = [r["subject"] for r in fold_rows]
    if len(set(subjects)) != EXPECTED_FOLD_COUNT:
        fail(f"{results_dir}: {EXPECTED_FOLD_COUNT} folds resolved to {len(set(subjects))} distinct "
             f"subjects {sorted(subjects)} -- the resolution is not a bijection")

    s_pred = np.concatenate(pred_parts)
    s_true = np.concatenate(true_parts)
    w_pred = np.concatenate(win_pred_parts)
    w_true = np.concatenate(win_true_parts)

    s_f1 = f1_score(s_true, s_pred, labels=ALL_LABELS, average=None, zero_division=0)
    w_f1 = f1_score(w_true, w_pred, labels=ALL_LABELS, average=None, zero_division=0)
    s_prec, s_rec, _, s_support = precision_recall_fscore_support(
        s_true, s_pred, labels=ALL_LABELS, zero_division=0)

    return {
        "dir": results_dir,
        "name": display_name(results_dir),
        "sw_length": sw_length, "sw_overlap": sw_overlap,
        "win_len": win_len, "step": step,
        "folds": fold_rows,
        "subjects": sorted(set(subjects)),
        "n_windows": int(len(w_true)),
        "n_raw": int(sum(r["n_samples"] for r in fold_rows)),
        "n_covered": int(len(s_true)),
        "coverage": len(s_true) / sum(r["n_samples"] for r in fold_rows),
        "sample_f1": s_f1,
        "sample_macro": float(f1_score(s_true, s_pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "window_f1": w_f1,
        "window_macro": float(f1_score(w_true, w_pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "sample_prec": s_prec, "sample_rec": s_rec, "sample_support": s_support,
        "window_support": np.bincount(w_true, minlength=N_CLASSES),
    }


# --------------------------------------------------------------------------- #
# dense (per-sample) processing
# --------------------------------------------------------------------------- #

DENSE_NPZ_RE = re.compile(r"^preds_([0-9a-f]{4})_.*\.npz$")


def resolve_dense_folds(results_dir, cfg):
    """
    The fold list for a --dense run: cfg's when it names one, else the npz filenames.

    A run over every subject records `loso_subjects: []` (main.py resolves an unset
    --loso_subjects to an empty list, and cross_participant_cv then iterates all
    subjects). The previous `cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)`
    turned that empty list into the five BASELINE folds, so an all-subject run was
    scored on 5 of its 14 npz files -- reporting coverage 1.00000 and a plausible
    macro F1, with nothing in the output showing that 9 folds had been dropped. The
    coverage column cannot catch it either: coverage is measured against the folds
    that were loaded, so discarding folds entirely leaves it at 1.0.

    Deriving the folds from the filenames removes the guess. When cfg does name folds
    they must agree with the files exactly, in both directions, so a mismatch is a
    hard fail rather than a silent intersection.
    """
    cfg_folds = list(cfg.get("loso_subjects") or [])

    by_fold = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "preds_*.npz"))):
        base = os.path.basename(path)
        if "_aug" in base:
            continue
        m = DENSE_NPZ_RE.match(base)
        if m:
            by_fold.setdefault(m.group(1), []).append(base)

    if not by_fold:
        fail(f"{results_dir}: no preds_<fold>_*.npz files found; cannot resolve the folds.")

    found = sorted(by_fold)
    if cfg_folds and set(cfg_folds) != set(found):
        fail(f"{results_dir}: cfg loso_subjects lists {sorted(cfg_folds)} but the npz files "
             f"are {found}. Refusing to score a fold set the run does not claim.")
    return sorted(cfg_folds) if cfg_folds else found


def load_seam_map(path):
    """The segment map written by scripts/build_seam_map.py, used only in --dense mode."""
    if not os.path.isfile(path):
        fail(f"seam map not found at {path!r}. --dense needs it to rebuild the per-sample "
             "ground truth; build it with scripts/build_seam_map.py.")
    with open(path) as fid:
        payload = json.load(fid)
    for key in ("total_samples", "segments"):
        if key not in payload:
            fail(f"{path}: missing key {key!r}")
    return payload


def dense_expected_truth(subject, segments, label_ids, min_segment_len):
    """
    The per-sample ground truth a dense fold MUST have produced, rebuilt from the raw label
    stream: the kept segments of `subject`, concatenated in seam-map (== raw stream) order.

    This is the dense replacement for resolve_fold's certification. It is strictly stronger:
    the windowed check matched a subject on its last-sample label sequence, one label per 25
    samples, while this matches every sample. A wrong fold->subject mapping, a wrong
    min_segment_len, or a different label encoding all fail it.
    """
    parts = [label_ids[s["start_row"]:s["end_row"]] for s in segments
             if s["subject_name"] == subject and s["length"] >= min_segment_len]
    if not parts:
        fail(f"seam map holds no kept segments for subject {subject!r} at "
             f"min_segment_len={min_segment_len}")
    return np.concatenate(parts).astype(np.int64)


def process_dir_dense(results_dir, labels_df, seam, npz_pattern, allow_partial):
    """
    Score a --dense run. No fold->subject brute force and no expansion: the npz already holds
    one prediction per raw sample, in stream order, so the only thing to establish is that its
    y_true is the subject's raw label stream -- which is exactly what the certification below
    checks, sample for sample.
    """
    if not os.path.isdir(results_dir):
        fail(f"--results_dir does not exist or is not a directory: {results_dir}")

    cfg = load_cfg(results_dir)
    if not cfg.get("dense", False):
        fail(f"{results_dir}: --dense was passed but this run's cfg.txt has dense={cfg.get('dense')!r}. "
             "Scoring a windowed run as dense would compare a per-window array against a "
             "per-sample timeline; refusing.")

    min_segment_len = int(cfg.get("dense_min_seg", 25))
    seq_len = int(cfg.get("dense_seq_len", 500))
    segments = seam["segments"]

    label_ids = labels_df["label_id"].to_numpy().astype(np.int64)
    if len(label_ids) != seam["total_samples"]:
        fail(f"labels file holds {len(label_ids)} samples, seam map describes "
             f"{seam['total_samples']}. They must be the same stream in the same order.")
    subj_col = labels_df["subject"].to_numpy()
    for seg in segments:
        if subj_col[seg["start_row"]] != seg["subject_name"]:
            fail(f"seam map segment at row {seg['start_row']} says subject "
                 f"{seg['subject_name']!r} but the labels file says "
                 f"{subj_col[seg['start_row']]!r}; the two are not aligned.")

    folds = resolve_dense_folds(results_dir, cfg)
    expect = len(folds) if allow_partial else len(folds)
    if not allow_partial and len(folds) not in (EXPECTED_FOLD_COUNT, ALL_SUBJECT_FOLD_COUNT):
        fail(f"{results_dir}: resolved {len(folds)} folds, expected "
             f"{EXPECTED_FOLD_COUNT} (the LOSO baseline) or {ALL_SUBJECT_FOLD_COUNT} "
             f"(every subject): {folds}. Pass --allow_partial_folds to score a "
             "partial run (a smoke test) anyway.")
    npz_paths = discover_fold_npz(results_dir, folds, npz_pattern, expect_count=expect)

    fold_rows, pred_parts, true_parts = [], [], []
    for fold in sorted(npz_paths):
        y_pred, y_true = load_npz(npz_paths[fold])

        expected = dense_expected_truth(fold, segments, label_ids, min_segment_len)
        if len(y_true) != len(expected):
            fail(f"{results_dir} fold {fold}: npz holds {len(y_true)} samples, the raw stream "
                 f"gives {len(expected)} kept samples at min_segment_len={min_segment_len}. "
                 "Reconstruction not certified.")
        if not np.array_equal(y_true, expected):
            n_bad = int((y_true != expected).sum())
            fail(f"{results_dir} fold {fold}: npz y_true differs from the raw per-sample labels "
                 f"at {n_bad} of {len(expected)} samples. Reconstruction not certified.")

        raw_total = sum(s["length"] for s in segments if s["subject_name"] == fold)
        fold_rows.append({
            "fold": fold, "subject": fold, "npz": os.path.basename(npz_paths[fold]),
            "n_windows": 0, "n_samples": raw_total, "n_covered": len(y_true),
            "coverage": len(y_true) / raw_total, "token_ok": True,
        })
        pred_parts.append(y_pred)
        true_parts.append(y_true)

    s_pred = np.concatenate(pred_parts)
    s_true = np.concatenate(true_parts)
    s_f1 = f1_score(s_true, s_pred, labels=ALL_LABELS, average=None, zero_division=0)
    s_prec, s_rec, _, s_support = precision_recall_fscore_support(
        s_true, s_pred, labels=ALL_LABELS, zero_division=0)

    return {
        "dir": results_dir, "name": display_name(results_dir), "dense": True,
        "sw_length": seq_len / SAMPLING_RATE, "sw_overlap": int(round(cfg.get("dense_overlap", 0.5) * 100)),
        "win_len": seq_len, "step": int(seq_len * (1 - cfg.get("dense_overlap", 0.5))),
        "min_segment_len": min_segment_len,
        "folds": fold_rows, "subjects": sorted(r["subject"] for r in fold_rows),
        "n_windows": 0,
        "n_raw": int(sum(r["n_samples"] for r in fold_rows)),
        "n_covered": int(len(s_true)),
        "coverage": len(s_true) / sum(r["n_samples"] for r in fold_rows),
        "sample_f1": s_f1,
        "sample_macro": float(f1_score(s_true, s_pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "window_f1": np.full(N_CLASSES, np.nan),
        "window_macro": float("nan"),
        "sample_prec": s_prec, "sample_rec": s_rec, "sample_support": s_support,
        "window_support": np.zeros(N_CLASSES, dtype=int),
    }


def display_name(path):
    base = os.path.basename(os.path.normpath(path))
    stripped = TS_PREFIX_RE.sub("", base)
    return stripped or base


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

CW = 10          # per-class column width; longest class name is 'dribbling' (9)


def metric_table(records, header, note, values_of, macro_of, signed=False):
    """One row per results_dir, one column per class, plus the macro."""
    name_w = max(max(len(r["name"]) for r in records), 12)
    sign = "+" if signed else ""      # sign flag precedes the width in a format spec

    print(f"\n{header}")
    print(note)
    line = (f"  {'config':<{name_w}} {'sw_len':>7} {'n_windows':>10} {'n_samples':>10} {'coverage':>9}  "
            + "".join(f"{n:>{CW}}" for n in CLASS_NAMES)
            + f"{'macro_F1':>{CW + 2}}")
    print("\n" + line)
    print("  " + "-" * (len(line) - 2))
    for r in records:
        vals = values_of(r)
        print(f"  {r['name']:<{name_w}} {r['sw_length']:>7.1f} {r['n_windows']:>10} "
              f"{r['n_covered']:>10} {r['coverage']:>9.5f}  "
              + "".join(f"{v:>{sign}{CW}.4f}" for v in vals)
              + f"{macro_of(r):>{sign}{CW + 2}.4f}")


def report_per_dir(records):
    print("\n" + "=" * 100)
    print("[1] PER-CONFIG RECONSTRUCTION -- fold -> subject, certified per fold")
    print("=" * 100)
    print("\nEach fold is accepted only on window count + exact last-sample-label equality against")
    print("the raw per-sample labels. The filename token is a fast path, never evidence")
    print("(_loso_common.py:337-341). 'token' flags whether it happened to be right.\n")

    for r in records:
        print(f"  {r['name']}   [{r['dir']}]")
        if r.get("dense"):
            print(f"    DENSE: seq_len={r['win_len']} samples ({r['sw_length']}s), "
                  f"step={r['step']}, min_segment_len={r['min_segment_len']}  ->  one "
                  f"prediction per raw sample, no expansion")
            print("    Certified sample-for-sample against the raw label stream rebuilt from the")
            print("    seam map, which is stricter than the windowed last-sample check.")
        else:
            print(f"    sw_length={r['sw_length']}s  sw_overlap={r['sw_overlap']}%  ->  "
                  f"win_len={r['win_len']} samples, step={r['step']} samples "
                  f"(win_len//2 would be {r['win_len'] // 2})")
        print(f"    {'fold':<6} {'subject':<8} {'token':<6} {'windows':>9} {'samples':>9} "
              f"{'covered':>9} {'coverage':>9}  npz")
        for f in r["folds"]:
            print(f"    {f['fold']:<6} {f['subject']:<8} {'ok' if f['token_ok'] else 'WRONG':<6} "
                  f"{f['n_windows']:>9} {f['n_samples']:>9} {f['n_covered']:>9} "
                  f"{f['coverage']:>9.5f}  {f['npz']}")
        print(f"    {'TOTAL':<6} {'':<8} {'':<6} {r['n_windows']:>9} {r['n_raw']:>9} "
              f"{r['n_covered']:>9} {r['coverage']:>9.5f}")
        print()


def report_tables(records):
    print("=" * 100)
    print("[2] SAMPLE-LEVEL PER-CLASS F1 -- the cross-config metric")
    print("=" * 100)
    metric_table(
        records,
        "  Window predictions expanded onto the raw 50 Hz timeline (last-window-wins),",
        "  pooled over the 5 LOSO folds, scored against the original per-sample labels.",
        lambda r: r["sample_f1"], lambda r: r["sample_macro"],
    )

    if all(r.get("dense") for r in records):
        print("\n" + "=" * 100)
        print("[3]/[4] WINDOW-LEVEL TABLES -- not applicable to a dense run")
        print("=" * 100)
        print("\n  A dense run emits one prediction per sample; there is no window population to")
        print("  score, and no last-sample rule to diverge from. Table [2] is the whole result.")
        return

    print("\n" + "=" * 100)
    print("[3] WINDOW-LEVEL PER-CLASS F1 -- diagnostic context only")
    print("=" * 100)
    metric_table(
        records,
        "  f1_score over the pooled npz windows, unexpanded.",
        "  NOT comparable across rows: n_windows and label purity both move with sw_length.\n"
        "  Read each row against its own row in [2], not against another row here.",
        lambda r: r["window_f1"], lambda r: r["window_macro"],
    )

    print("\n" + "=" * 100)
    print("[4] DIVERGENCE -- sample-level minus window-level, within each config")
    print("=" * 100)
    metric_table(
        records,
        "  How far the two scoring rules pull apart at each window length.",
        "  A term growing in magnitude down the sw_length column is the last-sample labeling\n"
        "  rule losing its grip on what a window contains as the window lengthens.",
        lambda r: r["sample_f1"] - r["window_f1"],
        lambda r: r["sample_macro"] - r["window_macro"],
        signed=True,
    )


def report_delta(records):
    if any(r.get("dense") for r in records):
        print("\n" + "=" * 100)
        print("[5] DELTA vs THE 1s BASELINE -- skipped (dense run present)")
        print("=" * 100)
        print("\n  Both a dense and a windowed run are scored on the raw 50 Hz timeline, so their")
        print("  table [2] rows ARE comparable and should be read side by side. They are not")
        print("  differenced automatically here because the two denominators are not identical:")
        print("  the windowed path drops a trailing tail per fold, the dense path drops whole")
        print("  segments shorter than min_segment_len. Check the coverage column before")
        print("  differencing by hand.")
        return

    base = [r for r in records if abs(r["sw_length"] - BASELINE_SW_LENGTH) < 1e-9]
    if len(records) < 2:
        return
    if not base:
        print("\n" + "=" * 100)
        print("[5] DELTA vs THE 1s BASELINE -- skipped")
        print("=" * 100)
        print(f"\n  No --results_dir has sw_length == {BASELINE_SW_LENGTH}; nothing to difference against.")
        return
    if len(base) > 1:
        print(f"\n  NOTE: {len(base)} configs have sw_length == {BASELINE_SW_LENGTH} "
              f"({', '.join(b['name'] for b in base)}); differencing against the first.")
    ref = base[0]

    print("\n" + "=" * 100)
    print(f"[5] DELTA vs THE 1s BASELINE ({ref['name']}) -- sample-level")
    print("=" * 100)
    others = [r for r in records if r is not ref]
    metric_table(
        others,
        "  Sample-level per-class F1 minus the baseline's, on the shared raw-timeline denominator.",
        "  This is the only differencing in this script that is licensed: both terms are scored\n"
        "  over the same samples. The window-level table above must not be differenced this way.",
        lambda r: r["sample_f1"] - ref["sample_f1"],
        lambda r: r["sample_macro"] - ref["sample_macro"],
        signed=True,
    )


def report_precision_recall(records):
    print("\n" + "=" * 100)
    print("[6] SAMPLE-LEVEL PRECISION / RECALL -- per config, diagnostic")
    print("=" * 100)
    for r in records:
        print(f"\n  {r['name']}   sw_length={r['sw_length']}s  win_len={r['win_len']}  step={r['step']}")
        print(f"    {'class':<12} {'support':>10} {'share':>8} {'precision':>10} {'recall':>8} "
              f"{'F1':>8}   {'win_F1':>8} {'win_supp':>9}")
        for c, name in enumerate(CLASS_NAMES):
            tail = ("" if r.get("dense") else
                    f"   {r['window_f1'][c]:>8.4f} {int(r['window_support'][c]):>9}")
            print(f"    {name:<12} {int(r['sample_support'][c]):>10} "
                  f"{r['sample_support'][c] / r['sample_support'].sum():>8.4f} "
                  f"{r['sample_prec'][c]:>10.4f} {r['sample_rec'][c]:>8.4f} {r['sample_f1'][c]:>8.4f}"
                  + tail)
        tail = ("" if r.get("dense") else
                f"   {r['window_macro']:>8.4f} {r['n_windows']:>9}")
        print(f"    {'MACRO':<12} {int(r['sample_support'].sum()):>10} {'':>8} {'':>10} {'':>8} "
              f"{r['sample_macro']:>8.4f}" + tail)


def report_sanity(records):
    print("\n" + "=" * 100)
    print("[7] SANITY CHECKS -- printed, never asserted")
    print("=" * 100)

    if all(r.get("dense") for r in records):
        reb = CLASS_NAMES.index("rebound")
        print("\n  Dense run: the window-level anchors below do not apply. Coverage and the")
        print("  rebound line are printed; everything keyed to win_len/step is omitted.\n")
        print(f"      {'config':<24} {'raw':>10} {'covered':>10} {'gap':>7} {'coverage':>10} "
              f"{'reb_F1':>8} {'reb_supp':>9}")
        for r in records:
            print(f"      {r['name']:<24} {r['n_raw']:>10} {r['n_covered']:>10} "
                  f"{r['n_raw'] - r['n_covered']:>7} {r['coverage']:>10.5f} "
                  f"{r['sample_f1'][reb]:>8.4f} {int(r['sample_support'][reb]):>9}")
        print("\n      The gap is the samples in segments shorter than min_segment_len, which")
        print("      the loader discards; it is 0 at the default min_segment_len=25.")
        return

    reb = CLASS_NAMES.index("rebound")

    print("\n  (a) coverage: covered samples vs the raw sample count for the 5 LOSO subjects")
    print("      The gap is the trailing tail past the final window -- at most `step` per fold,")
    print("      so at most 5*step overall.")
    print(f"\n      {'config':<24} {'raw':>10} {'covered':>10} {'gap':>7} {'max (5*step)':>13} {'coverage':>10}")
    for r in records:
        print(f"      {r['name']:<24} {r['n_raw']:>10} {r['n_covered']:>10} "
              f"{r['n_raw'] - r['n_covered']:>7} {5 * r['step']:>13} {r['coverage']:>10.5f}")

    print("\n  (b) sample-level vs window-level macro F1, within each config")
    print("      Expect close but not identical: the two score different populations.")
    print(f"\n      {'config':<24} {'sw_len':>7} {'sample_macro':>13} {'window_macro':>13} {'diff':>9}")
    for r in records:
        print(f"      {r['name']:<24} {r['sw_length']:>7.1f} {r['sample_macro']:>13.4f} "
              f"{r['window_macro']:>13.4f} {r['sample_macro'] - r['window_macro']:>+9.4f}")

    print("\n  (c) rebound, sample-level vs window-level")
    print(f"\n      {'config':<24} {'sw_len':>7} {'sample_F1':>10} {'window_F1':>10} {'diff':>9} "
          f"{'sample_supp':>12} {'window_supp':>12}")
    for r in records:
        print(f"      {r['name']:<24} {r['sw_length']:>7.1f} {r['sample_f1'][reb]:>10.4f} "
              f"{r['window_f1'][reb]:>10.4f} {r['sample_f1'][reb] - r['window_f1'][reb]:>+9.4f} "
              f"{int(r['sample_support'][reb]):>12} {int(r['window_support'][reb]):>12}")

    print("\n  (d) recorded window-level anchors, for comparison with the 1s row of table [3]")
    print("      Verbatim from grid_read.py:61-62. These are BASELINE numbers -- they apply only")
    print("      to a 1s run of the unaugmented baseline, and only at the stated seed.")
    print(f"\n      seed 1          : rebound={ANCHOR_WINDOW_SEED1['rebound']:.4f}  "
          f"layup={ANCHOR_WINDOW_SEED1['layup']:.4f}  walking={ANCHOR_WINDOW_SEED1['walking']:.4f}  "
          f"macro={ANCHOR_WINDOW_SEED1['macro']:.4f}")
    print(f"      mean(seeds 1,2,3): rebound={ANCHOR_WINDOW_MEAN123['rebound']:.4f}  "
          f"walking={ANCHOR_WINDOW_MEAN123['walking']:.4f}")
    print("\n      NOTE on the 0.1969 rebound figure: it is the mean over seeds 1/2/3, not a")
    print("      single run. A single-seed 1s baseline should land on 0.2053, not 0.1969.")
    print("      NOTE on macro F1: the recorded seed-1 baseline macro is 0.5273. If you are")
    print("      expecting ~0.616, that figure is not recorded anywhere in this repo and does")
    print("      not come from grid_read.py's anchors -- check which run it came from before")
    print("      reading any agreement or disagreement here as meaningful.")

    for r in records:
        if abs(r["sw_length"] - BASELINE_SW_LENGTH) > 1e-9:
            continue
        print(f"\n      {r['name']} (1s) window-level vs the seed-1 anchors:")
        for cls in ("rebound", "layup", "walking"):
            c = CLASS_NAMES.index(cls)
            d = r["window_f1"][c] - ANCHOR_WINDOW_SEED1[cls]
            print(f"        {cls:<9} observed={r['window_f1'][c]:.4f}  anchor={ANCHOR_WINDOW_SEED1[cls]:.4f}  "
                  f"diff={d:+.4f}")
        d = r["window_macro"] - ANCHOR_WINDOW_SEED1["macro"]
        print(f"        {'macro':<9} observed={r['window_macro']:.4f}  "
              f"anchor={ANCHOR_WINDOW_SEED1['macro']:.4f}  diff={d:+.4f}")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", action="append", nargs="+", required=True, metavar="DIR",
                        help="Run dir holding cfg.txt and the 5 per-fold npz files. Repeatable, "
                             "and each occurrence accepts several dirs; all are pooled into one "
                             "comparison table in the order given.")
    parser.add_argument("--labels", default="labels_export.csv.gz",
                        help="Exported per-sample labels (columns: subject, label), original row order.")
    parser.add_argument("--dense", action="store_true",
                        help="Score a --dense run: the npz already holds one prediction per raw "
                             "sample, so the fold->subject brute force and the last-window-wins "
                             "expansion are both bypassed. Ground truth is certified sample for "
                             "sample against the raw stream rebuilt from --seam_map.")
    parser.add_argument("--seam_map", default="data/seam_map.json",
                        help="Seam map used by --dense to rebuild each fold's kept-sample stream.")
    parser.add_argument("--allow_partial_folds", action="store_true",
                        help="Score a run with fewer than 5 folds (a smoke test). Off by default "
                             "so a truncated run cannot be mistaken for a complete one.")
    parser.add_argument("--npz_pattern", default=None,
                        help="Explicit glob containing '{fold}', overriding npz auto-detection. "
                             "Only needed when a run dir holds more than one npz per fold.")
    args = parser.parse_args()

    results_dirs = [d for group in args.results_dir for d in group]
    seen = set()
    for d in results_dirs:
        norm = os.path.normpath(d)
        if norm in seen:
            fail(f"--results_dir {d} given more than once")
        seen.add(norm)

    print("=" * 100)
    print("SAMPLE-LEVEL F1 ACROSS THE WINDOW-LENGTH SWEEP")
    print("=" * 100)
    print(f"\n  sampling rate  : {SAMPLING_RATE} Hz (hardcoded at preprocess_data.py:31; never in cfg.txt)")
    print(f"  labels         : {args.labels}")
    print(f"  configs        : {len(results_dirs)}")
    print("  windowing      : derived per config from cfg.txt; window_purity.py's pinned")
    print(f"                   WIN_LEN={wp.WIN_LEN}/STEP={wp.STEP} are deliberately NOT used here.")

    labels_df = wp.load_labels(args.labels)
    print(f"\n  loaded {len(labels_df)} per-sample labels, {labels_df['subject'].nunique()} subjects, "
          f"{labels_df['label_id'].nunique()} classes")

    if args.dense:
        seam = load_seam_map(args.seam_map)
        print(f"  seam map       : {args.seam_map} ({seam['total_segments']} segments)")
        records = [process_dir_dense(d, labels_df, seam, args.npz_pattern, args.allow_partial_folds)
                   for d in results_dirs]
    else:
        candidate_cache = {}
        records = [process_dir(d, labels_df, candidate_cache, args.npz_pattern) for d in results_dirs]

    subject_sets = {tuple(r["subjects"]) for r in records}
    if len(subject_sets) != 1:
        detail = "; ".join(f"{r['name']} -> {r['subjects']}" for r in records)
        fail("configs resolved to different LOSO subject sets, so their scores are computed over "
             f"different data and cannot be compared: {detail}")
    print(f"  all {len(records)} config(s) resolved to the same {len(records[0]['subjects'])} "
          f"subject(s): {records[0]['subjects']}")

    report_per_dir(records)
    report_tables(records)
    report_delta(records)
    report_precision_recall(records)
    report_sanity(records)

    print("\n" + "=" * 100)
    print("Read table [2] across rows. Table [3] is per-row diagnostic context only.")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
