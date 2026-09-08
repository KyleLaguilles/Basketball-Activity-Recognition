#!/usr/bin/env python3
"""
Shared loader for the dense (per-sample) diagnostic scripts.

Not a runnable script -- the leading underscore marks it as a helper module, the same
role analysis/_loso_common.py plays for the windowed scripts.

WHY THIS EXISTS. A dense npz holds exactly two arrays, y_pred and y_true, both one
entry per raw 50 Hz sample. It carries NO subject column and NO segment marker: the
array is that subject's kept seam-map segments concatenated end to end, and the joins
are invisible inside it. Every diagnostic here needs those joins back --

  * event_rebound_recall.py must not merge two rebounds across a splice into one event
  * dense_pass_overcount.py must not measure a distance that crosses a splice

-- so this module reconstructs the segment offsets in the concatenated coordinate
system and certifies them against sample_level_f1's reconstruction.

CERTIFICATION IS REUSED WHOLESALE from sample_level_f1 (slf.dense_expected_truth):
y_true must equal the subject's raw per-sample label stream, sample for sample. A
wrong fold->subject mapping, a wrong min_segment_len, or a different label encoding
all fail it. The segment offsets built here are additionally asserted to concatenate
to that same certified array, so an offset table that disagrees with slf's ordering
cannot survive.

FOLD DISCOVERY IS FROM THE FILENAMES, NOT FROM cfg['loso_subjects']. The all-14-subject
dense run records `loso_subjects: []`, and slf.process_dir_dense's
`cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)` fallback would silently resolve
that to the five baseline folds and score 5 of the 14 npz files while reporting
success. Here the fold set comes from the npz basenames and is cross-checked against
cfg when cfg names any, so the two can never disagree silently.
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_purity as wp          # noqa: E402  -- module-level defs, __main__-guarded
import sample_level_f1 as slf       # noqa: E402  -- certified reconstruction, reused wholesale

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
HardFail = wp.HardFail
fail = slf.fail
SAMPLING_RATE = slf.SAMPLING_RATE

EXPECTED_FOLD_COUNT = slf.EXPECTED_FOLD_COUNT      # 5, the LOSO baseline
EXPECTED_FOLD_COUNT_ALL = 14                       # every subject in the seam map

# validation.py:324-325 names these preds_<fold>_<fraction>[_seed<N>].npz; the fold
# token is a 4-hex-character subject code. Augmentation runs carry a further _aug
# suffix and are a different model, so they are excluded rather than swept in.
NPZ_RE = re.compile(r"^preds_([0-9a-f]{4})_.*\.npz$")


def class_id(name):
    """Class name -> id, hard-failing rather than returning a silent -1."""
    if name not in CLASS_NAMES:
        fail(f"unknown class {name!r}; known classes are {CLASS_NAMES}")
    return CLASS_NAMES.index(name)


def add_common_args(parser):
    """The flag set every dense diagnostic shares, spelled identically to slf's."""
    parser.add_argument("--results_dir", required=True, metavar="DIR",
                        help="A --dense run directory (must contain cfg.txt with dense=true).")
    parser.add_argument("--labels", default="labels_export.csv.gz",
                        help="Raw per-sample label export used to certify y_true.")
    parser.add_argument("--seam_map", default="data/seam_map.json",
                        help="Segment map from scripts/build_seam_map.py.")
    parser.add_argument("--npz_pattern", default=None,
                        help="Glob containing '{fold}', to disambiguate multiple npz per fold.")
    parser.add_argument("--allow_partial_folds", action="store_true",
                        help="Score a run that covers neither 5 nor 14 folds (a smoke test).")
    return parser


def discover_folds(results_dir, cfg, npz_pattern, allow_partial):
    """
    fold token -> npz path, derived from the filenames and cross-checked against cfg.

    A fold resolving to zero or several files is a hard fail; --npz_pattern (which must
    contain '{fold}') resolves the ambiguity explicitly rather than by guess.
    """
    cfg_folds = list(cfg.get("loso_subjects") or [])

    if npz_pattern:
        if "{fold}" not in npz_pattern:
            fail(f"--npz_pattern must contain '{{fold}}', got {npz_pattern!r}")
        if not cfg_folds:
            fail("--npz_pattern needs cfg['loso_subjects'] to name the folds, but it is empty. "
                 "Drop --npz_pattern so the folds can be read from the filenames.")
        found = {}
        for fold in cfg_folds:
            hits = sorted(glob.glob(os.path.join(results_dir, npz_pattern.format(fold=fold))))
            if len(hits) != 1:
                fail(f"{results_dir} fold {fold}: expected exactly 1 npz, found {len(hits)}: "
                     f"{[os.path.basename(h) for h in hits]}")
            found[fold] = hits[0]
    else:
        by_fold = {}
        for path in sorted(glob.glob(os.path.join(results_dir, "preds_*.npz"))):
            base = os.path.basename(path)
            if "_aug" in base:
                continue
            m = NPZ_RE.match(base)
            if m:
                by_fold.setdefault(m.group(1), []).append(path)
        for fold, hits in sorted(by_fold.items()):
            if len(hits) != 1:
                fail(f"{results_dir} fold {fold}: expected exactly 1 npz, found {len(hits)}: "
                     f"{[os.path.basename(h) for h in hits]}. Pass --npz_pattern to disambiguate.")
        found = {f: h[0] for f, h in by_fold.items()}

    if not found:
        fail(f"{results_dir}: no preds_<fold>_*.npz files found.")

    # cfg names folds -> the filenames must agree with it exactly, in both directions.
    if cfg_folds and set(cfg_folds) != set(found):
        fail(f"{results_dir}: cfg lists folds {sorted(cfg_folds)} but the npz files are "
             f"{sorted(found)}. Refusing to score a set the run does not claim.")

    n = len(found)
    if not allow_partial and n not in (EXPECTED_FOLD_COUNT, EXPECTED_FOLD_COUNT_ALL):
        fail(f"{results_dir}: resolved {n} folds ({sorted(found)}), expected "
             f"{EXPECTED_FOLD_COUNT} (LOSO baseline) or {EXPECTED_FOLD_COUNT_ALL} (all subjects). "
             "Pass --allow_partial_folds to score a partial run anyway.")
    return found


def load_dense_run(results_dir, labels_path, seam_path, npz_pattern=None, allow_partial=False):
    """
    Load and certify one --dense run.

    Returns a dict with the concatenated arrays plus a per-fold breakdown carrying, for
    each fold, the segment offsets in that fold's own concatenated coordinates:
        folds[i]["segs"] = [{"local_start", "local_end", "length", "recording",
                             "segment_idx", "start_row"}, ...]
    """
    if not os.path.isdir(results_dir):
        fail(f"--results_dir does not exist or is not a directory: {results_dir}")

    cfg = slf.load_cfg(results_dir)
    if not cfg.get("dense", False):
        fail(f"{results_dir}: cfg.txt has dense={cfg.get('dense')!r}. These diagnostics read a "
             "per-sample timeline; a windowed run holds one prediction per window. Refusing.")

    seam = slf.load_seam_map(seam_path)
    labels_df = wp.load_labels(labels_path)

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

    npz_paths = discover_folds(results_dir, cfg, npz_pattern, allow_partial)

    folds = []
    for fold in sorted(npz_paths):
        y_pred, y_true = slf.load_npz(npz_paths[fold])       # note: (pred, true) order

        # Certification, reused wholesale from sample_level_f1.
        expected = slf.dense_expected_truth(fold, segments, label_ids, min_segment_len)
        if len(y_true) != len(expected):
            fail(f"{results_dir} fold {fold}: npz holds {len(y_true)} samples, the raw stream "
                 f"gives {len(expected)} kept samples at min_segment_len={min_segment_len}. "
                 "Reconstruction not certified.")
        if not np.array_equal(y_true, expected):
            n_bad = int((y_true != expected).sum())
            fail(f"{results_dir} fold {fold}: npz y_true differs from the raw per-sample labels "
                 f"at {n_bad} of {len(expected)} samples. Reconstruction not certified.")

        # Segment offsets in this fold's concatenated coordinates, in seam-map order --
        # the identical filter and order dense_expected_truth used above.
        kept = [s for s in segments
                if s["subject_name"] == fold and s["length"] >= min_segment_len]
        segs, cursor = [], 0
        for s in kept:
            segs.append({"local_start": cursor, "local_end": cursor + s["length"],
                         "length": s["length"], "recording": s["recording"],
                         "segment_idx": s["segment_idx"], "start_row": s["start_row"]})
            cursor += s["length"]
        if cursor != len(y_true):
            fail(f"{results_dir} fold {fold}: segment offsets tile {cursor} samples but the npz "
                 f"holds {len(y_true)}. The offset table disagrees with the certified stream.")

        folds.append({"fold": fold, "subject": fold, "npz": os.path.basename(npz_paths[fold]),
                      "y_true": y_true, "y_pred": y_pred, "segs": segs,
                      "n_samples": int(len(y_true))})

    return {
        "results_dir": results_dir,
        "name": slf.display_name(results_dir),
        "cfg": cfg,
        "min_segment_len": min_segment_len,
        "seq_len": seq_len,
        "n_folds": len(folds),
        "all_subjects": len(folds) == EXPECTED_FOLD_COUNT_ALL,
        "folds": folds,
        "y_true": np.concatenate([f["y_true"] for f in folds]),
        "y_pred": np.concatenate([f["y_pred"] for f in folds]),
    }


def run_header(run, extra=()):
    """The provenance block every diagnostic prints before its numbers."""
    cfg = run["cfg"]
    print("=" * 78)
    print(f"run            : {run['name']}")
    print(f"dir            : {run['results_dir']}")
    print(f"folds          : {run['n_folds']}  ({', '.join(f['fold'] for f in run['folds'])})")
    print(f"samples        : {len(run['y_true'])}  "
          f"({len(run['y_true']) / SAMPLING_RATE / 3600:.2f} h at {SAMPLING_RATE} Hz)")
    print(f"dense_seq_len  : {run['seq_len']}   dense_min_seg: {run['min_segment_len']}   "
          f"batch_size: {cfg.get('batch_size')}   seed: {cfg.get('seed')}")
    if cfg.get("no_bilstm"):
        print("NOTE           : this run is the --no_bilstm ablation (no dense-head BiLSTM).")
    for line in extra:
        print(line)
    print("=" * 78)


def contiguous_runs(mask):
    """[start, end) bounds of every contiguous True run in a 1-D boolean mask."""
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def blocks_of(fold, span_seams):
    """
    The [start, end) blocks within which contiguity is meaningful for one fold.

    span_seams=False (default) -> one block per seam-map segment, so a run of labels
    interrupted by dropped samples is two events, not one.
    span_seams=True -> the fold's whole concatenated stream as a single block, which is
    the rule that reproduces the recorded 161-event total.
    """
    if span_seams:
        return [(0, fold["n_samples"])]
    return [(s["local_start"], s["local_end"]) for s in fold["segs"]]
