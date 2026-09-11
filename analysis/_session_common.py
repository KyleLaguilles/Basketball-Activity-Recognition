#!/usr/bin/env python3
"""
Shared per-SESSION loader for the duration/calibration diagnostics.

Not a runnable script -- the leading underscore marks it as a helper module, the same
role _dense_common.py plays for the dense diagnostics and _loso_common.py for the
windowed ones.

WHAT A SESSION IS
-----------------
The dataset holds 24 participants with one recording each: data/raw/<recording>.csv,
carried on every segment of data/seam_map.json as the `recording` field
(scripts/build_seam_map.py). A participant is keyed by that filename stem,
<4-hex id>_<eu|na>. The 4-hex id alone is NOT a participant: it is only unique within
a site (data/raw/meta.txt is keyed by location), and the 10 ids that appear under both
suffixes belong to different people -- 2dd9_eu and 2dd9_na differ in age, sex, height
and skill level. Training keys its LOSO subjects the same way
(preprocess_data.participant_keys), so every unit here is one and the same:

    session := recording := participant := LOSO subject      (24 in the full dataset)

and the seam map's subject_name equals its recording on every segment.

The suffix names the capture site, and each site was a single capture event, checked
against the raw timestamp column:

    every recording spans exactly ONE calendar date
    the 13 *_eu recordings  -> 2022-02-26, all starting within ~4 s of each other,
                               each spanning ~1.83 h
    the 11 *_na recordings  -> 2022-05-20, all starting within ~4 s of each other,
                               each spanning ~1.44 h

The two capture events are reported for context only (session_header lists sessions
by site); they are never the unit -- n=2 makes every per-class statistic meaningless.

COORDINATES, and why both prediction paths land in the same one
---------------------------------------------------------------
A windowed npz holds one entry per WINDOW; a dense npz holds one per SAMPLE. Neither
carries a session marker, and none is needed: each fold is one participant, so one
session. Both are reduced here to one array per subject in RAW SUBJECT-LOCAL
coordinates -- index i is the i-th row of that subject's slice of the label export --
and the session is the whole of that array:

    4d70_eu = [0, 61539)      4d70_na = [0, 65300)      (two different participants)

Three facts make that mapping exact, and all three are asserted in load_sessions()
rather than assumed:
  1. seam-map segments tile the global stream contiguously and in ascending row order
  2. every subject's global rows are contiguous
  3. slf.dense_expected_truth(subj) equals the boolean-mask slice labels[subject==subj]

Fact 3 is what lets the dense array be scattered back into raw coordinates: the dense
stream is the subject's KEPT segments concatenated, so each kept segment's dense slice
maps to its raw slice. At the current dense_min_seg=25 nothing is dropped (all 763
segments are >= 25 samples) and the two coordinate systems coincide -- but the scatter
is done properly anyway, so raising min_seg cannot silently misattribute a session.

THE COMMON COVERED MASK, and why it is not optional
---------------------------------------------------
The two paths do not cover the same samples:

  windowed  loses up to `step` trailing samples per subject, because the last window
            ends before the stream does (slf.expand_last_window_wins leaves them
            UNCOVERED) -- always at the tail of that participant's one session.
  dense     loses whole segments shorter than dense_min_seg (zero of them at 25).

Comparing a windowed estimate against a truth the dense run scored but the windowed
run never saw would credit dense with a difference that is bookkeeping, not accuracy.
So every paired number is computed on the INTERSECTION of the two covered masks, and
the two paths' truths are asserted equal there before either estimate is counted.

CERTIFICATION IS REUSED WHOLESALE, never restated: the windowed path goes through
slf.build_candidates -> slf.resolve_fold (window count + exact last-sample-label
equality), the dense path through _dense_common.load_dense_run (y_true equal to the
raw per-sample label stream, sample for sample). A wrong fold->subject mapping, a
wrong min_segment_len, or a different label encoding fails one of those before any
session is formed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_purity as wp          # noqa: E402  -- module-level defs, __main__-guarded
import sample_level_f1 as slf       # noqa: E402  -- certified reconstruction, reused wholesale
import _dense_common as dc          # noqa: E402  -- certified dense loader, reused wholesale

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
HardFail = wp.HardFail
fail = slf.fail
SAMPLING_RATE = slf.SAMPLING_RATE
UNCOVERED = slf.UNCOVERED

# One session per participant (each participant is one recording), so the baseline's
# session count is its fold count and the full dataset's is 24.
EXPECTED_SESSION_COUNT_ALL = 24
EXPECTED_SESSION_COUNT_BASELINE = len(wp.EXPECTED_FOLDS)

# The five basketball classes; rebound/layup/shot are the jump-and-land events an
# injury model is built around. Spelled as in boracle_duration_bias.py so the two
# scripts' tables can be read side by side.
BALL_ACTIONS = ("dribbling", "shot", "pass", "rebound", "layup")
EXPLOSIVE = ("shot", "rebound", "layup")


def secs(samples):
    """Sample count -> seconds. n_samples / SAMPLING_RATE, the only conversion used."""
    return np.asarray(samples, dtype=float) / SAMPLING_RATE


def hms(seconds):
    s = int(round(float(seconds)))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def rel_err_pct(est_samples, true_samples):
    """(est - true) / true * 100, NaN where truth is zero. Never a silent inf."""
    est, true = np.asarray(est_samples, float), np.asarray(true_samples, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(true > 0, (est - true) / true * 100.0, np.nan)


# --------------------------------------------------------------------------- #
# seam-map geometry
# --------------------------------------------------------------------------- #

def check_seam_geometry(segments, labels_df, total_samples):
    """
    The three facts the raw-subject-local coordinate system rests on.

    Asserted once per run rather than trusted: if any of them fails, every session
    boundary below is wrong and no number this module produces means anything.
    """
    for i in range(len(segments) - 1):
        if segments[i]["end_row"] != segments[i + 1]["start_row"]:
            fail(f"seam map segments do not tile: segment {i} ends at "
                 f"{segments[i]['end_row']} but segment {i + 1} starts at "
                 f"{segments[i + 1]['start_row']}. Session bounds would be wrong.")
    if segments[0]["start_row"] != 0 or segments[-1]["end_row"] != total_samples:
        fail(f"seam map spans rows [{segments[0]['start_row']}, {segments[-1]['end_row']}) "
             f"but the stream holds {total_samples} samples.")

    subj_col = labels_df["subject"].to_numpy()
    for subj in sorted(set(subj_col)):
        idx = np.flatnonzero(subj_col == subj)
        if idx[-1] - idx[0] + 1 != len(idx):
            fail(f"subject {subj}'s rows are not contiguous in the label export (span "
                 f"{idx[-1] - idx[0] + 1} rows for {len(idx)} samples). The raw "
                 "subject-local coordinate system does not exist for this data.")


def sessions_of(subject, segments):
    """
    recording -> (local_start, local_end) in RAW subject-local coordinates.

    Half-open, and asserted contiguous and tiling: a session that is not one interval
    means the seam map interleaves recordings within a subject, which the windowed
    path could not attribute at all.
    """
    base = min(s["start_row"] for s in segments if s["subject_name"] == subject)
    end = max(s["end_row"] for s in segments if s["subject_name"] == subject)
    bounds = {}
    for s in segments:
        if s["subject_name"] != subject:
            continue
        lo, hi = s["start_row"] - base, s["end_row"] - base
        if s["recording"] in bounds:
            b = bounds[s["recording"]]
            bounds[s["recording"]] = (min(b[0], lo), max(b[1], hi))
        else:
            bounds[s["recording"]] = (lo, hi)

    ordered = sorted(bounds.items(), key=lambda kv: kv[1][0])
    covered = 0
    cursor = 0
    for rec, (lo, hi) in ordered:
        if lo != cursor:
            fail(f"subject {subject}: session {rec} starts at local {lo} but the previous "
                 f"session ends at {cursor}. Sessions do not tile the subject's stream.")
        cursor = hi
        covered += hi - lo
    if cursor != end - base or covered != end - base:
        fail(f"subject {subject}: sessions tile {covered} samples but the subject holds "
             f"{end - base}.")
    return ordered


# --------------------------------------------------------------------------- #
# per-path loading, both into raw subject-local coordinates
# --------------------------------------------------------------------------- #

def _windowed_subject_arrays(results_dir, labels_df, npz_pattern, allow_partial):
    """
    subject -> (pred, true) in raw subject-local coordinates, pred UNCOVERED past the
    last window. Reconstruction certified by slf.resolve_fold, unchanged.
    """
    cfg = slf.load_cfg(results_dir)
    if cfg.get("dense", False):
        fail(f"{results_dir}: cfg.txt has dense=true, but this argument wants the WINDOWED "
             "baseline. Refusing to read a per-sample npz as windowed.")

    sw_length, sw_overlap, win_len, step = slf.derive_windowing(cfg, results_dir)
    folds = cfg.get("loso_subjects") or list(wp.EXPECTED_FOLDS)
    expect = len(folds)
    if not allow_partial and len(folds) not in (slf.EXPECTED_FOLD_COUNT, slf.ALL_SUBJECT_FOLD_COUNT):
        fail(f"{results_dir}: cfg loso_subjects lists {len(folds)} folds, expected "
             f"{slf.EXPECTED_FOLD_COUNT} (the LOSO baseline) or {slf.ALL_SUBJECT_FOLD_COUNT} "
             f"(every subject): {folds}. Pass --allow_partial_folds for a smoke test.")

    npz_paths = slf.discover_fold_npz(results_dir, folds, npz_pattern, expect_count=expect)
    candidates = slf.build_candidates(labels_df, win_len, step)

    out = {}
    for fold in sorted(npz_paths):
        y_pred_win, y_true_win = slf.load_npz(npz_paths[fold])
        subj = slf.resolve_fold(fold, y_true_win, candidates, results_dir, win_len, step)
        if subj in out:
            fail(f"{results_dir}: folds {fold} and an earlier one both resolve to subject "
                 f"{subj}; the fold -> subject resolution is not a bijection.")

        n_samples = candidates[subj]["n_samples"]
        y_true = labels_df.loc[labels_df["subject"] == subj, "label_id"].to_numpy().astype(np.int64)
        if len(y_true) != n_samples:
            fail(f"fold {fold} -> {subj}: raw label slice is {len(y_true)} rows, candidate "
                 f"table says {n_samples}")

        # Restate the certification against the raw timeline, as boracle_duration_bias does.
        idx_last = np.arange(len(y_true_win), dtype=np.int64) * step + win_len - 1
        if not np.array_equal(y_true[idx_last], y_true_win):
            n_bad = int((y_true[idx_last] != y_true_win).sum())
            fail(f"{results_dir} fold {fold} -> {subj}: reconstructed last-sample labels differ "
                 f"from the npz y_true at {n_bad} of {len(y_true_win)} windows. "
                 "Reconstruction not certified.")

        pred = slf.expand_last_window_wins(y_pred_win, n_samples, win_len, step)
        out[subj] = {"pred": pred, "true": y_true, "fold": fold,
                     "npz": os.path.basename(npz_paths[fold])}

    return out, {"sw_length": sw_length, "sw_overlap": sw_overlap, "win_len": win_len,
                 "step": step, "cfg": cfg, "name": slf.display_name(results_dir),
                 "results_dir": results_dir, "dense": False}


def _dense_subject_arrays(results_dir, labels_path, seam_path, segments, npz_pattern,
                          allow_partial):
    """
    subject -> (pred, true) in raw subject-local coordinates, scattered out of the dense
    (kept-segment) coordinate system. Certification is _dense_common's, unchanged.
    """
    run = dc.load_dense_run(results_dir, labels_path, seam_path, npz_pattern, allow_partial)

    out = {}
    for f in run["folds"]:
        subj = f["fold"]
        base = min(s["start_row"] for s in segments if s["subject_name"] == subj)
        n_raw = max(s["end_row"] for s in segments if s["subject_name"] == subj) - base

        pred = np.full(n_raw, UNCOVERED, dtype=np.int64)
        true = np.full(n_raw, UNCOVERED, dtype=np.int64)
        for s in f["segs"]:
            lo = s["start_row"] - base
            hi = lo + s["length"]
            d = slice(s["local_start"], s["local_end"])
            pred[lo:hi] = f["y_pred"][d]
            true[lo:hi] = f["y_true"][d]

        n_scattered = int((true != UNCOVERED).sum())
        if n_scattered != f["n_samples"]:
            fail(f"{results_dir} fold {subj}: scattered {n_scattered} samples back into raw "
                 f"coordinates but the certified stream holds {f['n_samples']}. The dense "
                 "-> raw mapping is not a bijection.")
        out[subj] = {"pred": pred, "true": true, "fold": subj, "npz": f["npz"]}

    return out, {"seq_len": run["seq_len"], "min_segment_len": run["min_segment_len"],
                 "cfg": run["cfg"], "name": run["name"], "results_dir": results_dir,
                 "dense": True}


def load_path(results_dir, labels_df, segments, labels_path, seam_path, dense,
              npz_pattern=None, allow_partial=False):
    """One prediction path (windowed or dense) as subject -> raw-local arrays + metadata."""
    if not os.path.isdir(results_dir):
        fail(f"results dir does not exist or is not a directory: {results_dir}")
    if dense:
        return _dense_subject_arrays(results_dir, labels_path, seam_path, segments,
                                     npz_pattern, allow_partial)
    return _windowed_subject_arrays(results_dir, labels_df, npz_pattern, allow_partial)


# --------------------------------------------------------------------------- #
# session tables
# --------------------------------------------------------------------------- #

def sessions_one_path(subject_arrays, segments):
    """
    One path -> {session: {"subject", "true", "est", "n_covered"}}, counts in SAMPLES.

    Scored on that path's own covered mask. Use paired_sessions() for anything that
    compares two paths.
    """
    out = {}
    for subj in sorted(subject_arrays):
        a = subject_arrays[subj]
        cov = (a["pred"] != UNCOVERED) & (a["true"] != UNCOVERED)
        for rec, (lo, hi) in sessions_of(subj, segments):
            c = cov[lo:hi]
            out[rec] = {
                "subject": subj,
                "true": np.bincount(a["true"][lo:hi][c], minlength=N_CLASSES).astype(np.int64),
                "est": np.bincount(a["pred"][lo:hi][c], minlength=N_CLASSES).astype(np.int64),
                "n_covered": int(c.sum()),
                "n_raw": int(hi - lo),
            }
    return out


def paired_sessions(win_arrays, den_arrays, segments):
    """
    {session: {"subject", "true", "win_est", "den_est", "n_common", "n_raw",
               "n_win_only", "n_den_only"}}, counts in SAMPLES, on the COMMON covered mask.

    Both paths' truths are asserted equal on that mask before either estimate is
    counted -- they are reconstructed independently, so agreement there is a real
    check on both reconstructions and not a tautology.
    """
    shared = sorted(set(win_arrays) & set(den_arrays))
    if not shared:
        fail(f"the two runs share no subject: windowed covers {sorted(win_arrays)}, "
             f"dense covers {sorted(den_arrays)}.")

    out = {}
    for subj in shared:
        w, d = win_arrays[subj], den_arrays[subj]
        if len(w["pred"]) != len(d["pred"]):
            fail(f"subject {subj}: windowed stream is {len(w['pred'])} samples, dense is "
                 f"{len(d['pred'])}. Both are raw subject-local; they must be equal.")

        w_cov = (w["pred"] != UNCOVERED) & (w["true"] != UNCOVERED)
        d_cov = (d["pred"] != UNCOVERED) & (d["true"] != UNCOVERED)
        common = w_cov & d_cov

        if not np.array_equal(w["true"][common], d["true"][common]):
            n_bad = int((w["true"][common] != d["true"][common]).sum())
            fail(f"subject {subj}: the windowed and dense reconstructions disagree about the "
                 f"ground truth at {n_bad} of {int(common.sum())} commonly covered samples. "
                 "One of the two reconstructions is wrong; refusing to compare them.")

        for rec, (lo, hi) in sessions_of(subj, segments):
            c = common[lo:hi]
            if not c.any():
                fail(f"session {rec}: no sample is covered by both paths; it cannot be scored.")
            out[rec] = {
                "subject": subj,
                "true": np.bincount(w["true"][lo:hi][c], minlength=N_CLASSES).astype(np.int64),
                "win_est": np.bincount(w["pred"][lo:hi][c], minlength=N_CLASSES).astype(np.int64),
                "den_est": np.bincount(d["pred"][lo:hi][c], minlength=N_CLASSES).astype(np.int64),
                "n_common": int(c.sum()),
                "n_raw": int(hi - lo),
                "n_win_only": int((w_cov[lo:hi] & ~d_cov[lo:hi]).sum()),
                "n_den_only": int((d_cov[lo:hi] & ~w_cov[lo:hi]).sum()),
            }

    # Conservation: on the common mask every sample carries exactly one prediction and
    # one truth, so all three vectors must sum to the same count, per session.
    for rec, r in out.items():
        for key in ("true", "win_est", "den_est"):
            if int(r[key].sum()) != r["n_common"]:
                fail(f"session {rec}: {key} sums to {int(r[key].sum())} samples but the common "
                     f"mask holds {r['n_common']}. The session bincounts do not conserve.")
    return out


def check_session_count(sessions, allow_partial, label=""):
    """A session set that is neither the baseline's sessions nor all 24 is a hard fail."""
    n = len(sessions)
    if not allow_partial and n not in (EXPECTED_SESSION_COUNT_BASELINE,
                                       EXPECTED_SESSION_COUNT_ALL):
        fail(f"{label}resolved {n} sessions ({sorted(sessions)}), expected "
             f"{EXPECTED_SESSION_COUNT_BASELINE} (the LOSO baseline, one session per "
             f"participant) or {EXPECTED_SESSION_COUNT_ALL} (all subjects). "
             "Pass --allow_partial_folds to score a partial run anyway.")
    return n


def add_session_args(parser):
    """Flags shared by the session diagnostics, spelled as in _dense_common.add_common_args."""
    parser.add_argument("--labels", default="labels_export.csv.gz",
                        help="Exported per-sample labels (columns: subject, label), "
                             "original row order.")
    parser.add_argument("--seam_map", default="data/seam_map.json",
                        help="Segment map from scripts/build_seam_map.py; supplies the "
                             "`recording` field this analysis calls a session.")
    parser.add_argument("--allow_partial_folds", action="store_true",
                        help="Score a run covering neither 5 nor 24 folds (a smoke test).")
    return parser


def load_common(args):
    """labels_df + certified seam-map segments, with the geometry checks applied."""
    labels_df = wp.load_labels(args.labels)
    seam = slf.load_seam_map(args.seam_map)
    segments = seam["segments"]

    if len(labels_df) != seam["total_samples"]:
        fail(f"labels file holds {len(labels_df)} samples, seam map describes "
             f"{seam['total_samples']}. They must be the same stream in the same order.")
    subj_col = labels_df["subject"].to_numpy()
    for seg in segments:
        if subj_col[seg["start_row"]] != seg["subject_name"]:
            fail(f"seam map segment at row {seg['start_row']} says subject "
                 f"{seg['subject_name']!r} but the labels file says "
                 f"{subj_col[seg['start_row']]!r}; the two are not aligned.")

    check_seam_geometry(segments, labels_df, seam["total_samples"])
    return labels_df, segments


def session_header(sessions, extra=()):
    """The session inventory every diagnostic prints before its numbers."""
    days = {}
    for rec in sorted(sessions):
        days.setdefault(rec.rsplit("_", 1)[-1], []).append(rec)
    print(f"sessions       : {len(sessions)} recordings over "
          f"{len({r['subject'] for r in sessions.values()})} subjects")
    for day, recs in sorted(days.items()):
        print(f"  capture day  : {day:<4} {len(recs)} session(s)  {', '.join(sorted(recs))}")
    for line in extra:
        print(line)
