#!/usr/bin/env python3
"""
Build data/seam_map.json -- the contiguous-segment map of the loso_G `train` array.

WHY. `load_dataset('subset_specific', 'loso_G')` returns one row per 50 Hz sample,
but that row stream is NOT continuous in time. Rows are deleted upstream, at CSV
creation time, by data_processing/data_creation.py:

    :46   sbj_data = sbj_data[(sbj_data['coarse'] != 'not_labeled')]
    :66-67 mask = sbj_data["basketball"].eq("not_labeled")
           sbj_data.loc[mask, "basketball"] = sbj_data.loc[mask, "locomotion"]
    :69   sbj_data = sbj_data[(sbj_data['basketball'] != 'not_labeled') &
                              (sbj_data['basketball'] != 'jumping')]
    :73   sbj_output_game = sbj_data[(sbj_data['coarse'] == 'game')]...

Every deletion splices two non-adjacent moments in time into adjacent rows. The
timestamp column that would reveal this is dropped at :52 and never reaches the
training pipeline, so a dense sequence cutter has no way to see the discontinuity.
This script recovers the seams from data/raw/*.csv and expresses them in train-array
row coordinates.

WHY THE RAW FILES AND NOT data/hangtime_game_data.csv. The game CSV is the deletion's
output: the seams are exactly the information it lost. Only data/raw/*.csv still holds
the timestamps. The game CSV is read here for VERIFICATION only (block sizes, acc/label
equality), never as the source of the segmentation.

THE COORDINATE CHAIN, and why game-CSV row i == train row i:
  1. data_creation.py:26  iterates `sorted(glob('data/raw/*.csv'))`, and :81-83
     `pd.concat`s each file's surviving game rows in that order -- so each raw file
     owns one contiguous block of the game CSV, blocks in sorted-filename order.
  2. preprocess_data.py:37 reads the game CSV with header=None (no reindex, no sort),
     :97 slices columns `iloc[:, 3:]`, :98/:112 build `train`. No row is reordered.
  3. preprocess_data.py:150 would drop `void_class` rows -- the game CSV contains
     zero of them (asserted below), so the row count and order survive intact.
Therefore train row i is game-CSV row i is the i-th surviving raw row in file order.

SEAM DEFINITION. Within a raw file, a seam sits between consecutive surviving game
rows whose raw timestamps differ by more than one sample period. Sensor cadence is a
clean 20 ms (verified: zero inter-sample gaps > 25 ms in the unfiltered raw stream
across all 24 files), so GAP_THRESHOLD_MS = 25 admits 20 ms and rejects >= 40 ms.
Recording boundaries are ALWAYS seams. Each recording is its own participant --
subject_code is keyed on the full filename stem <id>_<eu|na>, because the 4-hex id is
only unique within a site (data/raw/meta.txt) -- so these are the 23
participant-to-participant junctions.

Read-only apart from data/seam_map.json. Imports nothing from src/ -- the filters are
replayed independently, so the seam map does not depend on the training pipeline.

Usage (from repo root):
    python scripts/build_seam_map.py [--out data/seam_map.json]
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
GAME_CSV = "data/hangtime_game_data.csv"
DRILL_CSV = "data/hangtime_drill_data.csv"
WARMUP_CSV = "data/hangtime_warmup_data.csv"

SAMPLING_RATE_HZ = 50
SAMPLE_PERIOD_MS = 1000 // SAMPLING_RATE_HZ      # 20
GAP_THRESHOLD_MS = 25                             # > this == at least one deleted sample
SCHEMA_VERSION = 1

# data_creation.py:38 wrote the filename's 'na' site suffix into the location column as 'us'.
LOCATION_TO_SUFFIX = {"eu": "eu", "us": "na"}

# ---- anchors from docs/recon_dense_report.md §2.1 -------------------------------- #
ANCHOR_TOTAL_SAMPLES = 1377145
ANCHOR_TOTAL_SEGMENTS = 763
ANCHOR_SHORT_1500 = 615

ANCHOR_RECORDING_SAMPLES = {
    "05d8_eu": 61755, "0846_eu": 18684, "0846_na": 50885, "10f0_eu": 46161,
    "10f0_na": 65399, "2dd9_eu": 33465, "2dd9_na": 60820, "4991_eu": 67411,
    "4d70_eu": 61539, "4d70_na": 65300, "9bd4_eu": 62717, "9bd4_na": 58684,
    "a0da_eu": 62570, "a0da_na": 59560, "ac59_eu": 63109, "ac59_na": 55082,
    "b512_eu": 50572, "b512_na": 67014, "c6f3_na": 55213, "ce9d_eu": 62697,
    "ce9d_na": 64080, "e90f_eu": 58667, "f2ad_eu": 62122, "f2ad_na": 63639,
}
# One participant == one recording, so the 24 per-subject totals are the recording totals.
ANCHOR_SUBJECT_SAMPLES = ANCHOR_RECORDING_SAMPLES
ANCHOR_RECORDING_SEGMENTS = {
    "05d8_eu": 7, "0846_eu": 4, "0846_na": 98, "10f0_eu": 9, "10f0_na": 20,
    "2dd9_eu": 5, "2dd9_na": 42, "4991_eu": 3, "4d70_eu": 11, "4d70_na": 29,
    "9bd4_eu": 5, "9bd4_na": 146, "a0da_eu": 7, "a0da_na": 78, "ac59_eu": 6,
    "ac59_na": 75, "b512_eu": 6, "b512_na": 6, "c6f3_na": 92, "ce9d_eu": 10,
    "ce9d_na": 41, "e90f_eu": 7, "f2ad_eu": 5, "f2ad_na": 51,
}

SPOT_CHECK_RECORDING = "05d8_eu"
N_CONTENT_CHECKS = 5
CONTENT_CHECK_SEED = 0

_failures = []


def check(ok, label, detail=""):
    """Record and print one verification result. Never raises -- main() exits on any failure."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def fail_now(msg):
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def participant_keys(frame):
    """
    <id>_<eu|na> per row of a header=None hangtime_*_data.csv frame -- the raw filename
    stem, rebuilt from subject (col 3) and location (col 0). Replays
    preprocess_data.participant_keys, since this script imports nothing from src/.
    """
    suffix = frame[0].map(LOCATION_TO_SUFFIX)
    if suffix.isna().any():
        fail_now(f"unrecognized location value(s): {sorted(map(str, set(frame.loc[suffix.isna(), 0])))}")
    return frame[3].astype(str) + "_" + suffix


# --------------------------------------------------------------------------------- #
# replay of data_creation.py
# --------------------------------------------------------------------------------- #

def surviving_game_rows(path):
    """
    The rows of one raw CSV that reach hangtime_game_data.csv, in file order.

    Replicates data_creation.py:46, :66-67, :69, :73 exactly and in the same order.
    Returns the surviving frame with its timestamp column still attached (the column
    data_creation.py:52 drops), which is the whole point of going back to the raw file.
    """
    df = pd.read_csv(
        path,
        usecols=["timestamp", "acc_x", "acc_y", "acc_z", "coarse", "basketball", "locomotion"],
    )

    df = df[(df["coarse"] != "not_labeled")]                        # :46

    df = df.copy()
    mask = df["basketball"].eq("not_labeled")                       # :66
    df.loc[mask, "basketball"] = df.loc[mask, "locomotion"]         # :67

    df = df[(df["basketball"] != "not_labeled") & (df["basketball"] != "jumping")]   # :69
    df = df[(df["coarse"] == "game")]                               # :73

    return df.reset_index(drop=True)


def segment_bounds(ts_ms):
    """
    Local [start, end) segment bounds inside one recording, from its timestamp stream.

    A gap of more than GAP_THRESHOLD_MS between consecutive surviving rows means at
    least one sample was deleted between them, so they are not temporally adjacent and
    a segment ends. The recording's own two ends are always bounds.
    """
    n = len(ts_ms)
    if n == 0:
        return []
    d = np.diff(ts_ms)
    if n > 1 and d.min() <= 0:
        fail_now(f"non-monotonic timestamps: min inter-row delta is {d.min()} ms")
    cuts = np.flatnonzero(d > GAP_THRESHOLD_MS) + 1
    edges = np.concatenate(([0], cuts, [n]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def build_segments():
    """
    Walk every raw file in data_creation.py:26 order, accumulating a running row offset
    in train coordinates, and emit one record per contiguous segment.
    """
    paths = sorted(glob(os.path.join(RAW_DIR, "*.csv")))
    if not paths:
        fail_now(f"no raw CSVs found under {RAW_DIR}/")

    recordings = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    # One participant per recording: the subject is the full stem <id>_<eu|na>, because the
    # 4-hex id is only unique within a site. sorted() order is LabelEncoder order.
    subject_names = sorted(recordings)
    subject_code = {name: i for i, name in enumerate(subject_names)}

    segments = []
    per_recording_rows = {}
    raw_frames = {}
    offset = 0
    seg_idx_by_subject = {}

    for path, rec in zip(paths, recordings):
        sbj = rec
        df = surviving_game_rows(path)
        raw_frames[rec] = df

        ts_ms = pd.to_datetime(df["timestamp"]).to_numpy().astype("datetime64[ms]").astype(np.int64)
        bounds = segment_bounds(ts_ms)

        for lo, hi in bounds:
            k = seg_idx_by_subject.get(sbj, 0)
            seg_idx_by_subject[sbj] = k + 1
            segments.append({
                "subject_code": subject_code[sbj],
                "subject_name": sbj,
                "recording": rec,
                "segment_idx": k,
                "start_row": offset + lo,
                "end_row": offset + hi,
                "length": hi - lo,
                "length_sec": round((hi - lo) / SAMPLING_RATE_HZ, 2),
            })

        per_recording_rows[rec] = len(df)
        offset += len(df)
        print(f"    {rec:<10} rows={len(df):>7}  segments={len(bounds):>4}  "
              f"train rows [{offset - len(df)}, {offset})")

    return segments, per_recording_rows, raw_frames, subject_names, subject_code, offset


# --------------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------------- #

def verify(segments, per_recording_rows, raw_frames, subject_names, subject_code,
           total_rows, game):
    print("\n=== VERIFICATION ===\n")

    # (0) the coordinate-chain assumptions
    print("  (0) coordinate chain: game-CSV row i == train row i")
    n_void = int((game[7] == "void_class").sum())
    check(n_void == 0, "game CSV has no void_class rows (preprocess_data.py:150 drops none)",
          f"found {n_void}")
    check(len(game) == ANCHOR_TOTAL_SAMPLES, "game CSV row count == train row count",
          f"{len(game)} vs anchor {ANCHOR_TOTAL_SAMPLES}")
    game_key = participant_keys(game)
    sg = set(game_key.unique())
    sd = set(participant_keys(pd.read_csv(DRILL_CSV, header=None, usecols=[0, 3],
                                          dtype={0: str, 3: str})).unique())
    sw = set(participant_keys(pd.read_csv(WARMUP_CSV, header=None, usecols=[0, 3],
                                          dtype={0: str, 3: str})).unique())
    union = sd | sw | sg
    check(union == set(subject_names) and len(union) == 24,
          "LabelEncoder alphabet (drill+warmup+game participant keys) == raw filename stems",
          f"{len(union)} subjects")
    order_ok = all(
        game_key.iloc[segments[i]["start_row"]] == segments[i]["subject_name"]
        for i in range(0, len(segments), 37)
    )
    check(order_ok, "sampled segment start rows land on the right subject in the game CSV")

    # (1) total samples
    print("\n  (1) totals")
    tot = sum(s["length"] for s in segments)
    check(tot == ANCHOR_TOTAL_SAMPLES, "sum(segment lengths) == 1377145", f"got {tot}")
    check(total_rows == ANCHOR_TOTAL_SAMPLES, "running row offset ended at 1377145",
          f"got {total_rows}")

    # (2) segment count
    n_seg = len(segments)
    check(n_seg == ANCHOR_TOTAL_SEGMENTS, "segment count == 763", f"got {n_seg}")

    # (3) no gaps, no overlaps -- the segments must tile [0, total_rows) exactly
    print("\n  (3) tiling")
    starts = np.array([s["start_row"] for s in segments])
    ends = np.array([s["end_row"] for s in segments])
    check(starts[0] == 0, "first segment starts at row 0", f"got {starts[0]}")
    check(ends[-1] == ANCHOR_TOTAL_SAMPLES, "last segment ends at row 1377145", f"got {ends[-1]}")
    check(bool(np.all(ends[:-1] == starts[1:])),
          "every segment ends exactly where the next begins (no gap, no overlap)")
    check(bool(np.all(ends > starts)), "every segment is non-empty and end_row > start_row")
    codes = np.array([s["subject_code"] for s in segments])
    check(bool(np.all(np.diff(codes) >= 0)),
          "subject codes are non-decreasing (subject blocks are contiguous in train)")
    idx_ok = True
    seen = {}
    for s in segments:
        k = seen.get(s["subject_name"], 0)
        idx_ok &= (s["segment_idx"] == k)
        seen[s["subject_name"]] = k + 1
    check(idx_ok, "segment_idx runs 0..n-1 within each subject, in row order")

    # (4) per-subject sample totals
    print("\n  (4) per-subject sample totals vs recon §2.1")
    got = {}
    for s in segments:
        got[s["subject_name"]] = got.get(s["subject_name"], 0) + s["length"]
    bad = {k: (got.get(k), v) for k, v in ANCHOR_SUBJECT_SAMPLES.items() if got.get(k) != v}
    check(not bad, "all 24 subject sample totals match", f"mismatches: {bad}" if bad else "")

    # (5) per-recording sample totals
    print("\n  (5) per-recording sample totals vs recon §2.1")
    bad = {k: (per_recording_rows.get(k), v) for k, v in ANCHOR_RECORDING_SAMPLES.items()
           if per_recording_rows.get(k) != v}
    check(not bad, "all 24 recording sample totals match", f"mismatches: {bad}" if bad else "")

    # (6) per-recording segment counts
    print("\n  (6) per-recording segment counts vs recon §2.1")
    seg_per_rec = {}
    for s in segments:
        seg_per_rec[s["recording"]] = seg_per_rec.get(s["recording"], 0) + 1
    bad = {k: (seg_per_rec.get(k), v) for k, v in ANCHOR_RECORDING_SEGMENTS.items()
           if seg_per_rec.get(k) != v}
    check(not bad, "all 24 recording segment counts match", f"mismatches: {bad}" if bad else "")
    check(sum(ANCHOR_RECORDING_SEGMENTS.values()) == ANCHOR_TOTAL_SEGMENTS,
          "anchor arithmetic: 739 splices + 24 recordings == 763 segments")

    # (7) spot-check one recording's boundaries, re-derived independently
    print(f"\n  (7) spot check -- {SPOT_CHECK_RECORDING} boundaries re-derived from raw timestamps")
    rec = SPOT_CHECK_RECORDING
    df = raw_frames[rec]
    ts = pd.to_datetime(df["timestamp"])
    ts_ms = ts.to_numpy().astype("datetime64[ms]").astype(np.int64)
    d = np.diff(ts_ms)
    cut_local = np.flatnonzero(d > GAP_THRESHOLD_MS) + 1
    base = min(s["start_row"] for s in segments if s["recording"] == rec)
    rec_segs = sorted((s for s in segments if s["recording"] == rec), key=lambda s: s["start_row"])
    print(f"      block base row = {base}; {len(cut_local)} splice(s) found:")
    for c in cut_local:
        print(f"        gap {d[c - 1] / 1000:>8.2f}s  {ts.iloc[c - 1]} -> {ts.iloc[c]}"
              f"   train row {base + c}")
    expected_starts = [base] + [base + int(c) for c in cut_local]
    actual_starts = [s["start_row"] for s in rec_segs]
    check(expected_starts == actual_starts,
          f"{rec}: independently derived segment starts match the seam map",
          f"{expected_starts} vs {actual_starts}")
    check(len(rec_segs) == len(cut_local) + 1,
          f"{rec}: {len(cut_local)} splices give {len(cut_local) + 1} segments",
          f"got {len(rec_segs)}")
    check(all(rec_segs[i]["end_row"] == rec_segs[i + 1]["start_row"] for i in range(len(rec_segs) - 1)),
          f"{rec}: internal boundaries are shared (end_row == next start_row)")

    # (8) content check -- the rows a segment addresses are the rows it should address
    print(f"\n  (8) content check -- {N_CONTENT_CHECKS} random segments, acc + label vs raw")
    rng = np.random.default_rng(CONTENT_CHECK_SEED)
    picks = rng.choice(len(segments), size=N_CONTENT_CHECKS, replace=False)
    rec_base = {}
    for s in segments:
        rec_base[s["recording"]] = min(rec_base.get(s["recording"], s["start_row"]), s["start_row"])
    all_ok = True
    for p in sorted(picks.tolist()):
        s = segments[p]
        lo = s["start_row"] - rec_base[s["recording"]]
        raw = raw_frames[s["recording"]].iloc[lo:lo + s["length"]]
        # game CSV cols 4,5,6 are train cols 1,2,3 (preprocess_data.py:97 then :152);
        # train stores float32 (preprocess_data.py:160), so compare in float32.
        blk = game.iloc[s["start_row"]:s["end_row"]]
        acc_ok = np.allclose(
            blk[[4, 5, 6]].to_numpy(dtype=np.float32),
            raw[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32),
            atol=1e-5,
        )
        lbl_ok = bool((blk[7].to_numpy() == raw["basketball"].to_numpy()).all())
        sbj_ok = bool((game_key.iloc[s["start_row"]:s["end_row"]].to_numpy() == s["subject_name"]).all())
        ts_ms_seg = pd.to_datetime(raw["timestamp"]).to_numpy().astype("datetime64[ms]").astype(np.int64)
        cont_ok = len(ts_ms_seg) < 2 or bool(np.all(np.diff(ts_ms_seg) <= GAP_THRESHOLD_MS))
        ok = acc_ok and lbl_ok and sbj_ok and cont_ok
        all_ok &= ok
        print(f"      seg {p:>4} {s['recording']:<9} rows [{s['start_row']}, {s['end_row']}) "
              f"len={s['length']:<6} acc={acc_ok} label={lbl_ok} subject={sbj_ok} contiguous={cont_ok}")
    check(all_ok, "sampled segments address the right rows and are internally gap-free")

    # (9) short-segment census
    print("\n  (9) short-segment census")
    L = np.array([s["length"] for s in segments])
    n1500 = int((L < 1500).sum())
    check(n1500 == ANCHOR_SHORT_1500, "segments < 1500 samples (30 s) == 615", f"got {n1500}")
    for thr, unit in ((1500, "30 s"), (500, "10 s"), (50, "1 s")):
        m = L < thr
        print(f"      < {thr:>4} samples ({unit:>4}): {int(m.sum()):>3} segments, "
              f"{int(L[m].sum()):>7} samples ({L[m].sum() / L.sum() * 100:.3f}% of all samples)")

    return not _failures


def summary(segments, subject_names):
    L = np.array([s["length"] for s in segments])
    tot = int(L.sum())

    def band(m):
        return int(m.sum()), int(L[m].sum()), L[m].sum() / tot * 100

    b_long = band(L >= 500)
    b_mid = band((L >= 50) & (L < 500))
    b_short = band(L < 50)

    per_sub = {}
    for s in segments:
        per_sub[s["subject_name"]] = per_sub.get(s["subject_name"], 0) + 1

    shortest = min(segments, key=lambda s: s["length"])
    longest = max(segments, key=lambda s: s["length"])

    print("\n=== SEAM MAP SUMMARY ===")
    print("File: data/seam_map.json")
    print(f"Total samples: {tot}")
    print(f"Total segments: {len(segments)}")
    print(f"Segments >= 500 samples (10s): {b_long[0]} ({b_long[1]} samples, {b_long[2]:.3f}%)")
    print(f"Segments 50-499 samples: {b_mid[0]} ({b_mid[1]} samples, {b_mid[2]:.3f}%)")
    print(f"Segments < 50 samples (discarded): {b_short[0]} ({b_short[1]} samples, {b_short[2]:.3f}%)")
    print("Per-subject segment counts: " + ", ".join(f"{n}={per_sub.get(n, 0)}" for n in subject_names))
    print(f"Shortest segment: {shortest['length']} samples "
          f"(subject {shortest['subject_name']}, recording {shortest['recording']})")
    print(f"Longest segment: {longest['length']} samples "
          f"(subject {longest['subject_name']}, recording {longest['recording']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/seam_map.json")
    args = ap.parse_args()

    print("=== BUILD SEAM MAP ===")
    print(f"  raw dir          : {RAW_DIR}")
    print(f"  gap threshold    : > {GAP_THRESHOLD_MS} ms "
          f"(sample period {SAMPLE_PERIOD_MS} ms @ {SAMPLING_RATE_HZ} Hz)")
    print(f"  filters replayed : data_creation.py:46, :66-67, :69, :73\n")

    print("  replaying filters over data/raw/*.csv in sorted() order:")
    segments, per_recording_rows, raw_frames, subject_names, subject_code, total_rows = build_segments()

    print("\n  loading game CSV for verification only (never as the segmentation source)...")
    game = pd.read_csv(GAME_CSV, header=None, index_col=None, dtype={3: str})

    if not verify(segments, per_recording_rows, raw_frames, subject_names, subject_code,
                  total_rows, game):
        print(f"\nFAIL: {len(_failures)} verification(s) failed: {_failures}", file=sys.stderr)
        print("Refusing to write the seam map.", file=sys.stderr)
        sys.exit(1)

    payload = {
        "version": SCHEMA_VERSION,
        "total_samples": int(sum(s["length"] for s in segments)),
        "total_segments": len(segments),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "segments": segments,
    }
    with open(args.out, "w") as fid:
        json.dump(payload, fid, indent=2)
    print(f"\nWrote {args.out} ({os.path.getsize(args.out)} bytes)")

    summary(segments, subject_names)


if __name__ == "__main__":
    main()
