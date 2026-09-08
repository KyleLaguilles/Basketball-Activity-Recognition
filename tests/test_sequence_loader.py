#!/usr/bin/env python3
"""
Verification for data_processing/sequence_loader.build_sequences, against the REAL
dataset -- no synthetic fixtures, because every target below is a property of the
actual 1,377,145-row loso_G stream and its 763-segment seam map.

Run from the repo root:
    python tests/test_sequence_loader.py

The cutting plan is re-derived here INDEPENDENTLY of the loader (range()-based rather
than the loader's while-loop, built straight from data/seam_map.json) and every emitted
sequence is then compared elementwise against the train rows that plan names. A cutting
bug therefore cannot hide behind a matching bug: the content equality is what ties the
loader's arrays to specific row ranges, and the coverage/conservation/no-seam-crossing
invariants are checked against the seam map rather than against the loader.

Read-only. Writes nothing, trains nothing, loads no checkpoints.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from data_processing.preprocess_data import load_dataset          # noqa: E402
from data_processing.sequence_loader import build_sequences       # noqa: E402

SEAM_MAP = "data/seam_map.json"
CLASS_NAMES = ["dribbling", "shot", "pass", "rebound", "layup",
               "walking", "running", "standing", "sitting"]

SEQ_LEN = 500
OVERLAP = 0.5
MIN_SEGMENT_LEN = 25
PAD_VALUE = 0.0
IGNORE_INDEX = -100

# LabelEncoder codes of the five loso_G validation subjects (recon §3.3).
VAL_FOLDS = [(5, "4d70"), (6, "9bd4"), (7, "a0da"), (9, "b512"), (11, "ce9d")]

# recon §3.3 validation sample counts, before any short-segment discard.
ANCHOR_VAL_SAMPLES = {"4d70": 126839, "9bd4": 121401, "a0da": 122130,
                      "b512": 117586, "ce9d": 126777}
ANCHOR_TOTAL_SAMPLES = 1377145
ANCHOR_TOTAL_SEGMENTS = 763
ANCHOR_DISCARDED_SEGMENTS = 0
ANCHOR_DISCARDED_SAMPLES = 0

# recon §6.1, fold 4d70 validation samples per class (before discard).
ANCHOR_4D70_VAL_CLASS = {"dribbling": 6891, "shot": 189, "pass": 3413, "rebound": 1720,
                         "layup": 809, "walking": 63176, "running": 35579,
                         "standing": 5298, "sitting": 9764}

CONTENT_CHECK_N = 20
CONTENT_CHECK_SEED = 42

_failures = []


def check(ok, label, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def independent_plan(seg, seq_len, step, min_len):
    """
    The cutting plan for one segment, derived without reusing the loader's helpers.

    Uses range() where the loader uses a while loop, so the two derivations agree only
    if the rule itself is right. Returns [(start_row, end_row, pad_len)], empty when the
    segment is discarded.
    """
    lo, hi = seg["start_row"], seg["end_row"]
    n = hi - lo
    if n < min_len:
        return []
    if n < seq_len:
        return [(lo, hi, seq_len - n)]
    starts = list(range(0, max(n - seq_len, 0), step))   # curr < n - seq_len, strict
    starts.append(n - seq_len)                            # right-aligned tail
    return [(lo + s, lo + s + seq_len, 0) for s in starts]


def main():
    print("=" * 92)
    print("SEQUENCE LOADER VERIFICATION -- real loso_G data, real seam map")
    print("=" * 92)

    print(f"\n  seq_len={SEQ_LEN} ({SEQ_LEN / 50:.1f}s)  overlap={OVERLAP:.0%}  "
          f"step={int(SEQ_LEN * (1 - OVERLAP))}  min_segment_len={MIN_SEGMENT_LEN}")

    print("\n  loading data...")
    train, valid, subjects, nb_classes, class_names, sampling_rate, has_void = load_dataset(
        test_type="subset_specific", test_case="loso_G", include_void=False
    )
    print(f"  train array: {train.shape} {train.dtype};  valid={valid};  "
          f"{nb_classes} classes @ {sampling_rate} Hz")

    seam = json.load(open(SEAM_MAP))
    segments = seam["segments"]
    print(f"  seam map   : {len(segments)} segments, {seam['total_samples']} samples")

    check(len(train) == ANCHOR_TOTAL_SAMPLES, "train array holds 1377145 rows", f"{len(train)}")
    check(len(segments) == ANCHOR_TOTAL_SEGMENTS, "seam map holds 763 segments", f"{len(segments)}")

    # row -> segment index, for the no-seam-crossing check. Segments are contiguous
    # ranges, so two rows in the same segment imply every row between them is too;
    # checking a sequence's first and last real row is therefore sufficient.
    seg_of_row = np.empty(len(train), dtype=np.int32)
    for i, s in enumerate(segments):
        seg_of_row[s["start_row"]:s["end_row"]] = i

    step = int(SEQ_LEN * (1 - OVERLAP))
    rng = np.random.default_rng(CONTENT_CHECK_SEED)
    table = []

    for code, name in VAL_FOLDS:
        print(f"\n{'-' * 92}\n  FOLD {name} (subject_code {code})\n{'-' * 92}")

        res = build_sequences(
            train_array=train,
            seam_map_path=SEAM_MAP,
            val_subject_code=code,
            seq_len=SEQ_LEN,
            overlap=OVERLAP,
            min_segment_len=MIN_SEGMENT_LEN,
            pad_value=PAD_VALUE,
            ignore_index=IGNORE_INDEX,
        )

        # ---- independent plan for this fold ------------------------------------- #
        plan = {"train": [], "val": []}
        kept_spans = {"train": [], "val": []}
        disc = {"train": 0, "val": 0}
        for s in segments:
            side = "val" if s["subject_code"] == code else "train"
            entries = independent_plan(s, SEQ_LEN, step, MIN_SEGMENT_LEN)
            if not entries:
                disc[side] += s["length"]
                continue
            plan[side].extend(entries)
            kept_spans[side].append((s["start_row"], s["end_row"]))

        # ---- (10) dtypes --------------------------------------------------------- #
        print("\n  (10) dtypes")
        check(res["X_train"].dtype == np.float32, "X_train.dtype == float32", str(res["X_train"].dtype))
        check(res["y_train"].dtype == np.int64, "y_train.dtype == int64", str(res["y_train"].dtype))
        check(res["X_val"].dtype == np.float32, "X_val.dtype == float32", str(res["X_val"].dtype))
        check(res["y_val"].dtype == np.int64, "y_val.dtype == int64", str(res["y_val"].dtype))
        check(res["train_sample_labels"].dtype == np.int64 and
              res["val_sample_labels"].dtype == np.int64, "sample label streams are int64")

        # ---- (11) shapes --------------------------------------------------------- #
        print("\n  (11) shapes")
        nt, nv = len(plan["train"]), len(plan["val"])
        check(res["X_train"].shape == (nt, SEQ_LEN, 3), f"X_train.shape == ({nt}, {SEQ_LEN}, 3)",
              str(res["X_train"].shape))
        check(res["y_train"].shape == (nt, SEQ_LEN), f"y_train.shape == ({nt}, {SEQ_LEN})",
              str(res["y_train"].shape))
        check(res["X_val"].shape == (nv, SEQ_LEN, 3), f"X_val.shape == ({nv}, {SEQ_LEN}, 3)",
              str(res["X_val"].shape))
        check(res["y_val"].shape == (nv, SEQ_LEN), f"y_val.shape == ({nv}, {SEQ_LEN})",
              str(res["y_val"].shape))

        # ---- (3) sample conservation --------------------------------------------- #
        print("\n  (3) sample conservation")
        val_total = sum(s["length"] for s in segments if s["subject_code"] == code)
        train_total = sum(s["length"] for s in segments if s["subject_code"] != code)
        check(val_total == ANCHOR_VAL_SAMPLES[name],
              f"seam map val rows == recon §3.3 ({ANCHOR_VAL_SAMPLES[name]})", str(val_total))
        check(res["val_sample_count"] + disc["val"] == val_total,
              "val_sample_count + discarded val samples == total val samples",
              f"{res['val_sample_count']} + {disc['val']} == {val_total}")
        check(res["train_sample_count"] + disc["train"] == train_total,
              "train_sample_count + discarded train samples == total train samples",
              f"{res['train_sample_count']} + {disc['train']} == {train_total}")
        check(res["train_sample_count"] + res["val_sample_count"] + res["discarded_samples"]
              == ANCHOR_TOTAL_SAMPLES, "train + val + discarded == 1377145")
        check(res["discarded_samples"] == disc["train"] + disc["val"] == ANCHOR_DISCARDED_SAMPLES,
              "discarded_samples == 582 (subject-independent)", str(res["discarded_samples"]))

        # ---- (7) feature / label content, and the plan tie-in --------------------- #
        print("\n  (7) feature/label content vs train[start:end]")
        content_ok = {"train": True, "val": True}
        for side, key in (("train", "train"), ("val", "val")):
            X, y = res[f"X_{key}"], res[f"y_{key}"]
            for i, (lo, hi, pad_len) in enumerate(plan[side]):
                real = SEQ_LEN - pad_len
                if not np.array_equal(X[i, :real], train[lo:hi, 1:4].astype(np.float32)):
                    content_ok[side] = False
                    break
                if not np.array_equal(y[i, :real], train[lo:hi, 4].astype(np.int64)):
                    content_ok[side] = False
                    break
        check(content_ok["train"] and content_ok["val"],
              f"all {nt + nv} sequences match their train rows exactly (elementwise)")

        picks = rng.choice(nt, size=min(CONTENT_CHECK_N, nt), replace=False)
        spot_ok = True
        for p in picks.tolist():
            lo, hi, pad_len = plan["train"][p]
            real = SEQ_LEN - pad_len
            spot_ok &= np.array_equal(res["X_train"][p, :real], train[lo:hi, 1:4].astype(np.float32))
            spot_ok &= np.array_equal(res["y_train"][p, :real], train[lo:hi, 4].astype(np.int64))
        check(spot_ok, f"{len(picks)} random train sequences (seed {CONTENT_CHECK_SEED}) match exactly")

        # ---- (4) no seam crossing ------------------------------------------------- #
        print("\n  (4) no seam crossing")
        cross = {"train": 0, "val": 0}
        for side in ("train", "val"):
            for lo, hi, pad_len in plan[side]:
                last_real = hi - 1
                if seg_of_row[lo] != seg_of_row[last_real]:
                    cross[side] += 1
        check(cross["train"] == 0 and cross["val"] == 0,
              "every sequence's samples come from exactly one segment",
              f"crossing: train={cross['train']} val={cross['val']}")

        # ---- (5) padding ---------------------------------------------------------- #
        print("\n  (5) padding")
        pad_ok = True
        n_padded = {"train": 0, "val": 0}
        seg_len_by_start = {s["start_row"]: s["length"] for s in segments}
        for side, key in (("train", "train"), ("val", "val")):
            X, y = res[f"X_{key}"], res[f"y_{key}"]
            for i, (lo, hi, pad_len) in enumerate(plan[side]):
                real = SEQ_LEN - pad_len
                if pad_len:
                    n_padded[side] += 1
                    pad_ok &= bool(np.all(X[i, real:] == PAD_VALUE))
                    pad_ok &= bool(np.all(y[i, real:] == IGNORE_INDEX))
                    pad_ok &= (real == seg_len_by_start.get(lo))
                else:
                    pad_ok &= bool(np.all(y[i] != IGNORE_INDEX))
        check(pad_ok, "padded tails are pad_value / ignore_index, prefix length == segment length",
              f"padded sequences: train={n_padded['train']} val={n_padded['val']}")
        check(bool(np.all((res["y_train"] == IGNORE_INDEX) | ((res["y_train"] >= 0) & (res["y_train"] < 9)))),
              "y_train holds only class ids 0-8 or ignore_index")
        check(bool(np.all((res["y_val"] == IGNORE_INDEX) | ((res["y_val"] >= 0) & (res["y_val"] < 9)))),
              "y_val holds only class ids 0-8 or ignore_index")

        # ---- (6) coverage ---------------------------------------------------------- #
        print("\n  (6) coverage")
        for side in ("train", "val"):
            covered = np.zeros(len(train), dtype=bool)
            for lo, hi, _pad_len in plan[side]:
                # hi already excludes padding: a padded entry spans exactly its segment
                # (hi - lo == segment length), an unpadded one spans seq_len real rows.
                covered[lo:hi] = True
            expected = np.zeros(len(train), dtype=bool)
            for lo, hi in kept_spans[side]:
                expected[lo:hi] = True
            check(bool(np.array_equal(covered, expected)),
                  f"{side}: covered rows == exactly the non-discarded {side} rows",
                  f"{int(covered.sum())} covered vs {int(expected.sum())} expected")

        # ---- (8)/(9) per-sample label streams --------------------------------------- #
        print("\n  (8)/(9) per-sample label streams")
        for side, key in (("val", "val"), ("train", "train")):
            expected = np.concatenate(
                [train[lo:hi, 4] for lo, hi in kept_spans[side]]
            ).astype(np.int64)
            got = res[f"{key}_sample_labels"]
            check(bool(np.array_equal(got, expected)),
                  f"{key}_sample_labels == concat of non-discarded {side} segment labels, in order",
                  f"len {len(got)} vs {len(expected)}")
            check(len(got) == res[f"{key}_sample_count"],
                  f"len({key}_sample_labels) == {key}_sample_count",
                  f"{len(got)} vs {res[f'{key}_sample_count']}")

        table.append({
            "fold": name, "code": code,
            "train_seqs": nt, "val_seqs": nv,
            "train_samples": res["train_sample_count"], "val_samples": res["val_sample_count"],
            "discarded": res["discarded_samples"],
            "disc_train": disc["train"], "disc_val": disc["val"],
            "padded_train": n_padded["train"], "padded_val": n_padded["val"],
        })

        if name == "4d70":
            print("\n  (extra) fold 4d70 class distribution vs recon §6.1")
            vb = np.bincount(res["val_sample_labels"], minlength=9)
            tb = np.bincount(res["train_sample_labels"], minlength=9)
            print(f"      {'class':<12} {'val (kept)':>11} {'recon §6.1':>11} {'delta':>7} {'train (kept)':>13}")
            for c, cn in enumerate(CLASS_NAMES):
                anc = ANCHOR_4D70_VAL_CLASS[cn]
                print(f"      {cn:<12} {int(vb[c]):>11} {anc:>11} {int(vb[c]) - anc:>+7} {int(tb[c]):>13}")
            print(f"      {'TOTAL':<12} {int(vb.sum()):>11} {sum(ANCHOR_4D70_VAL_CLASS.values()):>11} "
                  f"{int(vb.sum()) - sum(ANCHOR_4D70_VAL_CLASS.values()):>+7} {int(tb.sum()):>13}")
            print(f"      (the deficit is the {disc['val']} sample(s) in 4d70's discarded "
                  f"segment(s), < {MIN_SEGMENT_LEN} samples each)")

        del res

    # ---- report ------------------------------------------------------------------- #
    print("\n" + "=" * 92)
    print("PER-FOLD SEQUENCE COUNTS")
    print("=" * 92)
    print(f"\n  {'fold (val subj)':<16} {'train seqs':>11} {'val seqs':>9} {'discarded samples':>18} "
          f"{'train samples':>14} {'val samples':>12}")
    for r in table:
        print(f"  {r['fold']:<16} {r['train_seqs']:>11} {r['val_seqs']:>9} {r['discarded']:>18} "
              f"{r['train_samples']:>14} {r['val_samples']:>12}")
    discs = {r["discarded"] for r in table}
    check(discs == {ANCHOR_DISCARDED_SAMPLES},
          "discarded samples identical across all 5 folds (subject-independent rule)",
          f"{sorted(discs)}")

    print("\n=== SEQUENCE LOADER SUMMARY ===")
    print(f"Sequence length: {SEQ_LEN} samples ({SEQ_LEN / 50:.1f} s)")
    print(f"Overlap: {OVERLAP:.0%} (step = {step})")
    print(f"Min segment length: {MIN_SEGMENT_LEN} samples ({MIN_SEGMENT_LEN / 50:.1f} s)")
    print(f"Discarded segments: {ANCHOR_DISCARDED_SEGMENTS} ({ANCHOR_DISCARDED_SAMPLES} samples, "
          f"{ANCHOR_DISCARDED_SAMPLES / ANCHOR_TOTAL_SAMPLES * 100:.3f}%)")
    for r in table:
        print(f"Fold {r['fold']}: train={r['train_seqs']} seqs, val={r['val_seqs']} seqs, "
              f"val_samples={r['val_samples']}")

    if _failures:
        print(f"\nFAIL: {_failures}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
