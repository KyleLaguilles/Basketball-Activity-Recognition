##################################################
# Seam-aware sequence cutting for the dense (per-sample) prediction path
##################################################
"""
Cut fixed-length training/validation sequences that never cross a seam.

This is the dense path's replacement for data_processing/sliding_window.py's
`apply_sliding_window`. It is NOT a wrapper around it: the windowed path collapses
each window to a single label (sliding_window.py:117 `output_y = [[i[-1]] for i in
output_y]`), which is exactly the labeling artifact the dense path exists to remove.
Here every timestep keeps its own label.

WHY SEAMS. `train` (from preprocess_data.load_dataset) has one row per 50 Hz sample,
but the row stream is not continuous in time. Rows were deleted upstream at CSV
creation (data_creation.py:46, :66-69, :73) and the timestamp column that would reveal
it was dropped at :52. The result is 739 intra-recording splices plus 24 recording
boundaries -- 763 contiguous segments in all, mapped in train-array coordinates by
scripts/build_seam_map.py into data/seam_map.json. `apply_sliding_window` groups only
by subject id (sliding_window.py:102-103), so its windows straddle these seams freely;
at 1 s that is a small effect, at 10 s or 30 s it is not. Every sequence this module
emits lies inside exactly one segment.

CUTTING RULE, per segment of length n:
  n <  min_segment_len   discarded (14 segments, 582 samples = 0.042% at the default 50)
  n <  seq_len           one sequence, left-aligned, right-padded to seq_len
                         (features `pad_value`, labels `ignore_index`)
  n >= seq_len           starts at 0, step, 2*step, ... while curr < n - seq_len
                         (sliding_window.py:35's strict '<'), then ONE final
                         right-aligned sequence starting at n - seq_len.
The right-aligned tail is the difference from `apply_sliding_window`, which drops it
(sliding_window.py:35 leaves up to `step` trailing samples uncovered). Here every
sample of every kept segment appears in at least one sequence, so the training set is
the whole timeline rather than a prefix of it. The tail can overlap the penultimate
sequence by more than `overlap`; it never duplicates it, because the strict '<' bound
excludes the start n - seq_len from the loop.

FEATURES are `train[start:end, 1:4]` -- acc_x/acc_y/acc_z only. Column 0 (the subject
code) is dropped, matching validation.py:206 `X_train, X_val = X_train[:, :, 1:],
X_val[:, :, 1:]`, so nb_channels stays 3. No normalization and no augmentation are
applied: raw g values, exactly as the windowed path feeds them.

OUTPUT LAYOUT. X is (N_seq, seq_len, 3) float32 and y is (N_seq, seq_len) int64.
float32/int64 are what train.py:455-456's `torch.from_numpy` needs to produce a float
input and a long target; `nn.CrossEntropyLoss` (train.py:210) consumes (B, C, T)
logits against (B, T) targets unchanged, and honours `ignore_index=-100` by default,
which is why padded timesteps carry -100 rather than a real class id.

ORDERING is deterministic and never shuffled: segments in seam-map order (which is
train-array row order), and within a segment ascending start position. Validation
predictions can therefore be stitched back onto the raw timeline by replaying the same
rule, with no index array in the npz.

SAMPLE COUNTS. `train_sample_count` / `val_sample_count` count each raw row ONCE --
they are the number of non-discarded rows on that side, not the number of non-padding
timesteps across sequences (overlap makes the latter larger). `train_sample_labels`
and `val_sample_labels` are those rows' labels in stream order, so
len(val_sample_labels) == val_sample_count holds by construction. The train array is
what class weights must be derived from for the dense path: train.py:415-418 counts
whatever label array it is handed, and the windowed y_train is not it.
"""

import json
import os

import numpy as np

# Column layout of the `train` array returned by preprocess_data.load_dataset:
# [subject_code, acc_x, acc_y, acc_z, label] (preprocess_data.py:97 slices iloc[:, 3:],
# :152 splits the label off, :112 concatenates them back).
SUBJECT_COL = 0
FEATURE_COLS = slice(1, 4)
LABEL_COL = 4
N_CHANNELS = 3

EXPECTED_SEAM_MAP_VERSION = 1


class SeamMapError(RuntimeError):
    """The seam map is missing, malformed, or does not describe this train array."""


def load_seam_map(seam_map_path, train_array=None):
    """
    Read data/seam_map.json and check it actually describes `train_array`.

    Validated here rather than trusted, because every slice this module takes is in the
    seam map's coordinates: a stale or mismatched map would silently hand the model rows
    from the wrong segment.

    :param seam_map_path: path to the JSON written by scripts/build_seam_map.py
    :param train_array: optional (N, 5) array to check the map against
    :return: list of segment dicts, in file order
    """
    if not os.path.isfile(seam_map_path):
        raise SeamMapError(
            f"seam map not found at {seam_map_path!r}. Build it first: "
            "python scripts/build_seam_map.py"
        )
    with open(seam_map_path) as fid:
        payload = json.load(fid)

    for key in ("version", "total_samples", "total_segments", "segments"):
        if key not in payload:
            raise SeamMapError(f"{seam_map_path}: missing key {key!r}; got {sorted(payload)}")

    if payload["version"] != EXPECTED_SEAM_MAP_VERSION:
        raise SeamMapError(
            f"{seam_map_path}: schema version {payload['version']}, this module expects "
            f"{EXPECTED_SEAM_MAP_VERSION}. Refusing to guess at the layout."
        )

    segments = payload["segments"]
    if len(segments) != payload["total_segments"]:
        raise SeamMapError(
            f"{seam_map_path}: header says {payload['total_segments']} segments, "
            f"the list holds {len(segments)}"
        )

    # Tiling: the segments must cover [0, total_samples) with no gap and no overlap,
    # in ascending row order. Anything else and 'segment order == row order' is false.
    cursor = 0
    for i, seg in enumerate(segments):
        for key in ("subject_code", "subject_name", "recording", "start_row", "end_row", "length"):
            if key not in seg:
                raise SeamMapError(f"{seam_map_path}: segment {i} missing key {key!r}")
        if seg["start_row"] != cursor:
            raise SeamMapError(
                f"{seam_map_path}: segment {i} starts at row {seg['start_row']}, expected "
                f"{cursor} (segments must tile the array with no gap or overlap)"
            )
        if seg["end_row"] - seg["start_row"] != seg["length"] or seg["length"] <= 0:
            raise SeamMapError(
                f"{seam_map_path}: segment {i} has start={seg['start_row']} end={seg['end_row']} "
                f"length={seg['length']}, which is inconsistent or empty"
            )
        cursor = seg["end_row"]

    if cursor != payload["total_samples"]:
        raise SeamMapError(
            f"{seam_map_path}: segments end at row {cursor}, header says "
            f"total_samples={payload['total_samples']}"
        )

    if train_array is not None:
        if train_array.ndim != 2 or train_array.shape[1] != 5:
            raise SeamMapError(
                f"train_array has shape {train_array.shape}, expected (N, 5) "
                "[subject_code, acc_x, acc_y, acc_z, label]"
            )
        if len(train_array) != payload["total_samples"]:
            raise SeamMapError(
                f"{seam_map_path} describes {payload['total_samples']} samples but the train "
                f"array holds {len(train_array)}. The seam map does not describe this array."
            )
        # Subject alignment: the map's subject_code must agree with column 0 at both ends
        # of every segment. This is what catches a map built against a different CSV.
        codes = train_array[:, SUBJECT_COL]
        for i, seg in enumerate(segments):
            lo, hi = seg["start_row"], seg["end_row"]
            if int(codes[lo]) != seg["subject_code"] or int(codes[hi - 1]) != seg["subject_code"]:
                raise SeamMapError(
                    f"{seam_map_path}: segment {i} ({seg['recording']}) claims subject_code "
                    f"{seg['subject_code']} but train rows {lo}/{hi - 1} carry "
                    f"{int(codes[lo])}/{int(codes[hi - 1])}"
                )

    return segments


def segment_starts(n, seq_len, step):
    """
    Sequence start offsets, local to a segment of length n >= seq_len.

    The loop bound is sliding_window.py:35's `while curr < len(data) - win_len`, so the
    start exactly at n - seq_len is never produced here; the right-aligned tail appended
    afterwards is therefore always a new sequence, never a duplicate of the last one.
    """
    starts = []
    curr = 0
    while curr < n - seq_len:
        starts.append(curr)
        curr += step
    starts.append(n - seq_len)      # right-aligned tail: guarantees full coverage
    return starts


def plan_segment(seg, seq_len, step, min_segment_len):
    """
    Cutting plan for one segment: list of (start_row, end_row, pad_len) in train coords.

    An empty list means the segment is discarded (shorter than min_segment_len).
    pad_len > 0 marks the short-segment case: the sequence holds `length` real samples
    left-aligned, then pad_len padded timesteps.
    """
    lo, hi, n = seg["start_row"], seg["end_row"], seg["length"]

    if n < min_segment_len:
        return []
    if n < seq_len:
        return [(lo, hi, seq_len - n)]
    return [(lo + s, lo + s + seq_len, 0) for s in segment_starts(n, seq_len, step)]


def build_sequences(
    train_array,
    seam_map_path,
    val_subject_code,
    seq_len=500,
    overlap=0.5,
    min_segment_len=50,
    pad_value=0.0,
    ignore_index=-100,
):
    """
    Cut seam-free sequences for one LOSO fold.

    The fold split is the windowed path's, unchanged: validation.py:121-122 splits on
    column 0 with `data[data[:, 0] != sbj]` / `data[data[:, 0] == sbj]`, and one segment
    belongs entirely to one subject, so splitting whole segments on
    `segment["subject_code"] == val_subject_code` selects exactly the same rows.

    :param train_array: (N, 5) float array from load_dataset('subset_specific', 'loso_G')
    :param seam_map_path: path to data/seam_map.json
    :param val_subject_code: LabelEncoder subject code held out for validation (0-13)
    :param seq_len: sequence length in samples (default 500 = 10 s at 50 Hz)
    :param overlap: overlap fraction between consecutive sequences of a long segment
    :param min_segment_len: segments shorter than this are discarded entirely
    :param pad_value: feature value written to padded timesteps
    :param ignore_index: label written to padded timesteps (CrossEntropyLoss default)
    :return: dict -- see the module docstring for the sample-count semantics
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1 (got {seq_len!r})")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1) (got {overlap!r})")
    if min_segment_len < 1:
        raise ValueError(f"min_segment_len must be >= 1 (got {min_segment_len!r})")

    step = int(seq_len * (1 - overlap))
    if step < 1:
        raise ValueError(
            f"overlap={overlap} with seq_len={seq_len} gives step={step}; step must be >= 1"
        )

    segments = load_seam_map(seam_map_path, train_array)

    known_codes = {seg["subject_code"] for seg in segments}
    if val_subject_code not in known_codes:
        raise ValueError(
            f"val_subject_code={val_subject_code} is not in the seam map "
            f"(known codes: {sorted(known_codes)})"
        )

    # ---- pass 1: plan, in seam-map order (== train row order) ---------------------- #
    plans = {"train": [], "val": []}
    label_spans = {"train": [], "val": []}     # kept rows, for the per-sample label streams
    sample_counts = {"train": 0, "val": 0}
    discarded_samples = 0

    for seg in segments:
        side = "val" if seg["subject_code"] == val_subject_code else "train"
        entries = plan_segment(seg, seq_len, step, min_segment_len)
        if not entries:
            discarded_samples += seg["length"]
            continue
        plans[side].extend(entries)
        label_spans[side].append((seg["start_row"], seg["end_row"]))
        sample_counts[side] += seg["length"]

    # ---- pass 2: materialize --------------------------------------------------------- #
    out = {}
    for side in ("train", "val"):
        entries = plans[side]
        X = np.full((len(entries), seq_len, N_CHANNELS), pad_value, dtype=np.float32)
        y = np.full((len(entries), seq_len), ignore_index, dtype=np.int64)

        for i, (lo, hi, pad_len) in enumerate(entries):
            real = seq_len - pad_len
            X[i, :real] = train_array[lo:hi, FEATURE_COLS]
            y[i, :real] = train_array[lo:hi, LABEL_COL]

        suffix = "train" if side == "train" else "val"
        out[f"X_{suffix}"] = X
        out[f"y_{suffix}"] = y

        spans = label_spans[side]
        if spans:
            stream = np.concatenate([train_array[lo:hi, LABEL_COL] for lo, hi in spans])
        else:
            stream = np.empty(0, dtype=train_array.dtype)
        out[f"{suffix}_sample_labels"] = stream.astype(np.int64)
        out[f"{suffix}_sample_count"] = int(sample_counts[side])

    out["discarded_samples"] = int(discarded_samples)
    return out
