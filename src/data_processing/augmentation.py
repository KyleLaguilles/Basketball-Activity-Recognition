##################################################
# Segment-level augmentation for rebound/layup training windows.
#
# jitter / scaling / magnitude-warp / time-warp transform conventions follow
# Um et al. 2017, "Data Augmentation of Wearable Sensor Data for Parkinson's
# Disease Monitoring using Convolutional Neural Networks" (arXiv:1706.00527).
# Rotation is a standard addition for orientation-invariant IMU training and
# is not from that paper. numpy-only (no scipy): magnitude/time warp use a
# hand-rolled natural cubic spline instead of scipy.interpolate.
#
# ------------------------------------------------------------------------
# WHY THIS IS SEGMENT-LEVEL, NOT PER-WINDOW: batch = context.
# ------------------------------------------------------------------------
# For this repo's context-aware networks (model/InceptionContext.py,
# model/DeepConvContext.py), the DataLoader batch dimension is NOT a set of
# independent samples -- it is read directly as the sequence axis of a
# cross-window BiLSTM/attention module. Concretely, InceptionContext.forward()
# does `x = x.unsqueeze(1)` on a (B, F) per-window feature tensor to get
# (B, 1, F), then feeds it to `self.context_lstm`, an nn.LSTM constructed with
# `batch_first=False` -- so PyTorch reads that shape as (seq_len, batch,
# input), meaning seq_len = B (the minibatch of windows) and batch is pinned
# to 1 (model/InceptionContext.py forward(), the unsqueeze(1)+context_lstm
# call). DeepConvContext does the identical `x.unsqueeze(1)` before its own
# batch_first=False context module, and for its transformer/self-attention
# variants additionally applies PositionalEncoding(max_len=batch_size) and a
# causal attention mask (model/DeepConvContext.py) -- both operations are
# only meaningful if the batch axis is a real, ordered temporal sequence.
# train.py's valloader is hardcoded shuffle=False and trainloader defaults to
# shuffle=False (--shuffling off) for exactly this reason.
#
# Consequence: any operation that changes which windows sit next to each
# other in the training array (subsampling, and this module's augmentation)
# inserts an artificial "temporal" discontinuity into whatever batch happens
# to straddle it -- structurally the same thing as the real subject-boundary
# splices already present in every LOSO training fold, where
# apply_sliding_window concatenates per-subject window blocks back-to-back
# in np.unique(subject) order (data_processing/sliding_window.py). A LOSO
# fold with 13 remaining subjects already contains 12 such splices that the
# model trains through with no special handling.
#
# This module reconstructs and transforms a whole contiguous multi-window
# *segment* at once (see augment_training_windows below) specifically so
# each appended run of windows is internally coherent -- one smooth,
# still-temporally-valid stretch of signal -- rather than manufacturing a
# fresh discontinuity between every single augmented window and its
# neighbor, the way independent per-window augmentation would.
##################################################

import numpy as np

RECIPES = {
    'conservative': ('jitter', 'scaling', 'magnitude_warp'),
    'warp': ('jitter', 'scaling', 'magnitude_warp', 'time_warp'),
    'rotate': ('jitter', 'scaling', 'magnitude_warp', 'time_warp', 'rotate'),
}

DEFAULT_PARAMS = dict(
    jitter_sigma_frac=0.03,   # fraction of per-channel std (see channel_std below)
    scaling_sigma=0.1,        # N(1, sigma) constant multiplier per channel, whole segment
    magwarp_sigma=0.15,       # N(1, sigma) smooth per-timestep multiplier per channel
    magwarp_knots=4,          # knots per win_len-worth of stream (i.e. per second, at sw_length=1.0)
    timewarp_sigma=0.15,      # N(1, sigma) smooth time-axis warp
    timewarp_knots=4,         # knots per win_len-worth of stream
    rotate_max_deg=15.0,      # cap on the single random rotation angle per segment
)


def _natural_cubic_spline(x, y, x_new):
    """
    Minimal natural cubic spline interpolator (numpy-only stand-in for
    scipy.interpolate.CubicSpline). x must be strictly increasing.
    """
    n = len(x)
    h = np.diff(x)

    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, rhs)

    b = np.empty(n - 1)
    d = np.empty(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    idx = np.clip(np.searchsorted(x, x_new, side='right') - 1, 0, n - 2)
    dx = x_new - x[idx]
    return y[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


def _smooth_random_curve(length, n_knots, rng, loc, sigma, n_channels):
    """Cubic-spline through n_knots evenly spaced random control points ~ N(loc, sigma)."""
    knot_x = np.linspace(0, length - 1, n_knots)
    knot_y = rng.normal(loc=loc, scale=sigma, size=(n_knots, n_channels))
    sample_x = np.arange(length)
    curve = np.empty((length, n_channels))
    for c in range(n_channels):
        curve[:, c] = _natural_cubic_spline(knot_x, knot_y[:, c], sample_x)
    return curve


def jitter(accel, sigma_frac, channel_std, rng):
    """Add per-sample Gaussian noise; sigma = sigma_frac * channel_std (fold-level, per-channel).
    Genuinely per-sample even at segment scale -- every row draws independent noise."""
    noise = rng.normal(loc=0.0, scale=sigma_frac * channel_std, size=accel.shape)
    return accel + noise


def scaling(accel, sigma, rng):
    """Multiply each channel by one constant random scalar ~ N(1, sigma) for the whole segment."""
    factor = rng.normal(loc=1.0, scale=sigma, size=(1, accel.shape[1]))
    return accel * factor


def magnitude_warp(accel, sigma, n_knots, rng):
    """Multiply each channel, elementwise over time, by one smooth random curve spanning the whole segment."""
    curve = _smooth_random_curve(accel.shape[0], n_knots, rng, loc=1.0, sigma=sigma, n_channels=accel.shape[1])
    return accel * curve


def time_warp(accel, sigma, n_knots, rng):
    """Smoothly warp the time axis across the whole segment, then resample back onto the original grid."""
    seg_len = accel.shape[0]
    warp_curve = _smooth_random_curve(seg_len, n_knots, rng, loc=1.0, sigma=sigma, n_channels=1)[:, 0]
    warp_curve = np.maximum(warp_curve, 0.05)
    cum_warp = np.cumsum(warp_curve)
    cum_warp = (cum_warp - cum_warp[0]) / (cum_warp[-1] - cum_warp[0]) * (seg_len - 1)
    orig_t = np.arange(seg_len)
    warped = np.empty_like(accel)
    for c in range(accel.shape[1]):
        warped[:, c] = np.interp(orig_t, cum_warp, accel[:, c])
    return warped


def _axis_angle_rotation_matrix(axis, angle):
    """Rodrigues' rotation formula."""
    x, y, z = axis
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    comp = 1 - cos_a
    return np.array([
        [x * x * comp + cos_a,     x * y * comp - z * sin_a, x * z * comp + y * sin_a],
        [y * x * comp + z * sin_a, y * y * comp + cos_a,     y * z * comp - x * sin_a],
        [z * x * comp - y * sin_a, z * y * comp + x * sin_a, z * z * comp + cos_a],
    ])


def rotate(accel, max_angle_deg, rng):
    """Apply one random 3D rotation (random axis, angle in [-max_angle_deg, max_angle_deg]) to the whole segment."""
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    R = _axis_angle_rotation_matrix(axis, angle)
    return accel @ R.T


def apply_recipe(accel, recipe, channel_std, rng, win_len, params=None):
    """
    Apply one draw of `recipe`'s transforms to a whole segment's accel stream: one rotation,
    one scaling factor, one magnitude-warp curve, one time-warp curve for the entire stream
    (jitter is the exception -- it stays per-sample by nature). Warp knot counts scale with
    stream length relative to win_len (e.g. 4 knots per win_len-worth of samples), so a
    5-window segment gets a proportionally smoother/longer curve than a single window would,
    rather than reusing a fixed knot count sized for one window's duration.

    :param accel: numpy float array, shape (stream_len, 3) -- the accel triplet only.
    :param recipe: one of RECIPES
    :param channel_std: numpy float array, shape (3,) -- fold-level per-channel std.
    :param rng: numpy.random.Generator
    :param win_len: int -- single-window length, used only to scale knot density.
    :return: augmented accel, shape (stream_len, 3)
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    steps = RECIPES[recipe]
    scale = accel.shape[0] / win_len
    out = accel
    if 'jitter' in steps:
        out = jitter(out, p['jitter_sigma_frac'], channel_std, rng)
    if 'scaling' in steps:
        out = scaling(out, p['scaling_sigma'], rng)
    if 'magnitude_warp' in steps:
        n_knots = max(2, round(p['magwarp_knots'] * scale))
        out = magnitude_warp(out, p['magwarp_sigma'], n_knots, rng)
    if 'time_warp' in steps:
        n_knots = max(2, round(p['timewarp_knots'] * scale))
        out = time_warp(out, p['timewarp_sigma'], n_knots, rng)
    if 'rotate' in steps:
        out = rotate(out, p['rotate_max_deg'], rng)
    return out


def _segment_index_range(X_train, i, k):
    """
    Widest [lo, hi] array-index range around i (inclusive) such that every row in
    [lo, hi] has the same subject id as row i, truncated to at most k steps each
    direction and to the array bounds. Does NOT guarantee temporal contiguity within
    that range (subsampling can drop windows without changing subject grouping) --
    that is verified separately by reconstruct_stream, which raises if it isn't true.
    """
    n = len(X_train)
    subj = X_train[i, 0, 0]
    lo = i
    for step in range(1, k + 1):
        j = i - step
        if j < 0 or X_train[j, 0, 0] != subj:
            break
        lo = j
    hi = i
    for step in range(1, k + 1):
        j = i + step
        if j >= n or X_train[j, 0, 0] != subj:
            break
        hi = j
    return lo, hi


def reconstruct_stream(X_train, indices, overlap_elements):
    """
    Reconstruct a contiguous raw [subject_id, acc_x, acc_y, acc_z] stream from consecutive
    windows at array positions `indices` (ascending). Window j's first `overlap_elements`
    samples must equal window j-1's last `overlap_elements` samples -- this is exactly what
    apply_sliding_window's stride guarantees for genuinely-adjacent windows (stride = win_len -
    overlap_elements, computed identically to sliding_window.py's sliding_window_seconds /
    sliding_window_samples, both of which use overlap_elements = int((overlap_ratio/100)*win_len)).

    HARD FAILS (raises RuntimeError) on any mismatch rather than silently stitching
    non-contiguous data -- this is the guard against subsampling having removed an
    intermediate window, which leaves array-adjacent-but-temporally-disjoint rows that
    still pass the subject-id check in _segment_index_range.

    :param X_train: numpy array, shape (n_windows, win_len, 4)
    :param indices: list of int, ascending, all same subject
    :param overlap_elements: int, samples shared between consecutive windows
    :return: stream, shape (win_len + (len(indices)-1)*stride, 4)
    """
    win_len = X_train.shape[1]
    stride = win_len - overlap_elements
    subj = X_train[indices[0], 0, 0]
    stream_len = win_len + (len(indices) - 1) * stride
    stream = np.empty((stream_len, X_train.shape[2]), dtype=X_train.dtype)
    stream[:win_len] = X_train[indices[0]]
    cursor = win_len

    for pos in range(1, len(indices)):
        prev_idx, cur_idx = indices[pos - 1], indices[pos]
        prev_w, cur_w = X_train[prev_idx], X_train[cur_idx]
        if cur_w[0, 0] != subj:
            raise RuntimeError(
                f"Segment reconstruction discontinuity: array index {cur_idx} has subject id "
                f"{cur_w[0, 0]} but segment subject is {subj}."
            )
        if overlap_elements > 0:
            prev_tail = prev_w[stride:]
            cur_head = cur_w[:overlap_elements]
            if not np.array_equal(prev_tail, cur_head):
                raise RuntimeError(
                    f"Segment reconstruction discontinuity between array indices {prev_idx} and "
                    f"{cur_idx}: expected an identical {overlap_elements}-sample overlap per "
                    f"apply_sliding_window's stride ({stride}), but the overlap regions differ "
                    f"(max abs diff={np.max(np.abs(prev_tail.astype(float) - cur_head.astype(float))):.6g}). "
                    f"These windows are not temporally contiguous -- most likely an intermediate "
                    f"window was removed by --subsample_classes. Refusing to synthesize a segment "
                    f"from non-contiguous data."
                )
        stream[cursor:cursor + stride] = cur_w[overlap_elements:]
        cursor += stride

    return stream


def _rewindow_stream(stream, win_len, stride, n_windows):
    """Re-window a reconstructed (and possibly transformed) stream at the same win_len/stride
    used to originally build it, producing exactly n_windows windows by direct slicing."""
    windows = np.empty((n_windows, win_len, stream.shape[1]), dtype=stream.dtype)
    for j in range(n_windows):
        start = j * stride
        windows[j] = stream[start:start + win_len]
    return windows


def augment_training_windows(X_train, y_train, target_classes, multiplier, recipe, seed,
                              class_names, context_k, sw_overlap, params=None):
    """
    Segment-level augmentation (Branch B -- see module docstring for why). For each training
    window i whose label is in `target_classes`, builds a segment spanning array indices
    [i-context_k, i+context_k] (truncated at subject boundaries by _segment_index_range),
    reconstructs its contiguous raw stream (reconstruct_stream -- raises on discontinuity),
    applies one shared draw of `recipe`'s transforms to the whole stream (apply_recipe), and
    re-windows it at the identical win_len/stride (_rewindow_stream). Each output window in
    the segment inherits the label and subject id of its *positional* source window (i.e. the
    window at the same offset in the original segment) -- so a segment anchored on a rebound
    window also contributes copies of whatever its neighbors were (walking, standing, or even
    another target class), not just rebound. `multiplier - 1` independently-drawn segments are
    generated per source window, so each target-class source window contributes `multiplier - 1`
    segments (each contributing exactly one full re-windowed copy at the anchor position).

    Only the anchor-position window of each segment is guaranteed to carry the source window's
    own target-class label; non-anchor positions ("spillover") carry whatever their true label
    already was, INCLUDING the case where a neighboring window happens to itself be a different
    (or the same) target class -- e.g. an augmented segment anchored on a layup window can add
    extra rebound-labeled windows if a rebound window sits within context_k of that layup anchor
    in the same subject's stream. This means a target class's final post-augmentation count is
    only guaranteed to be exactly `multiplier` x its pre-augmentation count when that class has
    zero spillover contamination from other targeted classes' segments; the returned `stats`
    dict separates anchor-driven counts (always exactly multiplier x by construction) from
    spillover-driven counts so this can be checked, not just asserted.

    Must be called BEFORE the subject-id column is stripped from X_train (i.e. before
    X_train = X_train[:, :, 1:] in cross_participant_cv) and AFTER subsample_training_windows,
    so the two compose -- reconstruct_stream's discontinuity check exists specifically to fail
    loudly, rather than silently fabricate data, if subsampling has broken contiguity near an
    augmentation target.

    Deterministic under `seed`: a single np.random.default_rng(seed) stream is consumed in
    (target class id ascending, source window index ascending, copy index ascending) order,
    with apply_recipe's internal draw order (jitter, scaling, magnitude_warp, time_warp, rotate)
    fixed per call -- re-running with the same seed reproduces byte-identical augmented segments.

    :param X_train: numpy array, shape (n_windows, win_len, 4)
    :param y_train: numpy array, shape (n_windows,)
    :param target_classes: list of str (class names, per class_names)
    :param multiplier: int; <=1 is a no-op
    :param recipe: one of RECIPES
    :param seed: int
    :param class_names: list of str, ordered (index == label id)
    :param context_k: int; segment radius in windows (segment = [i-context_k, i+context_k])
    :param sw_overlap: int/float; the run's --sw_overlap percentage (same value passed to
        apply_sliding_window), used to compute overlap_elements identically to sliding_window.py
    :param params: optional dict overriding DEFAULT_PARAMS
    :return: (X_train, y_train, stats); untouched (X_train, y_train unchanged) with an all-zero
        stats dict if multiplier <= 1 or no target_classes. stats keys:
        'n_segments' (int), 'anchor_added' (dict class_name->int),
        'spillover_added' (dict class_name->int), 'example_overlap_check' (dict or None)
    """
    empty_stats = {
        'n_segments': 0,
        'anchor_added': {c: 0 for c in class_names},
        'spillover_added': {c: 0 for c in class_names},
        'example_overlap_check': None,
    }
    if multiplier <= 1 or not target_classes:
        return X_train, y_train, empty_stats

    name_to_id = {n: i for i, n in enumerate(class_names)}
    target_ids = sorted(name_to_id[c] for c in target_classes)

    win_len = X_train.shape[1]
    # identical formula to sliding_window.py's sliding_window_seconds/sliding_window_samples
    overlap_elements = int((sw_overlap / 100) * win_len)
    stride = win_len - overlap_elements

    y_int = y_train.astype(int)
    rng = np.random.default_rng(seed)

    accel_all = X_train[:, :, 1:4].reshape(-1, 3)
    channel_std = accel_all.std(axis=0)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)

    new_X, new_y = [], []
    anchor_added = {c: 0 for c in class_names}
    spillover_added = {c: 0 for c in class_names}
    example_overlap_check = None
    n_segments = 0

    for cls_id in target_ids:
        for i in np.nonzero(y_int == cls_id)[0]:
            i = int(i)
            lo, hi = _segment_index_range(X_train, i, context_k)
            indices = list(range(lo, hi + 1))
            anchor_offset = i - lo

            for _ in range(multiplier - 1):
                stream = reconstruct_stream(X_train, indices, overlap_elements)
                aug_stream = stream.copy()
                aug_stream[:, 1:4] = apply_recipe(stream[:, 1:4], recipe, channel_std, rng, win_len, params)
                aug_windows = _rewindow_stream(aug_stream, win_len, stride, len(indices))

                if example_overlap_check is None and overlap_elements > 0 and len(aug_windows) > 1:
                    tail = aug_windows[0, -overlap_elements:, 1:4]
                    head = aug_windows[1, :overlap_elements, 1:4]
                    example_overlap_check = {
                        'exact_match': bool(np.array_equal(tail, head)),
                        'max_abs_diff': float(np.max(np.abs(tail.astype(float) - head.astype(float)))),
                    }

                for pos, src_idx in enumerate(indices):
                    new_X.append(aug_windows[pos])
                    src_label = y_train[src_idx]
                    new_y.append(src_label)
                    cname = class_names[int(src_label)]
                    if pos == anchor_offset:
                        anchor_added[cname] += 1
                    else:
                        spillover_added[cname] += 1
                n_segments += 1

    if not new_X:
        return X_train, y_train, empty_stats

    X_out = np.concatenate([X_train, np.stack(new_X, axis=0)], axis=0)
    y_out = np.concatenate([y_train, np.array(new_y, dtype=y_train.dtype)], axis=0)
    stats = {
        'n_segments': n_segments,
        'anchor_added': anchor_added,
        'spillover_added': spillover_added,
        'example_overlap_check': example_overlap_check,
    }
    return X_out, y_out, stats
