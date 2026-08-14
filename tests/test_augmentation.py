#!/usr/bin/env python3
"""
Standalone (no pytest) correctness checks for data_processing/augmentation.py's
segment-level augmentation, run against real apply_sliding_window output so
overlap/stride behavior matches production exactly.

Covers the Phase 2 hard-fail conditions for the rebound/layup augmentation study:
  - anchor-driven target-class counts are exactly pre_count * (multiplier - 1)
  - cross-target spillover contamination is detected and reported (not hidden)
  - determinism: same seed -> byte-identical output and spillover table
  - multiplier=1 is an exact no-op
  - reconstruct_stream HARD FAILS (raises RuntimeError) on a discontinuous
    segment, both directly and end-to-end through augment_training_windows,
    simulating what a --subsample_classes-induced gap looks like

Run: python tests/test_augmentation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from data_processing.sliding_window import apply_sliding_window
from data_processing.augmentation import (
    augment_training_windows, reconstruct_stream, _segment_index_range,
)

CLASS_NAMES = ['dribbling', 'shot', 'pass', 'rebound', 'layup',
               'walking', 'running', 'standing', 'sitting']
SW_OVERLAP = 60  # matches main.py's default --sw_overlap
WIN_LEN = 50


def _make_subject_raw(rng, subject_id, n_samples):
    accel = rng.normal(size=(n_samples, 3)).astype(np.float32)
    subj_col = np.full((n_samples, 1), subject_id, dtype=np.float32)
    return np.concatenate([subj_col, accel], axis=1)


def _build_fixture():
    """Two-subject synthetic fold with an adjacent rebound/layup region on subject 0
    (to force cross-target spillover) and an isolated rebound on subject 1."""
    rng = np.random.default_rng(123)
    X0 = _make_subject_raw(rng, 0.0, 900)
    X1 = _make_subject_raw(rng, 1.0, 900)
    y0 = np.zeros(900, dtype=np.uint8)
    y1 = np.zeros(900, dtype=np.uint8)
    y0[280:330] = 3  # rebound
    y0[330:380] = 4  # layup, immediately adjacent -> segments will overlap at context_k=2
    y1[500:520] = 3  # isolated rebound

    raw_X = np.concatenate([X0, X1], axis=0)
    raw_y = np.concatenate([y0, y1], axis=0)
    X_train, y_train = apply_sliding_window(
        raw_X, raw_y, sliding_window_size=WIN_LEN, unit='units',
        sampling_rate=50, sliding_window_overlap=SW_OVERLAP,
    )
    return X_train, y_train.astype(np.uint8)


def test_anchor_counts_exact_and_spillover_detected():
    X_train, y_train = _build_fixture()
    pre_counts = np.bincount(y_train.astype(int), minlength=9)

    X_aug, y_aug, stats = augment_training_windows(
        X_train, y_train, target_classes=['rebound', 'layup'], multiplier=4,
        recipe='conservative', seed=1, class_names=CLASS_NAMES,
        context_k=2, sw_overlap=SW_OVERLAP,
    )
    post_counts = np.bincount(y_aug.astype(int), minlength=9)
    rebound_id, layup_id = CLASS_NAMES.index('rebound'), CLASS_NAMES.index('layup')

    assert stats['anchor_added']['rebound'] == int(pre_counts[rebound_id]) * 3, \
        'anchor-driven rebound count must be exactly pre_count * (multiplier-1)'
    assert stats['anchor_added']['layup'] == int(pre_counts[layup_id]) * 3, \
        'anchor-driven layup count must be exactly pre_count * (multiplier-1)'

    total_spillover = sum(stats['spillover_added'].values())
    assert total_spillover > 0, 'expected cross-class spillover given adjacent rebound/layup regions'

    rebound_total_added = int(post_counts[rebound_id]) - int(pre_counts[rebound_id])
    expected_anchor_only = int(pre_counts[rebound_id]) * 3
    # This is the crux finding from the Phase 0b/Branch-B review: "exactly multiplier x" only
    # holds for the anchor-driven count, not the final bincount, when target classes co-occur
    # within context_k of each other. Assert the contamination is real in this fixture (by
    # construction) so the test would fail loudly if the spillover-accounting logic regressed
    # to silently hiding it.
    assert rebound_total_added > expected_anchor_only, \
        'expected final rebound bincount to exceed the anchor-only multiplier due to layup-segment spillover'
    print(f'[OK] anchor counts exact; rebound total_added={rebound_total_added} > '
          f'anchor-only={expected_anchor_only} (spillover contamination correctly surfaced)')


def test_overlap_consistency_of_output_windows():
    X_train, y_train = _build_fixture()
    _, _, stats = augment_training_windows(
        X_train, y_train, target_classes=['rebound', 'layup'], multiplier=4,
        recipe='conservative', seed=1, class_names=CLASS_NAMES,
        context_k=2, sw_overlap=SW_OVERLAP,
    )
    chk = stats['example_overlap_check']
    assert chk is not None
    assert chk['exact_match'] is True
    assert chk['max_abs_diff'] == 0.0
    print(f'[OK] adjacent augmented windows share identical overlap samples: {chk}')


def test_determinism():
    X_train, y_train = _build_fixture()
    kwargs = dict(target_classes=['rebound', 'layup'], multiplier=4, recipe='conservative',
                  class_names=CLASS_NAMES, context_k=2, sw_overlap=SW_OVERLAP)

    X_a, y_a, stats_a = augment_training_windows(X_train, y_train, seed=1, **kwargs)
    X_b, y_b, stats_b = augment_training_windows(X_train, y_train, seed=1, **kwargs)
    assert np.array_equal(X_a, X_b) and np.array_equal(y_a, y_b), \
        'same seed must reproduce byte-identical augmented output'
    assert stats_a['spillover_added'] == stats_b['spillover_added'], \
        'spillover table must be deterministic across same-seed runs'

    X_c, _, _ = augment_training_windows(X_train, y_train, seed=2, **kwargs)
    assert not np.array_equal(X_a, X_c), 'different seed should not reproduce identical output'
    print('[OK] determinism: same seed reproducible, different seed differs')


def test_multiplier_one_is_exact_noop():
    X_train, y_train = _build_fixture()
    X_noop, y_noop, stats = augment_training_windows(
        X_train, y_train, target_classes=['rebound', 'layup'], multiplier=1,
        recipe='rotate', seed=1, class_names=CLASS_NAMES, context_k=2, sw_overlap=SW_OVERLAP,
    )
    assert np.array_equal(X_noop, X_train) and np.array_equal(y_noop, y_train)
    assert stats['n_segments'] == 0
    print('[OK] multiplier=1 is an exact no-op')


def test_discontinuity_hard_fails():
    """Corrupt a window adjacent to a rebound anchor so its overlap no longer matches its
    neighbor, and confirm reconstruct_stream refuses to synthesize a segment from it --
    both directly and end-to-end through augment_training_windows."""
    X_train, y_train = _build_fixture()
    rng = np.random.default_rng(999)
    overlap_elements = int((SW_OVERLAP / 100) * WIN_LEN)

    rebound_id = CLASS_NAMES.index('rebound')
    rebound_indices = np.nonzero(y_train.astype(int) == rebound_id)[0]
    anchor_idx = int(rebound_indices[len(rebound_indices) // 2])

    X_corrupt = X_train.copy()
    X_corrupt[anchor_idx - 1, :, 1:4] = rng.normal(size=(WIN_LEN, 3))

    raised = False
    try:
        lo, hi = _segment_index_range(X_corrupt, anchor_idx, 2)
        reconstruct_stream(X_corrupt, list(range(lo, hi + 1)), overlap_elements)
    except RuntimeError:
        raised = True
    assert raised, 'reconstruct_stream must raise RuntimeError on a corrupted/discontinuous segment'
    print('[OK] reconstruct_stream HARD FAILS on a directly-corrupted segment')

    # simulate a --subsample_classes-induced gap: delete a window from the array (subject id
    # unchanged, but now array-adjacent rows are no longer temporally adjacent)
    X_gap = np.delete(X_train, anchor_idx - 1, axis=0)
    y_gap = np.delete(y_train, anchor_idx - 1, axis=0)
    raised_e2e = False
    try:
        augment_training_windows(
            X_gap, y_gap, target_classes=['rebound', 'layup'], multiplier=2,
            recipe='conservative', seed=1, class_names=CLASS_NAMES,
            context_k=2, sw_overlap=SW_OVERLAP,
        )
    except RuntimeError:
        raised_e2e = True
    assert raised_e2e, 'augment_training_windows must propagate the discontinuity error, not swallow it'
    print('[OK] augment_training_windows propagates the discontinuity error end-to-end '
          '(simulated subsampling gap)')


if __name__ == '__main__':
    tests = [
        test_anchor_counts_exact_and_spillover_detected,
        test_overlap_consistency_of_output_windows,
        test_determinism,
        test_multiplier_one_is_exact_noop,
        test_discontinuity_hard_fails,
    ]
    for t in tests:
        t()
    print('\nALL TESTS PASSED')
