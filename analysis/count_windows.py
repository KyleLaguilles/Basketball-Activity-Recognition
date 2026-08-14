"""
Read-only per-class window count check for the learning-curve go/no-go.

Mode 'loso_g' (default): replicates the exact LOSO fold split and windowing
calls used by cross_participant_cv (model/validation.py:113-132) for
test_case=loso_G, and prints per-fold train/val np.bincount per class.

Mode 'drill': replicates the single global windowing call used by
train_valid_split (model/validation.py:396-401) for test_case=split_DvsG,
whose 'train' array is the drill-only data (preprocess_data.py:121-122).
There is no per-subject LOSO loop for this test_case, so this prints one
pooled total across all drill subjects, not fold-by-fold.

Windowing config (sw_length/sw_unit/sw_overlap) is fixed to the canonical
loso_G sqrt-inverse W&B run config, NOT the module defaults in main.py
(which default sw_overlap=60) and NOT the job_scripts/ precedent value.
include_void is left at the main.py module default (False); flag this
script's output if that's wrong for the 61/15 reference numbers.

No writes, no side effects, CPU only. Run from the repo root:
    python analysis/count_windows.py                 # loso_G, fold-by-fold
    python analysis/count_windows.py --mode drill     # drill-set totals
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, 'src')
from data_processing.preprocess_data import load_dataset
from data_processing.sliding_window import apply_sliding_window

SW_LENGTH = 1.0
SW_UNIT = 'seconds'
SW_OVERLAP = 50
INCLUDE_VOID = False


def print_counts(title, y, class_names, nb_classes):
    counts = np.bincount(y.astype(int), minlength=nb_classes)
    print(f'  {title}: {int(len(y))} windows')
    for c in range(nb_classes):
        print(f'    {class_names[c]:<10} {int(counts[c])}')


def run_loso_g():
    train, valid, subjects, nb_classes, class_names, sampling_rate, has_void = \
        load_dataset(test_type='subset_specific', test_case='loso_G', include_void=INCLUDE_VOID)
    assert valid is None, 'loso_G should have valid=None (LOSO, no fixed split)'

    print(f'loso_G: {len(subjects)} subjects, {nb_classes} classes, sampling_rate={sampling_rate}')
    print(f'windowing: sw_length={SW_LENGTH} {SW_UNIT}, sw_overlap={SW_OVERLAP}%, sampling_rate={sampling_rate}\n')

    for sbj in np.unique(train[:, 0]):
        train_data = train[train[:, 0] != sbj]
        val_data = train[train[:, 0] == sbj]

        X_train, y_train = apply_sliding_window(
            train_data[:, :-1], train_data[:, -1],
            sliding_window_size=SW_LENGTH, unit=SW_UNIT,
            sampling_rate=sampling_rate, sliding_window_overlap=SW_OVERLAP,
        )
        X_val, y_val = apply_sliding_window(
            val_data[:, :-1], val_data[:, -1],
            sliding_window_size=SW_LENGTH, unit=SW_UNIT,
            sampling_rate=sampling_rate, sliding_window_overlap=SW_OVERLAP,
        )

        print(f'FOLD held-out subject {subjects[int(sbj)]}:')
        print_counts('train', y_train, class_names, nb_classes)
        print_counts('val', y_val, class_names, nb_classes)
        print()


def run_drill():
    train, valid, subjects, nb_classes, class_names, sampling_rate, has_void = \
        load_dataset(test_type='session_specific', test_case='split_DvsG', include_void=INCLUDE_VOID)
    assert valid is not None, 'split_DvsG should have a fixed valid split (game)'

    print(f'split_DvsG (train=drill only): {nb_classes} classes, sampling_rate={sampling_rate}')
    print(f'windowing: sw_length={SW_LENGTH} {SW_UNIT}, sw_overlap={SW_OVERLAP}%, sampling_rate={sampling_rate}')
    print('NOTE: no per-subject LOSO loop for this test_case; totals are pooled across all drill subjects.\n')

    X_train, y_train = apply_sliding_window(
        train[:, :-1], train[:, -1],
        sliding_window_size=SW_LENGTH, unit=SW_UNIT,
        sampling_rate=sampling_rate, sliding_window_overlap=SW_OVERLAP,
    )

    print_counts('drill train (pooled)', y_train, class_names, nb_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['loso_g', 'drill'], default='loso_g')
    args = parser.parse_args()

    if args.mode == 'loso_g':
        run_loso_g()
    else:
        run_drill()
