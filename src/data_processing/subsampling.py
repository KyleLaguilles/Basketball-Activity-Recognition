##################################################
# Deterministic, subject-stratified training-window subsampling
# for the rebound/layup learning-curve experiment.
##################################################

import numpy as np


def subsample_training_windows(X_train, y_train, target_classes, fraction, seed, class_names):
    """
    Deterministically subsample training windows belonging to `target_classes` down to
    `fraction` of their count, stratified by subject so every subject keeps the same
    keep-ratio for a targeted class rather than some subjects being dropped entirely.

    Must be called BEFORE the subject-id column is stripped from X_train (i.e. before
    X_train = X_train[:, :, 1:] in cross_participant_cv) -- stratification reads the
    subject id out of X_train[:, 0, 0], which is constant across a window's time axis.

    :param X_train: numpy array, shape (n_windows, win_len, n_features_incl_subject_id)
    :param y_train: numpy array, shape (n_windows,)
    :param target_classes: list of str
        Class names (per class_names) whose windows should be subsampled.
    :param fraction: float in [0, 1]
        Fraction of each targeted class's windows to keep, per subject.
        keep count = max(1, round(n * fraction)) -- a subject with any windows
        of a targeted class always keeps at least one, never zeroed out entirely.
        NOTE: this means fraction=0.0 is NOT "remove the class" -- every subject
        that has >=1 window of a targeted class still keeps exactly 1. A true
        zero-out would need a separate code path; this function never produces one.
    :param seed: int
        RNG seed; the same (seed, fraction, target_classes) always yields the same kept windows.
    :param class_names: list of str
        Ordered class names (index == label id), as returned by load_dataset.
    :return: (X_train, y_train) subsampled; untouched entirely if fraction >= 1.0
    """
    if fraction >= 1.0 or not target_classes:
        return X_train, y_train

    name_to_id = {n: i for i, n in enumerate(class_names)}
    target_ids = sorted(name_to_id[c] for c in target_classes)

    subj_ids = X_train[:, 0, 0]
    y_int = y_train.astype(int)
    keep_mask = np.ones(len(y_train), dtype=bool)

    rng = np.random.default_rng(seed)

    for cls_id in target_ids:
        cls_mask = (y_int == cls_id)
        for subj in sorted(np.unique(subj_ids[cls_mask])):
            idx = np.nonzero(cls_mask & (subj_ids == subj))[0]
            n_keep = max(1, int(round(len(idx) * fraction)))
            n_drop = len(idx) - n_keep
            if n_drop <= 0:
                continue
            drop = rng.choice(idx, size=n_drop, replace=False)
            keep_mask[drop] = False

    return X_train[keep_mask], y_train[keep_mask]
