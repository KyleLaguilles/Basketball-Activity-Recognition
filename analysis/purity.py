"""
Window purity analysis for Hang-Time HAR.

For each class: of the windows that CONTAIN it, what fraction are label-pure
(only that class) vs mixed? And of the windows the pipeline LABELS as a class,
how contaminated are they? Quantifies the windowing label-noise ceiling.

Reuses the pipeline's apply_sliding_window so windows match what the model sees
(it splits by subject internally). The raw CSV stores STRING labels and the
subject id may be a string hash, so we map both to ids here. Purity only depends
on whether two samples share a label, so the id ordering is irrelevant.

Label-assignment rule (from apply_sliding_window: output_y = [[i[-1]] ...]) is
LAST-sample, not majority vote; the assignment view below uses w[-1].

Run from the repo root:
    python analysis/window_purity.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from data_processing.sliding_window import apply_sliding_window

SW_LENGTH = 1.0
SW_UNIT = 'seconds'
SW_OVERLAP = 50
SAMPLING_RATE = 50          # confirmed: preprocess_data.py sets sampling_rate = 50
CLASS_NAMES = ['dribbling', 'shot', 'pass', 'rebound', 'layup',
               'walking', 'running', 'standing', 'sitting']

DATA_FILE = 'data/hangtime_drill_data.csv'
PURE_THRESHOLD = 0.80
OUTPUT_DIR = '.'


def load_windowed_labels(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    subj = pd.factorize(df.iloc[:, 0])[0].astype(float)          # subject grouping (hash-safe)
    raw = df.iloc[:, -1].astype(str).str.strip().str.lower().to_numpy()

    name2id = {n: i for i, n in enumerate(CLASS_NAMES)}
    unknown = sorted(set(raw) - set(name2id))                    # anything not a known class = void
    void_id = len(CLASS_NAMES)
    lab = np.array([name2id.get(s, void_id) for s in raw], dtype=float)
    names = CLASS_NAMES + (['void'] if unknown else [])
    ncls = len(names)

    print(f'{os.path.basename(path)}: {len(df)} samples, {len(np.unique(subj))} subjects')
    print(f'  labels found: {sorted(set(raw))}')
    if unknown:
        n_void = int((lab == void_id).sum())
        print(f'  NOTE: {unknown} not in the 9 classes -> bucketed as "void" '
              f'({n_void} samples, {100*n_void/len(lab):.1f}%). '
              f'Decide whether to keep or filter (see notes).')

    feats = np.column_stack([subj, lab])
    Xlab, _ = apply_sliding_window(feats, lab, SW_LENGTH, SW_UNIT, SAMPLING_RATE, SW_OVERLAP)
    return Xlab[:, :, 1].astype(int), ncls, names


def main():
    win, ncls, names = load_windowed_labels(DATA_FILE)
    n_win, win_len = win.shape

    contain = np.zeros(ncls, int)
    pure = np.zeros(ncls, int)
    eff_pure = np.zeros(ncls, int)
    contam = np.zeros((ncls, ncls))
    assigned_n = np.zeros(ncls, int)

    for w in win:
        counts = np.bincount(w, minlength=ncls)
        present = np.nonzero(counts)[0]
        frac = counts / win_len
        for c in present:
            contain[c] += 1
            if len(present) == 1:
                pure[c] += 1
            if frac[c] >= PURE_THRESHOLD:
                eff_pure[c] += 1
        a = w[-1]
        assigned_n[a] += 1
        contam[a] += frac

    pure_frac = np.divide(pure, contain, out=np.zeros(ncls), where=contain > 0)
    eff_frac = np.divide(eff_pure, contain, out=np.zeros(ncls), where=contain > 0)
    contam = np.divide(contam, assigned_n[:, None], out=np.zeros_like(contam),
                       where=assigned_n[:, None] > 0)
    self_purity = np.array([contam[c, c] for c in range(ncls)])

    containment = pd.DataFrame(dict(
        cls=names, n_windows_containing=contain,
        pct_pure=(100 * pure_frac).round(1),
        pct_mixed=(100 * (1 - pure_frac)).round(1),
        pct_pure_at_80=(100 * eff_frac).round(1)))

    top_contaminant, top_share = [], []
    for c in range(ncls):
        others = contam[c].copy()
        others[c] = -1
        j = int(others.argmax())
        top_contaminant.append(names[j] if others[j] > 0 else '-')
        top_share.append(round(100 * max(others[j], 0), 1))
    assignment = pd.DataFrame(dict(
        cls=names, n_windows_assigned=assigned_n,
        avg_self_purity_pct=(100 * self_purity).round(1),
        top_contaminant=top_contaminant, top_contaminant_pct=top_share))

    contam_df = pd.DataFrame((100 * contam).round(1), index=names, columns=names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(DATA_FILE))[0]
    containment.to_csv(os.path.join(OUTPUT_DIR, f'purity_containment_{tag}.csv'), index=False)
    assignment.to_csv(os.path.join(OUTPUT_DIR, f'purity_assignment_{tag}.csv'), index=False)
    contam_df.to_csv(os.path.join(OUTPUT_DIR, f'purity_contamination_{tag}.csv'))

    print(f'\n{n_win} windows, {win_len} samples each '
          f'({SW_LENGTH}{SW_UNIT[0]} @ {SAMPLING_RATE}Hz, {SW_OVERLAP}% overlap)\n')
    print('CONTAINMENT  -- of windows containing the class, how many are pure')
    print('%-10s %10s %8s %8s %10s' % ('class', 'n_contain', 'pure%', 'mixed%', 'pure@80%'))
    print('-' * 50)
    for i, c in enumerate(names):
        print('%-10s %10d %8.1f %8.1f %10.1f'
              % (c, contain[i], 100 * pure_frac[i], 100 * (1 - pure_frac[i]), 100 * eff_frac[i]))
    print('\nASSIGNMENT  -- of windows labeled the class (last-sample rule), how clean')
    print('%-10s %10s %12s %16s' % ('class', 'n_assign', 'self-purity%', 'top contaminant'))
    print('-' * 52)
    for i, c in enumerate(names):
        print('%-10s %10d %12.1f   %s (%.0f%%)'
              % (c, assigned_n[i], 100 * self_purity[i], top_contaminant[i], top_share[i]))
    print(f'\nsaved: purity_containment_{tag}.csv, purity_assignment_{tag}.csv, '
          f'purity_contamination_{tag}.csv')


if __name__ == '__main__':
    main()