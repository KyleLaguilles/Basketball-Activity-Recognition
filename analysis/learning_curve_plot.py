"""
Learning-curve plot for the rebound/layup synthetic-augmentation go/no-go.

Reads per-window val predictions saved by --save_val_npz (validation.py) as
preds_<subject>_<fraction>_seed<N>.npz (or, for the original seed-1 grid,
preds_<subject>_<fraction>.npz -- no suffix means seed=1 by construction,
since every one of those files came from a --seed 1 run) under a run's log
dir. POOLS predictions across the given fold subset per (fraction, seed) --
not mean-over-folds, since per-fold val support for rebound/layup is only
15-70 windows, too noisy to average -- and computes pooled per-class F1.
Where more than one seed exists for a fraction, reports mean +/- sd across
seeds; single-seed fractions get sd=0 (plotted with an n=1 annotation so
they're not mistaken for a multi-seed point).

Primary output: F1 vs. fraction, one line per --highlight_classes (default
rebound, layup) with error bars, all other classes collapsed into a single
muted "other classes" band for context.

Supplementary output: per-fold F1 vs. fraction for the highlighted classes
only (seed=1, every fraction has it), so pooling isn't silently hiding
fold-level heterogeneity.

--dirs lets you point at run directories that used the pre-seed-tagging
filename (e.g. concurrently-launched seed-2/3 endpoint jobs) without
rerunning them or renaming files on disk: pass <path>:<seed> to override
the filename-inferred seed for everything found under that directory.
Directories passed via --dirs are excluded from the default --log_root
recursive scan, so their unsuffixed filenames are never misattributed to
seed 1 there.

Read-only over logs/; writes only to --out_dir (default: figures/). Never
renames or moves any preds_*.npz file.

Run from the repo root, after Phase 3 runs have populated logs/ with
preds_*.npz files:
    python analysis/learning_curve_plot.py
    python analysis/learning_curve_plot.py --folds b512_na,a0da_eu,4d70_eu,ce9d_eu,9bd4_na \
        --fractions 0.25,0.5,0.75,1.0 --highlight_classes rebound,layup
    python analysis/learning_curve_plot.py \
        --dirs logs/subset_specific/loso_G/inceptioncontext/2026-07-17_12-00-00_lc_frac0.25_seed2:2 \
               logs/subset_specific/loso_G/inceptioncontext/2026-07-17_12-05-00_lc_frac0.25_seed3:3
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: this script only saves PNGs, never shows a window
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

CLASS_NAMES = ['dribbling', 'shot', 'pass', 'rebound', 'layup',
               'walking', 'running', 'standing', 'sitting']

DEFAULT_FOLDS = ['b512_na', 'a0da_eu', '4d70_eu', 'ce9d_eu', '9bd4_na']
DEFAULT_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_HIGHLIGHT = ['rebound', 'layup']

# palette (light mode) -- references/palette.md categorical slots 6 (red) and 1 (blue)
COLOR_HIGHLIGHT = {'rebound': '#e34948', 'layup': '#2a78d6'}
COLOR_OTHER = '#c3c2b7'
COLOR_INK = '#0b0b0b'
COLOR_MUTED = '#898781'
COLOR_GRID = '#e1e0d9'

SEED_SUFFIX_RE = re.compile(r'^seed(\d+)$')


def parse_pred_filename(path):
    """
    Parse preds_<subject>_<fraction>.npz (legacy; every such file was written by
    a --seed 1 run, so seed is implied = 1) or preds_<subject>_<fraction>_seed<N>.npz
    (current writer, validation.py). Parsing runs from the right: the fraction and
    optional seed<N> are the last one or two tokens and everything before them is the
    fold, so participant keys that contain an underscore (0846_eu) come through whole.

    Returns (fold, fraction, seed) or None if the filename matches neither form.
    """
    stem = os.path.basename(path)
    if not stem.endswith('.npz'):
        return None
    stem = stem[:-len('.npz')]
    if not stem.startswith('preds_'):
        return None
    rest = stem[len('preds_'):]
    parts = rest.split('_')
    if len(parts) < 2:
        return None

    seed_match = SEED_SUFFIX_RE.match(parts[-1])
    if seed_match:
        if len(parts) < 3:
            return None
        seed = int(seed_match.group(1))
        fraction_str = parts[-2]
        fold = '_'.join(parts[:-2])
    else:
        seed = 1
        fraction_str = parts[-1]
        fold = '_'.join(parts[:-1])

    if not fold:
        return None
    try:
        fraction = float(fraction_str)
    except ValueError:
        return None

    return fold, fraction, seed


def discover_npz(log_root, target_folds, target_fractions, exclude_dirs=()):
    """
    Recursively find preds_*.npz under log_root, skipping anything inside
    exclude_dirs (used to keep --dirs-covered directories out of this default
    scan, since their unsuffixed filenames would otherwise be misattributed to
    seed 1 here). Returns {(fold, fraction, seed): path}, keeping the most
    recently modified file when a triple appears under more than one run dir.
    """
    target_folds = set(target_folds)
    target_fraction_set = {float(f) for f in target_fractions}
    exclude_abs = [os.path.abspath(d) for d in exclude_dirs]

    best = {}
    for path in glob.glob(os.path.join(log_root, '**', 'preds_*.npz'), recursive=True):
        abs_dir = os.path.abspath(os.path.dirname(path))
        if any(abs_dir == d or abs_dir.startswith(d + os.sep) for d in exclude_abs):
            continue
        parsed = parse_pred_filename(path)
        if parsed is None:
            continue
        fold, fraction, seed = parsed
        if fold not in target_folds or fraction not in target_fraction_set:
            continue
        key = (fold, fraction, seed)
        if key not in best or os.path.getmtime(path) > os.path.getmtime(best[key]):
            best[key] = path
    return best


def parse_dirs_arg(dirs_arg):
    """
    Parse --dirs entries of the form <path> or <path>:<seed>. Returns a list of
    (path, seed_override_or_None).
    """
    parsed = []
    for entry in dirs_arg:
        path, sep, seed_str = entry.rpartition(':')
        if sep and seed_str.isdigit():
            parsed.append((path, int(seed_str)))
        else:
            parsed.append((entry, None))
    return parsed


def discover_npz_from_dirs(dirs_with_overrides, target_folds, target_fractions):
    """
    Like discover_npz, but scoped to explicit directories, with each directory's
    optional seed override taking precedence over filename inference for every
    preds_*.npz found under it (even legacy unsuffixed names).
    """
    target_folds = set(target_folds)
    target_fraction_set = {float(f) for f in target_fractions}

    best = {}
    for dir_path, seed_override in dirs_with_overrides:
        for path in glob.glob(os.path.join(dir_path, '**', 'preds_*.npz'), recursive=True):
            parsed = parse_pred_filename(path)
            if parsed is None:
                continue
            fold, fraction, inferred_seed = parsed
            seed = seed_override if seed_override is not None else inferred_seed
            if fold not in target_folds or fraction not in target_fraction_set:
                continue
            key = (fold, fraction, seed)
            if key not in best or os.path.getmtime(path) > os.path.getmtime(best[key]):
                best[key] = path
    return best


def pooled_f1_per_seed_fraction(files, target_fractions, target_folds):
    """
    Pools predictions across the fold subset per (fraction, seed). Returns a
    DataFrame with MultiIndex (fraction, seed), columns = CLASS_NAMES, values =
    pooled F1, plus {fraction: set(seeds found)}.
    """
    seeds_by_fraction = defaultdict(set)
    for (fold, fraction, seed) in files:
        seeds_by_fraction[fraction].add(seed)

    rows = {}
    for fraction in sorted(target_fractions):
        for seed in sorted(seeds_by_fraction.get(fraction, [])):
            y_pred_parts, y_true_parts = [], []
            found_folds = []
            for fold in target_folds:
                path = files.get((fold, fraction, seed))
                if path is None:
                    continue
                data = np.load(path)
                y_pred_parts.append(data['y_pred'])
                y_true_parts.append(data['y_true'])
                found_folds.append(fold)
            if not found_folds:
                continue
            missing = set(target_folds) - set(found_folds)
            if missing:
                print(f'  WARNING: fraction={fraction} seed={seed} missing folds {sorted(missing)}; '
                      f'pooling only {found_folds}.')
            y_pred = np.concatenate(y_pred_parts)
            y_true = np.concatenate(y_true_parts)
            f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
            rows[(fraction, seed)] = f1

    if not rows:
        df = pd.DataFrame(columns=CLASS_NAMES)
        df.index = pd.MultiIndex.from_tuples([], names=['fraction', 'seed'])
        return df, seeds_by_fraction

    df = pd.DataFrame.from_dict(rows, orient='index', columns=CLASS_NAMES)
    df.index = pd.MultiIndex.from_tuples(df.index, names=['fraction', 'seed'])
    df = df.sort_index()
    return df, seeds_by_fraction


def aggregate_across_seeds(df_per_seed):
    """
    df_per_seed: MultiIndex (fraction, seed) x CLASS_NAMES pooled F1.
    Returns a DataFrame indexed by fraction with <class>_mean, <class>_sd, n_seeds.
    sd uses ddof=0 so a single-seed fraction gets sd=0 (not NaN) -- it still plots
    as a point, just with a zero-length error bar and an n=1 annotation.
    """
    if df_per_seed.empty:
        cols = [f'{c}_mean' for c in CLASS_NAMES] + [f'{c}_sd' for c in CLASS_NAMES] + ['n_seeds']
        return pd.DataFrame(columns=cols)

    grouped = df_per_seed.groupby(level='fraction')
    mean_df = grouped.mean()
    sd_df = grouped.std(ddof=0)
    n_seeds = grouped.size()

    out = pd.DataFrame(index=mean_df.index)
    for c in CLASS_NAMES:
        out[f'{c}_mean'] = mean_df[c]
        out[f'{c}_sd'] = sd_df[c]
    out['n_seeds'] = n_seeds
    return out


def per_fold_f1_per_fraction(files, target_fractions, target_folds, classes_of_interest, seed=1):
    """
    Returns {class_name: DataFrame indexed by fraction, columns = fold, values = F1}
    for supplementary per-fold curves (highlighted classes only, fixed seed --
    this view is for eyeballing fold heterogeneity, not seed variance).
    """
    class_ids = {c: CLASS_NAMES.index(c) for c in classes_of_interest}
    out = {c: defaultdict(dict) for c in classes_of_interest}

    for fold in target_folds:
        for fraction in target_fractions:
            path = files.get((fold, fraction, seed))
            if path is None:
                continue
            data = np.load(path)
            y_pred, y_true = data['y_pred'], data['y_true']
            for c, cid in class_ids.items():
                f1 = f1_score(y_true, y_pred, average=None, labels=[cid], zero_division=0)[0]
                out[c][fraction][fold] = f1

    return {c: pd.DataFrame(v).sort_index() for c, v in out.items()}


def plot_primary(agg_df, highlight_classes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    if agg_df.empty:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    fractions = agg_df.index.to_numpy()
    n_seeds = agg_df['n_seeds']

    other_classes = [c for c in CLASS_NAMES if c not in highlight_classes]
    for i, c in enumerate(other_classes):
        ax.errorbar(fractions, agg_df[f'{c}_mean'], yerr=agg_df[f'{c}_sd'],
                    color=COLOR_OTHER, linewidth=1.2, alpha=0.55,
                    marker='o', markersize=4, capsize=3,
                    label='other classes' if i == 0 else None, zorder=2)

    # horizontal offset per highlighted class so n= labels don't collide when
    # two highlighted classes have close y-values at the same fraction
    label_dx = [-14, 14, 0, 0]
    for ci, c in enumerate(highlight_classes):
        color = COLOR_HIGHLIGHT.get(c, '#4a3aa7')
        ax.errorbar(fractions, agg_df[f'{c}_mean'], yerr=agg_df[f'{c}_sd'],
                    color=color, linewidth=2.2, marker='o', markersize=8,
                    markeredgecolor='#fcfcfb', markeredgewidth=1, capsize=4,
                    label=c, zorder=3)
        dx = label_dx[ci % len(label_dx)]
        for x, y, n in zip(fractions, agg_df[f'{c}_mean'], n_seeds):
            ax.annotate(f'n={int(n)}', (x, y), textcoords='offset points',
                        xytext=(dx, 12), fontsize=7, color=COLOR_MUTED, ha='center')

    ax.set_xlabel('Subsample fraction', color=COLOR_INK)
    ax.set_ylabel('Pooled per-class F1 (mean ± sd across seeds)', color=COLOR_INK)
    ax.set_title('Learning curve: F1 vs. training-window fraction (pooled, 5-fold subset)',
                 color=COLOR_INK, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#c3c2b7')
    ax.tick_params(colors=COLOR_MUTED)
    ax.legend(frameon=False, labelcolor=COLOR_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_supplementary(per_fold, out_path):
    n = len(per_fold)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor='#fcfcfb')
    if n == 1:
        axes = [axes]

    fold_colors = ['#2a78d6', '#1baf7a', '#eda100', '#4a3aa7', '#e87ba4', '#eb6834', '#e34948']

    for ax, (cls, fold_df) in zip(axes, per_fold.items()):
        ax.set_facecolor('#fcfcfb')
        for i, fold in enumerate(fold_df.columns):
            ax.plot(fold_df.index, fold_df[fold], color=fold_colors[i % len(fold_colors)],
                    linewidth=1.6, marker='o', markersize=5, alpha=0.85, label=fold)
        ax.set_title(f'{cls} (per-fold, supplementary, seed=1)', color=COLOR_INK, fontsize=10)
        ax.set_xlabel('Subsample fraction', color=COLOR_INK)
        ax.set_ylabel('F1', color=COLOR_INK)
        ax.set_ylim(0, 1)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=COLOR_MUTED)
        ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', default='logs', type=str)
    parser.add_argument('--folds', default=','.join(DEFAULT_FOLDS), type=str)
    parser.add_argument('--fractions', default=','.join(str(f) for f in DEFAULT_FRACTIONS), type=str)
    parser.add_argument('--highlight_classes', default=','.join(DEFAULT_HIGHLIGHT), type=str)
    parser.add_argument('--out_dir', default='figures', type=str)
    parser.add_argument('--dirs', nargs='*', default=[], type=str,
                         help='Explicit run directories to include, each optionally suffixed '
                              'with :<seed> (e.g. /path/to/rundir:2) to override filename-inferred '
                              'seed. Directories passed here are excluded from the default '
                              'recursive --log_root scan, so their files are never double-counted '
                              'or misattributed to seed 1.')
    args = parser.parse_args()

    folds = [f.strip() for f in args.folds.split(',') if f.strip()]
    fractions = [float(f.strip()) for f in args.fractions.split(',') if f.strip()]
    highlight_classes = [c.strip() for c in args.highlight_classes.split(',') if c.strip()]
    dirs_with_overrides = parse_dirs_arg(args.dirs)
    explicit_dir_paths = [d for d, _ in dirs_with_overrides]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Discovering preds_*.npz under {args.log_root} for folds={folds}, fractions={fractions} ...')
    if explicit_dir_paths:
        print(f'  excluding from --log_root scan (covered by --dirs instead): {explicit_dir_paths}')
    files = discover_npz(args.log_root, folds, fractions, exclude_dirs=explicit_dir_paths)
    print(f'  found {len(files)} matching (fold, fraction, seed) files under --log_root')

    if dirs_with_overrides:
        print(f'Discovering preds_*.npz under explicit --dirs: {dirs_with_overrides}')
        explicit_files = discover_npz_from_dirs(dirs_with_overrides, folds, fractions)
        print(f'  found {len(explicit_files)} matching (fold, fraction, seed) files under --dirs')
        files.update(explicit_files)

    print(f'  total: {len(files)} matching (fold, fraction, seed) files')

    seeds_by_fraction = defaultdict(set)
    for (fold, fraction, seed) in files:
        seeds_by_fraction[fraction].add(seed)
    print('\nDiscovered inventory (fraction -> seeds found):')
    for fraction in sorted(fractions):
        print(f'  {fraction}: seeds={sorted(seeds_by_fraction.get(fraction, []))}')

    df_per_seed, _ = pooled_f1_per_seed_fraction(files, fractions, folds)
    per_seed_csv = os.path.join(args.out_dir, 'learning_curve_pooled_f1_per_seed.csv')
    df_per_seed.to_csv(per_seed_csv)
    print(f'\nPooled per-class F1 by (fraction, seed):\n{df_per_seed}\n')
    print(f'saved: {per_seed_csv}')

    agg_df = aggregate_across_seeds(df_per_seed)
    agg_csv = os.path.join(args.out_dir, 'learning_curve_pooled_f1_agg.csv')
    agg_df.to_csv(agg_csv)
    print(f'\nAggregated (mean ± sd across seeds) by fraction:\n{agg_df}\n')
    print(f'saved: {agg_csv}')

    plot_primary(agg_df, highlight_classes, os.path.join(args.out_dir, 'learning_curve.png'))
    print(f'saved: {os.path.join(args.out_dir, "learning_curve.png")}')

    # supplementary per-fold view is fixed at seed=1 (every fraction has it); it's
    # for eyeballing fold heterogeneity, not seed variance
    per_fold = per_fold_f1_per_fraction(files, fractions, folds, highlight_classes, seed=1)
    plot_supplementary(per_fold, os.path.join(args.out_dir, 'learning_curve_per_fold.png'))
    print(f'saved: {os.path.join(args.out_dir, "learning_curve_per_fold.png")}')
