#!/usr/bin/env python3
"""
Screening-protocol diagnostic (d): k-NN adjacency in the 128-d Stage 1 embedding, for any
set of windowed InceptionContext runs that saved their checkpoints.

A driver around knn_adjacency.py, whose main() cannot be pointed at these runs: it
discovers npz and checkpoints over window_purity.EXPECTED_FOLDS (the 5-fold placeholder),
accepts only the three July BASELINE_DIRS as --baseline_dir, and asserts a per-seed
target count in [25, 50] sized for POOLED_ONSET_COUNT = 109. Everything that does the
work is imported from it unchanged -- load_cfg, load_checkpoint, build_val_sets,
build_net, forward_fold (the Stage 1 pre-hook capture), neighbor_classes, frac_of_class,
mean_std, group_line -- and the window reconstruction is window_purity's, with the same
keying shim as diag_24fold_windowed.py. What differs here:

  * folds come from each run's cfg.txt, npz from sample_level_f1.discover_fold_npz, and
    checkpoints from a glob on the fold token with a unique-hit requirement
  * --min_targets / --max_targets and --pooled_onset_count are optional; when unset the
    check is skipped and a note says so
  * --anchor_fold runs ONE fold and requires its reproduced y_pred to equal the saved
    npz y_pred exactly, then exits -- the go/no-go before a full run
  * several runs (seeds) per invocation, the data loaded once, with the headline figures
    tabulated per seed and as mean +/- sample std (ddof=1) across seeds

HEADLINE FIGURES. At --report_k (default 10), neighbors classified by TRUE label, mean
over query windows of the fraction of neighbors that are:
    target running   -- pure rebound windows predicted running: running neighbors
    target rebound   -- the same windows: rebound neighbors
    control running  -- correctly predicted rebound windows: running neighbors
    control rebound  -- the same windows: rebound neighbors
    sanity running   -- correctly predicted running windows: running neighbors
The screening run's "50% / 40% / ~15% / 82%" are target running / control running /
control rebound / sanity running at k=10 (logs/knn_seed{1,2,3}.txt).

Selection uses the SAVED npz predictions, as in knn_adjacency.py; the forward pass
supplies the embeddings and the certification. CPU only.

Usage (from repo root):
    # go/no-go: one fold must reproduce exactly
    python analysis/diag_24fold_knn.py --anchor_fold 05d8_eu --results_dir <seed1_dir>

    python analysis/diag_24fold_knn.py --results_dir <seed1_dir> <seed2_dir> <seed3_dir>
"""

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import knn_adjacency as ka          # noqa: E402  -- also puts src/ on sys.path
import sample_level_f1 as slf       # noqa: E402
import window_purity as wp          # noqa: E402

REBOUND, RUNNING = ka.REBOUND, ka.RUNNING
fail = ka.fail


def discover_checkpoints(run_dir, folds):
    """fold -> the unique checkpoint_best_<fold>_*.pth in run_dir."""
    found = {}
    for fold in folds:
        hits = sorted(glob.glob(os.path.join(run_dir, f"checkpoint_best_{fold}_*.pth")))
        if len(hits) != 1:
            fail(f"{run_dir} fold {fold}: expected exactly 1 checkpoint_best, found {len(hits)}: "
                 f"{[os.path.basename(h) for h in hits]}")
        found[fold] = hits[0]
    return found


def check_cfg(cfg, run_dir):
    if cfg["network"] != "inceptioncontext":
        fail(f"{run_dir}: network {cfg['network']!r}; the Stage 1 hook is InceptionContext's")
    if cfg.get("dense"):
        fail(f"{run_dir}: a --dense run; this analysis needs per-window predictions")
    if cfg["augment_classes"]:
        fail(f"{run_dir}: augmentation run (augment_classes={cfg['augment_classes']})")
    if cfg["nb_units_gru_ic"] != ka.EMBED_DIM_EXPECTED:
        fail(f"{run_dir}: nb_units_gru_ic {cfg['nb_units_gru_ic']} != {ka.EMBED_DIM_EXPECTED}")
    if not cfg.get("loso_subjects"):
        fail(f"{run_dir}: cfg loso_subjects is empty; refusing to guess the folds")


def load_data(cfg):
    """The full loso array, loaded once; cwd must be the repo root (literal 'data/')."""
    from data_processing.preprocess_data import load_dataset

    train_arr, valid_arr, subjects, nb_classes, class_names, sampling_rate, _ = load_dataset(
        test_type=cfg["test_type"], test_case=cfg["test_case"], include_void=cfg["include_void"])
    if valid_arr is not None:
        fail(f"test_case {cfg['test_case']!r} returned a split validation set; this assumes LOSO")
    if list(class_names) != ka.CLASS_NAMES:
        fail(f"dataset class_names {list(class_names)} != {ka.CLASS_NAMES}")
    if sampling_rate != 50:
        fail(f"sampling_rate {sampling_rate} != 50")
    return train_arr, subjects, nb_classes


def reproduce_fold(cfg, fold, val_sets, nb_classes, ckpt_path, npz_path):
    """Forward one fold from its checkpoint; returns (y_pred_repro, embed, y_pred_npz, y_true_npz)."""
    X_val, y_val = val_sets[fold]
    net = ka.build_net(cfg, X_val.shape[1], X_val.shape[2], nb_classes)
    ckpt = ka.load_checkpoint(ckpt_path)
    if "model_state_dict" not in ckpt:
        fail(f"{ckpt_path}: no 'model_state_dict' key (found {sorted(ckpt)})")
    net.load_state_dict(ckpt["model_state_dict"], strict=True)

    y_pred_repro, embed = ka.forward_fold(net, X_val, y_val, cfg["batch_size"])
    with np.load(npz_path) as z:
        y_pred_npz, y_true_npz = z["y_pred"], z["y_true"]
    if len(y_pred_repro) != len(y_pred_npz):
        fail(f"fold {fold}: reproduced {len(y_pred_repro)} predictions, npz has {len(y_pred_npz)}")
    if not np.array_equal(y_val.astype(y_true_npz.dtype), y_true_npz):
        fail(f"fold {fold}: rebuilt y_val disagrees with the npz y_true; val windowing differs")
    if embed.shape != (len(y_pred_npz), ka.EMBED_DIM_EXPECTED):
        fail(f"fold {fold}: embedding shape {embed.shape}")
    return y_pred_repro, embed, y_pred_npz, y_true_npz


def run_seed(run_dir, cfg, npz_paths, ckpt_paths, val_sets, nb_classes, table, seed, args, ks):
    folds = sorted(npz_paths)
    pred_col = f"pred_seed{seed}"
    embeds, repro, n_bad = [], [], 0
    print(f"\n  [A] checkpoint certification, seed {seed} (agreement >= {args.agree_tol})")
    for fold in folds:
        y_rep, emb, y_npz, _yt = reproduce_fold(cfg, fold, val_sets, nb_classes,
                                               ckpt_paths[fold], npz_paths[fold])
        bad = int((y_rep != y_npz).sum())
        agree = 1 - bad / len(y_npz)
        n_bad += bad
        print(f"      {fold:<8} n={len(y_npz):>5}  agreement={agree:.6f}  disagreements={bad}")
        if agree < args.agree_tol:
            fail(f"seed {seed} fold {fold}: agreement {agree:.6f} < {args.agree_tol}")
        embeds.append(emb)
        repro.append(y_rep)

    for fold, emb in zip(folds, embeds):
        if int((table["fold"] == fold).sum()) != len(emb):
            fail(f"fold {fold}: window table and embeddings disagree in length")
    if list(table["fold"].drop_duplicates()) != folds:
        fail("window table fold order != embedding fold order")

    X = np.concatenate(embeds)
    y_rep = np.concatenate(repro)
    y_true = table["last_label"].to_numpy().astype(int)
    y_pred = table[pred_col].to_numpy().astype(int)
    if not np.isfinite(X).all():
        fail(f"seed {seed}: non-finite embeddings")
    print(f"      total disagreements: {n_bad} of {len(X)} windows")

    is_reb = y_true == REBOUND
    pure = table["purity"].to_numpy() == 1.0
    target = np.flatnonzero(is_reb & (y_pred == RUNNING) & pure)
    ctrl_reb = np.flatnonzero(is_reb & (y_pred == REBOUND))
    ctrl_run = np.flatnonzero((y_true == RUNNING) & (y_pred == RUNNING))
    print(f"\n  [C] targets: pure rebound windows predicted running = {len(target)}  "
          f"(of {int((is_reb & pure).sum())} pure / {int(is_reb.sum())} rebound windows); "
          f"controls: correct rebound {len(ctrl_reb)}, correct running {len(ctrl_run)}")
    print(f"      targets touched by a CPU/GPU disagreement: "
          f"{int(np.isin(target, np.flatnonzero(y_rep != y_pred)).sum())}")
    if args.min_targets is None and args.max_targets is None:
        print("      target-count bounds: NOT CHECKED (--min_targets/--max_targets unset)")
    else:
        lo = args.min_targets if args.min_targets is not None else 0
        hi = args.max_targets if args.max_targets is not None else np.inf
        if not lo <= len(target) <= hi:
            fail(f"seed {seed}: {len(target)} targets outside [{lo}, {hi}]")
        print(f"      target-count bounds [{lo}, {hi}]: OK")
    if args.pooled_onset_count is None:
        print("      pooled onset count: NOT COMPARED (--pooled_onset_count unset; the 109 is a 5-fold figure)")
    else:
        print(f"      pooled onset count reference: {args.pooled_onset_count} "
              f"(per-seed expectation {args.pooled_onset_count / 3:.1f}; printed, not asserted)")
    if len(target) == 0 or len(ctrl_reb) == 0 or len(ctrl_run) == 0:
        fail(f"seed {seed}: an empty group; the comparison cannot discriminate")

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="euclidean").fit(X)
    groups = (("Pure rebound -> running", target, ""),
              ("Correct rebound", ctrl_reb, "   [control]"),
              ("Correct running", ctrl_run, "   [sanity]"))
    print("\n  [D/E] k-NN, euclidean, full 128-d, self excluded")
    headline = None
    for k in ks:
        print(f"    k = {k}   by neighbor TRUE label")
        nbs = {}
        for tag, idx, note in groups:
            nbs[tag] = ka.neighbor_classes(nn, X, idx, k, y_true)
            print("  " + ka.group_line(f"{tag} (n={len(idx)})", nbs[tag], note))
        print(f"    k = {k}   by neighbor PREDICTED label")
        for tag, idx, note in groups:
            print("  " + ka.group_line(f"{tag} (n={len(idx)})",
                                       ka.neighbor_classes(nn, X, idx, k, y_pred), note))
        if k == args.report_k:
            m = lambda tag, c: 100 * ka.mean_std(ka.frac_of_class(nbs[tag], c))[0]   # noqa: E731
            headline = {
                "target running": m("Pure rebound -> running", RUNNING),
                "target rebound": m("Pure rebound -> running", REBOUND),
                "control running": m("Correct rebound", RUNNING),
                "control rebound": m("Correct rebound", REBOUND),
                "sanity running": m("Correct running", RUNNING),
                "n_target": len(target), "n_ctrl": len(ctrl_reb),
            }
    return headline


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", nargs="+", required=True, metavar="DIR",
                        help="Windowed InceptionContext run dirs holding cfg.txt, preds_*.npz and "
                             "checkpoint_best_*.pth, one per seed, all over the same folds.")
    parser.add_argument("--labels", default="labels_export.csv.gz")
    parser.add_argument("--npz_pattern", default=None)
    parser.add_argument("--anchor_fold", default=None,
                        help="Reproduce only this fold of the FIRST run, require exact equality "
                             "with its npz y_pred, and exit.")
    parser.add_argument("--k_values", default="5,10,20,50")
    parser.add_argument("--report_k", default=10, type=int,
                        help="k for the per-seed headline table (default 10, the screening k).")
    parser.add_argument("--agree_tol", default=0.999, type=float)
    parser.add_argument("--min_targets", default=None, type=int)
    parser.add_argument("--max_targets", default=None, type=int)
    parser.add_argument("--pooled_onset_count", default=None, type=int)
    parser.add_argument("--threads", default=None, type=int, help="torch CPU threads.")
    args = parser.parse_args()

    ks = [int(k) for k in args.k_values.split(",") if k.strip()]
    if args.report_k not in ks:
        fail(f"--report_k {args.report_k} is not in --k_values {ks}")
    if not os.path.isdir("data") or not os.path.isdir("analysis"):
        fail("run from the repo root: preprocess_data.py joins a literal 'data/'")

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    runs = {}
    for d in args.results_dir:
        cfg, _ = ka.load_cfg(d)
        check_cfg(cfg, d)
        seed = int(cfg["seed"])
        if seed in runs or not 1 <= seed <= len(wp.BASELINE_DIRS):
            fail(f"{d}: seed {seed} repeated or outside 1..{len(wp.BASELINE_DIRS)}")
        folds = list(cfg["loso_subjects"])
        runs[seed] = {"dir": d, "cfg": cfg,
                      "npz": slf.discover_fold_npz(d, folds, args.npz_pattern, expect_count=len(folds)),
                      "ckpt": discover_checkpoints(d, folds)}
    if len({tuple(sorted(r["npz"])) for r in runs.values()}) != 1:
        fail("runs disagree on their fold sets")
    keys = ("sw_length", "sw_overlap", "sw_unit", "batch_size", "test_type", "test_case", "include_void")
    if len({tuple(r["cfg"][k] for k in keys) for r in runs.values()}) != 1:
        fail(f"runs disagree on {keys}; they cannot share one val set")

    first = runs[min(runs)]
    print("=" * 100)
    print("(d) k-NN ADJACENCY IN 128-d STAGE 1 FEATURE SPACE -- CPU only")
    print("=" * 100)
    for s in sorted(runs):
        print(f"  seed {s}  {runs[s]['dir']}   ({len(runs[s]['ckpt'])} checkpoint_best files)")

    train_arr, subjects, nb_classes = load_data(first["cfg"])
    val_sets = ka.build_val_sets(first["cfg"], subjects, train_arr)

    if args.anchor_fold:
        fold = args.anchor_fold
        if fold not in first["npz"]:
            fail(f"--anchor_fold {fold} is not one of the run's folds")
        y_rep, _e, y_npz, _yt = reproduce_fold(first["cfg"], fold, val_sets, nb_classes,
                                               first["ckpt"][fold], first["npz"][fold])
        bad = int((y_rep != y_npz).sum())
        print(f"\n  ANCHOR  seed {first['cfg']['seed']}  fold {fold}  n={len(y_npz)}  "
              f"disagreements={bad}  -> {'EXACT' if bad == 0 else 'NOT EXACT'}")
        print(f"          ckpt {first['ckpt'][fold]}")
        print(f"          npz  {first['npz'][fold]}")
        sys.exit(0 if bad == 0 else 2)

    labels_df = wp.load_labels(args.labels)
    candidates = wp.build_candidate_windows(labels_df)
    resolved = wp.resolve_and_certify({wp.BASELINE_DIRS[0]: first["npz"]}, candidates)
    table = wp.attach_predictions(wp.build_window_table(resolved),
                                  {wp.BASELINE_DIRS[s - 1]: r["npz"] for s, r in runs.items()},
                                  resolved)
    print(f"\n  window reconstruction certified: {len(resolved)} folds, {len(table)} val windows")
    missing = [f for f in resolved if f not in val_sets]
    if missing:
        fail(f"no val windows rebuilt for fold(s) {missing}")

    heads = {}
    for s in sorted(runs):
        r = runs[s]
        print("\n" + "-" * 100)
        print(f"  SEED {s}")
        print("-" * 100)
        heads[s] = run_seed(r["dir"], r["cfg"], r["npz"], r["ckpt"], val_sets, nb_classes,
                            table, s, args, ks)

    cols = ("target running", "target rebound", "control running", "control rebound", "sanity running")
    print("\n" + "=" * 100)
    print(f"  HEADLINE -- k = {args.report_k}, neighbors by TRUE label, % of neighbors "
          "(mean over query windows)")
    print("=" * 100)
    print(f"  {'':<8}{'n_tgt':>7}{'n_ctrl':>8}" + "".join(f"{c:>17}" for c in cols))
    for s in sorted(heads):
        h = heads[s]
        print(f"  {'seed ' + str(s):<8}{h['n_target']:>7}{h['n_ctrl']:>8}"
              + "".join(f"{h[c]:>17.1f}" for c in cols))
    if len(heads) > 1:
        v = np.array([[heads[s][c] for c in cols] for s in sorted(heads)])
        print(f"  {'mean':<8}{'':>15}" + "".join(f"{x:>17.1f}" for x in v.mean(axis=0)))
        print(f"  {'std':<8}{'':>15}" + "".join(f"{x:>17.1f}" for x in v.std(axis=0, ddof=1)))
    print("\n  target = pure rebound windows predicted running; control = correctly predicted rebound;")
    print("  sanity = correctly predicted running. std is the sample std (ddof=1) across seeds.")


if __name__ == "__main__":
    try:
        main()
    except wp.HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
