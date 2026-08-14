#!/usr/bin/env python3
"""
k-NN adjacency in Stage 1 feature space: do the pure-interior rebound windows sit
inside the running cluster?

THE QUESTION. Section 6.3 / onset_test.py established that rebound-labeled windows
predicted running are not mislabeled run-in -- the errors land on the PURER windows,
and a subset of them are 100% rebound with no lead-in at all (onset_test.py's "no
lead-in" column, i.e. lead_in_class == -1 <=> trailing_run == 50 <=> purity == 1.0).
Those windows are the cleanest possible rebound examples and the model still calls
them running. This script asks the geometric version of that question: in the 128-d
per-window representation the model actually computes, are those windows sitting
among running windows?

WHAT IS MEASURED. The Stage 1 embedding -- the per-window vector produced by
inception branches -> GRU -> attention pooling, BEFORE the context BiLSTM
(InceptionContext.py:182). The quantitative claim is the k-NN neighbor class
distribution in that 128-d space. No t-SNE, no 2-d projection, no visualization:
a neighbor histogram in the full space is the evidence, and a projection would only
be an illustration of it.

WHERE THE EMBEDDING COMES FROM. Stage 1's output is an inline tensor expression, not
a module output, so it cannot be captured with a forward hook. It is captured with a
forward PRE-hook on net.context_lstm, whose input is exactly that vector unsqueezed
to (B, 1, 128) -- InceptionContext.forward lines 185-186 are `x.unsqueeze(1)` then
`self.context_lstm(x)` with nothing in between. The capture is exact, not approximate.

CERTIFICATION BEFORE USE. Checkpoints are not trusted on their filename. Step A
re-runs the full forward pass for every fold and compares the reproduced y_pred
against the baseline npz y_pred. The baseline was produced on cuda:0; this script
runs on CPU, and cuDNN vs CPU kernels differ in the last mantissa bits, so a 9-way
softmax argmax can flip on near-ties. Byte-identity is therefore NOT asserted --
per-fold agreement >= --agree_tol (default 0.999) is, every disagreement is printed
individually, and any fold below the threshold is a hard fail.

WHICH WINDOWS. onset_test.py pools seeds 1/2/3 over the same 214 rebound windows, so
its "no lead-in / pred running" count of 109 counts (window, seed) pairs, not 109
distinct windows. Only seed 1's checkpoints exist, so only seed 1 defines an
embedding space, and the target set here is the SEED-1 SLICE of that pooled 109 --
expected around a third of it. The count is asserted to lie in [--min_targets,
--max_targets] (default [25, 50]) and printed; it is not asserted to equal 109.

RECONSTRUCTION IS NOT REIMPLEMENTED. window_purity.py's certified pipeline is
imported and called: load_labels -> build_candidate_windows -> resolve_and_certify ->
build_window_table -> attach_predictions. resolve_and_certify brute-forces each fold
to its subject and accepts only on window count + byte-identical last-sample labels
(dtype included); attach_predictions re-asserts it. Those are the certification,
reused rather than duplicated. window_purity.discover_npz is NOT used, because it
hardcodes the three baseline timestamps and the `preds_<fold>_1.0.npz` name; this
script discovers npz itself (see discover_fold_npz) and hands the result in through
the same dict shape those functions expect.

CPU only. Read-only: touches nothing under src/, modifies no file, writes no file.

Usage (from repo root, hangtime_har_py310 env):
    python analysis/knn_adjacency.py \
        --ckpt_dir     logs/subset_specific/loso_G/inceptioncontext/<TIMESTAMP>_lc_frac1.0_ckpt \
        --baseline_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --labels_path  labels_export.csv.gz \
        --data_dir     data
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))  # data_processing / model live here
import window_purity as wp  # noqa: E402  -- module-level defs, __main__-guarded

CLASS_NAMES = wp.CLASS_NAMES
N_CLASSES = wp.N_CLASSES
REBOUND, RUNNING = 3, 6

EMBED_DIM_EXPECTED = 128
POOLED_ONSET_COUNT = 109  # onset_test.py's 3-seed pooled "no lead-in / pred running" count
MAX_DISAGREE_PRINT = 50   # per fold, in Step A

# resolve_and_certify indexes npz_paths by BASELINE_DIRS[0] and attach_predictions
# maps that same key through SEED_OF_DIR to name the prediction column. Handing our
# own discovered paths in under that key is the shim: it reuses both functions
# unmodified and yields the seed-1 column this analysis wants.
SHIM_KEY = wp.BASELINE_DIRS[0]
PRED_COL = f"pred_seed{wp.SEED_OF_DIR[SHIM_KEY]}"


def fail(msg):
    """Hard-fail: print the observed value and exit nonzero."""
    print(f"\nHARD FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# npz / checkpoint discovery
# --------------------------------------------------------------------------- #

def discover_fold_npz(run_dir, pattern=None):
    """
    Map fold -> npz path inside one run dir, tolerating both npz naming schemes.

    The baseline dirs predate validation.py's seed tag and are named
    `preds_<fold>_1.0.npz`; runs saved after that change are named
    `preds_<fold>_<fraction>_seed<N>.npz`. Augmentation runs carry a further
    `_aug<mult><recipe>` suffix and are EXCLUDED -- they are a different model, and
    sweeping one in would silently mix conditions.

    --npz_pattern overrides the auto-detection with an explicit glob containing
    `{fold}`. A fold resolving to zero or more than one file is a hard fail: the
    ambiguity has to be resolved by the caller, not guessed at here.
    """
    if pattern and "{fold}" not in pattern:
        fail(f"--npz_pattern must contain '{{fold}}', got {pattern!r}")

    found = {}
    for fold in wp.EXPECTED_FOLDS:
        if pattern:
            hits = sorted(glob.glob(os.path.join(run_dir, pattern.format(fold=fold))))
        else:
            hits = sorted(glob.glob(os.path.join(run_dir, f"preds_{fold}_*.npz")))
            hits = [h for h in hits if "_aug" not in os.path.basename(h)]
        if len(hits) != 1:
            fail(f"fold {fold}: expected exactly 1 npz in {run_dir}, found {len(hits)}: "
                 f"{[os.path.basename(h) for h in hits]}. Pass --npz_pattern to disambiguate.")
        found[fold] = hits[0]
    return found


def discover_checkpoints(ckpt_dir):
    """
    Map fold -> checkpoint path.

    validation.py:295-297 writes checkpoint_{best,last}_<subject>_<name>.pth, so the
    run's --name is embedded and is not knowable from the fold alone. Globbing on the
    fold token and requiring a unique hit avoids hardcoding the name while still
    refusing to choose between candidates.
    """
    found = {}
    for fold in wp.EXPECTED_FOLDS:
        hits = sorted(glob.glob(os.path.join(ckpt_dir, f"checkpoint_*_{fold}_*.pth")))
        if len(hits) != 1:
            fail(f"fold {fold}: expected exactly 1 checkpoint in {ckpt_dir}, found {len(hits)}: "
                 f"{[os.path.basename(h) for h in hits]}")
        found[fold] = hits[0]
    return found


def load_checkpoint(path):
    """
    torch.load a checkpoint onto CPU.

    train.py:598-605 stores random/numpy/torch RNG states alongside the weights, so
    the file is not a pure tensor archive. torch >= 2.6 defaults weights_only=True and
    would refuse it; older versions do not accept the argument at all. Both are
    handled rather than pinning a torch version.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cfg(ckpt_dir):
    path = os.path.join(ckpt_dir, "cfg.txt")
    if not os.path.isfile(path):
        fail(f"cfg.txt not found in the checkpoint run dir: {path}")
    with open(path) as fid:
        return json.load(fid), path


# --------------------------------------------------------------------------- #
# model / data
# --------------------------------------------------------------------------- #

def build_val_sets(cfg, subjects, data):
    """
    Rebuild each fold's val windows exactly as cross_participant_cv does.

    validation.py:133-138 windows val_data independently of train_data, then
    validation.py:206 strips the leading subject-id column. Neither subsampling nor
    augmentation ever touches X_val, so the val set is a function of the raw data and
    the windowing args alone -- it does not depend on the run's subsample/augment
    settings, and is reproducible from cfg.txt without replaying training.

    Returns {fold_name: (X_val, y_val)} for the folds named in cfg["loso_subjects"].
    """
    from data_processing.sliding_window import apply_sliding_window

    loso = set(cfg["loso_subjects"]) if cfg["loso_subjects"] else None
    out = {}
    for sbj in np.unique(data[:, 0]):
        name = str(subjects[int(sbj)])
        if loso is not None and name not in loso:
            continue
        val_data = data[data[:, 0] == sbj]
        X_val, y_val = apply_sliding_window(
            val_data[:, :-1], val_data[:, -1],
            sliding_window_size=cfg["sw_length"],
            unit=cfg["sw_unit"],
            sampling_rate=50,  # hardcoded at preprocess_data.py:31; never reaches cfg.txt
            sliding_window_overlap=cfg["sw_overlap"],
        )
        out[name] = (X_val[:, :, 1:], y_val)
    return out


def build_net(cfg, window_size, nb_channels, nb_classes):
    """Instantiate InceptionContext with exactly validation.py:259-270's argument set."""
    from model.InceptionContext import InceptionContext

    return InceptionContext(
        cfg["batch_size"], nb_channels, nb_classes, window_size,
        lstm_units=cfg["nb_units_lstm"],
        lstm_layers=cfg["nb_layers_lstm"],
        dropout=cfg["drop_prob"],
        bidirectional=cfg["bidirectional"],
        filter_sizes=cfg["filter_sizes"],
        branch_filters=cfg["branch_filters"],
        nb_units_gru_ic=cfg["nb_units_gru_ic"],
        use_channel_affine=cfg["use_channel_affine"],
        branch_dilations=cfg["branch_dilations"],
    )


def forward_fold(net, X_val, y_val, batch_size):
    """
    One fold's full val forward pass: reproduced y_pred plus Stage 1 embeddings.

    BATCHING IS LOAD-BEARING. context_lstm is batch_first=False on input (B, 1, 128),
    so the BATCH DIMENSION IS THE TEMPORAL SEQUENCE for Stage 2 -- and it is
    bidirectional, so a window's prediction depends on which windows share its batch,
    on both sides. Measured: regrouping 237 windows from one batch into 100/100/37
    moves the logits by ~3e-2, three orders above the ~1e-6 float32 noise floor. This
    is a real dependence, not rounding. batch_size and shuffle=False must therefore
    match train.py:418-425 or y_pred is not comparable to the baseline.

    Stage 1 is per-window and MATHEMATICALLY batch-independent, so the embeddings
    carry no such dependence. They are not bit-identical across batch sizes either
    (~1e-6 relative, from conv/GRU kernels blocking differently), but that is float32
    noise well below any distance this analysis measures. The script uses cfg's
    batch_size throughout regardless, so the question does not arise in practice.

    The embedding is captured by a forward PRE-hook on context_lstm: its input is
    Stage 1's output unsqueezed to (B, 1, 128), with no dropout or projection between
    InceptionContext.py:182 and :186. Hooking net.dropout would be wrong -- the same
    Dropout instance is reused at :179 and :192 and would fire twice per forward.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    captured = []

    def pre_hook(_module, args):
        z = args[0]                       # (B, 1, nb_units_gru_ic)
        captured.append(z.squeeze(1).detach().cpu().numpy())
        return None                       # leave the input untouched

    handle = net.context_lstm.register_forward_pre_hook(pre_hook)
    try:
        ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        net.eval()
        preds = []
        with torch.no_grad():
            for x, _y in loader:
                logits = net(x)
                # train.py:518 softmaxes before argmax; softmax is monotonic, so the
                # argmax is identical and the extra op is skipped here.
                preds.append(np.argmax(logits.numpy(), axis=-1).astype(int))
    finally:
        handle.remove()

    return np.concatenate(preds), np.concatenate(captured, axis=0)


# --------------------------------------------------------------------------- #
# k-NN
# --------------------------------------------------------------------------- #

def neighbor_classes(nn_model, embeddings, query_idx, k, labels):
    """
    Class labels of each query's k nearest neighbors, EXCLUDING the query itself.

    Every query point is a member of the fitted reference set, so it is its own
    nearest neighbor at distance 0. k+1 neighbors are requested and the self-match is
    dropped by global index rather than by position -- with duplicate or coincident
    embeddings the self-match is not guaranteed to come back first, and dropping
    position 0 blindly would silently discard a real neighbor and keep the query.

    Returns (n_queries, k) array of neighbor class labels.
    """
    _dist, idx = nn_model.kneighbors(embeddings[query_idx], n_neighbors=k + 1)

    out = np.empty((len(query_idx), k), dtype=int)
    for r, self_i in enumerate(query_idx):
        row = idx[r]
        keep = row[row != self_i]
        if len(keep) < k:
            fail(f"self-exclusion left only {len(keep)} of {k} neighbors for window {self_i}; "
                 "duplicate embeddings collapsed the neighborhood")
        out[r] = labels[keep[:k]]
    return out


def frac_of_class(nb_cls, cls_id):
    """Per-query fraction of neighbors belonging to cls_id."""
    return (nb_cls == cls_id).mean(axis=1)


def mean_std(v):
    """Mean and SAMPLE standard deviation (ddof=1) across queries."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1))


def group_line(tag, nb_cls, note=""):
    """One group's running- and rebound-neighbor fractions, the two classes at issue."""
    run_m, run_s = mean_std(frac_of_class(nb_cls, RUNNING))
    reb_m, reb_s = mean_std(frac_of_class(nb_cls, REBOUND))
    return (f"      {tag:<34} running: {run_m * 100:5.1f}% +/- {run_s * 100:4.1f}%"
            f"    rebound: {reb_m * 100:5.1f}% +/- {reb_s * 100:4.1f}%{note}")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", required=True,
                        help="Run dir holding the per-fold checkpoints and its own cfg.txt (the lc_frac1.0_ckpt re-run).")
    parser.add_argument("--baseline_dir", required=True,
                        help="Run dir holding the baseline per-fold preds_*.npz (2026-07-17_07-56-30).")
    parser.add_argument("--labels_path", default="labels_export.csv.gz",
                        help="Exported per-sample labels (columns: subject, label), original row order.")
    parser.add_argument("--data_dir", default="data",
                        help="Directory holding hangtime_{drill,warmup,game}_data.csv. Its basename must be 'data' -- "
                             "preprocess_data.py:34-36 joins against a literal 'data/', so the process chdirs to its parent.")
    parser.add_argument("--npz_pattern", default=None,
                        help="Explicit glob for the baseline npz, containing {fold} (e.g. 'preds_{fold}_1.0.npz'). "
                             "Omit to auto-detect both the bare and the seed-tagged naming.")
    parser.add_argument("--k_values", default="5,10,20,50", help="Comma-separated k values to sweep.")
    parser.add_argument("--agree_tol", default=0.999, type=float,
                        help="Minimum per-fold y_pred agreement with the baseline npz in Step A.")
    parser.add_argument("--min_targets", default=25, type=int)
    parser.add_argument("--max_targets", default=50, type=int)
    args = parser.parse_args()

    ks = [int(k) for k in args.k_values.split(",") if k.strip()]

    # Resolve every path to absolute BEFORE the chdir below, so relative arguments are
    # interpreted against the invocation cwd rather than silently re-rooted.
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    baseline_dir = os.path.abspath(args.baseline_dir)
    labels_path = os.path.abspath(args.labels_path)
    data_dir = os.path.abspath(args.data_dir)

    for p, what in ((ckpt_dir, "--ckpt_dir"), (baseline_dir, "--baseline_dir"), (data_dir, "--data_dir")):
        if not os.path.isdir(p):
            fail(f"{what} is not a directory: {p}")
    if not os.path.isfile(labels_path):
        fail(f"--labels_path is not a file: {labels_path}")
    if os.path.basename(data_dir) != "data":
        fail(f"--data_dir basename must be 'data' (preprocess_data.py joins a literal 'data/'), got: {data_dir}")

    cfg, cfg_path = load_cfg(ckpt_dir)
    npz_paths = discover_fold_npz(baseline_dir, args.npz_pattern)
    ckpt_paths = discover_checkpoints(ckpt_dir)

    print("=" * 100)
    print("k-NN ADJACENCY IN 128-d STAGE 1 FEATURE SPACE -- loso_G, 5 folds, seed 1, CPU only")
    print("=" * 100)
    print("\nFILES LOADED (absolute paths):")
    print(f"  cfg            {cfg_path}")
    print(f"  labels         {labels_path}")
    print(f"  data dir       {data_dir}")
    for fold in wp.EXPECTED_FOLDS:
        print(f"  ckpt  {fold}   {ckpt_paths[fold]}")
    for fold in wp.EXPECTED_FOLDS:
        print(f"  npz   {fold}   {npz_paths[fold]}")

    print(f"\n  network={cfg['network']}  seed={cfg['seed']}  batch_size={cfg['batch_size']}  "
          f"valid_epoch={cfg['valid_epoch']}  subsample_fraction={cfg['subsample_fraction']}")
    if cfg["network"] != "inceptioncontext":
        fail(f"cfg network is {cfg['network']!r}, not 'inceptioncontext'; the Stage 1 hook does not apply")
    if cfg["augment_classes"]:
        fail(f"the checkpoint run is an augmentation run (augment_classes={cfg['augment_classes']}); "
             "it is not comparable to the baseline npz")

    # ---- reconstruction, reused wholesale from window_purity.py -----------------
    labels_df = wp.load_labels(labels_path)
    candidates = wp.build_candidate_windows(labels_df)
    shimmed = {SHIM_KEY: npz_paths}                       # see SHIM_KEY comment
    resolved = wp.resolve_and_certify(shimmed, candidates)
    table = wp.attach_predictions(wp.build_window_table(resolved), shimmed, resolved)

    folds = sorted(resolved)                              # build_window_table's row order
    print(f"\n  reconstruction certified: {len(folds)} folds, {len(table)} pooled val windows")
    for fold in folds:
        subj, _starts, mat, _yt = resolved[fold]
        print(f"    fold {fold} -> subject {subj}   windows={mat.shape[0]}")

    # ---- data + val windows ----------------------------------------------------
    os.chdir(os.path.dirname(data_dir))
    print(f"\n  chdir to {os.getcwd()} so preprocess_data.py's literal 'data/' resolves")

    from data_processing.preprocess_data import load_dataset

    train_arr, valid_arr, subjects, nb_classes, class_names, sampling_rate, _has_void = load_dataset(
        test_type=cfg["test_type"], test_case=cfg["test_case"], include_void=cfg["include_void"])
    if valid_arr is not None:
        fail(f"test_case {cfg['test_case']!r} returned a split validation set; this analysis assumes LOSO")
    if list(class_names) != CLASS_NAMES:
        fail(f"class_names from the dataset {list(class_names)} != window_purity's {CLASS_NAMES}")
    if sampling_rate != 50:
        fail(f"sampling_rate is {sampling_rate}, but window_purity.py's WIN_LEN/STEP assume 50")

    val_sets = build_val_sets(cfg, subjects, train_arr)
    missing = [f for f in folds if f not in val_sets]
    if missing:
        fail(f"cfg loso_subjects did not yield val windows for fold(s) {missing}")

    embed_dim = cfg["nb_units_gru_ic"]
    if embed_dim != EMBED_DIM_EXPECTED:
        fail(f"cfg nb_units_gru_ic is {embed_dim}, but this analysis is stated in terms of "
             f"{EMBED_DIM_EXPECTED}-d Stage 1 features")

    # =========================================================================== #
    # STEP A -- certify the checkpoints against the baseline predictions
    # =========================================================================== #
    print("\n" + "=" * 100)
    print("[A] CHECKPOINT CERTIFICATION -- reproduced y_pred vs baseline npz y_pred")
    print("=" * 100)
    print(f"\n  The baseline ran on {cfg['gpu']}; this runs on CPU. cuDNN and CPU kernels differ in the")
    print("  last mantissa bits, and y_pred is an argmax over a 9-way softmax, so near-ties can flip.")
    print(f"  Byte-identity is therefore NOT asserted. Per-fold agreement must be >= {args.agree_tol:.4f};")
    print("  every disagreement is printed. A fold below the threshold is a hard fail.\n")

    import torch

    fold_preds, fold_embed, fold_disagree = {}, {}, {}
    for fold in folds:
        X_val, y_val = val_sets[fold]
        window_size, nb_channels = X_val.shape[1], X_val.shape[2]

        net = build_net(cfg, window_size, nb_channels, nb_classes)
        ckpt = load_checkpoint(ckpt_paths[fold])
        if "model_state_dict" not in ckpt:
            fail(f"{ckpt_paths[fold]}: no 'model_state_dict' key (found {sorted(ckpt)})")
        # strict=True already raises on any missing/unexpected key, which is the
        # desired hard failure: a shape or name mismatch means cfg.txt does not
        # describe the architecture this checkpoint was trained with.
        net.load_state_dict(ckpt["model_state_dict"], strict=True)

        y_pred_repro, embed = forward_fold(net, X_val, y_val, cfg["batch_size"])
        if embed.shape[1] != embed_dim:
            fail(f"fold {fold}: captured embedding dim {embed.shape[1]} != nb_units_gru_ic {embed_dim}")

        with np.load(npz_paths[fold]) as z:
            y_pred_base, y_true_base = z["y_pred"], z["y_true"]

        if len(y_pred_repro) != len(y_pred_base):
            fail(f"fold {fold}: reproduced {len(y_pred_repro)} predictions but the npz has "
                 f"{len(y_pred_base)}; the val set was not rebuilt identically")
        if not np.array_equal(y_val.astype(y_true_base.dtype), y_true_base):
            fail(f"fold {fold}: rebuilt y_val disagrees with the npz y_true; the val windowing "
                 "does not match the run that produced the baseline")

        bad = np.flatnonzero(y_pred_repro != y_pred_base)
        agree = 1.0 - len(bad) / len(y_pred_base)
        fold_disagree[fold] = bad
        status = "OK" if agree >= args.agree_tol else "FAIL"
        print(f"  fold {fold}   n={len(y_pred_base):>5}   agreement={agree:.6f}   "
              f"disagreements={len(bad):>4}   [{status}]")
        # Capped: a wrong checkpoint disagrees on thousands of windows, and dumping
        # all of them buries the verdict. At the tolerance the expected count is a
        # handful, so the cap only bites when the run has already failed.
        for i in bad[:MAX_DISAGREE_PRINT]:
            print(f"      idx {int(i):>5}   baseline={CLASS_NAMES[int(y_pred_base[i])]:<10} "
                  f"reproduced={CLASS_NAMES[int(y_pred_repro[i])]:<10} "
                  f"(true={CLASS_NAMES[int(y_true_base[i])]})")
        if len(bad) > MAX_DISAGREE_PRINT:
            print(f"      ... {len(bad) - MAX_DISAGREE_PRINT} further disagreement(s) not printed")
        if agree < args.agree_tol:
            fail(f"fold {fold} agreement {agree:.6f} < {args.agree_tol:.4f} "
                 f"({len(bad)} disagreements of {len(y_pred_base)}). The checkpoint does not reproduce "
                 "the baseline predictions; nothing downstream is valid.")

        fold_preds[fold] = y_pred_repro
        fold_embed[fold] = embed

    total_disagree = sum(len(v) for v in fold_disagree.values())
    print(f"\n  all {len(folds)} folds certified at >= {args.agree_tol:.4f} agreement "
          f"({total_disagree} disagreement(s) in {len(table)} windows overall)")

    # =========================================================================== #
    # STEP B -- assemble the embedding matrix
    # =========================================================================== #
    print("\n" + "=" * 100)
    print("[B] STAGE 1 EMBEDDINGS")
    print("=" * 100)

    all_embeddings = np.concatenate([fold_embed[f] for f in folds], axis=0)
    all_y_pred_repro = np.concatenate([fold_preds[f] for f in folds], axis=0)
    all_fold_ids = np.concatenate([np.full(len(fold_embed[f]), i, dtype=int) for i, f in enumerate(folds)])

    # Row alignment: build_window_table concatenates over sorted(resolved) with
    # ignore_index=True, and the loop above iterates the same sorted order, so table
    # row i and embedding row i are the same window. Asserted per fold rather than
    # assumed, since every downstream index depends on it.
    for i, fold in enumerate(folds):
        n_tbl = int((table["fold"] == fold).sum())
        if n_tbl != len(fold_embed[fold]):
            fail(f"fold {fold}: table has {n_tbl} rows but {len(fold_embed[fold])} embeddings")
    if not np.array_equal(all_fold_ids, np.array([folds.index(f) for f in table["fold"]])):
        fail("embedding row order does not match the window table's fold order")

    all_y_true = table["last_label"].to_numpy().astype(int)
    all_y_pred = table[PRED_COL].to_numpy().astype(int)   # baseline npz predictions

    npz_total = 0
    for f in folds:
        with np.load(npz_paths[f]) as z:
            npz_total += len(z["y_pred"])
    if len(all_embeddings) != npz_total:
        fail(f"embedding count {len(all_embeddings)} != summed npz length {npz_total}")
    if len(all_embeddings) != len(table):
        fail(f"embedding count {len(all_embeddings)} != window table length {len(table)}")
    if all_embeddings.shape[1] != EMBED_DIM_EXPECTED:
        fail(f"embedding dim {all_embeddings.shape[1]} != expected {EMBED_DIM_EXPECTED}")
    if not np.isfinite(all_embeddings).all():
        fail(f"{int((~np.isfinite(all_embeddings)).sum())} non-finite value(s) in the embeddings")

    print(f"\n  all_embeddings {all_embeddings.shape}   all_y_true {all_y_true.shape}   "
          f"all_y_pred {all_y_pred.shape}   all_fold_ids {all_fold_ids.shape}")
    print(f"  hook: forward pre-hook on net.context_lstm, input (B, 1, {EMBED_DIM_EXPECTED}) -> "
          f"(B, {EMBED_DIM_EXPECTED})")
    print(f"  window count matches summed npz length ({npz_total}) and the certified table ({len(table)})")
    print(f"  embedding norms: mean={np.linalg.norm(all_embeddings, axis=1).mean():.4f}  "
          f"min={np.linalg.norm(all_embeddings, axis=1).min():.4f}  "
          f"max={np.linalg.norm(all_embeddings, axis=1).max():.4f}")
    print("\n  all_y_pred below is the BASELINE npz prediction (the quantity onset_test.py analysed).")
    print(f"  The reproduced predictions differ on {int((all_y_pred_repro != all_y_pred).sum())} window(s), "
          "all within the Step A tolerance.")

    # =========================================================================== #
    # STEP C -- the pure-interior rebound windows
    # =========================================================================== #
    print("\n" + "=" * 100)
    print("[C] TARGET SET -- pure-interior rebound windows predicted running (seed 1)")
    print("=" * 100)

    purity = table["purity"].to_numpy()
    is_reb = all_y_true == REBOUND

    target_mask = is_reb & (all_y_pred == RUNNING) & (purity == 1.0)
    target_idx = np.flatnonzero(target_mask)
    n_targets = len(target_idx)

    # Independent cross-check via onset_test.py's encoding of the same predicate.
    # onset_test derives "no lead-in" (lead_in_class == -1 <=> trailing_run == 50) by
    # walking the per-sample label sequence recovered from each window's certified
    # [start, end) span. Recomputing it that way goes back to labels_df rather than
    # reusing build_window_table's histogram, so it can actually catch a
    # reconstruction or span error -- re-deriving purity from the hist_ columns could
    # not, since purity IS those columns.
    label_lookup = {
        s: labels_df.loc[labels_df["subject"] == s, "label_id"].to_numpy()
        for s in labels_df["subject"].unique()
    }
    reb_rows = np.flatnonzero(is_reb)
    all_rebound_samples = np.zeros(len(table), dtype=bool)
    for i in reb_rows:
        row = table.iloc[i]
        seq = label_lookup[row["subject"]][int(row["start"]):int(row["end"])]
        if len(seq) != wp.WIN_LEN:
            fail(f"window {i} (fold {row['fold']}) recovered {len(seq)} samples, expected {wp.WIN_LEN}")
        all_rebound_samples[i] = bool((seq == REBOUND).all())

    xcheck = np.flatnonzero(all_rebound_samples & (all_y_pred == RUNNING))
    if not np.array_equal(target_idx, xcheck):
        fail(f"purity==1.0 selects {len(target_idx)} windows but re-walking the per-sample label "
             f"sequences selects {len(xcheck)}; the two derivations must coincide")

    n_pure = int((is_reb & (purity == 1.0)).sum())
    print(f"\n  rebound-labeled val windows:                    {int(is_reb.sum())}")
    print(f"  of those, purity == 1.0 (pure-interior):        {n_pure}   <- the whole universe, all seeds")
    print(f"  of those, predicted running by seed 1:          {n_targets}   <- the target set")
    print(f"\n  cross-check by re-walking the per-sample label sequences (onset_test.py's")
    print(f"  lead_in_class == -1 encoding, derived from labels_df rather than the")
    print(f"  histogram columns): agrees ({len(xcheck)})")
    print(f"\n  This is the SEED-1 SLICE of onset_test.py's pooled count of {POOLED_ONSET_COUNT}.")
    print(f"  That {POOLED_ONSET_COUNT} pools seeds 1/2/3 over the same 214 rebound windows, so it counts")
    print("  (window, seed) pairs rather than distinct windows. Only seed 1 has checkpoints, so only")
    print(f"  seed 1 defines an embedding space; {n_targets} is the expected order of magnitude, not a shortfall.")

    if not (args.min_targets <= n_targets <= args.max_targets):
        fail(f"target count {n_targets} outside the expected range "
             f"[{args.min_targets}, {args.max_targets}] for a single seed's slice of {POOLED_ONSET_COUNT} "
             f"(the pure-interior universe is {n_pure} windows, and {POOLED_ONSET_COUNT}/3 = "
             f"{POOLED_ONSET_COUNT / 3:.1f} is the per-seed expectation)")

    hit_by_disagreement = int(np.isin(target_idx, np.flatnonzero(all_y_pred_repro != all_y_pred)).sum())
    print(f"  targets affected by a Step A CPU/GPU disagreement: {hit_by_disagreement}")

    ctrl_reb_idx = np.flatnonzero(is_reb & (all_y_pred == REBOUND))
    ctrl_run_idx = np.flatnonzero((all_y_true == RUNNING) & (all_y_pred == RUNNING))
    print(f"\n  controls: correct rebound n={len(ctrl_reb_idx)}   correct running n={len(ctrl_run_idx)}")
    if len(ctrl_reb_idx) == 0 or len(ctrl_run_idx) == 0:
        fail("a control group is empty; the comparison cannot discriminate")

    # =========================================================================== #
    # STEP D/E -- k-NN sweep with controls
    # =========================================================================== #
    print("\n" + "=" * 100)
    print("[D/E] k-NN NEIGHBOR DISTRIBUTIONS -- euclidean, full 128-d space, self excluded")
    print("=" * 100)
    print(f"\n  Reference set: all {len(all_embeddings)} val windows across all {len(folds)} folds.")
    print("  Neighbors are classified by their TRUE label (the primary table) and, separately, by the")
    print("  model's PREDICTED label -- 'sits inside the running cluster' is a claim about where real")
    print("  running windows are, but where the model THINKS running is matters too, and they differ.")

    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(metric="euclidean").fit(all_embeddings)

    groups = (
        ("Pure-interior rebound", target_idx, ""),
        ("Correct rebound", ctrl_reb_idx, "   [control]"),
        ("Correct running", ctrl_run_idx, "   [sanity]"),
    )

    per_class_by_k = {}
    for k in ks:
        print(f"\n  {'-' * 96}")
        print(f"  k = {k}")
        print(f"  {'-' * 96}")

        print("\n    by neighbor TRUE label")
        for tag, idx, note in groups:
            nb = neighbor_classes(nn_model, all_embeddings, idx, k, all_y_true)
            if tag == "Pure-interior rebound":
                per_class_by_k[k] = nb
            print(group_line(f"{tag} (n={len(idx)})", nb, note))

        print("\n    by neighbor PREDICTED label")
        for tag, idx, note in groups:
            nb = neighbor_classes(nn_model, all_embeddings, idx, k, all_y_pred)
            print(group_line(f"{tag} (n={len(idx)})", nb, note))

    # ---- full 9-class neighbor profile of the target windows -------------------
    print("\n" + "=" * 100)
    print(f"[F] FULL 9-CLASS NEIGHBOR PROFILE -- the {n_targets} pure-interior rebound windows only")
    print("=" * 100)
    print("\n  Mean +/- sd across the target windows of the fraction of each window's k neighbors")
    print("  falling in each class, by neighbor TRUE label. Each COLUMN sums to 100% up to rounding.\n")

    header = "  " + f"{'class':<12}" + "".join(f"{'k=' + str(k):>16}" for k in ks)
    print(header)
    print("  " + "-" * (12 + 16 * len(ks)))
    for c, name in enumerate(CLASS_NAMES):
        cells = []
        for k in ks:
            m, s = mean_std(frac_of_class(per_class_by_k[k], c))
            cells.append(f"{m * 100:7.1f}% +/-{s * 100:4.1f}")
        marker = "  <-- true label" if c == REBOUND else ("  <-- predicted" if c == RUNNING else "")
        print(f"  {name:<12}" + "".join(f"{cell:>16}" for cell in cells) + marker)

    # ---- caveats ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CAVEATS")
    print("=" * 100)
    print("""
  1. STAGE 1 IS CONTEXT-FREE. What is measured here is the per-window representation
     before the context BiLSTM. Stage 2 sees cross-window context and could in
     principle rescue some of these windows: a rebound window whose own 50 samples
     look like running may still be recoverable from its neighbors in time. "Not
     separable in Stage 1" means the per-window features are insufficient on their
     own -- it does NOT mean the full model has no path to getting them right.

  2. THE CLAIM IS THE NEIGHBOR DISTRIBUTION, NOT A PICTURE. The quantity above is the
     k-NN class distribution in the full 128-d space. No t-SNE or other projection is
     computed, and none should be substituted for this: a 2-d embedding can separate
     or merge clusters that the 128-d metric does not, so a plot could contradict
     this table without either being wrong.

  3. SEED-1 SLICE. The target set is seed 1's share of onset_test.py's pooled count of
     {pooled}, which counts (window, seed) pairs across seeds 1/2/3. Only seed 1 has
     checkpoints, and only one seed's weights define one embedding space. The other
     seeds' errors fall on overlapping but not identical windows.

  4. CONTROLS ARE THE READ. A high running-neighbor fraction for the target windows is
     only informative relative to the correct-rebound control. If the two groups do
     not separate, the method is not discriminating and the number says nothing.

  5. CPU/GPU DIVERGENCE. Predictions were reproduced on CPU from checkpoints trained on
     {gpu}. Step A's disagreements are near-tie softmax flips, not evidence of a wrong
     checkpoint. Selection above uses the BASELINE npz predictions throughout, so the
     target set is exactly the one onset_test.py analysed.
""".format(pooled=POOLED_ONSET_COUNT, gpu=cfg["gpu"]))


if __name__ == "__main__":
    try:
        main()
    except wp.HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
