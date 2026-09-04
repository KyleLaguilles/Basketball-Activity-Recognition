##################################################
# All functions related to validating a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score

from data_processing.sliding_window import apply_sliding_window
from data_processing.subsampling import subsample_training_windows
from data_processing.augmentation import augment_training_windows
from misc.osutils import mkdir_if_missing
from model.DeepConvContext import DeepConvContext
from model.DeepConvLSTM import DeepConvLSTM
from model.AttendAndDiscriminate import AttendAndDiscriminate
from model.ShallowDeepConvLSTM import ShallowDeepConvLSTM
from model.TinyHAR import TinyHAR_Model
from model.TinierHAR import TinierHAR_Model
from model.ICGNet import ICGNet
from model.InceptionContext import InceptionContext
from model.train import train, init_optimizer, init_loss, init_scheduler
import wandb

class TinyHARWrapper(nn.Module):
    """Wraps TinyHAR to handle input shape: (B, T, C) -> (B, 1, T, C)"""
    use_fixup = False  # required by init_weights in train.py

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, T, C) -> (B, 1, T, C)
        return self.model(x)

class TinierHARWrapper(nn.Module):
    """Wraps TinierHAR to handle input shape: (B, T, C) -> (B, 1, T, C)"""
    use_fixup = False  # required by init_weights in train.py

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, T, C) -> (B, 1, T, C)
        return self.model(x)

def save_composite_confusion_matrix(v_conf_mat, class_names, log_dir, run=None, title='Confusion Matrix (All Subjects)'):
    """
    Save a styled composite confusion matrix heatmap similar to the paper's Figure 12.
    Uses seaborn for better color styling.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        v_conf_mat,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()

    mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
    save_path = os.path.join(log_dir, 'conf_mats', 'all_composite.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if run is not None:
        wandb.log({'conf_matrices/composite': wandb.Image(save_path)})

    return save_path

def dense_val_plan(data, args, sbj):
    """
    Re-derive the validation cutting plan for one fold, in train-array row coordinates.

    Uses sequence_loader's own plan_segment/load_seam_map rather than restating the rule, so
    this cannot drift from the arrays build_sequences actually produced.

    :return: (entries, kept_rows) where entries is [(start_row, end_row, pad_len)] in emission
        order and kept_rows is every non-discarded validation row, ascending -- which is
        exactly the order of build_sequences' val_sample_labels.
    """
    from data_processing.sequence_loader import load_seam_map, plan_segment

    segments = load_seam_map(args.dense_seam_map, data)
    step = int(args.dense_seq_len * (1 - args.dense_overlap))

    entries, kept = [], []
    for seg in segments:
        if seg['subject_code'] != int(sbj):
            continue
        seg_entries = plan_segment(seg, args.dense_seq_len, step, args.dense_min_seg)
        if not seg_entries:
            continue
        entries.extend(seg_entries)
        kept.append(np.arange(seg['start_row'], seg['end_row']))
    return entries, (np.concatenate(kept) if kept else np.empty(0, dtype=int))


def stitch_dense_predictions(val_output, entries, kept_rows, n_rows):
    """
    Collapse per-timestep predictions onto one prediction per raw sample.

    Overlapping sequences predict the same sample more than once, so the flat array coming out
    of train() is longer than the timeline: at seq_len=500/overlap=0.5 a fold's 126839 samples
    arrive as 244488 timesteps. Writing that to the npz would mean every downstream script
    scored a population that double-counts segment interiors relative to their edges.

    LAST-SEQUENCE-WINS, the same convention sample_level_f1.py:296-311 applies to overlapping
    windows: sequences are assigned in emission order, so the last one covering a sample sets
    it. Emission order is preserved end to end -- the val DataLoader is built with
    shuffle=False (train.py:477-484) and the per-batch padding mask keeps order -- so walking
    the flat array with a cursor of (end_row - start_row) per sequence is exact.

    :return: (n_kept, 2) array of [y_pred, y_true] in raw stream order, the same layout the
        windowed path returns.
    """
    preds = val_output[:, 0].astype(int)
    truth = val_output[:, 1].astype(int)

    expected = sum(hi - lo for lo, hi, _pad in entries)
    if len(preds) != expected:
        raise ValueError(
            f"dense stitching: train() returned {len(preds)} timesteps but the re-derived "
            f"validation plan accounts for {expected}. The plan and the emitted sequences "
            "disagree; refusing to stitch."
        )

    stitched_pred = np.full(n_rows, -1, dtype=int)
    stitched_true = np.full(n_rows, -1, dtype=int)
    cursor = 0
    for lo, hi, _pad in entries:
        real = hi - lo
        stitched_pred[lo:hi] = preds[cursor:cursor + real]
        stitched_true[lo:hi] = truth[cursor:cursor + real]
        cursor += real

    out_pred = stitched_pred[kept_rows]
    out_true = stitched_true[kept_rows]
    if (out_pred < 0).any() or (out_true < 0).any():
        raise ValueError("dense stitching: some kept validation rows were never covered by a "
                         "sequence, which contradicts the loader's coverage guarantee.")
    return np.vstack((out_pred, out_true)).T


def cross_participant_cv(data, args, log_dir=None, run=None):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).

    :param data: numpy array
        Data used for applying cross-validation
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_dir: string
        Logging directory
    :return pytorch model
        Trained network
    """
    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    all_train_output = None
    all_val_output = None
    orig_lr = args.learning_rate

    # per-subject best epoch and F1 tracking
    subject_best_records = []

    fold_subjects = np.unique(data[:, 0])
    if args.loso_subjects:
        fold_subjects = np.array([sbj for sbj in fold_subjects if str(args.subjects[int(sbj)]) in args.loso_subjects])

    for i, sbj in enumerate(fold_subjects):
        print('\n VALIDATING FOR SUBJECT {0}; {1} OF {2}'.format(args.subjects[int(sbj)], int(sbj) + 1, int(len(np.unique(data[:, 0])))))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
        args.learning_rate = orig_lr

        if args.dense:
            # Dense path: seam-free sequences with one label per timestep. build_sequences
            # takes the FULL array and does its own fold split on subject_code, which selects
            # exactly the rows the train_data/val_data masks above select (one seam-map segment
            # belongs entirely to one subject). apply_sliding_window is bypassed entirely --
            # its last-sample rule (sliding_window.py:117) is the labeling artifact this path
            # exists to remove -- and the acc-only channel slice normally applied at the
            # X_train[:, :, 1:] line below is already done by the loader.
            from data_processing.sequence_loader import build_sequences

            dense_res = build_sequences(
                train_array=data,
                seam_map_path=args.dense_seam_map,
                val_subject_code=int(sbj),
                seq_len=args.dense_seq_len,
                overlap=args.dense_overlap,
                min_segment_len=args.dense_min_seg,
            )
            X_train, y_train = dense_res['X_train'], dense_res['y_train']
            X_val, y_val = dense_res['X_val'], dense_res['y_val']
            # Per-sample label streams. Class weights must be derived from these, not from the
            # padded (N_seq, T) targets -- see train.py's weight block.
            args.dense_train_sample_labels = dense_res['train_sample_labels']
            args.dense_val_sample_labels = dense_res['val_sample_labels']
            dense_entries, dense_kept_rows = dense_val_plan(data, args, sbj)
            print(f'  Dense sequences  train: {X_train.shape}, val: {X_val.shape} '
                  f'(seq_len={args.dense_seq_len}, overlap={args.dense_overlap}, '
                  f'min_segment_len={args.dense_min_seg})')
            print(f'  Dense samples    train: {dense_res["train_sample_count"]}, '
                  f'val: {dense_res["val_sample_count"]}, '
                  f'discarded: {dense_res["discarded_samples"]}')
        else:
            # Sensor data is segmented using a sliding window mechanism
            X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                                    sliding_window_size=args.sw_length,
                                                    unit=args.sw_unit,
                                                    sampling_rate=args.sampling_rate,
                                                    sliding_window_overlap=args.sw_overlap,
                                                    )

            X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

        # subsample training windows of the targeted classes BEFORE the subject-id column
        # (X_train[:, 0, 0]) is stripped below -- stratification needs it. Eval (X_val, y_val)
        # was windowed independently from val_data above and is never touched here.
        if args.subsample_classes and not args.dense:
            train_counts_pre = np.bincount(y_train.astype(int), minlength=args.nb_classes)
            X_train, y_train = subsample_training_windows(
                X_train, y_train,
                target_classes=args.subsample_classes,
                fraction=args.subsample_fraction,
                seed=args.subsample_seed,
                class_names=args.class_names,
            )
            train_counts_post = np.bincount(y_train.astype(int), minlength=args.nb_classes)
            print(f'  Subsample classes={args.subsample_classes} fraction={args.subsample_fraction} seed={args.subsample_seed}:')
            print(f'    train windows: {int(train_counts_pre.sum())} -> {int(train_counts_post.sum())}')
            for c in range(args.nb_classes):
                if train_counts_pre[c] != train_counts_post[c]:
                    print(f'    {args.class_names[c]}: {int(train_counts_pre[c])} -> {int(train_counts_post[c])}')

        # augment training windows of the targeted classes -- runs AFTER subsampling so the two
        # compose (subsample first, then augment), and BEFORE the subject-id column is stripped
        # below since augment_training_windows also reads X_train[:, :, 0] to carry subject id
        # onto augmented copies. Eval (X_val, y_val) is never touched here. Segment-level (not
        # per-window) because batch = context for this repo's context-aware networks -- see
        # data_processing/augmentation.py's module docstring.
        if args.augment_classes and not args.dense:
            aug_counts_pre = np.bincount(y_train.astype(int), minlength=args.nb_classes)
            n_before = len(y_train)
            n_subject_splices_baseline = max(0, len(np.unique(X_train[:, 0, 0])) - 1)

            X_train, y_train, aug_stats = augment_training_windows(
                X_train, y_train,
                target_classes=args.augment_classes,
                multiplier=args.augment_multiplier,
                recipe=args.augment_recipe,
                seed=args.augment_seed,
                class_names=args.class_names,
                context_k=args.augment_context_k,
                sw_overlap=args.sw_overlap,
            )
            aug_counts_post = np.bincount(y_train.astype(int), minlength=args.nb_classes)

            print(f'  Augment classes={args.augment_classes} multiplier={args.augment_multiplier} '
                  f'recipe={args.augment_recipe} seed={args.augment_seed} context_k={args.augment_context_k}:')
            print(f'    train windows: {int(aug_counts_pre.sum())} -> {int(aug_counts_post.sum())}')
            print(f'    segments appended: {aug_stats["n_segments"]} '
                  f'(vs. {n_subject_splices_baseline} baseline subject-boundary splices already in this fold)')
            print(f'    target-class counts (anchor-driven; always exactly pre*(multiplier-1) by construction):')
            for c in args.augment_classes:
                cid = args.class_names.index(c)
                expected = int(aug_counts_pre[cid]) * (args.augment_multiplier - 1)
                actual_anchor = aug_stats['anchor_added'][c]
                actual_total = int(aug_counts_post[cid]) - int(aug_counts_pre[cid])
                flag = '' if actual_total == expected else '  <-- spillover contamination from a co-targeted class, NOT exactly N x'
                print(f'      {c}: {int(aug_counts_pre[cid])} -> {int(aug_counts_post[cid])} '
                      f'(anchor-added={actual_anchor}, expected={expected}, total-added={actual_total}){flag}')
            print(f'    spillover table (non-anchor windows added, by their own true class):')
            for c in args.class_names:
                sp = aug_stats['spillover_added'][c]
                if sp:
                    print(f'      {c}: +{sp}')
            if aug_stats['example_overlap_check'] is not None:
                chk = aug_stats['example_overlap_check']
                print(f'    example overlap-consistency check (first segment, adjacent augmented windows): '
                      f'exact_match={chk["exact_match"]} max_abs_diff={chk["max_abs_diff"]:.3g}')

        if not args.dense:
            X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

        if args.dense:
            # y_* are (N_seq, T) with ignore_index padding, so np.bincount cannot read them;
            # count the per-sample label streams instead. Same quantity, one row per sample.
            val_counts   = np.bincount(args.dense_val_sample_labels,   minlength=args.nb_classes)
            train_counts = np.bincount(args.dense_train_sample_labels, minlength=args.nb_classes)
        else:
            val_counts   = np.bincount(y_val.astype(int),   minlength=args.nb_classes)
            train_counts = np.bincount(y_train.astype(int), minlength=args.nb_classes)
        print(f'  Subject: {args.subjects[int(sbj)]}')
        print(f'  Windows — train: {len(y_train)}, val: {len(y_val)}')
        print(f'  Val   per-class: { {args.class_names[c]: int(val_counts[c])   for c in range(args.nb_classes)} }')
        print(f'  Train per-class: { {args.class_names[c]: int(train_counts[c]) for c in range(args.nb_classes)} }')

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        # network initialization
        if args.network == 'deepconvcontext':
            net = DeepConvContext(args.batch_size, args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob, args.bidirectional, args.context_type, args.nb_attention_heads, args.transformer_depth)
        elif args.network == 'deepconvlstm':
            net = DeepConvLSTM(args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob)
        elif args.network == 'attendanddiscriminate':        
            net = AttendAndDiscriminate(args.nb_channels, args.nb_classes, args.nb_units_lstm, args.nb_filters, args.filter_width, args.nb_layers_lstm, False, args.drop_prob, 0.5, 0.5, 'ReLU', 1, args.gpu, args.weights_init)
        elif args.network == 'shallow_deepconvlstm':     
            net = ShallowDeepConvLSTM(args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob)
        elif args.network == 'tinyhar':
            _tinyhar = TinyHAR_Model(
                input_shape=(1, 1, args.window_size, args.nb_channels),
                number_class=args.nb_classes,
                filter_num=args.filter_num,
                cross_channel_interaction_type=args.cross_channel_interaction_type,
                cross_channel_aggregation_type=args.cross_channel_aggregation_type,
                temporal_info_interaction_type=args.temporal_info_interaction_type,
                temporal_info_aggregation_type=args.temporal_info_aggregation_type
            )
            net = TinyHARWrapper(_tinyhar)
        elif args.network == 'tinierhar':
            _tinierhar = TinierHAR_Model(
                input_shape=(1, 1, args.window_size, args.nb_channels),
                nb_classes=args.nb_classes,
                filter_scaling_factor=1,
                config={
                    'nb_conv_blocks': args.nb_conv_blocks,
                    'nb_units_gru': args.nb_units_gru,
                    'nb_filters': args.nb_filters,
                    'drop_prob': args.drop_prob,
                }
            )
            net = TinierHARWrapper(_tinierhar)
        elif args.network == 'icgnet':
            net = ICGNet(
                channels=args.nb_channels,
                classes=args.nb_classes,
                window_size=args.window_size,
                drop_prob=args.drop_prob,
            )
        elif args.network == 'inceptioncontext':
            net = InceptionContext(
                args.batch_size, args.nb_channels, args.nb_classes, args.window_size,
                lstm_units=args.nb_units_lstm,
                lstm_layers=args.nb_layers_lstm,
                dropout=args.drop_prob,
                bidirectional=args.bidirectional,
                filter_sizes=args.filter_sizes,
                branch_filters=args.branch_filters,
                nb_units_gru_ic=args.nb_units_gru_ic,
                use_channel_affine=args.use_channel_affine,
                branch_dilations=args.branch_dilations,
                dense=args.dense,
            )
        else:
            print("Did not provide a valid network name!")

        # optimizer initialization
        opt = init_optimizer(net, args)

        # optimizer initialization
        loss = init_loss(args)

        # lr scheduler initialization
        if args.adj_lr:
            print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
            scheduler = init_scheduler(opt, args)
        else:
            scheduler = None

        net, checkpoint, val_output, train_output, best_epoch = train(X_train, y_train, X_val, y_val,
                                                          network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler,
                                                          config=vars(args), run=run, name='sbj_' + str(int(sbj))
                                                          )

        if args.dense:
            # One prediction per raw sample, not per (sequence, timestep) -- see
            # stitch_dense_predictions. Everything below (scores, csv dump, npz dump) then
            # consumes the same (N, 2) [y_pred, y_true] layout the windowed path produces.
            n_dense_timesteps = len(val_output)
            val_output = stitch_dense_predictions(val_output, dense_entries, dense_kept_rows,
                                                  len(data))
            print(f'  Dense stitching  {n_dense_timesteps} timesteps -> {len(val_output)} '
                  f'samples (last-sequence-wins over overlapping sequences)')
            if len(val_output) != len(args.dense_val_sample_labels):
                raise ValueError(
                    f"dense stitching produced {len(val_output)} samples but the loader kept "
                    f"{len(args.dense_val_sample_labels)}"
                )
            if not np.array_equal(val_output[:, 1], args.dense_val_sample_labels.astype(int)):
                raise ValueError(
                    "dense stitching: the stitched ground truth does not match the loader's "
                    "val_sample_labels; the sequence order assumption is violated."
                )

        if args.save_checkpoints:
            print('Saving checkpoint...')
            if args.valid_epoch == 'last':
                c_name = os.path.join(log_dir, "checkpoint_last_{}_{}.pth".format(args.subjects[int(sbj)], str(args.name)))
            else:
                c_name = os.path.join(log_dir, "checkpoint_best_{}_{}.pth".format(args.subjects[int(sbj)], str(args.name)))
            torch.save(checkpoint, c_name)

        if args.save_predictions:
            print('Saving predictions...')
            if args.valid_epoch == 'last':
                p_name = os.path.join(log_dir, "predictions_last_{}_{}.csv".format(args.subjects[int(sbj)], str(args.name)))
            else:
                p_name = os.path.join(log_dir, "predictions_best_{}_{}.csv".format(args.subjects[int(sbj)], str(args.name)))
            pd.DataFrame(val_output).to_csv(p_name)

        if args.save_val_npz:
            print('Saving per-window val predictions (npz)...')
            # augmentation runs get an _aug<mult><recipe> suffix so their filename does NOT match
            # learning_curve_plot.py's parse_pred_filename (which only recognizes a bare float or
            # seed<N> as the last underscore-token) -- this makes discover_npz's recursive
            # --log_root scan skip these files by construction instead of silently colliding with
            # a subsample_fraction=1.0 baseline run on the (fold, fraction, seed) key.
            aug_suffix = (
                f"_aug{args.augment_multiplier}{args.augment_recipe}" if args.augment_classes else ""
            )
            npz_name = os.path.join(log_dir, "preds_{}_{}_seed{}{}.npz".format(
                args.subjects[int(sbj)], args.subsample_fraction, args.seed, aug_suffix))
            np.savez(npz_name, y_pred=val_output[:, 0].astype(int), y_true=val_output[:, 1].astype(int))

        if all_val_output is None:
            all_train_output = train_output
            all_val_output = val_output
        else:
            all_train_output = np.concatenate((all_train_output, train_output), axis=0)
            all_val_output = np.concatenate((all_val_output, val_output), axis=0)

        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        t_conf_mat = confusion_matrix(train_output[:, 1], train_output[:, 0], normalize='true', labels=labels)
        t_acc = t_conf_mat.diagonal() / t_conf_mat.sum(axis=1)
        t_prec = precision_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        t_rec = recall_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        t_f1 = f1_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        
        v_conf_mat = confusion_matrix(val_output[:, 1], val_output[:, 0], normalize='true', labels=labels)
        v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
        v_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        v_rec = recall_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        v_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        
        cp_scores[0, :, int(sbj)] = v_acc
        cp_scores[1, :, int(sbj)] = v_prec
        cp_scores[2, :, int(sbj)] = v_rec
        cp_scores[3, :, int(sbj)] = v_f1

        # record best epoch and macro F1 for this subject
        subject_best_records.append({
            'subject': args.subjects[int(sbj)],
            'best_epoch': best_epoch,
            'best_macro_f1': float(np.nanmean(v_f1))
        })

        # fill values for train val gap evaluation
        train_val_gap[0, int(sbj)] = np.nanmean(t_acc) - np.nanmean(v_acc)
        train_val_gap[1, int(sbj)] = np.nanmean(t_prec) - np.nanmean(v_prec)
        train_val_gap[2, int(sbj)] = np.nanmean(t_rec) - np.nanmean(v_rec)
        train_val_gap[3, int(sbj)] = np.nanmean(t_f1) - np.nanmean(v_f1)

        print("SUBJECT {0} VALIDATION RESULTS: ".format(args.subjects[int(sbj)]))
        print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
        print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
        print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
        print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))
        
         # save per-subject confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        ax.set_title('Confusion Matrix Subject ' + str(int(sbj)))
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=v_conf_mat, display_labels=args.class_names)    
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
        plt.savefig(os.path.join(log_dir, 'conf_mats', 'sbj_' + str(int(sbj)) + '.png'))
        if run is not None:
            cm_path = os.path.join(log_dir, "conf_mats", f"sbj_{int(sbj)}.png")
            wandb.log({f"conf_matrices/sbj_{int(sbj)}": wandb.Image(cm_path)})

    if args.save_analysis:
        cp_score_acc = pd.DataFrame(cp_scores[0, :, :], index=None)
        cp_score_acc.index = args.class_names
        cp_score_acc.columns = args.subjects
        cp_score_prec = pd.DataFrame(cp_scores[1, :, :], index=None)
        cp_score_prec.index = args.class_names
        cp_score_prec.columns = args.subjects
        cp_score_rec = pd.DataFrame(cp_scores[2, :, :], index=None)
        cp_score_rec.index = args.class_names
        cp_score_rec.columns = args.subjects
        cp_score_f1 = pd.DataFrame(cp_scores[3, :, :], index=None)
        cp_score_f1.index = args.class_names
        cp_score_f1.columns = args.subjects
        tv_gap = pd.DataFrame(train_val_gap, index=None)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_gap.columns = args.subjects
        
        cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc_{}.csv'.format(args.name)))
        cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec_{}.csv').format(args.name))
        cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec_{}.csv').format(args.name))
        cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1_{}.csv').format(args.name))
        tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap_{}.csv').format(args.name))
        
        if run is not None:    
            for fname in [
                f"cp_scores_acc_{args.name}.csv",
                f"cp_scores_prec_{args.name}.csv",
                f"cp_scores_rec_{args.name}.csv",
                f"cp_scores_f1_{args.name}.csv",
                f"train_val_gap_{args.name}.csv",
            ]:
                wandb.save(os.path.join(log_dir, fname), policy="now")

    # fill values for normal evaluation
    labels = list(range(0, args.nb_classes))
    t_conf_mat = confusion_matrix(all_train_output[:, 1], all_train_output[:, 0], normalize='true', labels=labels)
    t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
    t_prec = precision_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_rec = recall_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_f1 = f1_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
        
    v_conf_mat = confusion_matrix(all_val_output[:, 1], all_val_output[:, 0], normalize='true', labels=labels)
    v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
    v_prec = precision_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_rec = recall_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_f1 = f1_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
        
    print("FINAL VALIDATION RESULTS: ")
    print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
    print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
    print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
    print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))
    
    print("FINAL VALIDATION RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(v_acc))
    print("Precision: {0}".format(v_prec))
    print("Recall: {0}".format(v_rec))
    print("F1: {0}".format(v_f1))

    print("GENERALIZATION GAP ANALYSIS: ")
    print("Train-Val-Accuracy Difference: {0}".format(np.nanmean(t_acc) - np.nanmean(v_acc)))
    print("Train-Val-Precision Difference: {0}".format(np.nanmean(t_prec) - np.nanmean(v_prec)))
    print("Train-Val-Recall Difference: {0}".format(np.nanmean(t_rec) - np.nanmean(v_rec)))
    print("Train-Val-F1 Difference: {0}".format(np.nanmean(t_f1) - np.nanmean(v_f1)))

    # print per-subject best epoch and F1 summary
    print('\nPER-SUBJECT BEST EPOCH & F1 SUMMARY')
    print('-------------------------------------')
    print('{:<20} {:>12} {:>12}'.format('Subject', 'Best Epoch', 'Best F1 (%)'))
    for record in subject_best_records:
        print('{:<20} {:>12} {:>12.2f}'.format(
            str(record['subject']), record['best_epoch'], record['best_macro_f1'] * 100))

    # save per-subject summary to CSV
    summary_df = pd.DataFrame(subject_best_records)
    summary_path = os.path.join(log_dir, 'subject_best_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print('\nSaved per-subject summary to: {}'.format(summary_path))
    if run is not None:
        wandb.save(summary_path, policy='now')
        wandb.log({'per_subject_summary': wandb.Table(dataframe=summary_df)})

    # save final composite confusion matrix
    save_composite_confusion_matrix(v_conf_mat, args.class_names, log_dir, run,
                                    title='Composite Confusion Matrix (All Subjects)')


def train_valid_split(train_data, valid_data, args, log_dir=None, run=None):
    """
    Method to apply normal cross-validation, i.e. one set split into train, validation and testing data.

    :param train_data: numpy array
        Data used for training
    :param valid_data: numpy array
        Data used for validation
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_dir: string
        Logging directory
    :return pytorch model
        Trained network
    """
    print('\nCALCULATING TRAIN-VALID-SPLIT SCORES.\n')
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

    X_val, y_val = apply_sliding_window(valid_data[:, :-1], valid_data[:, -1],
                                        sliding_window_size=args.sw_length,
                                        unit=args.sw_unit,
                                        sampling_rate=args.sampling_rate,
                                        sliding_window_overlap=args.sw_overlap,
                                        )

    X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

    args.window_size = X_train.shape[1]
    args.nb_channels = X_train.shape[2]

    # network initialization
    if args.network == 'deepconvcontext':
            net = DeepConvContext(args.batch_size, args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob, args.bidirectional, args.context_type, args.nb_attention_heads, args.transformer_depth)
    elif args.network == 'deepconvlstm':
        net = DeepConvLSTM(args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob)
    elif args.network == 'attendanddiscriminate':        
            net = AttendAndDiscriminate(args.nb_channels, args.nb_classes, args.nb_units_lstm, args.nb_filters, args.filter_width, args.nb_layers_lstm, False, args.drop_prob, 0.5, 0.5, 'ReLU', 1, args.gpu)
    elif args.network == 'shallow_deepconvlstm':     
            net = ShallowDeepConvLSTM(args.nb_channels, args.nb_classes, args.window_size, args.nb_filters, args.filter_width, args.nb_units_lstm, args.nb_layers_lstm, args.drop_prob)
    elif args.network == 'tinyhar':
        _tinyhar = TinyHAR_Model(
            input_shape=(1, 1, args.window_size, args.nb_channels),
            number_class=args.nb_classes,
            filter_num=args.filter_num,
            cross_channel_interaction_type=args.cross_channel_interaction_type,
            cross_channel_aggregation_type=args.cross_channel_aggregation_type,
            temporal_info_interaction_type=args.temporal_info_interaction_type,
            temporal_info_aggregation_type=args.temporal_info_aggregation_type
        )
        net = TinyHARWrapper(_tinyhar)
    elif args.network == 'tinierhar':
        _tinierhar = TinierHAR_Model(
            input_shape=(1, 1, args.window_size, args.nb_channels),
            nb_classes=args.nb_classes,
            filter_scaling_factor=1,
            config={
                'nb_conv_blocks': args.nb_conv_blocks,
                'nb_units_gru': args.nb_units_gru,
                'nb_filters': args.nb_filters,
                'drop_prob': args.drop_prob,
            }
        )
        net = TinierHARWrapper(_tinierhar)
    elif args.network == 'icgnet':
        net = ICGNet(
            channels=args.nb_channels,
            classes=args.nb_classes,
            window_size=args.window_size,
            drop_prob=args.drop_prob,
        )
    elif args.network == 'inceptioncontext':
        net = InceptionContext(
            args.batch_size, args.nb_channels, args.nb_classes, args.window_size,
            lstm_units=args.nb_units_lstm,
            lstm_layers=args.nb_layers_lstm,
            dropout=args.drop_prob,
            bidirectional=args.bidirectional,
            filter_sizes=args.filter_sizes,
            branch_filters=args.branch_filters,
            nb_units_gru_ic=args.nb_units_gru_ic,
            use_channel_affine=args.use_channel_affine,
            branch_dilations=args.branch_dilations,
        )
    else:
        print("Did not provide a valid network name!")

    # optimizer initialization
    opt = init_optimizer(net, args)

    # optimizer initialization
    loss = init_loss(args)

    # lr scheduler initialization
    if args.adj_lr:
        print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
        scheduler = init_scheduler(opt, args)
    else:
        scheduler = None

    net, checkpoint, val_output, train_output, best_epoch = train(X_train, y_train, X_val, y_val,
                                                      network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler,
                                                      config=vars(args), run=run, name='split'
                                                      )

    if args.save_checkpoints:
        print('Saving checkpoint...')
        if args.valid_epoch == 'last':
            c_name = os.path.join(log_dir, "checkpoint_last_{}.pth".format(str(args.name)))
        else:
            c_name = os.path.join(log_dir, "checkpoint_best_{}.pth".format(str(args.name)))
        torch.save(checkpoint, c_name)

    if args.save_predictions:
        print('Saving predictions...')
        if args.valid_epoch == 'last':
            p_name = os.path.join(log_dir, "predictions_last_{}.csv".format(str(args.name)))
        else:
            p_name = os.path.join(log_dir, "predictions_best_{}.csv".format(str(args.name)))
        pd.DataFrame(val_output).to_csv(p_name)

    # fill values for normal evaluation
    labels = list(range(0, args.nb_classes))
    t_conf_mat = confusion_matrix(train_output[:, 1], train_output[:, 0], normalize='true', labels=labels)
    t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
    t_prec = precision_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_rec = recall_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_f1 = f1_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        
    v_conf_mat = confusion_matrix(val_output[:, 1], val_output[:, 0], normalize='true', labels=labels)
    v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
    v_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_rec = recall_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)        
    
    print('VALIDATION RESULTS (macro): ')
    print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
    print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
    print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
    print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))

    print("VALIDATION RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(v_acc))
    print("Precision: {0}".format(v_prec))
    print("Recall: {0}".format(v_rec))
    print("F1: {0}".format(v_f1))

    print("GENERALIZATION GAP ANALYSIS: ")
    print("Train-Val-Accuracy Difference: {0}".format(np.nanmean(t_acc) - np.nanmean(v_acc)))
    print("Train-Val-Precision Difference: {0}".format(np.nanmean(t_prec) - np.nanmean(v_prec)))
    print("Train-Val-Recall Difference: {0}".format(np.nanmean(t_rec) - np.nanmean(v_rec)))
    print("Train-Val-F1 Difference: {0}".format(np.nanmean(t_f1) - np.nanmean(v_f1)))

    if args.save_analysis:
        tv_results = pd.DataFrame([v_acc, v_prec, v_rec, v_f1], columns=args.class_names)
        tv_results.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_gap = pd.DataFrame([t_acc - v_acc, t_prec - v_prec, t_rec - v_rec, t_f1 - v_f1],
                              columns=args.class_names)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_results.to_csv(os.path.join(log_dir, 'split_scores_{}.csv'.format(args.name)))
        tv_gap.to_csv(os.path.join(log_dir, 'tv_gap_{}.csv'.format(args.name)))

        if run is not None:    
            for fname in [
                f"split_scores_{args.name}.csv",
                f"tv_gap_{args.name}.csv",
            ]:
                wandb.save(os.path.join(log_dir, fname), policy="now")
    
    # save final postprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=v_conf_mat, display_labels=args.class_names)    
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
    plt.savefig(os.path.join(log_dir, 'conf_mats', 'all.png'))
    if run is not None:
        all_path = os.path.join(log_dir, "conf_mats", "all.png")
        wandb.log({"conf_matrices/all": wandb.Image(all_path)})
    
    # submit final values to wandb 
    if run is not None:
        run.summary["final_accuracy"] = float(np.nanmean(v_acc))
        run.summary["final_precision"] = float(np.nanmean(v_prec))
        run.summary["final_recall"] = float(np.nanmean(v_rec))
        run.summary["final_f1"] = float(np.nanmean(v_f1))

        run.summary["train-val-acc-diff"] = float(np.nanmean(t_acc) - np.nanmean(v_acc))
        run.summary["train-val-prec-diff"] = float(np.nanmean(t_prec) - np.nanmean(v_prec))
        run.summary["train-val-rec-diff"] = float(np.nanmean(t_rec) - np.nanmean(v_rec))
        run.summary["train-val-f1-diff"] = float(np.nanmean(t_f1) - np.nanmean(v_f1))
        
    return net
