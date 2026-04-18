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

    for i, sbj in enumerate(np.unique(data[:, 0])):
        print('\n VALIDATING FOR SUBJECT {0}; {1} OF {2}'.format(args.subjects[int(sbj)], int(sbj) + 1, int(len(np.unique(data[:, 0])))))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
        args.learning_rate = orig_lr

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

        X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

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
            net = TinyHAR_Model(
                input_shape=(1, 1, args.window_size, args.nb_channels),
                number_class=args.nb_classes,
                filter_num=args.filter_num,
                cross_channel_interaction_type=args.cross_channel_interaction_type,
                cross_channel_aggregation_type=args.cross_channel_aggregation_type,
                temporal_info_interaction_type=args.temporal_info_interaction_type,
                temporal_info_aggregation_type=args.temporal_info_aggregation_type
            )
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
        net = TinyHAR_Model(
            input_shape=(1, 1, args.window_size, args.nb_channels),
            number_class=args.nb_classes,
            filter_num=args.filter_num,
            cross_channel_interaction_type=args.cross_channel_interaction_type,
            cross_channel_aggregation_type=args.cross_channel_aggregation_type,
            temporal_info_interaction_type=args.temporal_info_interaction_type,
            temporal_info_aggregation_type=args.temporal_info_aggregation_type
        )
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
