##################################################
# Main script used to commence experiments
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import argparse
import datetime
import wandb
import json
import os
import time
import sys

from data_processing.preprocess_data import load_dataset
from model.validation import cross_participant_cv, train_valid_split

from misc.logging import Logger
from misc.torchutils import seed_torch

"""
DATASET OPTIONS:
- TEST_TYPE: type of experiment commenced (see data preprocess_data.py for insights)
- TEST_CASE: type of test case conducted (see data preprocess_data.py for insights)
- SW_LENGTH: length of sliding window
- SW_UNIT: unit in which length of sliding window is measured
- SW_OVERLAP: overlap ratio between sliding windows (in percent, i.e. 60 = 60%)
- INCLUDE_VOID: boolean whether to include void class in datasets
"""

TEST_TYPE = 'hypertuning'
TEST_CASE = 'loso_us'
SW_LENGTH = 1
SW_UNIT = 'seconds'
SW_OVERLAP = 60
INCLUDE_VOID = False

"""
NETWORK OPTIONS:
- NETWORK: network architecture to be used (e.g. 'deepconvlstm')
- NB_UNITS_LSTM: number of hidden units in each LSTM layer
- NB_LAYERS_LSTM: number of layers in LSTM
- NB_FILTERS: number of convolution filters employed in each layer of convolution blocks
- FILTER_WIDTH: width of convolution filters (e.g. 11 = 11x1 filter)
- DROP_PROB: dropout probability in dropout layers
- BIDIRECTIONAL: enable the use of a bidirectional LSTM (currently only affects deepconvcontext)
- TYPEOFCONTEXT: select the type of deepconvcontext architecture (e.g. 'lstm', 'self-attention', or 'transformer')
- NB_ATTENTION_HEADS: Number of attention heads for self-attention and transformer models
- TRANSFORMER_DEPTH: Depth of the transformer model (currently only affects deepconvcontext)
"""

NETWORK = 'deepconvlstm'
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
NB_FILTERS = 64
FILTER_WIDTH = 11
DROP_PROB = 0.5
POOL_TYPE = 'max'
POOL_KERNEL_WIDTH = 2
BIDIRECTIONAL = False
TYPE_OF_CONTEXT = 'lstm'
NB_ATTENTION_HEADS = 4
TRANSFORMER_DEPTH = 3
FILTER_NUM = 20
CROSS_CHANNEL_INTERACTION_TYPE = 'attn'
NB_CONV_BLOCKS = 4
NB_UNITS_GRU = 64
FILTER_SIZES = (1, 3, 5, 11)
BRANCH_FILTERS = (32, 64, 64, 64)
BRANCH_DILATIONS = (1, 1, 1, 1)
NB_UNITS_GRU_IC = 128
CROSS_CHANNEL_AGGREGATION_TYPE = 'FC'
TEMPORAL_INFO_INTERACTION_TYPE = 'gru'
TEMPORAL_INFO_AGGREGATION_TYPE = 'FC'

"""
TRAINING OPTIONS:
- SEED: random seed which is to be employed
- VALID_TYPE: (cross-)validation type; either 'cross-participant', 'split' or 'kfold'
- VALID_EPOCH: which epoch used for evaluation; either 'best' or 'last'
- BATCH_SIZE: size of the batches
- EPOCHS: number of epochs during training
- OPTIMIZER: optimizer to use; either 'rmsprop', 'adadelta' or 'adam'
- LR: learning rate to employ for optimizer
- WEIGHT_DECAY: weight decay to employ for optimizer
- WEIGHTS_INIT: weight initialization method to use to initialize network
- LOSS: loss to use ('cross_entropy', 'maxup')
- SMOOTHING: degree of label smoothing employed if cross-entropy used
- GPU: name of GPU to use (e.g. 'cuda:0')
- WEIGHTED: boolean whether to use weighted loss calculation based on support of each class
- SHUFFLING: boolean whether to use shuffling during training
- ADJ_LR: boolean whether to adjust learning rate if no improvement
- LR_SCHEDULER: type of learning rate scheduler to employ ('step_lr', 'reduce_lr_on_plateau')
- LR_STEP: step size of learning rate scheduler (patience if plateau).
- LR_DECAY: decay factor of learning rate scheduler.
- EARLY_STOPPING: boolean whether to stop the network training early if no improvement 
- ES_PATIENCE: patience (i.e. number of epochs) after which network training is stopped if no improvement
"""

SEED = 1
VALID_TYPE = 'cross-participant'
VALID_EPOCH = 'best'
BATCH_SIZE = 100
EPOCHS = 30
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross_entropy'
SMOOTHING = 0.0
GPU = 'cuda:0'
WEIGHTED = False
SHUFFLING = False
ADJ_LR = False
LR_SCHEDULER = 'step_lr'
LR_STEP = 10
LR_DECAY = 0.9
EARLY_STOPPING = False
ES_PATIENCE = 10

"""
LOGGING OPTIONS:
- NAME: name of the experiment; used for logging purposes
- WANDB: boolean whether to use wandb.ai for logging (please provide credentials below!)
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_PREDICTIONS: boolean whether to save predictions made by models
- SAVE_MODEL: boolean whether to save the model after last epoch as a checkpoint file
- SAVE_ANALYSIS: boolean whether to save analysis dataframe, i.e. csv containing all scores
"""

NAME = 'test_experiment'
WANDB = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_PREDICTIONS = False
SAVE_CHECKPOINTS = False
SAVE_ANALYSIS = False

def main(args):
    if args.wandb:
        run = wandb.init(
            project="hangtime_har",
            name=f"{args.test_type}_{args.test_case}_{args.network}",
            config=vars(args),
        )
    else:
        run = None

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    safe_ts = ts.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('logs', args.test_type, args.test_case, args.network, f"{safe_ts}_{args.name}")
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    cfg_path = os.path.join(log_dir, 'cfg.txt')
    with open(cfg_path, 'w') as fid:
        json.dump(vars(args), fid, indent=2)

    # upload cfg to W&B (shows under the run's Files)
    if run is not None:
        wandb.save(cfg_path, policy="now")

    # apply the chosen random seed to all relevant parts
    seed_torch(args.seed)

    ################################################## DATA LOADING ####################################################

    print('Loading data...')
    train, valid, subjects, nb_classes, class_names, sampling_rate, has_void = \
        load_dataset(test_type=args.test_type, test_case=args.test_case, include_void=args.include_void)

    args.subjects = subjects
    args.sampling_rate = sampling_rate
    args.nb_classes = nb_classes
    args.class_names = class_names
    args.has_void = has_void

    if args.subsample_classes:
        unknown = sorted(set(args.subsample_classes) - set(class_names))
        if unknown:
            raise ValueError(f"--subsample_classes contains unknown class name(s): {unknown}. "
                              f"Valid class names: {class_names}")
    if args.augment_classes:
        unknown = sorted(set(args.augment_classes) - set(class_names))
        if unknown:
            raise ValueError(f"--augment_classes contains unknown class name(s): {unknown}. "
                              f"Valid class names: {class_names}")
    if args.loso_subjects:
        known_subjects = {str(s) for s in subjects}
        unknown = sorted(set(args.loso_subjects) - known_subjects)
        if unknown:
            raise ValueError(f"--loso_subjects contains unknown subject name(s): {unknown}. "
                              f"Valid subjects: {sorted(known_subjects)}")

    ############################################# TRAINING #############################################################

    if valid is None:
        print("LOSO dataset with size: | {0} |".format(train.shape))
    else:
        print("Split datasets with size: | train {0} | valid {1} |".format(train.shape, valid.shape))

    if valid is None:
        _ = cross_participant_cv(train, args, log_dir, run)
    else:
        _ = train_valid_split(train, valid, args, log_dir, run)

    print("\nALL FINISHED")

    if args.wandb and run is not None:
        run.finish()  # close W & B run properly

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATASET OPTIONS
    parser.add_argument('--print_freq', default=PRINT_FREQ, type=int)
    parser.add_argument('--test_type', default=TEST_TYPE, type=str)
    parser.add_argument('--test_case', default=TEST_CASE, type=str)
    parser.add_argument('--sw_length', default=SW_LENGTH, type=float)
    parser.add_argument('--sw_unit', default=SW_UNIT, type=str)
    parser.add_argument('--sw_overlap', default=SW_OVERLAP, type=int)
    parser.add_argument('--include_void', default=INCLUDE_VOID, action='store_true')

    # NETWORK OPTIONS
    parser.add_argument('--network', default=NETWORK, type=str)
    parser.add_argument('--nb_units_lstm', default=NB_UNITS_LSTM, type=int)
    parser.add_argument('--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int)
    parser.add_argument('--nb_filters', default=NB_FILTERS, type=int)
    parser.add_argument('--filter_width', default=FILTER_WIDTH, type=int)
    parser.add_argument('--drop_prob', default=DROP_PROB, type=float)
    parser.add_argument('--pool_type', default=POOL_TYPE, type=str)
    parser.add_argument('--pool_kernel_width', default=POOL_KERNEL_WIDTH, type=int)
    parser.add_argument('--bidirectional', default=BIDIRECTIONAL, action='store_true')
    parser.add_argument('--use_channel_affine', action='store_true', help='Enable learnable per-channel input scaling (gamma*x + beta) before the inception branches.')
    parser.add_argument('--context_type', default=TYPE_OF_CONTEXT, type=str)
    parser.add_argument('--nb_attention_heads', default=NB_ATTENTION_HEADS, type=str)
    parser.add_argument('--transformer_depth', default=TRANSFORMER_DEPTH, type=int)

    # TRAINING OPTIONS
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--valid_type', default=VALID_TYPE, type=str)
    parser.add_argument('--valid_epoch', default=VALID_EPOCH, type=str)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str)
    parser.add_argument('--learning_rate', default=LR, type=float)
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    parser.add_argument('--weights_init', default=WEIGHTS_INIT, type=str)
    parser.add_argument('--loss', default=LOSS, type=str)
    parser.add_argument('--smoothing', default=SMOOTHING, type=float)
    parser.add_argument('--gpu', default=GPU, type=str)
    parser.add_argument('--weighted', default=WEIGHTED, action='store_true')
    parser.add_argument('--weight_scheme', default=None, choices=['none', 'inverse', 'sqrt_inverse', 'capped', 'power_inverse'], type=str,
                         help='Class weighting scheme for the loss. Bare --weighted maps to inverse; '
                              'omitting both --weighted and --weight_scheme maps to none.')
    parser.add_argument('--weight_cap', default=None, type=float,
                         help='Required iff --weight_scheme capped: max/min balanced-weight ratio allowed after capping.')
    parser.add_argument('--weight_exponent', default=0.5, type=float,
                         help='Exponent for --weight_scheme power_inverse: class_weights = (1/class_counts) ** exponent.')
    parser.add_argument('--pin_weights_to_pre_augmentation', default=False, action='store_true',
                         help='Derive class weights from the PRE-augmentation training counts instead of '
                              'the post-augmentation ones. Augmentation still runs and the training set '
                              'still grows -- only the loss weights are pinned. Isolates the augmentation '
                              'effect from the weight confound (a target class gets more windows AND a '
                              'smaller weight, so the two changes are entangled by default). Applies to '
                              '--weight_scheme sqrt_inverse/capped only; ignored (with a warning) for the '
                              'other schemes, and a no-op on folds where no augmentation ran.')
    parser.add_argument('--shuffling', default=SHUFFLING, action='store_true')
    parser.add_argument('--adj_lr', default=ADJ_LR, action='store_true')
    parser.add_argument('--lr_scheduler', default=LR_SCHEDULER, type=str)
    parser.add_argument('--lr_step', default=LR_STEP, type=int)
    parser.add_argument('--lr_decay', default=LR_DECAY, type=float)
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, action='store_true')
    parser.add_argument('--es_patience', default=ES_PATIENCE, type=int)
    parser.add_argument('--filter_num', default=FILTER_NUM, type=int)
    parser.add_argument('--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int)
    parser.add_argument('--nb_units_gru', default=NB_UNITS_GRU, type=int)
    parser.add_argument('--filter_sizes', default=FILTER_SIZES, nargs='+', type=int)
    parser.add_argument('--branch_filters', default=BRANCH_FILTERS, nargs='+', type=int)
    parser.add_argument('--branch_dilations', default=BRANCH_DILATIONS, nargs='+', type=int)
    parser.add_argument('--nb_units_gru_ic', default=NB_UNITS_GRU_IC, type=int)
    parser.add_argument('--cross_channel_interaction_type', default=CROSS_CHANNEL_INTERACTION_TYPE, type=str)
    parser.add_argument('--cross_channel_aggregation_type', default=CROSS_CHANNEL_AGGREGATION_TYPE, type=str)
    parser.add_argument('--temporal_info_interaction_type', default=TEMPORAL_INFO_INTERACTION_TYPE, type=str)
    parser.add_argument('--temporal_info_aggregation_type', default=TEMPORAL_INFO_AGGREGATION_TYPE, type=str)

    # LEARNING-CURVE / SUBSAMPLING OPTIONS (default-off; no effect unless --subsample_classes is set)
    parser.add_argument('--subsample_classes', default=None, type=str,
                         help='Comma-separated class names whose training windows are subsampled '
                              '(e.g. "rebound,layup"). Unset = no subsampling (default pipeline behavior).')
    parser.add_argument('--subsample_fraction', default=1.0, type=float,
                         help='Fraction of each listed class\'s training windows to keep per subject. '
                              'No-op at 1.0.')
    parser.add_argument('--subsample_seed', default=0, type=int,
                         help='RNG seed for deterministic, subject-stratified subsampling.')
    parser.add_argument('--loso_subjects', default=None, type=str,
                         help='Comma-separated subject names to restrict the LOSO fold loop to '
                              '(e.g. "b512,a0da"). Unset = run all folds (default pipeline behavior).')
    parser.add_argument('--save_val_npz', default=False, action='store_true',
                         help='Save per-window val predictions/true labels for each fold to '
                              'preds_<fold>_<fraction>.npz under the run log dir.')

    # AUGMENTATION OPTIONS (default-off; no effect unless --augment_classes is set)
    parser.add_argument('--augment_classes', default=None, type=str,
                         help='Comma-separated class names whose training windows are augmented '
                              '(e.g. "rebound,layup"). Unset = no augmentation (default pipeline behavior).')
    parser.add_argument('--augment_multiplier', default=1, type=int,
                         help='Each real window of a target class yields augment_multiplier-1 '
                              'augmented copies (total augment_multiplier x). 1 = off (no-op).')
    parser.add_argument('--augment_recipe', default='conservative', type=str,
                         choices=['conservative', 'warp', 'rotate'],
                         help='conservative = jitter+scale+magwarp; warp = conservative+timewarp; '
                              'rotate = warp+3D rotation of the accel triplet.')
    parser.add_argument('--augment_seed', default=0, type=int,
                         help='RNG seed for deterministic augmentation transform sampling.')
    parser.add_argument('--augment_context_k', default=2, type=int,
                         help='Segment radius in windows: each augmented copy is built from the '
                              'contiguous span [i-k, i+k] around the target window i (truncated at '
                              'subject boundaries), not the window alone -- required because this '
                              'repo\'s context-aware networks (InceptionContext, DeepConvContext) '
                              'read the DataLoader batch dimension as a temporal sequence.')

    # LOGGING OPTIONS
    parser.add_argument('--name', default=NAME, type=str)
    parser.add_argument('--wandb', default=WANDB, action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--verbose', default=VERBOSE, action='store_true')
    parser.add_argument('--save_predictions', default=SAVE_PREDICTIONS, action='store_true')
    parser.add_argument('--save_checkpoints', default=SAVE_CHECKPOINTS, action='store_true')
    parser.add_argument('--save_analysis', default=SAVE_ANALYSIS, action='store_true')

    args = parser.parse_args()

    # resolve --weighted / --weight_scheme into a single args.weight_scheme; hard-error on contradictions
    if args.weight_scheme is None:
        args.weight_scheme = 'inverse' if args.weighted else 'none'
    elif args.weighted and args.weight_scheme != 'inverse':
        parser.error(
            f"--weighted implies weight_scheme=inverse, but --weight_scheme {args.weight_scheme!r} "
            "was also given. Remove --weighted or set --weight_scheme inverse to resolve the contradiction."
        )
    if (args.weight_scheme == 'capped') != (args.weight_cap is not None):
        parser.error(
            "--weight_cap is required iff --weight_scheme capped "
            f"(weight_scheme={args.weight_scheme!r}, weight_cap={args.weight_cap!r})."
        )
    if args.weight_scheme == 'capped' and args.weight_cap <= 1:
        parser.error(f"--weight_cap must be > 1 (got {args.weight_cap!r}).")

    # resolve comma-separated list flags; unset -> empty list -> no-op downstream
    args.subsample_classes = (
        [c.strip() for c in args.subsample_classes.split(',') if c.strip()]
        if args.subsample_classes else []
    )
    args.augment_classes = (
        [c.strip() for c in args.augment_classes.split(',') if c.strip()]
        if args.augment_classes else []
    )
    args.loso_subjects = (
        [s.strip() for s in args.loso_subjects.split(',') if s.strip()]
        if args.loso_subjects else []
    )
    if args.subsample_classes and not (0.0 <= args.subsample_fraction <= 1.0):
        parser.error(f"--subsample_fraction must be in [0, 1] (got {args.subsample_fraction!r}).")
    if args.augment_classes and args.augment_multiplier < 1:
        parser.error(f"--augment_multiplier must be >= 1 (got {args.augment_multiplier!r}).")
    if args.augment_classes and args.augment_context_k < 0:
        parser.error(f"--augment_context_k must be >= 0 (got {args.augment_context_k!r}).")

    main(args)
