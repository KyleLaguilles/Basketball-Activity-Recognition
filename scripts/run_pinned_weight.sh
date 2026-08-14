#!/usr/bin/env bash
#
# Pinned-weight m2_conservative run.
#
# Every flag below is verbatim from the aug grid's m2_conservative cell
#   logs/subset_specific/loso_G/inceptioncontext/2026-07-17_17-55-45_aug_m2_conservative_seed1/cfg.txt
# with exactly one addition: --pin_weights_to_pre_augmentation.
#
# WHAT THAT CHANGES: augmentation still runs identically (same seed, same recipe, same
# segments -- the training set still grows by the same windows), but the sqrt_inverse class
# weights are derived from the PRE-augmentation counts instead of the post-augmentation ones.
# In the m2 grid cell, doubling rebound/layup drops their loss weight by ~18%, so the grid's
# m2 cells conflate "more data for the target class" with "less weight on the target class"
# (the confound documented in v5 section 4). This run holds the weights at their baseline
# values so the remaining delta vs. the unaugmented baseline is the augmentation effect alone.
#
# Diff this run's log against the original m2_conservative log: the
# [pin_weights_to_pre_augmentation] block prints per-class pre-aug vs post-aug counts, and the
# [weight_scheme=sqrt_inverse] block should now match the BASELINE run's weights, not the m2 run's.
#
# Note on comparing weights across runs: sqrt_inverse weights are normalized to mean 1, so
# every class's printed weight shifts when any one class's support changes. Compare the RATIO
# of a touched class to an untouched one (e.g. rebound/sitting) across runs, not a single
# class's absolute weight -- see print_weight_summary's docstring in src/model/train.py.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

$PYTHON src/main.py \
  --network inceptioncontext \
  --name aug_m2_conservative_seed1_pinnedw \
  --test_type subset_specific \
  --test_case loso_G \
  --seed 1 \
  --sw_length 1.0 \
  --sw_unit seconds \
  --sw_overlap 50 \
  --valid_type cross-participant \
  --valid_epoch best \
  --batch_size 100 \
  --epochs 40 \
  --optimizer adam \
  --learning_rate 1e-4 \
  --weight_decay 1e-06 \
  --weights_init xavier_normal \
  --loss cross_entropy \
  --smoothing 0.0 \
  --gpu cuda:0 \
  --weight_scheme sqrt_inverse \
  --weight_exponent 0.5 \
  --pin_weights_to_pre_augmentation \
  --lr_scheduler step_lr \
  --lr_step 10 \
  --lr_decay 0.9 \
  --es_patience 10 \
  --nb_units_lstm 128 \
  --nb_layers_lstm 1 \
  --nb_filters 64 \
  --filter_width 11 \
  --drop_prob 0.5 \
  --pool_type max \
  --pool_kernel_width 2 \
  --bidirectional \
  --context_type lstm \
  --transformer_depth 3 \
  --filter_num 20 \
  --nb_conv_blocks 4 \
  --nb_units_gru 64 \
  --filter_sizes 1 3 5 11 \
  --branch_filters 32 64 64 64 \
  --branch_dilations 1 1 1 1 \
  --nb_units_gru_ic 128 \
  --cross_channel_interaction_type attn \
  --cross_channel_aggregation_type FC \
  --temporal_info_interaction_type gru \
  --temporal_info_aggregation_type FC \
  --loso_subjects b512,a0da,4d70,ce9d,9bd4 \
  --subsample_fraction 1.0 \
  --subsample_seed 0 \
  --save_val_npz \
  --augment_classes rebound,layup \
  --augment_multiplier 2 \
  --augment_recipe conservative \
  --augment_seed 1 \
  --augment_context_k 2 \
  --print_freq 100 \
  --wandb
