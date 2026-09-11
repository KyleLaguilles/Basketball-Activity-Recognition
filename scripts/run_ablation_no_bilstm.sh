#!/bin/bash
# Ablation control for the dense path: identical to scripts/run_dense.sh except for
# --no_bilstm, which replaces the dense head's BiLSTM + Linear(256, 9) with a single
# Linear(128, 9) applied directly to the GRU output. Same dense labels, same sequences,
# same batch size, same training loop -- so the delta against run_dense.sh attributes the
# rebound F1 change to the BiLSTM's temporal smoothing rather than to dense labeling.
# Seed is pinned to 1. All flags are stated explicitly, including those matching defaults.
set -euo pipefail
python src/main.py \
    --loso_subjects b512_na,a0da_eu,4d70_eu,ce9d_eu,9bd4_na --save_val_npz \
    --test_type subset_specific --test_case loso_G --network inceptioncontext \
    --sw_length 1.0 --sw_unit seconds --sw_overlap 50 --epochs 40 --batch_size 10 \
    --optimizer adam --learning_rate 0.0001 --weight_decay 1e-06 --weights_init xavier_normal \
    --weight_scheme sqrt_inverse --weight_exponent 0.5 --bidirectional --context_type lstm \
    --nb_units_lstm 128 --nb_layers_lstm 1 --nb_filters 64 --filter_width 11 --drop_prob 0.5 \
    --nb_units_gru_ic 128 --filter_sizes 1 3 5 11 --branch_filters 32 64 64 64 \
    --branch_dilations 1 1 1 1 --loss cross_entropy --valid_epoch best --lr_scheduler step_lr \
    --lr_step 10 --lr_decay 0.9 --subsample_classes rebound,layup --subsample_fraction 1.0 \
    --gpu cuda:0 --dense --no_bilstm \
    --dense_seq_len 500 --dense_overlap 0.5 --dense_min_seg 25 \
    --dense_seam_map data/seam_map.json \
    --seed 3 --name ablation_no_bilstm_seed3

