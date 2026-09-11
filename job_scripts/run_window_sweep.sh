#!/bin/bash
# Window-length sweep: seed 1 screen
# Pre-registration: docs/prereg_window_sweep.md
# Only --sw_length and --name differ from baseline cfg.txt

set -e

# --- 1.5s ---
python src/main.py \
    --network inceptioncontext \
    --name window_1.5s_seed1 \
    --test_type subset_specific \
    --test_case loso_G \
    --seed 1 \
    --epochs 40 \
    --batch_size 100 \
    --learning_rate 0.0001 \
    --weight_decay 1e-06 \
    --weight_scheme sqrt_inverse \
    --bidirectional \
    --sw_length 1.5 \
    --sw_unit seconds \
    --sw_overlap 50 \
    --nb_units_lstm 128 \
    --nb_layers_lstm 1 \
    --nb_filters 64 \
    --filter_width 11 \
    --drop_prob 0.5 \
    --nb_units_gru_ic 128 \
    --filter_sizes 1 3 5 11 \
    --branch_filters 32 64 64 64 \
    --branch_dilations 1 1 1 1 \
    --optimizer adam \
    --weights_init xavier_normal \
    --valid_epoch best \
    --context_type lstm \
    --pool_type max \
    --pool_kernel_width 2 \
    --loss cross_entropy \
    --lr_scheduler step_lr \
    --lr_step 10 \
    --lr_decay 0.9 \
    --loso_subjects b512_na,a0da_eu,4d70_eu,ce9d_eu,9bd4_na \
    --subsample_classes rebound,layup \
    --subsample_fraction 1.0 \
    --subsample_seed 0 \
    --save_val_npz \
    --save_checkpoints \
    --gpu cuda:0 \
    --wandb

echo "=== 1.5s done ==="

# --- 2.0s ---
python src/main.py \
    --network inceptioncontext \
    --name window_2.0s_seed1 \
    --test_type subset_specific \
    --test_case loso_G \
    --seed 1 \
    --epochs 40 \
    --batch_size 100 \
    --learning_rate 0.0001 \
    --weight_decay 1e-06 \
    --weight_scheme sqrt_inverse \
    --bidirectional \
    --sw_length 2.0 \
    --sw_unit seconds \
    --sw_overlap 50 \
    --nb_units_lstm 128 \
    --nb_layers_lstm 1 \
    --nb_filters 64 \
    --filter_width 11 \
    --drop_prob 0.5 \
    --nb_units_gru_ic 128 \
    --filter_sizes 1 3 5 11 \
    --branch_filters 32 64 64 64 \
    --branch_dilations 1 1 1 1 \
    --optimizer adam \
    --weights_init xavier_normal \
    --valid_epoch best \
    --context_type lstm \
    --pool_type max \
    --pool_kernel_width 2 \
    --loss cross_entropy \
    --lr_scheduler step_lr \
    --lr_step 10 \
    --lr_decay 0.9 \
    --loso_subjects b512_na,a0da_eu,4d70_eu,ce9d_eu,9bd4_na \
    --subsample_classes rebound,layup \
    --subsample_fraction 1.0 \
    --subsample_seed 0 \
    --save_val_npz \
    --save_checkpoints \
    --gpu cuda:0 \
    --wandb

echo "=== 2.0s done ==="

# --- 3.0s ---
python src/main.py \
    --network inceptioncontext \
    --name window_3.0s_seed1 \
    --test_type subset_specific \
    --test_case loso_G \
    --seed 1 \
    --epochs 40 \
    --batch_size 100 \
    --learning_rate 0.0001 \
    --weight_decay 1e-06 \
    --weight_scheme sqrt_inverse \
    --bidirectional \
    --sw_length 3.0 \
    --sw_unit seconds \
    --sw_overlap 50 \
    --nb_units_lstm 128 \
    --nb_layers_lstm 1 \
    --nb_filters 64 \
    --filter_width 11 \
    --drop_prob 0.5 \
    --nb_units_gru_ic 128 \
    --filter_sizes 1 3 5 11 \
    --branch_filters 32 64 64 64 \
    --branch_dilations 1 1 1 1 \
    --optimizer adam \
    --weights_init xavier_normal \
    --valid_epoch best \
    --context_type lstm \
    --pool_type max \
    --pool_kernel_width 2 \
    --loss cross_entropy \
    --lr_scheduler step_lr \
    --lr_step 10 \
    --lr_decay 0.9 \
    --loso_subjects b512_na,a0da_eu,4d70_eu,ce9d_eu,9bd4_na \
    --subsample_classes rebound,layup \
    --subsample_fraction 1.0 \
    --subsample_seed 0 \
    --save_val_npz \
    --save_checkpoints \
    --gpu cuda:0 \
    --wandb

echo "=== 3.0s done ==="
echo "Window sweep seed-1 screen complete."

python analysis/sample_level_f1.py \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30 \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-04-47_window_1.5s_seed1 \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-15-51_window_2.0s_seed1 \
        --results_dir logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-24-55_window_3.0s_seed1 \
        --labels labels_export.csv.gz 2>&1

diff <(cat logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30/cfg.txt) \
     <(cat logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-04-47_window_1.5s_seed1/cfg.txt)

diff <(cat logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30/cfg.txt) \
     <(cat logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-15-51_window_2.0s_seed1/cfg.txt)

diff <(cat logs/subset_specific/loso_G/inceptioncontext/2026-07-17_07-56-30/cfg.txt) \
     <(cat logs/subset_specific/loso_G/inceptioncontext/2026-08-24_19-24-55_window_3.0s_seed1/cfg.txt)