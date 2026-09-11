#!/usr/bin/env bash
# scripts/run_seqlen_sweep.sh
#
# Sequence-length sweep (prereg: docs/prereg_seq_len_sweep.md)
# Usage:
#   bash scripts/run_seqlen_sweep.sh 250
#   bash scripts/run_seqlen_sweep.sh 1000
#
# The seq_len=500 baseline already exists (2026-09-04_16-04-41_dense_b10_seed1).
# Do NOT re-run it. This script is for the two new sweep points only.
#
# Pre-launch checklist (prereg §8):
#   - cfg.txt diff against seed-1: only dense_seq_len should differ
#   - Fold count: 5
#   - Total sample count: 614,733
#   - Coverage: 1.00000

set -euo pipefail

SEQ_LEN="${1:?Usage: $0 <seq_len>  (250 or 1000)}"

if [ "$SEQ_LEN" = "500" ]; then
    echo "ERROR: seq_len=500 is the existing baseline. Do not re-run." >&2
    exit 1
fi

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
    --gpu cuda:0 --dense \
    --dense_seq_len 1000 --dense_overlap 0.5 --dense_min_seg 25 \
    --dense_seam_map data/seam_map.json \
    --seed 1 --name "sweep_seqlen1000_seed1"