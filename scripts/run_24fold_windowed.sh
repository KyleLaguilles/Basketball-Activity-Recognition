#!/usr/bin/env bash
#
# Full 24-fold LOSO, WINDOWED mode, seeds 1-3 -- one main.py process per seed.
#
# Each process holds out every participant in turn (main.py's fold loop over
# --loso_subjects), so each seed lands in ONE log dir holding all 24 folds:
#   logs/subset_specific/loso_G/inceptioncontext/<timestamp>_24fold_windowed_seed<N>/
# which is the one-results_dir-per-run layout the analysis scripts read.
#
# Flags are the windowed loso_G baseline (scripts/run_baseline_ckpt.sh) with:
#   --loso_subjects   all 24 participant keys instead of the 5 placeholders
#   added             --save_predictions --save_checkpoints --save_analysis --wandb
# Every main.py flag is stated explicitly, defaults included. The few that cannot be
# stated without changing behavior are listed in the NOT-passed block at the end of ARGS.
#
# W&B: main.py:143-147 hardcodes project="hangtime_har" and the SAME run name
# ("subset_specific_loso_G_inceptioncontext") for every run. Find these runs by:
#   group        24fold_windowed                  (WANDB_RUN_GROUP)
#   tags         24fold, windowed, seed<N>        (WANDB_TAGS)
#   config.name  24fold_windowed_seed<N>          (--name; also the log dir suffix)
# wandb.init's explicit project/name win over env vars, but group/tags are not passed
# there, so the env vars take effect (checked against wandb 0.25.0).
#
# Runtime: the 5-fold windowed baseline (2026-07-17_07-56-30) took 1.77 h, ~0.35 h/fold,
# so expect ~8.5 h per seed and ~25 h for three seeds back to back.
#
# Usage (from anywhere -- the script cds to the repo root):
#   DRY_RUN=1 bash scripts/run_24fold_windowed.sh        # print the commands, run nothing
#   nohup bash scripts/run_24fold_windowed.sh > logs/24fold_windowed.out 2>&1 &
#   SEEDS="2 3" bash scripts/run_24fold_windowed.sh      # a subset, e.g. one seed per pod

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3}"
WANDB_GROUP="24fold_windowed"

# All 24 participants (<4-hex id>_<eu|na>), in LabelEncoder code order 0..23.
PARTICIPANTS=(
    05d8_eu 0846_eu 0846_na 10f0_eu 10f0_na 2dd9_eu 2dd9_na 4991_eu
    4d70_eu 4d70_na 9bd4_eu 9bd4_na a0da_eu a0da_na ac59_eu ac59_na
    b512_eu b512_na c6f3_na ce9d_eu ce9d_na e90f_eu f2ad_eu f2ad_na
)
LOSO_SUBJECTS="$(IFS=,; echo "${PARTICIPANTS[*]}")"

# main.py rejects unknown --loso_subjects names but silently runs fewer folds if one is
# missing, so check the list is exactly the participant set in the label export.
expected="$(zcat labels_export.csv.gz | tail -n +2 | cut -d, -f1 | sort -u | paste -sd, -)"
given="$(printf '%s\n' "${PARTICIPANTS[@]}" | sort -u | paste -sd, -)"
if [ "${#PARTICIPANTS[@]}" -ne 24 ] || [ "$given" != "$expected" ]; then
    echo "ERROR: PARTICIPANTS (${#PARTICIPANTS[@]}) != the 24 keys in labels_export.csv.gz" >&2
    echo "  given:    $given" >&2
    echo "  expected: $expected" >&2
    exit 1
fi

ARGS=(
    # --- dataset ---
    --test_type subset_specific
    --test_case loso_G
    --sw_length 1.0
    --sw_unit seconds
    --sw_overlap 50

    # --- network: InceptionContext (flags for the other architectures are inert, stated anyway) ---
    --network inceptioncontext
    --nb_units_lstm 128
    --nb_layers_lstm 1
    --nb_filters 64
    --filter_width 11
    --drop_prob 0.5
    --pool_type max
    --pool_kernel_width 2
    --bidirectional
    --context_type lstm
    --nb_attention_heads 4
    --transformer_depth 3
    --filter_num 20
    --nb_conv_blocks 4
    --nb_units_gru 64
    --filter_sizes 1 3 5 11
    --branch_filters 32 64 64 64
    --branch_dilations 1 1 1 1
    --nb_units_gru_ic 128
    --cross_channel_interaction_type attn
    --cross_channel_aggregation_type FC
    --temporal_info_interaction_type gru
    --temporal_info_aggregation_type FC

    # --- training ---
    --valid_type cross-participant
    --valid_epoch best
    --batch_size 100
    --epochs 40
    --optimizer adam
    --learning_rate 0.0001
    --weight_decay 1e-06
    --weights_init xavier_normal
    --loss cross_entropy
    --smoothing 0.0
    --gpu cuda:0
    --weight_scheme sqrt_inverse
    --weight_exponent 0.5
    --lr_scheduler step_lr
    --lr_step 10
    --lr_decay 0.9
    --es_patience 10

    # --- folds: every participant ---
    --loso_subjects "$LOSO_SUBJECTS"

    # --- subsampling: fraction 1.0 is a no-op, kept so cfg.txt matches the baseline ---
    --subsample_classes rebound,layup
    --subsample_fraction 1.0
    --subsample_seed 0

    # --- augmentation: off (--augment_classes unset); main.py's default values ---
    --augment_multiplier 1
    --augment_recipe conservative
    --augment_seed 0
    --augment_context_k 2

    # --- dense: off in this script; main.py's default values, inert without --dense ---
    --dense_seq_len 500
    --dense_overlap 0.5
    --dense_min_seg 25
    --dense_seam_map data/seam_map.json

    # --- outputs / logging ---
    --print_freq 100
    --save_val_npz
    --save_predictions
    --save_checkpoints
    --save_analysis
    --wandb

    # NOT passed -- store_true switches left off, or options whose default is "unset";
    # passing any of them changes behavior:
    #   --include_void --use_channel_affine --weighted --shuffling --adj_lr
    #   --early_stopping --verbose --pin_weights_to_pre_augmentation --dense --no_bilstm
    #   --weight_cap       (valid only with --weight_scheme capped)
    #   --augment_classes  (unset = no augmentation)
    # --seed and --name are set per iteration below.
)

for SEED in $SEEDS; do
    NAME="24fold_windowed_seed${SEED}"
    TAGS="24fold,windowed,seed${SEED}"
    cmd=("$PYTHON" src/main.py "${ARGS[@]}" --seed "$SEED" --name "$NAME")

    echo "=== $(date '+%F %T')  seed ${SEED}  name=${NAME}  wandb group=${WANDB_GROUP} tags=${TAGS} ==="
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'WANDB_RUN_GROUP=%q WANDB_TAGS=%q ' "$WANDB_GROUP" "$TAGS"
        printf '%q ' "${cmd[@]}"
        echo
        continue
    fi
    WANDB_RUN_GROUP="$WANDB_GROUP" WANDB_TAGS="$TAGS" "${cmd[@]}"
    echo "=== $(date '+%F %T')  seed ${SEED} done ==="
done
