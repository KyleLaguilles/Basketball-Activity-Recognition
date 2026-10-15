#!/usr/bin/env bash
#
# Last-epoch robustness check: full 24-fold LOSO, InceptionContext, WINDOWED and DENSE,
# one seed, scored at the LAST epoch instead of the best one.
#
# Why: the headline runs use --valid_epoch best, which keeps the epoch with the highest
# macro F1 on the held-out participant. This rerun asks whether the windowed -> dense gap
# survives when no epoch is chosen with the test participant's labels.
#
# Flags are copied verbatim from scripts/run_24fold_windowed.sh and
# scripts/run_24fold_dense.sh. The ONLY intended differences from those runs are:
#   --valid_epoch   best -> last
#   --name          24fold_<mode>_seed<N>  ->  24fold_<mode>_last_seed<N>
# Training does not depend on --valid_epoch (it only decides which epoch's predictions
# are returned, train.py:747-780), and training is deterministic for a given seed, so each
# run should follow the same trajectory as its original and stop at epoch 40.
#
# Check before reading any number: diff each new cfg.txt against the original run's
# cfg.txt for the same mode and seed. Expected differences: valid_epoch and name only.
#
# Output dirs:
#   logs/subset_specific/loso_G/inceptioncontext/<timestamp>_24fold_windowed_last_seed<N>/
#   logs/subset_specific/loso_G/inceptioncontext/<timestamp>_24fold_dense_last_seed<N>/
#
# Usage (from anywhere -- the script cds to the repo root):
#   DRY_RUN=1 bash scripts/run_24fold_lastepoch.sh            # print the commands, run nothing
#   nohup bash scripts/run_24fold_lastepoch.sh > logs/24fold_lastepoch.out 2>&1 &
#   MODES="dense" bash scripts/run_24fold_lastepoch.sh        # one mode only
#   PARALLEL=0 bash scripts/run_24fold_lastepoch.sh           # run the modes one after another
#
# By default both modes run at the same time on the one GPU (PARALLEL=1), which keeps pod
# utilization up. Each process seeds itself, so running them together does not affect
# determinism. Per-mode output goes to logs/24fold_<mode>_last_seed<N>.out.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SEED="${SEED:-1}"
MODES="${MODES:-windowed dense}"
PARALLEL="${PARALLEL:-1}"
WANDB_GROUP="24fold_lastepoch"
SEAM_MAP="data/seam_map.json"

# All 24 participants (<4-hex id>_<eu|na>), in LabelEncoder code order 0..23.
PARTICIPANTS=(
    05d8_eu 0846_eu 0846_na 10f0_eu 10f0_na 2dd9_eu 2dd9_na 4991_eu
    4d70_eu 4d70_na 9bd4_eu 9bd4_na a0da_eu a0da_na ac59_eu ac59_na
    b512_eu b512_na c6f3_na ce9d_eu ce9d_na e90f_eu f2ad_eu f2ad_na
)
LOSO_SUBJECTS="$(IFS=,; echo "${PARTICIPANTS[*]}")"

expected="$(zcat labels_export.csv.gz | tail -n +2 | cut -d, -f1 | sort -u | paste -sd, -)"
given="$(printf '%s\n' "${PARTICIPANTS[@]}" | sort -u | paste -sd, -)"
if [ "${#PARTICIPANTS[@]}" -ne 24 ] || [ "$given" != "$expected" ]; then
    echo "ERROR: PARTICIPANTS (${#PARTICIPANTS[@]}) != the 24 keys in labels_export.csv.gz" >&2
    exit 1
fi

for MODE in $MODES; do
    case "$MODE" in
        windowed|dense) ;;
        *) echo "ERROR: unknown mode '$MODE' (expected windowed or dense)" >&2; exit 1 ;;
    esac
done

if [[ " $MODES " == *" dense "* ]]; then
    "$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from data_processing.sequence_loader import load_seam_map
segs = load_seam_map('$SEAM_MAP')
print(f'seam map ok: {len(segs)} segments, {len({s[\"subject_code\"] for s in segs})} participant codes')
"
fi

# Shared by both modes: identical to the two original scripts except --valid_epoch.
COMMON_ARGS=(
    # --- dataset ---
    --test_type subset_specific
    --test_case loso_G
    --sw_length 1.0
    --sw_unit seconds
    --sw_overlap 50

    # --- network: InceptionContext ---
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
    --valid_epoch last                 # <-- the one change from the original runs
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

    # --- subsampling: fraction 1.0 is a no-op (and skipped under --dense) ---
    --subsample_classes rebound,layup
    --subsample_fraction 1.0
    --subsample_seed 0

    # --- augmentation: off ---
    --augment_multiplier 1
    --augment_recipe conservative
    --augment_seed 0
    --augment_context_k 2

    # --- dense settings: inert without --dense ---
    --dense_seq_len 500
    --dense_overlap 0.5
    --dense_min_seg 25
    --dense_seam_map "$SEAM_MAP"

    # --- outputs / logging ---
    --print_freq 100
    --save_val_npz
    --save_predictions
    --save_checkpoints
    --save_analysis
    --wandb

    # NOT passed (as in the originals): --include_void --use_channel_affine --weighted
    #   --shuffling --adj_lr --early_stopping --verbose --pin_weights_to_pre_augmentation
    #   --no_bilstm --weight_cap --augment_classes
)

run_mode() {
    local MODE="$1"
    local MODE_ARGS NAME TAGS
    if [ "$MODE" = "dense" ]; then
        MODE_ARGS=(--batch_size 10 --dense)
    else
        MODE_ARGS=(--batch_size 100)
    fi
    NAME="24fold_${MODE}_last_seed${SEED}"
    TAGS="24fold,${MODE},lastepoch,seed${SEED}"
    local cmd=("$PYTHON" src/main.py "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" --seed "$SEED" --name "$NAME")

    echo "=== $(date '+%F %T')  ${MODE}  seed ${SEED}  name=${NAME} ==="
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'WANDB_RUN_GROUP=%q WANDB_TAGS=%q ' "$WANDB_GROUP" "$TAGS"
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi
    WANDB_RUN_GROUP="$WANDB_GROUP" WANDB_TAGS="$TAGS" "${cmd[@]}"
    echo "=== $(date '+%F %T')  ${MODE} seed ${SEED} done ==="
}

mkdir -p logs
if [ "$PARALLEL" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    pids=()
    for MODE in $MODES; do
        run_mode "$MODE" > "logs/24fold_${MODE}_last_seed${SEED}.out" 2>&1 &
        pids+=($!)
        echo "launched ${MODE} (pid $!), output in logs/24fold_${MODE}_last_seed${SEED}.out"
        sleep 5   # distinct log-dir timestamps
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    exit "$status"
else
    for MODE in $MODES; do
        run_mode "$MODE"
    done
fi
