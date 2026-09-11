#!/usr/bin/env bash
#
# Full 24-fold LOSO, DENSE mode (batch_size 10), seeds 1-3 -- one main.py process per seed.
#
# Each process holds out every participant in turn (main.py's fold loop over
# --loso_subjects), so each seed lands in ONE log dir holding all 24 folds:
#   logs/subset_specific/loso_G/inceptioncontext/<timestamp>_24fold_dense_seed<N>/
# which is what analysis/sample_level_f1.py --dense and _dense_common.py read (24 folds
# is their EXPECTED_FOLD_COUNT_ALL / ALL_SUBJECT_FOLD_COUNT).
#
# Flags are the dense_b10 baseline (scripts/run_dense.sh at --batch_size 10, the dense
# settings as stated in scripts/run_seq_len_sweep.sh at seq_len 500) with:
#   --loso_subjects   all 24 participant keys instead of the 5 placeholders
#   added             --save_predictions --save_checkpoints --save_analysis --wandb
# Every main.py flag is stated explicitly, defaults included. The few that cannot be
# stated without changing behavior are listed in the NOT-passed block at the end of ARGS.
#
# W&B: main.py:143-147 hardcodes project="hangtime_har" and the SAME run name
# ("subset_specific_loso_G_inceptioncontext") for every run. Find these runs by:
#   group        24fold_dense                     (WANDB_RUN_GROUP)
#   tags         24fold, dense, seed<N>           (WANDB_TAGS)
#   config.name  24fold_dense_seed<N>             (--name; also the log dir suffix)
# wandb.init's explicit project/name win over env vars, but group/tags are not passed
# there, so the env vars take effect (checked against wandb 0.25.0).
#
# Requires data/seam_map.json built with participant keying (gitignored -- on a fresh
# checkout run `python scripts/build_seam_map.py` first). Checked before launching.
#
# Runtime: 5-fold dense_b10 runs took 0.29-0.42 h, ~0.06-0.08 h/fold, so expect
# ~1.5-2 h per seed and ~4.5-6 h for three seeds back to back.
#
# Usage (from anywhere -- the script cds to the repo root):
#   DRY_RUN=1 bash scripts/run_24fold_dense.sh           # print the commands, run nothing
#   nohup bash scripts/run_24fold_dense.sh > logs/24fold_dense.out 2>&1 &
#   SEEDS="2 3" bash scripts/run_24fold_dense.sh         # a subset, e.g. one seed per pod

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3}"
WANDB_GROUP="24fold_dense"
SEAM_MAP="data/seam_map.json"

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

# A stale (14-code) or missing seam map would otherwise only fail inside the first fold.
"$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from data_processing.sequence_loader import load_seam_map
segs = load_seam_map('$SEAM_MAP')
print(f'seam map ok: {len(segs)} segments, {len({s[\"subject_code\"] for s in segs})} participant codes')
"

ARGS=(
    # --- dataset (sw_* are inert under --dense, stated so cfg.txt matches the baseline) ---
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
    --batch_size 10
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

    # --- subsampling: skipped under --dense (validation.py gates it on `not args.dense`);
    #     kept so cfg.txt matches the dense baseline ---
    --subsample_classes rebound,layup
    --subsample_fraction 1.0
    --subsample_seed 0

    # --- augmentation: off (--augment_classes unset); main.py's default values ---
    --augment_multiplier 1
    --augment_recipe conservative
    --augment_seed 0
    --augment_context_k 2

    # --- dense ---
    --dense
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

    # NOT passed -- store_true switches left off, or options whose default is "unset";
    # passing any of them changes behavior:
    #   --include_void --use_channel_affine --weighted --shuffling --adj_lr
    #   --early_stopping --verbose --pin_weights_to_pre_augmentation --no_bilstm
    #   --weight_cap       (valid only with --weight_scheme capped)
    #   --augment_classes  (unset = no augmentation; --dense rejects multiplier > 1 anyway)
    # --seed and --name are set per iteration below.
)

for SEED in $SEEDS; do
    NAME="24fold_dense_seed${SEED}"
    TAGS="24fold,dense,seed${SEED}"
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
