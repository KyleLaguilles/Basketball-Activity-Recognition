for spec in "4 warp" "4 rotate" "8 conservative" "8 warp" "8 rotate"; do
  set -- $spec
  M=$1; R=$2
  nohup python src/main.py --network inceptioncontext --test_type subset_specific --test_case loso_G \
    --loso_subjects b512,a0da,4d70,ce9d,9bd4 --seed 1 --epochs 40 --batch_size 100 \
    --sw_length 1.0 --sw_unit seconds --sw_overlap 50 \
    --nb_units_lstm 128 --nb_layers_lstm 1 --bidirectional \
    --filter_sizes 1 3 5 11 --branch_filters 32 64 64 64 --nb_units_gru_ic 128 \
    --drop_prob 0.5 --learning_rate 1e-4 --weight_decay 1e-06 \
    --optimizer adam --weights_init xavier_normal --weight_scheme sqrt_inverse \
    --augment_classes rebound,layup --augment_multiplier $M --augment_recipe $R \
    --augment_context_k 2 --augment_seed 1 \
    --gpu cuda:0 --wandb --save_val_npz --save_checkpoints --name aug_m${M}_${R}_seed1_v2 \
    > logs/aug_m${M}_${R}_seed1_v2.log 2>&1 &
  sleep 2
done