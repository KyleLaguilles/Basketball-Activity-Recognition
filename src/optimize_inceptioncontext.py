# requires: pip install optuna
##################################################
# Bayesian hyperparameter optimization for InceptionContext
# Uses Optuna TPE + MedianPruner, logs each trial to W&B
##################################################

import json
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from data_processing.preprocess_data import load_dataset
from data_processing.sliding_window import apply_sliding_window
from model.InceptionContext import InceptionContext
from model.train import init_weights
from misc.torchutils import seed_worker

# ---------------------------------------------------------------------------
# Fixed config — matches the split_DandWvsG evaluation setup
# ---------------------------------------------------------------------------
SEED          = 1
BATCH_SIZE    = 100
EPOCHS        = 30
WEIGHT_DECAY  = 1e-6
WEIGHTS_INIT  = 'xavier_normal'
FILTER_SIZES  = (1, 3, 5, 11)
NB_LAYERS_LSTM = 1
SW_LENGTH     = 1.0
SW_UNIT       = 'seconds'
SW_OVERLAP    = 50

# Class ordering verified from data_processing/preprocess_data.py
CLASS_NAMES = [
    'dribbling',   # 0
    'shot',        # 1
    'pass',        # 2
    'rebound',     # 3
    'layup',       # 4
    'walking',     # 5
    'running',     # 6
    'standing',    # 7
    'sitting',     # 8
]



def objective(trial, X_train, y_train, X_val, y_val, nb_classes, nb_channels, window_size, device):
    # ---- Reproducibility ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Fixed units ----
    nb_units_lstm   = 128
    nb_units_gru_ic = 128

    # ---- Sample hyperparameters ----
    bf_k1   = trial.suggest_categorical('bf_k1',  [32, 64])
    bf_k3   = trial.suggest_categorical('bf_k3',  [64, 128])
    bf_k5   = trial.suggest_categorical('bf_k5',  [64, 128])
    bf_k11  = trial.suggest_categorical('bf_k11', [64, 128])
    branch_filters = (bf_k1, bf_k3, bf_k5, bf_k11)

    drop_prob     = trial.suggest_float('drop_prob',     0.3,  0.6)
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
    smoothing     = trial.suggest_float('label_smoothing', 0.0, 0.05)
    lr_patience   = trial.suggest_int('lr_patience', 3, 8)

    # ---- Build model ----
    net = InceptionContext(
        batch_size=BATCH_SIZE,
        channels=nb_channels,
        classes=nb_classes,
        window_size=window_size,
        lstm_units=nb_units_lstm,
        lstm_layers=NB_LAYERS_LSTM,
        dropout=drop_prob,
        bidirectional=True,
        filter_sizes=FILTER_SIZES,
        branch_filters=branch_filters,
        nb_units_gru_ic=nb_units_gru_ic,
    )
    net = init_weights(net, WEIGHTS_INIT)
    net.to(device)

    # ---- Loss with balanced class weights ----
    base_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(base_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)

    # ---- DataLoaders ----
    g = torch.Generator()
    g.manual_seed(SEED)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    use_pin = (device.type == 'cuda') and torch.cuda.is_available()
    trainloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        worker_init_fn=seed_worker, generator=g, pin_memory=use_pin,
    )
    valloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=seed_worker, generator=g, pin_memory=use_pin,
    )

    labels = list(range(nb_classes))
    best_val_f1 = 0.0

    # ---- Training loop ----
    for epoch in range(EPOCHS):
        net.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y.long())
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        net.eval()
        val_preds_parts, val_gt_parts, val_loss_parts = [], [], []
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                val_loss_parts.append(criterion(logits, y.long()).item())
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                gt    = y.cpu().numpy().flatten()
                val_preds_parts.append(preds.astype(int))
                val_gt_parts.append(gt.astype(int))

        avg_val_loss = float(np.mean(val_loss_parts))
        scheduler.step(avg_val_loss)

        val_preds = np.concatenate(val_preds_parts)
        val_gt    = np.concatenate(val_gt_parts)
        val_f1    = float(f1_score(val_gt, val_preds, average='macro', zero_division=1, labels=labels))
        best_val_f1 = max(best_val_f1, val_f1)

        wandb.log({
            'trial':        trial.number,
            'epoch':        epoch + 1,
            'val/f1_macro': val_f1,
            'val/loss':     avg_val_loss,
        }, step=trial.number * EPOCHS + epoch)

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---- Load data (once, outside the trial loop) ----
    print('Loading data...')
    train_data, valid_data, subjects, nb_classes, class_names, sampling_rate, has_void = load_dataset(
        test_type='session_specific',
        test_case='split_DandWvsG',
        include_void=False,
    )

    assert class_names == CLASS_NAMES, (
        f'Unexpected class ordering from dataset: {class_names}\n'
        f'Expected: {CLASS_NAMES}'
    )

    X_train, y_train = apply_sliding_window(
        train_data[:, :-1], train_data[:, -1],
        sliding_window_size=SW_LENGTH,
        unit=SW_UNIT,
        sampling_rate=sampling_rate,
        sliding_window_overlap=SW_OVERLAP,
    )
    X_val, y_val = apply_sliding_window(
        valid_data[:, :-1], valid_data[:, -1],
        sliding_window_size=SW_LENGTH,
        unit=SW_UNIT,
        sampling_rate=sampling_rate,
        sliding_window_overlap=SW_OVERLAP,
    )

    # Strip subject-ID column (column 0 of sensor block)
    X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

    window_size = X_train.shape[1]
    nb_channels = X_train.shape[2]

    print(f'X_train: {X_train.shape}  X_val: {X_val.shape}')
    print(f'nb_classes={nb_classes}  nb_channels={nb_channels}  window_size={window_size}')
    print(f'class_names: {class_names}')

    # ---- Single W&B run for the entire sweep ----
    run = wandb.init(
        project='hangtime_har',
        name='optuna_ic_sweep',
        config={
            'n_trials':     80,
            'seed':         SEED,
            'epochs':       EPOCHS,
            'batch_size':   BATCH_SIZE,
            'filter_sizes': list(FILTER_SIZES),
        },
    )

    wandb_callback = WeightsAndBiasesCallback(
        wandb_kwargs={'project': 'hangtime_har'},
        as_multirun=False,
    )

    # ---- Optuna study ----
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner()
    study   = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val,
            nb_classes, nb_channels, window_size, device,
        ),
        n_trials=80,
        callbacks=[wandb_callback],
    )

    run.finish()

    # ---- Print best results ----
    best = study.best_trial
    print('\n' + '=' * 60)
    print(f'Best trial:     #{best.number}')
    print(f'Best macro F1:  {best.value:.4f}')
    print('Best hyperparameters:')
    for k, v in best.params.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    # ---- Save JSON ----
    os.makedirs('results', exist_ok=True)
    best_params = {
        'nb_units_lstm':   128,
        'nb_units_gru_ic': 128,
        'bf_k1':           best.params['bf_k1'],
        'bf_k3':           best.params['bf_k3'],
        'bf_k5':           best.params['bf_k5'],
        'bf_k11':          best.params['bf_k11'],
        'drop_prob':       round(best.params['drop_prob'], 4),
        'learning_rate':   round(best.params['learning_rate'], 7),
        'label_smoothing': round(best.params['label_smoothing'], 4),
        'lr_patience':     best.params['lr_patience'],
        'weighted':        'balanced',
    }
    out_path = os.path.join('results', 'optuna_inceptioncontext_best.json')
    with open(out_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'\nBest hyperparameters saved to: {out_path}')


if __name__ == '__main__':
    main()
