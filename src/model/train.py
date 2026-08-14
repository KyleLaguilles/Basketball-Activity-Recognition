##################################################
# All functions related to training a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################

import os
import random

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from misc.osutils import mkdir_if_missing
from misc.torchutils import count_parameters, seed_worker
import wandb

class TinyHARWrapper(nn.Module):
    """Wraps TinyHAR to handle input shape: (B, T, C) -> (B, 1, T, C)"""
    use_fixup = False  # required by init_weights in train.py
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, T, C) -> (B, 1, T, C)
        return self.model(x)

def init_weights(network, weight_init):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    Args:
        network: torch.nn.Module
            The network to initialize.
        weight_init: str
            The weight initialization method. Options are 'normal', 'orthogonal', 'xavier_uniform', 'xavier_normal',
            'kaiming_uniform', 'kaiming_normal'.
            
    Returns:
        network: torch.nn.Module
            The initialized network.
    """
    for m in network.modules():
        # conv initialisation
        if isinstance(m, nn.Conv2d):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                m.bias.data.fill_(0.0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    if weight_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif weight_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif weight_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif weight_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif weight_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif weight_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
        elif isinstance(m, nn.LayerNorm):
            # Typically, the scale (weight) is initialized to 1 and the bias to 0.
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        # Transformer-related: MultiheadAttention and TransformerEncoderLayer
        elif isinstance(m, nn.MultiheadAttention):
            for attr in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                param = m
                for part in attr.split('.'):
                    param = getattr(param, part)
                if 'weight' in attr:
                    if weight_init == 'normal':
                        nn.init.normal_(param)
                    elif weight_init == 'orthogonal':
                        nn.init.orthogonal_(param)
                    elif weight_init == 'xavier_uniform':
                        nn.init.xavier_uniform_(param)
                    elif weight_init == 'xavier_normal':
                        nn.init.xavier_normal_(param)
                    elif weight_init == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(param)
                    elif weight_init == 'kaiming_normal':
                        nn.init.kaiming_normal_(param)
                else:
                    nn.init.constant_(param, 0)

        elif isinstance(m, nn.TransformerEncoderLayer):
            # Applies to self_attn, linear1, linear2
            init_weights(m.self_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)

        elif isinstance(m, nn.TransformerDecoderLayer):
            init_weights(m.self_attn, weight_init)
            init_weights(m.multihead_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)
            init_weights(m.norm3, weight_init)
    return network

class Maxup(torch.nn.Module):
    """
    A meta-augmentation, returning the worst result from a range of augmentations.
    As in the orignal paper, https://arxiv.org/abs/2002.09024,
    Implementation inspired by https://github.com/JonasGeiping/data-poisoning/
    see forest / data / mixing_data_augmentations.py
    """

    def __init__(self, given_data_augmentation, ntrials=4):
        """Initialize with a given data augmentation module."""
        super().__init__()
        self.augment = given_data_augmentation
        self.ntrials = ntrials
        self.max_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        additional_x, additional_labels = [], []
        for trial in range(self.ntrials):
            x_out, y_out = self.augment(x, y)
            additional_x.append(x_out)
            additional_labels.append(y_out)

        additional_x = torch.cat(additional_x, dim=0)
        additional_labels = torch.cat(additional_labels, dim=0)

        return additional_x, additional_labels

    def maxup_loss(self, outputs, extra_labels):
        """Compute loss. Here the loss is computed as worst-case estimate over the trials."""
        batch_size = outputs.shape[0] // self.ntrials
        correct_preds = (torch.argmax(outputs.data, dim=1) == extra_labels).sum().item() / self.ntrials
        stacked_loss = self.max_criterion(outputs, extra_labels).view(batch_size, self.ntrials, -1)
        loss = stacked_loss.max(dim=1)[0].mean()

        return loss, correct_preds


def my_noise_addition_augmenter(x, y):
    """
    Noise augmenter for maxup loss

    :param x: numpy array
        Features
    :param y: numpy array
        Labels
    :return: numpy array, numpy array
        Features with added noise and labels
    """
    sigma = 0.5
    return x + sigma*torch.randn_like(x), y


def init_loss(config):
    """
    Initialises an loss object for a given network.

    :param config: dict
        General setting dictionary
    :return: loss object
    """
    if config.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)
    elif config.loss == 'maxup':
        return None
    else:
        print("Did not provide a valid loss name!")
        return None
    return criterion


def init_optimizer(network, config):
    """
    Initialises an optimizer object for a given network.

    :param network: pytorch model
        Network for which optimizer and loss are to be initialised
    :param config: dict
        General setting dictionary
    :return: optimizer object
    """
    # define optimizer and loss
    if config.optimizer == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        print("Did not provide a valid optimizer name!")
        return None
    return opt


def init_scheduler(optimizer, config):
    """

    :param optimizer: optimizer object
        Optimizer object used during training
    :param config: dict
        General setting dictionary
    :return:
    """
    if config.lr_scheduler == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_decay)
    elif config.lr_scheduler == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.lr_step, factor=config.lr_decay)
    else:
        print("Did not provide a valid learning scheduler name!")
        return None
    return scheduler

def print_weight_summary(config, scheme, present_labels, all_class_weights):
    """
    Print the per-class weight vector actually applied to the loss this fold:
    scheme name, cap value if any, and class_name: weight (absent classes marked).

    Sum-normalized class weights: for sqrt_inverse/capped/power_inverse, weights
    are rescaled to mean 1 (sum == nb_classes) before printing. Changing one
    class's support shifts every other class's printed weight too, since they
    all share the same fixed budget -- comparing a single class's weight across
    two runs is not meaningful on its own. To check that subsampling recomputed
    weights correctly, compare the RATIO of two classes' weights (one touched,
    one untouched) across runs -- that ratio-of-ratios cancels the normalization
    and isolates the actual frequency-driven change.
    """
    present = set(int(l) for l in present_labels)
    cap_str = f" cap={config['weight_cap']}" if scheme == "capped" else ""
    exp_str = f" exponent={config['weight_exponent']}" if scheme == "power_inverse" else ""
    print(f"\n[weight_scheme={scheme}{cap_str}{exp_str}]")
    class_names = config["class_names"]
    for i in range(config["nb_classes"]):
        w = float(all_class_weights[i])
        tag = "" if i in present else " (absent from fold)"
        print(f"  {class_names[i]}: {w:.4f}{tag}")


def pinned_weight_labels(train_labels, config):
    """
    Return the label array the class-weight formula should be derived from.

    Default (--pin_weights_to_pre_augmentation off): returns `train_labels` itself, so every
    weight scheme sees exactly the array it always saw and results are unchanged.

    With the flag on: rebuilds a synthetic label array from the PRE-augmentation per-class
    counts recorded by cross_participant_cv (args.pre_aug_train_counts, set in
    model/validation.py right before augment_training_windows runs) and returns that instead.
    The augmented windows still train the network -- only the loss weights are pinned, which
    is the point: it separates "the target class got more data" from "the target class got a
    smaller weight because it now has more data", two effects that move together by default.

    Reconstructing labels rather than short-circuiting the formula keeps the existing
    compute_class_weight call untouched. This is exact, not an approximation, for
    sqrt_inverse: balanced weights are N/(K*n_c), and the subsequent division by the mean
    cancels the shared N/K factor, so only the relative counts n_c survive -- identical to
    what the fold would have produced had augmentation never run.

    :param train_labels: numpy array, the actual (post-augmentation) training labels
    :param config: dict, general setting dictionary
    :return: numpy array, labels to derive class weights from
    """
    if not config.get("pin_weights_to_pre_augmentation", False):
        return train_labels

    pre_counts = config.get("pre_aug_train_counts")
    if pre_counts is None:
        print("\n[pin_weights_to_pre_augmentation] ON, but no augmentation ran on this fold -- "
              "pre- and post-augmentation counts are identical, so pinning is a no-op here.")
        return train_labels

    pre_counts = np.asarray(pre_counts, dtype=np.int64)
    post_counts = np.bincount(train_labels.astype(int), minlength=len(pre_counts))
    pinned = np.repeat(np.arange(len(pre_counts)), pre_counts).astype(train_labels.dtype)

    class_names = config["class_names"]
    print("\n[pin_weights_to_pre_augmentation] ON -- class weights derived from PRE-augmentation "
          "counts; the augmented windows still train the network.")
    print(f"  train windows used for weights: {int(pre_counts.sum())} "
          f"(actual training set: {int(post_counts.sum())})")
    print(f"  {'class':<20} {'pre-aug':>10} {'post-aug':>10} {'delta':>10}")
    for c in range(len(pre_counts)):
        delta = int(post_counts[c]) - int(pre_counts[c])
        print(f"  {class_names[c]:<20} {int(pre_counts[c]):>10} {int(post_counts[c]):>10} "
              f"{delta:>+10}")

    return pinned


def train(train_features, train_labels, val_features, val_labels, network, optimizer, loss, config, name=None, run=None, lr_scheduler=None,
):
    """
    Method to train a PyTorch network.

    :param train_features: numpy array
        Training features
    :param train_labels: numpy array
        Training labels
    :param val_features: numpy array
        Validation features
    :param val_labels: numpy array
        Validation labels
    :param network: pytorch model
        DeepConvLSTM network object
    :param optimizer: optimizer object
        Optimizer object
    :param loss: loss object
        Loss object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for logging
    :param log_timestamp: string
        Timestamp used for logging
    :param lr_scheduler: scheduler object, default: None
        Learning rate scheduler object
    :return pytorch model, numpy array, numpy array
        Trained network and training and validation predictions with ground truth
    """

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network, config["weights_init"])

    # normalize / derive device
    device = config.get("gpu", "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # send network to device
    network.to(device)
    network.train()

    # if weighted loss chosen, calculate weights based on training dataset; else each class is weighted equally
    scheme = config["weight_scheme"]
    if config.get("pin_weights_to_pre_augmentation", False) and scheme not in ("sqrt_inverse", "capped"):
        print(f"\n[pin_weights_to_pre_augmentation] WARNING: flag is set but weight_scheme={scheme!r} -- "
              "pinning is only implemented for sqrt_inverse/capped and is being IGNORED. Weights below "
              "are derived from the post-augmentation counts as usual.")
    if scheme == "inverse":
        all_class_weights = torch.from_numpy(np.ones(config["nb_classes"])).float()
        class_weights = torch.from_numpy(
            compute_class_weight(
                "balanced", classes=np.unique(train_labels + 1), y=train_labels + 1
            )
        ).float()
        for i, lbl in enumerate(np.unique(train_labels)):
            all_class_weights[int(lbl)] = class_weights[i]

        if config["loss"] == "cross_entropy":
            loss.weight = all_class_weights.to(device)

        print("Applied weighted class weights: ")
        print(class_weights)
        print_weight_summary(config, scheme, np.unique(train_labels), all_class_weights)
    elif scheme == "none":
        all_class_weights = torch.from_numpy(np.ones(config["nb_classes"])).float()
        class_weights = torch.from_numpy(
            compute_class_weight(None, classes=np.unique(train_labels + 1), y=train_labels + 1)
        ).float()
        for i, lbl in enumerate(np.unique(train_labels)):
            all_class_weights[int(lbl)] = class_weights[i]
        if config["loss"] == "cross_entropy":
            loss.weight = all_class_weights.to(device)
    elif scheme in ("sqrt_inverse", "capped"):
        all_class_weights = torch.from_numpy(np.ones(config["nb_classes"])).float()
        # identical to train_labels unless --pin_weights_to_pre_augmentation is set
        weight_labels = pinned_weight_labels(train_labels, config)
        balanced_weights = compute_class_weight(
            "balanced", classes=np.unique(weight_labels + 1), y=weight_labels + 1
        )
        if scheme == "sqrt_inverse":
            derived_weights = np.sqrt(balanced_weights)
            derived_weights = derived_weights / derived_weights.mean()
        else:  # capped -- clip the ceiling to floor * RATIO (floor = the fold's smallest
               # balanced weight, i.e. the most-frequent class), then renormalize to mean 1
            floor = balanced_weights.min()
            ceiling = floor * config["weight_cap"]
            derived_weights = np.minimum(balanced_weights, ceiling)
            derived_weights = derived_weights / derived_weights.mean()
        class_weights = torch.from_numpy(derived_weights).float()
        for i, lbl in enumerate(np.unique(weight_labels)):
            all_class_weights[int(lbl)] = class_weights[i]
        if config["loss"] == "cross_entropy":
            loss.weight = all_class_weights.to(device)
        print_weight_summary(config, scheme, np.unique(weight_labels), all_class_weights)
    elif scheme == "power_inverse":
        all_class_weights = torch.from_numpy(np.ones(config["nb_classes"])).float()
        present_labels, class_counts = np.unique(train_labels, return_counts=True)
        derived_weights = (1.0 / class_counts) ** config["weight_exponent"]
        derived_weights = derived_weights / derived_weights.mean()
        class_weights = torch.from_numpy(derived_weights).float()
        for i, lbl in enumerate(present_labels):
            all_class_weights[int(lbl)] = class_weights[i]
        if config["loss"] == "cross_entropy":
            loss.weight = all_class_weights.to(device)
        print_weight_summary(config, scheme, present_labels, all_class_weights)
    else:
        raise ValueError(f"Unknown weight_scheme: {scheme!r}")

    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    if config["loss"] == "maxup":
        maxup = Maxup(my_noise_addition_augmenter, ntrials=4)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train_labels)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_features), torch.from_numpy(val_labels)
    )

    g = torch.Generator()
    g.manual_seed(config["seed"])

    use_pin = (device.type == "cuda") and torch.cuda.is_available()

    trainloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=config["shuffling"],
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=use_pin,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=use_pin,
    )

    # counters and objects used for early stopping and learning rate adjustment
    best_metric = 0.0
    best_epoch = 1
    best_val_preds = None
    best_train_preds = None
    early_stop = False
    es_pt_counter = 0
    labels = list(range(0, config["nb_classes"]))

    # store best checkpoint (so "best" truly returns best weights)
    best_checkpoint = None

    # choose default name for logging namespace
    if name is None:
        name = "train"

    # training loop; iterates through epochs
    for e in range(config["epochs"]):
        """
        TRAINING
        """
        train_preds_parts = []
        train_gt_parts = []
        train_losses = []

        start_time = time.time()
        batch_num = 1

        for i, (x, y) in enumerate(trainloader):
            inputs, targets = x.to(device), y.to(device)

            opt.zero_grad()

            if config["loss"] == "maxup":
                inputs, targets = maxup(inputs, targets)

            train_output = network(inputs)

            if config["loss"] == "maxup":
                train_loss = maxup.maxup_loss(train_output, targets.long())[0]
            else:
                train_loss = criterion(train_output, targets.long())

            train_loss.backward()
            opt.step()

            train_losses.append(float(train_loss.item()))

            # predictions / gt (collect then concat once)
            y_preds = np.argmax(train_output.detach().cpu().numpy(), axis=-1)
            y_true = targets.detach().cpu().numpy().flatten()
            train_preds_parts.append(y_preds.astype(int))
            train_gt_parts.append(y_true.astype(int))

            if config["verbose"]:
                if batch_num % config["print_freq"] == 0 and batch_num > 0:
                    cur_loss = float(np.mean(train_losses))
                    elapsed = time.time() - start_time
                    print(
                        "| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | train loss {:5.2f}".format(
                            e, batch_num, elapsed * 1000 / config["batch_size"], cur_loss
                        )
                    )
                    start_time = time.time()
                batch_num += 1

        train_preds = np.concatenate(train_preds_parts) if train_preds_parts else np.array([], dtype=int)
        train_gt = np.concatenate(train_gt_parts) if train_gt_parts else np.array([], dtype=int)

        """
        VALIDATION
        """
        val_preds_parts = []
        val_gt_parts = []
        val_losses = []

        network.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valloader):
                inputs, targets = x.to(device), y.to(device)

                if config["loss"] == "maxup":
                    inputs, targets = maxup(inputs, targets)

                val_output = network(inputs)

                if config["loss"] == "maxup":
                    val_loss = maxup.maxup_loss(val_output, targets.long())[0]
                else:
                    val_loss = criterion(val_output, targets.long())

                val_output = torch.nn.functional.softmax(val_output, dim=1)

                val_losses.append(float(val_loss.item()))

                y_preds = np.argmax(val_output.detach().cpu().numpy(), axis=-1)
                y_true = targets.detach().cpu().numpy().flatten()
                val_preds_parts.append(y_preds.astype(int))
                val_gt_parts.append(y_true.astype(int))

        val_preds = np.concatenate(val_preds_parts) if val_preds_parts else np.array([], dtype=int)
        val_gt = np.concatenate(val_gt_parts) if val_gt_parts else np.array([], dtype=int)

        # evaluation metrics
        t_conf_mat = confusion_matrix(train_gt, train_preds, normalize="true", labels=labels)
        t_acc = t_conf_mat.diagonal() / t_conf_mat.sum(axis=1)
        t_prec = precision_score(train_gt, train_preds, average=None, zero_division=1, labels=labels)
        t_rec = recall_score(train_gt, train_preds, average=None, zero_division=1, labels=labels)
        t_f1 = f1_score(train_gt, train_preds, average=None, zero_division=1, labels=labels)

        v_conf_mat = confusion_matrix(val_gt, val_preds, normalize="true", labels=labels)
        v_acc = v_conf_mat.diagonal() / v_conf_mat.sum(axis=1)
        v_prec = precision_score(val_gt, val_preds, average=None, zero_division=1, labels=labels)
        v_rec = recall_score(val_gt, val_preds, average=None, zero_division=1, labels=labels)
        v_f1 = f1_score(val_gt, val_preds, average=None, zero_division=1, labels=labels)

        # print epoch evaluation results
        print(
            "EPOCH: {}/{}".format(e + 1, config["epochs"]),
            "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
            "Train Acc (M): {:>4.2f} (%)".format(np.nanmean(t_acc) * 100),
            "Train Prc (M): {:>4.2f} (%)".format(np.nanmean(t_prec) * 100),
            "Train Rcl (M): {:>4.2f} (%)".format(np.nanmean(t_rec) * 100),
            "Train F1 (M): {:>4.2f} (%)".format(np.nanmean(t_f1) * 100),
            "\nValid Loss: {:.4f}".format(np.mean(val_losses)),
            "Valid Acc (M): {:>4.2f} (%)".format(np.nanmean(v_acc) * 100),
            "Valid Prc (M): {:>4.2f} (%)".format(np.nanmean(v_prec) * 100),
            "Valid Rcl (M): {:>4.2f} (%)".format(np.nanmean(v_rec) * 100),
            "Valid F1 (M): {:>4.2f} (%)".format(np.nanmean(v_f1) * 100),
        )

        # W&B logging (current behavior: per-epoch)
        if run is not None:
            wandb.log(
                {
                    f"{name}/train_loss": float(np.mean(train_losses)),
                    f"{name}/val_loss": float(np.mean(val_losses)),
                    f"{name}/val/acc_macro": float(np.nanmean(v_acc)),
                    f"{name}/val/prec_macro": float(np.nanmean(v_prec)),
                    f"{name}/val/rec_macro": float(np.nanmean(v_rec)),
                    f"{name}/val/f1_macro": float(np.nanmean(v_f1)),
                    f"{name}/lr": float(opt.param_groups[0]["lr"]),
                })

        # adjust learning rate if enabled
        if config["adj_lr"]:
            if config["lr_scheduler"] == "reduce_lr_on_plateau":
                lr_scheduler.step(np.mean(val_losses))
            else:
                lr_scheduler.step()

        # employ early stopping if employed (based on macro F1)
        metric = f1_score(val_gt, val_preds, average="macro", labels=labels)

        if best_metric >= metric:
            if config["early_stopping"]:
                es_pt_counter += 1
                if es_pt_counter >= config["es_patience"]:
                    print(
                        "Stopping training early since no loss improvement over {} epochs.".format(
                            str(es_pt_counter)
                        )
                    )
                    early_stop = True
        else:
            print(f"Performance improved... ({best_metric}->{metric})")
            if config["early_stopping"]:
                es_pt_counter = 0

            best_metric = float(metric)
            best_epoch = e + 1  # epochs are 1-indexed for readability
            best_checkpoint = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()},
                "optim_state_dict": opt.state_dict(),
                "criterion_state_dict": criterion.state_dict() if hasattr(criterion, "state_dict") else {},
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            best_train_preds = train_preds
            best_val_preds = val_preds

        network.train()

        if early_stop:
            break

    # return validation, train and test predictions as numpy array with ground truth
    if config["valid_epoch"] == "best":
        if best_checkpoint is None:
            # fallback: behave like "last"
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "optim_state_dict": opt.state_dict(),
                "criterion_state_dict": criterion.state_dict() if hasattr(criterion, "state_dict") else {},
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T, best_epoch

        # restore best weights into the existing network object
        network.load_state_dict(best_checkpoint["model_state_dict"])
        network.to(device)

        return (
            network,
            best_checkpoint,
            np.vstack((best_val_preds, val_gt)).T,
            np.vstack((best_train_preds, train_gt)).T,
            best_epoch,
        )
    else:
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optim_state_dict": opt.state_dict(),
            "criterion_state_dict": criterion.state_dict() if hasattr(criterion, "state_dict") else {},
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T, best_epoch
