##################################################
# InceptionContext: Inception-branch Stage 1 + DeepConvContext Stage 2
##################################################

import torch
import torch.nn as nn


class ChannelAffine(nn.Module):
    """
    Learnable per-channel affine scale and shift: gamma * x + beta.

    Equivalent to the affine portion of LayerNorm with normalization disabled.
    Inputs are expected to already be z-scored upstream, so only the learnable
    rebalancing (not normalization) is applied here.

    Args:
        channels: int
            Number of input channels (C). Adds 2*C learnable parameters.
    """

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))   # (C,)
        self.beta  = nn.Parameter(torch.zeros(channels))  # (C,)

    def forward(self, x):
        # x: (B, T, C)
        return self.gamma * x + self.beta                  # broadcasts over B, T


class _InceptionBranch2d(nn.Module):
    """
    Single Conv2d inception branch.

    Uses Conv2d(kernel_size, 1) instead of Conv1d so the convolution sees
    cross-channel relationships (channels are preserved as the spatial W dim).
    Temporal 'same' padding keeps T unchanged for any dilation value:
    padding = ((kernel_size - 1) * dilation) // 2. For dilation=1 this
    reduces to kernel_size // 2, preserving bit-identical outputs.
    The 1x1 bottleneck conv is undilated.
    """

    def __init__(self, in_channels, out_channels, kernel_size, drop_prob, dilation=1):
        super().__init__()
        padding = (((kernel_size - 1) * dilation) // 2, 0)  # same-padding along T for any dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=padding, dilation=(dilation, 1))
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, (1, 1))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.dropout(x)
        x = self.relu2(self.conv2(x))
        return x


class InceptionContext(nn.Module):
    """
    Hybrid model combining ICGNet-style inception feature extraction (Stage 1)
    with DeepConvContext's cross-window BiLSTM context (Stage 2).

    Stage 1 — within-window feature extraction:
      (B, T, C) -> inception branches (Conv2d) -> GRU -> attention pooling -> (B, nb_units_gru_ic)

    Stage 2 — cross-window context:
      (B, nb_units_gru_ic) -> context_lstm (BiLSTM) -> classifier -> (B, classes)

    Args:
        batch_size: int
            Size of the input batch (used as max_len context for Stage 2).
        channels: int
            Number of sensor channels.
        classes: int
            Number of output classes.
        window_size: int
            Number of time steps per window.
        lstm_units: int
            Hidden units for the context BiLSTM (Stage 2).
        lstm_layers: int
            Number of layers in the context LSTM.
        dropout: float
            Dropout probability applied in branches and before classifier.
        bidirectional: bool
            Whether context_lstm is bidirectional.
        filter_sizes: tuple of int
            Temporal kernel sizes for each inception branch.
        branch_filters: tuple of int
            Number of filters per branch before the 1×1 halving conv.
        nb_units_gru_ic: int
            Hidden units for the within-window GRU (Stage 1).
        use_channel_affine: bool
            When True, prepend a ChannelAffine layer that applies a learnable
            per-channel gamma/beta to the (B, T, C) input before any other
            processing. Default False (preserves existing behavior exactly).
        branch_dilations: tuple of int
            Dilation factor for the temporal conv in each inception branch.
            Must have the same length as filter_sizes. Default (1, 1, 1, 1)
            preserves bit-identical outputs to the undilated baseline.
        dense: bool
            When True, replace the attention pooling + Stage 2 with a per-timestep
            head: a bidirectional LSTM over the WITHIN-sequence time axis followed by
            a linear classifier, returning (B, classes, T) logits for
            nn.CrossEntropyLoss. Stage 2's context_lstm cannot serve this purpose --
            it is batch_first=False and reads the minibatch as its sequence axis, so
            it only ever sees across windows, never within one. Default False leaves
            the windowed forward path bit-identical: the dense branch returns before
            any existing line runs, and the dense layers are not even constructed.
        no_bilstm: bool
            Ablation control for the dense head; only meaningful when dense=True.
            When True, the dense BiLSTM is skipped (and not constructed) and the GRU
            output is routed straight through a single Linear(nb_units_gru_ic, classes).
            This removes the head's temporal smoothing capacity while leaving the dense
            labels, sequences and training loop identical, which isolates how much of the
            rebound F1 gain came from dense labeling rather than from the BiLSTM.
            Default False leaves the dense path bit-identical.
    """

    def __init__(
        self,
        batch_size,
        channels,
        classes,
        window_size,
        lstm_units=128,
        lstm_layers=1,
        dropout=0.5,
        bidirectional=False,
        filter_sizes=(1, 3, 5, 11),
        branch_filters=(32, 64, 64, 64),
        nb_units_gru_ic=128,
        use_channel_affine=False,
        branch_dilations=(1, 1, 1, 1),
        dense=False,
        no_bilstm=False,
    ):
        super().__init__()

        if len(branch_dilations) != len(filter_sizes):
            raise ValueError(
                f"branch_dilations length ({len(branch_dilations)}) must match "
                f"filter_sizes length ({len(filter_sizes)})"
            )

        # --- Optional per-channel affine (prepended to Stage 1) ---
        self.use_channel_affine = use_channel_affine
        if use_channel_affine:
            self.channel_affine = ChannelAffine(channels)

        self.lstm_units = lstm_units
        self.bidirectional = bidirectional

        # --- Stage 1: inception branches ---
        self.branches = nn.ModuleList([
            _InceptionBranch2d(1, branch_filters[i], filter_sizes[i], dropout, branch_dilations[i])
            for i in range(len(filter_sizes))
        ])

        # Auto-configure GRU input dim via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size, channels)
            branch_outs = [branch(dummy) for branch in self.branches]
            dummy_cat = torch.cat(branch_outs, dim=1)   # (1, sum_filters, T, C)
            gru_input_dim = dummy_cat.shape[1] * dummy_cat.shape[3]  # sum_filters * C

        self.gru = nn.GRU(gru_input_dim, nb_units_gru_ic, batch_first=True)
        self.gru_attention = nn.Linear(nb_units_gru_ic, 1)
        self.dropout = nn.Dropout(dropout)

        # --- Stage 2: cross-window context ---
        self.context_lstm = nn.LSTM(
            nb_units_gru_ic,
            lstm_units,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.classifier = nn.Linear(2 * lstm_units, classes)
        else:
            self.classifier = nn.Linear(lstm_units, classes)

        # --- Dense head: per-timestep classification (replaces Stage 1 pooling + Stage 2) ---
        # Always bidirectional: a per-sample label benefits from context on both sides, and
        # the whole point of the dense path is that a rebound's lead-in and follow-through
        # are both visible. batch_first=True because here the sequence axis really is dim 1.
        self.dense = dense
        self.no_bilstm = no_bilstm
        if dense and no_bilstm:
            # Ablation: no BiLSTM, so the head's input is the GRU's own hidden size
            # (nb_units_gru_ic = 128 under the pre-registered config) rather than
            # 2 * lstm_units. The GRU is still recurrent, but the head adds no further
            # temporal smoothing on top of it.
            self.dense_head = nn.Linear(nb_units_gru_ic, classes)
        elif dense:
            self.dense_lstm = nn.LSTM(
                nb_units_gru_ic,
                lstm_units,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.dense_head = nn.Linear(2 * lstm_units, classes)

    def forward(self, x):
        # x: (B, T, C)
        if self.use_channel_affine:
            x = self.channel_affine(x)                       # (B, T, C) — per-channel scale/shift
        x = x.unsqueeze(1)                                   # (B, 1, T, C)

        # Stage 1 — inception branches
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)                    # (B, sum_filters, T, C)

        B, F, T, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, F * C)      # (B, T, sum_filters*C)
        x = self.dropout(x)
        x, _ = self.gru(x)                                   # (B, T, nb_units_gru_ic)

        if self.dense:
            # Dense head -- returns before the attention pooling below, which is the module
            # that collapses T in the windowed path. Nothing after this point runs.
            if not self.no_bilstm:
                x, _ = self.dense_lstm(x)                    # (B, T, 2 * lstm_units)
            x = self.dense_head(x)                           # (B, T, classes)
            return x.permute(0, 2, 1)                        # (B, classes, T) for CrossEntropyLoss

        attn_weights = torch.softmax(self.gru_attention(x), dim=1)  # (B, T, 1)
        x = torch.sum(attn_weights * x, dim=1)               # (B, nb_units_gru_ic)

        # Stage 2 — cross-window context
        x = x.unsqueeze(1)                                   # (B, 1, nb_units_gru_ic)
        x, _ = self.context_lstm(x)                          # (B, 1, lstm_units * directions)
        if self.bidirectional:
            x = x.view(-1, 2 * self.lstm_units)
        else:
            x = x.view(-1, self.lstm_units)

        x = self.dropout(x)
        return self.classifier(x)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
