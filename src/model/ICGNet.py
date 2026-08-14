##################################################
# ICGNet: Inception Inspired CNN-GRU Hybrid for HAR
# Based on: Dua et al. (2022) "Inception inspired CNN-GRU hybrid network
# for human activity recognition", Multimedia Tools and Applications 82:5369-5403
##################################################

import torch
import torch.nn as nn


class _InceptionBranch(nn.Module):
    """Single branch of the modified Inception CNN block."""

    def __init__(self, in_channels, out_channels, kernel_size, drop_prob):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding for odd kernel sizes
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        # 1x1 conv to reduce channels (channel pooling, as in paper)
        self.conv2 = nn.Conv1d(out_channels, out_channels // 2, kernel_size=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.dropout(x)
        x = self.relu2(self.conv2(x))
        return x


class ICGNet(nn.Module):
    """
    Inception-inspired CNN-GRU hybrid network for Human Activity Recognition.

    Architecture (Dua et al. 2022, Fig. 6b + Fig. 8):
      - 4 parallel 1D conv branches with kernel sizes (1, 3, 5, 11)
      - Per-branch: Conv1d -> ReLU -> Dropout -> Conv1d(1x1) -> ReLU
      - Concatenate branches -> MaxPool1d(2) -> 2-layer GRU -> Dense -> BN -> classifier

    Input:  (B, T, C)  — batch, time steps, sensor channels
    Output: (B, num_classes)  — logits
    """

    def __init__(
        self,
        channels,
        classes,
        window_size,
        drop_prob=0.3,
        filter_sizes=(1, 3, 5, 11),
        branch_filters=(32, 64, 64, 64),
        gru_units=(32, 16),
        dense_units=64,
    ):
        super().__init__()

        # 4 inception branches; output channels per branch = branch_filters[i] // 2
        self.branches = nn.ModuleList([
            _InceptionBranch(channels, branch_filters[i], filter_sizes[i], drop_prob)
            for i in range(len(filter_sizes))
        ])

        concat_channels = sum(f // 2 for f in branch_filters)  # 16+32+32+32 = 112

        self.pool = nn.MaxPool1d(kernel_size=2)

        # RNN block: two GRU layers
        self.gru1 = nn.GRU(concat_channels, gru_units[0], batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.gru2 = nn.GRU(gru_units[0], gru_units[1], batch_first=True)

        # Classifier
        self.fc = nn.Linear(gru_units[1], dense_units)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(dense_units)
        self.classifier = nn.Linear(dense_units, classes)

    def forward(self, x):
        # x: (B, T, C) -> permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)

        # Inception branches
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)   # (B, concat_channels, T)

        x = self.pool(x)                     # (B, concat_channels, T//2)
        x = x.permute(0, 2, 1)              # (B, T//2, concat_channels)

        # RNN block
        x, _ = self.gru1(x)                  # (B, T//2, gru_units[0])
        x = self.dropout(x)
        _, h = self.gru2(x)                  # h: (1, B, gru_units[1])
        x = h.squeeze(0)                     # (B, gru_units[1])

        # Classifier
        x = self.relu(self.fc(x))            # (B, dense_units)
        x = self.bn(x)
        x = self.classifier(x)               # (B, classes)
        return x
