# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM model as described in "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition" (https://doi.org/10.1145/3460421.3480419).
    Args:
    
    Args:
        channels: int
            Number of channels in the input data.
        classes: int
            Number of classes for classification.
        window_size: int
            Size of the input window.
        conv_kernels: int
            Number of convolutional kernels.
        conv_kernel_size: int
            Size of the convolutional kernels.
        lstm_units: int
            Number of LSTM units.
        lstm_layers: int
            Number of LSTM layers.
        dropout: float
            Dropout rate.
        dense: bool
            Dense (per-sample) prediction: classify every timestep instead of only the
            window's last one, so the forward pass returns (batch, classes, T) with T
            equal to the input sequence length. Two changes, both confined to the dense
            branch: the conv stack is same-padded so it stops shrinking the time axis,
            and the classifier is applied at every LSTM timestep. The conv kernels, the
            LSTM (unidirectional, same hidden size and layer count), the dropout and the
            classifier are the windowed path's own modules, so the dense model has exactly
            the same parameters -- same-padding adds none. Default False leaves the
            windowed forward path bit-identical: padding=0 is nn.Conv2d's default, so the
            modules are constructed as before, and the dense branch returns before any
            existing line runs.
    """
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, dense=False):
        super(DeepConvLSTM, self).__init__()

        # Same-padding in the dense branch ONLY. An even kernel cannot be centered on the
        # time axis, so it is a hard error rather than an off-by-one the assert in forward
        # would catch later.
        if dense and conv_kernel_size % 2 == 0:
            raise ValueError(
                f"dense=True requires an odd conv_kernel_size to center-pad the time axis "
                f"(got {conv_kernel_size}); an even kernel cannot preserve T."
            )
        pad = ((conv_kernel_size - 1) // 2, 0) if dense else 0

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1), padding=pad)
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=pad)
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=pad)
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=pad)
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size if dense else window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes
        self.dense = dense

    def forward(self, x):
        seq_len = x.shape[1]
        x = x.unsqueeze(1) # batch, 1, sequence, axes
        x = self.activation(self.conv1(x))# batch, kernels, sequence, axes
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        if self.dense:
            # Per-timestep classification. x is (T, batch, lstm_units) -- the LSTM is
            # batch_first=False here exactly as in the windowed path, so the dense branch
            # reuses the permute/reshape/LSTM lines above verbatim and only replaces the
            # x[-1] reduction below. The classifier and dropout are the same modules.
            if x.shape[0] != seq_len:
                raise ValueError(
                    f"dense forward: the conv stack returned {x.shape[0]} timesteps for an "
                    f"input of {seq_len}; same-padding must preserve the time axis."
                )
            x = self.dropout(x)
            x = self.classifier(x)                  # (T, batch, classes)
            return x.permute(1, 2, 0)               # (batch, classes, T) for CrossEntropyLoss
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
        