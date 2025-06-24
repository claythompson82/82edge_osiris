# core/chrono/tcn_layer.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .tcn_utils import Chomp1d
from .api import StepOutput

class TCNBlock(nn.Module):
    """
    A single TCN block using the pad-and-trim pattern for causal convs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNChronoLayer(nn.Module):
    """
    Micro-cognition layer: stacks multiple TCNBlocks and
    produces a single‐step prediction.
    """
    def __init__(
        self,
        input_dim: int,
        history_size: int,
        channels: list[int] = [32, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.history_size = history_size
        layers = []
        for i, ch in enumerate(channels):
            in_ch = input_dim if i == 0 else channels[i-1]
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.final = nn.Linear(channels[-1], 1)

    def step(self, history: torch.Tensor) -> StepOutput:
        """
        history: (history_size, input_dim)
        returns: scalar prediction and no embedding
        """
        # to (batch=1, channels=input_dim, seq_len=history_size)
        x = history.unsqueeze(0).permute(0, 2, 1)
        out = self.network(x)                      # (1, channels[-1], seq_len)
        last = out[:, :, -1]                       # (1, channels[-1])
        pred = self.final(last).squeeze(0).squeeze(-1)  # scalar tensor
        return StepOutput(prediction=pred, embedding=None)
