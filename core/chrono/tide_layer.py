# core/chrono/tide_layer.py

import torch
import torch.nn as nn

class TiDEChronoLayer(nn.Module):
    """
    TiDE micro-layer: simple MLP on flattened history.
    """

    def __init__(self, in_channels: int, history_size: int):
        super().__init__()
        input_dim = in_channels * history_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def step(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: tensor of shape [history_size, n_features]
        returns: scalar tensor prediction of next tick's price
        """
        flat = history.flatten()  # [history_size * n_features]
        return self.net(flat)
