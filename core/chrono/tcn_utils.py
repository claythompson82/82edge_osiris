# core/chrono/tcn_utils.py

import torch.nn as nn

class Chomp1d(nn.Module):
    """
    Removes padding from the end of a 1D tensor.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (batch, channels, seq_len + chomp_size)
        return x[:, :, :-self.chomp_size].contiguous()
