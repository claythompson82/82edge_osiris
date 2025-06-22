import torch, torch.nn as nn

class TiDEChronoLayer(nn.Module):
    def __init__(self, in_channels: int, history_size: int):
        super().__init__()
        input_dim = in_channels * history_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 32),        nn.GELU(),
            nn.Linear(32, 1),
        )

    def step(self, history: torch.Tensor) -> torch.Tensor:
        flat = history.flatten()
        return self.net(flat)
