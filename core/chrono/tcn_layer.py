import torch, torch.nn as nn

class TCNChronoLayer(nn.Module):
    def __init__(self, in_channels: int, history_size: int):
        super().__init__()
        layers, dilation = [], 1
        for _ in range(3):
            layers += [
                nn.Conv1d(in_channels, in_channels, 3, padding=dilation, dilation=dilation),
                nn.ReLU(),
            ]
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.project = nn.Linear(in_channels, 1)

    def step(self, history: torch.Tensor) -> torch.Tensor:
        x = history.T.unsqueeze(0)       # [1, features, history]
        y = self.net(x).squeeze(0)       # [features, history]
        last = y[:, -1]                  # last time slice
        return self.project(last)
