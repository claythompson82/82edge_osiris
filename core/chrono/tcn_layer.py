import torch
import torch.nn as nn

class TCNChronoLayer(nn.Module):
    """
    Temporal Convolutional Network micro-layer.
    Takes a [history_size × n_features] tensor and predicts the next price.
    """

    def __init__(self, in_channels: int, history_size: int):
        super().__init__()
        layers = []
        dilation = 1
        # three causal conv blocks with doubling dilations
        for _ in range(3):
            layers += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(),
            ]
            dilation *= 2
        self.net = nn.Sequential(*layers)
        # project from in_channels → 1 scalar
        self.project = nn.Linear(in_channels, 1)

    def step(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: tensor of shape [history_size, n_features]
        returns: scalar tensor prediction of next tick's price
        """
        # shape → [1, n_features, history_size]
        x = history.T.unsqueeze(0)
        y = self.net(x).squeeze(0)       # [n_features, history_size]
        last_feature_map = y[:, -1]      # last time-step, shape [n_features]
        return self.project(last_feature_map)
