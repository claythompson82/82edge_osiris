import torch
import torch.nn as nn
from typing import List, Dict, Any

from .base import ChronoLayer

class TiDEChronoLayer(ChronoLayer):
    """
    A ChronoLayer implementation using a TiDE-like MLP architecture.
    This version returns both the final prediction and the internal embedding
    from its last hidden layer for consumption by other layers.
    """
    def __init__(self,
                 input_dim: int,
                 history_size: int,
                 output_dim: int = 1,
                 hidden_dims: List[int] = [64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.history_size = history_size
        self.final_hidden_dim = hidden_dims[-1] if hidden_dims else (history_size * input_dim)

        flat_dim = input_dim * history_size
        
        layers: List[nn.Module] = []
        prev_dim = flat_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        
        self.encoder = nn.Sequential(*layers)
        self.readout = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of histories.
        """
        batch_size = x.size(0)
        flat_input = x.view(batch_size, -1)
        embedding = self.encoder(flat_input)
        prediction = self.readout(embedding)
        
        return {
            "prediction": prediction,
            "embedding": embedding
        }

    def step(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes a history buffer for a single-step prediction.
        Wraps the forward call in torch.no_grad() to prevent tracking
        gradients during inference.
        """
        with torch.no_grad():
            history_batch = history.unsqueeze(0)
            output_dict = self.forward(history_batch)
            return {key: val.squeeze(0) for key, val in output_dict.items()}
