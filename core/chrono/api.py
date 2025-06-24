# core/chrono/api.py

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class StepOutput:
    """
    Standardized output for any ChronoLayer.step() call.
    """
    prediction: torch.Tensor
    embedding: Optional[torch.Tensor] = None
