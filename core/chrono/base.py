# core/chrono/base.py
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any

class ChronoLayer(nn.Module, ABC):
    """
    Abstract base class for a stateful temporal cognition layer in Osiris.
    
    This class defines a common interface for all temporal models (micro, meso, macro),
    ensuring they can be composed and managed consistently within the agent's architecture.
    The key principle is stateful, step-wise processing appropriate for a real-time agent.
    """
    def __init__(self, **kwargs):
        """Initializes the layer."""
        super().__init__()
        # Each layer can manage its own internal state (e.g., hidden states for RNNs
        # or rolling input buffers for convolutional models).
        self.state: Any = None

    @abstractmethod
    def step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single new tick/observation and updates the internal state.
        This is the primary method called in the agent's real-time loop.

        Args:
            x (Dict[str, Any]): A dictionary of named input tensors or data. 
                                Keys might include 'micro_embedding', 'news_sentiment', etc.
        
        Returns:
            Dict[str, Any]: A dictionary of named output tensors or data, representing
                            the layer's updated state or prediction.
        """
        pass

    def get_state_for_persistence(self) -> Any:
        """
        Returns the current internal state of the layer for checkpointing.
        This is crucial for implementing warm-restarts.
        
        Returns:
            Any: A serializable representation of the layer's state.
        """
        return self.state

    def load_state_from_persistence(self, state: Any):
        """
        Loads the internal state from a checkpoint for warm-restarts.
        
        Args:
            state (Any): The state object to load.
        """
        self.state = state
