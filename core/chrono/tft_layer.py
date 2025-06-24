# core/chrono/tft_layer.py

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

from .base import ChronoLayer

class TFTChronoLayer(ChronoLayer):
    """
    Meso-cognition layer implemented with a real Temporal Fusion Transformer.
    Updated to use only the supported constructor arguments for the current
    PyTorch Forecasting API.
    """
    def __init__(
        self,
        history_size: int,
        micro_embedding_dim: int,
        # Default TFT model hyperparameters
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
    ):
        super().__init__()

        print("Meso-Layer (Real TFT) Initializing…")
        self.history_size = history_size
        self.micro_embedding_dim = micro_embedding_dim
        self.history_buffer = pd.DataFrame()

        # Instantiate the model with only supported args
        self.model = TemporalFusionTransformer(
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=1,
            loss=MAE(),
        )
        print("Meso-Layer (Real TFT) Initialized.")

    def _add_to_history(self, numpy_embedding: np.ndarray):
        """Appends a new observation to the internal history buffer."""
        new_row_data = {f"emb_{i}": float(val) for i, val in enumerate(numpy_embedding)}
        new_row_data["target"] = 0.0
        new_row_data["group_id"] = 0
        new_row_data["time_idx"] = len(self.history_buffer)

        new_row_df = pd.DataFrame([new_row_data])
        self.history_buffer = pd.concat([self.history_buffer, new_row_df], ignore_index=True)

        if len(self.history_buffer) > self.history_size:
            self.history_buffer = self.history_buffer.iloc[-self.history_size :]

    def _prepare_prediction_data(self) -> Dict[str, Any]:
        """
        Formats the history buffer into the dict expected by the TFT model.
        Note: this may need to be updated if you switch to `from_dataset` or
        a newer API for inference.
        """
        df = self.history_buffer.copy()
        # Prepare continuous predictors (embeddings + time_idx)
        x_cont = torch.tensor(
            df[[f"emb_{i}" for i in range(self.micro_embedding_dim)] + ["time_idx"]].values,
            dtype=torch.float32,
        ).unsqueeze(0)  # shape [1, seq_len, features]
        # Encode lengths for encoder/decoder (we're doing full-history forecasting)
        encoder_lengths = torch.tensor([self.history_size])
        decoder_lengths = torch.tensor([self.history_size])
        # True values for loss (not used at inference)
        y = (torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(0), None)

        return {
            "x_cont": x_cont,
            "x_cat": torch.empty(1, self.history_size, 0, dtype=torch.long),
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
            "y": y,
        }

    def step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a new embedding and returns a forecast once the buffer
        is full. Expects x = {"embedding": np.ndarray}.
        """
        numpy_embedding = x.get("embedding")
        if not isinstance(numpy_embedding, np.ndarray):
            return {}

        self._add_to_history(numpy_embedding)
        if len(self.history_buffer) < self.history_size:
            return {"status": "buffering", "buffered": len(self.history_buffer)}

        data = self._prepare_prediction_data()
        with torch.no_grad():
            out = self.model(data)
        # extract the first (and only) forecast
        forecast = out["prediction"][0][0].item()
        return {"meso_forecast": float(forecast)}
