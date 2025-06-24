# core/chrono/factory.py

from typing import Dict
import torch.nn as nn

from .tcn_layer import TCNChronoLayer
from .tide_layer import TiDEChronoLayer
# future: from .tgnn_layer import TGNNChronoLayer

def create_chrono_layers(cfg: Dict) -> Dict[str, nn.Module]:
    """
    Factory to instantiate enabled ChronoLayers, keyed by name.
    """
    layers: Dict[str, nn.Module] = {}
    cs = cfg.get("chrono_stack", {})

    if cs.get("micro_enabled", True):
        model = cs.get("micro_model", "tcn").lower()
        if model == "tcn":
            layers["micro"] = TCNChronoLayer(
                input_dim=cs["input_dim"],
                history_size=cs["history_size"],
                # you can pass cs.get("tcn_channels"), etc.
            )
        elif model == "tide":
            layers["micro"] = TiDEChronoLayer(
                input_dim=cs["input_dim"],
                history_size=cs["history_size"],
                hidden_dims=cs.get("tide_hidden_dims", [64, 32]),
                output_dim=cs.get("tide_output_dim", 1),
            )
        else:
            raise ValueError(f"Unknown micro_model: {model}")

    if cs.get("meso_enabled", False):
        from .tft_layer import TFTChronoLayer
        layers["meso"] = TFTChronoLayer(
            history_size=cs["history_size"],
            micro_embedding_dim=cs.get("tide_hidden_dims", [])[ -1 ]
        )

    # if cs.get("macro_enabled", False):
    #     layers["macro"] = TGNNChronoLayer(...)

    if not layers:
        raise RuntimeError("No ChronoStack layers enabled in config.")

    return layers
