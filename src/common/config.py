"""
Global configuration for Osiris.
This module exports a mutable dict `cfg` that holds
all of our runtime settings, including the chrono_stack.
"""
from typing import Any, Dict

# Default configuration. Adjust values as needed.
cfg: Dict[str, Any] = {
    "chrono_stack": {
        # enable/disable micro layer
        "micro_enabled": True,
        # use TiDE baseline instead of TCN
        "use_tide_baseline": False,
        # dimensions for the TCN micro layer
        "tcn_input_dim": 3,
        "tcn_output_dim": 1,
        "tcn_sequence_length": 16,
        # will be populated at startup
        "layers": [],
    },
    # add other top-level config sections here
    # "llm": {...},
    # "redis": {...},
    # "logging": {...},
}
