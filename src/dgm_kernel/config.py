from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ExternalLLMConfig:
    """Configuration for the external LLM service."""

    base_url: str
    api_key: str
    timeout: float

    @classmethod
    def from_env(cls) -> "ExternalLLMConfig":
        return cls(
            base_url=os.getenv("DGM_LLM_BASE_URL", "http://localhost:8080"),
            api_key=os.getenv("DGM_LLM_API_KEY", ""),
            timeout=float(os.getenv("DGM_LLM_TIMEOUT", "30")),
        )


CONFIG = ExternalLLMConfig.from_env()
