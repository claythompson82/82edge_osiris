"""Legacy import shim so tests can still do `from osiris.llm_client import LLMClient`."""
from llm_sidecar.client import LLMClient  # adjust path if your grep showed a different file
__all__ = ["LLMClient"]
