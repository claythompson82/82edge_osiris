"""llm_sidecar.loader

Helpers for model loading and adapter directory handling.
"""

from __future__ import annotations
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Allow external override (tests will patch this)
ADAPTER_ROOT = Path(os.getenv("ADAPTER_ROOT", "/adapters"))  # Or your real default path

def _is_dated_dir(p: Path) -> bool:
    """Check if path is a directory and name looks like a date (YYYY-MM-DD or YYYYMMDD)."""
    try:
        # Accept YYYY-MM-DD or YYYYMMDD
        return p.is_dir() and (
            (len(p.name) == 10 and p.name[:4].isdigit() and p.name[5:7].isdigit() and p.name[8:10].isdigit())
            or (len(p.name) == 8 and p.name.isdigit())
        )
    except Exception:
        return False

def get_latest_adapter_dir(base_dir: str | Path = None) -> Path | None:
    """
    Returns the Path of the latest dated adapter directory.
    Returns None if none found (caller must handle).
    """
    root = Path(base_dir) if base_dir else ADAPTER_ROOT
    if not root.exists() or not root.is_dir():
        return None

    dated = [p for p in root.iterdir() if _is_dated_dir(p)]
    if not dated:
        return None

    # Sort by name as YYYYMMDD or YYYY-MM-DD sorts correctly lexicographically
    latest = max(dated, key=lambda p: p.name)
    return latest

# ===== Dummy model loader functions for patching in tests =====

# Patch targets (tests expect these to exist for mocking)
def load_hermes_model():
    raise NotImplementedError("Should be patched by tests.")

def load_phi3_model():
    raise NotImplementedError("Should be patched by tests.")

def get_hermes_model_and_tokenizer():
    raise NotImplementedError("Should be patched by tests.")

def get_phi3_model_and_tokenizer():
    raise NotImplementedError("Should be patched by tests.")

# If you have any additional public imports, add them below.
