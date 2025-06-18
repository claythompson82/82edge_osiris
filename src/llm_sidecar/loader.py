"""
Stub loader: the real implementation would return actual models.

In the test-suite the two `get_*_model_and_tokenizer` functions are **always
monkey-patched**, so the bodies below never run.  They only exist so that
importing this module doesn’t crash.

`get_latest_adapter_dir()` is called by the /health endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple


# Default root – tests will monkey-patch this to a tmp directory.
ADAPTER_ROOT = Path(os.getenv("ADAPTER_ROOT", "/adapters"))


def get_latest_adapter_dir() -> Path | None:  # pragma: no cover
    """
    Return the most-recent adapter subdir – or **None** if none exist.

    We suppress permissions errors because the sandbox can’t create /adapters.
    """
    try:
        if not ADAPTER_ROOT.exists():
            return None
        subdirs = [p for p in ADAPTER_ROOT.iterdir() if p.is_dir()]
        return max(subdirs, default=None)
    except PermissionError:
        return None


# --- Model helpers – tests always patch these ---------------------------------


def get_phi3_model_and_tokenizer() -> Tuple[Any, Any]:  # pragma: no cover
    raise NotImplementedError("Should be patched by tests.")


def get_hermes_model_and_tokenizer() -> Tuple[Any, Any]:  # pragma: no cover
    raise NotImplementedError("Should be patched by tests.")
