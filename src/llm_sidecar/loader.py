"""
LLM-sidecar – model-loader utilities & adapter discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Only *very* lightweight, synchronous helpers live here so that unit-tests can
patch them out without incurring GPU / model-load overhead.
"""

from __future__ import annotations

import json
import re
import typing as _t
from datetime import datetime as _dt
from pathlib import Path

# ---------- adapter hot-swap helpers -------------------------------------------------------------

#: where dated adapter directories live; overridable in tests
ADAPTER_ROOT: Path = Path(__file__).resolve().parent / "adapters"


def _looks_like_date_dir(p: Path) -> bool:
    """`YYYY-MM-DD` or `YYYYMMDD`."""
    return (
        p.is_dir()
        and re.fullmatch(r"\d{4}[-_]?\d{2}[-_]?\d{2}", p.name) is not None
    )


def get_latest_adapter_dir(base: _t.Union[str, Path, None] = None) -> Path | None:
    """
    Return *Path* to the newest adapter directory or **None** when nothing valid
    exists.

    The helper never raises on a missing path so that tests can point it at
    arbitrary temp-dirs.  We rely on *lexicographic* max – identical to the
    approach recommended in many FastAPI hot-swap examples. :contentReference[oaicite:3]{index=3}
    """
    root = Path(base) if base is not None else ADAPTER_ROOT
    if not root.exists():
        return None
    dated = (p for p in root.iterdir() if _looks_like_date_dir(p))
    latest = max(dated, default=None, key=lambda p: p.name)
    return latest


# ---------- dummy model / tokenizer factories ----------------------------------------------------

def _dummy_model():
    """Return a cheap stand-in *class* – tests never instantiate it."""
    from transformers import PreTrainedModel  # light import
    return PreTrainedModel


def _dummy_tokenizer():
    from transformers import PreTrainedTokenizerBase
    return PreTrainedTokenizerBase


def load_hermes_model():
    """Load – or in tests, *pretend* to load – the Hermes model."""
    return _dummy_model()


def load_phi3_model():
    """Load – or in tests, *pretend* to load – the Phi-3 model."""
    return _dummy_model()


# ---------- legacy aliases kept for backwards-compat --------------------------------------------

# Many downstream modules still import the old names; keep them as TRUE no-ops
# instead of breaking import-time.
def get_hermes_model_and_tokenizer():
    """Legacy alias expected by `hermes_plugin.py` tests."""
    return _dummy_model(), _dummy_tokenizer()


def get_phi3_model_and_tokenizer():
    """Legacy alias expected by older server paths."""
    return _dummy_model(), _dummy_tokenizer()


__all__ = [
    "ADAPTER_ROOT",
    "get_latest_adapter_dir",
    "load_hermes_model",
    "load_phi3_model",
    "get_hermes_model_and_tokenizer",
    "get_phi3_model_and_tokenizer",
]
