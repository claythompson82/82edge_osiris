"""
Dummy “Hermes” scorer for unit-tests.

Rule: look for the **first digit** in any text we can coax from the tokenizer
and return digit / 10.0.
"""

from __future__ import annotations

import importlib
import re
import sys
from typing import Any, Tuple

# --------------------------------------------------------------------------- #
# Patch-friendly loader indirection
# --------------------------------------------------------------------------- #


def _patched_loader():
    return sys.modules.get("osiris.llm_sidecar.loader") or importlib.import_module(
        "llm_sidecar.loader"
    )


def get_hermes_model_and_tokenizer() -> Tuple[Any, Any]:
    return _patched_loader().get_hermes_model_and_tokenizer()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_digit_re = re.compile(r"([0-9])")


def _find_digit(text: str) -> int | None:
    m = _digit_re.search(str(text))  # ensure even MagicMock → str
    return int(m.group(1)) if m else None


def score_with_hermes(proposal: dict[str, Any], context: str | None = None) -> float:
    _model, tok = get_hermes_model_and_tokenizer()
    candidate_texts: list[str] = [str(proposal), context or ""]

    try:
        candidate_texts.append(tok.decode(tok.encode("irrelevant")))
    except Exception:
        pass

    try:
        candidate_texts.append(tok.decode([]))  # type: ignore[arg-type]
    except Exception:
        pass

    candidate_texts.append(getattr(tok, "out", ""))

    for txt in candidate_texts:
        d = _find_digit(txt)
        if d is not None:
            return d / 10.0
    return 0.0
