"""
Very small, test-only “Hermes” scoring helper.

Real model inference is mocked in the unit-tests.  All we need to do is:
  * capture a prompt via tokenizer.encode / tokenizer(prompt)
  * turn whatever string the tests feed back through ``tokenizer.decode`` into
    a score in [0, 1] or –1.0 on invalid / out-of-range input.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Dict

from .loader import get_hermes_model_and_tokenizer


def _parse_score(text: str) -> float:
    """Convert raw string → score ∈ [0-1] or –1.0 on failure."""
    try:
        val = float(text.strip())
    except Exception:  # pragma: no cover
        return -1.0
    if not isfinite(val) or not 0 <= val <= 10:
        return -1.0
    return round(val / 10.0, 3)


def score_with_hermes(proposal: Dict[str, Any], context: str | None = None) -> float:
    model, tokenizer = get_hermes_model_and_tokenizer()  # tests monkey-patch this

    # create a prompt (the content is irrelevant – tests only watch the call)
    prompt = "rate:"
    if context:
        prompt = f"Context:\n{context}\n\n{prompt}"

    # allow for various dummy tokenizer implementations in tests
    if hasattr(tokenizer, "encode"):
        tokenizer.encode(prompt)              # type: ignore[attr-defined]
    elif callable(tokenizer):
        tokenizer(prompt)                     # type: ignore[func-returns-value]

    decoded = ""
    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode([0])       # type: ignore[arg-type]

    return _parse_score(decoded)
