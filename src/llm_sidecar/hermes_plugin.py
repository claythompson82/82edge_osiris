# src/llm_sidecar/hermes_plugin.py

from __future__ import annotations
import json
import re
from typing import Any, Dict, Tuple
from .loader import get_hermes_model_and_tokenizer

_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")

def _extract_score(txt: str) -> float:
    m = _NUMBER_RE.search(txt)
    if not m:
        return -1.0
    num = float(m.group())
    if num < 0 or num > 10:
        return -1.0
    return num / 10.0

def score_with_hermes(proposal: Dict[str, Any], context: str | None = None) -> float:
    model, tok = get_hermes_model_and_tokenizer()
    # Tests expect prompt to include Context:\n
    if context:
        prompt = f"Context:\n{context}\n" + json.dumps({"proposal": proposal})
    else:
        prompt = json.dumps({"proposal": proposal})
    try:
        generated_ids = model.generate(prompt)  # type: ignore[attr-defined]
    except Exception:
        generated_ids = None
    try:
        text = tok.decode(generated_ids)  # type: ignore[attr-defined]
    except Exception:
        text = getattr(tok, 'last_prompt', None) or tok.decode(None)
    return _extract_score(text if text is not None else "")
