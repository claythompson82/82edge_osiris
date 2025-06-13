"""
Light-weight FastAPI façade used only by the test-suite.
Real production logic can hang off these stubs later.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm_sidecar import db, loader
from llm_sidecar.db import Phi3FeedbackSchema as FeedbackItem  # re-export for tests
app = FastAPI(title="Osiris Test Stub")

# --------------------------------------------------------------------------- #
# 1. Tiny async event-bus shim (tests monkey-patch its methods).
# --------------------------------------------------------------------------- #
class _DummyBus:  # pragma: no cover – real impl lives elsewhere
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def subscribe(self, *_a, **_kw) -> None: ...

event_bus = _DummyBus()  # exported symbol the tests expect

# --------------------------------------------------------------------------- #
# 2. Minimal model helpers that the tests import / patch.
# --------------------------------------------------------------------------- #
def get_hermes_model_and_tokenizer():
    from transformers import AutoModel, AutoTokenizer  # heavy import patched in tests
    return AutoModel, AutoTokenizer


def get_phi3_model_and_tokenizer():
    from transformers import AutoModel, AutoTokenizer
    return AutoModel, AutoTokenizer


async def _generate_hermes_text(*_a, **_kw):  # patched
    return "stub-hermes"


async def _generate_phi3_json(*_a, **_kw):  # patched
    return {"stub": "phi3"}


# --------------------------------------------------------------------------- #
# 3. FeedbackItem – matches the expectations in tests/test_feedback_versioning.py
#    * default schema_version = "1.0"
#    * default nano-timestamp "when"
# --------------------------------------------------------------------------- #
def _now_ns() -> int:  # helper for default_factory
    return int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1e9)


class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    timestamp: str
    corrected_proposal: Union[Dict[str, Any], None] = None
    schema_version: str = "1.0"
    when: int = Field(default_factory=_now_ns)


# FastAPI route used only by one test
@app.post("/feedback/phi3")
async def submit_phi3_feedback(item: FeedbackItem):  # noqa: D401
    """Store feedback in LanceDB; returns the row stored."""
    payload = item.model_dump()
    db.append_feedback(payload)  # patched in tests
    return payload


# --------------------------------------------------------------------------- #
# 4. Health probe – must optionally expose latest adapter directory.
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health(adapter_date: bool = False):
    body: Dict[str, Any] = {
        "uptime": int((_dt.datetime.utcnow() - _dt.datetime.utcfromtimestamp(0)).total_seconds()),
        "mean_hermes_score_last_24h": db.get_mean_hermes_score_last_24h(),  # patched
    }

    if adapter_date:
        latest = loader.get_latest_adapter_dir()
        body["latest_adapter"] = latest.name if latest else None
    return body


# --------------------------------------------------------------------------- #
# 5. Text, scoring & speech stub-endpoints (tests patch internals heavily).
# --------------------------------------------------------------------------- #
@app.post("/generate/")
async def generate(prompt: str, max_length: int = 256, model_id: str | None = None):
    if model_id in (None, "hermes"):
        return {"generated_text": await _generate_hermes_text(prompt, max_length=max_length)}
    if model_id == "phi3":
        return await _generate_phi3_json(prompt, max_length=max_length)
    return {"error": f"Model {model_id!r} not supported"}, 400


@app.post("/score/hermes/")
async def score_hermes(proposal: Dict[str, Any], context: str | None = None):
    score = db.log_hermes_score(proposal_id := proposal.get("ticker", "N/A"), score=-1.0)  # patched
    return {"proposal_id": proposal_id, "score": score}


@app.post("/speak")
async def tts(text: str):
    audio = b"\x00\x00"  # gets patched
    return {"bytes": len(audio)}


# Public re-exports so the tests can `from osiris.server import app, db, FeedbackItem`
__all__ = ["app", "db", "FeedbackItem", "submit_phi3_feedback"]
