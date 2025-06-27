"""
Tiny FastAPI façade sufficient for the Osiris test-suite.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Union

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from llm_sidecar import db as _db
from llm_sidecar import hermes_plugin as _hermes_plugin
from llm_sidecar import loader as _loader
from llm_sidecar.db import _coerce_ts  # noqa: F401 (used below)

# --------------------------------------------------------------------------- #
#  Loader wrappers – always forward to *current* _loader attr (patch-safe)
# --------------------------------------------------------------------------- #


def get_phi3_model_and_tokenizer():
    return _loader.get_phi3_model_and_tokenizer()


def get_hermes_model_and_tokenizer():
    return _loader.get_hermes_model_and_tokenizer()


# --------------------------------------------------------------------------- #
#  Patchable stubs – heavy lifting mocked in tests
# --------------------------------------------------------------------------- #
async def _generate_hermes_text(prompt: str, max_length: int) -> str:
    return prompt[::-1][:max_length]


async def _generate_phi3_json(prompt: str, max_length: int) -> Dict[str, Any]:
    return {"output": prompt.upper()[:max_length]}


def text_to_speech(text: str) -> bytes:  # Patched in tests
    return b""


# --------------------------------------------------------------------------- #
#  FastAPI setup
# --------------------------------------------------------------------------- #
# Define tags for OpenAPI
openapi_tags_metadata = [
    {"name": "AZR Planner", "description": "Alpha-Zero-Risk planner"}
    # Add other tags here if needed
]

app = fastapi.FastAPI(
    title="Osiris-stub",
    version="test",
    openapi_tags=openapi_tags_metadata
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TS = time.time()

# --------------------------------------------------------------------------- #
#  Typed payloads
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    model_id: str | None = "hermes"
    max_length: int | None = 256


class ScoreHermesRequest(BaseModel):
    proposal: Dict[str, Any]
    context: str | None = None


class SpeakRequest(BaseModel):
    text: str


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.post("/generate/")
async def generate(req: GenerateRequest):
    if req.model_id == "phi3":
        return await _generate_phi3_json(req.prompt, req.max_length or 256)
    text_out = await _generate_hermes_text(req.prompt, req.max_length or 256)
    return {"output": text_out}


@app.post("/score/hermes/")
async def score_hermes(req: ScoreHermesRequest):
    score = _hermes_plugin.score_with_hermes(req.proposal, req.context)
    _db.log_hermes_score(score)
    return {"score": score}


@app.post("/speak/")
async def speak(req: SpeakRequest):
    wav_bytes = text_to_speech(req.text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
def health(adapter_date: bool = False):
    latest_adapter: str | None = None
    if adapter_date:
        latest_dir: Path | None = _loader.get_latest_adapter_dir()
        latest_adapter = latest_dir.name if latest_dir else None

    return JSONResponse(
        {
            "uptime": time.time() - _START_TS,
            "phi_ok": bool(get_phi3_model_and_tokenizer()[0]),  # PATCHABLE
            "hermes_ok": bool(get_hermes_model_and_tokenizer()[0]),  # PATCHABLE
            "mean_hermes_score_last_24h": _db.get_mean_hermes_score_last_24h(),
            "latest_adapter": latest_adapter,
        }
    )


# --------------------------------------------------------------------------- #
#  Helper used by tests/test_feedback_versioning.py
# --------------------------------------------------------------------------- #
async def submit_phi3_feedback(item) -> Dict[str, str]:
    """
    Persist a feedback record in LanceDB.
    """
    schema_ver: str = getattr(item, "schema_version", None) or "1.0"

    raw_ts = getattr(item, "timestamp", None)
    ts: Union[int, float] = _coerce_ts(raw_ts)

    content = getattr(item, "feedback_content", "{}")
    if not isinstance(content, str):
        content = json.dumps(content)

    row = _db.Phi3FeedbackSchema(  # type: ignore[attr-defined]
        transaction_id=item.transaction_id,
        feedback_type=item.feedback_type,
        feedback_content=content,
        schema_version=schema_ver,
        ts=ts,
    ).model_dump()

    _db.append_feedback(row)
    return {"status": "OK", "stored_schema_version": schema_ver}


# --------------------------------------------------------------------------- #
#  AZR Planner (internal, test only)
# --------------------------------------------------------------------------- #
import os
# Ensure imports are conditional or handled if azr_planner is optional
if os.getenv("OSIRIS_TEST") == "1": # More specific check for "1"
    # Import specific models needed
    from azr_planner.schemas import PlanningContext, TradeProposal # Changed TradePlan to TradeProposal
    from azr_planner.engine import generate_plan as azr_generate_plan # Alias to avoid name clash

    # router_azr name is fine as it's local to this conditional block
    router_azr = fastapi.APIRouter(
        prefix="/azr_api/internal/azr/planner", # Full prefix for the router
        tags=["AZR Planner"], # Tags defined on the router apply to all its routes
    )

    @router_azr.post(
        "/propose_trade", # Path is now relative to the router's prefix
        response_model=TradeProposal, # Changed to TradeProposal
        # Tags are inherited from APIRouter, but can be overridden or extended here
    )
    async def propose_trade(ctx: PlanningContext) -> TradeProposal: # Changed to TradeProposal
        """
        Thin HTTP wrapper around azr_planner.engine.generate_plan.
        """
        # The engine function is now imported as azr_generate_plan
        return azr_generate_plan(ctx)

    app.include_router(router_azr) # No prefix needed here as it's on APIRouter

# Ensure app.include_router is also conditional if router_azr itself is conditional
# The above structure handles this: router_azr and app.include_router are within the OSIRIS_TEST check.
