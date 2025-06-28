"""
src/osiris/server/__init__.py
─────────────────────────────
Minimal FastAPI façade used by the Osiris test-suite.

✦  “Public” AZR v1 endpoint lives at  /azr_api/v1/propose_trade
✦  “Internal” test-only endpoint (enabled when OSIRIS_TEST=1) at
   /azr_api/internal/azr/planner/propose_trade
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from llm_sidecar import db as _db
from llm_sidecar import hermes_plugin as _hermes_plugin
from llm_sidecar import loader as _loader
from llm_sidecar.loader import LoadedAdapterComponents

# --------------------------------------------------------------------------- #
# Loader wrappers (kept patch-safe)
# --------------------------------------------------------------------------- #
def get_phi3_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:  # pragma: no cover
    return _loader.get_phi3_model_and_tokenizer()


def get_hermes_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:  # pragma: no cover
    return _loader.get_hermes_model_and_tokenizer()


# --------------------------------------------------------------------------- #
# Patchable stubs – overridden in the test-suite
# --------------------------------------------------------------------------- #
async def _generate_hermes_text(prompt: str, max_length: int) -> str:  # pragma: no cover
    return prompt[::-1][: max_length]


async def _generate_phi3_json(prompt: str, max_length: int) -> Dict[str, Any]:  # pragma: no cover
    return {"output": prompt.upper()[: max_length]}


def text_to_speech(text: str) -> bytes:  # pragma: no cover
    return b""


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "AZR Planner", "description": "Public Alpha-Zero-Risk planner API (v1)"},
    {
        "name": "AZR Planner Internal",
        "description": "Internal/testing AZR planner route (enabled when OSIRIS_TEST=1)",
    },
]

app = fastapi.FastAPI(
    title="Osiris-stub",
    version="test",
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TS = time.time()

# --------------------------------------------------------------------------- #
#  Request payload schemas
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    model_id: str | None = Field(default="hermes", examples=["hermes", "phi3"])
    max_length: int | None = Field(default=256, ge=1, le=4096)


class ScoreHermesRequest(BaseModel):
    proposal: Dict[str, Any]
    context: str | None = None


class SpeakRequest(BaseModel):
    text: str


# --------------------------------------------------------------------------- #
#  Routes – generic LLM helpers
# --------------------------------------------------------------------------- #
@app.post("/generate/")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    if req.model_id == "phi3":
        return await _generate_phi3_json(req.prompt, req.max_length or 256)
    text_out = await _generate_hermes_text(req.prompt, req.max_length or 256)
    return {"output": text_out}


@app.post("/score/hermes/")
async def score_hermes(req: ScoreHermesRequest) -> Dict[str, Any]:
    score = _hermes_plugin.score_with_hermes(req.proposal, req.context)
    _db.log_hermes_score(score)
    return {"score": score}


@app.post("/speak/")
async def speak(req: SpeakRequest) -> Response:
    wav_bytes = text_to_speech(req.text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
def health(adapter_date: bool = False) -> JSONResponse:
    latest_adapter: str | None = None
    if adapter_date:
        latest_dir: Path | None = _loader.get_latest_adapter_dir()
        latest_adapter = latest_dir.name if latest_dir else None

    phi3_ok = bool(get_phi3_model_and_tokenizer())
    hermes_ok = bool(get_hermes_model_and_tokenizer())

    return JSONResponse(
        {
            "uptime": time.time() - _START_TS,
            "phi_ok": phi3_ok,
            "hermes_ok": hermes_ok,
            "mean_hermes_score_last_24h": _db.get_mean_hermes_score_last_24h(),
            "latest_adapter": latest_adapter,
        }
    )


# --------------------------------------------------------------------------- #
#  Helper used by tests/test_feedback_versioning.py
# --------------------------------------------------------------------------- #
async def submit_phi3_feedback(item: Any) -> Dict[str, str]:
    """Persist a feedback record in LanceDB (dummy in tests)."""
    schema_ver = getattr(item, "schema_version", "1.0")
    ts = _db._coerce_ts(getattr(item, "timestamp", None))

    content = getattr(item, "feedback_content", "{}")
    if not isinstance(content, str):
        content = json.dumps(content)

    row = _db.Phi3FeedbackSchema(
        transaction_id=item.transaction_id,
        feedback_type=item.feedback_type,
        feedback_content=content,
        schema_version=schema_ver,
        ts=ts,
    ).model_dump()

    _db.append_feedback(row)
    return {"status": "OK", "stored_schema_version": schema_ver}


# --------------------------------------------------------------------------- #
#  AZR Planner – PUBLIC v1
# --------------------------------------------------------------------------- #
from azr_planner.schemas import PlanningContext as AZRPlanningContext, TradeProposal as AZRTradeProposal
from azr_planner.engine import generate_plan as _azr_generate_plan

router_azr_v1 = fastapi.APIRouter(
    prefix="/azr_api/v1",
    tags=["AZR Planner"],
)


@router_azr_v1.post(
    "/propose_trade",
    response_model=AZRTradeProposal,
    summary="Propose a trade based on market context",
)
async def propose_trade_v1(ctx: AZRPlanningContext) -> AZRTradeProposal:
    return _azr_generate_plan(ctx)


app.include_router(router_azr_v1)

# --------------------------------------------------------------------------- #
#  AZR Planner – INTERNAL (enabled only when OSIRIS_TEST=1)
# --------------------------------------------------------------------------- #
if os.getenv("OSIRIS_TEST") == "1":
    router_azr_internal = fastapi.APIRouter(
        prefix="/azr_api/internal/azr/planner",
        tags=["AZR Planner Internal"],
    )

    @router_azr_internal.post(
        "/propose_trade",
        response_model=AZRTradeProposal,
        summary="[TEST] Propose trade (internal)",
    )
    async def propose_trade_internal(ctx: AZRPlanningContext) -> AZRTradeProposal:  # pragma: no cover
        return _azr_generate_plan(ctx)

    app.include_router(router_azr_internal)
