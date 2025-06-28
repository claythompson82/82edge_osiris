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
from llm_sidecar.loader import LoadedAdapterComponents # Import the type
from typing import Optional # For Optional type hint

def get_phi3_model_and_tokenizer() -> Optional[LoadedAdapterComponents]: # Added return type
    return _loader.get_phi3_model_and_tokenizer()


def get_hermes_model_and_tokenizer() -> Optional[LoadedAdapterComponents]: # Added return type
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
async def generate(req: GenerateRequest) -> Dict[str, Any]: # Added return type
    if req.model_id == "phi3":
        return await _generate_phi3_json(req.prompt, req.max_length or 256)
    text_out = await _generate_hermes_text(req.prompt, req.max_length or 256)
    return {"output": text_out}


@app.post("/score/hermes/")
async def score_hermes(req: ScoreHermesRequest) -> Dict[str, Any]: # Added return type
    score = _hermes_plugin.score_with_hermes(req.proposal, req.context)
    _db.log_hermes_score(score)
    return {"score": score}


@app.post("/speak/")
async def speak(req: SpeakRequest) -> Response: # Added return type
    wav_bytes = text_to_speech(req.text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
def health(adapter_date: bool = False) -> JSONResponse: # Added return type
    latest_adapter: str | None = None
    if adapter_date:
        latest_dir: Path | None = _loader.get_latest_adapter_dir()
        latest_adapter = latest_dir.name if latest_dir else None

    phi3_components = get_phi3_model_and_tokenizer()
    hermes_components = get_hermes_model_and_tokenizer()

    phi3_ok = bool(phi3_components and len(phi3_components) > 0 and phi3_components[0] is not None)
    hermes_ok = bool(hermes_components and len(hermes_components) > 0 and hermes_components[0] is not None)

    return JSONResponse(
        {
            "uptime": time.time() - _START_TS,
            "phi_ok": phi3_ok,  # PATCHABLE
            "hermes_ok": hermes_ok,  # PATCHABLE
            "mean_hermes_score_last_24h": _db.get_mean_hermes_score_last_24h(),
            "latest_adapter": latest_adapter,
        }
    )

# AZR-13: Prometheus Metrics Endpoint
# These imports are needed globally if /metrics is global
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST # Moved to global for /metrics
from starlette.responses import Response as StarletteResponse # Moved to global for /metrics

@app.get("/metrics", response_class=StarletteResponse)
async def metrics() -> StarletteResponse:
    """Exposes Prometheus metrics."""
    return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- #
#  Helper used by tests/test_feedback_versioning.py
# --------------------------------------------------------------------------- #
async def submit_phi3_feedback(item: Any) -> Dict[str, str]: # Added type hint for item
    """
    Persist a feedback record in LanceDB.
    """
    schema_ver: str = getattr(item, "schema_version", None) or "1.0"

    raw_ts = getattr(item, "timestamp", None)
    ts: Union[int, float] = _coerce_ts(raw_ts)

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
#  AZR Planner Public API V1
# --------------------------------------------------------------------------- #
# These imports should be generally available, not conditional on OSIRIS_TEST
# as this is a public endpoint.
from azr_planner.schemas import PlanningContext as AZRPlanningContext, TradeProposal as AZRTradeProposal
from azr_planner.engine import generate_plan as azr_generate_plan_engine

router_azr_v1 = fastapi.APIRouter(
    prefix="/azr_api/v1",
    tags=["AZR Planner"], # Use the same tag for grouping in OpenAPI
)

@router_azr_v1.post(
    "/propose_trade",
    response_model=AZRTradeProposal,
    summary="Propose a trade based on market context and planner logic",
)
async def propose_trade_v1(ctx: AZRPlanningContext) -> AZRTradeProposal:
    """
    Accepts a PlanningContext and returns a TradeProposal generated by the AZR Planner engine.
    This is the primary public endpoint for the AZR Planner.
    """
    return azr_generate_plan_engine(ctx)

app.include_router(router_azr_v1)


# --------------------------------------------------------------------------- #
#  AZR Planner (internal, test only - can be kept for internal testing if needed)
# --------------------------------------------------------------------------- #
import os
# Ensure imports are conditional or handled if azr_planner is optional
if os.getenv("OSIRIS_TEST") == "1":
    # Schemas and engine already imported above, but can be re-imported if preferred for clarity
    # from azr_planner.schemas import PlanningContext, TradeProposal # Already imported as AZRPlanningContext etc.
    # from azr_planner.engine import generate_plan as azr_generate_plan # Already imported as azr_generate_plan_engine
    from azr_planner.risk_gate import accept as risk_gate_accept, RiskGateConfig # AZR-12 Import
    # from fastapi import HTTPException # No longer needed as using JSONResponse directly
    # AZR-13: Prometheus client imports
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    # Response class from Starlette, also available via fastapi.responses
    from starlette.responses import Response as StarletteResponse


    router_azr_internal = fastapi.APIRouter(
        prefix="/azr_api/internal/azr/planner",
        tags=["AZR Planner Internal"], # Differentiate tag for internal use
    )

    @router_azr_internal.post(
        "/propose_trade",
        response_model=AZRTradeProposal, # On success
        # Add responses for OpenAPI documentation for the 409 case
        responses={
            200: {"description": "Trade proposal accepted"},
            409: {"description": "Trade proposal rejected by Risk Gate",
                  "content": {"application/json": {"example": {"detail": "risk_gate_reject", "reason": "high_risk"}}}}
        }
    )
    async def propose_trade_internal(ctx: AZRPlanningContext) -> Union[AZRTradeProposal, JSONResponse]: # Return type can be Union
        """
        Internal test wrapper around azr_planner.engine.generate_plan.
        Includes AZR-12 Risk Gate check.
        """
        trade_proposal = azr_generate_plan_engine(ctx)

        # AZR-12: Call Risk Gate
        # Using default RiskGateConfig for now
        accepted, reason = risk_gate_accept(trade_proposal, cfg=None)

        if not accepted:
            # Respond with HTTP 409 and specified JSON body
            # To match the exact flat JSON structure {"detail": "risk_gate_reject", "reason": "<string>"}
            # we should use JSONResponse directly. HTTPException tends to nest details.
            return JSONResponse(
                status_code=fastapi.status.HTTP_409_CONFLICT,
                content={"detail": "risk_gate_reject", "reason": reason or "unknown_reason"}
            )

        return trade_proposal

    app.include_router(router_azr_internal)
