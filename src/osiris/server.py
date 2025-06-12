# File: src/osiris/server.py

# -----------------------------------------------------------------------------
# Shim httpx.Client to accept `app=` for FastAPI TestClient
# -----------------------------------------------------------------------------
import httpx
_original_httpx_init = httpx.Client.__init__

def _shim_httpx_init(self, *args, app=None, **kwargs):
    if 'app' in kwargs:
        kwargs.pop('app')
    return _original_httpx_init(self, *args, **kwargs)

httpx.Client.__init__ = _shim_httpx_init

# -----------------------------------------------------------------------------
# Standard imports
# -----------------------------------------------------------------------------
import os
import json
import uuid
import datetime
import asyncio
import logging
import io

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Third-party stubs & optional integrations
# -----------------------------------------------------------------------------
try:
    from sentry_sdk.integrations.logging import LoggingIntegration
except ImportError:
    LoggingIntegration = None

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

try:
    from common.otel_init import init_otel
except ImportError:
    def init_otel(app=None):
        pass

# -----------------------------------------------------------------------------
# Feature flags & device
# -----------------------------------------------------------------------------
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() in ("1", "true", "yes")
print(f"[Side-car] metrics enabled: {ENABLE_METRICS}")
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() in ("1", "true", "yes")
print(f"[Side-car] profiling enabled: {ENABLE_PROFILING}")
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
print(f"[Side-car] running on {DEVICE}")

# -----------------------------------------------------------------------------
# Loader imports (no import-time loads)
# -----------------------------------------------------------------------------
from llm_sidecar.loader import (
    get_hermes_model_and_tokenizer,
    get_phi3_model_and_tokenizer,
)
import llm_sidecar.loader as loader  # for phi3_adapter_date

# -----------------------------------------------------------------------------
# TTS, DB, EventBus, Plugins
# -----------------------------------------------------------------------------
from llm_sidecar.tts import ChatterboxTTS
from llm_sidecar.db import append_feedback, log_hermes_score
import llm_sidecar.db as db
from llm_sidecar.event_bus import EventBus
from llm_sidecar.hermes_plugin import score_with_hermes

# DEV stubs for event handlers
try:
    from osiris.event_handlers import (
        handle_proposal_created,
        handle_proposal_assessed,
        handle_feedback_submitted_event,
    )
except ImportError:
    async def handle_proposal_created(*args, **kwargs): pass
    async def handle_proposal_assessed(*args, **kwargs): pass
    async def handle_feedback_submitted_event(*args, **kwargs): pass

# -----------------------------------------------------------------------------
# Feedback & schema files
# -----------------------------------------------------------------------------
PHI3_FEEDBACK_DATA_FILE = "/app/phi3_feedback_data.jsonl"
PHI3_FEEDBACK_LOG_FILE  = "/app/phi3_feedback_log.jsonl"

# -----------------------------------------------------------------------------
# FastAPI app & lifespan
# -----------------------------------------------------------------------------
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sentry
    dsn = os.getenv("SENTRY_DSN")
    if dsn and LoggingIntegration:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)]
        )

    # EventBus
    if os.getenv("REDIS_URL"):
        await event_bus.connect()
        await event_bus.subscribe("phi3.feedback.submitted", handle_feedback_submitted_event)

    yield

    await event_bus.close()

app.router.lifespan_context = lifespan

if ENABLE_METRICS and Instrumentator:
    Instrumentator().instrument(app).expose(app)

init_otel(app)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: dict
    timestamp: str
    corrected_proposal: dict = None
    schema_version: str = "1.0"

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 256

class UnifiedPromptRequest(PromptRequest):
    model_id: str = "hermes"

# -----------------------------------------------------------------------------
# Feedback endpoint
# -----------------------------------------------------------------------------
@app.post("/feedback/phi3/", tags=["feedback"])
async def submit_phi3_feedback(feedback: FeedbackItem):
    """
    Store feedback and return acknowledgment.
    """
    # ensure directory exists (ignore if not permitted)
    try:
        os.makedirs(os.path.dirname(PHI3_FEEDBACK_DATA_FILE), exist_ok=True)
    except PermissionError:
        pass

    # write feedback (ignore if cannot open file)
    try:
        with open(PHI3_FEEDBACK_DATA_FILE, "a") as f:
            f.write(json.dumps(feedback.dict()))
            f.write("\n")
    except (FileNotFoundError, PermissionError):
        pass

    return {"message": "Feedback received successfully", "transaction_id": feedback.transaction_id}

# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    hermes_ok, _ = get_hermes_model_and_tokenizer()
    phi3_ok, _ = get_phi3_model_and_tokenizer()
    return {
        "status": "ok" if (hermes_ok and phi3_ok) else "error",
        "hermes_loaded": hermes_ok,
        "phi3_loaded": phi3_ok,
        "phi3_adapter_date": loader.phi3_adapter_date,
    }

# -----------------------------------------------------------------------------
# Propose Trade Adjustments endpoint
# -----------------------------------------------------------------------------
@app.post("/propose_trade_adjustments/", tags=["strategy"])
async def propose_trade_adjustments(request: Request):
    """
    Generate a Phi-3 proposal and Hermes assessment, logging both.
    """
    data = await request.json()
    prompt = data.get("prompt")
    max_length = data.get("max_length", 256)
    context = data.get("context", None)

    # Generate Phi3 proposal
    phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()
    from llm_sidecar.server import _generate_phi3_json, _generate_hermes_text  # import helpers
    phi3_proposal = await _generate_phi3_json(prompt, max_length, phi3_model, phi3_tokenizer)

    # Evaluate with Hermes
    hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()
    hermes_assessment = await _generate_hermes_text(
        json.dumps(phi3_proposal), max_length, hermes_model, hermes_tokenizer
    )

    # ensure log directory exists (ignore if not permitted)
    try:
        os.makedirs(os.path.dirname(PHI3_FEEDBACK_LOG_FILE), exist_ok=True)
    except PermissionError:
        pass

    # append to log file (ignore if cannot open)
    try:
        with open(PHI3_FEEDBACK_LOG_FILE, "a") as f:
            entry = {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "phi3_proposal": phi3_proposal,
                "hermes_assessment": hermes_assessment,
            }
            f.write(json.dumps(entry))
            f.write("\n")
    except (FileNotFoundError, PermissionError):
        pass

    return {"phi3_proposal": phi3_proposal, "hermes_assessment": hermes_assessment}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    event_bus = EventBus(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    # Initialize TTS only on startup
    tts_model = ChatterboxTTS(
        model_dir=os.getenv("CHATTERBOX_MODEL_DIR", "/models/tts/chatterbox"),
        device=DEVICE,
        event_bus=event_bus,
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
