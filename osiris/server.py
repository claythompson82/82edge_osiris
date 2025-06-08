# server.py
# -----------------------------------------------------------------------------
# FastAPI side-car that serves Hermes-3-8B-GPTQ (text) and Phi-3-mini-4k-int8
# (JSON-structured) plus a feedback loop for nightly QLoRA / DPO training.
# -----------------------------------------------------------------------------

import os
import json
import uuid
import datetime
import traceback
from typing import Optional, Dict, Any, List
import asyncio
import logging
import base64
import sentry_sdk
import anyio._backends._asyncio # Preload for tests

try:
    from sentry_sdk.integrations.logging import LoggingIntegration
except Exception:
    LoggingIntegration = None
import io

import torch
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except Exception:
    Instrumentator = None
from pydantic import BaseModel

try:
    from common.otel_init import init_otel
except Exception:
    def init_otel(app=None):
        return

ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() in ("1", "true", "yes")
print(f"[Side-car] metrics enabled: {ENABLE_METRICS}")
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() in ("1", "true", "yes")
print(f"[Side-car] profiling enabled: {ENABLE_PROFILING}")

if ENABLE_PROFILING:
    from pyinstrument import Profiler
    from starlette.middleware.base import BaseHTTPMiddleware

from llm_sidecar.loader import (
    load_hermes_model,
    load_phi3_model,
    MICRO_LLM_MODEL_PATH,
)
from osiris.llm_sidecar import loader
from llm_sidecar.tts import ChatterboxTTS

try:
    from outlines import generate as outlines_generate
except Exception:
    outlines_generate = None

from llm_sidecar.db import append_feedback, log_hermes_score
import llm_sidecar.db as db
from llm_sidecar.event_bus import EventBus
from llm_sidecar.hermes_plugin import score_with_hermes

# ---------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------
CHATTERBOX_MODEL_DIR = "/models/tts/chatterbox"
PHI3_FEEDBACK_LOG_FILE = "/app/phi3_feedback_log.jsonl"
PHI3_FEEDBACK_DATA_FILE = "/app/phi3_feedback_data.jsonl"

JSON_SCHEMA_STR = """
{
  "type": "object",
  "properties": {
    "ticker":         { "type": "string",  "description": "Ticker symbol" },
    "action":         { "type": "string",  "enum": ["adjust", "pass", "abort"] },
    "side":           { "type": "string",  "enum": ["LONG", "SHORT"] },
    "new_stop_pct":   { "type": ["number","null"] },
    "new_target_pct": { "type": ["number","null"] },
    "confidence":     { "type": "number",  "minimum": 0, "maximum": 1 },
    "rationale":      { "type": "string",  "description": "One-sentence justification" }
  },
  "required": ["ticker", "action", "confidence", "rationale"],
  "additionalProperties": false
}
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Side-car] running on {DEVICE}")

last_profile_html = ""

# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 256

class UnifiedPromptRequest(PromptRequest):
    model_id: str = "hermes"

class SpeakRequest(BaseModel):
    text: str
    exaggeration: Optional[float] = 0.5
    ref_wav_b64: Optional[str] = None

class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Any
    timestamp: str
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"

class ScoreRequest(BaseModel):
    proposal: Dict[str, Any]
    context: Optional[str] = None

# ---------------------------------------------------------------------
# FastAPI initialisation
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sentry initialization
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)])
        logger.info("[Sentry] SDK initialized.")
    else:
        logger.info("[Sentry] SENTRY_DSN not found, skipping initialization.")

    logger.info("FastAPI startup: Connecting EventBus and subscribing to channels...")
    try:
        await event_bus.connect()
        await event_bus.subscribe("phi3.proposal.created", handle_proposal_created)
        await event_bus.subscribe("phi3.proposal.assessed", handle_proposal_assessed)
        await event_bus.subscribe("phi3.feedback.submitted", handle_feedback_submitted_event)
        logger.info("EventBus connected and subscriptions active.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    logger.info("FastAPI shutdown: Closing EventBus...")
    await event_bus.close()
    logger.info("EventBus closed.")

app = FastAPI(lifespan=lifespan)

if ENABLE_METRICS and Instrumentator:
    Instrumentator().instrument(app).expose(app)
init_otel(app)

if ENABLE_PROFILING:
    class ProfilingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            profiler = Profiler()
            profiler.start()
            response = await call_next(request)
            profiler.stop()
            global last_profile_html
            last_profile_html = profiler.output_html()
            return response
    app.add_middleware(ProfilingMiddleware)

event_bus = EventBus(redis_url="redis://localhost:6379/0")
logger = logging.getLogger(__name__)

print("[Side-car] loading models …")
load_hermes_model()
load_phi3_model()
phi3_adapter_date = loader.phi3_adapter_date
tts_model = ChatterboxTTS(model_dir=CHATTERBOX_MODEL_DIR, device=DEVICE, event_bus=event_bus)
print("[Side-car] models ready.")

def get_phi3_model_and_tokenizer():
    return loader.get_phi3_model_and_tokenizer()

def get_hermes_model_and_tokenizer():
    return loader.get_hermes_model_and_tokenizer()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_feedback_cache: List[Dict[str, Any]] = []
_feedback_mtime: Optional[float] = None

def _load_recent_feedback(max_examples: int = 3) -> List[Dict[str, Any]]:
    """Load recent correction feedback examples from disk."""
    global _feedback_cache, _feedback_mtime
    if not os.path.exists(PHI3_FEEDBACK_DATA_FILE):
        return []

    try:
        mtime = os.path.getmtime(PHI3_FEEDBACK_DATA_FILE)
        if _feedback_mtime is not None and mtime == _feedback_mtime:
            return _feedback_cache[-max_examples:]

        items: List[Dict[str, Any]] = []
        with open(PHI3_FEEDBACK_DATA_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if obj.get("feedback_type") == "correction" and isinstance(obj.get("corrected_proposal"), dict):
                            items.append(obj)
                    except json.JSONDecodeError:
                        continue
        
        _feedback_cache = items
        _feedback_mtime = mtime
        return items[-max_examples:]
    except Exception as e:
        logger.error(f"Error reading feedback file {PHI3_FEEDBACK_DATA_FILE}: {e}")
        return []

load_recent_feedback = _load_recent_feedback

# ---------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------

async def _generate_phi3_json(
    prompt: str,
    max_length: int,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """Generate a JSON proposal with Phi-3, optionally augmenting the prompt."""
    if model is None or tokenizer is None:
        model, tokenizer = get_phi3_model_and_tokenizer()

    schema = json.loads(JSON_SCHEMA_STR)
    if outlines_generate is None:
        raise RuntimeError("outlines library not available")

    generator = outlines_generate.json(model, schema, tokenizer=tokenizer)

    feedback_items = load_recent_feedback(max_examples=3)
    if feedback_items:
        examples = "\n".join(
            json.dumps(item["corrected_proposal"], indent=2)
            for item in feedback_items
            if isinstance(item.get("corrected_proposal"), dict)
        )
        prompt = (
            "Based on past feedback, here are examples of desired JSON outputs:\n"
            f"{examples}\n\nNow, considering the above, please process the following request:\n{prompt}"
        )

    return await generator(prompt, max_tokens=max_length)


async def _generate_hermes_text(
    prompt: str,
    max_length: int,
    model=None,
    tokenizer=None,
) -> str:
    """Generate text with Hermes. Real logic is stubbed for tests."""
    if model is None or tokenizer is None:
        model, tokenizer = get_hermes_model_and_tokenizer()

    if outlines_generate is None:
        raise RuntimeError("outlines library not available")

    generator = outlines_generate.text(model, tokenizer=tokenizer)
    return await generator(prompt, max_tokens=max_length)

# ---------------------------------------------------------------------
# End-points
# ---------------------------------------------------------------------
async def audio_byte_stream_generator(request: Request):
    queue = asyncio.Queue()
    await event_bus.register_listener_queue("audio.bytes", queue)
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                if message:
                    yield f"data: {message}\n\n"
                queue.task_done()
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    finally:
        await event_bus.unregister_listener_queue("audio.bytes", queue)

@app.post("/feedback/phi3/", tags=["feedback"])
async def submit_phi3_feedback(feedback: FeedbackItem):
    feedback.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # ``model_dump`` exists on Pydantic v2 models. Our tests use v1 so fall back
    # to ``dict`` for compatibility.
    if hasattr(feedback, "model_dump"):
        feedback_dict = feedback.model_dump()
    else:
        feedback_dict = feedback.dict()

    try:
        # Log to local file for training data
        with open(PHI3_FEEDBACK_DATA_FILE, "a") as f:
            f.write(json.dumps(feedback_dict) + "\n")

        # Add to LanceDB and publish event
        db.append_feedback(feedback_dict)
        if getattr(event_bus, "pubsub", None):
            await event_bus.publish("phi3.feedback.submitted", json.dumps(feedback_dict))
        
        return {
            "message": "Feedback received successfully",
            "transaction_id": feedback.transaction_id,
        }
    except Exception as e:
        logger.error(
            f"Error submitting feedback or publishing event for {feedback.transaction_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {e}")


@app.post("/generate/", tags=["strategy"])
async def generate(request: UnifiedPromptRequest):
    """Unified text/JSON generation endpoint."""
    if request.model_id == "hermes":
        model, tokenizer = get_hermes_model_and_tokenizer()
        text = await _generate_hermes_text(request.prompt, request.max_length, model, tokenizer)
        return {"generated_text": text}
    elif request.model_id == "phi3":
        model, tokenizer = get_phi3_model_and_tokenizer()
        return await _generate_phi3_json(request.prompt, request.max_length, model, tokenizer)
    else:
        raise HTTPException(status_code=422, detail="Invalid model_id")


@app.post("/propose_trade_adjustments/", tags=["strategy"])
async def propose_trade_adjustments(request: PromptRequest):
    """Generate a Phi-3 proposal and Hermes assessment, logging both."""
    phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()
    hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()

    phi3_proposal = await _generate_phi3_json(request.prompt, request.max_length, phi3_model, phi3_tokenizer)

    hermes_prompt = json.dumps(phi3_proposal, indent=2)
    hermes_assessment = await _generate_hermes_text(hermes_prompt, request.max_length, hermes_model, hermes_tokenizer)

    log_entry = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phi3_proposal": phi3_proposal,
        "hermes_assessment": hermes_assessment,
    }

    with open(PHI3_FEEDBACK_LOG_FILE, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

    if getattr(event_bus, "pubsub", None):
        await event_bus.publish("phi3.proposal.created", json.dumps(phi3_proposal))
        await event_bus.publish(
            "phi3.proposal.assessed",
            json.dumps({"proposal": phi3_proposal, "assessment": hermes_assessment}),
        )

    return {"phi3_proposal": phi3_proposal, "hermes_assessment": hermes_assessment}


@app.post("/score/hermes/", tags=["strategy"])
async def score_proposal_with_hermes(request: ScoreRequest):
    try:
        score = score_with_hermes(request.proposal, request.context)
        db.log_hermes_score({"proposal": request.proposal, "score": score})
        return {"proposal_id": request.proposal.get("id", str(uuid.uuid4())), "score": score}
    except Exception as e:
        logger.error(f"Error scoring proposal with Hermes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to score proposal: {e}")


@app.post("/speak", tags=["tts"])
async def speak(request: SpeakRequest):
    audio_bytes = tts_model.synth(request.text, exaggeration=request.exaggeration, ref_wav_b64=request.ref_wav_b64)
    return Response(content=audio_bytes, media_type="audio/wav")


# For now, keep the health endpoint as it's small.

@app.get("/health", tags=["meta"])
async def health():
    hermes_ok = all(get_hermes_model_and_tokenizer())
    phi3_ok = all(get_phi3_model_and_tokenizer())
    status = "ok" if hermes_ok and phi3_ok else "error"
    
    return {
        "status": status,
        "hermes_loaded": hermes_ok,
        "phi3_loaded": phi3_ok,
        "phi3_adapter_date": loader.phi3_adapter_date,
    }

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)