import os
import io
import sys
import time
import requests
import datetime as _dt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union

# --- LLM Loader Imports ---
try:
    from osiris.llm_sidecar.loader import (
        get_hermes_model_and_tokenizer,
        get_phi3_model_and_tokenizer
    )
except ImportError:
    def get_hermes_model_and_tokenizer():
        raise NotImplementedError("Should be patched by tests.")
    def get_phi3_model_and_tokenizer():
        raise NotImplementedError("Should be patched by tests.")

# --- DB Helper Imports ---
try:
    from osiris.llm_sidecar.db import get_mean_hermes_score_last_24h
except ImportError:
    def get_mean_hermes_score_last_24h():
        return 0.0

# --- Resemble.ai Chatterbox TTS ---
RESEMBLE_API_KEY = os.getenv("RESEMBLE_API_KEY")
RESEMBLE_PROJECT_UUID = os.getenv("RESEMBLE_PROJECT_UUID")  # Must be set!

def text_to_speech(text: str) -> bytes:
    """
    Uses Resemble.ai's Chatterbox API to generate TTS audio.
    Handles both immediate and async jobs with polling.
    """
    if not RESEMBLE_API_KEY or not RESEMBLE_PROJECT_UUID:
        raise RuntimeError("Resemble.ai API key or project UUID not set in environment variables.")
    url = f"https://app.resemble.ai/api/v2/projects/{RESEMBLE_PROJECT_UUID}/clips"
    headers = {
        "Authorization": f"Token {RESEMBLE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "title": "Chat Response",
        "body": text,
        "voice_uuid": None,  # Set a specific voice UUID if you have one
        "output_format": "wav"
    }
    response = requests.post(url, json=payload, headers=headers)
    if not response.ok:
        raise HTTPException(status_code=500, detail=f"TTS error: {response.text}")
    result = response.json()
    audio_url = result.get("audio_src")
    clip_uuid = result.get("uuid")

    # Poll if audio is not instantly ready (Resemble async fallback)
    if not audio_url and clip_uuid:
        poll_url = f"https://app.resemble.ai/api/v2/projects/{RESEMBLE_PROJECT_UUID}/clips/{clip_uuid}"
        for _ in range(10):
            poll_resp = requests.get(poll_url, headers=headers)
            poll_result = poll_resp.json()
            audio_url = poll_result.get("audio_src")
            if audio_url:
                break
            time.sleep(2)
    if not audio_url:
        raise HTTPException(status_code=500, detail="TTS response did not include audio link.")
    audio_resp = requests.get(audio_url)
    if not audio_resp.ok:
        raise HTTPException(status_code=500, detail="Error downloading TTS audio.")
    return audio_resp.content

# --- FastAPI App ---
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "hermes"
    max_length: int = 256

class ScoreHermesRequest(BaseModel):
    proposal: dict
    context: str

@app.post("/generate/")
async def generate(req: GenerateRequest):
    """
    Route for LLM generation. Uses Hermes by default; supports phi3.
    """
    if req.model_id == "phi3":
        return await _generate_phi3_json(req.prompt, req.max_length)
    else:
        return await _generate_hermes_text(req.prompt, req.max_length)

@app.post("/score/hermes/")
def score_hermes(req: ScoreHermesRequest):
    """
    Route for scoring proposals using Hermes.
    """
    score = score_with_hermes(req.proposal, req.context)
    return {"score": score}

@app.get("/health")
def health(adapter_date: bool = False):
    """
    Returns uptime, Hermes score, and (optionally) latest adapter date.
    """
    resp = {
        "uptime": int((_dt.datetime.utcnow() - _dt.datetime.utcfromtimestamp(0)).total_seconds()),
        "hermes_mean_score": get_mean_hermes_score_last_24h(),
        "latest_adapter": None,
    }
    if adapter_date:
        adapter_root = getattr(sys.modules.get("osiris.llm_sidecar.loader"), "ADAPTER_ROOT", None)
        if adapter_root:
            dirs = [
                d for d in os.listdir(adapter_root)
                if (adapter_root / d).is_dir() and d[:4].isdigit()
            ]
            if dirs:
                resp["latest_adapter"] = sorted(dirs)[-1]
    return resp

@app.post("/speak")
async def speak(request: Request):
    """
    TTS endpoint: returns audio/wav for input text using Resemble.
    """
    data = await request.json()
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    audio_bytes = text_to_speech(text)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

# --- FeedbackItem & submit_phi3_feedback for test compatibility ---
class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    timestamp: str
    schema_version: Optional[str] = Field(default="1.0")

async def submit_phi3_feedback(item: FeedbackItem):
    """
    Simulates storing feedback in the DB.
    Patch in real append_feedback logic or patch in tests.
    """
    # To be patched/mocked in tests or call db.append_feedback(item.dict())
    pass

# --- Dummy Implementations for Patch/Test ---
async def _generate_phi3_json(prompt, max_length):
    raise NotImplementedError("Should be patched by tests.")

async def _generate_hermes_text(prompt, max_length):
    raise NotImplementedError("Should be patched by tests.")

def score_with_hermes(proposal, context):
    raise NotImplementedError("Should be patched by tests.")
