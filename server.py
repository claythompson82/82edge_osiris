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

import torch
import redis.asyncio as redis # Import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_sidecar.event_bus import publish
from llm_sidecar.loader import (
    load_hermes_model,
    load_phi3_model,
    get_hermes_model_and_tokenizer,
    get_phi3_model_and_tokenizer,
    MICRO_LLM_MODEL_PATH,
    phi3_adapter_date,
)

# outlines (schema-guided generation for Phi-3)
from outlines import generate as outlines_generate
from llm_sidecar.db import append_feedback

# ---------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 256


class UnifiedPromptRequest(PromptRequest):
    model_id: str = "hermes"  # "hermes" | "phi3"


class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str  # "correction" | "rating" | "qualitative_comment"
    feedback_content: Any
    timestamp: str
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"


# ---------------------------------------------------------------------
# FastAPI initialisation
# ---------------------------------------------------------------------
app = FastAPI()

print("[Side-car] loading models …")
load_hermes_model()
load_phi3_model()
print("[Side-car] models ready.")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_recent_feedback(max_examples: int = 3) -> List[Dict[str, Any]]:
    if not os.path.exists(PHI3_FEEDBACK_DATA_FILE):
        return []
    items: List[Dict[str, Any]] = []
    with open(PHI3_FEEDBACK_DATA_FILE, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if obj.get("feedback_type") == "correction" and isinstance(
                    obj.get("corrected_proposal"), dict
                ):
                    items.append(obj)
            except Exception as err:
                print(f"[Feedback-load] skipping line: {err}")
    return items[-max_examples:]


async def _generate_phi3_json(
    prompt: str, max_length: int, model, tokenizer
) -> Dict[str, Any]:
    if not model or not tokenizer:
        return {"error": "Phi-3 model / tokenizer unavailable."}

    examples = _load_recent_feedback()
    aug = ""
    if examples:
        header = (
            "Based on recent feedback, here are examples of desired JSON outputs:\n"
        )
        ex_txt = "\n".join(
            f"Example {i+1}:\n{json.dumps(e['corrected_proposal'], indent=2)}"
            for i, e in enumerate(examples)
        )
        aug = f"{header}{ex_txt}\n\nNow consider the following request:\n"

    effective_prompt = f"{aug}{prompt}"

    try:
        gen = outlines_generate.json(model, JSON_SCHEMA_STR, tokenizer=tokenizer)
        return gen(effective_prompt, max_tokens=max_length)
    except Exception as e:
        print("[Phi-3] generation error:", e)
        return {
            "error": "Phi-3 JSON generation failed.",
            "details": traceback.format_exc(),
        }


async def _generate_hermes_text(
    prompt: str, max_length: int, model, tokenizer
) -> str:
    if not model or not tokenizer:
        return "Error: Hermes model / tokenizer unavailable."
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **inputs, max_length=max_length, do_sample=False, eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print("[Hermes] generation error:", e)
        return f"Error generating text with Hermes: {e}"


# ---------------------------------------------------------------------
# End-points
# ---------------------------------------------------------------------
@app.post("/generate/", tags=["unified"])
async def generate_unified(req: UnifiedPromptRequest):
    if req.model_id == "phi3":
        model, tok = get_phi3_model_and_tokenizer()
        if not model:
            return {"error": "Phi-3 model not loaded."}
        return await _generate_phi3_json(req.prompt, req.max_length, model, tok)

    if req.model_id == "hermes":
        model, tok = get_hermes_model_and_tokenizer()
        if not model:
            return {"error": "Hermes model not loaded."}
        txt = await _generate_hermes_text(req.prompt, req.max_length, model, tok)
        if txt.startswith("Error"):
            return {"error": txt}
        return {"generated_text": txt}

    return {
        "error": "Invalid model_id. Use 'hermes' or 'phi3'.",
        "specified_model_id": req.model_id,
    }


@app.post("/generate/hermes/", tags=["hermes"])
async def generate_hermes(req: PromptRequest):
    model, tok = get_hermes_model_and_tokenizer()
    if not model:
        return {"error": "Hermes model not loaded."}
    txt = await _generate_hermes_text(req.prompt, req.max_length, model, tok)
    if txt.startswith("Error"):
        return {"error": txt}
    return {"generated_text": txt}


@app.post("/generate/phi3/", tags=["phi3"])
async def generate_phi3(req: PromptRequest):
    model, tok = get_phi3_model_and_tokenizer()
    if not model:
        return {"error": "Phi-3 model not loaded."}
    return await _generate_phi3_json(req.prompt, req.max_length, model, tok)


@app.post("/propose_trade_adjustments/", tags=["strategy"])
async def propose_trade(req: PromptRequest):
    phi3_model, phi3_tok = get_phi3_model_and_tokenizer()
    hermes_model, hermes_tok = get_hermes_model_and_tokenizer()

    if not phi3_model or not hermes_model:
        return {"error": "Required model(s) not loaded."}

    phi3_json = await _generate_phi3_json(
        req.prompt, req.max_length, phi3_model, phi3_tok
    )
    if "error" in phi3_json:
        return {"error": "Phi-3 failed.", "details": phi3_json}

    await publish("phi3.proposal.generated", phi3_json) # Added event publishing

    hermes_prompt = (
        "Assess the following JSON trade proposal and provide a brief critique:\n\n"
        f"{json.dumps(phi3_json, indent=2)}\n\nAssessment:"
    )
    hermes_text = await _generate_hermes_text(
        hermes_prompt, req.max_length * 2, hermes_model, hermes_tok
    )

    await publish("hermes.assessment.generated", {"assessment": hermes_text}) # Added event publishing

    # log
    try:
        entry = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "phi3_proposal": phi3_json,
            "hermes_assessment": hermes_text,
        }
        with open(PHI3_FEEDBACK_LOG_FILE, "a") as f:
            json.dump(entry, f)
            f.write("\n")
    except Exception as e:
        print("[Log] write error:", e)

    return {"phi3_proposal": phi3_json, "hermes_assessment": hermes_text}


@app.post("/feedback/phi3/", tags=["feedback"])
async def submit_phi3_feedback(feedback: FeedbackItem): # Renamed from submit_feedback to submit_phi3_feedback as per issue
    feedback.timestamp = datetime.datetime.utcnow().isoformat()
    append_feedback(feedback.model_dump()) # Use model_dump() as specified
    await publish("feedback.received", feedback.model_dump()) # Added event publishing
    return {
        "message": "Feedback stored in LanceDB",
        "transaction_id": feedback.transaction_id,
    }


@app.get("/health", tags=["meta"])
async def health():
    hermes_ok = all(get_hermes_model_and_tokenizer())
    phi3_ok = all(get_phi3_model_and_tokenizer())
    phi3_file = os.path.exists(MICRO_LLM_MODEL_PATH)

    status = "ok"
    if not hermes_ok and not phi3_ok:
        status = "error"
    elif not hermes_ok or not phi3_ok:
        status = "partial_error"

    redis_connected = False
    try:
        # Use connection details from event_bus.py or make them globally accessible
        r = await redis.Redis(host="localhost", port=6379, socket_connect_timeout=1) # Short timeout
        await r.ping()
        redis_connected = True
        await r.close()
    except Exception:
        redis_connected = False

    return {
        "status": status,
        "hermes_loaded": hermes_ok,
        "phi3_loaded": phi3_ok,
        "phi3_model_file_exists": phi3_file,
        "device": DEVICE,
        "phi3_adapter_date": phi3_adapter_date,
        "redis_connected": redis_connected, # New flag
    }


# Local dev entry-point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
