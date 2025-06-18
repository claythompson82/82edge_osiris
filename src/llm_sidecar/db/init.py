"""Osiris FastAPI server – thin wrappers around helper modules.

Key points
──────────
•  Exposes *module-level* loader functions so tests can patch them via
   ``mock.patch("osiris.server.get_phi3_model_and_tokenizer")``.
•  Imports `llm_sidecar.hermes_plugin` at call time – patches then work.
"""

from __future__ import annotations

import glob
import io
import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from osiris.schemas import (
    GenerateRequest,
    ScoreHermesRequest,
    HealthResponse,
    FeedbackItem,
    SpeakRequest,
)

# ------------------------------------------------------------------ #
# Model loader wrappers (tests patch these *directly* on this module)
# ------------------------------------------------------------------ #
from llm_sidecar.loader import (  # noqa: E402  keep after std-lib imports
    get_phi3_model_and_tokenizer,
    get_hermes_model_and_tokenizer,
)

import llm_sidecar.db as _db  # noqa: E402
import llm_sidecar.hermes_plugin as _hermes_plugin  # noqa: E402

from .text import text_to_speech  # noqa: E402

app = FastAPI()

# ------------------------------------------------------------------ #
# Async generation helpers
# ------------------------------------------------------------------ #
async def _generate_hermes_text(prompt: str, max_length: int) -> str:
    model, _ = get_hermes_model_and_tokenizer()
    return await model.generate(prompt, max_length)  # type: ignore[attr-defined]


async def _generate_phi3_json(prompt: str, max_length: int) -> dict:
    model, _ = get_phi3_model_and_tokenizer()
    return await model.generate_json(prompt, max_length)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #
@app.post("/generate/")
async def generate_endpoint(req: GenerateRequest):
    if (req.model_id or "hermes") == "hermes":
        out = await _generate_hermes_text(req.prompt, req.max_length)
        return JSONResponse({"output": out})
    return JSONResponse(await _generate_phi3_json(req.prompt, req.max_length))


@app.post("/score/hermes/")
def score_hermes_endpoint(req: ScoreHermesRequest):
    score = _hermes_plugin.score_with_hermes(req.proposal, req.context)
    _db.log_hermes_score(int(datetime.utcnow().timestamp() * 1e9), score)
    return JSONResponse({"score": score})


@app.post("/feedback/phi3/")
async def submit_phi3_feedback(req: FeedbackItem):
    _db.append_feedback(req.model_dump(by_alias=True, exclude_none=True))
    return JSONResponse({"status": "success"})


@app.get("/health", response_model=HealthResponse)
def health(adapter_date: bool = False):
    """Cheap liveness endpoint plus optional adapter metadata."""
    def _flag(v):  # tuples, bools, None → bool
        return bool(v[0] if isinstance(v, tuple) else v)

    info = {
        "phi_ok": _flag(get_phi3_model_and_tokenizer()),
        "hermes_ok": _flag(get_hermes_model_and_tokenizer()),
    }

    if adapter_date:
        from llm_sidecar import loader as _loader  # local to pick up patches
        dirs = [d for d in glob.glob(os.path.join(_loader.ADAPTER_ROOT, "*")) if os.path.isdir(d)]
        dirs.sort()
        info["latest_adapter"] = os.path.basename(dirs[-1]) if dirs else None
        info["mean_hermes_score_last_24h"] = _db.get_mean_hermes_score_last_24h()

    return info


@app.post("/speak")
def speak_endpoint(req: SpeakRequest):
    audio = text_to_speech(req.text)
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
