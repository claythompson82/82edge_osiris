"""
Shared pydantic data-models for the Osiris API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    """UTC timestamp in ISO-8601 format (used as default factory)."""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Inference / generation
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int
    model_id: str = Field("hermes", alias="model_id")


class GenerateResponse(BaseModel):
    output: str


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class ScoreHermesRequest(BaseModel):
    proposal: Any
    context: Optional[str] = None


class ScoreHermesResponse(BaseModel):
    score: float


# --------------------------------------------------------------------------- #
# Health-check
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    # always present
    phi_ok: bool
    hermes_ok: bool

    # optional – populated when ?adapter_date=true
    latest_adapter: Optional[str] = None
    mean_hermes_score_last_24h: Optional[float] = None
    uptime: Optional[int] = Field(
        default=None,
        description="Process uptime in seconds (present only when adapter_date=true)",
    )

    class Config:
        extra = "allow"  # tolerate future keys without validation errors


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Any
    timestamp: str = Field(default_factory=_now_iso)
    schema_version: str = Field("1.0", alias="schema_version")


class FeedbackResponse(BaseModel):
    status: str = "ok"


# --------------------------------------------------------------------------- #
# Text-to-speech
# --------------------------------------------------------------------------- #
class SpeakRequest(BaseModel):
    text: str
