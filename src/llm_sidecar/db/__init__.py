import os
import json
import pathlib
from typing import Any, Dict, Optional, List
from datetime import datetime

import lancedb
from pydantic import BaseModel, Field

# --- DB Connection Setup ---
DB_ROOT = os.environ.get("OSIRIS_DB_ROOT", str(pathlib.Path(__file__).parent.parent.parent / "db_data"))
_db = lancedb.connect(DB_ROOT)
_tables = {
    "phi3_feedback": _db.open_table("phi3_feedback"),
    "orchestrator_runs": _db.open_table("orchestrator_runs"),
    "hermes_scores": _db.open_table("hermes_scores"),
}

# --- Pydantic Schemas ---

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Optional[dict] = None  # dict, now always deserialized
    schema_version: str = Field(default="1.0")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class OrchestratorRunSchema(BaseModel):
    run_id: str
    timestamp: str
    status: str
    user: Optional[str] = None
    config: Optional[dict] = None

class HermesScoreSchema(BaseModel):
    proposal_id: str
    proposal: dict
    score: float
    timestamp: str

# --- Logging/Insertion Functions ---

def log_phi3_feedback(feedback: Phi3FeedbackSchema) -> None:
    """Insert a feedback record (deserialized) into the phi3_feedback table."""
    # Convert feedback_content to dict (already a dict per schema).
    _tables["phi3_feedback"].add([feedback.model_dump()])

def log_run(run: OrchestratorRunSchema) -> None:
    """Insert an orchestrator run record."""
    _tables["orchestrator_runs"].add([run.model_dump()])

def log_hermes_score(score: HermesScoreSchema) -> None:
    """Insert a Hermes score record."""
    _tables["hermes_scores"].add([score.model_dump()])

# --- Utility Functions ---

def get_mean_hermes_score_last_24h() -> Optional[float]:
    """Calculate the mean Hermes score from the last 24 hours."""
    import pandas as pd
    from datetime import timedelta

    df = _tables["hermes_scores"].to_pandas()
    if df.empty or "timestamp" not in df.columns or "score" not in df.columns:
        return None

    now = datetime.utcnow()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff = now - timedelta(hours=24)
    recent_scores = df[df["timestamp"] >= cutoff]["score"]
    if recent_scores.empty:
        return None
    return float(recent_scores.mean())

# --- Exports (for explicit imports elsewhere) ---
__all__ = [
    "log_phi3_feedback",
    "log_run",
    "log_hermes_score",
    "get_mean_hermes_score_last_24h",
    "Phi3FeedbackSchema",
    "OrchestratorRunSchema",
    "HermesScoreSchema",
]
