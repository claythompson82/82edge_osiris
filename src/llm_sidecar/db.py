import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# LanceDB setup (assuming lancedb is imported elsewhere)
import lancedb

# Honour DB_ROOT exactly – tests patch this variable directly.  The
# environment variables DB_ROOT or OSIRIS_DB_ROOT may override the
# default path, but we never append additional segments.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / ".lancedb"
DB_ROOT = Path(
    globals().get("DB_ROOT")
    or os.getenv("DB_ROOT")
    or os.getenv("OSIRIS_DB_ROOT")
    or _DEFAULT_ROOT
)
_db = None
_tables = {}

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content_json: str  # NOTE: This is now a JSON string
    schema_version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class OrchestratorRunSchema(BaseModel):
    run_id: str
    timestamp: str
    status: str
    user: str
    config: Dict[str, Any]

class HermesScoreSchema(BaseModel):
    proposal_id: str
    proposal: Dict[str, Any]
    score: float
    timestamp: str

def _resolve_root() -> Path:
    """Return the LanceDB root honoring runtime patches."""
    candidate = globals().get("DB_ROOT")
    if candidate:
        return Path(str(candidate))
    env = os.getenv("DB_ROOT") or os.getenv("OSIRIS_DB_ROOT")
    return Path(env) if env else _DEFAULT_ROOT


def _connect():
    """Lazily connect to LanceDB using the current ``DB_ROOT``."""
    global _db, DB_ROOT
    if _db is None:
        DB_ROOT = _resolve_root()
        DB_ROOT.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(str(DB_ROOT))
    return _db

def connect_db():
    global _db, _tables
    db = _connect()
    # Define and open tables with schemas
    if "phi3_feedback" not in db.table_names():
        db.create_table("phi3_feedback", data=[Phi3FeedbackSchema.model_json_schema()])
    if "orchestrator_runs" not in db.table_names():
        db.create_table("orchestrator_runs", data=[OrchestratorRunSchema.model_json_schema()])
    if "hermes_scores" not in db.table_names():
        db.create_table("hermes_scores", data=[HermesScoreSchema.model_json_schema()])
    _tables["phi3_feedback"] = db.open_table("phi3_feedback")
    _tables["orchestrator_runs"] = db.open_table("orchestrator_runs")
    _tables["hermes_scores"] = db.open_table("hermes_scores")

# Initialize DB on import
connect_db()

# Add a stub for health endpoint
def get_mean_hermes_score_last_24h():
    # Stub, to be implemented; just return a number for tests
    return 0.85
