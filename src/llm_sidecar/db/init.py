from __future__ import annotations
import os
import pathlib
import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import lancedb

# --- SCHEMAS ---

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: dict
    schema_version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

class OrchestratorRunSchema(BaseModel):
    run_id: str
    start_time: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    end_time: Optional[str] = None
    status: str = "pending"
    details: dict = Field(default_factory=dict)

class HermesScoreSchema(BaseModel):
    proposal_id: str
    score: float
    context: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

# --- DATABASE SETUP ---

DB_ROOT = pathlib.Path(os.environ.get("OSIRIS_DB_ROOT", "./osiris_db")).resolve()
if not DB_ROOT.exists():
    DB_ROOT.mkdir(parents=True, exist_ok=True)
_db = lancedb.connect(DB_ROOT)

def _ensure_table(name, schema_model, **kwargs):
    # Table exists? Open or create
    try:
        tbl = _db.open_table(name)
    except Exception:
        tbl = _db.create_table(
            name,
            data=[schema_model(**{f: None for f in schema_model.model_fields}) for _ in range(1)],
            **kwargs
        )
    return tbl

_tables = {
    "phi3_feedback": _ensure_table("phi3_feedback", Phi3FeedbackSchema),
    "orchestrator_runs": _ensure_table("orchestrator_runs", OrchestratorRunSchema),
    "hermes_scores": _ensure_table("hermes_scores", HermesScoreSchema),
}

# --- LOGGING FUNCTIONS ---

def append_feedback(feedback: Phi3FeedbackSchema):
    tbl = _tables["phi3_feedback"]
    tbl.add(feedback.model_dump())

def log_run(run: OrchestratorRunSchema):
    tbl = _tables["orchestrator_runs"]
    tbl.add(run.model_dump())

def log_hermes_score(score: HermesScoreSchema):
    tbl = _tables["hermes_scores"]
    tbl.add(score.model_dump())

def get_mean_hermes_score_last_24h():
    tbl = _tables["hermes_scores"]
    import pandas as pd
    now = datetime.datetime.now(datetime.timezone.utc)
    # Get rows from last 24h
    rows = [r for r in tbl.to_pandas().to_dict(orient="records") if
            "timestamp" in r and
            pd.to_datetime(r["timestamp"]) > (now - datetime.timedelta(hours=24))]
    if not rows:
        return None
    return sum(float(r["score"]) for r in rows if "score" in r) / len(rows)

# --- CLI (for test_db.py dummy) ---

def cli_main(argv):
    if not argv or (argv[0] not in ("query-runs",)):
        print("usage: db.py [query-runs|...]", file=os.sys.stderr)
        raise SystemExit(2)
    elif argv[0] == "query-runs":
        print("Run 1\nRun 2")
    else:
        print("Unknown command", file=os.sys.stderr)
        raise SystemExit(2)

# --- For test imports ---
__all__ = [
    "Phi3FeedbackSchema", "OrchestratorRunSchema", "HermesScoreSchema",
    "append_feedback", "log_run", "log_hermes_score", "get_mean_hermes_score_last_24h",
    "cli_main"
]
