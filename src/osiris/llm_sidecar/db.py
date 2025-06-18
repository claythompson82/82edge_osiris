```python
# src/osiris/llm_sidecar/db.py
# ----------------------------
# LanceDB-backed database helpers for Osiris
# ----------------------------

import os
import pathlib
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import lancedb
from pydantic import BaseModel, Field

# Configuration: directory for LanceDB files (can be overridden via env)
DB_ROOT = pathlib.Path(os.environ.get("LANCE_DB_ROOT", "db"))

# Internal state
_db: Optional[lancedb.LanceDB] = None
_tables: Dict[str, lancedb.LanceTable] = {}

# -----------------------
# 1. Pydantic Schemas
# -----------------------

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Dict[str, Any]
    schema_version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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

# -----------------------
# 2. LanceDB Connection
# -----------------------

def connect(db_root: Optional[pathlib.Path] = None) -> lancedb.LanceDB:
    """
    Initializes the LanceDB connection and ensures required tables exist.
    Idempotent: safe to call multiple times.
    """
    global _db, _tables
    if _db is not None:
        return _db

    # Determine root directory
    root = db_root or DB_ROOT
    root.mkdir(parents=True, exist_ok=True)

    # Connect to LanceDB
    _db = lancedb.connect(root)
    _tables.clear()

    # Internal helper: create or open a table
    def _ensure_table(name: str, schema_cls: BaseModel):
        if name not in _db.table_names():
            # Bootstrap with one record matching the schema
            _db.create_table(
                name,
                data=[schema_cls().model_dump()],
                mode="overwrite",
            )
        _tables[name] = _db.open_table(name)

    # Ensure all tables are present
    _ensure_table("phi3_feedback", Phi3FeedbackSchema)
    _ensure_table("orchestrator_runs", OrchestratorRunSchema)
    _ensure_table("hermes_scores", HermesScoreSchema)

    return _db


def get_table(name: str) -> lancedb.LanceTable:
    """
    Returns a LanceDB table instance, initializing the DB if necessary.
    """
    if name not in _tables:
        connect()
    return _tables[name]

# ------------------------
# 3. Logging Functions
# ------------------------

def log_phi3_feedback(feedback: Phi3FeedbackSchema) -> None:
    """Appends a feedback entry to the phi3_feedback table."""
    tbl = get_table("phi3_feedback")
    tbl.add([feedback.model_dump()])


def log_orchestrator_run(run: OrchestratorRunSchema) -> None:
    """Appends an orchestrator run entry to the orchestrator_runs table."""
    tbl = get_table("orchestrator_runs")
    tbl.add([run.model_dump()])


def log_hermes_score(score: HermesScoreSchema) -> None:
    """Appends a score entry to the hermes_scores table."""
    tbl = get_table("hermes_scores")
    tbl.add([score.model_dump()])

# ------------------------
# 4. Query Functions
# ------------------------

def get_mean_hermes_score_last_24h() -> Optional[float]:
    """
    Returns the mean Hermes score over the last 24 hours, or None if no data.
    """
    import pandas as pd

    tbl = get_table("hermes_scores")
    df = tbl.to_pandas()
    if df.empty:
        return None

    # Parse timestamp to datetime, assume ISO with UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=24)
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return None

    return float(recent["score"].mean())

# ------------------------
# 5. Auto-bootstrap on Import
# ------------------------

try:
    # Initialize DB on module load to avoid missing-table errors in tests
    connect()
except Exception:
    # In environments lacking LanceDB, allow tests to patch functions as needed
    pass
```
