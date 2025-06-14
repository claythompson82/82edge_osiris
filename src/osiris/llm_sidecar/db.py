# osiris/llm_sidecar/db.py

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import lancedb
import os
import pathlib

# Change these as appropriate for your setup
DB_ROOT = pathlib.Path(os.environ.get("LANCE_DB_ROOT", "db"))
_db = None
_tables = {}

# -----------------------
# 1. Pydantic Schemas
# -----------------------

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Dict[str, Any]
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

# -----------------------
# 2. LanceDB Connection
# -----------------------

def _table_schema_for_model(model_cls):
    # Returns the schema fields for the LanceDB table.
    # For this demo, simply using pyarrow's automatic schema detection via model.dict()
    return None  # Let LanceDB infer

def connect(db_root=DB_ROOT):
    global _db, _tables
    if _db is not None:
        return _db

    db_root = pathlib.Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    _db = lancedb.connect(db_root)
    _tables = {}

    # Feedback Table
    if "phi3_feedback" not in _db.table_names():
        _db.create_table(
            "phi3_feedback",
            data=[Phi3FeedbackSchema(
                transaction_id="bootstrap",
                feedback_type="test",
                feedback_content={"msg": "bootstrap"},
            ).model_dump()],
            mode="overwrite"
        )
    _tables["phi3_feedback"] = _db.open_table("phi3_feedback")

    # Orchestrator Runs Table
    if "orchestrator_runs" not in _db.table_names():
        _db.create_table(
            "orchestrator_runs",
            data=[OrchestratorRunSchema(
                run_id="bootstrap",
                timestamp=datetime.utcnow().isoformat(),
                status="bootstrap",
                user="system",
                config={"msg": "bootstrap"}
            ).model_dump()],
            mode="overwrite"
        )
    _tables["orchestrator_runs"] = _db.open_table("orchestrator_runs")

    # Hermes Scores Table
    if "hermes_scores" not in _db.table_names():
        _db.create_table(
            "hermes_scores",
            data=[HermesScoreSchema(
                proposal_id="bootstrap",
                proposal={"test": 1},
                score=1.0,
                timestamp=datetime.utcnow().isoformat()
            ).model_dump()],
            mode="overwrite"
        )
    _tables["hermes_scores"] = _db.open_table("hermes_scores")

    return _db

def get_table(name):
    if name not in _tables:
        connect()  # triggers connection and table discovery
    return _tables[name]

# ------------------------
# 3. Logging Functions
# ------------------------

def log_phi3_feedback(feedback: Phi3FeedbackSchema):
    tbl = get_table("phi3_feedback")
    tbl.add([feedback.model_dump()])

def log_orchestrator_run(run: OrchestratorRunSchema):
    tbl = get_table("orchestrator_runs")
    tbl.add([run.model_dump()])

def log_hermes_score(score: HermesScoreSchema):
    tbl = get_table("hermes_scores")
    tbl.add([score.model_dump()])

# ------------------------
# 4. Query (Example only)
# ------------------------

def get_mean_hermes_score_last_24h():
    import pandas as pd
    tbl = get_table("hermes_scores")
    df = tbl.to_pandas()
    if df.empty:
        return None
    # Filter for last 24h
    now = datetime.utcnow()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = now - pd.Timedelta(hours=24)
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return None
    return float(recent["score"].mean())
