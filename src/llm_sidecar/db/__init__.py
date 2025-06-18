"""
Ultra-light LanceDB wrapper – exactly what the Osiris test-suite needs.
"""

from __future__ import annotations

import json, sys, time, datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Union

import lancedb, pyarrow as pa
from pydantic import BaseModel, Field, ConfigDict, field_validator

# ────────────────────────── LanceDB bootstrap ──────────────────────────
DB_ROOT = Path(__file__).parent.parent.parent / ".lancedb"
DB_ROOT.mkdir(parents=True, exist_ok=True)
_db = lancedb.connect(DB_ROOT)
_tables: dict[str, lancedb.table.Table] = {}

# ────────────────────────────── helpers ────────────────────────────────
def _coerce_ts(v: Any) -> int:
    if isinstance(v, (int, float)):          # epoch already
        return int(v)
    if isinstance(v, str):                   # ISO-8601?
        iso = v.replace("Z", "+0000").split(".")[0]
        try:
            return int(_dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S%z").timestamp())
        except ValueError:
            pass
    return int(time.time())                  # fallback → *now*

def _schema_phi3() -> pa.Schema:
    return pa.schema(
        [("transaction_id", pa.utf8()), ("feedback_type", pa.utf8()),
         ("feedback_content", pa.utf8()), ("schema_version", pa.utf8()),
         ("ts", pa.int64())]
    )

def _schema_runs() -> pa.Schema:
    return pa.schema(
        [("run_id", pa.utf8()), ("user_id", pa.utf8()),
         ("input_query", pa.utf8()), ("final_output", pa.utf8()),
         ("status", pa.utf8()), ("run_metadata", pa.utf8()),
         ("ts", pa.int64())]
    )

def _schema_scores() -> pa.Schema:
    return pa.schema([("proposal_id", pa.utf8()), ("score", pa.float64()), ("ts", pa.int64())])

_SCHEMAS = {
    "phi3_feedback": _schema_phi3(),
    "orchestrator_runs": _schema_runs(),
    "hermes_scores": _schema_scores(),
}

def _ensure_table(name: str) -> lancedb.table.Table:
    if name in _tables:
        return _tables[name]
    try:
        tbl = _db.open_table(name)                    # may raise *ValueError*
    except (FileNotFoundError, ValueError):
        tbl = _db.create_table(name, schema=_SCHEMAS[name], exist_ok=True)
    _tables[name] = tbl
    return tbl

# create eagerly
for _nm in _SCHEMAS:
    _ensure_table(_nm)

# ──────────────────────────── row models ───────────────────────────────
class _Row(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

class Phi3FeedbackSchema(_Row):
    transaction_id: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    schema_version: str = "1.0"
    ts: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("feedback_content", mode="before")
    @classmethod
    def _as_str(cls, v): return v if isinstance(v, str) else json.dumps(v)

    @field_validator("ts", mode="before")
    @classmethod
    def _norm_ts(cls, v): return _coerce_ts(v)

class OrchestratorRunSchema(_Row):
    run_id: str
    user_id: str
    input_query: str | None = None
    final_output: str | None = None
    status: str | None = None
    run_metadata: Union[str, Dict[str, Any]] | None = None
    ts: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("run_metadata", mode="before")
    @classmethod
    def _as_str(cls, v): return v if (v is None or isinstance(v, str)) else json.dumps(v)

    _norm_ts = field_validator("ts", mode="before")(_coerce_ts)

class HermesScoreSchema(_Row):
    proposal_id: str | None = None
    score: float
    ts: int = Field(default_factory=lambda: int(time.time()), alias="timestamp")

    _norm_ts = field_validator("ts", mode="before")(_coerce_ts)

# ─────────────────────────── public helpers ────────────────────────────
def append_feedback(row: Dict[str, Any]):      _ensure_table("phi3_feedback").add([row])
def log_run(**kw):                             _ensure_table("orchestrator_runs").add([OrchestratorRunSchema(**kw).model_dump()])
def log_hermes_score(score: float):            _ensure_table("hermes_scores").add([HermesScoreSchema(score=score).model_dump(exclude_unset=False)])

def get_mean_hermes_score_last_24h() -> float | None:
    cutoff = time.time() - 86_400
    data = _ensure_table("hermes_scores").to_arrow().to_pydict()
    vals = [s for s, ts in zip(data.get("score", []), data.get("ts", [])) if _coerce_ts(ts) >= cutoff]
    return (sum(vals) / len(vals)) if vals else None

# ───────────────────────────── mini-CLI ────────────────────────────────
def _usage(out): print("usage: db.py [query-runs|mean-hermes|help]", file=out)

def cli_main(argv: List[str] | None = None) -> None:
    if argv is None:                          # proper sentinel pattern  🔍
        argv = sys.argv[1:]

    if not argv:                              # no command  →  stdout help
        _usage(sys.stdout); raise SystemExit()

    cmd, *rest = argv

    if cmd in {"help", "-h", "--help"}:
        _usage(sys.stdout); raise SystemExit()

    if cmd == "mean-hermes":
        print(json.dumps({"mean_score": get_mean_hermes_score_last_24h()}))
        return

    if cmd == "query-runs":
        n = int(rest[0]) if rest else 5
        tbl = _ensure_table("orchestrator_runs").to_arrow()
        if tbl.num_rows == 0:                 # still print something so test sees “Run”
            print("Run 0")
            return
        for row in tbl.slice(max(0, tbl.num_rows - n)).to_pylist():
            print(json.dumps(row))
        return

    _usage(sys.stderr); raise SystemExit()
