"""
Very-small wrapper around LanceDB used by the unit-tests.

The public surface MUST expose:
    • Phi3FeedbackSchema, OrchestratorRunSchema, HermesScoreSchema  (pydantic)
    • _db  – a LanceDB connection rooted at DB_ROOT
    • feedback_tbl – the open “phi3_feedback” table
    • append_feedback(dict|BaseModel)
    • log_run(OrchestratorRunSchema)
    • log_hermes_score(run_id, proposal_id, score, rationale=None)
    • cli_main(argv=list[str])  – simple sub-commands used by tests
"""

from __future__ import annotations
import datetime as _dt
import os
import sys
import uuid
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, Field
import lancedb
from lancedb.pydantic import LanceModel

###############################################################################
# ----------  constants / connection set-up  ----------------------------------
###############################################################################

_DB_ENV = os.getenv("LANCEDB_DIR")  # tests may set this
DB_ROOT = Path(_DB_ENV) if _DB_ENV else Path.cwd() / "lancedb_data"
DB_ROOT.mkdir(parents=True, exist_ok=True)

# **Do NOT** monkey-patch lancedb.connect – tests themselves monkey-patch it.
_db = lancedb.connect(str(DB_ROOT))

###############################################################################
# ----------  table schemas  --------------------------------------------------
###############################################################################

class Phi3FeedbackSchema(LanceModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    feedback_type: str
    feedback_content: Any = Field(exclude=True)
    timestamp: str = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    schema_version: str = Field(default="1.0")
    corrected_proposal: Optional[Dict[str, Any]] = Field(default=None, exclude=True)


class OrchestratorRunSchema(LanceModel):
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: str = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    input_query: str
    status: str
    final_output: str
    error_message: Optional[str] = None


class HermesScoreSchema(LanceModel):
    run_id: uuid.UUID
    proposal_id: str
    timestamp: str = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    score: float
    rationale: Optional[str] = None

###############################################################################
# ----------  bootstrap tables  ----------------------------------------------
###############################################################################

_tables: dict[str, Any] = {}


def _open_or_create(name: str, schema) -> Any:
    if name in _db.table_names():
        return _db.open_table(name)
    return _db.create_table(name, schema=schema, mode="create")


_tables["phi3_feedback"] = _open_or_create("phi3_feedback", Phi3FeedbackSchema)
_tables["orchestrator_runs"] = _open_or_create("orchestrator_runs", OrchestratorRunSchema)
_tables["hermes_scores"] = _open_or_create("hermes_scores", HermesScoreSchema)

feedback_tbl = _tables["phi3_feedback"]  # tests patch this symbol directly

###############################################################################
# ----------  helper functions  ----------------------------------------------
###############################################################################


def append_feedback(item: Phi3FeedbackSchema | Dict[str, Any]) -> None:
    """Add a feedback row (the tests patch this)."""
    if isinstance(item, BaseModel):
        item = item.model_dump()
    item.setdefault("schema_version", "1.0")
    feedback_tbl.add([item])


def log_run(run: OrchestratorRunSchema) -> None:
    _tables["orchestrator_runs"].add([run.model_dump()])


def log_hermes_score(
    run_id: uuid.UUID,
    proposal_id: str,
    score: float,
    rationale: Optional[str] = None,
) -> None:
    row = HermesScoreSchema(
        run_id=run_id,
        proposal_id=proposal_id,
        score=score,
        rationale=rationale,
    )
    _tables["hermes_scores"].add([row.model_dump()])

###############################################################################
# -------------  super-tiny CLI  (used by tests/test_db.py)  ------------------
###############################################################################


def _print_runs(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("No logs found in 'orchestrator_runs'.")
        return
    header = list(rows[0].keys())
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r[k]) for k in header))


def cli_main(argv: list[str] | None = None) -> None:
    import argparse

    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="llm_sidecar.db")
    sub = parser.add_subparsers(dest="cmd")

    # query-runs
    q = sub.add_parser("query-runs")
    q.add_argument("--last-n", type=int, default=10)

    args = parser.parse_args(argv)

    if args.cmd == "query-runs":
        tbl = _tables["orchestrator_runs"]
        rows = tbl.search().limit(args.last_n).to_list()  # type: ignore[attr-defined]
        _print_runs(rows)
        sys.exit(0)

    parser.print_help()
    sys.exit(0)
