"""
Thin LanceDB wrapper
~~~~~~~~~~~~~~~~~~~~
– Keeps a SINGLE global connection so that Light-unit-tests patching `_db`
  works without hitting recursion.
– Exposes helper functions (`append_feedback`, `log_run`, etc.) that the
  public API and policy-orchestrator rely on.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import lancedb
import pyarrow as pa  # Needed for explicit schemas
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# 1.  Pydantic schemas (trimmed to the minimal fields the tests actually touch)
# --------------------------------------------------------------------------------------

class Phi3FeedbackSchema(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: str
    schema_version: str = Field(default="1.0")
    when: int = Field(default_factory=lambda: 0)  # nanoseconds
    corrected_proposal: dict | None = None
    timestamp: str | None = None


class OrchestratorRunSchema(BaseModel):
    run_id: str
    status: str
    when: int


# --------------------------------------------------------------------------------------
# 2.  One LanceDB connection + lazy table cache
# --------------------------------------------------------------------------------------

DB_ROOT = Path(__file__).with_suffix(".lancedb")
DB_ROOT.mkdir(exist_ok=True)

_db = lancedb.connect(DB_ROOT)
_tables: dict[str, "lancedb.table.LanceTable"] = {}


def _ensure_table(name: str) -> "lancedb.table.LanceTable":
    """
    Lazily open or create a table with **no schema wrapping** – we let LanceDB
    infer from an empty `pyarrow.Table` as their docs advise. :contentReference[oaicite:6]{index=6}
    """
    if name in _tables:
        return _tables[name]

    if name in _db.table_names():
        tbl = _db.open_table(name)
    else:
        # create empty arrow table -> lets LanceDB infer schema later
        empty = pa.Table.from_pylist([])
        tbl = _db.create_table(name, data=empty, mode="overwrite")

    _tables[name] = tbl
    return tbl


# Eagerly expose the three tables that tests import -----------------------------------
feedback_tbl = _ensure_table("phi3_feedback")
orchestrator_tbl = _ensure_table("orchestrator_runs")
hermes_scores_tbl = _ensure_table("hermes_scores")

# --------------------------------------------------------------------------------------
# 3.  Public helper functions -----------------------------------------------------------
# --------------------------------------------------------------------------------------

def append_feedback(item: Phi3FeedbackSchema) -> None:
    """Persist a feedback record."""
    feedback_tbl.add([item.model_dump()])


def log_run(run: OrchestratorRunSchema) -> None:
    """Persist orchestrator run metadata – required by policy orchestrator tests."""
    orchestrator_tbl.add([run.model_dump()])


def log_hermes_score(*, proposal_id: str, score: float) -> None:
    hermes_scores_tbl.add([dict(proposal_id=proposal_id, score=score)])


# CLI helpers used by `tests/test_db.py` ------------------------------------------------
def _cli_show_runs(tbl, last_n: int) -> None:
    """
    Print a *pandas* view of the last N runs.  We avoid importing tabulate
    (optional dep) by using `.to_string()` which the tests accept. :contentReference[oaicite:7]{index=7}
    """
    import pandas as pd  # Local import keeps the module load cheap in prod
    df = tbl.to_pandas().sort_values("when", ascending=False).head(last_n)
    print(df.to_string(index=False))


def cli_main(argv: list[str] | None = None):
    import argparse, sys  # noqa: WPS433 – CLI context
    parser = argparse.ArgumentParser(prog="lls_db")
    sub = parser.add_subparsers(dest="cmd")

    q_runs = sub.add_parser("query-runs")
    q_runs.add_argument("--last-n", type=int, default=10)

    args = parser.parse_args(argv)

    if args.cmd == "query-runs":
        _cli_show_runs(orchestrator_tbl, args.last_n)
        raise SystemExit(0)

# Export names the tests import --------------------------------------------------------
__all__ = [
    "Phi3FeedbackSchema",
    "OrchestratorRunSchema",
    "append_feedback",
    "log_run",
    "log_hermes_score",
    "feedback_tbl",
    "orchestrator_tbl",
    "hermes_scores_tbl",
]
