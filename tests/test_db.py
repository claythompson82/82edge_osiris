import datetime
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import llm_sidecar.db as lls_db
from llm_sidecar.db import Phi3FeedbackSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def setup_temp_db(monkeypatch, tmp_path):
    """Create a temporary isolated LanceDB for testing."""
    new_db_root = tmp_path / "lancedb_test"
    new_db_root.mkdir()

    # Connect to the temporary database
    db_connection = lls_db.lancedb.connect(new_db_root)
    monkeypatch.setattr(lls_db, "_db", db_connection)

    # Create the table required for tests
    schema = lls_db.Phi3FeedbackSchema
    table = db_connection.create_table("phi3_feedback", schema=schema, mode="overwrite")

    # Swap the module-level table so app code points to this temp table
    monkeypatch.setattr(lls_db, "feedback_tbl", table)

    yield db_connection
    # tmp_path fixture cleans up automatically


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_feedback_append(setup_temp_db):
    """Ensure feedback rows can be inserted."""
    initial = lls_db.feedback_tbl.to_lance().count_rows()

    item = Phi3FeedbackSchema(
        transaction_id="tid",
        feedback_type="rating",
        feedback_content="✅",
    )
    lls_db.feedback_tbl.add([item.dict()])

    final = lls_db.feedback_tbl.to_lance().count_rows()
    assert final > initial, "Row count should increase after append."


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------
sample_run_data: Dict[str, Any] = [
    {
        "run_id": "run1",
        "timestamp": "2023-01-01T11:00:00Z",
        "input_query": "query1",
        "status": "SUCCESS",
        "error_message": None,
        "final_output": json.dumps({"data": "output1"}),
    },
    {
        "run_id": "run2",
        "timestamp": "2023-01-01T11:01:00Z",
        "input_query": "query2",
        "status": "FAILURE",
        "error_message": "err2",
        "final_output": json.dumps({"data": "output2"}),
    },
]


class TestDbCLI(unittest.TestCase):
    # ---------- helpers ----------------------------------------------------
    def _mock_table(self, mock_conn, rows):
        mockdb = MagicMock()
        mock_conn.return_value = mockdb

        mock_table = MagicMock()
        mock_dataset = MagicMock()
        mock_arrow = MagicMock()

        mock_dataset.to_table.return_value = mock_arrow
        mock_arrow.to_pylist.return_value = rows
        mock_table.to_lance_dataset.return_value = mock_dataset
        return mock_table

    # ---------- list runs --------------------------------------------------
    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_show_runs_default_last_n(self, mock_conn, _):
        tbl = self._mock_table(mock_conn, sample_run_data)
        with patch.dict(lls_db._tables, {"orchestrator_runs": tbl}):
            cap = io.StringIO()
            with redirect_stdout(cap):
                lls_db.cli_main(["query-runs"])
        self.assertIn("run_id", cap.getvalue())

    # ---------- help output (relaxed) --------------------------------------
    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_no_command_shows_help(self, mock_conn, _):
        tbl = self._mock_table(mock_conn, [])
        with patch.dict(lls_db._tables, {"orchestrator_runs": tbl}):
            cap = io.StringIO()
            with redirect_stdout(cap), self.assertRaises(SystemExit) as cm:
                lls_db.cli_main([])
        self.assertEqual(cm.exception.code, 0)

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_invalid_command_shows_help(self, mock_conn, _):
        tbl = self._mock_table(mock_conn, [])
        with patch.dict(lls_db._tables, {"orchestrator_runs": tbl}):
            cap_err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(cap_err), self.assertRaises(SystemExit) as cm:
                lls_db.cli_main(["invalid-command"])
        self.assertIn("invalid choice", cap_err.getvalue())
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

