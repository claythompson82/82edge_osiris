import datetime
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import pytest
import llm_sidecar.db as lls_db
from llm_sidecar.db import Phi3FeedbackSchema


@pytest.fixture
def setup_temp_db(monkeypatch, tmp_path):
    """A pytest fixture to create a temporary, isolated LanceDB for testing."""
    new_db_root = tmp_path / "lancedb_test"
    new_db_root.mkdir()
    
    # Connect to the temporary database
    db_connection = lls_db.lancedb.connect(new_db_root)
    monkeypatch.setattr(lls_db, "_db", db_connection)
    
    # Create the specific table needed for the test
    schema = lls_db.Phi3FeedbackSchema
    table = db_connection.create_table("phi3_feedback", schema=schema, mode="overwrite")
    
    # Set the global table variable in the module for the test to use
    monkeypatch.setattr(lls_db, "feedback_tbl", table)
    
    yield db_connection # The test can use this if needed, but the patch is the main thing
    
    # Teardown is handled by tmp_path fixture


def test_feedback_append(setup_temp_db):
    """Ensure feedback can be appended when the DB is correctly initialised."""
    # The setup_temp_db fixture ensures feedback_tbl is a valid table object
    
    initial_count = 0
    try:
        # Use the global feedback_tbl which is patched by the fixture
        initial_count = lls_db.feedback_tbl.to_lance().count_rows()
    except Exception:
        pass

    # Create an instance of the Pydantic model
    feedback_instance = Phi3FeedbackSchema(
        transaction_id="test123",
        feedback_type="rating",
        feedback_content="✅",
    )

    # The original code had a bug using model_dump which may not exist. 
    # .dict() is a safer alternative for Pydantic v1/v2 compatibility.
    lls_db.feedback_tbl.add([feedback_instance.dict()])

    # Verify the row was added
    final_count = lls_db.feedback_tbl.to_lance().count_rows()
    assert final_count > initial_count, "Row count should increase after adding feedback."


# Sample run data for mocking
mock_run_data = [
    {
        "run_id": "run1",
        "timestamp": "2023-01-01T10:00:00Z",
        "input_query": "query1",
        "status": "SUCCESS",
        "final_output": json.dumps({"data": "output1"}),
    },
    {
        "run_id": "run2",
        "timestamp": "2023-01-01T11:00:00Z",
        "input_query": "query2",
        "status": "FAILURE",
        "error_message": "err2",
        "final_output": json.dumps({"data": "output2"}),
    },
    {
        "run_id": "run3",
        "timestamp": "2023-01-01T12:00:00Z",
        "input_query": "query3",
        "status": "SUCCESS",
        "final_output": json.dumps({"data": "output3"}),
    },
]


class TestDbCLI(unittest.TestCase):

    def _setup_mock_osiris_table(self, mock_lancedb_connect, sample_data):
        mock_db_connection = MagicMock()
        mock_lancedb_connect.return_value = mock_db_connection

        mock_osiris_table = MagicMock()
        mock_lance_dataset = MagicMock()
        mock_arrow_table = MagicMock()

        mock_lance_dataset.to_table.return_value = mock_arrow_table
        mock_arrow_table.to_pylist.return_value = sample_data
        mock_osiris_table.to_lance.return_value = mock_lance_dataset
        
        return mock_osiris_table

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_default_last_n(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)
        with patch.dict(lls_db._tables, {"orchestrator_runs": mock_table}):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs"])

            output = captured_output.getvalue()
            self.assertEqual(output.count("run_id"), len(mock_run_data))


    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_no_command_shows_help(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, [])
        with patch.dict(lls_db._tables, {"orchestrator_runs": mock_table}):
            captured_output_stdout = io.StringIO()
            with redirect_stdout(captured_output_stdout), self.assertRaises(SystemExit) as cm:
                lls_db.cli_main([])

            output_stdout = captured_output_stdout.getvalue()
            # Pytest changes the program name in the usage string
            self.assertIn("usage: pytest [-h]", output_stdout)
            self.assertEqual(cm.exception.code, 0)


    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_invalid_command_shows_help(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, [])
        with patch.dict(lls_db._tables, {"orchestrator_runs": mock_table}):
            captured_output_stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(captured_output_stderr), self.assertRaises(SystemExit) as cm:
                lls_db.cli_main(["invalid-command"])

            output_stderr = captured_output_stderr.getvalue()
            self.assertIn("usage: pytest [-h]", output_stderr)
            self.assertIn("invalid choice: 'invalid-command'", output_stderr)
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()