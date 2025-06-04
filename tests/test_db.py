from llm_sidecar.db import append_feedback, feedback_tbl, Phi3FeedbackSchema
import datetime  # Added for timestamp construction
import json  # For serializing dicts in mock_run_data


def test_feedback_append():
    # Ensure the table is clean before testing, or that tests account for existing data.
    # For simplicity, we'll assert count increases. A more robust test might clear the table
    # or use a dedicated test table if the main table is persistent across tests.
    initial_count = 0
    try:
        initial_count = feedback_tbl.count_rows()
    except (
        Exception
    ):  # Handle case where table might not exist before first append or is empty
        pass

    # Create an instance of the Pydantic model
    feedback_instance = Phi3FeedbackSchema(
        transaction_id="test123",
        # timestamp will use default factory
        feedback_type="rating",
        feedback_content="✅",
        # schema_version will use default "1.0"
        # corrected_proposal will use default None
    )
    # We can manually set timestamp if we need to control it for the test,
    # otherwise the model's default_factory will handle it.
    # For this test, default factory is fine. If specific timestamp needed:
    # feedback_instance.timestamp = datetime.datetime.utcnow().isoformat()

    append_feedback(feedback_instance)

    # Verify the row was added
    final_count = feedback_tbl.count_rows()
    assert (
        final_count > initial_count
    ), "Row count should increase after adding feedback."

    # Optional: Query the table to verify the content of the added row.
    # This requires knowing more about how to query LanceDB, e.g., using SQL or a vector search.
    # For now, just checking the count as per the issue's test structure.
    # Example of a more specific check if data retrieval is straightforward:
    # result_df = feedback_tbl.search().limit(final_count).to_df() # Get all rows
    # assert not result_df[result_df['transaction_id'] == 'test123'].empty, "Test row not found by transaction_id"


import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout, redirect_stderr  # Added redirect_stderr
import json
from llm_sidecar import db as lls_db  # To call lls_db.cli_main()

# Sample run data for mocking
# final_output is now a JSON string
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
    {
        "run_id": "run4",
        "timestamp": "2023-01-01T13:00:00Z",
        "input_query": "query4",
        "status": "SUCCESS",
        "final_output": json.dumps({"data": "output4"}),
    },
    {
        "run_id": "run5",
        "timestamp": "2023-01-01T14:00:00Z",
        "input_query": "query5",
        "status": "SUCCESS",
        "final_output": json.dumps({"data": "output5"}),
    },
]


class TestDbCLI(unittest.TestCase):

    def _setup_mock_osiris_table(self, mock_lancedb_connect, sample_data):
        mock_db_connection = MagicMock()
        mock_lancedb_connect.return_value = mock_db_connection

        mock_osiris_table = MagicMock()
        # Mock the chain osiris_runs_tbl.to_lance().to_table().to_pylist()
        mock_lance_dataset = MagicMock()
        mock_arrow_table = MagicMock()

        mock_lance_dataset.to_table.return_value = mock_arrow_table
        mock_arrow_table.to_pylist.return_value = sample_data
        mock_osiris_table.to_lance.return_value = mock_lance_dataset

        # This global variable `osiris_runs_tbl` in db.py is what cli_main uses.
        # We patch it directly within the llm_sidecar.db module.
        # ALSO, we need to ensure cli_main uses this mock via the _tables dict
        # So, we patch _tables as well for the duration of the test.
        # The key is that lls_db.init_db() populates _tables.
        # If cli_main calls init_db(), it might overwrite our mock if connect isn't fully mocked.
        # A robust way is to ensure init_db itself is controlled or _tables is definitively set.

        # Let's patch _tables directly for tests that query.
        # And ensure init_db doesn't run with real connections during these CLI tests.
        return mock_osiris_table

    @patch("llm_sidecar.db.init_db")  # Prevent real init_db during CLI test
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_default_last_n(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs"])  # Default N is 5

            output = captured_output.getvalue()
            # Expect 5 runs (all of them, sorted by timestamp desc)
            self.assertEqual(output.count("run_id"), 5)
            self.assertTrue(output.find("run5") < output.find("run1"))  # run5 is latest

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_custom_last_n(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "2"])

            output = captured_output.getvalue()
            self.assertEqual(output.count("run_id"), 2)
            self.assertIn("run5", output)  # Latest
            self.assertIn("run4", output)  # Second latest
            self.assertNotIn("run3", output)

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_more_than_available(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(
            mock_lancedb_connect, mock_run_data[:3]
        )  # Only 3 runs
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "5"])  # Request 5

            output = captured_output.getvalue()
            # Should show all 3 available runs
            self.assertEqual(output.count("run_id"), 3)

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_empty_table(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, [])  # No runs
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs"])

            output = captured_output.getvalue()
            self.assertIn(
                "No logs found in 'orchestrator_runs'.", output
            )  # Updated expected message
            self.assertEqual(output.count("run_id"), 0)

    @patch("llm_sidecar.db.init_db")
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_zero_last_n(self, mock_lancedb_connect, mock_init_db):
        # --last 0 should show all runs
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "0"])

            output = captured_output.getvalue()
            self.assertEqual(output.count("run_id"), len(mock_run_data))  # All runs

    @patch("llm_sidecar.db.init_db")  # Prevent real init_db
    @patch("llm_sidecar.db.lancedb.connect")
    def test_no_command_shows_help(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, [])
        # For help, _tables needs to be populated for choices in args
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output_stdout = io.StringIO()
            # Argparse help for no command goes to stdout and exits with 0
            with (
                redirect_stdout(captured_output_stdout),
                self.assertRaises(SystemExit) as cm,
            ):
                lls_db.cli_main([])  # No command

            output_stdout = captured_output_stdout.getvalue()
            self.assertIn("usage: __main__.py [-h]", output_stdout)
            self.assertIn("Available commands", output_stdout)
            self.assertIn("query-runs", output_stdout)  # Check specific command
            self.assertEqual(cm.exception.code, 0)

    @patch("llm_sidecar.db.init_db")  # Prevent real init_db
    @patch("llm_sidecar.db.lancedb.connect")
    def test_invalid_command_shows_help(self, mock_lancedb_connect, mock_init_db):
        mock_table = self._setup_mock_osiris_table(mock_lancedb_connect, [])
        # For help, _tables needs to be populated for choices in args
        with patch.dict(
            lls_db._tables,
            {
                "orchestrator_runs": mock_table,
                "phi3_feedback": MagicMock(),
                "hermes_scores": MagicMock(),
            },
        ):
            captured_output_stderr = io.StringIO()
            # Argparse error for invalid command goes to stderr and exits with 2
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(captured_output_stderr),
                self.assertRaises(SystemExit) as cm,
            ):
                lls_db.cli_main(["invalid-command"])

            output_stderr = captured_output_stderr.getvalue()
            self.assertIn(
                "usage: __main__.py [-h]", output_stderr
            )  # argparse prints usage to stderr for errors
            self.assertIn("invalid choice: 'invalid-command'", output_stderr)
            self.assertEqual(cm.exception.code, 2)

    @patch("llm_sidecar.db.init_db")  # Prevent real init_db
    @patch("llm_sidecar.db.lancedb.connect")
    def test_query_runs_table_not_initialized(self, mock_lancedb_connect, mock_init_db):
        original_tables = lls_db._tables
        lls_db._tables = (
            {}
        )  # Directly set _tables to empty to make choices for --table empty
        try:
            captured_stderr = io.StringIO()
            with redirect_stderr(captured_stderr), self.assertRaises(SystemExit) as cm:
                lls_db.cli_main(["query-runs", "--table", "orchestrator_runs"])

            self.assertEqual(cm.exception.code, 2)
            stderr_output = captured_stderr.getvalue()
            self.assertIn(
                "argument --table: invalid choice: 'orchestrator_runs'", stderr_output
            )
            self.assertIn("(choose from )", stderr_output)  # choices are empty
        finally:
            lls_db._tables = original_tables  # Restore original _tables


# To run these tests if this file is executed directly (though typically run via pytest or unittest runner)
if __name__ == "__main__":
    unittest.main()
