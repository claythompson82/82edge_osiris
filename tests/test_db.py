from llm_sidecar.db import append_feedback, feedback_tbl
import datetime # Added for timestamp construction

def test_feedback_append():
    # Ensure the table is clean before testing, or that tests account for existing data.
    # For simplicity, we'll assert count increases. A more robust test might clear the table
    # or use a dedicated test table if the main table is persistent across tests.
    initial_count = 0
    try:
        initial_count = feedback_tbl.count_rows()
    except Exception: # Handle case where table might not exist before first append or is empty
        pass

    row = {
        "transaction_id": "test123",
        "timestamp": datetime.datetime.utcnow().isoformat(), # Using actual timestamp
        "feedback_type": "rating",
        "feedback_content": "✅"
    }
    append_feedback(row)
    
    # Verify the row was added
    final_count = feedback_tbl.count_rows()
    assert final_count > initial_count, "Row count should increase after adding feedback."

    # Optional: Query the table to verify the content of the added row.
    # This requires knowing more about how to query LanceDB, e.g., using SQL or a vector search.
    # For now, just checking the count as per the issue's test structure.
    # Example of a more specific check if data retrieval is straightforward:
    # result_df = feedback_tbl.search().limit(final_count).to_df() # Get all rows
    # assert not result_df[result_df['transaction_id'] == 'test123'].empty, "Test row not found by transaction_id"

import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout
import json
from llm_sidecar import db as lls_db # To call lls_db.cli_main()

# Sample run data for mocking
mock_run_data = [
    {"run_id": "run1", "timestamp": "2023-01-01T10:00:00Z", "input_query": "query1", "status": "SUCCESS", "final_output": {"data": "output1"}},
    {"run_id": "run2", "timestamp": "2023-01-01T11:00:00Z", "input_query": "query2", "status": "FAILURE", "error_message": "err2", "final_output": {"data": "output2"}},
    {"run_id": "run3", "timestamp": "2023-01-01T12:00:00Z", "input_query": "query3", "status": "SUCCESS", "final_output": {"data": "output3"}},
    {"run_id": "run4", "timestamp": "2023-01-01T13:00:00Z", "input_query": "query4", "status": "SUCCESS", "final_output": {"data": "output4"}},
    {"run_id": "run5", "timestamp": "2023-01-01T14:00:00Z", "input_query": "query5", "status": "SUCCESS", "final_output": {"data": "output5"}},
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
        return mock_osiris_table

    @patch('llm_sidecar.db.lancedb.connect') # Patch where connect is called in db.py
    def test_query_runs_default_last_n(self, mock_lancedb_connect):
        # Setup mock for osiris_runs_tbl to return our sample data
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs"]) # Default N is 5
            
            output = captured_output.getvalue()
            # Expect 5 runs (all of them, sorted by timestamp desc)
            self.assertEqual(output.count("run_id"), 5)
            self.assertTrue(output.find("run5") < output.find("run1")) # run5 is latest

    @patch('llm_sidecar.db.lancedb.connect')
    def test_query_runs_custom_last_n(self, mock_lancedb_connect):
         with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "2"])
            
            output = captured_output.getvalue()
            self.assertEqual(output.count("run_id"), 2)
            self.assertIn("run5", output) # Latest
            self.assertIn("run4", output) # Second latest
            self.assertNotIn("run3", output)

    @patch('llm_sidecar.db.lancedb.connect')
    def test_query_runs_more_than_available(self, mock_lancedb_connect):
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data[:3])): # Only 3 runs available
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "5"]) # Request 5
            
            output = captured_output.getvalue()
            # Should show all 3 available runs
            self.assertEqual(output.count("run_id"), 3)

    @patch('llm_sidecar.db.lancedb.connect')
    def test_query_runs_empty_table(self, mock_lancedb_connect):
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, [])): # No runs
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs"])
            
            output = captured_output.getvalue()
            self.assertIn("No run logs found.", output)
            self.assertEqual(output.count("run_id"), 0)

    @patch('llm_sidecar.db.lancedb.connect')
    def test_query_runs_zero_last_n(self, mock_lancedb_connect):
        # --last 0 should show all runs
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, mock_run_data)):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                lls_db.cli_main(["query-runs", "--last", "0"])
            
            output = captured_output.getvalue()
            self.assertEqual(output.count("run_id"), len(mock_run_data)) # All runs

    @patch('llm_sidecar.db.lancedb.connect')
    def test_no_command_shows_help(self, mock_lancedb_connect):
        # This doesn't strictly need the table mock, but safe to keep consistent
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, [])):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                # Suppress SystemExit that argparse calls on --help or no command
                with self.assertRaises(SystemExit) as cm:
                    lls_db.cli_main([]) # No command
            
            output = captured_output.getvalue()
            self.assertIn("usage: __main__.py [-h] {query-runs}", output) # argparse output uses program name
            self.assertIn("Available commands", output)
            # Check that the exit code is 0 for help display
            self.assertEqual(cm.exception.code, 0)

    @patch('llm_sidecar.db.lancedb.connect')
    def test_invalid_command_shows_help(self, mock_lancedb_connect):
        with patch('llm_sidecar.db.osiris_runs_tbl', self._setup_mock_osiris_table(mock_lancedb_connect, [])):
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                 with self.assertRaises(SystemExit) as cm:
                    lls_db.cli_main(["invalid-command"])
            
            output = captured_output.getvalue()
            # argparse for invalid command prints error to stderr and help to stdout (or stderr depending on version/config)
            # For this test, we'll check that it tries to print help.
            # The actual error message for "invalid choice" goes to stderr, which we are not capturing here.
            # We are checking that it attempts to print help (which it does before exiting).
            # The exit code for argument errors is typically 2.
            self.assertIn("usage: __main__.py [-h] {query-runs}", output)
            self.assertEqual(cm.exception.code, 2)

    @patch('llm_sidecar.db.lancedb.connect')
    @patch('llm_sidecar.db.osiris_runs_tbl', None) # Simulate table not being initialized
    def test_query_runs_table_not_initialized(self, mock_lancedb_connect):
        # No need to use _setup_mock_osiris_table here, we are patching osiris_runs_tbl to be None
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            lls_db.cli_main(["query-runs"])
        
        output = captured_output.getvalue()
        self.assertIn("Error: osiris_runs table is not initialized.", output)

# To run these tests if this file is executed directly (though typically run via pytest or unittest runner)
if __name__ == '__main__':
    unittest.main()
