import os
import sys
import json
import unittest
from unittest.mock import patch, mock_open
import pyarrow as pa
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Add project root to sys.path to allow for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set, we can import the script
from osiris.scripts import harvest_feedback

class TestHarvestFeedback(unittest.TestCase):

    def setUp(self):
        """Set up a mock database and sample data for tests."""
        self.db_path = "/tmp/test_harvest_db"
        self.db = lancedb.connect(self.db_path)
        self.table_name = "phi3_feedback"
        
        # Define a schema that matches what the script might expect
        class Feedback(LanceModel):
            id: str
            feedback_type: str
            assessment: str
            proposal: str
            corrected_proposal: str
            when: str
            user_id: str

        try:
            self.table = self.db.create_table(self.table_name, schema=Feedback, mode="overwrite")
        except Exception:
            # Fallback for existing table
            self.table = self.db.open_table(self.table_name)

        self.sample_data = [
            {
                "id": "1", "feedback_type": "rating", "assessment": "good", "proposal": "p1", 
                "corrected_proposal": "", "when": "2025-06-07T12:00:00Z", "user_id": "user1"
            },
            {
                "id": "2", "feedback_type": "correction", "assessment": "", "proposal": "p2",
                "corrected_proposal": '{"action": "BUY"}', "when": "2025-06-07T13:00:00Z", "user_id": "user1"
            }
        ]
        if self.table.count_rows() == 0:
             self.table.add(self.sample_data)


    def test_default_extraction(self):
        """Test that the script runs and extracts data."""
        output_jsonl = os.path.join(self.db_path, "output.jsonl")
        args = ["--db-path", self.db_path, "--output", output_jsonl]
        
        with patch('osiris.scripts.harvest_feedback.lancedb.connect') as mock_connect:
            mock_connect.return_value = self.db
            harvest_feedback.main(args)

        self.assertTrue(os.path.exists(output_jsonl))
        with open(output_jsonl, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0, "Output file should not be empty")
