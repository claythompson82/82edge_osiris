import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import lancedb
from lancedb.pydantic import LanceModel

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
            when: int
            user_id: str
            schema_version: str

        try:
            self.table = self.db.create_table(
                self.table_name, schema=Feedback, mode="overwrite"
            )
        except Exception:
            # Fallback for existing table
            self.table = self.db.open_table(self.table_name)

        self.sample_data = [
            {
                "id": "1",
                "feedback_type": "rating",
                "assessment": "good",
                "proposal": "p1",
                "corrected_proposal": "",
                "when": 1749297600000000000,
                "user_id": "user1",
                "schema_version": "1.0",
            },
            {
                "id": "2",
                "feedback_type": "correction",
                "assessment": "",
                "proposal": "p2",
                "corrected_proposal": '{"action": "BUY"}',
                "when": 1749301200000000000,
                "user_id": "user1",
                "schema_version": "1.0",
            },
        ]
        if self.table.count_rows() == 0:
            self.table.add(self.sample_data)

    def test_default_extraction(self):
        """Test that the script runs and extracts data."""
        output_jsonl = os.path.join(self.db_path, "output.jsonl")
        args = ["--out", output_jsonl]

        with (
            patch("osiris.scripts.harvest_feedback.lancedb.connect") as mock_connect,
            patch("os.path.exists", return_value=True),
        ):
            mock_connect.return_value = self.db
            with patch("sys.argv", ["harvest_feedback.py"] + args):
                harvest_feedback.main()

        self.assertTrue(os.path.exists(output_jsonl))
        with open(output_jsonl, "r") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0, "Output file should not be empty")
