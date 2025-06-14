import os
from unittest.mock import patch
from osiris.scripts import harvest_feedback

def test_default_extraction(tmp_path, monkeypatch):
    """
    Test the harvest_feedback script: reads from phi3_feedback, writes output JSONL.
    """
    output_jsonl = os.path.join(tmp_path, "output.jsonl")

    # Sample feedback record matching your schema (as a dict)
    sample_rows = [
        {
            "transaction_id": "test1",
            "feedback_type": "bug",
            "feedback_content": {"msg": "sample"},  # Correct type (dict, not string)
            "schema_version": "1.0",
            "timestamp": "2025-06-13T00:00:00Z"
        }
    ]

    class DummyTable:
        def to_arrow(self):
            import pyarrow as pa
            return pa.Table.from_pylist(sample_rows)

    class DummyDB:
        def __init__(self, tables):
            self._tables = tables
        def table(self, name):
            return DummyTable()

    # Patch lancedb.connect to use DummyDB with our sample data
    with patch("osiris.scripts.harvest_feedback.lancedb.connect", return_value=DummyDB({"phi3_feedback": sample_rows})):
        # Patch sys.argv for main()
        with patch("sys.argv", ["harvest_feedback.py", "--out", output_jsonl]):
            harvest_feedback.main()

    # Now verify the output file exists and contains at least one line
    assert os.path.exists(output_jsonl)
    with open(output_jsonl, "r") as f:
        lines = f.readlines()
    assert len(lines) > 0, "Output file should not be empty"
