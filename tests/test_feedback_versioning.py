import asyncio
import json
import pathlib
import tempfile
import importlib
import datetime
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

import pytest
import lancedb
from lancedb.pydantic import LanceModel
from pydantic import BaseModel

# Correctly import the modules we need to test
import osiris.llm_sidecar.db as db
from osiris.server import submit_phi3_feedback, FeedbackItem
from osiris.scripts.harvest_feedback import main as harvest_main
from osiris.scripts.migrate_feedback import main as migrate_main

# Define a Pydantic model for testing that matches the data structure
class FeedbackSchema(LanceModel):
    transaction_id: str
    timestamp: str
    feedback_type: str
    feedback_content: Any
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"
    # The 'when' field is used by the harvest script
    when: Optional[datetime.datetime] = None


@pytest.fixture
def temp_db():
    """Create a temporary LanceDB for the duration of a test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = pathlib.Path(tmpdir)
        db_conn = lancedb.connect(db_path)
        yield db_conn

def test_lancedb_table_creation_with_schema(temp_db):
    """Tests that a table can be created with a Pydantic schema."""
    table_name = "test_table"
    # The correct way to create a table with a Pydantic schema
    table = temp_db.create_table(table_name, schema=FeedbackSchema, mode="overwrite")
    assert table.name == table_name
    assert "transaction_id" in table.schema.names

def test_feedback_submission_and_retrieval(temp_db, mocker):
    """Tests the full feedback loop from submission to harvesting."""
    # Mock the global DB connection used by the server and scripts
    mocker.patch("osiris.llm_sidecar.db._db", temp_db)
    
    # 1. Create the table using the correct schema
    table_name = "phi3_feedback"
    table = temp_db.create_table(table_name, schema=FeedbackSchema, mode="overwrite")
    mocker.patch.dict(db._tables, {table_name: table})

    # 2. Submit feedback
    feedback_item = FeedbackItem(
        transaction_id="tid123",
        feedback_type="correction",
        feedback_content={"old": "a", "new": "b"},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corrected_proposal={"action": "BUY"}
    )
    
    # Directly call the function to avoid server complexity
    async def run_submission():
        await submit_phi3_feedback(feedback_item)

    asyncio.run(run_submission())
    
    # 3. Verify feedback was added
    assert table.count_rows() == 1

    # 4. Run the harvest script
    output_file = temp_db._uri / "harvested_data.jsonl"
    harvest_args = ["--db-path", temp_db._uri, "--output", str(output_file)]
    
    # The script uses a different lancedb.connect call, so we patch that too
    with patch('osiris.scripts.harvest_feedback.lancedb.connect', return_value=temp_db):
        harvest_main(harvest_args)

    # 5. Verify the harvest file was created and has content
    assert output_file.exists()
    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['id'] == 'tid123'
