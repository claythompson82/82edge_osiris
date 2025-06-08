"""
Tests for feedback submission, versioning, and migration scripts.
This file contains tests to ensure that feedback data is handled
correctly, including schema versioning upon submission and data
migration for older records.
"""
import pytest
import lancedb
from typing import Any, Dict, Optional
from pydantic import BaseModel

# Assume these are the correct import paths based on project structure.
# If these paths are incorrect, they should be adjusted to match the actual location
# of the FeedbackItem model, submit_phi3_feedback function, and migrate_main function.
from llm_sidecar.server import FeedbackItem, submit_phi3_feedback
from osiris.scripts.migrate_feedback import main as migrate_main

@pytest.mark.asyncio
async def test_submit_phi3_feedback_stores_version(mocker):
    """
    Tests that the `submit_phi3_feedback` function correctly handles the
    schema_version field. It should add schema_version="1.0" if it's not
    provided, and respect a custom version if it is.
    """
    # Patch the underlying database append function to intercept its input
    mocked_append = mocker.patch("llm_sidecar.db.append_feedback")

    # Test Case 1: Default schema_version
    # A FeedbackItem is created without a schema_version.
    feedback_item_default = FeedbackItem(
        transaction_id="tid1",
        feedback_type="correction",
        feedback_content="some content",
        timestamp="ts1",
    )
    # The function under test is called.
    await submit_phi3_feedback(feedback_item_default)

    # Assert that the mocked database function was called exactly once.
    mocked_append.assert_called_once()
    # Inspect the arguments passed to the mocked function.
    call_args_default = mocked_append.call_args[0][0]
    
    # Verify the payload is a dictionary and has the correct default schema version.
    assert isinstance(call_args_default, dict)
    assert call_args_default.get("schema_version") == "1.0"
    assert call_args_default.get("transaction_id") == "tid1"

    # Reset the mock for the next test case.
    mocked_append.reset_mock()

    # Test Case 2: Custom schema_version
    # A FeedbackItem is created with a custom schema_version.
    feedback_item_custom = FeedbackItem(
        transaction_id="tid2",
        feedback_type="rating",
        feedback_content={"score": 1},
        timestamp="ts2",
        schema_version="custom_v2",
    )
    # The function under test is called again.
    await submit_phi3_feedback(feedback_item_custom)

    # Assert that the mocked database function was called exactly once.
    mocked_append.assert_called_once()
    call_args_custom = mocked_append.call_args[0][0]

    # Verify the payload is a dictionary and that the custom schema version was preserved.
    assert isinstance(call_args_custom, dict)
    assert call_args_custom.get("schema_version") == "custom_v2"
    assert call_args_custom.get("transaction_id") == "tid2"


@pytest.mark.asyncio
async def test_migrate_feedback_py_script(tmp_path_factory, monkeypatch):
    """
    Integration-like test for the `scripts/migrate_feedback.py` script.
    This test verifies that the migration script correctly identifies records
    with missing or outdated schema versions and updates them to "1.0".
    """
    # Create a temporary directory for the LanceDB database for this test.
    db_path = tmp_path_factory.mktemp("lancedb_migrate_test")

    # Connect to the temporary database.
    db_conn = lancedb.connect(db_path)
    table_name = "phi3_feedback"  # The script is hardcoded to use this table name.

    # --- Test Data Setup ---
    # Record A: Simulates a very old record with no schema_version field.
    record_A_dict = {
        "transaction_id": "a",
        "timestamp": "ts_a",
        "feedback_type": "type_a",
        "feedback_content": "content_a",
        "corrected_proposal": {"key": "val_a"},
    }
    # Record B: Simulates an older record with an outdated schema version.
    record_B_dict = {
        "transaction_id": "b",
        "timestamp": "ts_b",
        "feedback_type": "type_b",
        "feedback_content": "content_b",
        "corrected_proposal": {"key": "val_b"},
        "schema_version": "0.8",
    }
    # Record C: Simulates a modern record that should not be changed.
    record_C_dict = {
        "transaction_id": "c",
        "timestamp": "ts_c",
        "feedback_type": "type_c",
        "feedback_content": "content_c",
        "corrected_proposal": {"key": "val_c"},
        "schema_version": "1.0",
    }
    
    # Clean up any pre-existing table.
    if table_name in db_conn.table_names():
        db_conn.drop_table(table_name)

    # Define a temporary Pydantic schema to create the initial table.
    # This is needed because LanceDB requires a schema to create a table.
    class TempSchemaForInitialData(BaseModel):
        transaction_id: str
        timestamp: str
        feedback_type: str
        feedback_content: Any
        corrected_proposal: Optional[Dict[str, Any]] = None
        schema_version: Optional[str] = None

    # Try creating the table with an explicit schema first.
    try:
        table = db_conn.create_table(
            table_name, schema=TempSchemaForInitialData, mode="overwrite"
        )
        table.add([record_A_dict, record_B_dict, record_C_dict])
    except Exception:
        # Fallback for LanceDB versions that prefer schema inference from data.
        if table_name in db_conn.table_names():
            db_conn.drop_table(table_name)
        table = db_conn.create_table(
            table_name,
            data=[record_A_dict, record_B_dict, record_C_dict],
            mode="overwrite",
        )

    # --- Mocking Dependencies for the Script ---
    # Mock lancedb.connect to return our temporary DB connection.
    def mock_lancedb_connect(path):
        return db_conn

    monkeypatch.setattr(
        "osiris.scripts.migrate_feedback.lancedb.connect", mock_lancedb_connect
    )
    # Mock os.makedirs to prevent the script from creating directories.
    monkeypatch.setattr(
        "osiris.scripts.migrate_feedback.os.makedirs",
        lambda path, exist_ok=False: None,
        raising=False,
    )

    # --- Execute the Migration Script ---
    migrate_main()

    # --- Assertions ---
    # Verify the results of the migration.
    migrated_table = db_conn.open_table(table_name)
    results = migrated_table.search().to_list()
    assert len(results) == 3

    # Check that all records now have the correct schema_version.
    for record in results:
        assert record.get("schema_version") == "1.0"
