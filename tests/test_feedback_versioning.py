"""
Tests for feedback submission, versioning, harvesting, and migration scripts.

This file contains tests to ensure that feedback data is handled
correctly throughout its lifecycle, including:
- Schema versioning upon submission.
- Harvesting recent feedback based on specific criteria.
- Data migration for older records to a new schema.
"""
import pytest
import lancedb
import datetime
import json
import uuid
import sys
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union

# --- Main Application Imports ---
# Imports the Pydantic model and server function for feedback submission.
from llm_sidecar.server import FeedbackItem, submit_phi3_feedback

# --- Script Imports for Testing ---
# Imports the main functions from the utility scripts to be tested.
from osiris.scripts.harvest_feedback import main as harvest_main
from osiris.scripts.migrate_feedback import main as migrate_main


# --- Helper Pydantic Models for Test Data ---

class FeedbackSchemaForHarvestTest(BaseModel):
    """
    A Pydantic schema used to create structured data for the
    `test_harvest_feedback_py_filters_by_version` test. It includes
    all fields the harvest script might query.
    """
    transaction_id: str
    timestamp: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    corrected_proposal: Optional[str] = None  # Stored as JSON string
    schema_version: str
    when: int  # Nanosecond timestamp for time-based queries


class TempSchemaForMigrationTest(BaseModel):
    """
    A temporary Pydantic schema representing the structure of "old" feedback data
    before it has been migrated. Note that `schema_version` is Optional to
    simulate records that are missing this field.
    """
    transaction_id: str
    timestamp: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: Optional[str] = None


# --- Test Functions ---

@pytest.mark.asyncio
async def test_feedback_item_model_defaults_version():
    """
    Tests that the FeedbackItem Pydantic model correctly assigns a default
    schema_version of "1.0" when not provided, and respects a custom
    version when it is provided.
    """
    # Test case 1: Default version
    item1 = FeedbackItem(
        transaction_id=str(uuid.uuid4()),
        feedback_type="rating",
        feedback_content={"score": 5},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    assert item1.schema_version == "1.0"

    # Test case 2: Custom version override
    item2 = FeedbackItem(
        transaction_id=str(uuid.uuid4()),
        feedback_type="qualitative_comment",
        feedback_content={"comment": "Great!"},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        schema_version="custom_test_version"
    )
    assert item2.schema_version == "custom_test_version"


@pytest.mark.asyncio
async def test_submit_phi3_feedback_stores_version(mocker):
    """
    Tests that the `submit_phi3_feedback` server function correctly handles
    the schema_version field. It should use the default "1.0" if not provided
    and preserve a custom version if one is passed in the FeedbackItem.
    """
    # Patch the underlying database append function to intercept its input
    mocked_append = mocker.patch("llm_sidecar.db.append_feedback")

    # Test Case 1: Default schema_version
    feedback_item_default = FeedbackItem(
        transaction_id="tid1",
        feedback_type="correction",
        feedback_content="some content",
        timestamp="ts1",
    )
    await submit_phi3_feedback(feedback_item_default)

    mocked_append.assert_called_once()
    call_args_default = mocked_append.call_args[0][0]

    # Verify the payload is a dictionary and has the correct default schema version.
    assert isinstance(call_args_default, dict)
    assert call_args_default.get("schema_version") == "1.0"
    assert call_args_default.get("transaction_id") == "tid1"

    # Reset the mock for the next test case.
    mocked_append.reset_mock()

    # Test Case 2: Custom schema_version
    feedback_item_custom = FeedbackItem(
        transaction_id="tid2",
        feedback_type="rating",
        feedback_content={"score": 1},
        timestamp="ts2",
        schema_version="custom_v2",
    )
    await submit_phi3_feedback(feedback_item_custom)

    mocked_append.assert_called_once()
    call_args_custom = mocked_append.call_args[0][0]

    # Verify the payload is a dictionary and that the custom schema version was preserved.
    assert isinstance(call_args_custom, dict)
    assert call_args_custom.get("schema_version") == "custom_v2"
    assert call_args_custom.get("transaction_id") == "tid2"


@pytest.mark.asyncio
async def test_harvest_feedback_py_filters_by_version(tmp_path_factory, monkeypatch):
    """
    Integration-like test for the `scripts/harvest_feedback.py` script.
    This test verifies that the script correctly filters records from LanceDB
    based on the --schema-version, --days-back, and feedback_type criteria.
    """
    # Setup temporary paths for the database and output file
    db_path = tmp_path_factory.mktemp("lancedb_harvest_test")
    output_file = db_path / "harvested_data.jsonl"
    db = lancedb.connect(db_path)
    table_name = "phi3_feedback"

    # --- Test Data Setup ---
    now_ns = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)
    old_ns = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)).timestamp() * 1e9)

    # Define a set of records with varying attributes to test filtering logic
    records_to_add = [
        # Should be harvested: version 1.0, type 'correction', recent
        {"transaction_id": "rec1_v1_correct_recent", "schema_version": "1.0", "feedback_type": "correction", "when": now_ns},
        # Should NOT be harvested: wrong version
        {"transaction_id": "rec2_v0.9_correct_recent", "schema_version": "0.9", "feedback_type": "correction", "when": now_ns},
        # Should NOT be harvested: wrong feedback_type
        {"transaction_id": "rec3_v1_other_recent", "schema_version": "1.0", "feedback_type": "other_type", "when": now_ns},
        # Should NOT be harvested: too old
        {"transaction_id": "rec4_v1_correct_old", "schema_version": "1.0", "feedback_type": "correction", "when": old_ns},
        # Should be harvested: version 1.0.1 (matches "1.0" prefix), recent, correct type
        {"transaction_id": "rec5_v1.0.1_correct_recent", "schema_version": "1.0.1", "feedback_type": "correction", "when": now_ns},
    ]
    # Add common data to all records
    for rec in records_to_add:
        rec.update({
            "feedback_content": "test content",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "corrected_proposal": json.dumps({"action": "adjust", "ticker": "TEST"})
        })

    # --- Create and Populate LanceDB Table ---
    # This block robustly creates the table, handling potential differences in LanceDB versions
    try:
        table = db.create_table(table_name, schema=FeedbackSchemaForHarvestTest, mode="overwrite")
        table.add(records_to_add)
    except Exception:
        if table_name in db.table_names():
            db.drop_table(table_name)
        table = db.create_table(table_name, data=records_to_add, mode="overwrite")

    # --- Mock Dependencies for the Script ---
    monkeypatch.setattr(lancedb, "connect", lambda path: db)
    # Mock sys.argv to simulate running the script from the command line
    test_argv = [
        "scripts/harvest_feedback.py",
        "--days-back", "1",
        "--schema-version", "1.0",
        "--out", str(output_file),
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    # --- Execute the Harvest Script ---
    harvest_main()

    # --- Assertions ---
    assert output_file.exists(), "Harvest script did not create an output file."
    harvested_records = []
    with open(output_file, "r") as f:
        for line in f:
            harvested_records.append(json.loads(line))

    assert len(harvested_records) == 2, "Harvest script did not filter records correctly."
    harvested_tids = {rec['transaction_id'] for rec in harvested_records}
    expected_tids = {"rec1_v1_correct_recent", "rec5_v1.0.1_correct_recent"}
    assert harvested_tids == expected_tids


@pytest.mark.asyncio
async def test_migrate_feedback_py_script(tmp_path_factory, monkeypatch):
    """
    Integration-like test for the `scripts/migrate_feedback.py` script.
    This test verifies that the migration script correctly identifies records
    with missing or outdated schema versions and updates them to "1.0".
    """
    db_path = tmp_path_factory.mktemp("lancedb_migrate_test")
    db_conn = lancedb.connect(db_path)
    table_name = "phi3_feedback"

    # --- Test Data Setup ---
    # Record A: Simulates a very old record with no schema_version field.
    record_A_dict = {
        "transaction_id": "a", "timestamp": "ts_a", "feedback_type": "type_a",
        "feedback_content": "content_a", "corrected_proposal": {"key": "val_a"},
    }
    # Record B: Simulates an older record with an outdated schema version.
    record_B_dict = {
        "transaction_id": "b", "timestamp": "ts_b", "feedback_type": "type_b",
        "feedback_content": "content_b", "corrected_proposal": {"key": "val_b"},
        "schema_version": "0.8",
    }
    # Record C: Simulates a modern record that should not be changed.
    record_C_dict = {
        "transaction_id": "c", "timestamp": "ts_c", "feedback_type": "type_c",
        "feedback_content": "content_c", "corrected_proposal": {"key": "val_c"},
        "schema_version": "1.0",
    }

    if table_name in db_conn.table_names():
        db_conn.drop_table(table_name)
        
    # Create the initial table, letting LanceDB infer the schema from data
    # that includes records with and without the 'schema_version' field.
    table = db_conn.create_table(
        table_name,
        data=[record_A_dict, record_B_dict, record_C_dict],
        mode="overwrite",
    )
    
    # --- Mocking Dependencies for the Script ---
    monkeypatch.setattr("osiris.scripts.migrate_feedback.lancedb.connect", lambda path: db_conn)
    monkeypatch.setattr("osiris.scripts.migrate_feedback.os.makedirs", lambda path, exist_ok=False: None, raising=False)

    # --- Execute the Migration Script ---
    migrate_main()

    # --- Assertions ---
    migrated_table = db_conn.open_table(table_name)
    results = migrated_table.search().to_list()
    assert len(results) == 3

    # Check that ALL records now have the correct schema_version "1.0".
    for record in results:
        assert record.get("schema_version") == "1.0", f"Record {record.get('transaction_id')} was not migrated correctly."

