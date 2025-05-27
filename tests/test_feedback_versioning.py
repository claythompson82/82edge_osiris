import pytest
import asyncio
import datetime
import json
import os
import uuid
import lancedb
import argparse
import sys
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from server import FeedbackItem, submit_phi3_feedback # For tests 1 and 2
from llm_sidecar.db import append_feedback as actual_append_feedback # For test 2, to mock its module
# For tests 3 and 4 (script testing)
from scripts.harvest_feedback import main as harvest_main
from scripts.migrate_feedback import main as migrate_main
from scripts.migrate_feedback import FeedbackSchemaWithVersion as MigrationOutputSchema # To check migrated data

# --- Helper Pydantic Models for Test Data & Table Creation ---

class FeedbackSchemaWithVersionForHarvestTest(BaseModel):
    transaction_id: str
    timestamp: str # ISO format
    feedback_type: str
    feedback_content: Any
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str
    when: int # Nanosecond timestamp, as used by harvest_feedback.py query

class FeedbackSchemaOldForMigrationTest(BaseModel):
    # Schema for data *before* migration, some fields might be missing if not required by old system
    # The migration script itself uses `FeedbackSchemaWithVersion` to create the *new* table.
    # This model is for defining the data we *insert* into the table *before* migration.
    transaction_id: str
    timestamp: str # ISO format
    feedback_type: str
    feedback_content: Any
    corrected_proposal: Optional[Dict[str, Any]] = None
    # schema_version is intentionally missing here for some test records
    # 'when' is also specific to harvest test, not strictly needed for migration test data structure
    # but if it exists in the actual DB, migrate should carry it over if not explicitly dropped.
    # The migration script implicitly carries over all existing fields.
    # For simplicity, we'll make migration test data simpler and not include 'when'.


# --- Test Functions ---

@pytest.mark.asyncio
async def test_feedback_item_model_defaults_version():
    """Tests that FeedbackItem model defaults schema_version and allows override."""
    item1 = FeedbackItem(
        transaction_id=str(uuid.uuid4()),
        feedback_type="rating",
        feedback_content={"score": 5},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    assert item1.schema_version == "1.0"

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
    """Tests that submit_phi3_feedback passes the schema_version to append_feedback."""
    # Mock llm_sidecar.db.append_feedback
    mocked_append = mocker.patch("llm_sidecar.db.append_feedback")

    # Test with default schema_version
    feedback_item_default = FeedbackItem(
        transaction_id="tid1",
        feedback_type="correction",
        feedback_content="some content",
        timestamp="ts1"
    )
    await submit_phi3_feedback(feedback_item_default)
    mocked_append.assert_called_once()
    call_args_default = mocked_append.call_args[0][0]
    assert isinstance(call_args_default, dict)
    assert call_args_default.get("schema_version") == "1.0"
    assert call_args_default.get("transaction_id") == "tid1"

    mocked_append.reset_mock()

    # Test with explicit schema_version
    feedback_item_custom = FeedbackItem(
        transaction_id="tid2",
        feedback_type="rating",
        feedback_content={"score": 1},
        timestamp="ts2",
        schema_version="custom_v2"
    )
    await submit_phi3_feedback(feedback_item_custom)
    mocked_append.assert_called_once()
    call_args_custom = mocked_append.call_args[0][0]
    assert isinstance(call_args_custom, dict)
    assert call_args_custom.get("schema_version") == "custom_v2"
    assert call_args_custom.get("transaction_id") == "tid2"


@pytest.mark.asyncio
async def test_harvest_feedback_py_filters_by_version(tmp_path_factory, monkeypatch):
    """Integration-like test for scripts/harvest_feedback.py filtering."""
    db_path = tmp_path_factory.mktemp("lancedb_harvest_test")
    output_file = tmp_path_factory.mktemp("output") / "harvested_data.jsonl"

    # Connect to temporary DB and create table with schema
    db = lancedb.connect(db_path)
    # The harvest script queries 'when', so the table needs it.
    # It also expects corrected_proposal to be filterable.
    # The FeedbackSchemaWithVersionForHarvestTest is suitable here.
    table_name = "phi3_feedback" # Script uses this name
    
    try:
        table = db.create_table(table_name, schema=FeedbackSchemaWithVersionForHarvestTest, mode="overwrite")
    except Exception as e:
        # Fallback for older lancedb that might not like schema on create_table with overwrite
        # or Pydantic schema directly. In that case, create then add with schema inference.
        if table_name in db.table_names(): # Ensure clean state
            db.drop_table(table_name)
        # LanceDB can infer schema from first batch of data if schema not passed or pydantic schema not directly usable
        # For test robustness, explicit schema is better. Assuming create_table with Pydantic schema works.
        # If not, this test would need adjustment for schema creation.
        # For now, let's assume it works.
        print(f"Warning: Initial table creation with schema failed: {e}. This might affect test if schema not inferred correctly.")
        # If create_table with schema fails, one might need to create without schema and add data,
        # relying on LanceDB's schema inference, which is less robust for tests.
        # pytest.fail(f"LanceDB table creation with schema failed: {e}") # Option to fail fast
        table = db.create_table(table_name, mode="overwrite") # Create without explicit Pydantic schema


    now_ns = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1_000_000_000)
    old_ns = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)).timestamp() * 1_000_000_000)

    common_data = {
        "feedback_content": "test content",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "corrected_proposal": {"action": "adjust", "ticker": "TEST"}, # Must be non-empty for filter
    }

    records_to_add = [
        FeedbackSchemaWithVersionForHarvestTest(transaction_id="rec1_v1_correct_recent", **common_data, schema_version="1.0", feedback_type="correction", when=now_ns).model_dump(),
        FeedbackSchemaWithVersionForHarvestTest(transaction_id="rec2_v0.9_correct_recent", **common_data, schema_version="0.9", feedback_type="correction", when=now_ns).model_dump(),
        FeedbackSchemaWithVersionForHarvestTest(transaction_id="rec3_v1_other_recent", **common_data, schema_version="1.0", feedback_type="other_type", when=now_ns).model_dump(), # Filtered by type
        FeedbackSchemaWithVersionForHarvestTest(transaction_id="rec4_v1_correct_old", **common_data, schema_version="1.0", feedback_type="correction", when=old_ns).model_dump(), # Filtered by date
        FeedbackSchemaWithVersionForHarvestTest(transaction_id="rec5_v1.0.1_correct_recent", **common_data, schema_version="1.0.1", feedback_type="correction", when=now_ns).model_dump(), # Should not be picked by --schema-version "1.0"
    ]
    if records_to_add:
         table.add(records_to_add)

    # Monkeypatch lancedb.connect to use the temporary DB
    def mock_lancedb_connect(path):
        assert str(path) == str(db_path) # Ensure script uses the path we expect
        return lancedb.connect(path) # Connect to the actual temp path
    monkeypatch.setattr(lancedb, "connect", mock_lancedb_connect)
    
    # Monkeypatch sys.argv for harvest_feedback.py
    # Target: --schema-version "1.0", --days-back 1 (to include 'now_ns'), --out output_file.jsonl
    # Default feedback_type is 'correction' in the script's query
    test_argv = [
        "scripts/harvest_feedback.py",
        "--days-back", "1",
        "--schema-version", "1.0",
        "--out", str(output_file),
        # "--max", "10" # optional
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    # Run the harvest script's main function
    harvest_main()

    # Assertions
    assert output_file.exists()
    harvested_records = []
    with open(output_file, "r") as f:
        for line in f:
            harvested_records.append(json.loads(line))
    
    assert len(harvested_records) == 1
    # The only record matching all criteria: schema_version="1.0", feedback_type="correction", recent
    # The prompt for harvest_feedback.py is 'assessment' or 'proposal'.
    # Our test data doesn't have these keys, so the 'prompt' field in output will be empty string.
    # This is fine, we are testing filtering, not content transformation logic of harvest script here.
    # The script extracts 'prompt' from 'assessment' or 'proposal' keys, which are not in FeedbackSchema.
    # This implies harvest_feedback.py might be designed for a slightly different raw data structure
    # than what FeedbackItem/FeedbackSchema strictly define.
    # For the purpose of this test, we check if the *correct record* (rec1) was selected based on filtering criteria.
    # The output format is {"prompt": "...", "response": "{...corrected_proposal_json...}"}
    
    # To make the assertion more robust, let's check the content of the 'response' field,
    # which should be the JSON dump of 'corrected_proposal' of 'rec1_v1_correct_recent'.
    found_rec1 = False
    for rec in harvested_records:
        response_data = json.loads(rec["response"])
        if response_data == common_data["corrected_proposal"]: # Simple check
             # To be more specific, we'd need to know which record's proposal this is.
             # Since only one record is expected, this is okay.
             # If we want to ensure it's from "rec1_v1_correct_recent", we'd need transaction_id
             # in the output of harvest_feedback.py, which it doesn't currently do.
             found_rec1 = True
             break
    assert found_rec1

    # Cleanup: tmp_path_factory handles automatic cleanup of db_path and output_file parent dir

@pytest.mark.asyncio
async def test_migrate_feedback_py_script(tmp_path_factory, monkeypatch):
    """Integration-like test for scripts/migrate_feedback.py."""
    db_path = tmp_path_factory.mktemp("lancedb_migrate_test")
    
    # Connect to temporary DB
    db = lancedb.connect(db_path)
    table_name = "phi3_feedback" # Script uses this name

    # For migration, we need to simulate data that *might* not have schema_version.
    # LanceDB can create schema from data. We'll add dicts.
    # The migration script reads this data, adds schema_version, and writes to a *new*
    # table defined by its internal `FeedbackSchemaWithVersion`.
    
    # Initial data: one with schema_version, one without
    # Timestamps for 'when' are not strictly part of migration logic test,
    # but if they exist, they should be carried over.
    # The migration script's `FeedbackSchemaWithVersion` does not include 'when'.
    # So 'when' will be carried over as an "extra" field if present in source.
    # Let's simplify and use data that matches `FeedbackSchemaOldForMigrationTest` more closely.
    
    record_A_dict = { # Simulating old data, no schema_version
        "transaction_id": "a", "timestamp": "ts_a", "feedback_type": "type_a", 
        "feedback_content": "content_a", "corrected_proposal": {"key": "val_a"}
    }
    record_B_dict = { # Simulating data that somehow already has a version (e.g. partial previous migration)
        "transaction_id": "b", "timestamp": "ts_b", "feedback_type": "type_b", 
        "feedback_content": "content_b", "corrected_proposal": {"key": "val_b"}, "schema_version": "0.8" # Old version
    }
    record_C_dict = { # Simulating data that is already up-to-date
        "transaction_id": "c", "timestamp": "ts_c", "feedback_type": "type_c", 
        "feedback_content": "content_c", "corrected_proposal": {"key": "val_c"}, "schema_version": "1.0"
    }

    # Create table and add initial data.
    # The schema of the initial table should allow for schema_version to be missing.
    # LanceDB will infer if no schema provided.
    if table_name in db.table_names(): # Clean slate
        db.drop_table(table_name)
    
    # We can create the table with the schema that the *migration script expects to write to*
    # (MigrationOutputSchema), but set schema_version as Optional for initial insert.
    # Or, let LanceDB infer from data that includes optional schema_version.
    # For robustness, let's try to create with a schema that allows optional schema_version
    
    class TempSchemaForInitialData(BaseModel):
        transaction_id: str
        timestamp: str
        feedback_type: str
        feedback_content: Any
        corrected_proposal: Optional[Dict[str, Any]] = None
        schema_version: Optional[str] = None # Key for initial data

    try:
        # Create with a schema that allows schema_version to be None (or missing)
        table = db.create_table(table_name, schema=TempSchemaForInitialData, mode="overwrite")
        table.add([record_A_dict, record_B_dict, record_C_dict])
    except Exception as e:
        # Fallback if Pydantic schema with Optional doesn't work as expected on create
        print(f"Warning: Initial table creation with TempSchemaForInitialData failed: {e}. Relying on schema inference.")
        if table_name in db.table_names(): db.drop_table(table_name)
        table = db.create_table(table_name, mode="overwrite") # No explicit schema
        # Add data as dicts, LanceDB will infer types. `schema_version` might be missing.
        table.add([record_A_dict, record_B_dict, record_C_dict])


    # Monkeypatch lancedb.connect for the script
    def mock_lancedb_connect_migrate(path):
        assert str(path) == str(db_path)
        return lancedb.connect(path) # Connect to the actual temp path
    monkeypatch.setattr("scripts.migrate_feedback.lancedb.connect", mock_lancedb_connect_migrate)
    # Also mock os.makedirs in migrate_feedback if it's there, to avoid issues with tmp_path
    monkeypatch.setattr("scripts.migrate_feedback.os.makedirs", lambda path, exist_ok=False: None, raising=False)


    # Run the migration script's main function
    migrate_main()

    # Assertions: Check data in the table after migration
    migrated_table = db.open_table(table_name)
    results = migrated_table.search().to_list()
    assert len(results) == 3
    
    found_a, found_b, found_c = False, False, False
    for record in results:
        assert record.get("schema_version") == "1.0" # All should be updated
        if record["transaction_id"] == "a":
            assert record["feedback_content"] == "content_a"
            found_a = True
        elif record["transaction_id"] == "b":
            assert record["feedback_content"] == "content_b"
            # schema_version was "0.8", now "1.0"
            found_b = True
        elif record["transaction_id"] == "c":
            assert record["feedback_content"] == "content_c"
            # schema_version was already "1.0", should remain "1.0"
            found_c = True
            
    assert found_a and found_b and found_c

    # Test idempotency: Run migration again
    print("Running migration script again for idempotency check...")
    migrate_main()
    
    migrated_table_again = db.open_table(table_name)
    results_again = migrated_table_again.search().to_list()
    assert len(results_again) == 3
    for record in results_again:
        assert record.get("schema_version") == "1.0"
        # Quick check on one record's content
        if record["transaction_id"] == "a":
            assert record["feedback_content"] == "content_a"
    print("Idempotency check passed.")

    # Cleanup is handled by tmp_path_factory
