import pytest
import tempfile
import pathlib
import importlib # For reloading the module
import json # For fields that are stored as JSON strings

# Initial import of the module. Its state will be patched and reloaded.
import osiris.llm_sidecar.db as llm_sidecar_db

def test_db_initialization_and_operations(monkeypatch):
    """
    Tests database initialization, table creation, data insertion, and row counts
    using a temporary directory for DB_ROOT.
    """
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmp_db_root = pathlib.Path(tmpdir_name)

        # Patch _tables and _db to ensure they are reset for the reloaded module
        monkeypatch.setattr(llm_sidecar_db, '_tables', {})
        monkeypatch.setattr(llm_sidecar_db, '_db', None)

        # Keep a reference to the original lancedb.connect
        # This is assuming lancedb is an attribute of the llm_sidecar.db module
        original_lancedb_connect = llm_sidecar_db.lancedb.connect

        # Define a wrapper for lancedb.connect that always uses the temp directory
        def mock_lancedb_connect(path, **kwargs):
            # Force connection to the temporary directory
            return original_lancedb_connect(tmp_db_root, **kwargs)

        monkeypatch.setattr(llm_sidecar_db.lancedb, 'connect', mock_lancedb_connect)

        # Also, ensure DB_ROOT in the module itself is set for any direct uses of it,
        # although the connect patch is the primary mechanism now.
        monkeypatch.setattr(llm_sidecar_db, 'DB_ROOT', tmp_db_root)


        # Reload the module. This will execute `_db = lancedb.connect(DB_ROOT)`
        # which will now call our mock_lancedb_connect.
        # It also calls init_db() at the end of the module.
        db_module = importlib.reload(llm_sidecar_db)

        # 1. Assert that the three tables exist
        # Access _db via the reloaded module instance. It should be connected to tmp_db_root.
        assert db_module._db.uri == str(tmp_db_root), \
            f"Database connected to {db_module._db.uri} instead of {tmp_db_root}"
        table_names = db_module._db.table_names()
        assert "phi3_feedback" in table_names, f"Expected 'phi3_feedback' in {table_names}"
        assert "orchestrator_runs" in table_names, f"Expected 'orchestrator_runs' in {table_names}"
        assert "hermes_scores" in table_names, f"Expected 'hermes_scores' in {table_names}"

        # 2. Create one sample Pydantic model instance for each schema
        # Access schemas via the reloaded module instance
        feedback_sample = db_module.Phi3FeedbackSchema(
            transaction_id="txn_bootstrap_123",
            feedback_type="rating",
            feedback_content="Bootstrap test good",
            # schema_version and timestamp will use defaults
        )

        orchestrator_run_sample = db_module.OrchestratorRunSchema(
            input_query="What is bootstrap testing?",
            # run_id and timestamp will use defaults
            status="SUCCESS",
            final_output=json.dumps({"result": "Bootstrap test successful"}),
            error_message=None
        )

        hermes_score_sample = db_module.HermesScoreSchema(
            run_id=orchestrator_run_sample.run_id, # Link to the created run
            # timestamp will use default
            score=0.95,
            rationale="Bootstrap test rationale"
        )

        # 3. Add these sample rows
        # Access logging functions via the reloaded module instance
        db_module.append_feedback(feedback_sample)
        db_module.log_run(orchestrator_run_sample)
        db_module.log_hermes_score(hermes_score_sample)

        # 4. For each table, retrieve the table object and assert that table.count_rows() is 1
        # Access _tables via the reloaded module instance

        feedback_table = db_module._tables.get("phi3_feedback")
        assert feedback_table is not None, "phi3_feedback table object not found in _tables"
        assert feedback_table.count_rows() == 1

        orchestrator_runs_table = db_module._tables.get("orchestrator_runs")
        assert orchestrator_runs_table is not None, "orchestrator_runs table object not found in _tables"
        assert orchestrator_runs_table.count_rows() == 1

        hermes_scores_table = db_module._tables.get("hermes_scores")
        assert hermes_scores_table is not None, "hermes_scores table object not found in _tables"
        assert hermes_scores_table.count_rows() == 1

        # Verify data content (optional, but good for sanity check)
        retrieved_feedback = feedback_table.search().limit(1).to_list()
        assert len(retrieved_feedback) == 1
        assert retrieved_feedback[0]['transaction_id'] == "txn_bootstrap_123"

        retrieved_runs = orchestrator_runs_table.search().limit(1).to_list()
        assert len(retrieved_runs) == 1
        assert retrieved_runs[0]['input_query'] == "What is bootstrap testing?"
        assert json.loads(retrieved_runs[0]['final_output']) == {"result": "Bootstrap test successful"}

        retrieved_scores = hermes_scores_table.search().limit(1).to_list()
        assert len(retrieved_scores) == 1
        assert retrieved_scores[0]['run_id'] == orchestrator_run_sample.run_id

        # Temporary directory tmpdir_name is automatically cleaned up.
        # Close the db connection from the reloaded module to release file locks.
        if hasattr(db_module._db, 'close'):
             db_module._db.close()

# To ensure that the reloaded module's state doesn't interfere with other tests,
# it might be good practice to restore the original module state if pytest doesn't fully isolate it.
# However, pytest's monkeypatch and test execution model usually provide good isolation.
# For module-level state like DB_ROOT and _db, reloading is a robust way to re-initialize.
# The last close() helps with file handles.
# If tests were to run in parallel on the same module without care, it could be an issue,
# but pytest typically runs tests in separate processes or manages imports carefully.

# One final consideration: if other tests import llm_sidecar.db before this test runs,
# they will use the original DB_ROOT. This test specifically tests the bootstrap with a new root.
# This is generally fine as test order shouldn't matter for well-isolated tests.
# The monkeypatching and reloading are scoped to this test function.
# However, to be extremely cautious about module state affecting subsequent tests within the same session
# (if not fully isolated by pytest runner), one might consider restoring the original DB_ROOT and reloading again
# in a fixture or at the end of the test. For now, this structure should be okay.
# The .close() is the most important cleanup for LanceDB here.

# After the test, the original llm_sidecar.db module (if imported by other tests)
# would still point to its original DB_ROOT. The reloaded instance is what this test uses.
# Pytest test collection/execution should handle this.
# If `llm_sidecar.db` was imported at the top of this test file using `import llm_sidecar.db as original_db_module`
# and then inside the test `reloaded_db_module = importlib.reload(llm_sidecar.db)`, all references inside
# the test would be to `reloaded_db_module.some_function`. This is what's happening implicitly.
# The `from llm_sidecar.db import ...` statements at the top might grab functions/objects from the
# originally loaded module. This could be an issue if not careful.

# Let's adjust imports to be safer with reloading.
# We should get the functions and classes from the *reloaded* module instance.
# So, instead of `from llm_sidecar.db import X`, use `reloaded_db_module.X`.

# Revised structure for clarity with reloading:

# import llm_sidecar.db # Initial import
# (keep other top-level imports like pytest, tempfile, pathlib, importlib, json)

# def test_db_initialization_and_operations(monkeypatch):
#     with tempfile.TemporaryDirectory() as tmpdir_name:
#         tmp_db_root = pathlib.Path(tmpdir_name)
#         monkeypatch.setattr(llm_sidecar.db, 'DB_ROOT', tmp_db_root)
#         monkeypatch.setattr(llm_sidecar.db, '_db', None) # Ensure connection is re-established
#         monkeypatch.setattr(llm_sidecar.db, '_tables', {}) # Ensure tables are re-cached

#         db_module = importlib.reload(llm_sidecar.db) # Reload and get the new module instance

#         db_module.init_db() # Call init_db from the reloaded module

#         # Access everything via db_module.
#         table_names = db_module._db.table_names()
#         # ...
#         feedback_sample = db_module.Phi3FeedbackSchema(...)
#         # ...
#         db_module.append_feedback(feedback_sample)
#         # ...
#         feedback_table = db_module._tables.get("phi3_feedback")
#         # ...
#         if hasattr(db_module._db, 'close'):
#             db_module._db.close()

# This revised structure is more robust. I will use this.
