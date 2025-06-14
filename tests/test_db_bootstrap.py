import importlib
import tempfile
import pathlib
import pytest

import llm_sidecar.db

def test_db_initialization_and_operations(monkeypatch):
    """
    Tests database initialization, table creation, data insertion, and row counts
    using a temporary directory for DB_ROOT. Adapts to the correct feedback_content field.
    """
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmp_db_root = pathlib.Path(tmpdir_name)

        # Patch _tables and _db to ensure they are reset for the reloaded module
        monkeypatch.setattr(llm_sidecar.db, "_tables", {})
        monkeypatch.setattr(llm_sidecar.db, "_db", None)

        # Keep a reference to the original lancedb.connect
        original_lancedb_connect = llm_sidecar.db.lancedb.connect

        # Define a wrapper for lancedb.connect that always uses the temp directory
        def mock_lancedb_connect(path, **kwargs):
            return original_lancedb_connect(tmp_db_root, **kwargs)

        monkeypatch.setattr(llm_sidecar.db.lancedb, "connect", mock_lancedb_connect)
        monkeypatch.setattr(llm_sidecar.db, "DB_ROOT", tmp_db_root)

        db_module = importlib.reload(llm_sidecar.db)

        # 1. Assert that the three tables exist
        assert db_module._db.uri == str(tmp_db_root), f"Database connected to {db_module._db.uri} instead of {tmp_db_root}"
        table_names = db_module._db.table_names()
        assert "phi3_feedback" in table_names, f"Expected 'phi3_feedback' in {table_names}"
        assert "orchestrator_runs" in table_names, f"Expected 'orchestrator_runs' in {table_names}"
        assert "hermes_scores" in table_names, f"Expected 'hermes_scores' in {table_names}"

        # 2. Create one sample Pydantic model instance for each schema
        feedback_sample = db_module.Phi3FeedbackSchema(
            transaction_id="txn_bootstrap_123",
            feedback_type="rating",
            feedback_content={"note": "Bootstrap test good"},
            # schema_version and timestamp will use defaults
        )
        orchestrator_sample = db_module.OrchestratorRunSchema(
            run_id="run_bootstrap_123",
            user_id="user_abc",
            run_metadata={"boot": True}
        )
        hermes_sample = db_module.HermesScoreSchema(
            proposal_id="prop_bootstrap_123",
            score=0.42,
            timestamp="2025-06-13T00:00:00Z"
        )

        # 3. Append samples to each table
        db_module._tables["phi3_feedback"].add([feedback_sample.model_dump()])
        db_module._tables["orchestrator_runs"].add([orchestrator_sample.model_dump()])
        db_module._tables["hermes_scores"].add([hermes_sample.model_dump()])

        # 4. Check table row counts
        assert db_module._tables["phi3_feedback"].count_rows() == 1
        assert db_module._tables["orchestrator_runs"].count_rows() == 1
        assert db_module._tables["hermes_scores"].count_rows() == 1
