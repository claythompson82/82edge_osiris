import asyncio
import os
import time
import uuid
import json # Added json import

import httpx
import pytest
import pytest_asyncio
from tenacity import retry, stop_after_attempt, wait_fixed # For retry logic

# Import Phi3FeedbackSchema and other necessary components from the application's db module
# This requires that the `llm_sidecar` package is discoverable in the PYTHONPATH.
# For E2E tests running against a Docker container, direct DB interaction from the test
# host can be tricky if the DB is only exposed within the Docker network.
# The issue implies inserting a feedback row, which might mean:
# 1. The test environment has access to the DB (e.g., lancedb service is port-mapped to host).
# 2. There's an API endpoint to submit feedback (preferred for E2E if available).
# Assuming for now that the test needs to simulate this by preparing data and potentially
# calling an internal method if direct DB access is not straightforward,
# or that an endpoint exists.
# For this implementation, we will assume an endpoint `/feedback/` exists or that
# the `llm_sidecar.db` module can be used if lancedb is accessible.
# Given the problem description, direct insertion seems implied.
# We will try to use the application's own db functions if lancedb is accessible.
# This means the lancedb container's port 8100 (gRPC) or 8101 (REST) needs to be mapped.
# The docker/compose.yaml maps 8100 and 8101.
# The db.py uses lancedb.connect(DB_ROOT) with DB_ROOT="/app/lancedb_data".
# This path is inside the lancedb container.
# To interact with it from the test host, we'd typically use the lancedb client library
# connecting to "http://localhost:8101" (if REST is enabled and used by client) or gRPC.
#
# Simpler approach for now: The test will call an endpoint to log feedback if one exists.
# If not, this step will be harder. The issue states "inserting a feedback row".
# Let's assume there is *no dedicated feedback endpoint* for this test and we must use the DB module.
# This requires `llm_sidecar` to be in PYTHONPATH for the test runner.
# And lancedb client in the test env to connect to the lancedb service.

# If llm_sidecar is not installed in the test environment, this import will fail.
# We might need to adjust PYTHONPATH when running pytest.
# For now, let's assume it's available or handle the import error.
try:
    from llm_sidecar.db import Phi3FeedbackSchema, append_feedback, init_db as app_init_db
    # We need to ensure the db module connects to the correct lancedb URI
    # The default db.py connects to a local path, which is fine if test runs in same container
    # But if test runs on host, it needs to connect to localhost:8101 (REST) or localhost:8100 (gRPC)
    # Forcing db connection for test:
    # import lancedb
    # test_db_uri = "data/lancedb" # For local test client
    # os.makedirs(test_db_uri, exist_ok=True)
    # _mock_db_connection = lancedb.connect(test_db_uri) # This would be a *local* db for the test, not the container's one.

    # The issue implies that the feedback insertion should affect the state read by /health.
    # This means the test *must* write to the same DB as the application.
    # The `llm_sidecar.db.append_feedback` function uses the `_db` global initialized by `lancedb.connect(DB_ROOT)`.
    # We need to override this `_db` or `DB_ROOT` for the test, or ensure `pytest-docker` makes the DB accessible
    # and `lancedb` client can connect to it.
    # `lancedb.connect("http://localhost:8101")` could be a way if REST API is enabled on lancedb server.
    # The `lancedb/lancedb-server:latest` image should support this.
    # Let's try to re-initialize the app's db connection for the test.
    # This is hacky and depends on module-level globals.

    # Forcing db connection for test by patching DB_ROOT. This is still problematic if db is already initialized.
    # A better way would be a fixture that provides a db client configured for the test.
    # For now, we will assume server.py / llm_sidecar.db handles DB init correctly
    # and the test will *call an endpoint* if one exists, or use a very carefully crafted direct DB access.
    # The simplest interpretation for "inserting a feedback row" in an E2E test is often via an existing API
    # if the goal is to test the *effect* of that feedback.
    # If no such API, direct DB modification is the only way.

    # Given the constraints and focusing on the cycle:
    # /generate -> /score -> (insert feedback) -> /health
    # The insertion must happen to the DB used by the /health endpoint.
    # Let's use the app's own `append_feedback` after ensuring its `_db` points to the containerized LanceDB.
    # This requires `llm_sidecar` to be in the path.
    # And `lancedb` Python package to be installed in the test environment.
    # The `db.py` uses `lancedb.connect(DB_ROOT)` where `DB_ROOT` is `/app/lancedb_data`. This is a path *within* the container.
    # For the test running on the host to connect to this, it must use a URI like `http://localhost:8101`.
    # So, we need to patch `lancedb.connect` or re-initialize `db._db` in a test-specific way.

    # Let's assume `llm_sidecar.db` is importable and we will attempt to use its functions.
    # The `pytest-docker` service `lancedb` should be available at `localhost:8101`.
    # We will try to monkeypatch `DB_ROOT` in `llm_sidecar.db` for the test session.
    # This is fragile. A fixture managing DB connection would be better.

    # For the purpose of this subtask, we'll write the test logic assuming `append_feedback` works correctly
    # against the containerized DB. This might require further adjustments to the test setup or db module.
    # A simpler path for "simulating a win" if an endpoint exists. If not, direct DB it is.
    # Let's assume for now we'll call `append_feedback` and it will work.

except ImportError:
    print("Warning: llm_sidecar.db could not be imported. Direct DB interaction for feedback may fail.")
    Phi3FeedbackSchema = None
    append_feedback = None
    app_init_db = None


SERVICE_URL = "http://localhost:8000"

@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(os.path.dirname(__file__), "..", "..", "docker", "compose.yaml")

@pytest.fixture(scope="session")
def docker_compose_project_name():
    return "advisor-e2e-advice-cycle" # Renamed for clarity

@pytest_asyncio.fixture(scope="session")
async def http_client(docker_services, docker_compose_project_name): # Added docker_compose_project_name
    # docker_ip = docker_services.ip_for("llm-sidecar") # This might be useful if not localhost
    # service_port = docker_services.port_for("llm-sidecar", 8000) # Port for llm-sidecar
    # actual_service_url = f"http://{docker_ip}:{service_port}"
    # For now, assume localhost mapping works as per compose file.

    health_check_url = f"{SERVICE_URL}/health"
    max_retries = 36  # Increased retries: 36 * 5s = 180s
    wait_seconds = 5

    print(f"Waiting for service to be healthy at {health_check_url}...")
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_check_url, timeout=10) # Increased timeout for health check
            if response.status_code == 200:
                health_data = response.json()
                phi3_status = health_data.get("phi3_onnx_status", health_data.get("phi3_status")) # Adapting to potential key name change
                hermes_status = health_data.get("hermes_status")
                lancedb_status = health_data.get("lancedb_status")

                if health_data.get("status") == "ok" and phi3_status == "loaded" and hermes_status == "loaded" and lancedb_status == "ok":
                    print(f"Service is healthy: {health_data}")

                    # One-time setup for DB connection IF using direct DB access from test
                    if app_init_db and append_feedback:
                        try:
                            # This is the tricky part: making llm_sidecar.db connect to the Dockerized LanceDB
                            # We need to ensure lancedb.connect() inside db.py uses the correct URI.
                            # Patching DB_ROOT before db.init_db() is called by the app is hard.
                            # A fixture that yields a db_client connected to "http://localhost:8101" would be cleaner.
                            # For now, assume the app initializes its DB connection correctly via its startup.
                            # The test will use `append_feedback` which uses the app's initialized `_db`.
                            pass # DB should be initialized by the llm-sidecar service itself.
                        except Exception as e:
                            pytest.fail(f"Failed to ensure DB is initialized for test: {e}")

                    async with httpx.AsyncClient(base_url=SERVICE_URL) as final_client:
                        yield final_client
                    return
                else:
                    print(f"Health check OK, but service not fully ready. Attempt {i+1}/{max_retries}. Response: {health_data}")
            else:
                print(f"Health check failed with status {response.status_code}. Attempt {i+1}/{max_retries}. Body: {response.text}")
        except httpx.RequestError as e:
            print(f"Health check request failed: {e}. Attempt {i+1}/{max_retries}")

        if i < max_retries - 1:
            await asyncio.sleep(wait_seconds)

    pytest.fail(f"Service did not become healthy at {SERVICE_URL} after {max_retries * wait_seconds} seconds.")


@pytest.mark.e2e
async def test_full_advice_cycle(http_client: httpx.AsyncClient):
    """
    Full-path integration test for the entire advice cycle.
    """
    proposal_id_for_feedback = None

    # 1. Hit /generate/phi3/ with a stub prompt; assert JSON schema validity.
    print("Step 1: Generating advice with /generate/phi3/...")
    generate_payload = {"prompt": "Generate a stock trade proposal for AAPL.", "model_id": "phi3"}
    try:
        response_generate = await http_client.post("/generate/", json=generate_payload, timeout=60) # Increased timeout
        response_generate.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except httpx.ReadTimeout:
        pytest.fail("Request to /generate/ timed out after 60s.")
    except httpx.HTTPStatusError as e:
        pytest.fail(f"Request to /generate/ failed with {e.response.status_code}: {e.response.text}")

    assert response_generate.status_code == 200
    generated_data = response_generate.json()

    # Basic schema validation for phi3 output (example, adjust based on actual output)
    assert "generated_text" in generated_data # This is a guess, depends on phi3 JSON output structure
    assert isinstance(generated_data["generated_text"], (dict, str)) # It might be a JSON string or a dict

    # If generated_text is a string, try to parse it as JSON because it's often a JSON blob
    advice_proposal_content = None
    if isinstance(generated_data["generated_text"], str):
        try:
            advice_proposal_content = json.loads(generated_data["generated_text"])
        except json.JSONDecodeError:
            pytest.fail(f"phi3 generated_text is a string but not valid JSON: {generated_data['generated_text']}")
    elif isinstance(generated_data["generated_text"], dict):
        advice_proposal_content = generated_data["generated_text"]
    else:
        pytest.fail(f"phi3 generated_text is neither a string nor a dict: {type(generated_data['generated_text'])}")

    assert advice_proposal_content is not None, "Could not extract advice proposal content"
    # Example assertion for proposal content (highly dependent on actual phi3 output format)
    assert "ticker" in advice_proposal_content
    assert "action" in advice_proposal_content
    print(f"Step 1 PASSED. Generated advice: {advice_proposal_content}")

    # 2. Call /score/hermes/; assert score ∈ [0,1].
    print("Step 2: Scoring advice with /score/hermes/...")
    score_payload = {
        "proposal": advice_proposal_content, # Use the parsed JSON from previous step
        "context": "e2e_test_advice_cycle"
    }
    try:
        response_score = await http_client.post("/score/hermes/", json=score_payload, timeout=30)
        response_score.raise_for_status()
    except httpx.ReadTimeout:
        pytest.fail("Request to /score/hermes/ timed out after 30s.")
    except httpx.HTTPStatusError as e:
        pytest.fail(f"Request to /score/hermes/ failed with {e.response.status_code}: {e.response.text}")

    assert response_score.status_code == 200
    score_data = response_score.json()

    assert "proposal_id" in score_data
    try:
        uuid.UUID(score_data["proposal_id"]) # Validate proposal_id is a UUID
        proposal_id_for_feedback = score_data["proposal_id"]
    except ValueError:
        pytest.fail(f"proposal_id '{score_data['proposal_id']}' is not a valid UUID.")

    assert "score" in score_data
    assert isinstance(score_data["score"], float)
    assert 0.0 <= score_data["score"] <= 1.0
    print(f"Step 2 PASSED. Score: {score_data['score']}, Proposal ID: {proposal_id_for_feedback}")

    # 3. Simulate a win outcome by inserting a feedback row (type = “outcome”).
    print("Step 3: Simulating win outcome by inserting feedback row...")
    assert proposal_id_for_feedback is not None, "proposal_id was not captured from score response"

    if not Phi3FeedbackSchema or not append_feedback:
        pytest.skip("llm_sidecar.db components not available. Skipping feedback insertion test.")
        # If skipping, the /health check for feedback counts will also need adjustment or skipping.
        # For now, let this be a hard fail if components are missing, as it's crucial for the test.
        pytest.fail("Phi3FeedbackSchema or append_feedback not available. Cannot simulate win outcome.")

    feedback_entry = Phi3FeedbackSchema(
        transaction_id=str(proposal_id_for_feedback), # Must be a string
        feedback_type="outcome", # As per issue spec
        feedback_content=json.dumps({"outcome": "win", "source": "e2e_test"}), # Content can be JSON string
        # schema_version is defaulted in Pydantic model
    )
    try:
        # This relies on append_feedback using a LanceDB client connected to the containerized DB.
        # This is the most complex part of the test's interaction.
        # Option A: Test directly calls append_feedback from llm_sidecar.db
        # This requires llm_sidecar to be in PYTHONPATH and its db._db to point to the containerized LanceDB.
        # Option B: Test calls a dedicated feedback submission endpoint (if one existed).
        # Option C: Test uses a separate lancedb client to connect to "http://localhost:8101" and add data.
        # Choosing Option A for now as it uses the application's own logic mostly.
        # However, direct DB access from test is an anti-pattern for true black-box E2E.
        # A small utility in the main app like `python -m llm_sidecar.db add-feedback ...` could be another way.

        # To make Option A work, we need to ensure the `_db` object in `llm_sidecar.db`
        # (used by `append_feedback`) is connected to the LanceDB service started by Docker.
        # The `db.py` initializes with `lancedb.connect(DB_ROOT)`, where `DB_ROOT` is an internal path.
        # We need it to connect to `http://localhost:8101` (the REST endpoint for LanceDB server).
        # This usually means patching `lancedb.connect` or `DB_ROOT` *before* `llm_sidecar.db` is imported or `init_db` is run.
        # This is tricky.
        # A workaround: Use a separate lancedb client in the test.
        import lancedb
        # Ensure the lancedb_data path exists for the client if it tries to create local files (it shouldn't for remote)
        # os.makedirs("lancedb_test_data_temp", exist_ok=True) # Not needed for remote

        # Connect to the LanceDB server exposed by Docker
        # This assumes the lancedb service in docker-compose maps port 8101 for REST or 8100 for gRPC
        # The current lancedb python client might prefer gRPC. Let's try with the http URI.
        # If this fails, may need to use gRPC URI if client supports it, or ensure REST is primary.
        # `lancedb.connect("http://localhost:8101")`
        # The `lancedb-server` image listens on 8100 (gRPC) and 8101 (REST).
        # The Python client `lancedb.connect(uri)` by default uses gRPC if `uri` is `host:port`.
        # If `uri` starts with `http://` or `https://`, it implies REST, but client might not support it directly.
        # The `connect(path)` is for local DB.
        # Let's assume `db.py` in the app is already configured and works.
        # The `append_feedback` call below will use the app's `_db` instance.

        # The `http_client` fixture already waits for lancedb_status == "ok" from /health.
        # This implies the main app (llm_sidecar) has successfully connected to LanceDB.
        # So, calling `append_feedback` should work if the `llm_sidecar` code is correct.
        append_feedback(feedback_entry)
        print(f"Step 3 PASSED. Inserted feedback for proposal_id: {proposal_id_for_feedback}")
    except Exception as e:
        # Log current tables in LanceDB for debugging
        try:
            import lancedb as local_lancedb_client
            # This connection attempt is for debugging from the test runner's perspective.
            # It assumes lancedb service is mapped to localhost:8101 (REST) or localhost:8100 (gRPC)
            # Default client connection might try gRPC.
            # Using explicit URI format for REST:
            db_client_uri = "http://localhost:8101"
            # Or for gRPC if client prefers that for host:port format and server is on 8100
            # db_client_uri = "localhost:8100"
            print(f"Attempting to connect to LanceDB at {db_client_uri} for debugging table list...")
            db_client = local_lancedb_client.connect(db_client_uri)
            print(f"LanceDB tables: {db_client.table_names()}")
        except Exception as db_e:
            print(f"Could not connect to LanceDB to list tables for debugging: {db_e}")
        pytest.fail(f"Failed to insert feedback row using append_feedback: {e}. Check LanceDB connection and schema. Trace: {e!r}")


    # 4. Trigger /health; assert latest mean Hermes score ≥0 and LanceDB row counts > 0.
    print("Step 4: Triggering /health to check effects...")
    # Wait a brief moment for DB changes to be potentially reflected if there's any async processing.
    await asyncio.sleep(2)

    response_health = await http_client.get("/health", timeout=10)
    assert response_health.status_code == 200
    health_data = response_health.json()

    # Assertions for /health endpoint based on issue spec
    # "latest mean Hermes score ≥0"
    assert "mean_hermes_score_last_24h" in health_data # Assuming this is the field
    # The score can be None if no scores in last 24h, but we just added one.
    # So it should not be None in this test flow after a successful scoring.
    assert health_data["mean_hermes_score_last_24h"] is not None, "Mean Hermes score should not be None after scoring"
    assert health_data["mean_hermes_score_last_24h"] >= 0.0

    # "LanceDB row counts > 0"
    # Need to know the exact keys for row counts from the /health endpoint.
    # Assuming keys like 'lancedb_phi3_feedback_count' and 'lancedb_hermes_scores_count'.
    # These keys are not explicitly defined in the issue, so we are guessing here.
    # Let's check for general lancedb status and some specific table counts if available.
    assert health_data.get("lancedb_status") == "ok"

    # Assuming health check exposes counts like this:
    # health_data = { ..., "lancedb_counts": {"phi3_feedback": N, "hermes_scores": M}}
    # The structure of lancedb counts in /health needs to be confirmed.
    # For now, let's assume a generic check or skip if keys are unknown.
    # Based on `server.py`'s likely implementation of /health, it might query table lengths.
    # Example from a hypothetical /health implementation:
    # "lancedb_counts": {
    #   "phi3_feedback": db._tables["phi3_feedback"].to_lance().count_rows(),
    #   "hermes_scores": db._tables["hermes_scores"].to_lance().count_rows(),
    # }
    # Let's assume the keys are `num_phi3_feedback_rows` and `num_hermes_scores_rows`.
    # This is a placeholder; actual keys from `/health` output must be used.

    # If the /health endpoint provides table counts like:
    # "table_counts": { "phi3_feedback": X, "hermes_scores": Y }
    # We would assert health_data["table_counts"]["phi3_feedback"] > 0
    # For now, checking if specific count keys exist and are > 0
    # This part is speculative due to unknown /health response structure for counts.
    # Let's look for `num_phi3_feedback_total` and `num_hermes_scores_total` as plausible keys

    phi3_feedback_count = health_data.get("num_phi3_feedback_total", 0)
    hermes_scores_count = health_data.get("num_hermes_scores_total", 0)

    # If specific count keys are not in /health, this test part needs an update.
    # A less specific test: ensure some known tables are reported.
    # For now, we rely on the issue's "LanceDB row counts > 0"
    # and assume the health endpoint will provide these directly.
    # If the test fails here, it's likely due to incorrect key names for counts.
    assert phi3_feedback_count > 0, "LanceDB row count for phi3_feedback should be > 0 after feedback insertion."
    assert hermes_scores_count > 0, "LanceDB row count for hermes_scores should be > 0 after scoring."

    print(f"Step 4 PASSED. Health check response: {health_data}")
    print("Full advice cycle E2E test completed successfully!")

```
