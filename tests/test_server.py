import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import osiris.server as server

app = server.app

def test_generate_hermes_default_model_id():
    # Tests /generate/ with default model_id (hermes)
    client = TestClient(app)
    with patch("osiris.server._generate_hermes_text", new_callable=AsyncMock, return_value="Hermes mock output"):
        with patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_hermes: # type: MagicMock
            response = client.post("/generate/", json={"prompt": "test prompt for hermes default", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == {"output": "Hermes mock output"}

def test_generate_phi3_explicit_model_id():
    # Tests /generate/ with explicit model_id='phi3'
    client = TestClient(app)
    mock_phi3_output = {"phi3_mock_output": "success"}
    with patch("osiris.server._generate_phi3_json", new_callable=AsyncMock, return_value=mock_phi3_output):
        with patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_phi3: # type: MagicMock
            response = client.post("/generate/", json={"prompt": "test prompt for phi3", "model_id": "phi3", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == mock_phi3_output

def test_score_proposal_with_hermes_success():
    # Tests /score/hermes/ endpoint successful scoring
    client = TestClient(app)
    with (
        patch("osiris.llm_sidecar.hermes_plugin.score_with_hermes", return_value=0.75) as mock_score_hermes, # type: MagicMock
        patch("osiris.llm_sidecar.db.log_hermes_score", return_value=None) as mock_log_score, # type: MagicMock
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_hermes # type: MagicMock
    ):
        response = client.post("/score/hermes/", json={"proposal": {"foo": "bar"}, "context": "ctx"})
    assert response.status_code == 200
    assert response.json() == {"score": 0.75}

def test_health_endpoint_with_adapter_date(monkeypatch, tmp_path):
    # Tests /health?adapter_date=true endpoint for adapter dir
    client = TestClient(app)
    # Patch adapter root with fake directories
    (tmp_path / "2025-06-10").mkdir()
    monkeypatch.setattr(
        "osiris.llm_sidecar.loader.ADAPTER_ROOT", tmp_path, raising=False
    )
    with (
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_hermes, # type: MagicMock
        patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_phi3, # type: MagicMock
        patch("osiris.llm_sidecar.db.get_mean_hermes_score_last_24h", return_value=0.95) as mock_get_mean_score, # type: MagicMock
    ):
        response = client.get("/health?adapter_date=true")
        assert response.status_code == 200
        body = response.json()
        assert body["latest_adapter"] == "2025-06-10"

def test_speak_endpoint(monkeypatch):
    # Tests /speak endpoint (dummy test, patching all dependencies)
    client = TestClient(app)
    with (
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_hermes, # type: MagicMock
        patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_phi3, # type: MagicMock
        patch("osiris.server.text_to_speech", return_value=b"dummy_audio") as mock_tts, # type: MagicMock
    ):
        response = client.post("/speak/", json={"text": "Say something"})
        assert response.status_code == 200
        assert response.content == b"dummy_audio"

# --- Tests for AZR Planner ---
import os
from datetime import datetime, timezone
import importlib # For reloading server module

# Sample data for testing the AZR Planner endpoint
# Needs to be defined globally for use in tests.
# Using camelCase for aliased fields
SAMPLE_PLANNING_CONTEXT_DATA_AZR = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "equityCurve": [100.0 + i for i in range(35)], # Alias: equityCurve
    "volSurface": {"MES": 0.15, "M2K": 0.20},   # Alias: volSurface
    "riskFreeRate": 0.02,                      # Alias: riskFreeRate
}

@pytest.fixture(scope="module", autouse=True)
def _manage_osiris_test_env_var_for_azr_module():
    """
    Ensure OSIRIS_TEST is set for AZR tests in this module and restore afterward.
    Also reloads the server module to apply the env var change for app configuration.
    """
    original_value = os.environ.get("OSIRIS_TEST")
    os.environ["OSIRIS_TEST"] = "1"

    # Reload server module to apply the env var change for app configuration
    importlib.reload(server)
    # Update the global 'app' variable to the reloaded server's app
    # This is tricky because 'app' is already imported at the top level.
    # The TestClient instances should ideally pick up the reloaded 'server.app'.
    # To be absolutely sure, TestClient could be instantiated inside tests
    # or a fixture could provide a client with the reloaded app.

    # For simplicity, we assume TestClient(server.app) will use the reloaded one.
    # If issues persist, client instantiation within tests might be needed.

    yield #ปล่อยให้ test functions ทำงาน

    if original_value is None:
        if "OSIRIS_TEST" in os.environ:
            del os.environ["OSIRIS_TEST"]
    else:
        os.environ["OSIRIS_TEST"] = original_value

    # Reload again to revert to original state for other test modules if any
    importlib.reload(server)

# Note: The 'app' variable used by TestClient is captured when 'tests/test_server.py'
# is first imported. The fixture above reloads 'server' but 'TestClient(app)'
# might still use the initially imported 'app'.
# A common pattern is to have a fixture that yields a TestClient with the correctly configured app.

@pytest.fixture(scope="function") # Use function scope if app state changes per test due to OSIRIS_TEST
def test_client_azr() -> TestClient:
    """Provides a TestClient instance with the reloaded server.app."""
    # Ensure server module used by TestClient is the one reloaded by the module fixture
    return TestClient(server.app)


def test_azr_planner_smoke_endpoint_exists(test_client_azr: TestClient) -> None:
    """
    Smoke test for the AZR Planner endpoint. (OSIRIS_TEST=1 active)
    """
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR)
    assert response.status_code == 200, f"Response content: {response.text}"
    data = response.json()

    # Expect TradePlan response. Actual content depends on mocked latent_risk or default placeholder.
    # For a generic smoke test without mocking latent_risk here, we check structure.
    # The engine's new latent_risk might result in 'ENTER', 'HOLD', or 'EXIT'.
    assert "action" in data
    assert data["action"] in ["HOLD", "ENTER", "EXIT"] # Check it's one of the valid actions
    assert "rationale" in data
    assert isinstance(data["rationale"], str)
    assert "latent_risk" in data
    assert isinstance(data["latent_risk"], float)
    assert 0.0 <= data["latent_risk"] <= 1.0
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

    # Verify confidence calculation consistency
    import math
    assert math.isclose(data["confidence"], round(1.0 - data["latent_risk"], 3))

    if data["action"] == "ENTER":
        assert "legs" in data
        assert isinstance(data["legs"], list)
        assert len(data["legs"]) == 1
        leg = data["legs"][0]
        assert leg["instrument"] == "MES"
        assert leg["direction"] == "LONG"
        assert leg["size"] == 1.0
    elif data["action"] == "EXIT":
        assert "legs" in data
        assert isinstance(data["legs"], list)
        assert len(data["legs"]) == 1
        leg = data["legs"][0]
        assert leg["instrument"] == "MES"
        assert leg["direction"] == "SHORT"
        assert leg["size"] == 1.0 # Stub size
    elif data["action"] == "HOLD":
        assert data.get("legs") is None # For HOLD, legs should be None


@patch('azr_planner.engine.calculate_latent_risk') # Mock at engine level
def test_azr_planner_action_enter_on_low_risk(mock_calc_lr: MagicMock, test_client_azr: TestClient) -> None:
    """Test endpoint returns ENTER action when latent risk < 0.30."""
    test_risk = 0.15
    mock_calc_lr.return_value = test_risk

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR)
    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "ENTER"
    assert data["rationale"] == "Latent risk is low, favorable for new positions."
    assert data["latent_risk"] == test_risk
    assert data["confidence"] == round(1.0 - test_risk, 3)
    assert len(data["legs"]) == 1
    leg = data["legs"][0]
    assert leg["instrument"] == "MES"
    assert leg["direction"] == "LONG"
    assert leg["size"] == 1.0

@patch('azr_planner.engine.calculate_latent_risk')
def test_azr_planner_action_hold_on_moderate_risk(mock_calc_lr: MagicMock, test_client_azr: TestClient) -> None:
    """Test endpoint returns HOLD action when 0.30 <= latent risk <= 0.70."""
    test_risk = 0.5
    mock_calc_lr.return_value = test_risk # This was missing

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR)
    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "HOLD"
    assert data["rationale"] == "Latent risk is moderate, maintaining current positions."
    assert data["latent_risk"] == test_risk
    assert data["confidence"] == round(1.0 - test_risk, 3)
    assert data.get("legs") is None

@patch('azr_planner.engine.calculate_latent_risk')
def test_azr_planner_action_exit_on_high_risk(mock_calc_lr: MagicMock, test_client_azr: TestClient) -> None:
    """Test endpoint returns EXIT action when latent risk > 0.70."""
    test_risk = 0.85
    mock_calc_lr.return_value = test_risk

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR)
    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "EXIT"
    assert data["rationale"] == "Latent risk is high, reducing exposure."
    assert data["latent_risk"] == test_risk
    assert data["confidence"] == round(1.0 - test_risk, 3)
    assert len(data["legs"]) == 1
    leg = data["legs"][0]
    assert leg["instrument"] == "MES"
    assert leg["direction"] == "SHORT"
    assert leg["size"] == 1.0 # Stub size


def test_azr_planner_invalid_input_equity_curve_too_short(test_client_azr: TestClient) -> None:
    """Test AZR planner with invalid input (equity curve too short)."""
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR.copy()
    invalid_context["equityCurve"] = [1.0] * 20
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # print("DEBUG: test_azr_planner_invalid_input_equity_curve_too_short - data['detail']:", data["detail"]) # DEBUG PRINT removed
    # Pydantic error loc for FastAPI body fields is typically ('body', <field_name_or_alias>)
    # Error loc from FastAPI is a list: ['body', 'equityCurve']
    assert any(error.get("loc") == ['body', 'equityCurve'] and error.get("msg", "").startswith("List should have at least 30 items") for error in data["detail"])

def test_azr_planner_invalid_input_missing_required_field(test_client_azr: TestClient) -> None:
    """Test AZR planner with invalid input (missing timestamp)."""
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR.copy()
    del invalid_context["timestamp"]
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("timestamp" in error.get("loc", []) and "Field required" in error.get("msg", "") for error in data["detail"])

def test_azr_planner_prefix_and_tags_available_when_osiris_test_set(test_client_azr: TestClient) -> None:
    """Test if the AZR router has correct prefix and tags when OSIRIS_TEST=1."""
    # This test relies on the _manage_osiris_test_env_var_for_azr_module fixture
    # having set OSIRIS_TEST=1 and reloaded the server.app for the test_client_azr.
    openapi_schema = test_client_azr.get("/openapi.json").json()

    path_key = "/azr_api/internal/azr/planner/propose_trade"
    assert path_key in openapi_schema["paths"], f"Path {path_key} not found. OSIRIS_TEST may not have been effective."

    operation = openapi_schema["paths"][path_key].get("post")
    assert operation is not None, "POST operation not found."
    assert "AZR Planner" in operation.get("tags", []), "Tag 'AZR Planner' not applied to operation."

    # Check tag definition (optional, but good for completeness)
    defined_tags = [tag["name"] for tag in openapi_schema.get("tags", [])]
    assert "AZR Planner" in defined_tags, "Tag 'AZR Planner' not defined in OpenAPI schema."
