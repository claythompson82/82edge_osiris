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
import math # <--- Added import math
from datetime import datetime, timezone
import importlib # For reloading server module

from azr_planner.schemas import Instrument, Direction, Leg # For constructing currentPositions

# --- Helper function to generate HLC data (can be shared or defined in a conftest.py later) ---
def _generate_hlc_data_for_server_test(num_periods: int, start_price: float = 100.0, daily_change: float = 0.1, spread: float = 0.5) -> list[tuple[float, float, float]]:
    data = []
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + abs(daily_change * math.sin(i*0.1))
        low = current_close - spread - abs(daily_change * math.cos(i*0.1))
        close = (high + low) / 2 + (math.sin(i*0.5) * spread*0.1)
        low = min(low, high - 0.01)
        close = max(min(close, high), low)
        data.append((round(high,2), round(low,2), round(close,2)))
        current_close = close
    return data

# Updated sample data for the new PlanningContext schema
# Using field names directly as aliases match for new fields.
# MIN_HISTORY_POINTS_SERVER_TEST needs to be consistent with engine's requirements.
# From engine: max(ASSUMED_EMA_LONG_PERIOD, ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK, 15)
# Let's use a value that satisfies typical placeholder values (e.g., 60 for Kelly lookback + buffer)
MIN_HISTORY_POINTS_SERVER_TEST = 65

SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "equityCurve": [10000.0 + i*10 for i in range(35)],
    "dailyHistoryHLC": _generate_hlc_data_for_server_test(MIN_HISTORY_POINTS_SERVER_TEST),
    "dailyVolume": [10000 + i*100 for i in range(MIN_HISTORY_POINTS_SERVER_TEST)],
    "currentPositions": [
        Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0).model_dump() # Use model_dump for FastAPI
    ],
    "volSurface": {"MES": 0.15, "M2K": 0.20},
    "riskFreeRate": 0.02,
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
    # Re-assign app to the reloaded server's app to be sure TestClient uses it.
    global app
    app = server.app

    yield #ปล่อยให้ test functions ทำงาน

    if original_value is None:
        if "OSIRIS_TEST" in os.environ:
            del os.environ["OSIRIS_TEST"]
    else:
        os.environ["OSIRIS_TEST"] = original_value

    # Reload again to revert to original state for other test modules if any
    importlib.reload(server)
    app = server.app # Reset app global

# Note: The 'app' variable used by TestClient is captured when 'tests/test_server.py'
# is first imported. The fixture above reloads 'server' but 'TestClient(app)'
# might still use the initially imported 'app'.
# A common pattern is to have a fixture that yields a TestClient with the correctly configured app.

@pytest.fixture(scope="function") # Use function scope if app state changes per test due to OSIRIS_TEST
def test_client_azr() -> TestClient:
    """Provides a TestClient instance with the reloaded server.app."""
    # Ensure server module used by TestClient is the one reloaded by the module fixture
    # The module fixture should handle reloading 'server' and updating the 'app' global.
    return TestClient(app)


def test_azr_planner_new_engine_smoke_test(test_client_azr: TestClient) -> None:
    """
    Smoke test for the AZR Planner endpoint with the new engine. (OSIRIS_TEST=1 active)
    POSTs a valid new PlanningContext and checks for a 200 OK and valid TradeProposal structure.
    """
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 200, f"Response content: {response.text}"

    data = response.json()
    from azr_planner.schemas import TradeProposal as TradeProposalSchema # For validation

    # Validate response against the TradeProposal schema
    try:
        TradeProposalSchema.model_validate(data)
    except Exception as e:
        pytest.fail(f"Response does not match TradeProposal schema: {e}\nResponse data: {data}")

    assert data["action"] in ["HOLD", "ENTER", "EXIT", "ADJUST"] # Valid actions
    assert isinstance(data["rationale"], str) and len(data["rationale"]) > 0
    assert isinstance(data["confidence"], float) and 0.0 <= data["confidence"] <= 1.0

    # Check for new optional fields (they can be None)
    assert "signal_value" in data
    assert "atr_value" in data
    assert "kelly_fraction_value" in data
    assert "target_position_size" in data

    if data.get("atr_value") is not None:
        assert isinstance(data["atr_value"], float) and data["atr_value"] >= 0
    if data.get("kelly_fraction_value") is not None:
        assert isinstance(data["kelly_fraction_value"], float) and data["kelly_fraction_value"] >= 0
    if data.get("target_position_size") is not None:
        assert isinstance(data["target_position_size"], float) and data["target_position_size"] >= 0

    # Latent risk is now optional in TradeProposal
    if data.get("latent_risk") is not None:
         assert isinstance(data["latent_risk"], float) and 0.0 <= data["latent_risk"] <= 1.0

    if data["action"] in ["ENTER", "EXIT", "ADJUST"]: # These actions usually have legs
        if data.get("target_position_size", 0.0) > 0 or (data["action"] == "EXIT" and SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW.get("currentPositions")):
            assert data["legs"] is not None, f"Action {data['action']} should have legs if target size > 0 or exiting existing position."
            assert isinstance(data["legs"], list)
            if data["legs"]: # If list is not empty
                leg = data["legs"][0]
                assert "instrument" in leg
                assert "direction" in leg
                assert "size" in leg and leg["size"] > 0
        # else: # target_position_size is 0 or None, or not exiting an existing position
            # legs might be None or empty list, which is acceptable.
    elif data["action"] == "HOLD":
        # For HOLD, legs can be None or an empty list depending on engine specifics.
        # The current placeholder engine might set it to None.
        pass


# --- Comment out old AZR planner tests based on latent_risk and old schema ---
# @patch('azr_planner.engine.calculate_latent_risk') # Mock at engine level
# def test_azr_planner_action_enter_on_low_risk(mock_calc_lr: MagicMock, test_client_azr: TestClient) -> None:
# ... (rest of old tests commented out) ...

# def test_azr_planner_invalid_input_equity_curve_too_short(test_client_azr: TestClient) -> None:
# ... (this test might still be valid if equityCurve min_length is kept, but other fields are now needed)

# def test_azr_planner_invalid_input_missing_required_field(test_client_azr: TestClient) -> None:
# ... (this test needs to be updated for new required fields like dailyHistoryHLC)


# --- New tests for invalid inputs based on the updated PlanningContext ---
def test_azr_planner_invalid_input_daily_history_hlc_too_short(test_client_azr: TestClient) -> None:
    """Test AZR planner with invalid input (dailyHistoryHLC too short)."""
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW.copy()
    invalid_context["dailyHistoryHLC"] = [(100,99,99.5)] * 5 # Too short (min_length is 15 in schema)
    if invalid_context.get("dailyVolume"): # Adjust volume if present
        invalid_context["dailyVolume"] = [1000] * 5

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422 # Expect Pydantic validation error
    data = response.json()
    assert "detail" in data
    assert any(
        error.get("loc") == ['body', 'dailyHistoryHLC'] and
        "List should have at least 15 items" in error.get("msg", "")
        for error in data["detail"]
    ), f"Error details: {data['detail']}"


def test_azr_planner_invalid_input_missing_daily_history_hlc(test_client_azr: TestClient) -> None:
    """Test AZR planner with invalid input (missing dailyHistoryHLC)."""
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW.copy()
    del invalid_context["dailyHistoryHLC"]
    if "dailyVolume" in invalid_context: # Remove volume if HLC is removed
        del invalid_context["dailyVolume"]

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any(
        "dailyHistoryHLC" in error.get("loc", []) and
        "Field required" in error.get("msg", "")
        for error in data["detail"]
    ), f"Error details: {data['detail']}"


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
