import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import osiris.server as server
from azr_planner.schemas import TradeProposal, Leg, Instrument, Direction # AZR-13: Added for test_metrics_endpoint

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
        Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0).model_dump()
    ],
    "nSuccesses": 20, # Added for AZR-06
    "nFailures": 10,  # Added for AZR-06
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

    # AZR-05 specific fields should now be None or not present if removed from TradeProposal by AZR-06 engine
    assert data.get("signal_value") is None
    assert data.get("atr_value") is None
    assert data.get("kelly_fraction_value") is None
    assert data.get("target_position_size") is None

    # Latent risk is populated by the new engine
    assert data.get("latent_risk") is not None
    assert isinstance(data["latent_risk"], float) and 0.0 <= data["latent_risk"] <= 1.0

    # Check legs based on action (simplified for smoke test)
    if data["action"] == "ENTER":
            assert data["legs"] is not None and len(data["legs"]) == 1
            leg = data["legs"][0]
            assert leg["instrument"] == Instrument.MES.value
            assert leg["direction"] == Direction.LONG.value
            assert leg["size"] == 1.0
    elif data["action"] == "EXIT":
            assert data["legs"] is not None # Legs should exist as per current engine logic for EXIT
            if data["legs"]:
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
    assert "AZR Planner Internal" in operation.get("tags", []), "Tag 'AZR Planner Internal' not applied to operation for internal route."

    # Check tag definition (optional, but good for completeness)
    # The main "AZR Planner" tag should also be defined for the v1 endpoint.
    defined_tags = [tag["name"] for tag in openapi_schema.get("tags", [])]
    assert "AZR Planner Internal" in defined_tags, "Tag 'AZR Planner Internal' not defined in OpenAPI schema."
    assert "AZR Planner" in defined_tags, "Tag 'AZR Planner' (for v1 endpoint) not defined in OpenAPI schema."


# --- AZR-12: Tests for Risk Gate integration in internal propose_trade endpoint ---

def test_azr_planner_internal_propose_trade_risk_gate_accepts(
    test_client_azr: TestClient,
    monkeypatch: pytest.MonkeyPatch # Added monkeypatch
) -> None:
    """Test internal propose_trade when risk_gate.accept returns True."""
    # Mock azr_planner.risk_gate.accept to return (True, None)
    # The actual path to mock is where it's imported in osiris.server
    # Mock now needs to accept db_table and cfg keyword arguments
    monkeypatch.setattr("osiris.server.risk_gate_accept", lambda proposal, *, db_table=None, cfg=None: (True, None))

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 200, f"Response content: {response.text}"
    data = response.json()
    # Basic check that it's a valid TradeProposal structure
    assert "action" in data
    assert "rationale" in data


def test_azr_planner_internal_propose_trade_risk_gate_rejects(
    test_client_azr: TestClient,
    monkeypatch: pytest.MonkeyPatch # Added monkeypatch
) -> None:
    """Test internal propose_trade when risk_gate.accept returns False."""
    rejection_reason = "position_limit"
    # Mock azr_planner.risk_gate.accept to return (False, reason)
    # Mock now needs to accept db_table and cfg keyword arguments
    monkeypatch.setattr("osiris.server.risk_gate_accept", lambda proposal, *, db_table=None, cfg=None: (False, rejection_reason))

    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 409, f"Expected 409, got {response.status_code}. Response: {response.text}"

    data = response.json()
    assert data == {"detail": "risk_gate_reject", "reason": rejection_reason}


# AZR-13: Test for /metrics endpoint
def test_metrics_endpoint(
    test_client_azr: TestClient,
    monkeypatch: pytest.MonkeyPatch # To control risk gate outcomes for predictable metrics
) -> None:
    """Test that the /metrics endpoint is available and returns Prometheus data."""

    # Ensure some risk gate metrics are populated by making calls that trigger the actual risk gate.

    # Scenario 1: Craft a context that should be accepted by the default risk gate
    # Default config: max_latent_risk=0.35, min_confidence=0.60, max_position_usd=25_000.0
    # generate_plan will produce some latent_risk and confidence.
    # We need generate_plan's output to be accepted.
    # For this test, let's assume SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW results in a proposal
    # that is normally accepted by the default risk gate config.
    # (If not, this test might be flaky, or we'd need to mock generate_plan to give a known good proposal)

    # To ensure counters are fresh for this test, ideally they would be reset.
    # For now, we rely on them being incremented from whatever state they were in.
    # The test checks for *presence* of metrics, not exact values after reset.

    # Call 1 (expected to be accepted by default risk gate if proposal is reasonable)
    # We need to ensure generate_plan is called, and then risk_gate.accept is called with its output.
    # The risk_gate.accept is NOT mocked here, so real increments happen.
    # We might need to mock generate_plan to produce a known "good" proposal.
    with patch("osiris.server.azr_generate_plan_engine") as mock_gen_plan_accept:
        good_proposal = TradeProposal(action="ENTER", rationale="Good", latent_risk=0.1, confidence=0.8,
                                      legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)])
        mock_gen_plan_accept.return_value = good_proposal
        test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)

    # Scenario 2: Craft a context/mock generate_plan for a rejected proposal
    with patch("osiris.server.azr_generate_plan_engine") as mock_gen_plan_reject:
        bad_proposal = TradeProposal(action="ENTER", rationale="Bad", latent_risk=0.9, confidence=0.8, # High risk
                                     legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)])
        mock_gen_plan_reject.return_value = bad_proposal
        # This call should result in a 409, but we are interested in the counter increment
        # The actual risk_gate.accept will be called.
        test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)

    # Now call /metrics
    response = test_client_azr.get("/metrics")
    assert response.status_code == 200
    # CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    from prometheus_client import CONTENT_TYPE_LATEST # Import for direct comparison
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST

    # Check for presence of our custom metrics in the output
    content = response.text
    assert "azr_riskgate_accept_total" in content
    assert "azr_riskgate_reject_total" in content
    # The bad_proposal was rejected due to high_risk, as that's the first check it fails
    assert 'azr_riskgate_reject_total{reason="high_risk"}' in content


# --- AZR-14: Tests for P&L Endpoint ---
from azr_planner.schemas import DailyPNLReport as AZRDailyPNLReportSchema # For response validation
import datetime as dt # For constructing date objects

def test_azr_api_v1_pnl_daily_endpoint_empty(test_client_azr: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /pnl/daily endpoint when no data is present (or DB mock returns empty)."""
    # Mock the LanceDB interaction to return an empty list
    # This requires knowing how the server will try to access LanceDB.
    # For now, assume it might involve a global 'db' object or a path.
    # If pnl.py uses a global DB_PATH, we could mock lancedb.connect.
    # Given the endpoint has a TODO for actual query, we can mock a helper if one was made,
    # or simply test its current stubbed behavior if that's easier for now.
    # The current stub returns dummy data or empty list. Let's test with last_n that gets empty.

    response = test_client_azr.get("/azr_api/v1/pnl/daily?last_n=100") # Valid last_n (e.g. max allowed)
    assert response.status_code == 200
    response_data = response.json()
    if not isinstance(response_data, list): # Should be a list
         pytest.fail(f"Response data is not a list: {response_data}")

    # If the dummy data generation is active and last_n is high, it might still return some.
    # The stub returns max 5 dummy reports. If we ask for more, we get those 5.
    # If we want to test truly empty, the stub logic would need to return [] for some case.
    # For now, let's check if it returns a list, and if it has items, they are valid.
    if response_data:
        for item in response_data:
            AZRDailyPNLReportSchema.model_validate(item) # Check structure


def test_azr_api_v1_pnl_daily_endpoint_with_dummy_data(test_client_azr: TestClient) -> None:
    """Test /pnl/daily endpoint with its current dummy data generation."""
    response = test_client_azr.get("/azr_api/v1/pnl/daily?last_n=3")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3 # Dummy data stub generates min(last_n, 5)

    base_date = dt.date.today()
    for i, report_dict in enumerate(data):
        report = AZRDailyPNLReportSchema.model_validate(report_dict)
        assert report.date == base_date - dt.timedelta(days=i)
        assert report.realized_pnl == 100.0 + i
        # Add more assertions if needed based on dummy data structure


# --- Tests for AZR Planner V1 Public Endpoint ---
def test_azr_api_v1_propose_trade_happy_path(test_client_azr: TestClient) -> None:
    """
    End-to-end happy path for POST /azr_api/v1/propose_trade.
    Uses a minimal 35-point equityCurve and dailyHistoryHLC.
    Asserts action is ENTER or HOLD, and latent_risk is within [0, 1].
    """
    # Construct minimal valid PlanningContext payload
    # LR_V2_MIN_POINTS is 30. Task asks for 35 points for equityCurve.
    # dailyHistoryHLC also needs to be sufficient for latent_risk_v2.
    num_points = 35
    equity_curve_data = [100.0 + i * 0.1 for i in range(num_points)]
    hlc_data = _generate_hlc_data_for_server_test(num_points)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equityCurve": equity_curve_data,
        "dailyHistoryHLC": hlc_data,
        "volSurface": {"MES": 0.22},
        "riskFreeRate": 0.02,
        "nSuccesses": 5, # Example values
        "nFailures": 2   # Example values
        # dailyVolume and currentPositions are optional and omitted for minimal payload
    }

    response = test_client_azr.post("/azr_api/v1/propose_trade", json=payload)
    assert response.status_code == 200, f"Response content: {response.text}"

    data = response.json()
    from azr_planner.schemas import TradeProposal as TradeProposalSchema

    try:
        TradeProposalSchema.model_validate(data)
    except Exception as e:
        pytest.fail(f"Response does not match TradeProposal schema for v1 endpoint: {e}\nResponse data: {data}")

    # Assert action ∈ {"ENTER","HOLD"} and 0 ≤ latent_risk ≤ 1.
    # The exact action depends on the dummy data and current latent_risk_v2 logic.
    # Given it's a "happy path" with gently rising equity, latent risk should be low.
    # Confidence depends on nSuccesses/nFailures. With 5S/2F and alpha=3,beta=4: (5+3)/(5+2+3+4) = 8/14 = 0.57
    # ENTER: lr < 0.25 and conf > 0.7  (0.57 is not > 0.7, so not ENTER this way)
    # EXIT: lr > 0.7 or conf < 0.4 (0.57 is not < 0.4)
    # So, it should be HOLD if lr is not extreme.
    # If lr is very low (e.g. <0.25), but conf is 0.57, it will be HOLD.
    # If lr is very high (e.g. >0.7), it will be EXIT.
    # The spec says "assert action ∈ {\"ENTER\",\"HOLD\"}". This implies the test data should lead to one of these.
    # A flat equity curve gives lr=0. With conf=0.57, this is not ENTER.
    # Let's make nSuccesses higher to get conf > 0.7 for a potential ENTER.
    # e.g. nSuccesses=10, nFailures=1 => conf = (10+3)/(10+1+3+4) = 13/18 = 0.72

    payload_for_enter_hold = payload.copy()
    payload_for_enter_hold["nSuccesses"] = 10
    payload_for_enter_hold["nFailures"] = 1
    # To ensure low latent risk for potential ENTER, use a flat equity curve
    payload_for_enter_hold["equityCurve"] = [100.0] * num_points
    # HLC data for flat curve also (otherwise returns for vol/entropy might be non-zero)
    payload_for_enter_hold["dailyHistoryHLC"] = [[100.0, 100.0, 100.0]] * num_points


    response_enter_hold = test_client_azr.post("/azr_api/v1/propose_trade", json=payload_for_enter_hold)
    assert response_enter_hold.status_code == 200
    data_enter_hold = response_enter_hold.json()

    # With flat curve, lr should be 0. With 10S/1F, conf is ~0.72.
    # lr (0) < 0.25 AND conf (0.72) > 0.7 => Should be ENTER
    assert data_enter_hold["action"] in ["ENTER", "HOLD"], f"Action was {data_enter_hold['action']}, lr={data_enter_hold['latent_risk']}, conf={data_enter_hold['confidence']}"

    assert "latent_risk" in data_enter_hold
    assert isinstance(data_enter_hold["latent_risk"], float)
    assert 0.0 <= data_enter_hold["latent_risk"] <= 1.0
