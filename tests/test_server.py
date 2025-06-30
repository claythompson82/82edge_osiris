import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import osiris.server as server # Main import of server
# Import specific schemas needed for tests
from azr_planner.schemas import TradeProposal, Leg, Instrument, Direction, DailyPNLReport as AZRDailyPNLReportSchema
from azr_planner.replay.schemas import ReplayReport as AZRReplayReportSchema

# AZR-15 Replay Test Imports for creating fixture data
import pandas as pd # <--- ADDED IMPORT
import pyarrow as pa # type: ignore[import-untyped]
import pyarrow.parquet as pq # type: ignore[import-untyped]
import gzip
import io
from pathlib import Path
import os # For OSIRIS_TEST env var
import math
from datetime import datetime, timezone, timedelta, date as dt_date # Specific import for date
import importlib # For reloading server module
from typing import List, Tuple, Dict, Any # <--- ADDED IMPORT

# This global 'app' will be updated by the fixture _manage_osiris_test_env_var_for_azr_module
app = server.app

# Standard LLM sidecar/TTS tests
def test_generate_hermes_default_model_id() -> None: # Added return type
    client = TestClient(app)
    with patch("osiris.server._generate_hermes_text", new_callable=AsyncMock, return_value="Hermes mock output"), \
         patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        response = client.post("/generate/", json={"prompt": "test prompt for hermes default", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == {"output": "Hermes mock output"}

def test_generate_phi3_explicit_model_id() -> None: # Added return type
    client = TestClient(app)
    mock_phi3_output = {"phi3_mock_output": "success"}
    with patch("osiris.server._generate_phi3_json", new_callable=AsyncMock, return_value=mock_phi3_output), \
         patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        response = client.post("/generate/", json={"prompt": "test prompt for phi3", "model_id": "phi3", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == mock_phi3_output

def test_score_proposal_with_hermes_success() -> None: # Added return type
    client = TestClient(app)
    with patch("osiris.llm_sidecar.hermes_plugin.score_with_hermes", return_value=0.75), \
         patch("osiris.llm_sidecar.db.log_hermes_score", return_value=None), \
         patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        response = client.post("/score/hermes/", json={"proposal": {"foo": "bar"}, "context": "ctx"})
    assert response.status_code == 200
    assert response.json() == {"score": 0.75}

def test_health_endpoint_with_adapter_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None: # Added type hints
    client = TestClient(app)
    (tmp_path / "2025-06-10").mkdir()
    monkeypatch.setattr("osiris.llm_sidecar.loader.ADAPTER_ROOT", tmp_path, raising=False)
    with patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())), \
         patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())), \
         patch("osiris.llm_sidecar.db.get_mean_hermes_score_last_24h", return_value=0.95):
        response = client.get("/health?adapter_date=true")
        assert response.status_code == 200
        assert response.json()["latest_adapter"] == "2025-06-10"

def test_speak_endpoint(monkeypatch: pytest.MonkeyPatch) -> None: # Added type hints
    client = TestClient(app)
    with patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())), \
         patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())), \
         patch("osiris.server.text_to_speech", return_value=b"dummy_audio"):
        response = client.post("/speak/", json={"text": "Say something"})
    assert response.status_code == 200
    assert response.content == b"dummy_audio"

# --- Tests for AZR Planner ---
def _generate_hlc_data_for_server_test(num_periods: int, start_price: float = 100.0, daily_change: float = 0.1, spread: float = 0.5) -> List[Tuple[float, float, float]]:
    data_list: List[Tuple[float, float, float]] = [] # Explicit type for list
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + abs(daily_change * math.sin(i*0.1))
        low = current_close - spread - abs(daily_change * math.cos(i*0.1))
        close = (high + low) / 2 + (math.sin(i*0.5) * spread*0.1)
        low = min(low, high - 0.01); close = max(min(close, high), low)
        data_list.append((round(high,2), round(low,2), round(close,2)))
        current_close = close
    return data_list

MIN_HISTORY_POINTS_SERVER_TEST = 65
SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW: Dict[str, Any] = { # Explicit type for dict
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "equityCurve": [10000.0 + i*10 for i in range(35)],
    "dailyHistoryHLC": _generate_hlc_data_for_server_test(MIN_HISTORY_POINTS_SERVER_TEST),
    "dailyVolume": [float(10000 + i*100) for i in range(MIN_HISTORY_POINTS_SERVER_TEST)],
    "currentPositions": [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0).model_dump()],
    "nSuccesses": 20, "nFailures": 10,
    "volSurface": {"MES": 0.15, "M2K": 0.20}, "riskFreeRate": 0.02,
}

@pytest.fixture(scope="module", autouse=True)
def _manage_osiris_test_env_var_for_azr_module() -> None: # Added return type
    global app # Declare app as global at the start of the fixture
    original_value = os.environ.get("OSIRIS_TEST")
    os.environ["OSIRIS_TEST"] = "1"
    importlib.reload(server)
    app = server.app # Rebind the module-level app
    yield
    if original_value is None:
        if "OSIRIS_TEST" in os.environ: del os.environ["OSIRIS_TEST"]
    else: os.environ["OSIRIS_TEST"] = original_value
    importlib.reload(server)
    app = server.app # Rebind to potentially original state; uses 'global app' from fixture start

@pytest.fixture(scope="function")
def test_client_azr() -> TestClient: return TestClient(app)

def test_azr_planner_new_engine_smoke_test(test_client_azr: TestClient) -> None:
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 200
    TradeProposal.model_validate(response.json())

def test_azr_planner_internal_propose_trade_risk_gate_accepts(test_client_azr: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("osiris.server.risk_gate_accept", lambda proposal, *, db_table=None, cfg=None, registry=None: (True, None))
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 200

def test_azr_planner_internal_propose_trade_risk_gate_rejects(test_client_azr: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    rejection_reason = "position_limit"
    monkeypatch.setattr("osiris.server.risk_gate_accept", lambda proposal, *, db_table=None, cfg=None, registry=None: (False, rejection_reason))
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    assert response.status_code == 409
    assert response.json() == {"detail": "risk_gate_reject", "reason": rejection_reason}

def test_metrics_endpoint(test_client_azr: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with patch("osiris.server.azr_generate_plan_engine") as mock_gen_plan_accept:
        mock_gen_plan_accept.return_value = TradeProposal(action="ENTER", rationale="Good", latent_risk=0.1, confidence=0.8, legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)])
        test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)
    with patch("osiris.server.azr_generate_plan_engine") as mock_gen_plan_reject:
        mock_gen_plan_reject.return_value = TradeProposal(action="ENTER", rationale="Bad", latent_risk=0.9, confidence=0.8, legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)])
        test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW)

    sample_replay_data = [{"timestamp": datetime.now(timezone.utc), "instrument": "MESU24", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}]
    df = pd.DataFrame(sample_replay_data); table = pa.Table.from_pandas(df, preserve_index=False)
    pq_buffer = io.BytesIO(); pq.write_table(table, pq_buffer); pq_buffer.seek(0)
    gz_buffer = io.BytesIO();
    with gzip.GzipFile(fileobj=gz_buffer, mode='wb') as gz_file: gz_file.write(pq_buffer.read())
    gz_buffer.seek(0)
    files_for_replay = {'file': ('dummy.parquet.gz', gz_buffer, 'application/octet-stream')}
    mock_report_obj = AZRReplayReportSchema(replay_id="metrics-test-id", source_file="dummy.parquet.gz",
        replay_start_time_utc=datetime.now(timezone.utc), replay_end_time_utc=datetime.now(timezone.utc),
        replay_duration_seconds=0.01, total_bars_processed=1, proposals_generated=0, proposals_accepted=0, proposals_rejected=0,
        initial_equity=100000.0, final_equity=100000.0, total_return_pct=0.0, max_drawdown_pct=0.0,
        mean_planner_decision_ms=None, equity_curve=[])
    # We are testing if the /metrics endpoint can expose the counter.
    # The actual replay logic and its own counter increment are tested elsewhere.
    # Here, we can directly increment the counter to ensure it's picked up by /metrics.
    # The post to /azr_api/internal/azr/replay is just to ensure the server-side setup for replay
    # (like router registration) doesn't break. The actual replay logic is mocked.
    with patch("osiris.server.replay_run_replay", return_value=(mock_report_obj, [])):
        test_client_azr.post("/azr_api/internal/azr/replay", files=files_for_replay) # Params don't matter due to mock

    # Directly increment the actual counter that /metrics should expose
    from azr_planner.replay.runner import REPLAY_RUNS_TOTAL as actual_server_replay_counter
    actual_server_replay_counter.labels(granularity="server_replay_bars", instrument_group="dummy").inc()

    response = test_client_azr.get("/metrics")
    assert response.status_code == 200
    from prometheus_client import CONTENT_TYPE_LATEST
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    content = response.text
    assert "azr_riskgate_accept_total" in content
    assert 'azr_riskgate_reject_total{reason="high_risk"}' in content
    assert "azr_pnl_reports_total" in content
    assert "azr_replay_runs_total" in content
    assert 'azr_replay_runs_total{granularity="server_replay_bars",instrument_group="dummy"}' in content

def test_azr_api_v1_pnl_daily_endpoint_empty(test_client_azr: TestClient) -> None:
    response = test_client_azr.get("/azr_api/v1/pnl/daily?last_n=100")
    assert response.status_code == 200; response_data = response.json()
    assert isinstance(response_data, list)
    if response_data:
        for item in response_data: AZRDailyPNLReportSchema.model_validate(item)

def test_azr_api_v1_pnl_daily_endpoint_with_dummy_data(test_client_azr: TestClient) -> None:
    response = test_client_azr.get("/azr_api/v1/pnl/daily?last_n=3")
    assert response.status_code == 200; data = response.json()
    assert isinstance(data, list); assert len(data) == 3
    base_date = dt_date.today() # Use dt_date alias
    for i, report_dict in enumerate(data):
        report = AZRDailyPNLReportSchema.model_validate(report_dict)
        assert report.date == base_date - timedelta(days=i) # Use timedelta from datetime
        assert report.realized_pnl == 100.0 + i

def test_azr_api_v1_propose_trade_happy_path(test_client_azr: TestClient) -> None:
    num_points = 35; equity_curve_data = [100.0 + i * 0.1 for i in range(num_points)]
    hlc_data = _generate_hlc_data_for_server_test(num_points)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "equityCurve": equity_curve_data,
               "dailyHistoryHLC": hlc_data, "volSurface": {"MES": 0.22}, "riskFreeRate": 0.02,
               "nSuccesses": 10, "nFailures": 1}
    payload["equityCurve"] = [100.0] * num_points
    payload["dailyHistoryHLC"] = [[100.0,100.0,100.0]] * num_points
    response = test_client_azr.post("/azr_api/v1/propose_trade", json=payload)
    assert response.status_code == 200; data = response.json()
    TradeProposal.model_validate(data)
    assert data["action"] == "ENTER"
    assert 0.0 <= data["latent_risk"] <= 0.01

def test_azr_replay_endpoint_smoke(test_client_azr: TestClient, tmp_path: Path) -> None:
    sample_replay_data = [{"timestamp": datetime.now(timezone.utc) - timedelta(minutes=30), "instrument": "MESU24", "open": 4500.0, "high": 4505.0, "low": 4499.0, "close": 4502.0, "volume": 100.0}]
    df = pd.DataFrame(sample_replay_data); table = pa.Table.from_pandas(df, preserve_index=False)
    pq_buffer = io.BytesIO(); pq.write_table(table, pq_buffer); pq_buffer.seek(0)
    gz_buffer = io.BytesIO();
    with gzip.GzipFile(fileobj=gz_buffer, mode='wb') as gz_file: gz_file.write(pq_buffer.read())
    gz_buffer.seek(0)
    files = {'file': ('bars_mini.parquet.gz', gz_buffer, 'application/octet-stream')}
    mock_report_obj = AZRReplayReportSchema(replay_id="mock-server-replay-id", source_file="bars_mini.parquet.gz",
        replay_start_time_utc=datetime.now(timezone.utc), replay_end_time_utc=datetime.now(timezone.utc),
        replay_duration_seconds=0.01, total_bars_processed=1, proposals_generated=0, proposals_accepted=0, proposals_rejected=0,
        initial_equity=100000.0, final_equity=100000.0, total_return_pct=0.0, max_drawdown_pct=0.0,
        mean_planner_decision_ms=None, equity_curve=[])
    with patch("osiris.server.replay_run_replay", return_value=(mock_report_obj, [])) as mock_runner:
        response = test_client_azr.post("/azr_api/internal/azr/replay", files=files)
    assert response.status_code == 200
    data = response.json(); assert data["replay_id"] == "mock-server-replay-id"; assert data["total_bars_processed"] == 1
    mock_runner.assert_called_once()
    txt_buffer = io.BytesIO(b"text"); files_txt = {'file':('data.txt.gz',txt_buffer,'application/octet-stream')}
    response_txt = test_client_azr.post("/azr_api/internal/azr/replay", files=files_txt)
    assert response_txt.status_code == 400; assert "Invalid file type" in response_txt.json()["detail"]

def test_azr_planner_invalid_input_daily_history_hlc_too_short(test_client_azr: TestClient) -> None:
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW.copy()
    invalid_context["dailyHistoryHLC"] = [(100.0,99.0,99.5)] * 5 # Corrected tuple values
    if invalid_context.get("dailyVolume"): invalid_context["dailyVolume"] = [1000.0] * 5
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422
    assert any("dailyHistoryHLC" in e.get("loc",[]) and "List should have at least 15 items" in e.get("msg","") for e in response.json()["detail"])

def test_azr_planner_invalid_input_missing_daily_history_hlc(test_client_azr: TestClient) -> None:
    invalid_context = SAMPLE_PLANNING_CONTEXT_DATA_AZR_NEW.copy()
    del invalid_context["dailyHistoryHLC"]
    if "dailyVolume" in invalid_context: del invalid_context["dailyVolume"]
    response = test_client_azr.post("/azr_api/internal/azr/planner/propose_trade", json=invalid_context)
    assert response.status_code == 422
    assert any("dailyHistoryHLC" in e.get("loc",[]) and "Field required" in e.get("msg","") for e in response.json()["detail"])

def test_azr_planner_prefix_and_tags_available_when_osiris_test_set(test_client_azr: TestClient) -> None:
    openapi_schema = test_client_azr.get("/openapi.json").json()
    # Path for internal planner was changed to /azr_api/internal/azr/planner/propose_trade
    path_key = "/azr_api/internal/azr/planner/propose_trade"
    assert path_key in openapi_schema["paths"]
    operation = openapi_schema["paths"][path_key].get("post")
    assert operation is not None and "AZR Planner Internal" in operation.get("tags", [])
    defined_tags = [tag["name"] for tag in openapi_schema.get("tags", [])]
    assert "AZR Planner Internal" in defined_tags and "AZR Planner" in defined_tags
