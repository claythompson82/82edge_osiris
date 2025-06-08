import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import io
import wave
import datetime
import uuid

# Import the FastAPI app instance from server.py
from osiris.server import app, db, FeedbackItem

@pytest.fixture(autouse=True)
def patch_model_loaders():
    """Automatically patch model loaders for all tests in this file."""
    with (
        patch("osiris.server.load_hermes_model", return_value=None),
        patch("osiris.server.load_phi3_model", return_value=None),
        patch("osiris.server.event_bus.connect", new_callable=AsyncMock),
        patch("osiris.server.event_bus.close", new_callable=AsyncMock),
        patch("osiris.server.event_bus.subscribe", new_callable=AsyncMock),
    ):
        yield

def test_generate_hermes_default_model_id():
    """Test /generate/ with default model_id (hermes)"""
    with (
        patch("osiris.server.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_hermes,
        patch("osiris.server._generate_hermes_text", new_callable=AsyncMock, return_value="Hermes mock output") as mock_generate_hermes,
    ):
        client = TestClient(app)
        response = client.post("/generate/", json={"prompt": "test prompt for hermes default", "max_length": 256})
        assert response.status_code == 200
        assert response.json() == {"generated_text": "Hermes mock output"}
        mock_get_hermes.assert_called_once()

def test_generate_phi3_explicit_model_id():
    """Test /generate/ with explicit model_id='phi3'"""
    mock_phi3_output = {"phi3_mock_output": "success"}
    with (
        patch("osiris.server.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())) as mock_get_phi3,
        patch("osiris.server._generate_phi3_json", new_callable=AsyncMock, return_value=mock_phi3_output) as mock_generate_phi3,
    ):
        client = TestClient(app)
        response = client.post("/generate/", json={"prompt": "test prompt for phi3", "model_id": "phi3", "max_length": 256})
        assert response.status_code == 200
        assert response.json() == mock_phi3_output
        mock_get_phi3.assert_called_once()

@pytest.mark.parametrize("bad_id", ["invalid_model", "nonsense", "bad"])
def test_generate_invalid_model_id(bad_id: str):
    """/generate/ rejects unknown model_id values with 422"""
    client = TestClient(app)
    response = client.post("/generate/", json={"prompt": "test prompt", "model_id": bad_id, "max_length": 256})
    assert response.status_code == 422

def test_score_proposal_with_hermes_success():
    """Test /score/hermes/ endpoint successful scoring."""
    with (
        patch("osiris.server.score_with_hermes", return_value=0.75) as mock_score_func,
        patch("osiris.server.db.log_hermes_score", return_value=None) as mock_log_score,
    ):
        client = TestClient(app)
        payload = {"proposal": {"ticker": "XYZ", "action": "BUY"}, "context": "Test context"}
        response = client.post("/score/hermes/", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "proposal_id" in response_data
        assert response_data["score"] == 0.75
        mock_score_func.assert_called_once_with(payload["proposal"], payload["context"])
        mock_log_score.assert_called_once()

def test_health_endpoint_db_query_exception():
    """Test /health endpoint when DB query raises an exception."""
    mock_table = MagicMock()
    mock_table.search.side_effect = Exception("Simulated DB error")
    with (
        patch("osiris.server.logger.error") as mock_logger_error,
        patch.dict(db._tables, {"hermes_scores": mock_table}),
    ):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["mean_hermes_score_last_24h"] is None
        mock_logger_error.assert_called_with("Error calculating mean Hermes score for health check: Simulated DB error")

def test_speak_endpoint():
    """Test /speak endpoint for TTS"""
    mock_audio_data = b"RIFFxxxxWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00dataxxxx"
    with patch(
        "osiris.server.tts_model.synth",
        new_callable=AsyncMock,
        return_value=mock_audio_data,
    ) as mock_synth:
        client = TestClient(app)
        response = client.post("/speak", json={"text": "hello world"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")