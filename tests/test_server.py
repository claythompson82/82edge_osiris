import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import osiris.server as server

app = server.app

def test_generate_hermes_default_model_id():
    # Tests /generate/ with default model_id (hermes)
    client = TestClient(app)
    with patch("osiris.server._generate_hermes_text", new_callable=AsyncMock, return_value="Hermes mock output"):
        with patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
            response = client.post("/generate/", json={"prompt": "test prompt for hermes default", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == {"output": "Hermes mock output"}

def test_generate_phi3_explicit_model_id():
    # Tests /generate/ with explicit model_id='phi3'
    client = TestClient(app)
    mock_phi3_output = {"phi3_mock_output": "success"}
    with patch("osiris.server._generate_phi3_json", new_callable=AsyncMock, return_value=mock_phi3_output):
        with patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
            response = client.post("/generate/", json={"prompt": "test prompt for phi3", "model_id": "phi3", "max_length": 256})
    assert response.status_code == 200
    assert response.json() == mock_phi3_output

def test_score_proposal_with_hermes_success():
    # Tests /score/hermes/ endpoint successful scoring
    client = TestClient(app)
    with (
        patch("osiris.llm_sidecar.hermes_plugin.score_with_hermes", return_value=0.75),
        patch("osiris.llm_sidecar.db.log_hermes_score", return_value=None),
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock()))
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
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
        patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
        patch("osiris.llm_sidecar.db.get_mean_hermes_score_last_24h", return_value=0.95),
    ):
        response = client.get("/health?adapter_date=true")
        assert response.status_code == 200
        body = response.json()
        assert body["latest_adapter"] == "2025-06-10"

def test_speak_endpoint(monkeypatch):
    # Tests /speak endpoint (dummy test, patching all dependencies)
    client = TestClient(app)
    with (
        patch("osiris.llm_sidecar.loader.get_hermes_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
        patch("osiris.llm_sidecar.loader.get_phi3_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
        patch("osiris.server.text_to_speech", return_value=b"dummy_audio") as mock_tts,
    ):
        response = client.post("/speak/", json={"text": "Say something"})
        assert response.status_code == 200
        assert response.content == b"dummy_audio"
