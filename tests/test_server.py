import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Import the FastAPI app instance from server.py
# Ensure server.py is in the Python path or adjust as necessary.
# For this environment, assuming server.py is at the root and discoverable.
from osiris.server import app


@pytest.mark.asyncio
async def test_generate_hermes_default_model_id():
    """Test /generate/ with default model_id (hermes)"""
    with (
        patch(
            "osiris.server.get_hermes_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_get_hermes,
        patch(
            "osiris.server._generate_hermes_text",
            new_callable=AsyncMock,
            return_value="Hermes mock output",
        ) as mock_generate_hermes,
    ):

        with TestClient(app) as client:
            response = client.post(
                "/generate/", json={"prompt": "test prompt for hermes default"}
            )

        assert response.status_code == 200
        assert response.json() == {"generated_text": "Hermes mock output"}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_called_once_with(
            "test prompt for hermes default",
            256,
            mock_get_hermes.return_value[0],
            mock_get_hermes.return_value[1],
        )


@pytest.mark.asyncio
async def test_generate_hermes_explicit_model_id():
    """Test /generate/ with explicit model_id='hermes'"""
    with (
        patch(
            "osiris.server.get_hermes_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_get_hermes,
        patch(
            "osiris.server._generate_hermes_text",
            new_callable=AsyncMock,
            return_value="Hermes mock output explicit",
        ) as mock_generate_hermes,
    ):

        with TestClient(app) as client:
            response = client.post(
                "/generate/",
                json={
                    "prompt": "test prompt for hermes explicit",
                    "model_id": "hermes",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"generated_text": "Hermes mock output explicit"}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_called_once_with(
            "test prompt for hermes explicit",
            256,
            mock_get_hermes.return_value[0],
            mock_get_hermes.return_value[1],
        )


@pytest.mark.asyncio
async def test_generate_phi3_explicit_model_id():
    """Test /generate/ with explicit model_id='phi3'"""
    mock_phi3_output = {"phi3_mock_output": "success"}
    with (
        patch(
            "osiris.server.get_phi3_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_get_phi3,
        patch(
            "osiris.server._generate_phi3_json",
            new_callable=AsyncMock,
            return_value=mock_phi3_output,
        ) as mock_generate_phi3,
    ):

        with TestClient(app) as client:
            response = client.post(
                "/generate/",
                json={"prompt": "test prompt for phi3", "model_id": "phi3"},
            )

        assert response.status_code == 200
        assert response.json() == mock_phi3_output
        mock_get_phi3.assert_called_once()
        mock_generate_phi3.assert_called_once_with(
            "test prompt for phi3",
            256,
            mock_get_phi3.return_value[0],
            mock_get_phi3.return_value[1],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["invalid_model", "nonsense", "bad"])
async def test_generate_invalid_model_id(bad_id: str):
    """/generate/ rejects unknown model_id values with 422"""
    with TestClient(app) as client:
        response = client.post(
            "/generate/", json={"prompt": "test prompt", "model_id": bad_id}
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid model_id specified. Choose 'hermes' or 'phi3'."
    }


@pytest.mark.asyncio
async def test_generate_hermes_model_not_loaded():
    """Test /generate/ with hermes model not loaded"""
    with (
        patch(
            "osiris.server.get_hermes_model_and_tokenizer", return_value=(None, None)
        ) as mock_get_hermes,
        patch(
            "osiris.server._generate_hermes_text", new_callable=AsyncMock
        ) as mock_generate_hermes,
    ):  # Should not be called

        with TestClient(app) as client:
            response = client.post(
                "/generate/", json={"prompt": "test prompt", "model_id": "hermes"}
            )

        assert response.status_code == 200  # Server returns error as JSON with 200 OK
        assert response.json() == {"error": "Hermes model not loaded."}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_not_called()


@pytest.mark.asyncio
async def test_generate_phi3_model_not_loaded():
    """Test /generate/ with phi3 model not loaded"""
    with (
        patch(
            "osiris.server.get_phi3_model_and_tokenizer", return_value=(None, None)
        ) as mock_get_phi3,
        patch(
            "osiris.server._generate_phi3_json", new_callable=AsyncMock
        ) as mock_generate_phi3,
    ):  # Should not be called

        with TestClient(app) as client:
            response = client.post(
                "/generate/", json={"prompt": "test prompt", "model_id": "phi3"}
            )

        assert response.status_code == 200  # Server returns error as JSON with 200 OK
        assert response.json() == {"error": "Phi-3 model not loaded."}
        mock_get_phi3.assert_called_once()
        mock_generate_phi3.assert_not_called()


# To run these tests, you would typically use:
# pytest tests/test_server.py
# Ensure pytest, pytest-asyncio, and httpx are installed in your environment.
# pip install pytest pytest-asyncio httpx
# The server.py file should be in the PYTHONPATH.
# If running from the root of the project, it usually works out.
# Example: PYTHONPATH=. pytest tests/test_server.py
#
# Also, the server.py uses MICRO_LLM_MODEL_PATH which is loaded at startup.
# While tests mock out model loading for specific endpoints, the initial model loading
# at server startup might still try to access this path.
# For a fully isolated test, one might need to mock os.path.exists or the load_..._model functions
# globally if they interfere with test setup, or set an environment variable for MICRO_LLM_MODEL_PATH
# if the server code is designed to be configurable that way for testing.
# However, the tests above focus on the endpoint logic and mock out the direct interactions
# during the request-response cycle, so they should be fine as long as server.py can be imported.
# The `load_hermes_model()` and `load_phi3_model()` are called at import time in server.py.
# We should patch these out too for truly isolated tests.


@pytest.fixture(autouse=True)
def patch_model_loaders():
    """Automatically patch model loaders for all tests in this file."""
    with (
        patch("osiris.server.load_hermes_model", return_value=None) as mock_load_hermes,
        patch("osiris.server.load_phi3_model", return_value=None) as mock_load_phi3,
        patch("osiris.server.event_bus.connect", new_callable=AsyncMock) as mock_connect,
        patch("osiris.server.event_bus.close", new_callable=AsyncMock) as mock_close,
        patch("osiris.server.event_bus.subscribe", new_callable=AsyncMock) as mock_subscribe,
    ):
        yield mock_load_hermes, mock_load_phi3, mock_connect, mock_close, mock_subscribe


# The above fixture will mock the global model loading functions called at server startup.
# This makes the tests more robust by preventing side effects from these startup calls.
# For example, if MICRO_LLM_MODEL_PATH was not set, server.py might log errors or fail
# during import if load_phi3_model() isn't robust against it.
# This fixture ensures these startup loaders are benign during testing.
# The individual tests then mock get_..._model_and_tokenizer for endpoint-specific behavior.

# Note on AsyncMock:
# unittest.mock.AsyncMock is available in Python 3.8+.
# If using an older Python, an alternative like `asynctest.mock.CoroutineMock` (from asynctest library)
# or `MagicMock(return_value=asyncio.Future())` and setting result on future might be needed.
# Assuming Python 3.8+ for AsyncMock.
# The prompt environment should support this.

# Final check on assertions:
# The max_length is defaulted to 256 in the Pydantic model.
# The calls to _generate_hermes_text and _generate_phi3_json inside the endpoint
# use request.max_length, so the mocked functions should be asserted with this default value.
# Added this to the `assert_called_once_with` for relevant tests.

# For test_generate_invalid_model_id we now expect a 422 response. The endpoint
# raises an HTTPException with the detail string
# "Invalid model_id specified. Choose 'hermes' or 'phi3'." whenever the
# `model_id` is anything other than "hermes" or "phi3". The test above
# parameterizes several bad values and confirms the JSON body matches.

import io
import wave
import datetime
import uuid  # For checking proposal_id type in score tests
from osiris.server import db  # To access db._tables for mocking if needed


@pytest.mark.asyncio
async def test_score_proposal_with_hermes_success():
    """Test /score/hermes/ endpoint successful scoring."""
    with (
        patch("osiris.server.score_with_hermes", return_value=0.75) as mock_score_func,
        patch("osiris.server.db.log_hermes_score", return_value=None) as mock_log_score,
    ):

        with TestClient(app) as client:
            payload = {
                "proposal": {"ticker": "XYZ", "action": "BUY"},
                "context": "Test context",
            }
            response = client.post("/score/hermes/", json=payload)

        assert response.status_code == 200
        response_data = response.json()
        assert "proposal_id" in response_data
        # Attempt to parse proposal_id as UUID to ensure it's a valid UUID string
        try:
            uuid.UUID(response_data["proposal_id"])
        except ValueError:
            pytest.fail(
                f"proposal_id '{response_data['proposal_id']}' is not a valid UUID string."
            )

        assert response_data["score"] == 0.75

        # server.py runs score_with_hermes in an executor, so the mock should still capture the call.
        # The actual function passed to run_in_executor is score_with_hermes from the server's scope.
        # So, 'server.score_with_hermes' is the correct target.
        # Check call args on the mock_score_func. If run_in_executor is used,
        # the direct call might be harder to assert perfectly without knowing executor internals
        # or if the mock behaves differently with it. Assuming standard patching works:
        # For run_in_executor, the function and its arguments are passed to the executor.
        # The mock should capture the call to the *original* function if that's what's passed.
        # The patch replaces 'server.score_with_hermes' with a mock. This mock is then passed
        # to run_in_executor. So, the mock itself is called.
        mock_score_func.assert_called_once_with(payload["proposal"], payload["context"])

        mock_log_score.assert_called_once()
        args, _ = mock_log_score.call_args
        assert args[0].score == 0.75
        assert isinstance(args[0].proposal_id, uuid.UUID)


@pytest.mark.asyncio
async def test_score_proposal_with_hermes_failure():
    """Test /score/hermes/ endpoint when scoring fails."""
    with (
        patch("osiris.server.score_with_hermes", return_value=-1.0) as mock_score_func,
        patch("osiris.server.db.log_hermes_score") as mock_log_score,
    ):  # Should not be called

        with TestClient(app) as client:
            payload = {"proposal": {"ticker": "ABC", "action": "SELL"}}
            response = client.post("/score/hermes/", json=payload)

        assert response.status_code == 500
        response_data = response.json()
        assert "Failed to score proposal" in response_data.get("detail", "")

        mock_score_func.assert_called_once_with(payload["proposal"], None)
        mock_log_score.assert_not_called()


@pytest.mark.asyncio
async def test_score_proposal_with_hermes_db_log_failure():
    """Test /score/hermes/ endpoint when DB logging fails after successful scoring."""
    with (
        patch("osiris.server.score_with_hermes", return_value=0.8) as mock_score_func,
        patch(
            "osiris.server.db.log_hermes_score", side_effect=Exception("DB log error")
        ) as mock_log_score,
    ):

        with TestClient(app) as client:
            payload = {
                "proposal": {"ticker": "DEF", "action": "BUY"},
                "context": "DB log fail test",
            }
            response = client.post("/score/hermes/", json=payload)

        assert response.status_code == 500
        response_data = response.json()
        # Check for the specific error message related to logging failure
        assert "Score generated but failed to log" in response_data.get("detail", "")
        assert "DB log error" in response_data.get(
            "detail", ""
        )  # Ensure original error is mentioned

        mock_score_func.assert_called_once_with(payload["proposal"], payload["context"])
        mock_log_score.assert_called_once()


@pytest.mark.asyncio
async def test_health_endpoint_with_recent_scores():
    """Test /health endpoint when recent Hermes scores are present."""
    now = datetime.datetime.now(datetime.timezone.utc)
    scores_data = [
        {"score": 0.8, "timestamp": (now - datetime.timedelta(hours=1)).isoformat()},
        {"score": 0.6, "timestamp": (now - datetime.timedelta(hours=2)).isoformat()},
    ]
    expected_mean = (0.8 + 0.6) / 2

    mock_table = MagicMock()
    mock_search_result = MagicMock()
    mock_where_result = MagicMock()
    mock_select_result = MagicMock()

    mock_table.search.return_value = mock_search_result
    mock_search_result.where.return_value = mock_where_result
    mock_where_result.select.return_value = mock_select_result
    mock_select_result.to_list.return_value = scores_data

    with patch.dict(db._tables, {"hermes_scores": mock_table}):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mean_hermes_score_last_24h"] == expected_mean
    assert data["num_hermes_scores_last_24h"] == len(scores_data)
    assert "status" in data
    assert "hermes_loaded" in data


@pytest.mark.asyncio
async def test_health_endpoint_no_recent_scores():
    """Test /health endpoint when no recent Hermes scores are present."""
    mock_table = MagicMock()
    mock_search_result = MagicMock()
    mock_where_result = MagicMock()
    mock_select_result = MagicMock()

    mock_table.search.return_value = mock_search_result
    mock_search_result.where.return_value = mock_where_result
    mock_where_result.select.return_value = mock_select_result
    mock_select_result.to_list.return_value = []

    with patch.dict(db._tables, {"hermes_scores": mock_table}):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mean_hermes_score_last_24h"] is None
    assert data["num_hermes_scores_last_24h"] == 0
    assert "status" in data


@pytest.mark.asyncio
async def test_health_endpoint_db_table_not_found():
    """Test /health endpoint when hermes_scores table is not found."""
    with patch.dict(db._tables, {"hermes_scores": None}):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mean_hermes_score_last_24h"] is None
    assert data["num_hermes_scores_last_24h"] == 0
    assert "status" in data


@pytest.mark.asyncio
async def test_health_endpoint_db_query_exception():
    """Test /health endpoint when DB query raises an exception."""
    mock_table = MagicMock()
    mock_table.search.side_effect = Exception("Simulated DB error")

    # Also mock logger to check if the error is logged
    with (
        patch("osiris.server.logger.error") as mock_logger_error,
        patch.dict(db._tables, {"hermes_scores": mock_table}),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mean_hermes_score_last_24h"] is None
    assert data["num_hermes_scores_last_24h"] == 0
    assert "status" in data
    # Ensure the expected error was logged at least once
    assert mock_logger_error.call_args_list[-1][0][0] == (
        "Error calculating mean Hermes score for health check: Simulated DB error"
    )


@pytest.mark.asyncio
async def test_speak_endpoint():
    """Test /speak endpoint for TTS"""
    mock_audio_data = b"RIFFxxxxWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00dataxxxx"  # Dummy WAV
    # Create a more realistic dummy WAV for duration check
    # Parameters: 1 channel, 16-bit depth, 22050 Hz sample rate, 1 second duration
    samplerate = 22050
    duration_seconds = 1
    n_channels = 1
    sampwidth = 2  # bytes per sample (16-bit)
    n_frames = samplerate * duration_seconds
    comptype = "NONE"
    compname = "not compressed"

    # Create a valid WAV file in memory
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(samplerate)
            wf.setnframes(n_frames)
            wf.setcomptype(comptype, compname)
            # Write some dummy audio frames (e.g., silence)
            wf.writeframes(b"\x00\x00" * n_frames)
        mock_audio_data = wav_io.getvalue()

    # Patch the tts_model.synth method within the server's context
    # Ensure that server.tts_model is already initialized or mock its initialization if necessary
    # For this test, we assume tts_model is an attribute of the app or otherwise accessible
    # and its `synth` method is what we need to mock.
    # If tts_model is initialized globally in server.py like `tts_model = ChatterboxTTS(...)`,
    # then patching 'server.tts_model.synth' is correct.

    # We also need to ensure tts_model itself is not None if server.py has logic like:
    # if not tts_model: raise HTTPException(...)
    # The patch_model_loaders fixture handles LLM models, but not necessarily TTS model.
    # Let's assume tts_model is initialized. If not, we might need to patch 'server.ChatterboxTTS'.

    with patch(
        "osiris.server.tts_model.synth", return_value=mock_audio_data
    ) as mock_synth:
        with TestClient(app) as client:
            response = client.post("/speak", json={"text": "hello world"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

        # Check for WAV header
        assert response.content.startswith(b"RIFF")
        assert (
            b"WAVE" in response.content[:12]
        )  # WAVE chunk ID is typically at offset 8

        # Check audio duration
        with io.BytesIO(response.content) as audio_buffer:
            with wave.open(audio_buffer, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                assert duration >= 0.5  # Check if duration is > 0.5 seconds

        mock_synth.assert_called_once_with(
            text="hello world", ref_wav=None, exaggeration=0.5
        )
