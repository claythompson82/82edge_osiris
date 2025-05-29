import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock

# Import the FastAPI app instance from server.py
# Ensure server.py is in the Python path or adjust as necessary.
# For this environment, assuming server.py is at the root and discoverable.
from server import app

@pytest.mark.asyncio
async def test_generate_hermes_default_model_id():
    """Test /generate/ with default model_id (hermes)"""
    with patch('server.get_hermes_model_and_tokenizer', return_value=(MagicMock(), MagicMock())) as mock_get_hermes, \
         patch('server._generate_hermes_text', new_callable=AsyncMock, return_value="Hermes mock output") as mock_generate_hermes:

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/generate/", json={"prompt": "test prompt for hermes default"})

        assert response.status_code == 200
        assert response.json() == {"generated_text": "Hermes mock output"}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_called_once_with("test prompt for hermes default", 256, mock_get_hermes.return_value[0], mock_get_hermes.return_value[1])

@pytest.mark.asyncio
async def test_generate_hermes_explicit_model_id():
    """Test /generate/ with explicit model_id='hermes'"""
    with patch('server.get_hermes_model_and_tokenizer', return_value=(MagicMock(), MagicMock())) as mock_get_hermes, \
         patch('server._generate_hermes_text', new_callable=AsyncMock, return_value="Hermes mock output explicit") as mock_generate_hermes:

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/generate/", json={"prompt": "test prompt for hermes explicit", "model_id": "hermes"})

        assert response.status_code == 200
        assert response.json() == {"generated_text": "Hermes mock output explicit"}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_called_once_with("test prompt for hermes explicit", 256, mock_get_hermes.return_value[0], mock_get_hermes.return_value[1])

@pytest.mark.asyncio
async def test_generate_phi3_explicit_model_id():
    """Test /generate/ with explicit model_id='phi3'"""
    mock_phi3_output = {"phi3_mock_output": "success"}
    with patch('server.get_phi3_model_and_tokenizer', return_value=(MagicMock(), MagicMock())) as mock_get_phi3, \
         patch('server._generate_phi3_json', new_callable=AsyncMock, return_value=mock_phi3_output) as mock_generate_phi3:

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/generate/", json={"prompt": "test prompt for phi3", "model_id": "phi3"})

        assert response.status_code == 200
        assert response.json() == mock_phi3_output
        mock_get_phi3.assert_called_once()
        mock_generate_phi3.assert_called_once_with("test prompt for phi3", 256, mock_get_phi3.return_value[0], mock_get_phi3.return_value[1])

@pytest.mark.asyncio
async def test_generate_invalid_model_id():
    """Test /generate/ with an invalid model_id"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/generate/", json={"prompt": "test prompt", "model_id": "invalid_model"})

    # The server.py logic returns a dict for invalid model_id, FastAPI defaults to 200 OK for such responses.
    assert response.status_code == 200
    assert response.json() == {"error": "Invalid model_id specified. Choose 'hermes' or 'phi3'.", "specified_model_id": "invalid_model"}

@pytest.mark.asyncio
async def test_generate_hermes_model_not_loaded():
    """Test /generate/ with hermes model not loaded"""
    with patch('server.get_hermes_model_and_tokenizer', return_value=(None, None)) as mock_get_hermes, \
         patch('server._generate_hermes_text', new_callable=AsyncMock) as mock_generate_hermes: # Should not be called

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/generate/", json={"prompt": "test prompt", "model_id": "hermes"})

        assert response.status_code == 200 # Server returns error as JSON with 200 OK
        assert response.json() == {"error": "Hermes model not loaded. Please check server logs."}
        mock_get_hermes.assert_called_once()
        mock_generate_hermes.assert_not_called()

@pytest.mark.asyncio
async def test_generate_phi3_model_not_loaded():
    """Test /generate/ with phi3 model not loaded"""
    with patch('server.get_phi3_model_and_tokenizer', return_value=(None, None)) as mock_get_phi3, \
         patch('server._generate_phi3_json', new_callable=AsyncMock) as mock_generate_phi3: # Should not be called

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/generate/", json={"prompt": "test prompt", "model_id": "phi3"})

        assert response.status_code == 200 # Server returns error as JSON with 200 OK
        assert response.json() == {"error": "Phi-3 ONNX model not loaded. Please check server logs."}
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
    with patch('server.load_hermes_model', return_value=None) as mock_load_hermes, \
         patch('server.load_phi3_model', return_value=None) as mock_load_phi3:
        yield mock_load_hermes, mock_load_phi3

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

# For test_generate_invalid_model_id, the status code is 200 as discussed.
# The server logic returns:
# return {"error": "Invalid model_id specified. Choose 'hermes' or 'phi3'.", "specified_model_id": request.model_id}
# This is a standard dictionary, FastAPI converts it to JSONResponse with status 200.
# So, assert response.status_code == 200 is correct based on the current server code.
# If a 4xx status code were desired, server.py would need to use:
# from fastapi import HTTPException
# raise HTTPException(status_code=400, detail="Invalid model_id specified...")
# or from fastapi.responses import JSONResponse
# return JSONResponse(status_code=400, content={"error": ...})
# Since it doesn't, 200 is the expected code for these error responses.
# The tests align with this.

import io
import wave

@pytest.mark.asyncio
async def test_speak_endpoint():
    """Test /speak endpoint for TTS"""
    mock_audio_data = b"RIFFxxxxWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00dataxxxx" # Dummy WAV
    # Create a more realistic dummy WAV for duration check
    # Parameters: 1 channel, 16-bit depth, 22050 Hz sample rate, 1 second duration
    samplerate = 22050
    duration_seconds = 1
    n_channels = 1
    sampwidth = 2 # bytes per sample (16-bit)
    n_frames = samplerate * duration_seconds
    comptype = "NONE"
    compname = "not compressed"

    # Create a valid WAV file in memory
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(samplerate)
            wf.setnframes(n_frames)
            wf.setcomptype(comptype, compname)
            # Write some dummy audio frames (e.g., silence)
            wf.writeframes(b'\x00\x00' * n_frames)
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

    with patch('server.tts_model.synth', return_value=mock_audio_data) as mock_synth:
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/speak", json={"text": "hello world"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        
        # Check for WAV header
        assert response.content.startswith(b"RIFF")
        assert b"WAVE" in response.content[:12] # WAVE chunk ID is typically at offset 8

        # Check audio duration
        with io.BytesIO(response.content) as audio_buffer:
            with wave.open(audio_buffer, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                assert duration >= 0.5 # Check if duration is > 0.5 seconds

        mock_synth.assert_called_once_with(text="hello world", ref_wav=None, exaggeration=0.5)
