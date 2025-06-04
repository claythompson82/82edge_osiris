from unittest.mock import patch, MagicMock, AsyncMock
import uvicorn

# Patch heavy model loading before importing server
with (
    patch("llm_sidecar.loader.load_hermes_model"),
    patch("llm_sidecar.loader.load_phi3_model"),
    patch("llm_sidecar.tts.ChatterboxTTS"),
):
    import osiris.server as server

# Patch generation helpers to return quickly
server._generate_hermes_text = AsyncMock(return_value="ok")
server._generate_phi3_json = AsyncMock(return_value={"ok": True})
server.hermes_model = MagicMock()
server.hermes_tokenizer = MagicMock()
server.phi3_model = MagicMock()
server.phi3_tokenizer = MagicMock()

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
