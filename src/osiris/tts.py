import os
import requests

RESEMBLE_API_KEY = os.environ.get("RESEMBLE_API_KEY", "your-key-here")
RESEMBLE_PROJECT_UUID = os.environ.get("RESEMBLE_PROJECT_UUID", "project-uuid-here")
RESEMBLE_VOICE_UUID = os.environ.get("RESEMBLE_VOICE_UUID", "voice-uuid-here")
RESEMBLE_BASE_URL = "https://api.resemble.ai/v2"

def text_to_speech(text: str) -> bytes:
    """
    Send text to Resemble AI and return audio bytes.
    """
    url = f"{RESEMBLE_BASE_URL}/projects/{RESEMBLE_PROJECT_UUID}/clips"
    headers = {
        "Authorization": f"Token {RESEMBLE_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "text": text,
        "voice": RESEMBLE_VOICE_UUID,
        "output_format": "wav"
    }
    resp = requests.post(url, json=body, headers=headers)
    resp.raise_for_status()
    # The Resemble API response will include a "path" (URL to download audio)
    audio_url = resp.json().get("path")
    if not audio_url:
        raise RuntimeError("No audio path returned by Resemble API")
    audio_resp = requests.get(audio_url)
    audio_resp.raise_for_status()
    return audio_resp.content


def speak(text: str) -> None:
    """Send ``text`` to the sidecar ``/speak`` endpoint."""
    base_url = os.environ.get("OSIRIS_SIDECAR_URL", "http://localhost:8000").rstrip("/")
    try:
        requests.post(f"{base_url}/speak", json={"text": text}, timeout=5).raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        if os.getenv("OSIRIS_TEST") == "1":
            # In tests the sidecar may not be running.
            print(f"[tts] Chatterbox unavailable: {exc}")
        else:
            raise
