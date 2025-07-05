import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

for mod in ["numpy", "sounddevice", "whisper", "torch", "webrtcvad", "intent_router"]:
    sys.modules.setdefault(mod, MagicMock())

osiris_mod = types.ModuleType("osiris")
tts_mod = types.ModuleType("osiris.tts")
tts_mod.speak = MagicMock()
osiris_mod.tts = tts_mod
sys.modules.setdefault("osiris", osiris_mod)
sys.modules.setdefault("osiris.tts", tts_mod)

import live_transcribe


def test_voice_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    utterances = ["hello", "world", "bye"]
    responses = [f"resp-{i}" for i in range(3)]

    def fake_worker() -> None:
        for text in utterances:
            resp = live_transcribe.intent_router.route_and_respond(text, live_transcribe.ctx)
            live_transcribe.tts.speak(resp)

    monkeypatch.setattr(live_transcribe, "transcribe_worker", fake_worker)

    class DummyStream:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(live_transcribe.sd, "InputStream", DummyStream)
    monkeypatch.setattr("builtins.input", lambda *a, **k: None)
    monkeypatch.setattr(live_transcribe, "audio_q", SimpleNamespace(put=lambda *a, **k: None))

    def fake_route(text: str, ctx: dict) -> str:
        assert ctx is live_transcribe.ctx
        return responses.pop(0)

    monkeypatch.setattr(live_transcribe.intent_router, "route_and_respond", fake_route)
    speak_mock = MagicMock()
    monkeypatch.setattr(live_transcribe.tts, "speak", speak_mock)

    live_transcribe.main()

    assert speak_mock.call_count == 3
    assert speak_mock.call_args_list[0].args[0] == "resp-0"
    assert speak_mock.call_args_list[1].args[0] == "resp-1"
    assert speak_mock.call_args_list[2].args[0] == "resp-2"
