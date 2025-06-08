import unittest
from unittest.mock import patch, MagicMock
import sys
import requests

# Stub heavy optional dependencies
for name in [
    "torch",
    "torchaudio",
    "chatterbox",
    "sentry_sdk",
    "tenacity",
    "lancedb",
    "lancedb.pydantic",
    "transformers",
    "optimum",
    "optimum.onnxruntime",
    "peft",
]:
    sys.modules.setdefault(name, MagicMock())

# Stub redis modules
import types

redis_module = types.ModuleType("redis")
redis_asyncio = types.ModuleType("redis.asyncio")
client_mod = types.ModuleType("redis.asyncio.client")
client_mod.PubSub = type("_DummyPubSub", (), {})
redis_module.asyncio = redis_asyncio
sys.modules.setdefault("redis", redis_module)
sys.modules.setdefault("redis.asyncio", redis_asyncio)
sys.modules.setdefault("redis.asyncio.client", client_mod)
exceptions_mod = types.ModuleType("redis.exceptions")
exceptions_mod.RedisError = Exception
sys.modules.setdefault("redis.exceptions", exceptions_mod)

# Dynamically import LLMClient without triggering osiris __init__
import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "osiris" / "llm_client.py"
spec = importlib.util.spec_from_file_location("osiris.llm_client", module_path)
llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_module)
LLMClient = llm_module.LLMClient


def _mock_response(status_code=200, json_data=None, raise_for_status=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_data)
    mock_resp.text = str(json_data)
    if raise_for_status:
        mock_resp.raise_for_status = MagicMock(side_effect=raise_for_status)
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestLLMClient(unittest.TestCase):
    def test_retry_on_5xx(self):
        client = LLMClient(
            base_url="http://test", timeout=1, retries=2, backoff_factor=0
        )
        first = _mock_response(status_code=500)
        second = _mock_response(status_code=200, json_data={"ok": True})
        with patch.object(
            client.session, "request", side_effect=[first, second]
        ) as mock_request:
            result = client.generate("phi3", "prompt")
            self.assertEqual(result, {"ok": True})
            self.assertEqual(mock_request.call_count, 2)
            self.assertEqual(mock_request.call_args_list[0][1]["timeout"], 1)

    def test_exceed_retries_raises(self):
        client = LLMClient(base_url="http://test", retries=2, backoff_factor=0)
        error_resp = _mock_response(status_code=500)
        with patch.object(
            client.session, "request", side_effect=[error_resp, error_resp, error_resp]
        ):
            with self.assertRaises(requests.exceptions.HTTPError):
                client.generate("phi3", "prompt")
