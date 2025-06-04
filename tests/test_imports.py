import importlib
import sys
import types
from unittest.mock import MagicMock

# Provide stubs for heavy optional dependencies
for name in [
    'torch',
    'torchaudio',
    'chatterbox',
    'sentry_sdk',
    'lancedb',
    'lancedb.pydantic',
    'transformers',
    'optimum',
    'optimum.onnxruntime',
    'peft',
]:
    sys.modules.setdefault(name, MagicMock())

# Build stub modules for redis
redis_module = types.ModuleType('redis')
redis_asyncio = types.ModuleType('redis.asyncio')
client_mod = types.ModuleType('redis.asyncio.client')
exceptions_mod = types.ModuleType('redis.exceptions')
class _DummyPubSub: pass
client_mod.PubSub = _DummyPubSub
class _DummyRedis: pass
redis_asyncio.Redis = _DummyRedis
redis_asyncio.from_url = MagicMock(return_value=_DummyRedis())
redis_module.asyncio = redis_asyncio
exceptions_mod.RedisError = Exception
sys.modules.setdefault('redis', redis_module)
sys.modules.setdefault('redis.asyncio', redis_asyncio)
sys.modules.setdefault('redis.asyncio.client', client_mod)
sys.modules.setdefault('redis.exceptions', exceptions_mod)

# Patch heavy model loading and TTS initialisation before importing server
import llm_sidecar.loader as loader
import llm_sidecar.tts as tts

loader.load_hermes_model = MagicMock()
loader.load_phi3_model = MagicMock()

class _DummyTTS:
    def __init__(self, *args, **kwargs):
        pass

tts.ChatterboxTTS = _DummyTTS

MODULES = [
    "osiris",
    "osiris.llm_sidecar",
    "osiris.llm_sidecar.server",
    "llm_sidecar.event_bus",
    "llm_sidecar.loader",
    "llm_sidecar.hermes_plugin",
    "llm_sidecar.reward",
]

def test_all_modules_importable():
    for mod in MODULES:
        importlib.import_module(mod)
