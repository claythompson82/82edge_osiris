#!/usr/bin/env python
import json
from pathlib import Path
from unittest.mock import patch
import sys
from types import ModuleType
from fastapi.openapi.utils import get_openapi

mock_loader = ModuleType('llm_sidecar.loader')
mock_loader.load_hermes_model = lambda: None
mock_loader.load_phi3_model = lambda: None
mock_loader.get_hermes_model_and_tokenizer = lambda: (None, None)
mock_loader.get_phi3_model_and_tokenizer = lambda: (None, None)
mock_loader.MICRO_LLM_MODEL_PATH = ''
mock_loader.phi3_adapter_date = None
mock_tts = ModuleType('llm_sidecar.tts')
mock_tts.ChatterboxTTS = lambda *a, **k: None
mock_torch = ModuleType('torch')
mock_torch.cuda = ModuleType('torch.cuda')
mock_torch.cuda.is_available = lambda: False
mock_outlines = ModuleType('outlines')
mock_outlines.generate = lambda *a, **k: None
mock_lancedb = ModuleType('lancedb')
mock_lancedb.pydantic = ModuleType('lancedb.pydantic')
mock_lancedb.pydantic.LanceModel = object
mock_lancedb.connect = lambda *a, **k: type('DB', (), {'open_table': lambda *a, **k: None})()
mock_redis = ModuleType('redis')
mock_redis.asyncio = ModuleType('redis.asyncio')
mock_redis.asyncio.client = ModuleType('redis.asyncio.client')
mock_redis.asyncio.client.PubSub = object
mock_redis.exceptions = ModuleType('redis.exceptions')
mock_redis.exceptions.RedisError = Exception
mock_redis.asyncio.from_url = lambda *a, **k: None

with patch.dict(sys.modules, {
    'llm_sidecar.loader': mock_loader,
    'llm_sidecar.tts': mock_tts,
    'torch': mock_torch,
    'lancedb': mock_lancedb,
    'lancedb.pydantic': mock_lancedb.pydantic,
    'outlines': mock_outlines,
    'redis.asyncio': mock_redis.asyncio,
    'redis.asyncio.client': mock_redis.asyncio.client,
    'redis.exceptions': mock_redis.exceptions,
    'redis': mock_redis,
}):
    import importlib
    dummy_llm = ModuleType('osiris.llm_sidecar')
    sys.modules['osiris.llm_sidecar'] = dummy_llm
    server = importlib.import_module('osiris.server')

schema = get_openapi(
    title=server.app.title,
    version=server.app.version,
    description=server.app.description,
    routes=server.app.routes,
)

Path("docs").mkdir(exist_ok=True)
with open("docs/openapi.json", "w") as f:
    json.dump(schema, f, indent=2)
print("OpenAPI schema written to docs/openapi.json")
