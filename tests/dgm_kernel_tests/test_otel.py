from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from opentelemetry import trace
import sys
import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _setup_tracing() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


import pytest


@pytest.mark.asyncio
async def test_loop_once_spans(monkeypatch):
    exporter = _setup_tracing()
    # import after tracer provider configured
    dummy_crypto = types.ModuleType("dgm_kernel.crypto_sign")
    dummy_crypto.sign_patch = lambda diff: "sig"
    dummy_crypto.verify_patch = lambda diff, sig: True
    sys.modules["dgm_kernel.crypto_sign"] = dummy_crypto

    from dgm_kernel import meta_loop

    monkeypatch.setattr(meta_loop, "PATCH_RATE_LIMIT_SECONDS", 0, raising=False)
    monkeypatch.setattr(meta_loop, "_last_patch_time", 0.0, raising=False)
    monkeypatch.setattr(meta_loop, "fetch_recent_traces", AsyncMock(return_value=[{"id": "t1"}]))
    monkeypatch.setattr(meta_loop, "generate_patch", AsyncMock(return_value={"target": "t.py", "before": "", "after": ""}))
    monkeypatch.setattr(meta_loop, "_verify_patch", AsyncMock(return_value=True))
    monkeypatch.setattr(meta_loop, "prove_patch", lambda diff: 1.0)
    monkeypatch.setattr(meta_loop, "run_patch_in_sandbox", lambda *_, **__: (True, "", 0, 0.0, 0.0))
    monkeypatch.setattr(meta_loop, "_apply_patch", lambda patch: True)
    monkeypatch.setattr(meta_loop, "_record_patch_history", lambda entry: None)

    await meta_loop.loop_once()

    names = {s.name for s in exporter.get_finished_spans()}
    assert names == {"meta_loop.iteration", "fetch_traces", "generate_patch", "verify_patch", "sandbox"}



