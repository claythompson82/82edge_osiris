"""meta_loop end-to-end + unit tests (schema-aware)."""

from __future__ import annotations

import asyncio
import difflib
import importlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# -----------------------------------------------------------------------------
# In-memory Redis shim – placed *before* importing module-under-test
# -----------------------------------------------------------------------------
class _MiniRedis:
    """Tiny subset of redis-py used by meta_loop tests."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = defaultdict(list)

    # list-ops ---------------------------------------------------------------
    def lpush(self, name: str, value: str) -> None:
        self._store[name].insert(0, value)

    def rpop(self, name: str) -> str | None:
        lst = self._store[name]
        return lst.pop() if lst else None

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        if end == -1:
            end = len(self._store[name]) - 1
        return self._store[name][start : end + 1]

    # helpers ----------------------------------------------------------------
    def llen(self, name: str) -> int:
        return len(self._store[name])

    def ping(self) -> bool:  # noqa: D401 – parity with redis-py
        return True


class _RedisModule:
    class Redis:  # noqa: D101 – stub wrapper
        def __init__(self, *_, **__) -> None:
            self._client = _MiniRedis()

        # delegate everything to _client
        def __getattr__(self, attr):
            return getattr(self._client, attr)

    class exceptions:  # noqa: D101 – stub namespace
        class RedisError(Exception): ...


import sys  # noqa: E402 – needed before meta_loop import

sys.modules.setdefault("redis", _RedisModule())

# -----------------------------------------------------------------------------
# Module under test
# -----------------------------------------------------------------------------
from dgm_kernel import meta_loop  # noqa: E402

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_trace(i: int) -> Dict[str, Any]:
    """Return a **schema-valid** trace."""
    return {
        "id": f"t{i}",
        "timestamp": float(i),
        "pnl": float(i),
        "data": f"trace_{i}",
    }


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def fake_redis() -> _MiniRedis:
    """
    Return the in-memory Redis stub used by *meta_loop* and **reset its store
    for each test**.
    """
    wrapper = meta_loop.REDIS          # redis.Redis wrapper
    assert hasattr(wrapper, "_client"), "redis shim not installed"
    core: _MiniRedis = wrapper._client
    core._store.clear()
    return core


@pytest.fixture(autouse=True)
def _patch_redis(monkeypatch, fake_redis) -> None:
    """Inject our stub into meta_loop for every test."""
    monkeypatch.setattr(meta_loop, "REDIS", fake_redis)


# -----------------------------------------------------------------------------
# fetch_recent_traces
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_recent_empty() -> None:
    assert await meta_loop.fetch_recent_traces(5) == []


@pytest.mark.asyncio
async def test_fetch_recent_n_items(fake_redis: _MiniRedis) -> None:
    for i in range(10):
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_make_trace(i)))

    got = await meta_loop.fetch_recent_traces(7)
    assert len(got) == 7
    assert got[0]["id"] == "t0"
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 3


@pytest.mark.asyncio
async def test_fetch_recent_less_than_n(fake_redis: _MiniRedis) -> None:
    for i in range(3):
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_make_trace(i)))

    got = await meta_loop.fetch_recent_traces(5)
    assert [t["id"] for t in got] == ["t0", "t1", "t2"]
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 0


@pytest.mark.asyncio
async def test_fetch_recent_json_error(fake_redis: _MiniRedis, caplog) -> None:
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_make_trace(1)))
    fake_redis.lpush(meta_loop.TRACE_QUEUE, "bad json")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_make_trace(2)))

    with caplog.at_level(logging.ERROR):
        got = await meta_loop.fetch_recent_traces(3)

    assert [t["id"] for t in got] == ["t1", "t2"]
    assert any("bad json" in m for m in caplog.messages)


# -----------------------------------------------------------------------------
# _verify_patch – quick smoke test for dangerous-token gate
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verify_patch_rejects_token() -> None:
    bad = {"after": 'import os\nos.system("rm -rf /")', "target": "dummy.py"}
    ok = await meta_loop._verify_patch([], bad)
    assert ok is False


# -----------------------------------------------------------------------------
# loop_forever – regression: should still call _generate_patch after rollback
# -----------------------------------------------------------------------------
def test_loop_forever_rolls_back_and_mutates(monkeypatch) -> None:
    calls = {"rollback": 0, "generate": 0}

    async def always_fail_verify(_, __):  # returns False every time
        return False

    async def fake_generate(_):
        calls["generate"] += 1
        return {"target": "t.py", "before": "", "after": ""}

    def fake_rollback(_):
        calls["rollback"] += 1

    async def fake_fetch(*_, **__):
        return [_make_trace(0)]

    # stop after two sleep() calls
    sleep_cnt = {"n": 0}

    def stop_after_two(_):
        sleep_cnt["n"] += 1
        if sleep_cnt["n"] == 2:
            raise StopIteration

    monkeypatch.setattr(meta_loop, "_verify_patch", always_fail_verify)
    monkeypatch.setattr(meta_loop, "_generate_patch_async", fake_generate)
    monkeypatch.setattr(meta_loop, "_rollback", fake_rollback)
    monkeypatch.setattr(meta_loop, "fetch_recent_traces", fake_fetch)
    monkeypatch.setattr(time, "sleep", stop_after_two)

    with pytest.raises(StopIteration):
        meta_loop.loop_forever()

    assert calls == {"rollback": 1, "generate": 1}
