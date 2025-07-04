"""meta-loop end-to-end + unit tests (schema-aware)."""

from __future__ import annotations

import asyncio
import difflib
import importlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# ────────────────────────────  minimal Redis shim  ───────────────────────────
class _MiniRedis:
    """In-memory subset of redis-py for tests."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = defaultdict(list)

    # list operations --------------------------------------------------------
    def lpush(self, key: str, val: str) -> None:
        self._data[key].insert(0, val)

    def rpop(self, key: str) -> str | None:
        lst = self._data[key]
        return lst.pop() if lst else None

    def llen(self, key: str) -> int:
        return len(self._data[key])

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        if end == -1:
            end = len(self._data[key]) - 1
        return self._data[key][start : end + 1]

    # misc -------------------------------------------------------------------
    def ping(self) -> bool:  # parity with redis-py
        return True


class _RedisModule:
    class Redis:  # wrapper that delegates to a fresh _MiniRedis each instantiation
        def __init__(self, *_, **__) -> None:
            self._core = _MiniRedis()

        def __getattr__(self, item):
            return getattr(self._core, item)

    class exceptions:
        class RedisError(Exception): ...


# install shim *before* importing code under test
sys.modules.setdefault("redis", _RedisModule())

# ────────────────────────────  module under test  ────────────────────────────
from dgm_kernel import meta_loop  # noqa: E402

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _trace(i: int) -> Dict[str, Any]:
    return {
        "id": f"t{i}",
        "timestamp": float(i),
        "pnl": float(i),
        "data": f"trace_{i}",
    }


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------
@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _MiniRedis:
    """Inject a fresh stub into *meta_loop* each test."""
    stub = _MiniRedis()
    monkeypatch.setattr(meta_loop, "REDIS", stub, raising=False)
    return stub


# -----------------------------------------------------------------------------
# fetch_recent_traces
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_recent_empty() -> None:
    assert await meta_loop.fetch_recent_traces(5) == []


@pytest.mark.asyncio
async def test_fetch_recent_n_items(fake_redis: _MiniRedis) -> None:
    for i in range(10):
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_trace(i)))

    got = await meta_loop.fetch_recent_traces(7)
    assert len(got) == 7
    assert got[0]["id"] == "t0"
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 3


@pytest.mark.asyncio
async def test_fetch_recent_less_than_n(fake_redis: _MiniRedis) -> None:
    for i in range(3):
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_trace(i)))

    got = await meta_loop.fetch_recent_traces(5)
    assert [t["id"] for t in got] == ["t0", "t1", "t2"]
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 0


@pytest.mark.asyncio
async def test_fetch_recent_json_error(fake_redis: _MiniRedis, caplog) -> None:
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_trace(1)))
    fake_redis.lpush(meta_loop.TRACE_QUEUE, "bad json")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(_trace(2)))

    with caplog.at_level(logging.ERROR):
        got = await meta_loop.fetch_recent_traces(3)

    assert [t["id"] for t in got] == ["t1", "t2"]
    assert any("bad json" in m for m in caplog.messages)


# -----------------------------------------------------------------------------
# verify patch – dangerous-token gate
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verify_patch_rejects_token() -> None:
    bad = {"after": 'import os\nos.system("rm -rf /")', "target": "dummy.py"}
    assert await meta_loop._verify_patch([], bad) is False


# -----------------------------------------------------------------------------
# loop_forever regression – should generate after rollback
# -----------------------------------------------------------------------------
def test_loop_forever_mutates_after_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"rollback": 0, "generate": 0}

    async def always_fail(*_) -> bool:
        return False

    async def fake_gen(*_) -> Dict[str, str]:
        calls["generate"] += 1
        return {"target": "x.py", "before": "", "after": ""}

    def fake_rb(*_) -> None:
        calls["rollback"] += 1

    async def fake_fetch(*_, **__) -> list[Dict[str, Any]]:
        return [_trace(0)]

    sleep_cnt = {"n": 0}

    def stop_after_two(_: float) -> None:
        sleep_cnt["n"] += 1
        if sleep_cnt["n"] == 2:
            raise StopIteration

    monkeypatch.setattr(meta_loop, "_verify_patch", always_fail)
    monkeypatch.setattr(meta_loop, "_generate_patch_async", fake_gen)
    monkeypatch.setattr(meta_loop, "_rollback", fake_rb)
    monkeypatch.setattr(meta_loop, "fetch_recent_traces", fake_fetch)
    monkeypatch.setattr(time, "sleep", stop_after_two)

    with pytest.raises(StopIteration):
        meta_loop.loop_forever()

    assert calls == {"rollback": 1, "generate": 1}
