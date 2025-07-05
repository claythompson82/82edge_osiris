"""Test-time shim for missing `FakeRedis.from_server`."""
from __future__ import annotations

from typing import Any

import fakeredis

# Patch fakeredis.aioredis if needed ----------------------------
if not hasattr(fakeredis.aioredis.FakeRedis, "from_server"):
    class _FakeRedis(fakeredis.aioredis.FakeRedis):  # type: ignore[misc]
        @classmethod
        def from_server(cls, server: Any, **kw: Any) -> "_FakeRedis":
            return cls(decode_responses=kw.get("decode_responses", False))

    fakeredis.aioredis.FakeRedis = _FakeRedis  # type: ignore[assignment]

# Re-export patched objects ------------------------------------
FakeRedis = fakeredis.aioredis.FakeRedis  # type: ignore[misc]
FakeServer = fakeredis.aioredis.FakeServer

__all__ = ["FakeRedis", "FakeServer"]
