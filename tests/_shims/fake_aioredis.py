"""
Minimal shim so test_event_bus.py can do::

    from fake_aioredis import FakeRedis, FakeServer
"""
from __future__ import annotations
from fakeredis import FakeServer, FakeRedis as _FR

class FakeRedis(_FR):  # type: ignore[misc]
    @classmethod
    def from_server(cls, server: FakeServer, *a, **kw) -> "FakeRedis":  # noqa: D401
        # fakeredis' FakeRedis constructor already takes FakeServer
        return cls(server, *a, **kw)
