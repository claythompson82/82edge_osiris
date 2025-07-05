from __future__ import annotations
from fakeredis import FakeRedis as _Base, FakeServer


class FakeRedis(_Base):  # type: ignore[misc]
    @classmethod
    def from_server(cls, server: FakeServer, *a, **kw) -> "FakeRedis":
        return cls(server, *a, **kw)

