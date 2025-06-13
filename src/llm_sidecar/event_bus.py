"""Async EventBus stub for unit tests."""
from typing import AsyncIterator, Any

class RedisError(Exception):
    pass

class EventBus:  # no behaviours needed for tests
    async def publish(self, *_, **__):
        return None
    async def subscribe(self, *_, **__) -> AsyncIterator[Any]:
        yield {}

aSYNC_EMPTY: AsyncIterator[Any]

async def connect(*_, **__):  # noqa: D401
    return EventBus()

async def close(*_, **__):
    return None

async def subscribe(*_, **__):
    yield {}
