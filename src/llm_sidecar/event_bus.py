"""Minimal async EventBus used in the test-suite."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from enum import Enum
from typing import Any, AsyncIterator, Dict, List

from pydantic import Field
from pydantic.dataclasses import dataclass


class RedisError(Exception):
    """Placeholder Redis error used by the stub."""


class Channel(str, Enum):
    """Event bus topics."""

    SPEECH_TEXT = "speech.text"  # Whisper → Router
    TTS_TEXT = "speech.tts"  # Router → TTS


@dataclass
class SpeechMessage:
    """Payload published on :pydata:`Channel.SPEECH_TEXT`."""

    text: str
    ts: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Very small in-memory publish/subscribe helper."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[asyncio.Queue[str]]] = {}

    async def publish(self, channel: str, payload: str) -> None:
        for q in list(self._subs.get(channel, [])):
            await q.put(payload)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subs.setdefault(channel, []).append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs.get(channel, []).remove(q)

    async def publish_speech(self, text: str, **meta: Any) -> None:
        msg = SpeechMessage(text=text, metadata=dict(meta))
        await self.publish(Channel.SPEECH_TEXT.value, json.dumps(asdict(msg)))


async def connect(*_: Any, **__: Any) -> EventBus:  # noqa: D401
    return EventBus()


async def close(*_: Any, **__: Any) -> None:
    return None


async def subscribe(*_: Any, **__: Any) -> AsyncIterator[Any]:
    yield {}
