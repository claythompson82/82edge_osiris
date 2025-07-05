import asyncio
import json

import pytest

from llm_sidecar.event_bus import EventBus, Channel


@pytest.mark.asyncio
async def test_publish_subscribe_round_trip():
    bus = EventBus()
    received: list[str] = []

    async def listener():
        async for payload in bus.subscribe(Channel.SPEECH_TEXT.value):
            received.append(payload)
            break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0)  # ensure listener starts
    await bus.publish_speech("hello", foo="bar")
    await asyncio.wait_for(task, timeout=1)

    assert len(received) == 1
    data = json.loads(received[0])
    assert data["text"] == "hello"
    assert data["metadata"] == {"foo": "bar"}
    assert isinstance(data["ts"], float)
