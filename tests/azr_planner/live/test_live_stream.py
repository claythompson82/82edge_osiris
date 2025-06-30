from __future__ import annotations

import pytest
import asyncio
import datetime

from azr_planner.live.stream import MockWebSocketBarStream
from azr_planner.replay.schemas import Bar # Using existing Bar schema

@pytest.mark.asyncio
async def test_mock_web_socket_bar_stream_default_bars() -> None:
    symbol = "TESTSYM"
    streamer = MockWebSocketBarStream(symbol=symbol, delay_seconds=0.001)

    bars_received: list[Bar] = []
    count = 0
    async for bar in streamer.stream():
        bars_received.append(bar)
        count += 1
        if count >= 5: # Limit for test duration
            break

    assert len(bars_received) == 5
    for bar in bars_received:
        assert isinstance(bar, Bar)
        assert bar.instrument == symbol
        assert bar.timestamp is not None
        assert bar.open >= bar.low
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.close >= bar.low

@pytest.mark.asyncio
async def test_mock_web_socket_bar_stream_predefined_bars() -> None:
    symbol = "PREDEF"
    now = datetime.datetime.now(datetime.timezone.utc)
    predefined_bars = [
        Bar(timestamp=now, instrument=symbol, open=10, high=11, low=9, close=10.5, volume=100),
        Bar(timestamp=now + datetime.timedelta(seconds=15), instrument=symbol, open=10.5, high=11.5, low=10, close=11, volume=120),
        Bar(timestamp=now + datetime.timedelta(seconds=30), instrument=symbol, open=11, high=12, low=10.5, close=10.8, volume=110),
    ]
    streamer = MockWebSocketBarStream(symbol=symbol, predefined_bars=predefined_bars, delay_seconds=0.001)

    bars_received = []
    async for bar in streamer.stream():
        bars_received.append(bar)

    assert len(bars_received) == len(predefined_bars)
    for i, received_bar in enumerate(bars_received):
        expected_bar = predefined_bars[i]
        assert received_bar.timestamp == expected_bar.timestamp
        assert received_bar.instrument == expected_bar.instrument
        assert received_bar.close == expected_bar.close

@pytest.mark.asyncio
async def test_mock_web_socket_bar_stream_delay() -> None:
    symbol = "DELAYTEST"
    delay = 0.05 # 50ms
    streamer = MockWebSocketBarStream(symbol=symbol, delay_seconds=delay) # Uses default bars

    start_time = asyncio.get_event_loop().time()
    count = 0
    async for _ in streamer.stream():
        count += 1
        if count >= 3:
            break
    end_time = asyncio.get_event_loop().time()

    assert count == 3
    # Total time should be at least (count - 1) * delay, as the last bar doesn't have a delay *after* it in this loop
    # And first bar is yielded almost immediately. So 2 delays for 3 bars.
    # Allow for some scheduling overhead.
    assert (end_time - start_time) >= (count -1) * delay * 0.9 # A bit of tolerance for timing

@pytest.mark.asyncio
async def test_mock_web_socket_bar_stream_empty_predefined_list() -> None:
    symbol = "EMPTY"
    streamer = MockWebSocketBarStream(symbol=symbol, predefined_bars=[], delay_seconds=0.001)
    bars_received = []
    async for bar in streamer.stream():
        bars_received.append(bar)
    assert not bars_received
