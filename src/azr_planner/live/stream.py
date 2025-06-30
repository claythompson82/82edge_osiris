from __future__ import annotations

import asyncio
import datetime
from abc import ABC, abstractmethod
from typing import AsyncIterable, List, Optional

# Using the Bar definition from replay.schemas for consistency
from azr_planner.replay.schemas import Bar

class AbstractBarStream(ABC):
    """Abstract base class for a real-time bar stream."""

    @abstractmethod
    async def stream(self) -> AsyncIterable[Bar]:
        """Yields Bar objects as they arrive from the source."""
        # This is an abstract method, so it needs to be implemented by subclasses.
        # The 'yield' here is just to make Python recognize it as an async generator method signature.
        # It will not be executed directly.
        if False: # pragma: no cover
            yield


class MockWebSocketBarStream(AbstractBarStream):
    """
    A mock implementation of a bar stream, simulating a WebSocket feed.
    For testing, it yields a predefined sequence of bars or can be extended
    to use Hypothesis for more complex bar generation.
    """
    def __init__(self, symbol: str, predefined_bars: Optional[List[Bar]] = None, delay_seconds: float = 0.1):
        self.symbol = symbol
        self.delay_seconds = delay_seconds # Simulate network latency / bar interval

        if predefined_bars is not None:
            self._bars_to_yield = predefined_bars
        else:
            # Default sequence if none provided
            self._bars_to_yield = self._generate_default_bars()

    def _generate_default_bars(self, count: int = 100) -> List[Bar]:
        """Generates a simple sequence of bars for the given symbol."""
        bars: List[Bar] = []
        start_time = datetime.datetime.now(datetime.timezone.utc)
        price = 100.0
        for i in range(count):
            ts = start_time + datetime.timedelta(seconds=i * 15) # 15-second bars
            open_p, high_p, low_p, close_p = price, price + 0.5, price - 0.5, price + (i % 3 - 1) * 0.2
            bars.append(Bar(
                timestamp=ts,
                instrument=self.symbol, # Ensure Bar uses the configured symbol
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=100.0 + i
            ))
            price += (i % 5 - 2) * 0.1 # Simple price movement
        return bars

    async def stream(self) -> AsyncIterable[Bar]:
        """Yields Bar objects from the predefined sequence with a delay."""
        for bar in self._bars_to_yield:
            # Ensure the bar's instrument matches the stream's configured symbol,
            # especially if _bars_to_yield could come from various sources.
            # For this mock, _generate_default_bars already sets it.
            # If Bar was a dataclass, one might use `dataclasses.replace(bar, instrument=self.symbol)`
            # but it's a Pydantic model (or simple class). We assume bars are correctly symbolized.
            if bar.instrument != self.symbol:
                # This case should ideally not happen if bars are generated correctly for this stream.
                # Or, one might choose to skip/log if strict symbol matching is required.
                # For now, we assume correct generation.
                pass # pragma: no cover

            yield bar
            await asyncio.sleep(self.delay_seconds)
