from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _setup_tracer() -> Any:
    """Return a global tracer named 'dgm', fallback to no-op if unavailable."""
    try:  # pragma: no cover - optional dependency
        from opentelemetry import trace as ot_trace
    except Exception:
        logger.info("OpenTelemetry not available; spans disabled.")

        class _DummySpan:
            def __enter__(self) -> None:
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        class _DummyTracer:
            def start_as_current_span(self, _name: str) -> _DummySpan:
                return _DummySpan()

        return _DummyTracer()

    return ot_trace.get_tracer("dgm")


tracer = _setup_tracer()
