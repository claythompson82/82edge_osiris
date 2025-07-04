from __future__ import annotations

import logging
from typing import Any, Literal, Type

from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)


def _setup_tracer() -> Tracer:
    """Return a global tracer named 'dgm', fallback to no-op if unavailable."""
    try:  # pragma: no cover - optional dependency
        from opentelemetry import trace as ot_trace
    except Exception:
        logger.info("OpenTelemetry not available; spans disabled.")

        class _DummySpan:
            def set_attribute(self, key: str, value: AttributeValue) -> None:
                pass

            def __enter__(self) -> _DummySpan:
                return self

            def __exit__(
                self,
                exc_type: Type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: Any | None,
            ) -> Literal[False]:
                return False

        class _DummyTracer:
            def start_as_current_span(self, _name: str) -> _DummySpan:
                return _DummySpan()

        return _DummyTracer()  # type: ignore[return-value]

    return ot_trace.get_tracer("dgm")


tracer: Tracer = _setup_tracer()
