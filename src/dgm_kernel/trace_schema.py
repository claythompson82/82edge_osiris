from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, ValidationError

from dgm_kernel import metrics

log = logging.getLogger(__name__)


class Trace(BaseModel):
    """Execution trace record."""

    id: str
    timestamp: int
    pnl: float
    patch_id: str | None = None


def validate_traces(traces: List[dict]) -> List[Trace]:
    """Validate raw trace dicts, dropping invalid rows."""
    valid: List[Trace] = []
    for idx, row in enumerate(traces):
        try:
            valid.append(Trace.model_validate(row))
        except ValidationError as exc:  # pragma: no cover - error path tested
            log.error(
                "Trace validation failed for trace: %s. Error: %s. Trace index: %s",
                row,
                exc,
                idx,
            )
            metrics.trace_validation_fail_total.inc()
    return valid


__all__ = ["Trace", "validate_traces"]
