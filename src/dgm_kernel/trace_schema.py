from __future__ import annotations

import logging
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, ValidationError

from dgm_kernel import metrics

log = logging.getLogger(__name__)


class HistoryEntry(TypedDict, total=False):
    """A dictionary representing an entry in the patch history."""

    patch_id: str
    timestamp: float
    diff: str
    reward: float
    sandbox_exit_code: int
    sig: str


class Trace(BaseModel):
    """Execution trace record."""

    id: str
    timestamp: int
    pnl: float
    patch_id: str | None = None

    model_config = ConfigDict(extra="ignore")


def validate_traces(traces: list[dict[str, Any]]) -> list[Trace]:
    """Validate raw trace dicts, dropping invalid rows."""
    valid: list[Trace] = []
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


__all__ = ["Trace", "validate_traces", "HistoryEntry"]
