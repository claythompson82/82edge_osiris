from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import requests

from .config import CONFIG
from .trace_schema import Trace

log = logging.getLogger(__name__)


class PatchDict(TypedDict):
    """Wire-format for a generated code patch."""
    target: str
    before: str
    after: str
    diff: str
    sig: str


def _row_to_dict(row: Any) -> dict[str, Any]:
    """
    Helper that converts a single trace row to a plain `dict`
    without raising.  Handles:

    • `Trace` Pydantic objects (preferred)
    • Arbitrary objects that expose `.model_dump()`
    • A row that is *already* a mapping
    """
    if isinstance(row, Trace):
        return row.model_dump()

    if hasattr(row, "model_dump"):  # type: ignore[attr-defined]
        try:
            return cast(Any, row).model_dump()
        except Exception:  # pragma: no cover – defensive
            pass

    # Fall-back: assume it’s already a plain mapping
    if isinstance(row, dict):
        return row

    log.warning("Un-recognised trace row type %s – sending as‐is", type(row))
    return cast(dict[str, Any], row)


def draft_patch(traces: list[Any]) -> PatchDict | None:  # noqa: D401
    """Ask the external LLM sidecar to propose a patch for the given traces."""
    if not traces:
        log.warning("draft_patch called with no traces – returning None.")
        return None

    url = f"{CONFIG.base_url.rstrip('/')}/v1/draft_patch"
    headers = {"Authorization": f"Bearer {CONFIG.api_key}"} if CONFIG.api_key else {}

    try:
        payload = {"trace": [_row_to_dict(r) for r in traces]}
        resp = requests.post(url, json=payload, headers=headers, timeout=CONFIG.timeout)
    except Exception as exc:  # pragma: no cover – network / transport failures
        log.error("draft_patch HTTP request failed: %s", exc)
        return None

    if not 200 <= resp.status_code < 300:
        log.error("draft_patch failed with status %s: %s", resp.status_code, resp.text)
        return None

    try:
        return cast(PatchDict, resp.json())
    except Exception as exc:  # pragma: no cover – bad JSON
        log.error("draft_patch JSON decode failed: %s", exc)
        return None
