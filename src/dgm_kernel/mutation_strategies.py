from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import requests

from .config import CONFIG
from .trace_schema import Trace

log = logging.getLogger(__name__)


class PatchDict(TypedDict, total=False):
    """Dictionary describing a code-patch returned by the external LLM."""

    target: str
    before: str
    after: str
    diff: str     # optional – may be filled in by server
    sig: str      # cryptographic signature (if server signs)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert ``row`` (Trace or plain‐dict) into a serialisable dict."""
    if isinstance(row, Trace):   # runtime type
        return row.model_dump()
    if hasattr(row, "model_dump"):
        try:
            return cast(dict[str, Any], row.model_dump())
        except Exception:  # fall through to plain cast
            pass
    return cast(dict[str, Any], row)


def draft_patch(trace: list[Any]) -> PatchDict | None:
    """Ask the external LLM service to draft a patch for *trace*.

    Parameters
    ----------
    trace
        A list of ``Trace`` objects **or** raw dict rows.

    Returns
    -------
    PatchDict | None
        Parsed JSON on success, or *None* if the request/response failed.
    """
    if not trace:
        log.warning("draft_patch called with empty trace -> None")
        return None

    url = f"{CONFIG.base_url.rstrip('/')}/v1/draft_patch"
    headers = {"Authorization": f"Bearer {CONFIG.api_key}"} if CONFIG.api_key else {}

    payload = {"trace": [_row_to_dict(r) for r in trace]}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=CONFIG.timeout)
    except Exception as exc:       # pragma: no cover  (network errors)
        log.error("draft_patch HTTP request failed: %s", exc)
        return None

    if not 200 <= resp.status_code < 300:
        log.error("draft_patch failed (%s): %s", resp.status_code, resp.text)
        return None

    try:
        return cast(PatchDict, resp.json())
    except Exception as exc:
        log.error("draft_patch JSON decode failed: %s", exc)
        return None
