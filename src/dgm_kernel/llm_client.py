from __future__ import annotations

import logging
from typing import TypedDict, cast

import requests

from .config import CONFIG
from .trace_schema import Trace

log = logging.getLogger(__name__)


class PatchDict(TypedDict):
    """A dictionary representing a code patch."""

    target: str
    before: str
    after: str
    diff: str
    sig: str


def draft_patch(trace: list[Trace]) -> PatchDict | None:
    """Request a patch from the external LLM service."""
    if not trace:
        log.warning("draft_patch called with no traces, returning None.")
        return None

    url = f"{CONFIG.base_url.rstrip('/')}/v1/draft_patch"
    headers = {"Authorization": f"Bearer {CONFIG.api_key}"} if CONFIG.api_key else {}

    try:
        trace_json = [t.model_dump_json() for t in trace]
        resp = requests.post(
            url,
            json={"trace": trace_json},
            headers=headers,
            timeout=CONFIG.timeout,
        )
    except Exception as exc:  # pragma: no cover - network errors
        log.error("draft_patch HTTP request failed: %s", exc)
        return None

    if not 200 <= resp.status_code < 300:
        log.error(
            "draft_patch failed with status %s: %s", resp.status_code, resp.text
        )
        return None

    try:
        return cast(PatchDict, resp.json())
    except Exception as exc:
        log.error("draft_patch JSON decode failed: %s", exc)
        return None
