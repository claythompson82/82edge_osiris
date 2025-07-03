from __future__ import annotations

import logging
from typing import List, Dict, Optional

import requests

from .config import CONFIG

log = logging.getLogger(__name__)

def draft_patch(trace: List[Dict]) -> Optional[Dict]:
    """Request a patch from the external LLM service."""
    if not trace:
        log.warning("draft_patch called with no traces, returning None.")
        return None

    url = f"{CONFIG.base_url.rstrip('/')}/v1/draft_patch"
    headers = {"Authorization": f"Bearer {CONFIG.api_key}"} if CONFIG.api_key else {}

    try:
        resp = requests.post(
            url,
            json={"trace": trace},
            headers=headers,
            timeout=CONFIG.timeout,
        )
    except Exception as exc:  # pragma: no cover - network errors
        log.error("draft_patch HTTP request failed: %s", exc)
        return None

    if resp.status_code != 200:
        log.error("draft_patch failed with status %s: %s", resp.status_code, resp.text)
        return None

    try:
        return resp.json()
    except Exception as exc:
        log.error("draft_patch JSON decode failed: %s", exc)
        return None
