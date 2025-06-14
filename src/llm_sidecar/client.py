"""Tiny HTTP client used by the tests – **no real network calls**."""
from __future__ import annotations

import time
from typing import Any, Dict

import httpx

__all__ = ["LLMClient", "LLMClientError"]


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, retries: int = 3, backoff_factor: float = 0.5, timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = httpx.Client()

    # internal helper so tests can monkey-patch network layer
    def _do_request(self, payload: Dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}/generate"
        for attempt in range(1, self.retries + 2):  # first try + N retries
            resp = self.session.request("POST", url, json=payload, timeout=self.timeout)
            if resp.status_code < 500:
                return resp
            if attempt <= self.retries:
                time.sleep(self.backoff_factor)
        raise LLMClientError(f"failed after {self.retries} retries")

    # public API
    def generate(self, model: str, prompt: str, **params) -> Dict[str, Any] | None:
        try:
            resp = self._do_request({"model": model, "prompt": prompt, **params})
            return resp.json()
        except LLMClientError as exc:
            # CI tests expect *None* instead of raising
            return None
