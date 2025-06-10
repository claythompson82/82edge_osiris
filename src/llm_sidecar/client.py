"""
Minimal stand-in for the real LLMClient so unit-tests have something to interact with.
• Exposes `.session` (httpx.Client) and transparently delegates unknown attributes.
• Implements a basic `.request()` method with optional retry + back-off.
"""

import httpx
import random
import time
from typing import Any


class LLMClient:
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 5,
        retries: int = 0,
        backoff_factor: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.backoff = backoff_factor
        self.session = httpx.Client(timeout=timeout)

    # --------------------------------------------------------------------- #
    # Core request helper with naive exponential back-off
    # --------------------------------------------------------------------- #
    def request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        full_url = f"{self.base_url}/{url.lstrip('/')}"
        attempt = 0
        while True:
            response = self.session.request(method, full_url, **kw)
            if response.status_code < 500 or attempt >= self.retries:
                return response
            attempt += 1
            sleep_time = self.backoff * (2**attempt) + random.random() * 0.1
            time.sleep(sleep_time)

    # --------------------------------------------------------------------- #
    # Make attributes like `.get`, `.post`, etc. work transparently
    # --------------------------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying httpx.Client."""
        return getattr(self.session, name)
