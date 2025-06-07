import logging
from typing import Any, Dict

import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Simple HTTP client for interacting with the LLM sidecar."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    def _request_with_retry(
        self, method: str, endpoint: str, **kwargs
    ) -> requests.Response:
        url = f"{self.base_url}{endpoint}"

        def _retryable(resp: requests.Response | None) -> bool:
            return resp is None or resp.status_code >= 500

        for attempt in Retrying(
            reraise=True,
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=self.backoff_factor),
            retry=retry_if_exception_type(requests.exceptions.RequestException)
            | retry_if_result(_retryable),
        ):
            with attempt:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Server error: {resp.status_code}", response=resp
                    )
                return resp

    def generate(
        self, model_id: str, prompt: str, max_length: int = 256
    ) -> Dict[str, Any]:
        resp = self._request_with_retry(
            "POST",
            f"/generate?model_id={model_id}",
            json={"prompt": prompt, "max_length": max_length},
        )
        return resp.json()

    def propose_trade_adjustments(
        self, prompt: str, max_length: int = 256
    ) -> Dict[str, Any]:
        resp = self._request_with_retry(
            "POST",
            "/propose_trade_adjustments/",
            json={"prompt": prompt, "max_length": max_length},
        )
        return resp.json()
