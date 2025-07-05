from __future__ import annotations

from unittest import mock

import pytest

from dgm_kernel import llm_client
from dgm_kernel.config import ExternalLLMConfig
from dgm_kernel.trace_schema import Trace


class DummyResponse:
    def __init__(self, status: int, data: dict[str, object]):
        self.status_code = status
        self._data = data
        self.text = ""

    def json(self) -> dict[str, object]:
        return self._data


def test_draft_patch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"target": "t.py"}
    resp = DummyResponse(200, expected)
    monkeypatch.setattr(llm_client, "CONFIG", ExternalLLMConfig("http://x", "k", 1.0))
    post_mock = mock.Mock(return_value=resp)
    monkeypatch.setattr(llm_client.requests, "post", post_mock)

    trace = Trace(id="1", timestamp=1, pnl=0.0)
    result = llm_client.draft_patch([trace.model_dump()])
    assert result == expected
    post_mock.assert_called_once()


def test_draft_patch_bad_status(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = DummyResponse(500, {"error": "oops"})
    monkeypatch.setattr(llm_client, "CONFIG", ExternalLLMConfig("http://x", "", 1.0))
    post_mock = mock.Mock(return_value=resp)
    monkeypatch.setattr(llm_client.requests, "post", post_mock)

    trace = Trace(id="1", timestamp=1, pnl=0.0)
    result = llm_client.draft_patch([trace.model_dump()])
    assert result is None
    post_mock.assert_called_once()
