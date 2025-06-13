"""
Adapter hot-swap & health-endpoint tests for Osiris side-car.

These run under heavy mock patching to avoid real model loads.
"""
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient
from osiris.server import app   # FastAPI instance

# ──────────────────────────────────────────────────────────────────────────
# HELPER FIXTURES
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────
# ADAPTER LOAD SCENARIOS
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.xfail  # no adapter dir
def test_adapter_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llm_sidecar.loader.ADAPTER_ROOT",
        tmp_path / "does_not_exist",
        raising=False,
    )
    from llm_sidecar.loader import get_latest_adapter_dir
    assert get_latest_adapter_dir() is None


def test_adapter_success(tmp_path, monkeypatch):
    # create fake dated sub-dirs
    (tmp_path / "2025-06-11").mkdir()
    (tmp_path / "2025-06-12").mkdir()
    monkeypatch.setattr(
        "llm_sidecar.loader.ADAPTER_ROOT",
        tmp_path,
        raising=False,
    )
    from llm_sidecar.loader import get_latest_adapter_dir
    assert get_latest_adapter_dir().name == "2025-06-12"


# ──────────────────────────────────────────────────────────────────────────
# HEALTH ENDPOINT
# ──────────────────────────────────────────────────────────────────────────
import builtins
from unittest import mock

# NOTE: decorator order == argument order (outer → first param)
@mock.patch("httpx.Client")  # 1️⃣
@mock.patch("osiris.server.get_phi3_model_and_tokenizer")  # 2️⃣
@mock.patch("osiris.server.get_hermes_model_and_tokenizer")  # 3️⃣
def test_health_endpoint_with_adapter_date(
    mock_httpx_client,
    mock_phi3_model,
    mock_hermes_model,
    client,
    tmp_path,
    monkeypatch,
):
    # fake model returns so /health skips heavy loads
    mock_phi3_model.return_value = (True, True)
    mock_hermes_model.return_value = (True, True)

    # stub adapter root so server picks up a predictable date
    (tmp_path / "2025-06-10").mkdir()
    monkeypatch.setattr(
        "llm_sidecar.loader.ADAPTER_ROOT", tmp_path, raising=False
    )

    resp = client.get("/health?adapter_date=true")
    assert resp.status_code == 200
    body = resp.json()
    # response should carry latest dir name
    assert body["latest_adapter"] == "2025-06-10"
    # cached stats present
    assert "uptime" in body
