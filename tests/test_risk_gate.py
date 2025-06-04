import sys
import types

# Stub lancedb so risk_gate imports without the optional dependency
lancedb_stub = types.ModuleType("lancedb")
lancedb_stub.connect = lambda *args, **kwargs: None
sys.modules.setdefault("lancedb", lancedb_stub)

import pytest
from advisor.risk_gate import accept

DEFAULT_CONFIG = {"max_position_size_pct": 0.05, "max_daily_loss_pct": 0.02}


def test_accept_invalid_proposal():
    result = accept(None, 10000, 0, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "Invalid or empty proposal" in result["reason"]


def test_accept_missing_fields():
    result = accept({"quantity": 10}, 10000, 0, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "missing 'quantity' or 'price_estimate'" in result["reason"]


def test_accept_invalid_numbers():
    result = accept({"quantity": "bad", "price_estimate": 5}, 10000, 0, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "not valid numbers" in result["reason"]


def test_accept_non_positive_nav():
    result = accept({"quantity": 10, "price_estimate": 5}, 0, 0, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "non-positive" in result["reason"]


def test_accept_position_size_exceeds():
    result = accept({"quantity": 6, "price_estimate": 100}, 10000, 0, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "exceeds max" in result["reason"]


def test_accept_daily_pnl_breach():
    result = accept({"quantity": 1, "price_estimate": 100}, 10000, -300, DEFAULT_CONFIG)
    assert not result["accepted"]
    assert "daily loss limit" in result["reason"]


def test_accept_valid_proposal():
    result = accept({"quantity": 1, "price_estimate": 100}, 10000, 100, DEFAULT_CONFIG)
    assert result["accepted"]
    assert result["reason"] == "Proposal meets risk criteria."
