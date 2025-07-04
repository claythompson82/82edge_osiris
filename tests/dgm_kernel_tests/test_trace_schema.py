from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter

from dgm_kernel import metrics
from dgm_kernel.trace_schema import validate_traces


def setup_counter(monkeypatch):
    registry = CollectorRegistry(auto_describe=True)
    counter = Counter(
        "dgm_trace_validation_fail_total",
        "Number of traces dropped due to schema validation errors",
        registry=registry,
    )
    monkeypatch.setattr(metrics, "trace_validation_fail_total", counter)
    return counter


def test_validate_traces_valid(monkeypatch):
    counter = setup_counter(monkeypatch)
    rows = [
        {"id": "a", "timestamp": 1, "pnl": 0.5},
        {"id": "b", "timestamp": 2, "pnl": -1.0, "patch_id": "p1"},
    ]
    traces = validate_traces(rows)
    assert [t.id for t in traces] == ["a", "b"]
    assert counter._value.get() == 0.0


def test_validate_traces_invalid(monkeypatch):
    counter = setup_counter(monkeypatch)
    rows = [
        {"id": "good", "timestamp": 1, "pnl": 1.0},
        {"timestamp": 2, "pnl": 2.0},  # missing id
        {"id": "bad2", "timestamp": 3, "pnl": "bad"},  # pnl not float
    ]
    traces = validate_traces(rows)
    assert len(traces) == 1
    assert traces[0].id == "good"
    assert counter._value.get() == 2.0
