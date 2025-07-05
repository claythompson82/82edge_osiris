import types
from prometheus_client import CollectorRegistry, Counter, Histogram
import subprocess
import resource
import statistics
from collections import defaultdict
from unittest.mock import AsyncMock
import pytest

from dgm_kernel import meta_loop

from dgm_kernel import sandbox, metrics


class DummyProc:
    def __init__(self, rc: int = 0) -> None:
        self.stdout = ""
        self.stderr = ""
        self.returncode = rc


class _MiniRedis:
    """Simple in-memory Redis subset."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = defaultdict(list)

    def lpush(self, key: str, val: str) -> None:
        self._data[key].insert(0, val)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        lst = self._data[key]
        if end == -1:
            end = len(lst) - 1
        return lst[start : end + 1]

    def ltrim(self, key: str, start: int, end: int) -> None:
        lst = self._data[key]
        if end == -1:
            end = len(lst) - 1
        self._data[key] = lst[start : end + 1]

    def llen(self, key: str) -> int:
        return len(self._data[key])

    def ping(self) -> bool:
        return True


def test_run_patch_records_metrics(monkeypatch):
    registry = CollectorRegistry(auto_describe=True)
    cpu_counter = Counter(
        "dgm_sandbox_cpu_ms_total",
        "Total CPU ms",
        registry=registry,
    )
    ram_counter = Counter(
        "dgm_sandbox_ram_mb_total",
        "Total RAM mb",
        registry=registry,
    )
    monkeypatch.setattr(metrics, "sandbox_cpu_ms_total", cpu_counter)
    monkeypatch.setattr(metrics, "sandbox_ram_mb_total", ram_counter)

    # Fake subprocess.run
    monkeypatch.setattr(sandbox, "subprocess", types.SimpleNamespace(run=lambda *a, **k: DummyProc()))

    ru_vals = [
        types.SimpleNamespace(ru_utime=1.0, ru_stime=1.0, ru_maxrss=1000),
        types.SimpleNamespace(ru_utime=2.5, ru_stime=2.5, ru_maxrss=3000),
    ]
    def fake_getrusage(who):
        return ru_vals.pop(0)
    monkeypatch.setattr(sandbox.resource, "getrusage", fake_getrusage)

    ok, logs, code, cpu_ms, ram_mb = sandbox.run_patch_in_sandbox({"after": "print('hi')"})

    assert ok is True
    assert cpu_ms == ( (2.5+2.5) - (1.0+1.0) ) * 1000.0
    assert ram_mb == (3000 - 1000) / 1024.0
    assert cpu_counter._value.get() == cpu_ms
    assert ram_counter._value.get() == ram_mb


def test_timeout_helpers(monkeypatch):
    redis_stub = _MiniRedis()
    monkeypatch.setattr(sandbox, "REDIS", redis_stub)

    registry = CollectorRegistry(auto_describe=True)
    hist = Histogram(
        "dgm_sandbox_runtime_seconds",
        "Runtime",
        buckets=(0.1, 0.3, 1, 3, 10, 30),
        registry=registry,
    )
    monkeypatch.setattr(metrics, "sandbox_runtime_seconds", hist)
    monkeypatch.setattr(sandbox.metrics, "sandbox_runtime_seconds", hist)

    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    for s in samples:
        sandbox.record_runtime(s)

    expected = max(10.0, 4 * statistics.median(samples))
    assert sandbox.suggest_timeout() == pytest.approx(expected)
    assert redis_stub.llen(sandbox.RUNTIMES_KEY) == len(samples)


@pytest.mark.asyncio
async def test_meta_loop_records_runtime(monkeypatch):
    redis_stub = _MiniRedis()
    monkeypatch.setattr(sandbox, "REDIS", redis_stub)

    registry = CollectorRegistry(auto_describe=True)
    hist = Histogram(
        "dgm_sandbox_runtime_seconds",
        "Runtime",
        buckets=(0.1, 0.3, 1, 3, 10, 30),
        registry=registry,
    )
    monkeypatch.setattr(metrics, "sandbox_runtime_seconds", hist)
    monkeypatch.setattr(sandbox.metrics, "sandbox_runtime_seconds", hist)
    monkeypatch.setattr(meta_loop.metrics, "sandbox_runtime_seconds", hist)

    monkeypatch.setattr(meta_loop, "PATCH_RATE_LIMIT_SECONDS", 0, raising=False)
    monkeypatch.setattr(meta_loop, "_last_patch_time", 0.0, raising=False)
    monkeypatch.setattr(meta_loop, "_apply_patch", lambda patch: True)
    monkeypatch.setattr(meta_loop, "_record_patch_history", lambda entry: None)
    monkeypatch.setattr(meta_loop, "prove_patch", lambda diff: 1.0)

    monkeypatch.setattr(meta_loop, "fetch_recent_traces", AsyncMock(return_value=[{}]))
    monkeypatch.setattr(
        meta_loop,
        "generate_patch",
        AsyncMock(return_value={"target": "t.py", "before": "", "after": ""}),
    )
    monkeypatch.setattr(meta_loop, "_verify_patch", AsyncMock(return_value=True))

    monkeypatch.setattr(meta_loop, "suggest_timeout", lambda default=10.0: 2.0)
    tvals = [1.0, 3.0]

    def fake_perf() -> float:
        return tvals.pop(0)

    monkeypatch.setattr(meta_loop.time, "perf_counter", fake_perf)

    call = {}

    def fake_run(patch, timeout=None):
        call["timeout"] = timeout
        return True, "", 0, 0.0, 0.0

    monkeypatch.setattr(meta_loop, "run_patch_in_sandbox", fake_run)

    rec = {}

    def fake_record(dur: float) -> None:
        rec["dur"] = dur

    monkeypatch.setattr(meta_loop, "record_runtime", fake_record)

    await meta_loop.loop_once()

    assert call["timeout"] == 2.0
    assert rec["dur"] == pytest.approx(2.0)

