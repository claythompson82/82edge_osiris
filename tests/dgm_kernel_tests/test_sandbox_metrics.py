import types
from prometheus_client import CollectorRegistry, Counter
import subprocess
import resource

from dgm_kernel import sandbox, metrics


class DummyProc:
    def __init__(self, rc: int = 0) -> None:
        self.stdout = ""
        self.stderr = ""
        self.returncode = rc


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

