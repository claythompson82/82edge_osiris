import json
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge

from dgm_kernel import meta_loop, metrics


def test_patch_rate_metric(tmp_path, monkeypatch):
    history_file = tmp_path / "history.json"
    # Prepopulate with 11 entries spaced 60 seconds apart
    entries = []
    for i in range(11):
        entries.append({"timestamp": i * 60})
    history_file.write_text(json.dumps(entries))
    monkeypatch.setattr(meta_loop, "PATCH_HISTORY_FILE", history_file)

    registry = CollectorRegistry(auto_describe=True)
    gauge = Gauge(
        "dgm_patch_apply_minutes_average",
        "Average minutes between patch applications over the last 10 patches",
        registry=registry,
    )
    monkeypatch.setattr(metrics, "patch_apply_minutes_average", gauge)
    monkeypatch.setattr(meta_loop.metrics, "patch_apply_minutes_average", gauge)

    meta_loop._record_patch_history({"timestamp": 11 * 60})

    assert gauge._value.get() == 1.0
