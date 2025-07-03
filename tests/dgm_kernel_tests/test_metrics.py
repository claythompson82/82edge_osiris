import importlib
import asyncio
from pathlib import Path
from typing import Dict
import importlib

import pytest
from prometheus_client import CollectorRegistry, Counter

from dgm_kernel import meta_loop, metrics


def get_metric_value(registry: CollectorRegistry, name: str, labels: Dict[str, str]) -> float:
    value = registry.get_sample_value(name, labels=labels)
    return value if value is not None else 0.0


@pytest.mark.asyncio
async def test_generate_patch_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(meta_loop.metrics, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)

    monkeypatch.setattr(meta_loop, "draft_patch", lambda traces: {"target": "t.py", "before": "", "after": "code"})
    monkeypatch.setattr(meta_loop, "_generate_patch", lambda code: code)

    before_val = get_metric_value(registry, "dgm_patch_generation_total", {"mutation": meta_loop._MUTATION_NAME, "result": "success"})
    patch = await meta_loop.generate_patch([{}])
    assert patch is not None
    assert get_metric_value(registry, "dgm_patch_generation_total", {"mutation": meta_loop._MUTATION_NAME, "result": "success"}) == before_val + 1.0


@pytest.mark.asyncio
async def test_generate_patch_metrics_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(meta_loop.metrics, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)

    monkeypatch.setattr(meta_loop, "draft_patch", lambda traces: None)

    before_val = get_metric_value(registry, "dgm_patch_generation_total", {"mutation": meta_loop._MUTATION_NAME, "result": "failure"})
    patch = await meta_loop.generate_patch([{}])
    assert patch is None
    assert get_metric_value(registry, "dgm_patch_generation_total", {"mutation": meta_loop._MUTATION_NAME, "result": "failure"}) == before_val + 1.0


def test_apply_patch_metrics_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(meta_loop.metrics, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)

    target = tmp_path / "mod.py"
    patch = {"target": str(target), "after": "", "before": ""}

    before_val = get_metric_value(registry, "dgm_patches_applied_total", {"mutation": meta_loop._MUTATION_NAME, "result": "success"})
    assert meta_loop._apply_patch(patch) is True
    assert get_metric_value(registry, "dgm_patches_applied_total", {"mutation": meta_loop._MUTATION_NAME, "result": "success"}) == before_val + 1.0


def test_apply_patch_metrics_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(meta_loop.metrics, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)

    target = tmp_path / "mod.py"
    patch = {"target": str(target), "after": "", "before": ""}

    def fail_import(name: str):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    before_val = get_metric_value(registry, "dgm_patches_applied_total", {"mutation": meta_loop._MUTATION_NAME, "result": "failure"})
    assert meta_loop._apply_patch(patch) is False
    assert get_metric_value(registry, "dgm_patches_applied_total", {"mutation": meta_loop._MUTATION_NAME, "result": "failure"}) == before_val + 1.0


@pytest.mark.asyncio
async def test_unsafe_token_counter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    counter = Counter(
        "dgm_unsafe_token_found_total",
        "Number of patches rejected due to dangerous tokens",
        registry=registry,
    )
    monkeypatch.setattr(metrics, "unsafe_token_found_total", counter)
    monkeypatch.setattr(meta_loop.metrics, "unsafe_token_found_total", counter)

    patch = {
        "target": str(tmp_path / "mod.py"),
        "before": "",
        "after": 'os.system("rm -rf /")',
    }

    result = await meta_loop._verify_patch([], patch)
    assert result is False
    assert counter._value.get() == 1.0
