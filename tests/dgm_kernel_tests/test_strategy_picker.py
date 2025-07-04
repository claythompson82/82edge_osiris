import math
import random
from collections import Counter
from typing import Dict

import importlib

import pytest
from prometheus_client import CollectorRegistry

from dgm_kernel import mutation_strategies, meta_loop, metrics
from dgm_kernel.mutation_strategies import ASTInsertComment, ASTRenameIdentifier


def get_metric_value(registry: CollectorRegistry, name: str, labels: Dict[str, str]) -> float:
    val = registry.get_sample_value(name, labels=labels)
    return val if val is not None else 0.0


def chi2_pvalue(observed: list[float], expected: list[float]) -> float:
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))


def test_weighted_choice_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)

    for _ in range(80):
        metrics.increment_mutation_success(strategy="ASTInsertComment", registry=registry)
    for _ in range(20):
        metrics.increment_mutation_failure(strategy="ASTInsertComment", registry=registry)
    for _ in range(20):
        metrics.increment_mutation_success(strategy="ASTRenameIdentifier", registry=registry)
    for _ in range(80):
        metrics.increment_mutation_failure(strategy="ASTRenameIdentifier", registry=registry)

    strategies = [ASTInsertComment, ASTRenameIdentifier]
    counts: Counter[str] = Counter()
    random.seed(0)
    for _ in range(1000):
        strat = mutation_strategies.weighted_choice(strategies)
        counts[strat.name] += 1

    w1 = min(0.7, max(0.05, 80 / (80 + 20 + 1e-3)))
    w2 = min(0.7, max(0.05, 20 / (20 + 80 + 1e-3)))
    total = w1 + w2
    expected = [w1 / total * 1000, w2 / total * 1000]
    observed = [counts["ASTInsertComment"], counts["ASTRenameIdentifier"]]

    assert chi2_pvalue(observed, expected) > 0.05


def test_mutation_counters(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(metrics, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(meta_loop.metrics, "DEFAULT_REGISTRY", registry)

    target = tmp_path / "mod.py"
    patch = {"target": str(target), "after": "", "before": ""}

    before = get_metric_value(
        registry,
        "dgm_mutation_success_total",
        {"strategy": meta_loop._MUTATION_NAME},
    )
    assert meta_loop._apply_patch(patch) is True
    assert get_metric_value(
        registry,
        "dgm_mutation_success_total",
        {"strategy": meta_loop._MUTATION_NAME},
    ) == before + 1.0

    def fail_import(name: str):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    before_fail = get_metric_value(
        registry,
        "dgm_mutation_failure_total",
        {"strategy": meta_loop._MUTATION_NAME},
    )
    assert meta_loop._apply_patch(patch) is False
    assert get_metric_value(
        registry,
        "dgm_mutation_failure_total",
        {"strategy": meta_loop._MUTATION_NAME},
    ) == before_fail + 1.0

