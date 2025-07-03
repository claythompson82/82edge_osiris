from __future__ import annotations

from typing import Dict, List, Optional
from prometheus_client import Counter, CollectorRegistry, REGISTRY as DEFAULT_REGISTRY

_PATCH_APPLIED = "dgm_patches_applied_total"
_PATCH_GENERATION = "dgm_patch_generation_total"

_counters: Dict[CollectorRegistry, Dict[str, Counter]] = {}


def _get_or_create(name: str, labels: List[str], help_text: str, registry: CollectorRegistry) -> Counter:
    if registry not in _counters:
        _counters[registry] = {}
    metric = _counters[registry].get(name)
    if metric is None:
        metric = Counter(name, help_text, labels, registry=registry)
        _counters[registry][name] = metric
    if not isinstance(metric, Counter):
        raise TypeError(f"Metric {name} in registry is not a Counter")
    return metric


def increment_patch_generation(*, mutation: str, result: str, registry: Optional[CollectorRegistry] = None) -> None:
    reg = registry if registry is not None else DEFAULT_REGISTRY
    c = _get_or_create(_PATCH_GENERATION, ["mutation", "result"], "Total number of patches generated", reg)
    c.labels(mutation=mutation, result=result).inc()


def increment_patch_apply(*, mutation: str, result: str, registry: Optional[CollectorRegistry] = None) -> None:
    reg = registry if registry is not None else DEFAULT_REGISTRY
    c = _get_or_create(_PATCH_APPLIED, ["mutation", "result"], "Total number of patches applied", reg)
    c.labels(mutation=mutation, result=result).inc()
