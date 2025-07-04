from __future__ import annotations

from typing import Dict, List, Optional
from prometheus_client import Counter, Gauge, CollectorRegistry, REGISTRY as DEFAULT_REGISTRY

_PATCH_APPLIED = "dgm_patches_applied_total"
_PATCH_GENERATION = "dgm_patch_generation_total"
_MUTATION_SUCCESS = "dgm_mutation_success_total"
_MUTATION_FAILURE = "dgm_mutation_failure_total"
unsafe_token_found_total = Counter(
    "dgm_unsafe_token_found_total",
    "Number of patches rejected due to dangerous tokens",
)

# Counts how often the meta-loop backed off due to repeated rollbacks
rollback_backoff_total = Counter(
    "dgm_rollback_backoff_total",
    "Number of times meta-loop slept due to consecutive rollbacks",
)

# Counts invalid patch signatures during verification
patch_sig_invalid_total = Counter(
    "dgm_patch_sig_invalid_total",
    "Number of patches rejected due to invalid signature",
)

# Counts validation failures for incoming traces
trace_validation_fail_total = Counter(
    "dgm_trace_validation_fail_total",
    "Number of traces dropped due to schema validation errors",
)

# Tracks the minutes-per-patch average for the last 10 patches applied.
patch_apply_minutes_average = Gauge(
    "dgm_patch_apply_minutes_average",
    "Average minutes between patch applications over the last 10 patches",
)

# Tracks total CPU time and RAM usage consumed by sandbox executions
sandbox_cpu_ms_total = Counter(
    "dgm_sandbox_cpu_ms_total",
    "Total CPU milliseconds consumed by sandboxed patches",
)

sandbox_ram_mb_total = Counter(
    "dgm_sandbox_ram_mb_total",
    "Total RAM megabytes consumed by sandboxed patches",
)

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


def increment_mutation_success(*, strategy: str, registry: Optional[CollectorRegistry] = None) -> None:
    reg = registry if registry is not None else DEFAULT_REGISTRY
    c = _get_or_create(_MUTATION_SUCCESS, ["strategy"], "Total successful mutations", reg)
    c.labels(strategy=strategy).inc()


def increment_mutation_failure(*, strategy: str, registry: Optional[CollectorRegistry] = None) -> None:
    reg = registry if registry is not None else DEFAULT_REGISTRY
    c = _get_or_create(_MUTATION_FAILURE, ["strategy"], "Total failed mutations", reg)
    c.labels(strategy=strategy).inc()
