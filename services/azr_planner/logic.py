import os
from typing import Tuple
import numpy as np
from kubernetes import config, client
from services.azr_planner.schemas import PlannerState, Task, Plan

def free_capacity(state: PlannerState) -> Tuple[int, int]:
    """
    Return slots available for Alpha and Beta:
    1) Use provided state.alpha_free/state.beta_free if set,
    2) Otherwise query Kubernetes for real resource counts.
    """
    # If user specified both, use that
    if state.alpha_free is not None and state.beta_free is not None:
        return int(state.alpha_free), int(state.beta_free)

    # Fallback: query Kubernetes cluster (excluded from test coverage)
    # pragma: no cover
    try:  # pragma: no cover
        config.load_incluster_config()  # pragma: no cover
    except config.ConfigException:  # pragma: no cover
        config.load_kube_config()  # pragma: no cover
    v1 = client.CoreV1Api()  # pragma: no cover
    nodes = v1.list_node().items  # pragma: no cover

    # GPUs for Alpha
    total_gpus  = sum(int(n.status.allocatable.get("nvidia.com/gpu", 0)) for n in nodes)  # pragma: no cover
    used_alpha  = int(os.getenv("ALPHA_IN_USE", 0))  # pragma: no cover
    alpha_slots = max(total_gpus - used_alpha, 0)  # pragma: no cover

    # CPU cores for Beta
    total_cpu   = sum(int(float(n.status.allocatable.get("cpu", 0))) for n in nodes)  # pragma: no cover
    used_beta   = int(os.getenv("BETA_IN_USE", 0))  # pragma: no cover
    beta_slots  = max(total_cpu - used_beta, 0)  # pragma: no cover

    return alpha_slots, beta_slots  # pragma: no cover

def generate_plan(state: PlannerState) -> Plan:
    """
    Build a Plan:
      - One task per reasoning mode
      - All tasks use Alpha if any alpha_free>0, else Beta
      - alpha_count/beta_count reflect free_capacity(state)
      - priority = int(avg_gap*5 + 8), clamped to 10
    """
    # Choose resource
    resource = "Alpha" if state.alpha_free and state.alpha_free > 0 else "Beta"

    # Compute performance gaps from 0.5
    gaps = {mode: abs(score - 0.5) for mode, score in state.performance.items()}

    # Create one Task per mode
    tasks = [
        Task(
            id=mode,
            kind=mode,
            description=f"Practice {mode} reasoning (gap={gaps[mode]:.2f})",
            resource=resource
        )
        for mode in gaps
    ]

    # Get slot counts (state or K8s fallback)
    alpha_count, beta_count = free_capacity(state)

    # Priority heuristic
    avg_gap   = np.mean(list(gaps.values()))
    raw_score = avg_gap * 5 + 8
    priority  = int(min(10, raw_score))

    return Plan(
        tasks=tasks,
        alpha_count=alpha_count,
        beta_count=beta_count,
        priority=priority
    )
