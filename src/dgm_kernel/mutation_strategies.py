"""Mutation strategies for the Darwin-Gödel Machine (DGM).

DGM design PDF §2.3 describes two simple AST-based mutations:

* ``ASTInsertComment`` – inserts a harmless string literal as the first
  statement in a module (acts like a comment).
* ``ASTRenameIdentifier`` – renames the first function it finds by
  appending ``_renamed`` to the identifier.

Additional strategies can be plugged-in at runtime by setting the
``DGM_MUTATION`` environment variable.
"""

from __future__ import annotations

import ast
import random
from typing import Any, Mapping, Protocol, Sequence, Type, cast

from prometheus_client import CollectorRegistry, REGISTRY as DEFAULT_REGISTRY, Counter
from . import metrics

DECAY = 0.995

__all__ = [
    "MutationStrategy",
    "ASTInsertComment",
    "ASTRenameIdentifier",
    "DEFAULT_REGISTRY",
    "weighted_choice",
    "choose_mutation",
]


# --------------------------------------------------------------------------- #
# Strategy protocol & concrete AST strategies
# --------------------------------------------------------------------------- #
class MutationStrategy(Protocol):
    """Strategy interface for code mutation."""

    name: str

    def mutate(self, code: str) -> str: ...


class ASTInsertComment(MutationStrategy):
    """Insert a string literal at the start of the module (no-op)."""

    name = "ASTInsertComment"

    def mutate(self, code: str) -> str:
        module = ast.parse(code)
        module.body.insert(0, ast.Expr(value=ast.Constant("mutated")))
        ast.fix_missing_locations(module)
        return ast.unparse(module)


class ASTRenameIdentifier(MutationStrategy):
    """Rename the first *top-level* function definition."""

    name = "ASTRenameIdentifier"

    def mutate(self, code: str) -> str:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.name = f"{node.name}_renamed"
                break
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


# --------------------------------------------------------------------------- #
# Strategy selection helpers
# --------------------------------------------------------------------------- #
def _get_registry() -> CollectorRegistry:  # small helper for test monkey-patch
    return cast(CollectorRegistry, getattr(metrics, "DEFAULT_REGISTRY"))


def weighted_choice(strategies: Sequence[Type[MutationStrategy]]) -> Type[MutationStrategy]:
    """Choose a mutation weighted by past success ratio (Prometheus counters)."""
    if not strategies:
        raise ValueError("No strategies provided")

    registry = _get_registry()

    weights: list[float] = []
    for strat in strategies:
        succ_val = registry.get_sample_value(
            "dgm_mutation_success_total", labels={"strategy": strat.name}
        )
        fail_val = registry.get_sample_value(
            "dgm_mutation_failure_total", labels={"strategy": strat.name}
        )
        succ = (succ_val or 0.0) * DECAY
        fail = (fail_val or 0.0) * DECAY

        succ_metric = cast(Counter | None, registry._names_to_collectors.get("dgm_mutation_success"))
        if succ_metric is not None:
            succ_metric.labels(strategy=strat.name)._value.set(succ)
        fail_metric = cast(Counter | None, registry._names_to_collectors.get("dgm_mutation_failure"))
        if fail_metric is not None:
            fail_metric.labels(strategy=strat.name)._value.set(fail)

        if succ_val is None and fail_val is None:
            ratio = 0.5
        else:
            ratio = succ / (succ + fail + 1e-3)

        # keep ratios within a reasonable band so every strategy has a chance
        weights.append(min(0.7, max(0.05, ratio)))

    return random.choices(list(strategies), weights=weights, k=1)[0]


def choose_mutation(
    traces: Sequence[Mapping[str, Any]] | None = None,
) -> Type[MutationStrategy]:
    """Return the *class* of the mutation strategy to apply next.

    Parameters
    ----------
    traces
        Currently unused placeholder for future adaptive selection logic.
    """
    _ = traces  # reserved for future use
    return weighted_choice([ASTInsertComment, ASTRenameIdentifier])
