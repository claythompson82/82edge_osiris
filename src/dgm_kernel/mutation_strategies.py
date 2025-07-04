"""Mutation strategies for Darwin-Gödel Machine (DGM).

This module implements two simple AST-based transformations from
DGM design PDF § 2.3:

- ``ASTInsertComment`` inserts a no-op string literal at the start of a
  module, effectively acting as a comment.
- ``ASTRenameIdentifier`` renames the first function definition by
  appending ``_renamed`` to its name.

Both strategies parse the input code and return syntactically valid
Python source. Additional strategies can be plugged in via the
``DGM_MUTATION`` environment variable.
"""

from __future__ import annotations

import ast
import random
from typing import TYPE_CHECKING, Protocol, Sequence, Mapping, Any, Type

from prometheus_client import CollectorRegistry, REGISTRY as DEFAULT_REGISTRY

__all__ = [
    "ASTInsertComment",
    "ASTRenameIdentifier",
    "MutationStrategy",
    "DEFAULT_REGISTRY",
    "weighted_choice",
    "choose_mutation",
]

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from dgm_kernel import metrics


class MutationStrategy(Protocol):
    """Strategy interface for code mutation."""

    name: str

    def mutate(self, code: str) -> str:
        """Return a mutated version of ``code``."""
        raise NotImplementedError


class ASTInsertComment(MutationStrategy):
    """Insert a string literal at the beginning of the module."""

    name = "ASTInsertComment"

    def mutate(self, code: str) -> str:
        module = ast.parse(code)
        module.body.insert(0, ast.Expr(value=ast.Constant("mutated")))
        ast.fix_missing_locations(module)
        return ast.unparse(module)


class ASTRenameIdentifier(MutationStrategy):
    """Rename the first function definition found in the module."""

    name = "ASTRenameIdentifier"

    def mutate(self, code: str) -> str:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.name = f"{node.name}_renamed"
                break
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


def weighted_choice(strategies: list[MutationStrategy]) -> MutationStrategy:
    """Choose a strategy based on past success/failure metrics."""
    if not strategies:
        raise ValueError("No strategies provided")

    # Metrics registry may be overridden in tests via monkeypatch
    registry: CollectorRegistry = DEFAULT_REGISTRY
    weights = []
    for strat in strategies:
        succ = (
            registry.get_sample_value(
                "dgm_mutation_success_total", labels={"strategy": strat.name}
            )
            or 0.0
        )
        fail = (
            registry.get_sample_value(
                "dgm_mutation_failure_total", labels={"strategy": strat.name}
            )
            or 0.0
        )

        val = succ / (succ + fail + 1e-3)
        val = min(0.7, max(0.05, val))
        weights.append(val)

    chosen = random.choices(strategies, weights=weights, k=1)[0]
    return chosen


def choose_mutation(
    traces: Sequence[Mapping[str, Any]] | None = None,
) -> Type[MutationStrategy]:
    """Return the mutation class for the next patch."""

    _ = traces  # currently unused
    selected = weighted_choice([ASTInsertComment(), ASTRenameIdentifier()])
    return type(selected)
