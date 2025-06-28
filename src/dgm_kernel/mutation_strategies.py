"""Mutation strategies for Darwin-G\xf6del Machine (DGM).

This module implements two simple AST-based transformations from
DGM design PDF \xa7 2.3:

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
from typing import Protocol


class MutationStrategy(Protocol):
    """Strategy interface for code mutation."""

    def mutate(self, code: str) -> str:
        """Return a mutated version of ``code``."""
        raise NotImplementedError


class ASTInsertComment:
    """Insert a string literal at the beginning of the module."""

    def mutate(self, code: str) -> str:
        module = ast.parse(code)
        module.body.insert(0, ast.Expr(value=ast.Constant("mutated")))
        ast.fix_missing_locations(module)
        return ast.unparse(module)


class ASTRenameIdentifier:
    """Rename the first function definition found in the module."""

    def mutate(self, code: str) -> str:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.name = f"{node.name}_renamed"
                break
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
