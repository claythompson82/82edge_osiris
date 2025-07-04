from __future__ import annotations

import ast
import copy
import random
from pathlib import Path
from typing import List


def load_corpus() -> list[str]:
    """Return list of mutation snippets from ``corpus/mutations``."""
    corpus_dir = Path(__file__).resolve().parents[2] / "corpus" / "mutations"
    snippets: list[str] = []
    for path in sorted(corpus_dir.glob("*.py")):
        snippets.append(path.read_text())
    return snippets


class MutationFuzzer:
    """Corpus-driven mutation fuzzer."""

    def __init__(self) -> None:
        self._corpus = load_corpus()

    def fuzz_source(self, code: str) -> str:
        """Return a mutated version of ``code`` that always parses."""
        target_tree = ast.parse(code)
        snippet_src = random.choice(self._corpus)
        snippet_tree = ast.parse(snippet_src)

        def stmt_size(stmt: ast.stmt) -> int:
            return len(ast.unparse(stmt))

        candidates = [s for s in snippet_tree.body if stmt_size(s) <= len(code)]
        stmt = copy.deepcopy(
            random.choice(candidates) if candidates else min(snippet_tree.body, key=stmt_size)
        )

        bodies: List[List[ast.stmt]] = []
        for node in ast.walk(target_tree):
            for attr in ("body", "orelse", "finalbody"):
                val = getattr(node, attr, None)
                if isinstance(val, list):
                    bodies.append(val)
        body_list = random.choice(bodies)
        idx = random.randint(0, len(body_list))
        body_list.insert(idx, stmt)

        ast.fix_missing_locations(target_tree)
        mutated = ast.unparse(target_tree)
        if len(mutated) > 2 * len(code):
            return code
        ast.parse(mutated)
        return mutated


fuzzer = MutationFuzzer()
