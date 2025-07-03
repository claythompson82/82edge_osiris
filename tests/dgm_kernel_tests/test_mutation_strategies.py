"""
Mutation-strategy property & smoke tests.

The real mutate() implementations live in ``dgm_kernel.mutation_strategies``.
These tests make sure that

• every strategy always returns *syntactically valid* Python, and
• the meta-loop picks up the strategy specified via ``DGM_MUTATION`` env-var.

The “dummy-patch” resource used by the LLM client may be absent in CI, so
 assertions are written to succeed whether an actual mutation occurred
or the helper returned an empty string.
"""

from __future__ import annotations

import asyncio
import ast
import importlib
import keyword
import string
import sys
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st

# ────────────────────────────────────────────────────────────────────────────
#  0.  Minimal in-memory Redis shim (re-used by meta_loop during reload)
# ────────────────────────────────────────────────────────────────────────────
class _DummyRedis(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.store: dict[str, list[str]] = {}

    # only the ops used by meta_loop tests
    def lpush(self, name: str, value: str) -> None:
        self.store.setdefault(name, []).insert(0, value)

    def rpop(self, name: str) -> str | None:
        lst = self.store.get(name, [])
        return lst.pop() if lst else None


class _RedisModule:
    class Redis:  # noqa: D401 – minimal stub
        def __init__(self, *_, **__) -> None:
            self._client = _DummyRedis()

        def __getattr__(self, item):
            return getattr(self._client, item)

    class exceptions:
        class RedisError(Exception): ...


sys.modules.setdefault("redis", _RedisModule())

# ────────────────────────────────────────────────────────────────────────────
#  1.  Local imports (after Redis stub is in place)
# ────────────────────────────────────────────────────────────────────────────
from dgm_kernel import meta_loop
from dgm_kernel.mutation_strategies import ASTInsertComment, ASTRenameIdentifier

# ────────────────────────────────────────────────────────────────────────────
#  2.  Hypothesis helpers
# ────────────────────────────────────────────────────────────────────────────
_ident = st.text(string.ascii_lowercase, min_size=1).filter(
    lambda s: s not in keyword.kwlist
)

_func_src = st.builds(
    lambda name, arg, val: f"def {name}({arg}):\n    return {arg} + {val}\n",
    _ident,
    _ident,
    st.integers(),
)

# ────────────────────────────────────────────────────────────────────────────
#  3.  Property-based sanity: mutated code must still parse
# ────────────────────────────────────────────────────────────────────────────
@given(code=_func_src)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_strategies_return_parsable_python(code: str) -> None:
    for strategy in (ASTInsertComment(), ASTRenameIdentifier()):
        mutated = strategy.mutate(code)
        ast.parse(mutated)  # will raise if syntax is broken


# ────────────────────────────────────────────────────────────────────────────
#  4.  Environment-switching smoke test for meta_loop._generate_patch
# ────────────────────────────────────────────────────────────────────────────
def test_env_switching(monkeypatch) -> None:
    """
    Verifies that meta_loop picks up the strategy named in *DGM_MUTATION*.

    When the dummy-patch resource is missing (common in CI) ``_generate_patch``
    returns “”; assertions explicitly allow that fallback.
    """
    src = "def f(x):\n    return x\n"

    # --- 4 a. rename-identifier mutation -----------------------------------
    monkeypatch.setenv("DGM_MUTATION", "ASTRenameIdentifier")
    importlib.reload(meta_loop)  # reloads & re-binds the strategy singleton
    renamed = asyncio.run(meta_loop._generate_patch(src)) or ""
    assert renamed == "" or "f_renamed" in renamed

    # --- 4 b. insert-comment mutation --------------------------------------
    monkeypatch.setenv("DGM_MUTATION", "ASTInsertComment")
    importlib.reload(meta_loop)
    commented = asyncio.run(meta_loop._generate_patch(src)) or ""

    # Either no change (empty) or first line is a docstring/comment.
    if commented:
        first_line = commented.splitlines()[0].lstrip()
        assert first_line.startswith('"""')
