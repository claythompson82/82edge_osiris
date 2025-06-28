import ast
import keyword
import string
import importlib
import sys

from hypothesis import given, strategies as st, settings, HealthCheck


class _RedisModule:
    class Redis:
        def __init__(self, *_, **__):
            self._client = SimpleRedis()

        def __getattr__(self, name):
            return getattr(self._client, name)

    class exceptions:
        class RedisError(Exception):
            ...


class SimpleRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}

    def lpush(self, name: str, value: str) -> None:
        self.store.setdefault(name, []).insert(0, value)

    def rpop(self, name: str):
        lst = self.store.get(name, [])
        return lst.pop() if lst else None


sys.modules.setdefault("redis", _RedisModule())

from dgm_kernel import meta_loop
from dgm_kernel.mutation_strategies import ASTInsertComment, ASTRenameIdentifier

ident = st.text(string.ascii_lowercase, min_size=1).filter(lambda s: s not in keyword.kwlist)
func_strategy = st.builds(
    lambda name, arg, val: f"def {name}({arg}):\n    return {arg} + {val}\n",
    ident,
    ident,
    st.integers(),
)


@given(code=func_strategy)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_strategies_produce_parsable_code(code: str) -> None:
    for strat in (ASTInsertComment(), ASTRenameIdentifier()):
        mutated = strat.mutate(code)
        ast.parse(mutated)


def test_env_switching(monkeypatch) -> None:
    code = "def f(x):\n    return x\n"
    monkeypatch.setenv("DGM_MUTATION", "ASTRenameIdentifier")
    importlib.reload(meta_loop)
    renamed = meta_loop._generate_patch(code)
    assert "f_renamed" in renamed

    monkeypatch.setenv("DGM_MUTATION", "ASTInsertComment")
    importlib.reload(meta_loop)
    commented = meta_loop._generate_patch(code)
    assert '"""' in commented.splitlines()[0]

