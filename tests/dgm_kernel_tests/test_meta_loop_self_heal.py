import importlib
import os
import sys
import time
from collections import defaultdict
from prometheus_client import Counter, CollectorRegistry
from hypothesis import given, strategies as st, settings, HealthCheck
import pytest

class _RedisModule:
    class Redis:
        def __init__(self, *_, **__):
            self._client = SimpleRedis()

        def __getattr__(self, name):
            return getattr(self._client, name)

    class exceptions:
        class RedisError(Exception):
            ...


sys.modules.setdefault("redis", _RedisModule())

class SimpleRedis:
    def __init__(self):
        self.store = defaultdict(list)

    def lpush(self, name, value):
        self.store[name].insert(0, value)

    def rpop(self, name):
        lst = self.store[name]
        if lst:
            return lst.pop()
        return None

    def ping(self):
        return True


from dgm_kernel import meta_loop, metrics

@pytest.fixture
def fake_redis():
    r = SimpleRedis()
    r.ping()
    return r


@pytest.fixture(autouse=True)
def patch_redis_client(monkeypatch, fake_redis):
    monkeypatch.setattr(meta_loop, "REDIS", fake_redis)


@settings(max_examples=1, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.just(5))
def test_sleep_after_rollbacks(monkeypatch, num_failures):
    async def fake_fetch(*_args, **_kwargs):
        return [{}]

    async def gen_patch(traces):
        return {"target": "t.py", "before": "", "after": ""}

    async def verify(traces, patch):
        return False

    rollbacks = {"n": 0}

    def rb(_patch):
        rollbacks["n"] += 1

    sleep_calls = []

    def fake_sleep(s):
        sleep_calls.append(s)
        if rollbacks["n"] >= num_failures:
            raise StopIteration

    registry = CollectorRegistry(auto_describe=True)
    counter = Counter(
        "dgm_rollback_backoff_total",
        "Number of times meta-loop slept due to consecutive rollbacks",
        registry=registry,
    )

    monkeypatch.setattr(meta_loop.metrics, "rollback_backoff_total", counter)
    monkeypatch.setattr(metrics, "rollback_backoff_total", counter)

    monkeypatch.setenv("DGM_MUTATION", "OtherMutation")

    monkeypatch.setattr(meta_loop, "fetch_recent_traces", fake_fetch)
    monkeypatch.setattr(meta_loop, "_generate_patch_async", gen_patch)
    monkeypatch.setattr(meta_loop, "_verify_patch", verify)
    monkeypatch.setattr(meta_loop, "_rollback", rb)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    with pytest.raises(StopIteration):
        meta_loop.loop_forever()

    assert any(sec >= meta_loop.ROLLBACK_SLEEP_S for sec in sleep_calls)
    assert counter._value.get() >= 1.0
    assert os.environ["DGM_MUTATION"] == "ASTInsertComment"


def test_env_var_reload(monkeypatch):
    monkeypatch.setenv("DGM_MUTATION", "ASTRenameIdentifier")
    importlib.reload(meta_loop)
    assert type(meta_loop._mutation_strategy).__name__ == "ASTRenameIdentifier"

    async def fetch(*_args, **_kwargs):
        return [{}]

    async def none_patch(traces):
        return None

    calls = {"sleep": 0}

    def fake_sleep(_):
        calls["sleep"] += 1
        if calls["sleep"] > 3:
            raise StopIteration

    monkeypatch.setattr(meta_loop, "fetch_recent_traces", fetch)
    monkeypatch.setattr(meta_loop, "_generate_patch_async", none_patch)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    with pytest.raises(StopIteration):
        meta_loop.loop_forever()

    assert os.environ["DGM_MUTATION"] == "ASTInsertComment"
    importlib.reload(meta_loop)
    assert type(meta_loop._mutation_strategy).__name__ == "ASTInsertComment"
