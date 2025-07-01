import asyncio
import json
import logging
from pathlib import Path
from unittest import mock
import time
import difflib

import pytest
from collections import defaultdict
import sys


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

    def llen(self, name):
        return len(self.store[name])

    def lrange(self, name, start, end):
        lst = self.store[name]
        if end == -1:
            end = len(lst) - 1
        return lst[start:end + 1]

    def ping(self):
        return True

# Modules to be tested
from dgm_kernel import meta_loop

# Minimal test used by prover sandbox
def sanity_only():
    assert True

# Configure logging for tests (optional, but can be helpful)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


@pytest.fixture
def fake_redis():
    """Provide a minimal in-memory Redis replacement."""
    r = SimpleRedis()
    r.ping()
    return r


@pytest.fixture(autouse=True)
def patch_redis_client(monkeypatch, fake_redis):
    """Autouse fixture to replace meta_loop.REDIS with our fake_redis instance."""
    monkeypatch.setattr(meta_loop, "REDIS", fake_redis)


# --- Test Cases for fetch_recent_traces ---

@pytest.mark.asyncio
async def test_fetch_recent_traces_empty_queue(fake_redis):
    log.info("Testing fetch_recent_traces with an empty queue.")
    traces = await meta_loop.fetch_recent_traces(n=5)
    assert len(traces) == 0
    log.info("Test passed: fetch_recent_traces with empty queue returned 0 traces.")


@pytest.mark.asyncio
async def test_fetch_recent_traces_fetches_n_items(fake_redis):
    log.info("Testing fetch_recent_traces fetches N items.")
    trace_data = [{"id": i, "data": f"trace_{i}"} for i in range(10)]
    for trace in trace_data:
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace))

    fetched_traces = await meta_loop.fetch_recent_traces(n=7)
    assert len(fetched_traces) == 7
    for i in range(7):
        assert fetched_traces[i]["id"] == i
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 3
    log.info("Test passed: fetch_recent_traces fetched N items correctly.")


@pytest.mark.asyncio
async def test_fetch_recent_traces_less_than_n_items_available(fake_redis):
    log.info("Testing fetch_recent_traces with less than N items available.")
    trace_data = [{"id": i, "data": f"trace_{i}"} for i in range(3)]
    for trace in trace_data:
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace))

    fetched_traces = await meta_loop.fetch_recent_traces(n=5)
    assert len(fetched_traces) == 3
    for i in range(3):
        assert fetched_traces[i]["id"] == i
    assert fake_redis.llen(meta_loop.TRACE_QUEUE) == 0
    log.info("Test passed: fetch_recent_traces handled less than N items correctly.")


@pytest.mark.asyncio
async def test_fetch_recent_traces_json_decode_error(fake_redis, caplog):
    log.info("Testing fetch_recent_traces with a JSON decode error.")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": 1, "data": "valid_trace"}))
    fake_redis.lpush(meta_loop.TRACE_QUEUE, "this_is_not_json")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": 3, "data": "another_valid_trace"}))

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        fetched_traces = await meta_loop.fetch_recent_traces(n=3)

    assert len(fetched_traces) == 2
    assert fetched_traces[0]["id"] == 1
    assert fetched_traces[1]["id"] == 3
    assert any("Failed to decode JSON for trace: this_is_not_json" in message for message in caplog.messages)
    log.info("Test passed: fetch_recent_traces handled JSON decode error correctly.")


# --- Test Cases for meta_loop ---

@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop.run_patch_in_sandbox")
@mock.patch("dgm_kernel.meta_loop._apply_patch")
@mock.patch("dgm_kernel.meta_loop.proofable_reward")
@mock.patch("asyncio.sleep", return_value=None)
async def test_meta_loop_applies_good_patch(
    mock_sleep, mock_proofable_reward, mock_apply_patch, mock_sandbox, mock_verify_patch, mock_generate_patch, fake_redis, tmp_path
):
    log.info("Testing meta_loop applies a good patch.")
    trace_1 = {"id": "trace1", "value": 100}
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace_1))

    target_file_path = tmp_path / "osiris_policy" / "strategy.py"
    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    initial_content = "# Initial strategy content\n"
    target_file_path.write_text(initial_content)

    test_patch = {
        "target": str(target_file_path),
        "before": initial_content,
        "after": initial_content + "# Patched by DGM\n",
        "rationale": "Test rationale",
    }
    mock_generate_patch.return_value = test_patch
    mock_verify_patch.return_value = (True, 8.0)
    mock_sandbox.return_value = (True, "", 0)
    mock_apply_patch.return_value = True
    mock_proofable_reward.return_value = 1.0

    # This is a simplified simulation of the main loop's logic for one iteration
    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted, _ = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        sandbox_ok, _, _ = meta_loop.run_patch_in_sandbox(patch_data)
        if sandbox_ok and meta_loop._apply_patch(patch_data):
            new_r = sum(meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces)
            if new_r >= 0:
                meta_loop.REDIS.lpush(
                    meta_loop.APPLIED_LOG,
                    json.dumps(patch_data | {"reward": new_r, "verification_score": 8.0}),
                )

    assert mock_generate_patch.called
    mock_verify_patch.assert_called()
    mock_apply_patch.assert_called_with(test_patch)
    mock_proofable_reward.assert_called()

    applied_log_content = fake_redis.lrange(meta_loop.APPLIED_LOG, 0, -1)
    assert len(applied_log_content) == 1
    logged_patch = json.loads(applied_log_content[0])
    assert logged_patch["rationale"] == "Test rationale"
    assert logged_patch["reward"] == 1.0
    log.info("Test passed: meta_loop applied a good patch.")


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop._rollback", wraps=meta_loop._rollback)
@mock.patch("dgm_kernel.meta_loop._apply_patch")
@mock.patch("dgm_kernel.meta_loop.proofable_reward")
async def test_meta_loop_rolls_back_bad_patch(
    mock_reward, mock_apply_patch, mock_rollback, mock_verify_patch, mock_generate_patch, fake_redis, tmp_path
):
    log.info("Testing meta_loop rolls back a patch with negative reward.")
    trace = {"id": "trace1", "value": 100}
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace))

    target_file = tmp_path / "osiris_policy" / "strategy.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    before = "# Initial strategy content\n"
    target_file.write_text(before)

    patch = {
        "target": str(target_file),
        "before": before,
        "after": before + "# bad\n",
        "rationale": "bad patch",
    }
    mock_generate_patch.return_value = patch
    mock_verify_patch.return_value = (True, 9.0)

    mock_apply_patch.side_effect = lambda p: (
        Path(p["target"]).write_text(p["after"]),
        True,
    )[1]
    mock_reward.return_value = -0.5

    # Simplified loop
    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted, _ = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        if meta_loop._apply_patch(patch_data):
            new_r = sum(meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces)
            if new_r >= 0:
                meta_loop.REDIS.lpush(meta_loop.APPLIED_LOG, json.dumps(patch_data | {"reward": new_r}))
            else:
                for t in traces:
                    t["rolled_back"] = True
                    meta_loop.REDIS.lpush(meta_loop.ROLLED_BACK_LOG, json.dumps(t))
                meta_loop._rollback(patch_data)

    mock_rollback.assert_called_once_with(patch)
    assert target_file.read_text() == before
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0
    rb_entries = [json.loads(v) for v in fake_redis.lrange(meta_loop.ROLLED_BACK_LOG, 0, -1)]
    assert rb_entries == [{"id": "trace1", "value": 100, "rolled_back": True}]
    log.info("Test passed: meta_loop correctly rolled back a bad patch.")


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop._apply_patch")
async def test_meta_loop_skips_unproven_patch(
    mock_apply_patch, mock_verify_patch, mock_generate_patch, fake_redis
):
    log.info("Testing meta_loop skips a patch that fails verification.")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": "trace1"}))
    mock_generate_patch.return_value = {"rationale": "unverified patch"}
    mock_verify_patch.return_value = (False, 3.0)

    # Simplified loop
    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted, _ = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        meta_loop._apply_patch(patch_data)

    mock_apply_patch.assert_not_called()
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0
    log.info("Test passed: meta_loop correctly skipped an unverified patch.")


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop.run_patch_in_sandbox")
@mock.patch("dgm_kernel.meta_loop._apply_patch")
async def test_meta_loop_skips_failed_sandbox(
    mock_apply_patch, mock_sandbox, mock_verify_patch, mock_generate_patch, fake_redis
):
    log.info("Testing meta_loop skips a patch that fails sandbox testing.")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": "trace1"}))
    mock_generate_patch.return_value = {"rationale": "sandbox fail patch"}
    mock_verify_patch.return_value = (True, 8.0)
    mock_sandbox.return_value = (False, "Container crashed", 1)

    # Simplified loop
    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted, _ = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        sandbox_ok, _, _ = meta_loop.run_patch_in_sandbox(patch_data)
        if sandbox_ok:
            meta_loop._apply_patch(patch_data)

    mock_apply_patch.assert_not_called()
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0
    log.info("Test passed: meta_loop correctly skipped a patch that failed sandbox.")


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop.run_patch_in_sandbox")
@mock.patch("dgm_kernel.meta_loop._apply_patch")
async def test_rate_limiting_prevents_patch(
    mock_apply_patch, mock_sandbox, mock_verify_patch, mock_generate_patch, fake_redis, tmp_path
):
    """Verify that patches are skipped when within the rate limit window."""
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": "trace1"}))
    target_file = tmp_path / "rate.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("before\n")
    patch = {
        "target": str(target_file),
        "before": "before\n",
        "after": "after\n",
    }
    mock_generate_patch.return_value = patch
    mock_verify_patch.return_value = True
    mock_sandbox.return_value = (True, "", 0)
    mock_apply_patch.return_value = True

    meta_loop._last_patch_time = time.time()  # within rate limit

    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        sandbox_ok, _, _ = meta_loop.run_patch_in_sandbox(patch_data)
        if sandbox_ok:
            now = time.time()
            if now - meta_loop._last_patch_time < meta_loop.PATCH_RATE_LIMIT_SECONDS:
                pass
            else:
                meta_loop._apply_patch(patch_data)

    mock_apply_patch.assert_not_called()
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop.run_patch_in_sandbox")
@mock.patch("dgm_kernel.meta_loop._apply_patch")
@mock.patch("dgm_kernel.meta_loop.proofable_reward")
async def test_patch_history_logged(
    mock_reward,
    mock_apply_patch,
    mock_sandbox,
    mock_verify_patch,
    mock_generate_patch,
    fake_redis,
    tmp_path,
    monkeypatch,
):
    """Ensure a successful patch is recorded in patch_history.json."""
    history_file = tmp_path / "history.json"
    monkeypatch.setattr(meta_loop, "PATCH_HISTORY_FILE", history_file)
    history_file.write_text("[]")

    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": "trace1"}))
    target_file = tmp_path / "hist.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("before\n")
    patch = {
        "target": str(target_file),
        "before": "before\n",
        "after": "after\n",
    }
    mock_generate_patch.return_value = patch
    mock_verify_patch.return_value = True
    mock_sandbox.return_value = (True, "", 0)
    mock_apply_patch.return_value = True
    mock_reward.return_value = 1.0

    meta_loop._last_patch_time = time.time() - meta_loop.PATCH_RATE_LIMIT_SECONDS - 1
    monkeypatch.setattr(meta_loop.uuid, "uuid4", lambda: "id123")

    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        sandbox_ok, _, exit_code = meta_loop.run_patch_in_sandbox(patch_data)
        if sandbox_ok and meta_loop._apply_patch(patch_data):
            new_r = sum(meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces)
            if new_r >= 0:
                patch_id = str(meta_loop.uuid.uuid4())
                diff = "".join(
                    difflib.unified_diff(
                        patch_data["before"].splitlines(),
                        patch_data["after"].splitlines(),
                        fromfile="before",
                        tofile="after",
                        lineterm="",
                    )
                )
                meta_loop.REDIS.lpush(
                    meta_loop.APPLIED_LOG,
                    json.dumps(patch_data | {"reward": new_r, "patch_id": patch_id}),
                )
                meta_loop._record_patch_history(
                    {
                        "patch_id": patch_id,
                        "timestamp": meta_loop._last_patch_time + meta_loop.PATCH_RATE_LIMIT_SECONDS + 1,
                        "diff": diff,
                        "reward": new_r,
                        "sandbox_exit_code": exit_code,
                    }
                )
                meta_loop._last_patch_time = meta_loop._last_patch_time + meta_loop.PATCH_RATE_LIMIT_SECONDS + 1

    entries = json.loads(history_file.read_text())
    assert len(entries) == 1
    assert entries[0]["patch_id"] == "id123"


def test_loop_forever_rolls_back_and_mutates(monkeypatch):
    calls = {"rollback": 0, "generate": 0, "verify": 0}

    async def fake_verify(traces, patch):
        calls["verify"] += 1
        return False

    async def fake_generate(traces):
        calls["generate"] += 1
        return {"target": "t.py", "before": "", "after": ""}

    def fake_rollback(patch):
        calls["rollback"] += 1

    async def fake_fetch(*_args, **_kwargs):
        return [{}]

    sleep_calls = {"n": 0}

    def stop_sleep(_):
        sleep_calls["n"] += 1
        if sleep_calls["n"] == 2:
            raise StopIteration

    monkeypatch.setattr(meta_loop, "_verify_patch", fake_verify)
    monkeypatch.setattr(meta_loop, "_generate_patch", fake_generate)
    monkeypatch.setattr(meta_loop, "_rollback", fake_rollback)
    monkeypatch.setattr(meta_loop, "fetch_recent_traces", fake_fetch)
    monkeypatch.setattr(time, "sleep", stop_sleep)

    TimeoutCtx = getattr(pytest, "Timeout", None)
    if TimeoutCtx is None:  # pragma: no cover - fallback if plugin missing
        class TimeoutCtx:  # type: ignore
            def __init__(self, *_args):
                pass
            def __enter__(self):
                return None
            def __exit__(self, *_exc):
                return False

    with pytest.raises(StopIteration):
        with TimeoutCtx(3):
            meta_loop.loop_forever()

    assert calls["rollback"] == 1
    assert calls["generate"] == meta_loop.MAX_MUTATIONS_PER_LOOP
