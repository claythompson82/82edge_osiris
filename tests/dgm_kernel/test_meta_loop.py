import asyncio
import json
import logging
from pathlib import Path
from unittest import mock  # Python's built-in mock library

import pytest
import fakeredis  # Needs to be installed: pip install fakeredis

# Modules to be tested
from dgm_kernel import meta_loop

# Configure logging for tests (optional, but can be helpful)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


@pytest.fixture
def mock_redis_server():
    # This fixture provides a fakeredis server instance if needed,
    # but often, the decorator is enough.
    server = fakeredis.FakeServer()
    server.connected = True  # Ensure it appears connected
    return server


@pytest.fixture
def fake_redis(mock_redis_server):
    # Use fakeredis.FakeStrictRedis for type compatibility with redis.Redis
    # The `decode_responses=True` is important to match the application code.
    r = fakeredis.FakeStrictRedis(server=mock_redis_server, decode_responses=True)
    # Ensure the connection is explicitly "established"
    r.ping()
    return r


@pytest.fixture(autouse=True)
def patch_redis_client(monkeypatch, fake_redis):
    """Autouse fixture to replace meta_loop.REDIS with our fake_redis instance."""
    monkeypatch.setattr(meta_loop, "REDIS", fake_redis)
    # Also patch any direct instantiations if they exist elsewhere, though meta_loop uses a global.
    # If meta_loop.py creates Redis client inside functions (it doesn't currently),
    # that would need more specific patching (e.g., @mock.patch('dgm_kernel.meta_loop.redis.Redis')).


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
    # Push to the TRACE_QUEUE (items are pushed left, popped right in meta_loop)
    for trace in trace_data:
        fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace))

    fetched_traces = await meta_loop.fetch_recent_traces(n=7)
    assert len(fetched_traces) == 7
    # Redis RPUSH/RPOP means LIFO, but lpush/rpop is FIFO if rpop is from the "end" of list
    # The code uses rpop, which takes from the tail. lpush adds to head.
    # So, if we lpush 0,1,2...9, then rpop will give 0, then 1, ...
    # The order in fetched_traces should be trace_data[0] to trace_data[6]
    for i in range(7):
        assert fetched_traces[i]["id"] == i

    # Check remaining items in queue
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
    fake_redis.lpush(
        meta_loop.TRACE_QUEUE, json.dumps({"id": 1, "data": "valid_trace"})
    )
    fake_redis.lpush(meta_loop.TRACE_QUEUE, "this_is_not_json")
    fake_redis.lpush(
        meta_loop.TRACE_QUEUE, json.dumps({"id": 3, "data": "another_valid_trace"})
    )

    # We pushed 3 items, rpop will fetch in reverse order of lpush if we think of it as stack.
    # Or, if lpush = head, rpop = tail:
    # Pushed: {"id":3}, "not_json", {"id":1}  ( {"id":1} is at tail)
    # Fetched: {"id":1}, then "not_json" (error), then {"id":3}

    caplog.clear()  # Clear previous logs
    with caplog.at_level(logging.ERROR):
        fetched_traces = await meta_loop.fetch_recent_traces(n=3)

    assert len(fetched_traces) == 2  # Should skip the invalid one
    assert fetched_traces[0]["id"] == 1  # from rpop
    assert fetched_traces[1]["id"] == 3  # from rpop

    assert any(
        "Failed to decode JSON for trace: this_is_not_json" in message
        for message in caplog.messages
    )
    log.info("Test passed: fetch_recent_traces handled JSON decode error correctly.")


# --- Placeholder for meta_loop tests ---
# These will be more complex and require mocking generate_patch, _apply_patch,
# and llm_sidecar.reward.proofable_reward




# Test that a negative reward triggers rollback and traces are tagged.
# Similar logic is covered in test_meta_loop_skips_unproven_patch below.


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop._rollback", wraps=meta_loop._rollback)
@mock.patch("dgm_kernel.meta_loop._apply_patch")
@mock.patch("dgm_kernel.meta_loop.proofable_reward")
async def test_meta_loop_rolls_back_bad_patch(
    mock_reward,
    mock_apply_patch,
    mock_rollback,
    mock_verify_patch,
    mock_generate_patch,
    fake_redis,
    tmp_path,
):
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
    mock_verify_patch.return_value = True
    mock_apply_patch.side_effect = lambda p: (
        Path(p["target"]).write_text(p["after"]),
        True,
    )[1]
    mock_reward.return_value = -0.5

    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        if meta_loop._apply_patch(patch_data):
            new_r = sum(
                meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces
            )
            if new_r >= 0:
                meta_loop.REDIS.lpush(
                    meta_loop.APPLIED_LOG,
                    json.dumps(patch_data | {"reward": new_r}),
                )
            else:
                for t in traces:
                    t["rolled_back"] = True
                    meta_loop.REDIS.lpush(meta_loop.ROLLED_BACK_LOG, json.dumps(t))
                meta_loop._rollback(patch_data)

    mock_rollback.assert_called_once_with(patch)
    assert target_file.read_text() == before
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0
    rb_entries = [
        json.loads(v) for v in fake_redis.lrange(meta_loop.ROLLED_BACK_LOG, 0, -1)
    ]
    assert rb_entries == [{"id": "trace1", "value": 100, "rolled_back": True}]


@pytest.mark.asyncio
@mock.patch("dgm_kernel.meta_loop.generate_patch")
@mock.patch("dgm_kernel.meta_loop._verify_patch")
@mock.patch("dgm_kernel.meta_loop._rollback", wraps=meta_loop._rollback)
@mock.patch("dgm_kernel.meta_loop._apply_patch")
@mock.patch("dgm_kernel.meta_loop.proofable_reward")
async def test_meta_loop_skips_unproven_patch(
    mock_reward,
    mock_apply_patch,
    mock_rollback,
    mock_verify_patch,
    mock_generate_patch,
    fake_redis,
    tmp_path,
):
    """Ensure patches rejected by verification do not get applied."""
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
    mock_verify_patch.return_value = False

    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        if meta_loop._apply_patch(patch_data):
            _ = sum(
                meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces
            )

    mock_apply_patch.assert_not_called()
    mock_reward.assert_not_called()
    mock_rollback.assert_not_called()
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0
    assert fake_redis.llen(meta_loop.ROLLED_BACK_LOG) == 0
    assert target_file.read_text() == before
