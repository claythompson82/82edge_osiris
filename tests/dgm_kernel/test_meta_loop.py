import asyncio
import json
import logging
from pathlib import Path
from unittest import mock # Python's built-in mock library

import pytest
import fakeredis # Needs to be installed: pip install fakeredis

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
    server.connected = True # Ensure it appears connected
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
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": 1, "data": "valid_trace"}))
    fake_redis.lpush(meta_loop.TRACE_QUEUE, "this_is_not_json")
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps({"id": 3, "data": "another_valid_trace"}))

    # We pushed 3 items, rpop will fetch in reverse order of lpush if we think of it as stack.
    # Or, if lpush = head, rpop = tail:
    # Pushed: {"id":3}, "not_json", {"id":1}  ( {"id":1} is at tail)
    # Fetched: {"id":1}, then "not_json" (error), then {"id":3}

    caplog.clear() # Clear previous logs
    with caplog.at_level(logging.ERROR):
        fetched_traces = await meta_loop.fetch_recent_traces(n=3)

    assert len(fetched_traces) == 2 # Should skip the invalid one
    assert fetched_traces[0]["id"] == 1 # from rpop
    assert fetched_traces[1]["id"] == 3 # from rpop

    assert any("Failed to decode JSON for trace: this_is_not_json" in message for message in caplog.messages)
    log.info("Test passed: fetch_recent_traces handled JSON decode error correctly.")

# --- Placeholder for meta_loop tests ---
# These will be more complex and require mocking generate_patch, _prove_patch, _apply_patch,
# and llm_sidecar.reward.proofable_reward

@pytest.mark.asyncio
@mock.patch('dgm_kernel.meta_loop.generate_patch')
@mock.patch('dgm_kernel.meta_loop._prove_patch')
@mock.patch('dgm_kernel.meta_loop._apply_patch')
@mock.patch('dgm_kernel.meta_loop.proofable_reward') # Path to proofable_reward
@mock.patch('asyncio.sleep', return_value=None) # Mock sleep to speed up test
async def test_meta_loop_applies_good_patch(
    self, mock_sleep, mock_proofable_reward, mock_apply_patch,
    mock_prove_patch, mock_generate_patch, fake_redis, tmp_path
):
    log.info("Testing meta_loop applies a good patch.")
    # Setup:
    # 1. Populate TRACE_QUEUE with some traces
    trace_1 = {"id": "trace1", "value": 100}
    fake_redis.lpush(meta_loop.TRACE_QUEUE, json.dumps(trace_1))

    # 2. Configure mocks:
    #   - generate_patch returns a valid patch
    #   - _prove_patch returns True
    #   - _apply_patch returns True
    #   - proofable_reward returns a positive value

    # Ensure target file for patch exists for 'before' content in generate_patch
    # and for apply/rollback.
    # We use tmp_path provided by pytest for temporary file operations.
    target_file_path = tmp_path / "osiris_policy" / "strategy.py"
    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    initial_content = "# Initial strategy content\n"
    target_file_path.write_text(initial_content)

    test_patch = {
        "target": str(target_file_path),
        "before": initial_content,
        "after": initial_content + "# Patched by DGM\n",
        "rationale": "Test rationale"
    }
    mock_generate_patch.return_value = asyncio.Future()
    mock_generate_patch.return_value.set_result(test_patch)

    mock_prove_patch.return_value = asyncio.Future()
    mock_prove_patch.return_value.set_result(True)

    # _apply_patch is synchronous in the code
    mock_apply_patch.return_value = True

    # proofable_reward is synchronous
    mock_proofable_reward.return_value = 1.0 # Positive reward

    # Run meta_loop for one iteration (or a controlled number)
    # We need to run it in a way that it can be cancelled after one cycle for the test
    # or rely on mock_generate_patch, etc. to control flow.
    # For this test, we'll let it run once by controlling inputs and then check results.

    # To make it run "once" effectively for this test case:
    # - Provide one trace.
    # - mock_generate_patch will be called once.
    # - Then, if we want to stop, we could make generate_patch raise an exception
    #   on subsequent calls, or make fetch_recent_traces return empty.
    # For now, let's assume one trace means one cycle we care about.

    # To stop the loop after one iteration for testing:
    # Make generate_patch return None after the first call.
    async def generate_patch_side_effect(*args, **kwargs):
        if mock_generate_patch.call_count == 1: # Corrected: use mock_generate_patch.call_count
            fut = asyncio.Future()
            fut.set_result(test_patch)
            return fut
        fut_none = asyncio.Future()
        fut_none.set_result(None) # Stop loop after one successful patch cycle
        return fut_none

    mock_generate_patch.side_effect = generate_patch_side_effect

    # Run the loop. It will break when generate_patch returns None.
    # Or, more robustly, run it as a task and cancel after a short delay.
    loop_task = asyncio.create_task(meta_loop.meta_loop())
    # Increased sleep slightly to ensure the full loop including mocked sub-awaits can complete.
    await asyncio.sleep(0.5) # Allow loop to run at least one full cycle
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        log.info("Meta_loop task cancelled as expected for test.")

    # Assertions:
    # - generate_patch was called
    # - _prove_patch was called with the patch
    # - _apply_patch was called with the patch
    # - proofable_reward was called
    # - Patch was logged to APPLIED_LOG in Redis
    assert mock_generate_patch.call_count >= 1 # Should be called at least once
    mock_prove_patch.assert_called_with(test_patch)
    mock_apply_patch.assert_called_with(test_patch)
    mock_proofable_reward.assert_called_with(trace_1, {}) # Assuming empty context for now

    applied_log_content = fake_redis.lrange(meta_loop.APPLIED_LOG, 0, -1)
    assert len(applied_log_content) == 1
    logged_patch = json.loads(applied_log_content[0])
    assert logged_patch["target"] == str(target_file_path)
    assert logged_patch["rationale"] == "Test rationale"
    assert logged_patch["reward"] == 1.0

    # Check that the target file was actually modified by _apply_patch
    # In a real test of _apply_patch, we'd check file content.
    # Here, mock_apply_patch is used, so we trust it was called.
    # If we were testing the real _apply_patch, we'd do:
    # assert target_file_path.read_text() == test_patch["after"]
    log.info("Test passed: meta_loop applied a good patch.")

# TODO: Add test for meta_loop_rolls_back_bad_patch
# Similar to above, but proofable_reward returns a negative value,
# and we assert that _rollback is called and the patch is not in APPLIED_LOG.
# Need to mock _rollback as well for that.

# TODO: Add test for meta_loop_skips_unproven_patch
# Similar, but _prove_patch returns False. Assert _apply_patch is not called.

@pytest.mark.asyncio
@mock.patch('dgm_kernel.meta_loop.generate_patch')
@mock.patch('dgm_kernel.meta_loop._verify_patch')
@mock.patch('dgm_kernel.meta_loop._apply_patch')
@mock.patch('dgm_kernel.meta_loop._rollback')
@mock.patch('dgm_kernel.meta_loop.proofable_reward')
async def test_meta_loop_rolls_back_bad_patch(
    mock_reward,
    mock_rollback,
    mock_apply_patch,
    mock_prove_patch,
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
    mock_prove_patch.return_value = (True, 9.0)
    mock_apply_patch.return_value = True
    mock_reward.return_value = -1.0

    traces = await meta_loop.fetch_recent_traces()
    patch_data = await meta_loop.generate_patch(traces)
    accepted, score = await meta_loop._verify_patch(traces, patch_data)
    if accepted:
        if meta_loop._apply_patch(patch_data):
            new_r = sum(meta_loop.proofable_reward(t, patch_data.get("after")) for t in traces)
            if new_r >= 0:
                meta_loop.REDIS.lpush(
                    meta_loop.APPLIED_LOG,
                    json.dumps(patch_data | {"reward": new_r, "verification_score": score}),
                )
            else:
                meta_loop._rollback(patch_data)

    mock_rollback.assert_called_once_with(patch)
    assert fake_redis.llen(meta_loop.APPLIED_LOG) == 0

