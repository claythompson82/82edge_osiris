import pytest
import asyncio
import json
from fakeredis import aioredis as fakeredis_async
from llm_sidecar import event_bus

@pytest.fixture
async def mock_shared_redis_client(monkeypatch):
    # This fixture provides a single, shared FakeRedis instance for the duration of a test.
    # All calls to redis.Redis() within the event_bus module will use this instance.
    fake_client = fakeredis_async.FakeRedis() 
    
    async def mock_get_shared_client(*args, **kwargs):
        return fake_client # Always return the same instance
    
    monkeypatch.setattr("llm_sidecar.event_bus.redis.Redis", mock_get_shared_client)
    # Patching REDIS_HOST and REDIS_PORT to ensure they are not pointing to a real Redis during tests.
    # This is a safeguard in case these constants are used directly and not passed to Redis()
    monkeypatch.setattr(event_bus, "REDIS_HOST", "mock_redis_host")
    monkeypatch.setattr(event_bus, "REDIS_PORT", 0) # Port 0 is often invalid, good for mock
    
    yield fake_client # Provide the client to the test if needed for direct manipulation
    
    # Teardown: fakeredis clients are typically in-memory and don't need explicit close for FakeRedis,
    # but if it were a real pooled connection from fakeredis.create_redis_pool, it would be closed here.
    # For FakeRedis, this is mostly for completeness or if specific cleanup methods exist.
    # await fake_client.close() # Not usually needed for FakeRedis itself

@pytest.mark.asyncio
async def test_event_bus_publish_subscribe(mock_shared_redis_client):
    channel = "test_single_message_channel"
    payload = {"message": "event_bus_test_1"}
    queue = asyncio.Queue()

    async def listener():
        async for msg in event_bus.subscribe(channel):
            await queue.put(msg)
            break # We only expect one message for this test

    listener_task = asyncio.create_task(listener())
    
    # Allow a very brief moment for the subscriber to initialize
    await asyncio.sleep(0.01) 

    await event_bus.publish(channel, payload)

    try:
        received_message = await asyncio.wait_for(queue.get(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Timeout waiting for message on channel {channel}")
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass # Expected upon cancellation

    assert received_message == payload

@pytest.mark.asyncio
async def test_event_bus_subscribe_multiple_messages(mock_shared_redis_client):
    channel = "test_multiple_messages_channel"
    payloads = [
        {"message": "multi_test_1"},
        {"message": "multi_test_2"},
        {"message": "multi_test_3"}
    ]
    received_messages_list = []
    
    async def listener():
        async for msg in event_bus.subscribe(channel):
            received_messages_list.append(msg)
            if len(received_messages_list) == len(payloads):
                break

    listener_task = asyncio.create_task(listener())

    await asyncio.sleep(0.01) # Brief pause for subscriber setup

    for p in payloads:
        await event_bus.publish(channel, p)
        await asyncio.sleep(0.01) # Small delay between publishes if needed

    try:
        # Wait until the listener has collected all messages
        # The timeout should be generous enough for all messages to be processed
        await asyncio.wait_for(listener_task, timeout=len(payloads) * 0.5 + 1) 
    except asyncio.TimeoutError:
        pytest.fail(f"Timeout waiting for multiple messages on channel {channel}. Received: {len(received_messages_list)}/{len(payloads)}")
    finally:
        if not listener_task.done(): # Ensure cancellation if not already done
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
    
    assert len(received_messages_list) == len(payloads)
    for p in payloads:
        assert p in received_messages_list
