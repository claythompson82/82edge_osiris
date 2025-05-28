import pytest
import asyncio
import json
from unittest.mock import patch

import fakeredis.aioredis
from llm_sidecar.redis_bus import publish, subscribe

@pytest.mark.asyncio
async def test_redis_publish_subscribe_roundtrip():
    """
    Tests a full publish-subscribe roundtrip using a mocked Redis.
    """
    test_channel = "test_channel_roundtrip"
    test_message = {"key": "value", "num": 123, "nested": {"n_key": "n_val"}}
    received_messages = []

    # Patch 'redis.asyncio.from_url' to use the fake Redis client
    with patch("llm_sidecar.redis_bus.redis.asyncio.from_url", fakeredis.aioredis.FakeRedis.from_url):
        
        async def subscriber_task_func():
            try:
                async for message in subscribe(test_channel):
                    received_messages.append(message)
            except asyncio.CancelledError:
                # This is expected when the task is cancelled
                pass
            except Exception as e:
                pytest.fail(f"Subscriber task failed with: {e}")

        subscriber_task = asyncio.create_task(subscriber_task_func())
        
        # Give the subscriber a moment to connect and subscribe
        await asyncio.sleep(0.01) 

        # Publish the test message
        await publish(test_channel, json.dumps(test_message))
        
        # Wait for the subscriber to receive the message
        # A simple sleep is used here; for more complex scenarios,
        # a condition variable or event might be more robust.
        await asyncio.sleep(0.01) 

        # Assertions
        assert len(received_messages) == 1, "Subscriber should have received exactly one message"
        assert received_messages[0] == test_message, "Received message does not match the original"

        # Clean up the subscriber task
        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass # Expected

@pytest.mark.asyncio
async def test_publish_before_subscribe():
    """
    Tests that a message published before a subscriber connects is still received.
    (fakeredis should queue it).
    """
    test_channel = "test_channel_publish_first"
    test_message = {"event": "system_startup", "timestamp": "2023-01-01T00:00:00Z"}
    received_messages = []

    with patch("llm_sidecar.redis_bus.redis.asyncio.from_url", fakeredis.aioredis.FakeRedis.from_url):
        # Publish first
        await publish(test_channel, json.dumps(test_message))
        await asyncio.sleep(0.01) # Give Redis a moment to process

        async def subscriber_task_func():
            try:
                async for message in subscribe(test_channel):
                    received_messages.append(message)
                    if len(received_messages) >= 1: # Stop after one for this test
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                pytest.fail(f"Subscriber task failed with: {e}")

        subscriber_task = asyncio.create_task(subscriber_task_func())
        await asyncio.sleep(0.01) # Give subscriber time to connect and get message

        assert len(received_messages) == 1
        assert received_messages[0] == test_message

        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_multiple_messages():
    """
    Tests publishing and subscribing to multiple messages on the same channel.
    """
    test_channel = "test_channel_multiple"
    messages_to_send = [
        {"id": 1, "content": "first message"},
        {"id": 2, "content": "second message"},
        {"id": 3, "content": "third message"},
    ]
    received_messages = []

    with patch("llm_sidecar.redis_bus.redis.asyncio.from_url", fakeredis.aioredis.FakeRedis.from_url):
        async def subscriber_task_func():
            try:
                async for message in subscribe(test_channel):
                    received_messages.append(message)
                    if len(received_messages) == len(messages_to_send):
                        break 
            except asyncio.CancelledError:
                pass
            except Exception as e:
                pytest.fail(f"Subscriber task failed with: {e}")
        
        subscriber_task = asyncio.create_task(subscriber_task_func())
        await asyncio.sleep(0.01) # Allow subscriber to connect

        for msg in messages_to_send:
            await publish(test_channel, json.dumps(msg))
            await asyncio.sleep(0.005) # Short delay between publishes

        await asyncio.sleep(0.05) # Allow all messages to be processed

        assert len(received_messages) == len(messages_to_send)
        assert received_messages == messages_to_send

        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass
