import asyncio
import unittest
import json
import logging
from unittest.mock import patch, AsyncMock

from llm_sidecar.event_bus import EventBus, RedisError

# Ensure fakeredis is installed for testing (e.g., in requirements-dev.txt)
# pip install fakeredis[pyaio]
try:
    from fakeredis import aioredis as fake_aioredis
    from fakeredis.aioredis import FakeServer
except ImportError:
    # This is a fallback for environments where fakeredis might not be installed,
    # though tests will likely fail or be skipped.
    # For CI, ensure fakeredis[pyaio] is in test dependencies.
    fake_aioredis = None
    FakeServer = None
    logging.warning(
        "fakeredis.aioredis not found. EventBus tests may be skipped or fail."
    )


# Suppress most logging output during tests unless specifically testing for it
logging.basicConfig(level=logging.CRITICAL)


@unittest.skipIf(
    fake_aioredis is None,
    "fakeredis.aioredis is not installed, skipping EventBus tests",
)
class TestEventBus(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.fake_server = FakeServer()
        # Using from_server ensures we are using a server instance we control
        self.fake_redis_client = fake_aioredis.FakeRedis.from_server(
            self.fake_server, decode_responses=True
        )

        self.event_bus = EventBus("redis://dummy_url_for_fakeredis")
        # Replace the client created by from_url with our fake client
        # This is a common pattern when precise control over the client instance is needed
        if self.event_bus.redis_client:
            await self.event_bus.redis_client.close()  # Close client created by EventBus's __init__
        self.event_bus.redis_client = self.fake_redis_client

        # Now connect, which will use the fake_redis_client for ping and pubsub
        try:
            await self.event_bus.connect()
        except RedisError as e:
            self.fail(f"EventBus connect failed during setup with fakeredis: {e}")

        self.received_messages = {}  # To store messages for each handler_id
        self.handler_events = {}  # To store asyncio.Event for each handler_id

    async def _create_handler(self, handler_id: str):
        self.handler_events[handler_id] = asyncio.Event()
        self.received_messages[handler_id] = []

        async def _handler(payload: str):
            self.received_messages[handler_id].append(payload)
            self.handler_events[handler_id].set()
            # Reset event if you expect multiple messages for this handler in a single test
            # self.handler_events[handler_id].clear()

        return _handler

    async def asyncTearDown(self):
        if hasattr(self, "event_bus") and self.event_bus:
            await self.event_bus.close()
        # fakeredis FakeRedis client does not have an explicit close method that is separate from connection_pool.disconnect
        if hasattr(self, "fake_redis_client") and self.fake_redis_client:
            await self.fake_redis_client.connection_pool.disconnect()
        # FakeServer doesn't have an explicit close, it's in-memory
        self.fake_server = None

    async def test_publish_subscribe_single_handler(self):
        handler_id = "handler1"
        event_type = "test_event_single"
        payload = {"message": "hello single"}

        handler = await self._create_handler(handler_id)
        await self.event_bus.subscribe(event_type, handler)

        await self.event_bus.publish(event_type, json.dumps(payload))

        try:
            await asyncio.wait_for(self.handler_events[handler_id].wait(), timeout=1.0)
        except asyncio.TimeoutError:
            self.fail(
                f"Handler {handler_id} did not receive message for {event_type} in time."
            )

        self.assertEqual(len(self.received_messages[handler_id]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id][0]), payload)

    async def test_publish_subscribe_multiple_handlers_same_event(self):
        handler_id1 = "handler_multi1"
        handler_id2 = "handler_multi2"
        event_type = "test_event_multi"
        payload = {"message": "hello multi"}

        handler1 = await self._create_handler(handler_id1)
        handler2 = await self._create_handler(handler_id2)

        await self.event_bus.subscribe(event_type, handler1)
        await self.event_bus.subscribe(event_type, handler2)

        await self.event_bus.publish(event_type, json.dumps(payload))

        try:
            await asyncio.wait_for(self.handler_events[handler_id1].wait(), timeout=1.0)
            await asyncio.wait_for(self.handler_events[handler_id2].wait(), timeout=1.0)
        except asyncio.TimeoutError:
            self.fail(
                f"One or more handlers did not receive message for {event_type} in time."
            )

        self.assertEqual(len(self.received_messages[handler_id1]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id1][0]), payload)
        self.assertEqual(len(self.received_messages[handler_id2]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id2][0]), payload)

    async def test_multiple_event_types_and_isolation(self):
        handler_id_A = "handlerA"
        handler_id_B = "handlerB"
        event_type_A = "event_A"
        payload_A = {"message": "payload A"}
        event_type_B = "event_B"
        payload_B = {"message": "payload B"}

        handler_A = await self._create_handler(handler_id_A)
        handler_B = await self._create_handler(handler_id_B)

        await self.event_bus.subscribe(event_type_A, handler_A)
        await self.event_bus.subscribe(event_type_B, handler_B)

        await self.event_bus.publish(event_type_A, json.dumps(payload_A))
        await self.event_bus.publish(event_type_B, json.dumps(payload_B))

        try:
            await asyncio.wait_for(
                self.handler_events[handler_id_A].wait(), timeout=1.0
            )
            await asyncio.wait_for(
                self.handler_events[handler_id_B].wait(), timeout=1.0
            )
        except asyncio.TimeoutError:
            self.fail(
                "Handlers for event_A or event_B did not receive messages in time."
            )

        self.assertEqual(len(self.received_messages[handler_id_A]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id_A][0]), payload_A)

        self.assertEqual(len(self.received_messages[handler_id_B]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id_B][0]), payload_B)

        # Also check that handlerA did not get payloadB and vice-versa
        # This is implicitly tested by the length check above, but can be made explicit
        # if handlers were designed to receive multiple messages.
        # For this test, it means only one message should be in each list.

    async def test_handler_for_one_event_not_receive_other_events(self):
        handler_id_X = "handlerX"  # Subscribed to event_X
        handler_id_Y = "handlerY"  # Subscribed to event_Y
        event_type_X = "event_X"
        event_type_Y = "event_Y"
        payload_Y = {"message": "payload Y for Y only"}

        handler_X = await self._create_handler(handler_id_X)
        handler_Y = await self._create_handler(handler_id_Y)

        await self.event_bus.subscribe(event_type_X, handler_X)
        await self.event_bus.subscribe(event_type_Y, handler_Y)

        # Publish only to event_Y
        await self.event_bus.publish(event_type_Y, json.dumps(payload_Y))

        try:
            await asyncio.wait_for(
                self.handler_events[handler_id_Y].wait(), timeout=1.0
            )
        except asyncio.TimeoutError:
            self.fail(
                f"Handler {handler_id_Y} for {event_type_Y} did not receive message in time."
            )

        self.assertEqual(len(self.received_messages[handler_id_Y]), 1)
        self.assertEqual(json.loads(self.received_messages[handler_id_Y][0]), payload_Y)

        # Check that handler_X received nothing
        self.assertEqual(
            len(self.received_messages[handler_id_X]),
            0,
            f"Handler {handler_id_X} for {event_type_X} should not have received messages for {event_type_Y}.",
        )
        self.assertFalse(
            self.handler_events[handler_id_X].is_set(),
            f"Event for {handler_id_X} should not be set as no message for {event_type_X} was published.",
        )

    async def test_graceful_shutdown(self):
        handler_id = "shutdown_handler"
        event_type = "shutdown_event"

        handler = await self._create_handler(handler_id)
        await self.event_bus.subscribe(event_type, handler)

        self.assertTrue(
            len(self.event_bus._listener_tasks) > 0,
            "Listener task should have been created.",
        )

        await self.event_bus.close()

        self.assertTrue(
            self.event_bus._stop_event.is_set(), "Stop event should be set after close."
        )

        # Listener tasks should be empty because they are awaited in close() and removed from the list.
        # If close() is implemented correctly, it should join/cancel tasks.
        # The tasks list is cleared in EventBus.close()
        self.assertEqual(
            len(self.event_bus._listener_tasks),
            0,
            "Listener tasks list should be empty after close and cleanup.",
        )

        # Verify that tasks are actually completed (not just removed from list)
        # This requires tasks to be awaited in EventBus.close()
        # If tasks were stored, we could check task.done()
        # For now, checking the list is empty is the primary check based on current EventBus.close() impl.

        # Test that publishing after close doesn't work or is handled
        # The current EventBus.publish logs an error if redis_client is None.
        # After close(), redis_client is not None but its connection pool is disconnected.
        # Publishing should ideally fail or be handled by RedisError.
        with self.assertLogs(
            logger="llm_sidecar.event_bus", level="ERROR"
        ) as log_capture:
            await self.event_bus.publish(event_type, "message_after_close")
            # Check that an error related to publishing on a closed connection was logged
            self.assertTrue(
                any(
                    "Error publishing event" in record.getMessage()
                    for record in log_capture.records
                ),
                "Should log an error when publishing after close.",
            )

        # Handler should not have received the message sent after close
        self.assertEqual(
            len(self.received_messages[handler_id]),
            0,
            "Handler should not receive messages published after EventBus is closed.",
        )

    @patch("llm_sidecar.event_bus.redis.Redis.ping", new_callable=AsyncMock)
    async def test_connection_error_on_connect(self, mock_ping):
        mock_ping.side_effect = RedisError("Simulated Redis connection error")

        new_event_bus = EventBus("redis://dummy")
        # Replace client to ensure our patched ping is on the instance connect() will use
        new_event_bus.redis_client = fake_aioredis.FakeRedis.from_server(
            self.fake_server, decode_responses=True
        )
        # Now, specifically patch ping on this instance for the test
        new_event_bus.redis_client.ping = AsyncMock(
            side_effect=RedisError("Simulated Redis connection error")
        )

        with self.assertRaises(RedisError):
            await new_event_bus.connect()

        # Ensure pubsub is not initialized if connect fails
        self.assertIsNone(new_event_bus.pubsub)
        await new_event_bus.close()  # Clean up

    async def test_publish_error_handling(self):
        # Simulate publish error by patching the redis_client's publish method
        original_publish = self.event_bus.redis_client.publish
        self.event_bus.redis_client.publish = AsyncMock(
            side_effect=RedisError("Simulated publish error")
        )

        with self.assertLogs(
            logger="llm_sidecar.event_bus", level="ERROR"
        ) as log_capture:
            await self.event_bus.publish("error_test_event", "test_payload")
            self.assertTrue(
                any(
                    "Error publishing event" in record.getMessage()
                    and "Simulated publish error" in record.getMessage()
                    for record in log_capture.records
                ),
                "Should log an error when redis_client.publish fails.",
            )

        # Restore original method
        self.event_bus.redis_client.publish = original_publish

    async def test_subscribe_after_close_fails_or_logs(self):
        await self.event_bus.close()  # Close the main event bus

        handler = await self._create_handler("handler_after_close")
        # Attempting to subscribe after pubsub is closed (or None) should be handled
        # Current implementation: logs error if self.pubsub is None.
        # In close(), self.pubsub is closed then set to None by some Redis clients, or just closed.
        # EventBus.close() calls self.pubsub.close()
        # Let's check for the log
        with self.assertLogs(
            logger="llm_sidecar.event_bus", level="ERROR"
        ) as log_capture:
            await self.event_bus.subscribe("event_after_close", handler)
            # This check depends on whether pubsub is None or just closed.
            # If pubsub is None: "PubSub not initialized."
            # If pubsub is closed, trying to use it might raise RedisError or similar.
            # The current EventBus.subscribe checks `if not self.pubsub:`,
            # and `self.pubsub` is not set to None in `close()`.
            # Instead, `self.pubsub.subscribe()` on a closed pubsub object would raise an error.
            # The listener task would then fail.
            # Let's refine this test based on actual EventBus behavior.
            # The current EventBus.subscribe will attempt to use the closed pubsub object.
            # This will likely raise an error when `self.pubsub.subscribe()` is called.
            # The test should reflect that the listener task might not start or might exit quickly.
            # For now, let's assume the current check `if not self.pubsub` is the main guard.
            # The EventBus.close() does: `await self.pubsub.unsubscribe(); await self.pubsub.close()`.
            # It does not set `self.pubsub = None`.
            # So, `subscribe()` will proceed, but the `_listener` task's `self.pubsub.get_message()`
            # will likely raise an error immediately or return None indefinitely.

            # Given the current code, subscribe() will still create a listener task.
            # The error will occur inside the listener task when it tries to get a message.
            # This test is more about the robustness of the listener loop post-close.
            # The subscribe method itself might not log directly if pubsub object still exists.
            # A more direct test:
            self.assertTrue(
                self.event_bus._stop_event.is_set(), "Stop event should remain set."
            )
            # If subscribe is called after close, new tasks should ideally not be created,
            # or if they are, they should terminate quickly.
            # The current EventBus.subscribe doesn't prevent new task creation after close.
            # This might be an area for improvement in EventBus.
            # For now, this test asserts that an error is logged if we try to subscribe
            # assuming the pubsub object itself would throw an error upon usage.
            # This is a bit indirect. A direct check on `subscribe` might be better
            # if `subscribe` itself had pre-conditions related to `_stop_event`.

            # Let's focus on the fact that new subscriptions on a closed bus are problematic.
            # The listener task will start, but likely fail or do nothing.
            # No message will be processed.
            await self.event_bus.publish("event_after_close", "test")
            await asyncio.sleep(0.1)  # give time for potential processing
            self.assertEqual(len(self.received_messages["handler_after_close"]), 0)
            # The error might be logged inside the listener, not `subscribe` itself.
            # This test is becoming more of an integration test of listener resiliency.

            # A simpler check for "subscribe on closed bus":
            # If pubsub is truly unusable, .subscribe() on it should fail.
            # fakeredis's closed pubsub might behave differently than real Redis here.
            # Let's assume an error is logged or an exception is raised by the pubsub object itself.
            # The provided log check is a good starting point.
            # The current `subscribe` doesn't have a specific log for this, but listener will.
            # This test is a bit weak for `subscribe` method itself post-close.
            # The `PubSub not initialized` log is if `self.pubsub` was None from the start.
            # The critical aspect is that new subscriptions shouldn't successfully process messages.
            self.assertTrue(
                any(
                    "PubSub not initialized" in r.getMessage()
                    or "Error in listener" in r.getMessage()
                    for r in log_capture.records
                ),
                "Expected log indicating subscribe failed or listener error on closed EventBus.",
            )


if __name__ == "__main__":
    unittest.main()
