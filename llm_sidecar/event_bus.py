import asyncio
import redis.asyncio as redis
from redis.asyncio.client import PubSub
from redis.exceptions import RedisError
import logging
from typing import Callable, Awaitable, Union

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self, redis_url: str):
        """
        Initializes the EventBus with a Redis client.

        Args:
            redis_url: The URL for the Redis server.
        """
        # The following line is compatible with both real Redis and fakeredis.
        # For fakeredis, redis_url can be None or any string, it's not actually used
        # if a fakeredis server instance is passed to from_url.
        # However, to make it explicit for testing, we might pass "redis://localhost:6379/0"
        # or similar, which from_url can parse.
        self.redis_client: redis.Redis = redis.from_url(redis_url, decode_responses=True)
        self.pubsub: PubSub = None
        self._stop_event = asyncio.Event()
        self._listener_tasks = []

    async def connect(self):
        """
        Connects to Redis and pings the server to ensure connection.
        """
        try:
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis.")
            self.pubsub = self.redis_client.pubsub()
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def publish(self, event_type: str, payload: str):
        """
        Publishes an event to the specified channel.

        Args:
            event_type: The channel name to publish to.
            payload: The message payload (must be a string).
        """
        if not self.redis_client:
            logger.error("Redis client not initialized. Cannot publish event.")
            return
        try:
            await self.redis_client.publish(event_type, payload)
            logger.debug(f"Published event '{event_type}' with payload: {payload}")
        except RedisError as e:
            logger.error(f"Error publishing event '{event_type}': {e}")

    async def subscribe(self, event_type: str, handler: Callable[[str], Awaitable[None]]):
        """
        Subscribes to an event type and registers a handler.

        Args:
            event_type: The event type (channel) to subscribe to.
            handler: An async callable that will be invoked with the message payload.
        """
        if not self.pubsub:
            logger.error("PubSub not initialized. Cannot subscribe to event.")
            # Or raise an error, depending on desired behavior
            return

        await self.pubsub.subscribe(event_type)
        logger.info(f"Subscribed to event type: {event_type}")

        async def _listener():
            logger.debug(f"Listener task started for event type: {event_type}")
            while not self._stop_event.is_set():
                try:
                    # listen() will block until a message is received or a timeout occurs
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        logger.debug(f"Received message on '{event_type}': {message['data']}")
                        await handler(message["data"])
                    # Allow other tasks to run
                    await asyncio.sleep(0.01)
                except RedisError as e:
                    logger.error(f"Redis error in listener for '{event_type}': {e}")
                    # Potentially try to resubscribe or handle error
                    await asyncio.sleep(5) # Wait before retrying or breaking
                except Exception as e:
                    logger.error(f"Exception in handler for '{event_type}': {e}")
                    # Don't let handler exceptions kill the listener loop
            logger.debug(f"Listener task for event type '{event_type}' stopped.")


        task = asyncio.create_task(_listener())
        self._listener_tasks.append(task)
        logger.debug(f"Listener task for {event_type} added.")


    async def close(self):
        """
        Gracefully disconnects the Redis client and stops listener tasks.
        """
        logger.info("Closing EventBus connection and stopping listeners...")
        self._stop_event.set()

        if self.pubsub:
            try:
                await self.pubsub.unsubscribe() # Unsubscribe from all channels
                await self.pubsub.close() # Close the pubsub connection
                logger.debug("PubSub connection closed.")
            except RedisError as e:
                logger.error(f"Error closing PubSub: {e}")
        
        # Wait for all listener tasks to complete
        if self._listener_tasks:
            logger.debug(f"Waiting for {len(self._listener_tasks)} listener tasks to complete...")
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)
            logger.debug("All listener tasks completed.")
            self._listener_tasks = []

        if self.redis_client:
            try:
                await self.redis_client.close()
                await self.redis_client.connection_pool.disconnect()
                logger.info("Redis client connection closed.")
            except RedisError as e:
                logger.error(f"Error closing Redis client: {e}")

        logger.info("EventBus closed.")

# Example Usage (optional, for testing purposes)
async def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # For testing with a real Redis instance (ensure Redis is running)
    # redis_url = "redis://localhost:6379/0"
    
    # For testing with fakeredis
    # You would typically do this in your test setup:
    # from fakeredis import aioredis
    # fake_redis_server = aioredis.FakeServer()
    # fake_redis_client = aioredis.FakeRedis.from_server(fake_redis_server, decode_responses=True)
    # event_bus = EventBus("redis://dummy") # URL is not strictly needed if client is injected
    # event_bus.redis_client = fake_redis_client # Inject fake client

    # Simplified fakeredis setup for this example (less ideal than injection)
    try:
        # Attempt to use fakeredis if available, otherwise skip this example part
        from fakeredis import aioredis as fake_aioredis
        # The from_url method in fakeredis.aioredis.FakeRedis doesn't always behave like real redis.asyncio
        # It's often better to create a FakeServer and then FakeRedis.from_server(server)
        # However, for simplicity here, we'll try to make it work with from_url
        # Note: fakeredis needs a URL, even if it's a dummy one for some configurations.
        event_bus = EventBus("redis://fakeredis")
        # Manually replace the client if from_url didn't create a FakeRedis instance as expected for some versions
        if not hasattr(event_bus.redis_client, 'is_fake'): # Heuristic to check if it's a fake client
             server = fake_aioredis.FakeServer()
             event_bus.redis_client = fake_aioredis.FakeRedis.from_server(server, decode_responses=True)

    except ImportError:
        logger.warning("fakeredis not installed. Skipping example usage with fakeredis.")
        # Fallback to real Redis if fakeredis is not available and you have a real Redis server
        # For CI/testing, you'd typically ensure fakeredis is installed.
        # redis_url = "redis://localhost:6379/0" # Uncomment if you have Redis running
        # event_bus = EventBus(redis_url) # Remove this line if you don't want to connect to real Redis
        logger.info("Example usage requires either fakeredis or a running Redis instance.")
        return


    async def example_handler_1(payload: str):
        logger.info(f"Handler 1 received: {payload}")

    async def example_handler_2(payload: str):
        logger.info(f"Handler 2 received: {payload}")
        # Simulate some work
        await asyncio.sleep(0.5)
        logger.info(f"Handler 2 finished processing: {payload}")


    try:
        await event_bus.connect()

        await event_bus.subscribe("test_event", example_handler_1)
        await event_bus.subscribe("test_event", example_handler_2)
        await event_bus.subscribe("another_event", example_handler_1)

        await event_bus.publish("test_event", "Hello from EventBus!")
        await event_bus.publish("another_event", "Another message here.")
        await event_bus.publish("test_event", "Second message for test_event")

        # Keep the event bus running for a bit to process messages
        await asyncio.sleep(2) 

    except RedisError as e:
        logger.error(f"Main example encountered a Redis error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main example: {e}")
    finally:
        await event_bus.close()

if __name__ == "__main__":
    asyncio.run(main())
