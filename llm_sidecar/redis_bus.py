import os
import json
import redis.asyncio as redis
from typing import AsyncIterator

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def publish(channel: str, msg_json: str):
    """
    Publishes a JSON message to the specified Redis channel.

    Args:
        channel: The name of the Redis channel.
        msg_json: The JSON message string to publish.
    """
    r = await redis.from_url(REDIS_URL)
    await r.publish(channel, msg_json)
    await r.close()

async def subscribe(channel: str) -> AsyncIterator[dict]:
    """
    Subscribes to a Redis channel and yields messages as dictionaries.

    Args:
        channel: The name of the Redis channel to subscribe to.

    Yields:
        dict: Messages received from the channel, decoded from JSON.
    """
    r = await redis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            yield json.loads(data)
    await pubsub.unsubscribe(channel)
    await r.close()
