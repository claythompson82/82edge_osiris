import redis.asyncio as redis
import json
from typing import Generator, Dict

# TODO: Make Redis connection details configurable
REDIS_HOST = "localhost"
REDIS_PORT = 6379

async def publish(channel: str, payload: dict):
    # Publishes a message to the specified Redis channel.
    r = await redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    try:
        await r.publish(channel, json.dumps(payload))
    finally:
        await r.close()

async def subscribe(channel: str) -> Generator[Dict, None, None]:
    # Subscribes to a Redis channel and yields messages as they arrive.
    r = await redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0) # Timeout to allow graceful exit
            if message and message['type'] == 'message':
                yield json.loads(message['data'])
            # Add a small sleep to prevent tight loop if no messages and allow other async tasks
            # await asyncio.sleep(0.01) # Removed as get_message with timeout handles this
    except GeneratorExit: # Handle consumer stopping
        print(f"Subscription to {channel} closed.")
    finally:
        print(f"Unsubscribing from {channel} and closing connection.")
        await pubsub.unsubscribe(channel)
        await r.close()
