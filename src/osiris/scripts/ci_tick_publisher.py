#!/usr/bin/env python3

import redis
import time
import json
from datetime import datetime, timezone
import argparse
import signal
import sys

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_CHANNEL = "market.ticks"

# Global flag to handle script termination
running = True


def signal_handler(sig, frame):
    global running
    print("Signal received, stopping publisher...")
    running = False


def main():
    global running
    parser = argparse.ArgumentParser(description="Publish mock ticks to Redis.")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration in seconds to publish ticks."
    )
    parser.add_argument(
        "--redis-host", type=str, default=REDIS_HOST, help="Redis host."
    )
    parser.add_argument(
        "--redis-port", type=int, default=REDIS_PORT, help="Redis port."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=REDIS_CHANNEL,
        help="Redis channel to publish to.",
    )

    args = parser.parse_args()

    print(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    try:
        r = redis.Redis(
            host=args.redis_host, port=args.redis_port, decode_responses=True
        )
        r.ping()
        print("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_time = time.time()
    end_time = start_time + args.duration
    tick_count = 0

    print(
        f"Starting tick publisher for {args.duration} seconds. Publishing to channel '{args.channel}'."
    )

    try:
        while running and time.time() < end_time:
            timestamp = datetime.now(timezone.utc).isoformat()
            tick_message = {
                "timestamp": timestamp,
                "symbol": "MOCK_TICK_CI",  # Using a distinct symbol for CI ticks
                "close": round(
                    100.0 + (random.random() * 10 - 5), 2
                ),  # Add some minor price variation
            }
            message_json = json.dumps(tick_message)

            try:
                r.publish(args.channel, message_json)
                tick_count += 1
                if tick_count % 10 == 0:  # Log every 10 ticks
                    print(f"Published tick {tick_count}: {message_json}")
            except redis.exceptions.RedisError as e:
                print(f"Error publishing to Redis: {e}")
                # Attempt to reconnect or handle error appropriately
                time.sleep(5)  # Wait before retrying or exiting
                try:
                    r = redis.Redis(
                        host=args.redis_host,
                        port=args.redis_port,
                        decode_responses=True,
                    )
                    r.ping()
                except redis.exceptions.ConnectionError:
                    print("Reconnect failed. Exiting.")
                    break  # Exit loop if reconnect fails

            time.sleep(1)  # Publish every 1 second
    except Exception as e:
        print(f"An unexpected error occurred in the publisher loop: {e}")
    finally:
        print(f"Publisher finished. Published a total of {tick_count} ticks.")
        if r:
            r.close()


if __name__ == "__main__":
    # Need to add random for the price variation
    import random

    main()
