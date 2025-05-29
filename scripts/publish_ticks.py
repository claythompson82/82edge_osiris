# scripts/publish_ticks.py
import redis
import json
import time
import datetime
import random
import argparse
import logging

def publish_ticks(redis_url: str, channel: str, duration_s: int, interval_s: float):
    logging.info(f"Connecting to Redis at {redis_url} to publish on '{channel}'")
    r = redis.Redis.from_url(redis_url)

    start_time = time.time()
    try:
        while (time.time() - start_time) < duration_s:
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            price = round(100 + random.uniform(-5, 5), 2)
            tick_data = {
                "timestamp": timestamp,
                "symbol": "SIM_EURUSD", # Example symbol
                "open": round(price - random.uniform(0, 1), 2),
                "high": round(price + random.uniform(0, 1), 2),
                "low": round(price - random.uniform(0, 1), 2),
                "close": price,
                "volume": random.randint(100, 1000)
            }
            message = json.dumps(tick_data)
            r.publish(channel, message)
            logging.info(f"Published to {channel}: {message}")
            time.sleep(interval_s)
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}. Ensure Redis is running and accessible at {redis_url}.")
    except Exception as e:
        logging.error(f"Error publishing ticks: {e}", exc_info=True)
    finally:
        logging.info("Finished publishing ticks.")
        if 'r' in locals() and r:
            try:
                r.close()
            except Exception as e_close:
                logging.error(f"Error closing Redis connection: {e_close}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - PUBLISH_TICKS - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Simulate and publish market ticks to Redis.")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis URL.")
    parser.add_argument("--channel", type=str, default="market.ticks", help="Redis channel to publish to.")
    parser.add_argument("--duration", type=int, default=30, help="Duration to publish ticks for (seconds).")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between ticks (seconds).")
    
    args = parser.parse_args()
    
    publish_ticks(args.redis_url, args.channel, args.duration, args.interval)
