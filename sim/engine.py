import csv
import json
import time
import asyncio
import argparse
from datetime import datetime
from llm_sidecar.event_bus import EventBus, RedisError


async def publish_market_data(
    csv_filepath: str,
    redis_url: str,
    channel_name: str,
    delay_seconds: float = 1.0,
    from_date_str: str = None,
):
    """
    Reads OHLCV data from a CSV file and publishes each bar as a JSON dictionary
    to the specified Redis channel.
    """
    event_bus = EventBus(redis_url=redis_url)

    try:
        await event_bus.connect()
        print(f"Connected to Redis at {redis_url}")
    except RedisError as e:
        print(f"Error connecting to Redis: {e}")
        return

    print(
        f"Starting to publish market data from {csv_filepath} to channel '{channel_name}'"
    )

    try:
        with open(csv_filepath, mode="r", newline="") as csvfile:
            # Assuming the CSV has a header row
            # Example header: Timestamp,Open,High,Low,Close,Volume
            # Or: Date,Open,High,Low,Close,Volume,Adj Close (typical Yahoo Finance)
            reader = csv.DictReader(csvfile)

            required_headers = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]  # Timestamp/Date is also key
            # Try to identify common date/timestamp headers
            timestamp_header = None
            if "Timestamp" in reader.fieldnames:  # Typically includes time
                timestamp_header = "Timestamp"
            elif "Date" in reader.fieldnames:  # Typically date only
                timestamp_header = "Date"
            # Add other common variants if necessary
            elif "datetime" in reader.fieldnames:
                timestamp_header = "datetime"
            elif (
                "time" in reader.fieldnames
            ):  # Less common for OHLCV structure but possible
                timestamp_header = "time"
            else:
                print(
                    f"Error: CSV must contain a standard timestamp header (e.g., Timestamp, Date, datetime). Found: {reader.fieldnames}"
                )
                return

            missing_headers = [
                h for h in required_headers if h not in reader.fieldnames
            ]
            if missing_headers:
                print(
                    f"Error: CSV is missing required headers: {', '.join(missing_headers)}"
                )
                return

            start_date_obj = None
            if from_date_str:
                try:
                    start_date_obj = datetime.strptime(from_date_str, "%Y-%m-%d").date()
                    print(f"Filtering data from date: {start_date_obj}")
                except ValueError:
                    print(
                        f"Error: Invalid --from_date format. Please use YYYY-MM-DD. Got: {from_date_str}"
                    )
                    return

            processed_rows = 0
            for row_num, row in enumerate(reader):
                try:
                    current_row_ts_str = row[timestamp_header]

                    # Date filtering
                    if start_date_obj:
                        try:
                            # Attempt to parse the row's timestamp. This can be complex due to various formats.
                            # Common formats: YYYY-MM-DD HH:MM:SS, YYYY-MM-DD, MM/DD/YYYY HH:MM, etc.
                            # For simplicity, we'll try a few common ones. Robust parsing might need dateutil.parser.
                            row_date = None
                            if " " in current_row_ts_str:  # Likely datetime
                                row_date = datetime.strptime(
                                    current_row_ts_str.split(" ")[0], "%Y-%m-%d"
                                ).date()
                            elif (
                                "-" in current_row_ts_str
                                and len(current_row_ts_str) == 10
                            ):  # YYYY-MM-DD
                                row_date = datetime.strptime(
                                    current_row_ts_str, "%Y-%m-%d"
                                ).date()
                            elif (
                                "/" in current_row_ts_str
                                and len(current_row_ts_str.split("/")[2]) == 4
                            ):  # MM/DD/YYYY
                                parts = current_row_ts_str.split("/")
                                row_date = datetime(
                                    int(parts[2]), int(parts[0]), int(parts[1])
                                ).date()

                            if row_date and row_date < start_date_obj:
                                continue  # Skip this row
                        except ValueError as dve:
                            print(
                                f"Skipping row {row_num + 1} due to date parsing error for filtering: {dve} - Timestamp: '{current_row_ts_str}'"
                            )
                            continue

                    bar_data = {
                        "timestamp": current_row_ts_str,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                    }
                    if "Adj Close" in row:  # Optional: Add Adj Close if present
                        bar_data["adj_close"] = float(row["Adj Close"])

                    message_json = json.dumps(bar_data)
                    await event_bus.publish(channel_name, message_json)
                    processed_rows += 1
                    print(
                        f"Published bar {processed_rows} (Row {row_num + 1}): {message_json} to '{channel_name}'"
                    )

                    await asyncio.sleep(delay_seconds)  # Simulate real-time data feed

                except ValueError as ve:  # Handles float conversion errors primarily
                    print(
                        f"Skipping row {row_num + 1} due to data value conversion error: {ve} - Row: {row}"
                    )
                    continue
                except RedisError as re:
                    print(f"Error publishing to Redis: {re}. Attempting to continue...")
                    # Optional: add retry logic or attempt to reconnect to Redis here
                except Exception as e:
                    print(
                        f"An unexpected error occurred while processing row {row_num + 1}: {e}"
                    )
                    continue

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if event_bus.redis_client:
            await event_bus.close()
            print("Disconnected from Redis.")


async def main():
    parser = argparse.ArgumentParser(description="Market Data Simulation Engine")
    parser.add_argument(
        "csv_filepath", type=str, help="Path to the OHLCV CSV data file."
    )
    parser.add_argument(
        "--redis_url",
        type=str,
        default="redis://localhost:6379/0",
        help="Redis URL (default: redis://localhost:6379/0)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="market.ticks",
        help="Redis channel to publish market data to (default: market.ticks)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between publishing bars (default: 1.0). Cannot be used with --speed.",
    )
    group.add_argument(
        "--speed",
        type=str,
        default="1x",
        help="Speed multiplier for publishing (e.g., '1x', '2x', '0.5x'). Modifies a base delay of 1.0s. Cannot be used with --delay.",
    )

    parser.add_argument(
        "--from_date",
        type=str,
        default=None,
        help="Start publishing data from this date (YYYY-MM-DD).",
    )

    args = parser.parse_args()

    actual_delay = args.delay
    if args.speed:
        try:
            if args.speed.lower().endswith("x"):
                multiplier = float(args.speed[:-1])
                if multiplier <= 0:
                    raise ValueError("Speed multiplier must be positive.")
                actual_delay = (
                    1.0 / multiplier
                )  # Base delay is 1.0s for speed calculation
            else:
                raise ValueError("Speed format must be like '1x', '2.5x', etc.")
            if (
                hasattr(args, "delay") and args.delay != 1.0
            ):  # Check if --delay was also set from group
                if (
                    args.speed != "1x"
                ):  # if speed is set to non-default, and delay is also non-default (from group logic)
                    print(
                        "Warning: Both --delay and --speed were specified or defaulted in a way that might conflict. Using --speed's derived delay."
                    )
        except ValueError as e:
            print(f"Error: Invalid --speed value: {e}")
            return

    await publish_market_data(
        args.csv_filepath,
        args.redis_url,
        args.channel,
        delay_seconds=actual_delay,
        from_date_str=args.from_date,
    )


if __name__ == "__main__":
    asyncio.run(main())
