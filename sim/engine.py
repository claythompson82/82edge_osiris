import csv
import json
import asyncio
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from llm_sidecar.event_bus import EventBus, RedisError


@dataclass
class Portfolio:
    cash: float
    position: float = 0.0
    history: list = field(default_factory=list)

    def value(self, market_price: float) -> float:
        return self.cash + self.position * market_price


def calculate_slippage(
    order_size: float, bar_volume: float, market_price: float, slippage_impact: float
) -> float:
    """Return adjusted fill price accounting for slippage."""
    if bar_volume <= 0:
        return market_price
    price_adjustment = (abs(order_size) / bar_volume) * slippage_impact
    if order_size > 0:
        return market_price + price_adjustment
    else:
        return market_price - price_adjustment


def apply_transaction_cost(
    quantity: float, execution_price: float, commission_rate: float
) -> float:
    """Return commission cost for a trade."""
    return abs(quantity) * execution_price * commission_rate


def execute_trade(
    portfolio: Portfolio,
    quantity: float,
    market_price: float,
    bar_volume: float,
    commission_rate: float,
    slippage_impact: float,
    timestamp: str,
) -> None:
    """Execute a trade with slippage and commission."""

    fill_price = calculate_slippage(quantity, bar_volume, market_price, slippage_impact)
    commission = apply_transaction_cost(quantity, fill_price, commission_rate)

    portfolio.cash -= quantity * fill_price
    portfolio.cash -= commission
    portfolio.position += quantity
    portfolio.history.append(
        {
            "timestamp": timestamp,
            "quantity": quantity,
            "fill_price": fill_price,
            "commission": commission,
        }
    )


async def publish_market_data(
    csv_filepath: str,
    redis_url: str,
    channel_name: str,
    delay_seconds: float = 1.0,
    from_date_str: str = None,
    order_channel: Optional[str] = "market.orders",
    commission_rate: float = 0.001,
    slippage_impact: float = 0.1,
    starting_cash: float = 100_000.0,
):
    """
    Reads OHLCV data from a CSV file and publishes each bar as a JSON dictionary
    to the specified Redis channel.
    """
    event_bus = EventBus(redis_url=redis_url)
    portfolio = Portfolio(cash=starting_cash)
    latest_bar = None

    async def handle_order(message: str):
        nonlocal latest_bar
        if not latest_bar:
            print("Order received but no market data available yet. Skipping")
            return
        try:
            order = json.loads(message)
            qty = float(order.get("quantity", 0))
        except Exception as exc:
            print(f"Invalid order message '{message}': {exc}")
            return

        execute_trade(
            portfolio,
            quantity=qty,
            market_price=latest_bar["close"],
            bar_volume=latest_bar["volume"],
            commission_rate=commission_rate,
            slippage_impact=slippage_impact,
            timestamp=latest_bar["timestamp"],
        )
        print(
            f"Executed order for {qty} units at last price {latest_bar['close']}. Cash now {portfolio.cash:.2f}"
        )

    try:
        await event_bus.connect()
        print(f"Connected to Redis at {redis_url}")
        if order_channel:
            await event_bus.subscribe(order_channel, handle_order)
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

                    latest_bar = bar_data

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

            if latest_bar:
                final_value = portfolio.value(latest_bar["close"])
                pnl = final_value - starting_cash
                print(f"Simulation complete. Final P&L: {pnl:.2f}")

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

    parser.add_argument(
        "--order_channel",
        type=str,
        default="market.orders",
        help="Redis channel to listen for order messages (default: market.orders)",
    )
    parser.add_argument(
        "--commission_rate",
        type=float,
        default=0.001,
        help="Commission rate applied per trade (default: 0.001)",
    )
    parser.add_argument(
        "--slippage_impact",
        type=float,
        default=0.1,
        help="Slippage impact factor (default: 0.1)",
    )
    parser.add_argument(
        "--starting_cash",
        type=float,
        default=100000.0,
        help="Starting cash balance for the portfolio",
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
        order_channel=args.order_channel,
        commission_rate=args.commission_rate,
        slippage_impact=args.slippage_impact,
        starting_cash=args.starting_cash,
    )


if __name__ == "__main__":
    asyncio.run(main())
