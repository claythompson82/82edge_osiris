import argparse
import asyncio
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tsaug import TimeWarp

from llm_sidecar.event_bus import EventBus, RedisError


@dataclass
class Portfolio:
    cash: float
    position: float = 0.0
    history: list = field(default_factory=list)

    def value(self, market_price: float) -> float:
        return self.cash + self.position * market_price


def apply_gaussian_noise(
    bars: List[Dict[str, float]], std_ratio: float
) -> List[Dict[str, float]]:
    """Apply Gaussian noise to OHLCV values in-place."""
    for bar in bars:
        keys = ["open", "high", "low", "close", "volume"]
        if "adj_close" in bar:
            keys.append("adj_close")
        for k in keys:
            val = bar[k]
            noise = random.gauss(0, std_ratio * val)
            bar[k] = val + noise
    return bars


def apply_time_warp(bars: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Apply time warping to the OHLCV series using tsaug."""
    if not bars:
        return bars
    feature_keys = ["open", "high", "low", "close", "volume"]
    if "adj_close" in bars[0]:
        feature_keys.append("adj_close")

    data = np.array([[bar[k] for k in feature_keys] for bar in bars], dtype=float)
    aug = TimeWarp()
    warped = aug.augment(data.reshape(1, len(bars), len(feature_keys)))[0]

    for i, bar in enumerate(bars):
        for j, key in enumerate(feature_keys):
            value = float(warped[i, j])
            if key == "volume" and value < 0:
                value = 0.0
            bar[key] = value
    return bars


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
    augment: bool = False,
    noise_std_ratio: float = 0.001,
    order_channel: Optional[str] = "market.orders",
    commission_rate: float = 0.001,
    slippage_impact: float = 0.1,
    starting_cash: float = 100_000.0,
):
    """
    Reads OHLCV data from a CSV, optionally applies augmentation, and publishes
    each bar to Redis while listening for and executing trades.
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
        all_bars: List[Dict[str, float]] = []
        with open(csv_filepath, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            # ... (omitting header validation logic for brevity, assuming it's correct) ...
            for row in reader:
                # ... (omitting date filtering and parsing for brevity) ...
                bar_data = {
                    "timestamp": row.get("Timestamp") or row.get("Date"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
                if "Adj Close" in row:
                    bar_data["adj_close"] = float(row["Adj Close"])
                all_bars.append(bar_data)

        if augment:
            print("Applying data augmentation...")
            all_bars = apply_gaussian_noise(all_bars, noise_std_ratio)
            all_bars = apply_time_warp(all_bars)
            print("Augmentation complete.")

        processed_rows = 0
        for bar in all_bars:
            latest_bar = bar
            message_json = json.dumps(bar)
            await event_bus.publish(channel_name, message_json)
            processed_rows += 1
            print(f"Published bar {processed_rows}: {message_json} to '{channel_name}'")
            await asyncio.sleep(delay_seconds)

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
    parser.add_argument("csv_filepath", type=str, help="Path to the OHLCV CSV data file.")
    # ... (omitting argument parsing for delay/speed for brevity) ...
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--channel", type=str, default="market.ticks")
    parser.add_argument("--from_date", type=str, default=None)

    # Arguments for Data Augmentation
    parser.add_argument("--augment_data", action="store_true", help="Enable data augmentation.")
    parser.add_argument("--noise_std_ratio", type=float, default=0.001)

    # Arguments for Trading Frictions
    parser.add_argument("--order_channel", type=str, default="market.orders")
    parser.add_argument("--commission_rate", type=float, default=0.001)
    parser.add_argument("--slippage_impact", type=float, default=0.1)
    parser.add_argument("--starting_cash", type=float, default=100000.0)

    args = parser.parse_args()
    # ... (omitting delay/speed calculation for brevity) ...
    actual_delay = 1.0

    await publish_market_data(
        args.csv_filepath,
        args.redis_url,
        args.channel,
        delay_seconds=actual_delay,
        from_date_str=args.from_date,
        augment=args.augment_data,
        noise_std_ratio=args.noise_std_ratio,
        order_channel=args.order_channel,
        commission_rate=args.commission_rate,
        slippage_impact=args.slippage_impact,
        starting_cash=args.starting_cash,
    )


if __name__ == "__main__":
    asyncio.run(main())