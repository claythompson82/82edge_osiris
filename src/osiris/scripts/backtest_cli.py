from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Assuming azr_planner is installed or PYTHONPATH is set correctly
from azr_planner.schemas import PlanningContext
from azr_planner.engine import backtest_strategy as engine_run_walk_forward_backtest
from azr_planner.backtest.report import generate_html_report
from azr_planner.backtest.core import run_backtest as engine_run_single_backtest # For full period report data
from azr_planner.backtest.core import INITIAL_CASH # For buy and hold calc
from azr_planner.math_utils import LR_V2_MIN_POINTS # For PlanningContext construction


def parse_equity_csv(file_path: Path) -> List[Tuple[datetime.datetime, float]]:
    """Parses a CSV file with timestamp,price columns."""
    data: List[Tuple[datetime.datetime, float]] = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None or \
           'timestamp' not in reader.fieldnames or \
           'price' not in reader.fieldnames:
            raise ValueError("CSV must contain 'timestamp' and 'price' columns, and be non-empty.")
        for row in reader:
            try:
                # Try parsing with timezone first, then naive
                try:
                    ts = datetime.datetime.fromisoformat(row['timestamp'])
                except ValueError:
                    ts_naive = datetime.datetime.fromisoformat(row['timestamp'].replace('Z', ''))
                    ts = ts_naive.replace(tzinfo=datetime.timezone.utc) # Assume UTC if naive

                price = float(row['price'])
                data.append((ts, price))
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping row due to parsing error: {row} - {e}")
                continue
    if not data:
        raise ValueError(f"No valid data parsed from CSV file: {file_path}")
    # Sort by timestamp just in case
    data.sort(key=lambda x: x[0])
    return data

def create_planning_context_from_csv_data(
    timestamps: List[datetime.datetime],
    prices: List[float]
) -> PlanningContext:
    """
    Creates a full PlanningContext from parsed CSV data.
    Uses sensible defaults for fields not present in the CSV.
    """
    if not timestamps or not prices or len(timestamps) != len(prices):
        raise ValueError("Timestamps and prices lists must not be empty and must have the same length.")

    if len(prices) < LR_V2_MIN_POINTS: # LR_V2_MIN_POINTS is typically 30
        raise ValueError(f"Price data must contain at least {LR_V2_MIN_POINTS} points for planner lookback.")

    # For daily_history_hlc, assume H=L=C=price for simplicity from just price data
    daily_history_hlc = [(price, price, price) for price in prices]

    # Use the first timestamp as the reference for the context object itself.
    # The equity_curve and daily_history_hlc will contain the full series.
    context_timestamp = timestamps[0]

    return PlanningContext(
        timestamp=context_timestamp,
        equity_curve=prices, # This is the full price series, used as equity by planner
        daily_history_hlc=daily_history_hlc,
        daily_volume=None, # No volume data from this CSV type
        current_positions=None, # Start with no positions for a fresh backtest
        vol_surface={"MES": 0.20},  # Sensible default, actual value might not be critical for some planners
        risk_free_rate=0.02,        # Sensible default
        nSuccesses=50,              # Default to somewhat confident inputs
        nFailures=10
    )

def generate_daily_contexts_for_full_run(
    full_history_ctx_template: PlanningContext
) -> List[PlanningContext]:
    """
    Generates a list of daily PlanningContext objects from a full history template.
    Each daily context has the appropriate lookback.
    This mirrors the logic in `run_walk_forward` for its full period backtest.
    """
    daily_contexts: List[PlanningContext] = []
    num_data_points = len(full_history_ctx_template.equity_curve)
    min_planner_lookback = LR_V2_MIN_POINTS # Equity curve lookback for planner
    min_hlc_lookback = 15 # HLC lookback for planner (e.g. ATR)

    for k_full in range(num_data_points):
        current_master_idx_full = k_full

        planner_equity_lookback_start_full = current_master_idx_full - min_planner_lookback + 1
        planner_hlc_lookback_start_full = current_master_idx_full - min_hlc_lookback + 1

        if planner_equity_lookback_start_full < 0 or planner_hlc_lookback_start_full < 0:
            continue

        pc_equity_curve_full = full_history_ctx_template.equity_curve[planner_equity_lookback_start_full : current_master_idx_full + 1]
        pc_hlc_full = full_history_ctx_template.daily_history_hlc[planner_hlc_lookback_start_full : current_master_idx_full + 1]

        # Ensure lookback data meets minimum lengths required by PlanningContext schema itself
        if len(pc_equity_curve_full) < min_planner_lookback or len(pc_hlc_full) < min_hlc_lookback:
            continue

        pc_timestamp_full = full_history_ctx_template.timestamp + datetime.timedelta(days=current_master_idx_full)

        daily_ctx_full = PlanningContext(
            timestamp=pc_timestamp_full,
            equity_curve=pc_equity_curve_full,
            daily_history_hlc=pc_hlc_full,
            daily_volume=None, # Assuming no volume from input CSV
            current_positions=None, # Handled by run_backtest
            vol_surface=full_history_ctx_template.vol_surface,
            risk_free_rate=full_history_ctx_template.risk_free_rate,
            nSuccesses=full_history_ctx_template.n_successes, # Field name
            nFailures=full_history_ctx_template.n_failures   # Field name
        )
        daily_contexts.append(daily_ctx_full)
    return daily_contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest and generate HTML report.")
    parser.add_argument(
        "--equity-curve",
        type=Path,
        required=True,
        help="Path to CSV file with timestamp,price columns for the equity curve/price data."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the HTML report."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=31, # Default slice length, must be > MIN_PLANNER_LOOKBACK (30)
        help="Number of days in each rolling window slice for walk-forward analysis."
    )
    args = parser.parse_args()

    try:
        print(f"Loading equity data from: {args.equity_curve}")
        csv_data = parse_equity_csv(args.equity_curve)
        timestamps, prices = zip(*csv_data)

        print("Creating planning context...")
        # This context uses the full history from the CSV
        full_history_planning_ctx = create_planning_context_from_csv_data(list(timestamps), list(prices))

        print("Running walk-forward backtest (engine.backtest_strategy)...")
        # This returns the WalkForwardBacktestReport as a dict
        walk_forward_report_dict = engine_run_walk_forward_backtest(
            ctx=full_history_planning_ctx,
            window_days=args.window_days
        )

        print("Running full period backtest for detailed report data...")
        # Generate daily contexts for the full period run
        daily_contexts_for_full_run = generate_daily_contexts_for_full_run(full_history_planning_ctx)

        full_single_backtest_report_obj: Optional[Any] = None # Using Any for now for the object before model_dump
        full_single_backtest_report_dict: Optional[Dict[str, Any]] = None
        strategy_equity_curve: Optional[List[float]] = None
        strategy_equity_timestamps: Optional[List[datetime.datetime]] = None
        trade_pnls: Optional[List[float]] = [] # Initialize as empty list

        if len(daily_contexts_for_full_run) >= 2:
            full_single_backtest_report_obj = engine_run_single_backtest(daily_contexts_for_full_run)
            full_single_backtest_report_dict = full_single_backtest_report_obj.model_dump()
            strategy_equity_curve = full_single_backtest_report_obj.equity_curve

            # Extract timestamps for the strategy equity curve from its daily_results
            # Each daily_result corresponds to a step in the equity curve after the initial value.
            if full_single_backtest_report_obj.daily_results: # This check implies full_single_backtest_report_obj is not None
                # Equity curve has N+1 points, daily_results has N points.
                # First point is initial cash at 'start_timestamp' - 1 day effectively, or use result timestamps.
                # The daily_results[j].portfolio_state_after_trades.timestamp aligns with equity_curve[j+1]
                strategy_equity_timestamps = \
                    [full_single_backtest_report_obj.start_timestamp] + \
                    [dr.portfolio_state_after_trades.timestamp for dr in full_single_backtest_report_obj.daily_results]

            # Extract P&Ls
            # Ensure trade_pnls is a list before appending
            if not isinstance(trade_pnls, list): # Should have been initialized as list
                trade_pnls = []

            if full_single_backtest_report_obj.daily_results: # Same check as above
                for daily_res in full_single_backtest_report_obj.daily_results:
                    for trade in daily_res.trades_executed:
                        if trade.pnl is not None:
                            trade_pnls.append(trade.pnl) # Appending to the list
        else:
            print("Warning: Not enough data to run the full period backtest for detailed charts.")


        print("Generating buy-and-hold equity curve...")
        buy_and_hold_equity_curve: Optional[List[float]] = None
        buy_and_hold_timestamps: Optional[List[datetime.datetime]] = None
        if prices:
            first_price = prices[0]
            if first_price > 0:
                buy_and_hold_equity_curve = [INITIAL_CASH * (p / first_price) for p in prices]
                buy_and_hold_timestamps = list(timestamps) # Timestamps from CSV align with these prices

        print(f"Generating HTML report to: {args.out}")
        generate_html_report(
            walk_forward_report_data=walk_forward_report_dict,
            strategy_equity_curve=strategy_equity_curve,
            strategy_equity_timestamps=strategy_equity_timestamps,
            buy_and_hold_equity_curve=buy_and_hold_equity_curve,
            buy_and_hold_timestamps=buy_and_hold_timestamps,
            trade_pnls=trade_pnls if trade_pnls else None, # Pass None if empty
            out_path=args.out
        )
        print("Report generation complete.")

    except ValueError as ve:
        print(f"Error: {ve}")
    except FileNotFoundError:
        print(f"Error: Equity curve CSV file not found at {args.equity_curve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # This allows running the script directly, e.g., python src/osiris/scripts/backtest_cli.py ...
    # Ensure OSIRIS_TEST is set if azr_planner components require it (e.g. for engine endpoints if used differently)
    # For this CLI, direct imports should work if `pip install -e .` was done.
    main()
