from __future__ import annotations

import math
import datetime
from typing import List, Optional, Iterable

import numpy as np
import pandas as pd

from azr_planner.schemas import PlanningContext, Leg, Instrument
from azr_planner.backtest.schemas import WalkForwardBacktestReport, SingleBacktestReport
from azr_planner.backtest.core import run_backtest, INITIAL_CASH
from azr_planner.backtest.metrics import calculate_cagr, calculate_max_drawdown, calculate_sharpe_ratio


TRADING_DAYS_PER_YEAR = 252 # Standard value for annualization

def _create_window_planning_context(
    base_ctx: PlanningContext,
    window_equity_curve: List[float],
    window_hlc_data: List[tuple[float, float, float]],
    window_volume_data: Optional[List[float]],
    window_start_timestamp: datetime.datetime
) -> PlanningContext:
    """
    Creates a PlanningContext for a specific window of data.
    """
    # For a window, n_successes and n_failures might be reset or carried over.
    # For simplicity, let's assume they are reset for each window, or use the base_ctx's values.
    # Using base_ctx's values might be more realistic if they represent overall strategy calibration.
    return PlanningContext(
        timestamp=window_start_timestamp, # Timestamp for the start of this window's decision making
        equity_curve=window_equity_curve,
        daily_history_hlc=window_hlc_data,
        daily_volume=window_volume_data,
        current_positions=base_ctx.current_positions, # Initial positions for each window could be reset or carried. Resetting for isolated window tests.
        vol_surface=base_ctx.vol_surface, # Assumed constant or updated externally
        risk_free_rate=base_ctx.risk_free_rate, # Assumed constant for the backtest
        nSuccesses=base_ctx.n_successes,  # Using alias again
        nFailures=base_ctx.n_failures     # Using alias again
    )


def run_walk_forward(
    full_history_ctx: PlanningContext,
    window_days: int = 30,
    step_days: int = 1
) -> WalkForwardBacktestReport:
    """
    Produce a walk-forward back-test engine that uses the metrics from AZR-08
    and returns an aggregated WalkForwardBacktestReport.

    Args:
        full_history_ctx: PlanningContext containing the entire historical data.
        window_days: The number of days in each rolling window.
        step_days: The number of days to step forward for each new window.

    Returns:
        An aggregated WalkForwardBacktestReport.
    """
    if not full_history_ctx.equity_curve or len(full_history_ctx.equity_curve) < window_days:
        raise ValueError(f"Equity curve in full_history_ctx must be at least {window_days} days long.")
    if not full_history_ctx.daily_history_hlc or len(full_history_ctx.daily_history_hlc) < window_days:
        raise ValueError(f"Daily history HLC in full_history_ctx must be at least {window_days} days long.")
    if full_history_ctx.daily_volume and len(full_history_ctx.daily_volume) < window_days:
        raise ValueError(f"Daily volume in full_history_ctx must be at least {window_days} days long if provided.")
    if len(full_history_ctx.equity_curve) != len(full_history_ctx.daily_history_hlc):
        raise ValueError("Equity curve length must match daily_history_hlc length.")
    if full_history_ctx.daily_volume and len(full_history_ctx.equity_curve) != len(full_history_ctx.daily_volume):
        raise ValueError("Equity curve length must match daily_volume length if volume is provided.")


    all_sharpe_ratios: List[float] = []
    all_max_drawdowns: List[float] = []
    all_total_trades: List[int] = []
    all_window_equity_curves: List[List[float]] = []

    # For overall win rate calculation
    total_winning_trades_overall = 0
    total_trades_overall = 0

    num_data_points = len(full_history_ctx.equity_curve)

    overall_from_date = full_history_ctx.timestamp
    overall_to_date = full_history_ctx.timestamp + datetime.timedelta(days=num_data_points -1) # Initial estimate

    # This constant defines the minimum number of data points required by the planner
    # (e.g., for latent_risk_v2 which needs 30 points for equity_curve).
    MIN_PLANNER_LOOKBACK = 30
    MIN_HLC_LOOKBACK_FOR_PLANNER = 15


    for i in range(0, num_data_points - window_days + 1, step_days):
        start_idx = i
        end_idx = i + window_days

        # The contexts_for_this_window_run will be populated with daily PlanningContext objects
        # Each of these daily PlanningContext objects must have its own lookback history.
        contexts_for_this_window_run: List[PlanningContext] = []

        # Iterate for each day within the current slice [start_idx, end_idx)
        # to create the daily PlanningContext objects for run_backtest.
        for k_in_slice in range(window_days): # k_in_slice is the offset from start_idx
            current_master_idx = start_idx + k_in_slice # Index in the full_history_ctx.equity_curve

            # Determine the lookback period for the planner for this specific day (current_master_idx)
            planner_equity_lookback_start_idx = current_master_idx - MIN_PLANNER_LOOKBACK + 1
            planner_hlc_lookback_start_idx = current_master_idx - MIN_HLC_LOOKBACK_FOR_PLANNER + 1

            # Ensure there's enough historical data for the lookback
            if planner_equity_lookback_start_idx < 0 or planner_hlc_lookback_start_idx < 0:
                continue # Not enough history to form a valid PlanningContext for this day

            pc_equity_curve = full_history_ctx.equity_curve[planner_equity_lookback_start_idx : current_master_idx + 1]
            pc_hlc = full_history_ctx.daily_history_hlc[planner_hlc_lookback_start_idx : current_master_idx + 1]

            pc_volume = None
            if full_history_ctx.daily_volume:
                planner_volume_lookback_start_idx = current_master_idx - MIN_HLC_LOOKBACK_FOR_PLANNER + 1
                if planner_volume_lookback_start_idx < 0: # Should align with HLC check
                     continue
                pc_volume = full_history_ctx.daily_volume[planner_volume_lookback_start_idx : current_master_idx + 1]
                if len(pc_volume) < MIN_HLC_LOOKBACK_FOR_PLANNER:
                    continue


            # Final check on lengths for safety, though slice logic should ensure it if start_idx is valid
            if len(pc_equity_curve) < MIN_PLANNER_LOOKBACK or len(pc_hlc) < MIN_HLC_LOOKBACK_FOR_PLANNER:
                continue

            pc_timestamp = full_history_ctx.timestamp + datetime.timedelta(days=current_master_idx)

            daily_ctx = PlanningContext(
                timestamp=pc_timestamp,
                equity_curve=pc_equity_curve,
                daily_history_hlc=pc_hlc,
                daily_volume=pc_volume,
                current_positions=None, # run_backtest manages positions internally for generate_plan
                vol_surface=full_history_ctx.vol_surface,
                risk_free_rate=full_history_ctx.risk_free_rate,
                nSuccesses=full_history_ctx.n_successes, # These could be dynamic in a more complex setup # Using alias again
                nFailures=full_history_ctx.n_failures    # Using alias again
            )
            contexts_for_this_window_run.append(daily_ctx)

        if len(contexts_for_this_window_run) < 2:
            # run_backtest needs at least two contexts: one for decision, one for next-day fill prices.
            continue

        window_report: SingleBacktestReport = run_backtest(contexts_for_this_window_run)

        if window_report.metrics.sharpe_ratio is not None and math.isfinite(window_report.metrics.sharpe_ratio):
            all_sharpe_ratios.append(window_report.metrics.sharpe_ratio)

        if window_report.metrics.max_drawdown is not None:
            all_max_drawdowns.append(window_report.metrics.max_drawdown)

        all_total_trades.append(window_report.metrics.total_trades)
        # all_window_equity_curves.append(window_report.equity_curve) # Not used currently

        total_winning_trades_overall += window_report.metrics.winning_trades
        total_trades_overall += window_report.metrics.total_trades


    mean_sharpe_np_val = np.mean(all_sharpe_ratios) if all_sharpe_ratios else None
    mean_sharpe_final: Optional[float] = None
    if mean_sharpe_np_val is not None and math.isfinite(mean_sharpe_np_val):
        mean_sharpe_final = float(mean_sharpe_np_val)

    worst_drawdown = max(all_max_drawdowns) if all_max_drawdowns else None

    # Calculate overall total_return by running run_backtest once on the full period
    full_period_contexts: List[PlanningContext] = []
    for k_full in range(num_data_points):
        current_master_idx_full = k_full
        planner_equity_lookback_start_full = current_master_idx_full - MIN_PLANNER_LOOKBACK + 1
        planner_hlc_lookback_start_full = current_master_idx_full - MIN_HLC_LOOKBACK_FOR_PLANNER + 1

        if planner_equity_lookback_start_full < 0 or planner_hlc_lookback_start_full < 0: continue

        pc_equity_curve_full = full_history_ctx.equity_curve[planner_equity_lookback_start_full : current_master_idx_full + 1]
        pc_hlc_full = full_history_ctx.daily_history_hlc[planner_hlc_lookback_start_full : current_master_idx_full + 1]

        pc_volume_full = None
        if full_history_ctx.daily_volume:
            planner_volume_lookback_start_idx_full = current_master_idx_full - MIN_HLC_LOOKBACK_FOR_PLANNER + 1
            if planner_volume_lookback_start_idx_full < 0: continue
            pc_volume_full = full_history_ctx.daily_volume[planner_volume_lookback_start_idx_full : current_master_idx_full+1]
            if len(pc_volume_full) < MIN_HLC_LOOKBACK_FOR_PLANNER: continue


        if len(pc_equity_curve_full) < MIN_PLANNER_LOOKBACK or len(pc_hlc_full) < MIN_HLC_LOOKBACK_FOR_PLANNER: continue

        pc_timestamp_full = full_history_ctx.timestamp + datetime.timedelta(days=current_master_idx_full)
        daily_ctx_full = PlanningContext(
            timestamp=pc_timestamp_full, equity_curve=pc_equity_curve_full, daily_history_hlc=pc_hlc_full, daily_volume=pc_volume_full,
            current_positions=None, vol_surface=full_history_ctx.vol_surface, risk_free_rate=full_history_ctx.risk_free_rate,
            nSuccesses=full_history_ctx.n_successes, nFailures=full_history_ctx.n_failures ) # Using alias again
        full_period_contexts.append(daily_ctx_full)

    overall_total_return_val: Optional[float] = None
    if len(full_period_contexts) >= 2:
        full_period_report = run_backtest(full_period_contexts)
        if full_period_report.initial_cash > 0:
             calculated_return = (full_period_report.final_equity / full_period_report.initial_cash) - 1.0
             overall_total_return_val = max(calculated_return, -1.0) # Cap at -100% loss
        # Update overall_from_date and overall_to_date from this definitive run
        overall_from_date = full_period_report.start_timestamp
        overall_to_date = full_period_report.end_timestamp
    else: # Fallback if no full simulation could be run
        # Use initial estimates if full_period_contexts couldn't be formed
        # This ensures from_date and to_date are always set.
        # overall_from_date and overall_to_date retain their initial estimates here.
        pass


    sum_of_trades = sum(all_total_trades)
    overall_win_rate: Optional[float] = None
    if total_trades_overall > 0:
        overall_win_rate = total_winning_trades_overall / total_trades_overall

    # If no windows ran at all, all_sharpe_ratios etc will be empty.
    # from_date and to_date should reflect the input context's span if no simulation happened.
    if not all_sharpe_ratios and not all_max_drawdowns and not all_total_trades:
         # If no windows were processed, use the original full_history_ctx timestamp boundaries
         # This might happen if num_data_points is too small for any window.
         overall_from_date = full_history_ctx.timestamp
         overall_to_date = full_history_ctx.timestamp + datetime.timedelta(days=max(0, num_data_points-1))


    return WalkForwardBacktestReport(
        from_date=overall_from_date,
        to_date=overall_to_date,
        mean_sharpe=mean_sharpe_final, # Use the correctly typed variable
        worst_drawdown=worst_drawdown, # Already None if no data
        total_return=overall_total_return_val,
        trades=sum_of_trades,
        win_rate=overall_win_rate,
    )
