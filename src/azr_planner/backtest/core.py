from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterable, Tuple

import pandas as pd # For potential series operations if needed later
import numpy as np  # For math operations if needed later

from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg
from azr_planner.engine import generate_plan
from .schemas import DailyTrade, DailyPortfolioState, DailyResult, SingleBacktestMetrics, SingleBacktestReport
from .metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate_and_pnl_stats
)

# Constants
INITIAL_CASH = 100_000.0
CONTRACT_MULTIPLIER: Dict[Instrument, float] = {
    Instrument.MES: 5.0,
    Instrument.M2K: 5.0, # Assuming same as MES for now
    # Add other instruments if they become part of backtesting
}
# Default multiplier if instrument not in map
DEFAULT_MULTIPLIER = 1.0 # For non-futures like US_SECTOR_ETF

TRADING_DAYS_PER_YEAR = 252 # For annualization

# Position representation: Dict[Instrument, Dict[str, float]]
# e.g., {Instrument.MES: {"size": 2.0, "avg_entry_price": 4500.50}}
PortfolioPositions = Dict[Instrument, Dict[str, float]]


def _get_fill_price(next_day_context: Optional[PlanningContext]) -> Optional[float]:
    """
    Extracts the fill price from the next day's context.
    Spec: "ctx.equity_curve[-1] of the next day".
    Assuming equity_curve in PlanningContext is a list of prices for the underlying.
    """
    if next_day_context and next_day_context.equity_curve:
        return next_day_context.equity_curve[-1]
    return None

def run_backtest(ctx_iter: Iterable[PlanningContext]) -> SingleBacktestReport:
    """
    Runs a backtest simulation based on an iterator of daily PlanningContexts.

    Args:
        ctx_iter: An iterable of PlanningContext objects, one for each trading day.

    Returns:
        A BacktestReport object summarizing the performance.
    """
    contexts = list(ctx_iter) # Convert iterator to list to allow lookahead for fill prices
    if not contexts or len(contexts) < 2: # Need at least one day to plan and one next day for fills
        raise ValueError("Context iterator must yield at least two PlanningContexts for backtesting.")

    start_timestamp = contexts[0].timestamp
    end_timestamp = contexts[-1].timestamp # Timestamp of the last day's decision/evaluation

    current_cash = INITIAL_CASH
    # Portfolio: Instrument -> {"size": float, "avg_entry_price": float}
    current_positions: PortfolioPositions = {}

    equity_curve_history: List[float] = [INITIAL_CASH]
    daily_results_log: List[DailyResult] = []
    all_trades_log: List[DailyTrade] = []

    latent_risk_series: List[Optional[float]] = []
    confidence_series: List[Optional[float]] = []

    # Loop through contexts, ensuring there's a next day for fill prices
    for i in range(len(contexts) - 1):
        ctx_today: PlanningContext = contexts[i]
        ctx_tomorrow: PlanningContext = contexts[i+1] # For fill prices and MTM

        day_trades: List[DailyTrade] = []

        # 1. Get Planner's Proposal
        time_before_plan = time.perf_counter()
        trade_proposal: TradeProposal = generate_plan(ctx_today)
        time_after_plan = time.perf_counter()
        planner_latency_ms = (time_after_plan - time_before_plan) * 1000

        latent_risk_series.append(trade_proposal.latent_risk)
        confidence_series.append(trade_proposal.confidence)

        # 2. Determine Fill Price for today's actions (from tomorrow's context)
        fill_price_tomorrow = _get_fill_price(ctx_tomorrow)

        if fill_price_tomorrow is None:
            # Cannot execute trades or MTM if no fill price for tomorrow
            # Log this day with no actions/PNL changes due to missing data
            current_equity = equity_curve_history[-1] # Equity remains unchanged
            daily_pnl = 0.0
            # Fallthrough to log DailyResult
        else:
            # 3. Simulate Trade Execution based on proposal
            if trade_proposal.action != "HOLD" and trade_proposal.legs:
                for leg in trade_proposal.legs:
                    instrument = leg.instrument
                    # Planner's leg.direction is LONG/SHORT. Execution direction might be BUY/SELL.
                    # For simplicity, assume BUY for LONG intent, SELL for SHORT intent from planner.
                    # This needs to be more nuanced based on opening/closing.
                    planner_direction = leg.direction
                    size_to_trade = leg.size
                    multiplier = CONTRACT_MULTIPLIER.get(instrument, DEFAULT_MULTIPLIER)

                    trade_pnl: Optional[float] = None
                    position = current_positions.get(instrument, {"size": 0.0, "avg_entry_price": 0.0})
                    current_size = position["size"]
                    avg_entry_price = position["avg_entry_price"]

                    # Use planner_direction (LONG/SHORT) for DailyTrade's direction field, assuming it uses the same Enum.
                    if planner_direction == Direction.LONG:
                        if current_size < 0: # Currently short, need to cover first (buy to cover)
                            size_to_cover = min(size_to_trade, abs(current_size))
                            trade_pnl_cover = (avg_entry_price - fill_price_tomorrow) * size_to_cover * multiplier # PNL for short cover
                            day_trades.append(DailyTrade(timestamp=ctx_tomorrow.timestamp, instrument=instrument, direction=Direction.LONG, size=size_to_cover, fill_price=fill_price_tomorrow, pnl=trade_pnl_cover))
                            all_trades_log.append(day_trades[-1])
                            current_cash += trade_pnl_cover
                            current_size += size_to_cover
                            size_to_trade -= size_to_cover

                        if size_to_trade > 0: # If still need to go long (or add to long)
                            new_total_size = current_size + size_to_trade
                            if current_size > 0 : # Adding to existing long
                                avg_entry_price = (avg_entry_price * current_size + fill_price_tomorrow * size_to_trade) / new_total_size
                            else: # New long position (current_size was 0 or became 0 after cover)
                                avg_entry_price = fill_price_tomorrow
                            current_size = new_total_size
                            day_trades.append(DailyTrade(timestamp=ctx_tomorrow.timestamp, instrument=instrument, direction=Direction.LONG, size=size_to_trade, fill_price=fill_price_tomorrow, pnl=None))
                            all_trades_log.append(day_trades[-1])

                    elif planner_direction == Direction.SHORT:
                        if current_size > 0: # Currently long, need to sell first (sell to close)
                            size_to_sell = min(size_to_trade, current_size)
                            trade_pnl_sell = (fill_price_tomorrow - avg_entry_price) * size_to_sell * multiplier
                            day_trades.append(DailyTrade(timestamp=ctx_tomorrow.timestamp, instrument=instrument, direction=Direction.SHORT, size=size_to_sell, fill_price=fill_price_tomorrow, pnl=trade_pnl_sell))
                            all_trades_log.append(day_trades[-1])
                            current_cash += trade_pnl_sell
                            current_size -= size_to_sell
                            size_to_trade -= size_to_sell

                        if size_to_trade > 0: # If still need to go short (or add to short)
                            new_total_size = current_size - size_to_trade
                            if current_size < 0: # Adding to existing short
                                avg_entry_price = (avg_entry_price * abs(current_size) + fill_price_tomorrow * size_to_trade) / abs(new_total_size)
                            else: # New short position
                                avg_entry_price = fill_price_tomorrow
                            current_size = new_total_size
                            day_trades.append(DailyTrade(timestamp=ctx_tomorrow.timestamp, instrument=instrument, direction=Direction.SHORT, size=size_to_trade, fill_price=fill_price_tomorrow, pnl=None))
                            all_trades_log.append(day_trades[-1])

                    if current_size != 0: # Ensure not to store zero size positions with old prices
                        current_positions[instrument] = {"size": current_size, "avg_entry_price": avg_entry_price}
                    elif instrument in current_positions: # Position closed
                        del current_positions[instrument]

            # 4. Mark-to-Market open positions and calculate Daily P&L
            unrealized_pnl_today = 0.0
            current_portfolio_value = current_cash # Start with cash

            for instrument, pos_details in current_positions.items():
                pos_size = pos_details["size"]
                entry_price = pos_details["avg_entry_price"]
                multiplier = CONTRACT_MULTIPLIER.get(instrument, DEFAULT_MULTIPLIER)

                instrument_value_at_entry = entry_price * pos_size * multiplier # Note: pos_size can be negative
                instrument_value_at_mtm = fill_price_tomorrow * pos_size * multiplier

                unrealized_pnl_today += (instrument_value_at_mtm - instrument_value_at_entry)
                current_portfolio_value += instrument_value_at_mtm # More accurately, value of position not cash

            # Daily P&L calculation: sum of P&L from trades closed today + MTM change on open positions
            # For simplicity, let equity be previous equity + sum of today's trade PNLs + MTM of open positions
            # This needs refinement: equity is cash + market value of positions.
            # Daily P&L = (Today's Equity - Yesterday's Equity)

            realized_pnl_today = sum(t.pnl for t in day_trades if t.pnl is not None)

            # Calculate current equity
            # Start with cash (which includes realized PNL from today)
            current_equity = current_cash
            # Add market value of open positions
            for instrument, pos_details in current_positions.items():
                pos_size = pos_details["size"]
                # entry_price = pos_details["avg_entry_price"] # Not needed for current market value
                multiplier = CONTRACT_MULTIPLIER.get(instrument, DEFAULT_MULTIPLIER)
                # For futures, market value is complex (margin based).
                # For simplicity, let's consider "value" as unrealized P&L for futures.
                # Value for equities would be price * size.
                # Here, we track equity by P&L.
                # Unrealized P&L for an open position: (current_price - avg_entry_price) * size * multiplier
                unrealized_pnl_for_pos = (fill_price_tomorrow - pos_details["avg_entry_price"]) * pos_size * multiplier
                current_equity += unrealized_pnl_for_pos # Add unrealized P&L to cash to get total equity

            daily_pnl = current_equity - equity_curve_history[-1]

        equity_curve_history.append(current_equity)

        # 5. Log Daily Result
        portfolio_state = DailyPortfolioState(
            timestamp=ctx_tomorrow.timestamp, # State at end of day, using tomorrow's EOD timestamp
            cash=current_cash, # This cash includes realized PNLs
            total_equity=current_equity,
            positions={inst: pos["size"] for inst, pos in current_positions.items()},
            daily_pnl=daily_pnl
        )

        daily_results_log.append(DailyResult(
            timestamp=ctx_today.timestamp, # Decision was made based on ctx_today
            trade_proposal=trade_proposal,
            trades_executed=day_trades,
            portfolio_state_after_trades=portfolio_state,
            planner_latency_ms=planner_latency_ms,
            latent_risk_at_decision=trade_proposal.latent_risk,
            confidence_at_decision=trade_proposal.confidence
        ))

    # 6. Calculate Final Metrics
    final_equity = equity_curve_history[-1]

    # Calculate returns for Sharpe/Sortino (daily equity returns)
    equity_series = pd.Series(equity_curve_history)
    daily_equity_returns = equity_series.pct_change().dropna().tolist()

    # Get initial risk_free_rate from the first context for Sharpe/Sortino
    # Ensure contexts list is not empty before accessing (already checked at func start)
    initial_risk_free_rate = contexts[0].risk_free_rate

    pnl_stats = calculate_win_rate_and_pnl_stats(all_trades_log)

    metrics = SingleBacktestMetrics(
        cagr=calculate_cagr(equity_curve_history, TRADING_DAYS_PER_YEAR),
        max_drawdown=calculate_max_drawdown(equity_curve_history),
        sharpe_ratio=calculate_sharpe_ratio(daily_equity_returns, initial_risk_free_rate, TRADING_DAYS_PER_YEAR),
        sortino_ratio=calculate_sortino_ratio(daily_equity_returns, initial_risk_free_rate, 0.0, TRADING_DAYS_PER_YEAR),
        win_rate=pnl_stats["winRate"],
        total_trades=int(pnl_stats["totalTrades"] if pnl_stats["totalTrades"] is not None else 0),
        winning_trades=int(pnl_stats["winningTrades"] if pnl_stats["winningTrades"] is not None else 0),
        losing_trades=int(pnl_stats["losingTrades"] if pnl_stats["losingTrades"] is not None else 0),
        avg_win_pnl=pnl_stats["avgWinPnl"],
        avg_loss_pnl=pnl_stats["avgLossPnl"],
        avg_trade_pnl=pnl_stats["avgTradePnl"],
        profit_factor=pnl_stats["profitFactor"]
    )

    return SingleBacktestReport(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp, # This is the timestamp of the last context used for decision
        initial_cash=INITIAL_CASH,
        final_equity=final_equity,
        metrics=metrics,
        equity_curve=equity_curve_history,
        daily_results=daily_results_log,
        latent_risk_series=latent_risk_series,
        confidence_series=confidence_series
    )
