from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

# Assuming DailyTrade is defined in .schemas or ..schemas
# For now, let's use a relative import that should work if this file is part of the backtest package.
from .schemas import DailyTrade


def calculate_cagr(equity_curve: List[float], num_trading_days_per_year: int = 252) -> Optional[float]:
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
        equity_curve: A list of equity values, chronologically ordered.
        num_trading_days_per_year: Number of trading days in a year.

    Returns:
        The CAGR as a decimal (e.g., 0.1 for 10%), or None if calculation is not possible.
    """
    if not equity_curve or len(equity_curve) < 2:
        return None

    start_value = equity_curve[0]
    end_value = equity_curve[-1]

    if start_value == 0: # Avoid division by zero if starting equity is zero
        return None
    if start_value < 0 or end_value < 0: # CAGR is typically for positive values
        # Or handle as appropriate for strategy (e.g. always return None, or allow if contextually makes sense)
        return None # For now, assume positive equity path

    num_days = len(equity_curve) - 1
    if num_days == 0: # Single data point
        return 0.0

    num_years = num_days / num_trading_days_per_year
    if num_years == 0:
        # If num_days is non-zero but rounds to 0 years (e.g. < 1 day effectively if days_per_year is large)
        # or if num_days_per_year is extremely large.
        # Avoid division by zero if num_years is truly zero.
        # If num_days > 0, num_years will be > 0.
        # This case is more about very short periods leading to extreme CAGR.
        # For simplicity, if num_years ends up zero due to very few days,
        # it might be better to return 0 or None, but formula with small num_years is still valid.
        # The check `len(equity_curve) < 2` handles num_days = 0.
        # If num_years is positive but very small, CAGR can be huge.
        pass


    # Ensure num_years is not zero before division in exponent
    if num_years == 0: # This would only happen if num_days = 0, which is handled.
                      # Or if num_trading_days_per_year is float('inf'), not the case.
        return 0.0 # No change over zero years is zero growth.

    cagr = (end_value / start_value) ** (1.0 / num_years) - 1.0
    return float(cagr)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculates the Maximum Drawdown (MDD) from an equity curve.

    Args:
        equity_curve: A list of equity values, chronologically ordered.

    Returns:
        The maximum drawdown as a positive decimal (e.g., 0.2 for 20% drawdown).
        Returns 0.0 if no drawdown occurs or if data is insufficient.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity_series = pd.Series(equity_curve, dtype=float) # Ensure dtype is float for calculations

    # Calculate cumulative max up to each point
    # .cummax() is appropriate here and simpler than .expanding().max() for this use case
    cumulative_max = equity_series.cummax()

    # Drawdown series: (current_value / peak_value_so_far) - 1
    # This gives drawdown as a decimal, e.g., -0.1 for a 10% drawdown.
    # Avoid division by zero if cumulative_max can be zero (e.g. if equity can go to 0)
    # For equity curves, typically start > 0. If cummax becomes 0, it's problematic.
    # Let's assume cummax will be positive if start_value is positive.
    drawdown_series = (equity_series / cumulative_max) - 1.0

    # Max drawdown is the minimum value in the drawdown series (most negative)
    min_drawdown_value = drawdown_series.min() # This will be <= 0

    if pd.isna(min_drawdown_value) or not isinstance(min_drawdown_value, (float, np.floating)):
        # If all NaNs or unexpected type, means no valid drawdown calculable (e.g. single point curve after ops)
        return 0.0

    max_dd_as_positive_float = abs(float(min_drawdown_value))

    return max_dd_as_positive_float


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate_annual: float,
    num_trading_days_per_year: int = 252
) -> Optional[float]:
    """
    Calculates the Sharpe Ratio.

    Args:
        returns: A list of periodic returns (e.g., daily).
        risk_free_rate_annual: Annual risk-free rate (decimal, e.g., 0.02 for 2%).
        num_trading_days_per_year: Number of trading days in a year for annualization.

    Returns:
        The annualized Sharpe Ratio, or None if calculation is not possible.
    """
    if not returns or len(returns) < 2:
        return None

    returns_series = pd.Series(returns)

    # Convert annual risk-free rate to periodic
    risk_free_rate_periodic = (1 + risk_free_rate_annual) ** (1 / num_trading_days_per_year) - 1

    excess_returns = returns_series - risk_free_rate_periodic

    mean_excess_return = float(excess_returns.mean())
    std_dev_excess_return = float(excess_returns.std())

    if std_dev_excess_return == 0 or math.isnan(std_dev_excess_return) or math.isnan(mean_excess_return):
        return None # Sharpe is undefined if no volatility or if mean is NaN

    sharpe_ratio_periodic = mean_excess_return / std_dev_excess_return
    annualized_sharpe_ratio = sharpe_ratio_periodic * math.sqrt(num_trading_days_per_year)

    return float(annualized_sharpe_ratio)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate_annual: float,
    target_return_annual: float = 0.0, # Often the risk-free rate itself
    num_trading_days_per_year: int = 252
) -> Optional[float]:
    """
    Calculates the Sortino Ratio.

    Args:
        returns: A list of periodic returns (e.g., daily).
        risk_free_rate_annual: Annual risk-free rate.
        target_return_annual: Annual target return (Minimum Acceptable Return - MAR).
        num_trading_days_per_year: Number of trading days for annualization.

    Returns:
        The annualized Sortino Ratio, or None if calculation is not possible.
    """
    if not returns or len(returns) < 2:
        return None

    returns_series = pd.Series(returns)

    risk_free_rate_periodic = (1 + risk_free_rate_annual) ** (1 / num_trading_days_per_year) - 1
    target_return_periodic = (1 + target_return_annual) ** (1 / num_trading_days_per_year) - 1

    excess_returns_over_rf = returns_series - risk_free_rate_periodic # For numerator
    mean_excess_return = float(excess_returns_over_rf.mean())

    # Downside deviation
    downside_diff = returns_series - target_return_periodic
    downside_squares = downside_diff[downside_diff < 0].pow(2) # Only negative deviations from target

    if downside_squares.empty: # No returns below target
        # Sortino is undefined or infinite if there's positive excess return and no downside risk.
        # If mean_excess_return is also zero or negative, then Sortino could be 0 or negative.
        # For simplicity, if no downside deviation, might return very large number if mean_excess_return > 0
        # or 0 if mean_excess_return <=0. Or None. Let's return None as it's a special case.
        return None

    mean_downside_squares = float(downside_squares.mean()) # Cast to float
    downside_deviation_periodic = math.sqrt(mean_downside_squares)

    if downside_deviation_periodic == 0 or math.isnan(downside_deviation_periodic) or math.isnan(mean_excess_return):
        return None

    sortino_ratio_periodic = mean_excess_return / downside_deviation_periodic
    annualized_sortino_ratio = sortino_ratio_periodic * math.sqrt(num_trading_days_per_year)

    return float(annualized_sortino_ratio)


def calculate_win_rate_and_pnl_stats(trades: List[DailyTrade]) -> Dict[str, Optional[float]]:
    """
    Calculates win rate and various P&L statistics from a list of trades.
    Assumes each trade object has a 'pnl' attribute.

    Args:
        trades: A list of DailyTrade objects.

    Returns:
        A dictionary containing: win_rate, total_trades, winning_trades, losing_trades,
        avg_win_pnl, avg_loss_pnl, avg_trade_pnl, profit_factor.
        Values can be None if not calculable (e.g., no trades, no wins, or no losses).
    """
    if not trades:
        return {
            "winRate": None, "totalTrades": 0, "winningTrades": 0, "losingTrades": 0,
            "avgWinPnl": None, "avgLossPnl": None, "avgTradePnl": None, "profitFactor": None
        }

    pnls = [trade.pnl for trade in trades if trade.pnl is not None]
    if not pnls: # All trades had None PNL, or no trades with PNL
        return {
            "winRate": None, "totalTrades": 0, "winningTrades": 0, "losingTrades": 0, # totalTrades should be 0 if no PNLs
            "avgWinPnl": None, "avgLossPnl": None, "avgTradePnl": None, "profitFactor": None
        }

    total_trades = len(pnls)
    winning_trades_pnl = [p for p in pnls if p > 0]
    losing_trades_pnl = [p for p in pnls if p < 0] # pnl is negative for losses

    num_winning_trades = len(winning_trades_pnl)
    num_losing_trades = len(losing_trades_pnl)

    win_rate = num_winning_trades / total_trades if total_trades > 0 else None

    avg_win_pnl = sum(winning_trades_pnl) / num_winning_trades if num_winning_trades > 0 else None
    avg_loss_pnl = sum(losing_trades_pnl) / num_losing_trades if num_losing_trades > 0 else None # Will be negative or zero

    avg_trade_pnl = sum(pnls) / total_trades if total_trades > 0 else None

    gross_profit = sum(winning_trades_pnl)
    gross_loss = abs(sum(losing_trades_pnl)) # abs because losses are negative

    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0 and gross_loss == 0: # Only wins, no losses
        profit_factor = float('inf') # Or a very large number, or None
    # If gross_profit is 0 and gross_loss is 0, profit_factor is undefined (None is good)

    return {
        "winRate": win_rate,
        "totalTrades": total_trades,
        "winningTrades": num_winning_trades,
        "losingTrades": num_losing_trades,
        "avgWinPnl": avg_win_pnl,
        "avgLossPnl": avg_loss_pnl,
        "avgTradePnl": avg_trade_pnl,
        "profitFactor": profit_factor,
    }
