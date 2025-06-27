# This file makes the 'backtest' directory a Python package.
# It can also be used to expose parts of the package's API.

from .core import run_backtest
from .schemas import BacktestReport, DailyResult, DailyTrade, DailyPortfolioState, BacktestMetrics
from .metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate_and_pnl_stats
)

__all__ = [
    "run_backtest",
    "BacktestReport",
    "DailyResult",
    "DailyTrade",
    "DailyPortfolioState",
    "BacktestMetrics",
    "calculate_cagr",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_win_rate_and_pnl_stats",
]
