from __future__ import annotations # Added

from .core import run_backtest
from .runner import run_walk_forward # Added
from .schemas import (
    DailyTrade,
    DailyPortfolioState,
    DailyResult,
    SingleBacktestMetrics,   # Renamed
    SingleBacktestReport,    # Renamed
    WalkForwardBacktestReport # New
)
from .metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate_and_pnl_stats
)
# Example for TYPE_CHECKING if pandas was directly used and causing issues:
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     import pandas as pd # Not strictly needed here as pandas isn't directly in this API surface

__all__ = [
    "run_backtest",
    "run_walk_forward",          # Added
    "DailyTrade",
    "DailyPortfolioState",
    "DailyResult",
    "SingleBacktestMetrics",     # Renamed
    "SingleBacktestReport",      # Renamed
    "WalkForwardBacktestReport", # Added
    "calculate_cagr",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_win_rate_and_pnl_stats",
]
