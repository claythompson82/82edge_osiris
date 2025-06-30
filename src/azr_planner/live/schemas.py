from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

class LiveConfig(BaseModel):
    """Configuration for the live paper trading session."""
    symbol: str = Field(description="Trading symbol (e.g., 'MESU24', 'EUR/USD'). For simplicity, assumes a single symbol for the session.")
    initial_equity: float = Field(description="Initial equity for the paper trading session.", ge=0)

    # RiskGateConfig related fields - these will be used to construct a RiskGateConfig
    max_risk_per_trade_pct: float = Field(
        default=0.01, # 1%
        description="Maximum percentage of account equity to risk per trade.",
        ge=0, le=1.0
    )
    max_drawdown_pct_account: float = Field(
        default=0.10, # 10%
        description="Maximum percentage drawdown allowed for the account before ceasing new trades.",
        ge=0, le=1.0
    )

    model_config = ConfigDict(frozen=True)


class LivePosition(BaseModel):
    """Represents an open position in the live paper trading blotter."""
    instrument: str = Field(description="Instrument identifier (e.g., 'MES', 'ES').") # Corresponds to Instrument enum value typically
    quantity: float = Field(description="Number of contracts/shares. Positive for long, negative for short.")
    average_entry_price: float = Field(description="Average price at which the position was entered.", ge=0)
    unrealized_pnl: float = Field(default=0.0, description="Current mark-to-market profit or loss for this open position.")
    realized_pnl_session: float = Field(default=0.0, description="Profit or loss realized from this instrument during the current session.")

    model_config = ConfigDict(frozen=True)


class LivePnl(BaseModel):
    """Represents the overall P&L state of the live paper trading session."""
    timestamp: datetime.datetime = Field(description="Timestamp of when this P&L snapshot was generated.")
    total_equity: float = Field(description="Current total equity (initial equity + realized P&L + unrealized P&L).")
    session_realized_pnl: float = Field(description="Total P&L realized during this session across all trades.")
    session_unrealized_pnl: float = Field(description="Total unrealized P&L from all open positions, marked to current market prices.")
    open_positions_count: int = Field(description="Number of currently open positions.", ge=0)

    model_config = ConfigDict(frozen=True)
