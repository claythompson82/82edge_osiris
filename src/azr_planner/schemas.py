"""Pydantic models for AZR Planner API."""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Annotated

from pydantic import BaseModel, Field, ConfigDict


class Instrument(str, Enum):
    """Instrument types."""
    MES = "MES"
    M2K = "M2K"
    US_SECTOR_ETF = "US_SECTOR_ETF"
    ETH_OPT = "ETH_OPT"


class Direction(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class Leg(BaseModel):
    """Represents a single leg of a trade."""
    instrument: Instrument
    direction: Direction
    size: Annotated[float, Field(gt=0)]
    limit_price: Optional[Annotated[float, Field(ge=0)]] = Field(default=None)

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

# --- AZR-14: P&L Simulation Schemas ---
import datetime as dt # For date type hint

class DailyFill(BaseModel):
    """Represents a single trade fill for a day."""
    timestamp: dt.datetime = Field(description="Timestamp of the fill execution.")
    instrument: Instrument = Field(description="Instrument that was filled.")
    direction: Direction = Field(description="Direction of the fill (BUY/SELL). Note: This should align with trade directions, not position directions like LONG/SHORT.") # BUY or SELL
    qty: Annotated[float, Field(gt=0, description="Quantity filled (always positive).")]
    price: Annotated[float, Field(ge=0, description="Execution price of the fill.")]

    model_config = ConfigDict(populate_by_name=True, frozen=True)


class DailyPNLReport(BaseModel):
    """Report detailing daily Profit & Loss and portfolio state."""
    date: dt.date = Field(description="The date for which this P&L report is generated (typically EOD).")
    realized_pnl: float = Field(default=0.0, description="Profit and Loss realized on this day from closing trades.")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized Profit and Loss on open positions at EOD.")
    net_position_value: float = Field(default=0.0, description="Total market value of all open positions at EOD.")
    cash: float = Field(description="Cash balance at EOD after all fills and P&L.")
    total_equity: float = Field(description="Total portfolio equity at EOD (Cash + Net Position Value).")
    gross_exposure: float = Field(default=0.0, description="Sum of absolute market values of all positions.")
    net_exposure: float = Field(default=0.0, description="Net market value of all positions (longs - shorts value).")
    cumulative_max_equity: float = Field(description="Peak total equity observed up to and including this day.")
    current_drawdown: Annotated[float, Field(ge=0.0, le=1.0)] = Field(description="Current drawdown from peak equity (0.0 to 1.0).")
    equity_curve_points: List[float] = Field(description="Time series of EOD total_equity values, including the current day's.")

    model_config = ConfigDict(populate_by_name=True, frozen=True)


class PlanningContext(BaseModel):
    """Input context for trade planning."""
    timestamp: datetime
    equity_curve: Annotated[
        List[float],
        Field(alias="equityCurve", min_length=30, description="Time series of equity values.")
    ]
    # New fields for detailed price/volume data and current state
    daily_history_hlc: Annotated[
        List[tuple[float, float, float]],
        Field(
            alias="dailyHistoryHLC",
            min_length=15, # Minimum for a reasonable ATR(14) calculation (14 TRs -> 15 days)
            description="List of (High, Low, Close) tuples for each period in chronological order. Required for ATR and potentially signal generation."
        )
    ]
    daily_volume: Annotated[
        Optional[List[float]],
        Field(
            default=None,
            alias="dailyVolume",
            min_length=15, # Should match daily_history_hlc if provided
            description="Optional list of trading volumes per period, corresponding to daily_history_hlc. Used for volume-adjusted signals."
        )
    ]
    current_positions: Annotated[
        Optional[List[Leg]],
        Field(
            default=None,
            alias="currentPositions",
            description="Optional list of currently held positions (Legs). Used for determining EXIT actions or adjusting new position sizes."
        )
    ]
    # Existing fields
    vol_surface: Annotated[
        Dict[str, float],
        Field(alias="volSurface", description="Volatility surface data.")
    ]
    risk_free_rate: Annotated[
        float,
        Field(alias="riskFreeRate", ge=0.0, le=0.2, description="Current risk-free rate.")
    ]
    # New fields for Bayesian confidence input
    n_successes: int = Field(0, alias="nSuccesses", ge=0, description="Number of historical successes for confidence calibration.")
    n_failures: int = Field(0, alias="nFailures", ge=0, description="Number of historical failures for confidence calibration.")


    model_config = ConfigDict(
        populate_by_name=True, # Allows using either field name or alias
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

# The old TradeProposal is removed.
# TradePlan is renamed to TradeProposal and will serve as the main output schema.

class TradeProposal(BaseModel):
    """
    Output trade proposal from the AZR Planner engine.
    Contains the proposed action, risk metrics, and associated trade legs.
    """
    action: str = Field(..., examples=["HOLD", "ENTER", "EXIT"], description="The proposed trading action.")
    rationale: str = Field(..., description="Justification for the proposed action.")

    # Existing risk/confidence metrics - latent_risk might be deprecated or become optional
    # if the new engine logic fully replaces it. For now, keeping it.
    # The problem statement implies latent_risk is still part of a property test.
    latent_risk: Optional[float] = Field(default=None, ge=0, le=1, description="The calculated latent risk score (0-1), if applicable.")
    confidence: Annotated[float, Field(ge=0, le=1, description="Confidence in this trade proposal (0-1), based on new model.")]

    legs: Optional[List[Leg]] = Field(default=None, description="List of trade legs for the proposal. Typically None if action is HOLD.")

    # New optional fields for additional context from the new engine
    signal_value: Optional[float] = Field(
        default=None,
        description="Value of the generated trading signal (e.g., from EMA crossover, vol-adjusted)."
    )
    atr_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Calculated Average True Range (ATR) at the time of proposal."
    )
    kelly_fraction_value: Optional[float] = Field(
        default=None,
        ge=0, # Kelly fraction should be non-negative for sizing
        description="Calculated Kelly fraction (raw value, before any caps)."
    )
    target_position_size: Optional[float] = Field(
        default=None,
        ge=0, # Position size should be non-negative
        description="Target position size determined by the engine (e.g., after Kelly/ATR capping)."
    )
    expected_pnl_per_unit: Optional[float] = Field( # Renamed from expected_pnl to be more specific
        default=None,
        alias="expectedPnlPerUnit",
        description="Expected Profit and Loss (PnL) per unit of trade, if calculable by the model."
    )

    model_config = ConfigDict(
        populate_by_name=True, # Allow population by alias
        extra="forbid",
        str_strip_whitespace=True,
    )
