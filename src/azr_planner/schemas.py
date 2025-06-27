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
