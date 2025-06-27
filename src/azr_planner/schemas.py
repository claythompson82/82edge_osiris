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
        Field(alias="equityCurve", min_length=30)
    ]
    vol_surface: Annotated[ # Now required, no default_factory
        Dict[str, float],
        Field(alias="volSurface")
    ]
    risk_free_rate: Annotated[ # Now required, not Optional, no default
        float,
        Field(alias="riskFreeRate", ge=0.0, le=0.2)
    ]

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )


class TradeProposal(BaseModel):
    """Output trade proposal - to be kept as is."""
    latent_risk: Annotated[float, Field(ge=0, alias="latentRisk")]
    legs: List[Leg]

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

class TradePlan(BaseModel):
    """Simplified trade plan for current stub engine."""
    action: str = Field(..., examples=["HOLD", "ENTER", "EXIT"])
    rationale: str
    latent_risk: float  # Added: The computed latent risk that led to this plan
    confidence: Annotated[float, Field(ge=0, le=1)]  # Added: Confidence in this plan
    legs: Optional[List[Leg]] = Field(default=None)  # Added: Optional list of legs for the plan

    model_config = ConfigDict(
        populate_by_name=True, # Allow population by alias if fields get them later
        extra="forbid",
        str_strip_whitespace=True,
    )
