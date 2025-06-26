"""Pydantic models for AZR Planner API."""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Annotated

from pydantic import BaseModel, Field


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

    model_config = {"extra": "forbid"}


class PlanningContext(BaseModel):
    """Input context for trade planning."""
    timestamp: datetime
    equity_curve: Annotated[List[float], Field(min_length=30)]
    vol_surface: Dict[str, float] = Field(default_factory=dict)
    risk_free_rate: Optional[float] = Field(default=None)

    model_config = {"extra": "forbid"}


class TradeProposal(BaseModel):
    """Output trade proposal."""
    latent_risk: Annotated[float, Field(ge=0)]
    legs: List[Leg]

    model_config = {"extra": "forbid"}
