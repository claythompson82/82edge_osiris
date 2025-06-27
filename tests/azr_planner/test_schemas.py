"""Tests for AZR Planner Pydantic schemas."""

from datetime import datetime, timezone
from typing import List

import pytest
from pydantic import TypeAdapter, ValidationError

from azr_planner.schemas import (
    PlanningContext,
    TradeProposal,
    Leg,
    Instrument,
    Direction,
)


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def dt_utc(y: int, m: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------- #
# PlanningContext
# ----------------------------------------------------------------------------- #
def test_planning_context_valid() -> None:
    data = {
        "timestamp": "2023-01-01T12:00:00Z",
        "equityCurve": [1.0] * 30,
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(data)

    # round-trip (field names)
    assert adapter.dump_python(
        ctx, by_alias=False
    ) == {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equity_curve": [1.0] * 30,
        "vol_surface": {"MES": 0.2},
        "risk_free_rate": 0.01,
    }

    # round-trip (aliases)
    assert adapter.dump_python(
        ctx, by_alias=True
    ) == {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equityCurve": [1.0] * 30,
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }


def test_planning_context_invalid() -> None:
    adapter = TypeAdapter(PlanningContext)

    # missing timestamp
    with pytest.raises(ValidationError):
        adapter.validate_python({"equityCurve": [1.0] * 30, "volSurface": {"MES": 0.1}, "riskFreeRate": 0.01})

    # equity curve too short
    with pytest.raises(ValidationError, match="at least 30 items"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equityCurve": [1.0] * 29,
                "volSurface": {"MES": 0.1},
                "riskFreeRate": 0.01,
            }
        )


# ----------------------------------------------------------------------------- #
# TradeProposal
# ----------------------------------------------------------------------------- #
VALID_TP_BASE = {
    "action": "ENTER",
    "rationale": "stub",
    "latent_risk": 0.25,
    "confidence": 0.9,
}


def test_trade_proposal_valid() -> None:
    data = {
        **VALID_TP_BASE,
        "legs": [
            {"instrument": "MES", "direction": "LONG", "size": 10.0, "limit_price": 3000.0},
            {"instrument": "M2K", "direction": "SHORT", "size": 5.0},
        ],
    }
    adapter = TypeAdapter(TradeProposal)
    tp = adapter.validate_python(data)
    assert tp.legs and tp.legs[0].instrument == Instrument.MES


def test_trade_proposal_bounds() -> None:
    adapter = TypeAdapter(TradeProposal)

    # latent_risk / confidence out of bounds
    for field, bad in [("latent_risk", -0.1), ("latent_risk", 1.1), ("confidence", -0.1), ("confidence", 1.1)]:
        bad_tp = {**VALID_TP_BASE, field: bad}
        with pytest.raises(ValidationError):
            adapter.validate_python(bad_tp)

    # legs validation – size missing
    bad_size = {**VALID_TP_BASE, "legs": [{"instrument": "MES", "direction": "LONG"}]}
    with pytest.raises(ValidationError, match=r"legs\.0\.size"):
        adapter.validate_python(bad_size)

    # legs validation – size <= 0
    size_zero = {**VALID_TP_BASE, "legs": [{"instrument": "MES", "direction": "LONG", "size": 0}]}
    with pytest.raises(ValidationError, match=r"legs\.0\.size"):
        adapter.validate_python(size_zero)

    # legs validation – negative limit price
    bad_price = {
        **VALID_TP_BASE,
        "legs": [{"instrument": "MES", "direction": "LONG", "size": 1, "limit_price": -100}],
    }
    with pytest.raises(ValidationError, match=r"legs\.0\.limit_price.*greater than or equal to 0"):
        adapter.validate_python(bad_price)


# ----------------------------------------------------------------------------- #
# Enum helpers
# ----------------------------------------------------------------------------- #
def test_leg_instrument_enum() -> None:
    with pytest.raises(ValidationError):
        Leg.model_validate({"instrument": "INVALID", "direction": "LONG", "size": 1.0})


def test_leg_direction_enum() -> None:
    with pytest.raises(ValidationError):
        Leg.model_validate({"instrument": "MES", "direction": "INVALID", "size": 1.0})
