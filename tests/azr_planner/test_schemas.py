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
    # New required field: dailyHistoryHLC (min_length=15)
    # Optional fields: dailyVolume, currentPositions
    valid_hlc = [(100.0, 99.0, 99.5)] * 15
    data = {
        "timestamp": "2023-01-01T12:00:00Z",
        "equityCurve": [1.0] * 30,
        "dailyHistoryHLC": valid_hlc, # New required field
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
        # dailyVolume and currentPositions are optional, so not included here for minimal valid case
    }
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(data)

    # round-trip (field names) - check new fields too
    dumped_data_fields = adapter.dump_python(ctx, by_alias=False)
    assert dumped_data_fields == {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equity_curve": [1.0] * 30,
        "daily_history_hlc": valid_hlc,
        "daily_volume": None, # Default for optional
        "current_positions": None, # Default for optional
        "vol_surface": {"MES": 0.2},
        "risk_free_rate": 0.01,
    }

    # round-trip (aliases) - check new fields too
    dumped_data_aliases = adapter.dump_python(ctx, by_alias=True)
    assert dumped_data_aliases == {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equityCurve": [1.0] * 30,
        "dailyHistoryHLC": valid_hlc,
        "dailyVolume": None, # Default for optional
        "currentPositions": None, # Default for optional
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }


def test_planning_context_invalid() -> None:
    adapter = TypeAdapter(PlanningContext)
    valid_hlc = [(100.0, 99.0, 99.5)] * 15

    # missing timestamp
    with pytest.raises(ValidationError, match="timestamp"):
        adapter.validate_python({
            "equityCurve": [1.0] * 30,
            "dailyHistoryHLC": valid_hlc,
            "volSurface": {"MES": 0.1},
            "riskFreeRate": 0.01
        })

    # equity curve too short
    # Pydantic v2 error message for too_short is "List should have at least {min_length} items after validation, not {actual_length}"
    with pytest.raises(ValidationError, match=r"equityCurve\W*List should have at least 30 items after validation, not 29"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equityCurve": [1.0] * 29,
                "dailyHistoryHLC": valid_hlc,
                "volSurface": {"MES": 0.1},
                "riskFreeRate": 0.01,
            }
        )

    # dailyHistoryHLC missing
    # Pydantic v2 error message for missing field: "Field required"
    with pytest.raises(ValidationError, match=r"dailyHistoryHLC\W*Field required"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equityCurve": [1.0] * 30,
                # dailyHistoryHLC is missing
                "volSurface": {"MES": 0.1},
                "riskFreeRate": 0.01,
            }
        )

    # dailyHistoryHLC too short
    with pytest.raises(ValidationError, match=r"dailyHistoryHLC\W*List should have at least 15 items after validation, not 14"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equityCurve": [1.0] * 30,
                "dailyHistoryHLC": [(100.0, 99.0, 99.5)] * 14, # Too short
                "volSurface": {"MES": 0.1},
                "riskFreeRate": 0.01,
            }
        )

    # dailyVolume provided but length doesn't match dailyHistoryHLC (Pydantic v2 default doesn't cross-validate field lengths unless explicitly coded in a validator)
    # This specific validation (length matching) is not auto-added by Pydantic for Optional[List] based on another List's length.
    # Such a check would require a model_validator.
    # The schema has min_length for dailyVolume if provided, but not a cross-field length check.
    # So, this test is more about ensuring it passes if volume is None or has its own min_length.
    # If dailyVolume is provided with wrong number of items, it should pass Pydantic validation based on current schema,
    # but our engine logic might fail later if it assumes matching lengths.
    # For now, testing valid case with optional volume:
    data_with_volume = {
        "timestamp": "2023-01-01T12:00:00Z",
        "equityCurve": [1.0] * 30,
        "dailyHistoryHLC": valid_hlc,
        "dailyVolume": [1000.0] * 15, # Correct length
        "volSurface": {"MES": 0.1},
        "riskFreeRate": 0.01,
    }
    ctx_vol = adapter.validate_python(data_with_volume)
    assert ctx_vol.daily_volume == [1000.0] * 15

    data_with_short_volume = { # This should fail if min_length on dailyVolume is enforced when not None
        "timestamp": "2023-01-01T12:00:00Z",
        "equityCurve": [1.0] * 30,
        "dailyHistoryHLC": valid_hlc, # length 15
        "dailyVolume": [1000.0] * 10, # length 10, schema dailyVolume has min_length=15
        "volSurface": {"MES": 0.1},
        "riskFreeRate": 0.01,
    }
    with pytest.raises(ValidationError, match=r"dailyVolume\W*List should have at least 15 items after validation, not 10"):
        adapter.validate_python(data_with_short_volume)


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
    with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
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
