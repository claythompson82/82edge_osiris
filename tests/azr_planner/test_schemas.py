"""Tests for AZR Planner Pydantic schemas."""

import pytest
from pydantic import TypeAdapter, ValidationError
from datetime import datetime, timezone # Added timezone

from azr_planner.schemas import (
    PlanningContext,
    TradeProposal,
    Leg,
    Instrument,
    Direction,
)

# Helper function to create a datetime object with UTC timezone
def dt_utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def test_planning_context_valid() -> None:
    """Test valid PlanningContext."""
    data = {
        "timestamp": "2023-01-01T12:00:00Z",
        "equity_curve": [1.0] * 30,
        "vol_surface": {"MES": 0.2},
        "risk_free_rate": 0.01,
    }
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(data)
    assert ctx.timestamp == dt_utc(2023, 1, 1, 12) # Use helper for timezone
    assert len(ctx.equity_curve) == 30
    assert ctx.vol_surface["MES"] == 0.2
    assert ctx.risk_free_rate == 0.01

    # Test round trip
    # Pydantic v2 dump_python by default excludes None values if exclude_none=True is not model_config
    # or if the field has a default of None. Let's be explicit about expected.
    expected_dump = {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equity_curve": [1.0] * 30,
        "vol_surface": {"MES": 0.2},
        "risk_free_rate": 0.01,
    }
    assert adapter.dump_python(ctx) == expected_dump


def test_planning_context_invalid() -> None:
    """Test invalid PlanningContext."""
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    # Missing required timestamp
    with pytest.raises(ValidationError):
        adapter.validate_python({"equity_curve": [1.0] * 30}) # equity_curve
    # Equity curve too short
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equity_curve": [1.0] * 29} # equity_curve
        )
    # Additional properties not allowed
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equity_curve": [1.0] * 30, # equity_curve
                "extra_field": "bad",
            }
        )

def test_trade_proposal_valid() -> None:
    """Test valid TradeProposal."""
    data = {
        "latent_risk": 0.5, # latent_risk
        "legs": [
            {
                "instrument": "MES",
                "direction": "LONG",
                "size": 10.5,
                "limit_price": 3000.0, # limit_price
            },
            {
                "instrument": "M2K",
                "direction": "SHORT",
                "size": 5.0,
                # limit_price is optional, defaults to None
            },
        ],
    }
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    proposal = adapter.validate_python(data)
    assert proposal.latent_risk == 0.5
    assert len(proposal.legs) == 2
    assert proposal.legs[0].instrument == Instrument.MES
    assert proposal.legs[0].direction == Direction.LONG
    assert proposal.legs[0].size == 10.5
    assert proposal.legs[0].limit_price == 3000.0
    assert proposal.legs[1].instrument == Instrument.M2K
    assert proposal.legs[1].direction == Direction.SHORT
    assert proposal.legs[1].size == 5.0
    assert proposal.legs[1].limit_price is None # Explicitly check for None

    # Test round trip
    expected_dump = {
        "latent_risk": 0.5,
        "legs": [
            {
                "instrument": Instrument.MES, # Enums will be dumped as their value by default
                "direction": Direction.LONG,
                "size": 10.5,
                "limit_price": 3000.0,
            },
            {
                "instrument": Instrument.M2K,
                "direction": Direction.SHORT,
                "size": 5.0,
                "limit_price": None, # Pydantic includes fields with default None if value is None
            },
        ],
    }
    # dump_python by default uses enum members, not their values, if not configured otherwise.
    # For exact JSON schema matching, one might use `dump_json` or `model_dump(mode='json')`.
    # Here, we compare with Python dict matching Pydantic's default python dump behavior.
    dumped_data = adapter.dump_python(proposal)

    # Convert enums in expected_dump to their values for comparison if needed,
    # or ensure comparison handles enum members.
    # Pydantic's dump_python usually keeps enums as members.
    assert dumped_data["latent_risk"] == expected_dump["latent_risk"]

    # Help mypy understand the structure of dumped_data["legs"]
    legs_data = dumped_data.get("legs")
    assert isinstance(legs_data, list)
    expected_legs_data = expected_dump.get("legs")
    assert isinstance(expected_legs_data, list)

    assert len(legs_data) == len(expected_legs_data)
    for i in range(len(legs_data)):
        leg_item = legs_data[i]
        expected_leg_item = expected_legs_data[i]
        assert isinstance(leg_item, dict)
        assert isinstance(expected_leg_item, dict)
        assert leg_item.get("instrument") == expected_leg_item.get("instrument")
        assert leg_item.get("direction") == expected_leg_item.get("direction")
        assert leg_item.get("size") == expected_leg_item.get("size")
        assert leg_item.get("limit_price") == expected_leg_item.get("limit_price")


def test_trade_proposal_invalid() -> None:
    """Test invalid TradeProposal."""
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    # Negative latent risk
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python({"latent_risk": -0.1, "legs": []}) # latent_risk
    # Invalid leg (missing size)
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python(
            {
                "latent_risk": 0.0, # latent_risk
                "legs": [{"instrument": "MES", "direction": "LONG"}],
            }
        )
    # Invalid leg (size not > 0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        adapter.validate_python(
            {
                "latent_risk": 0.0, # latent_risk
                "legs": [{"instrument": "MES", "direction": "LONG", "size": 0}],
            }
        )
    # Invalid leg (limit_price < 0)
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python(
            {
                "latent_risk": 0.0, # latent_risk
                "legs": [{"instrument": "MES", "direction": "LONG", "size": 1, "limit_price": -100}], # limit_price
            }
        )

def test_leg_instrument_enum() -> None:
    """Test Instrument enum validation within Leg."""
    with pytest.raises(ValidationError, match="Input should be 'MES', 'M2K', 'US_SECTOR_ETF' or 'ETH_OPT'"):
        # Pydantic V2 model_validate should handle string to enum conversion.
        # Direct instantiation might be stricter or MyPy might not infer it well.
        # Forcing an invalid string to test the validation message.
        Leg.model_validate({"instrument":"INVALID_INSTRUMENT", "direction":"LONG", "size":1.0})


def test_leg_direction_enum() -> None:
    """Test Direction enum validation within Leg."""
    with pytest.raises(ValidationError, match="Input should be 'LONG' or 'SHORT'"):
        Leg.model_validate({"instrument":"MES", "direction":"INVALID_DIRECTION", "size":1.0})

# Sample payload for round-trip testing as per acceptance criteria
SAMPLE_PAYLOAD_PLANNING_CONTEXT = {
    "timestamp": "2024-07-30T10:00:00Z",
    "equity_curve": [100.0, 101.0, 100.5, 102.0, 103.0] * 6, # 30 items
    "vol_surface": {"MES": 0.15, "M2K": 0.20},
    "risk_free_rate": 0.02
}

SAMPLE_PAYLOAD_TRADE_PROPOSAL = {
    "latent_risk": 0.05, # latent_risk
    "legs": [
        {"instrument": "MES", "direction": "LONG", "size": 10, "limit_price": 4500.0}, # limit_price
        {"instrument": "M2K", "direction": "SHORT", "size": 5} # No limit_price
    ]
}

def test_planning_context_roundtrip_sample() -> None:
    """Round-trip test for PlanningContext with sample payload."""
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(SAMPLE_PAYLOAD_PLANNING_CONTEXT)

    expected_after_validation_dump = {
        "timestamp": dt_utc(2024,7,30,10), # datetime object
        "equity_curve": SAMPLE_PAYLOAD_PLANNING_CONTEXT["equity_curve"],
        "vol_surface": SAMPLE_PAYLOAD_PLANNING_CONTEXT["vol_surface"],
        "risk_free_rate": SAMPLE_PAYLOAD_PLANNING_CONTEXT["risk_free_rate"]
    }
    assert adapter.dump_python(ctx) == expected_after_validation_dump

def test_trade_proposal_roundtrip_sample() -> None:
    """Round-trip test for TradeProposal with sample payload."""
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    proposal = adapter.validate_python(SAMPLE_PAYLOAD_TRADE_PROPOSAL)

    expected_after_validation_dump = {
        "latent_risk": SAMPLE_PAYLOAD_TRADE_PROPOSAL["latent_risk"],
        "legs": [
            {
                "instrument": Instrument.MES,
                "direction": Direction.LONG,
                "size": 10.0, # Pydantic might convert int to float for confloat
                "limit_price": 4500.0
            },
            {
                "instrument": Instrument.M2K,
                "direction": Direction.SHORT,
                "size": 5.0,
                "limit_price": None
            }
        ]
    }
    dumped_proposal = adapter.dump_python(proposal)
    assert dumped_proposal == expected_after_validation_dump

def test_instrument_us_sector_etf() -> None:
    """Test US_SECTOR_ETF as a valid instrument type."""
    leg_data = {"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0}
    leg = Leg(**leg_data) # type: ignore[arg-type]
    assert leg.instrument == Instrument.US_SECTOR_ETF

    adapter: TypeAdapter[Leg] = TypeAdapter(Leg)
    dumped = adapter.dump_python(leg)
    assert dumped["instrument"] == Instrument.US_SECTOR_ETF # Will be enum member

    validated_leg = adapter.validate_python({"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0})
    assert validated_leg.instrument == Instrument.US_SECTOR_ETF
