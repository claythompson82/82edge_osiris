"""Tests for AZR Planner Pydantic schemas."""

import pytest
from pydantic import TypeAdapter, ValidationError
from datetime import datetime, timezone

from azr_planner.schemas import (
    PlanningContext,
    TradeProposal, # This is the correct, renamed schema
    Leg,
    Instrument,
    Direction,
)

def dt_utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def test_planning_context_valid() -> None:
    """Test valid PlanningContext."""
    data = {
        "timestamp": "2023-01-01T12:00:00Z",
        "equityCurve": [1.0] * 30,
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(data)
    assert ctx.timestamp == dt_utc(2023, 1, 1, 12)
    assert ctx.equity_curve == data["equityCurve"]
    assert ctx.vol_surface == data["volSurface"]
    assert ctx.risk_free_rate == data["riskFreeRate"]

    expected_dump_field_names = {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equity_curve": [1.0] * 30,
        "vol_surface": {"MES": 0.2},
        "risk_free_rate": 0.01,
    }
    assert adapter.dump_python(ctx, by_alias=False) == expected_dump_field_names

    expected_dump_aliases = {
        "timestamp": dt_utc(2023, 1, 1, 12),
        "equityCurve": [1.0] * 30,
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }
    assert adapter.dump_python(ctx, by_alias=True) == expected_dump_aliases


def test_planning_context_invalid() -> None:
    """Test invalid PlanningContext."""
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)

    # Missing required timestamp
    with pytest.raises(ValidationError) as exc_info_ts:
        adapter.validate_python({"equityCurve": [1.0] * 30, "volSurface": {"MES":0.1}, "riskFreeRate": 0.01})
    assert any(err["type"] == "missing" and err["loc"] == ("timestamp",) for err in exc_info_ts.value.errors())

    # Missing volSurface (now required)
    with pytest.raises(ValidationError) as exc_info_vs:
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equityCurve": [1.0] * 30, "riskFreeRate": 0.01}
        )
    # print("DEBUG: test_planning_context_invalid - missing volSurface errors:", exc_info_vs.value.errors()) # DEBUG PRINT
    assert any(err["type"] == "missing" and err["loc"] == ("volSurface",) for err in exc_info_vs.value.errors()) # loc uses alias

    # Missing riskFreeRate (now required)
    with pytest.raises(ValidationError) as exc_info_rfr:
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equityCurve": [1.0] * 30, "volSurface": {"MES":0.1}} # Input uses alias volSurface
        )
    assert any(err["type"] == "missing" and err["loc"] == ("riskFreeRate",) for err in exc_info_rfr.value.errors()) # loc uses alias (consistent with volSurface)

    # Equity curve too short
    with pytest.raises(ValidationError, match="List should have at least 30 items"):
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equityCurve": [1.0] * 29,
             "volSurface": {"MES":0.1}, "riskFreeRate": 0.01}
        )

    # riskFreeRate out of bounds
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equityCurve": [1.0] * 30,
             "volSurface": {"MES":0.1}, "riskFreeRate": -0.1}
        )
    with pytest.raises(ValidationError, match="Input should be less than or equal to 0.2"):
        adapter.validate_python(
            {"timestamp": "2023-01-01T12:00:00Z", "equityCurve": [1.0] * 30,
             "volSurface": {"MES":0.1}, "riskFreeRate": 0.21}
        )

    # Additional properties not allowed
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        adapter.validate_python(
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "equityCurve": [1.0] * 30,
                "volSurface": {"MES":0.1},
                "riskFreeRate": 0.01,
                "extra_field": "bad",
            }
        )

def test_trade_proposal_valid() -> None:
    """Test valid TradeProposal."""
    data = {
        "action": "ENTER",
        "rationale": "Good signal",
        "latent_risk": 0.25,
        "confidence": 0.75,
        "legs": [
            {"instrument": "MES", "direction": "LONG", "size": 10.5, "limit_price": 3000.0,},
            {"instrument": "M2K", "direction": "SHORT", "size": 5.0,},
        ],
    }
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    proposal = adapter.validate_python(data)
    assert proposal.action == data["action"]
    assert proposal.rationale == data["rationale"]
    assert proposal.latent_risk == data["latent_risk"]
    assert proposal.confidence == data["confidence"]
    assert proposal.legs is not None
    legs_list = proposal.legs
    assert len(legs_list) == 2
    assert legs_list[0].instrument == Instrument.MES
    assert legs_list[0].direction == Direction.LONG
    assert legs_list[0].size == 10.5
    assert legs_list[0].limit_price == 3000.0
    assert legs_list[1].instrument == Instrument.M2K
    assert legs_list[1].direction == Direction.SHORT
    assert legs_list[1].size == 5.0
    assert legs_list[1].limit_price is None

    # Test dumping (by_alias=False is default for model_dump but explicit for TypeAdapter)
    # The TradeProposal model does not use aliases for these fields.
    expected_dump_data = {
        "action": "ENTER",
        "rationale": "Good signal",
        "latent_risk": 0.25,
        "confidence": 0.75,
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.5, "limit_price": 3000.0,},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None,},
        ],
    }
    assert adapter.dump_python(proposal, by_alias=False) == expected_dump_data
    # by_alias=True should produce the same as by_alias=False since no aliases are defined for these fields in TradeProposal
    assert adapter.dump_python(proposal, by_alias=True) == expected_dump_data


def test_trade_proposal_invalid() -> None:
    """Test invalid TradeProposal."""
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)

    # Test missing required fields
    required_fields_data = {"action": "E", "rationale": "R", "latent_risk": 0.1, "confidence": 0.9}
    for field_name_to_remove in required_fields_data: # Corrected indentation
        invalid_data = required_fields_data.copy()
        del invalid_data[field_name_to_remove]
        with pytest.raises(ValidationError) as exc_info:
             adapter.validate_python(invalid_data)
        # Check if the specific field is reported as missing
        assert any(
            err["type"] == "missing" and err["loc"] == (field_name_to_remove,)
            for err in exc_info.value.errors()
        ), f"Expected '{field_name_to_remove}' to be reported as missing"

    # Test latent_risk bounds
    invalid_lr_low = required_fields_data.copy()
    invalid_lr_low["latent_risk"] = -0.1
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python(invalid_lr_low)

    invalid_lr_high = required_fields_data.copy()
    invalid_lr_high["latent_risk"] = 1.1
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        adapter.validate_python(invalid_lr_high)

    # Test confidence bounds (similar checks as latent_risk, but for confidence field)
    invalid_conf_low = required_fields_data.copy()
    invalid_conf_low["confidence"] = -0.1
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python(invalid_conf_low)

    invalid_conf_high = required_fields_data.copy()
    invalid_conf_high["confidence"] = 1.1
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        adapter.validate_python(invalid_conf_high)

    # Test invalid leg (e.g., missing size)
    invalid_leg_data = required_fields_data.copy()
    invalid_leg_data["legs"] = [{"instrument": "MES", "direction": "LONG"}] # Missing size
    with pytest.raises(ValidationError, match="Field required.*legs.0.size"):
        adapter.validate_python(invalid_leg_data)

    # Test invalid leg size
    invalid_leg_size_data = required_fields_data.copy()
    invalid_leg_size_data["legs"] = [{"instrument": "MES", "direction": "LONG", "size": 0}]
    with pytest.raises(ValidationError, match="Input should be greater than 0.*legs.0.size"):
        adapter.validate_python(invalid_leg_size_data)

    # Test invalid leg limit_price
    invalid_leg_price_data = required_fields_data.copy()
    invalid_leg_price_data["legs"] = [{"instrument": "MES", "direction": "LONG", "size": 1, "limit_price": -100}]
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0.*legs.0.limit_price"):
        adapter.validate_python(invalid_leg_price_data)


def test_leg_instrument_enum() -> None:
    with pytest.raises(ValidationError, match="Input should be 'MES', 'M2K', 'US_SECTOR_ETF' or 'ETH_OPT'"):
        Leg.model_validate({"instrument":"INVALID_INSTRUMENT", "direction":"LONG", "size":1.0})

def test_leg_direction_enum() -> None:
    with pytest.raises(ValidationError, match="Input should be 'LONG' or 'SHORT'"):
        Leg.model_validate({"instrument":"MES", "direction":"INVALID_DIRECTION", "size":1.0})

SAMPLE_PAYLOAD_PLANNING_CONTEXT = {
    "timestamp": "2024-07-30T10:00:00Z",
    "equityCurve": [100.0, 101.0, 100.5, 102.0, 103.0] * 6,
    "volSurface": {"MES": 0.15, "M2K": 0.20},
    "riskFreeRate": 0.02
}

SAMPLE_PAYLOAD_TRADE_PROPOSAL = {
    "action": "ENTER", # Added
    "rationale": "Sample rationale", # Added
    "latent_risk": 0.05, # Changed from latentRisk
    "confidence": 0.95, # Added
    "legs": [
        {"instrument": "MES", "direction": "LONG", "size": 10.0, "limit_price": 4500.0}, # Ensured size is float
        {"instrument": "M2K", "direction": "SHORT", "size": 5.0} # Ensured size is float
    ]
}

def test_planning_context_roundtrip_sample() -> None:
    adapter: TypeAdapter[PlanningContext] = TypeAdapter(PlanningContext)
    ctx = adapter.validate_python(SAMPLE_PAYLOAD_PLANNING_CONTEXT)

    expected_dump_field_names = {
        "timestamp": dt_utc(2024,7,30,10),
        "equity_curve": SAMPLE_PAYLOAD_PLANNING_CONTEXT["equityCurve"],
        "vol_surface": SAMPLE_PAYLOAD_PLANNING_CONTEXT["volSurface"],
        "risk_free_rate": SAMPLE_PAYLOAD_PLANNING_CONTEXT["riskFreeRate"]
    }
    assert adapter.dump_python(ctx, by_alias=False) == expected_dump_field_names

    expected_dump_aliases = {
        "timestamp": dt_utc(2024,7,30,10),
        "equityCurve": SAMPLE_PAYLOAD_PLANNING_CONTEXT["equityCurve"],
        "volSurface": SAMPLE_PAYLOAD_PLANNING_CONTEXT["volSurface"],
        "riskFreeRate": SAMPLE_PAYLOAD_PLANNING_CONTEXT["riskFreeRate"]
    }
    assert adapter.dump_python(ctx, by_alias=True) == expected_dump_aliases


def test_trade_proposal_roundtrip_sample() -> None:
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    proposal = adapter.validate_python(SAMPLE_PAYLOAD_TRADE_PROPOSAL)

    expected_dump_field_names = {
        "action": SAMPLE_PAYLOAD_TRADE_PROPOSAL["action"],
        "rationale": SAMPLE_PAYLOAD_TRADE_PROPOSAL["rationale"],
        "latent_risk": SAMPLE_PAYLOAD_TRADE_PROPOSAL["latent_risk"],
        "confidence": SAMPLE_PAYLOAD_TRADE_PROPOSAL["confidence"],
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.0, "limit_price": 4500.0},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None}
        ]
    }
    assert adapter.dump_python(proposal, by_alias=False) == expected_dump_field_names
    # Since TradeProposal fields do not have aliases different from their names for these core fields,
    # by_alias=True should yield the same result.
    assert adapter.dump_python(proposal, by_alias=True) == expected_dump_field_names


def test_instrument_us_sector_etf() -> None:
    leg_data = {"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0}
    leg = Leg.model_validate(leg_data)
    assert leg.instrument == Instrument.US_SECTOR_ETF

    adapter: TypeAdapter[Leg] = TypeAdapter(Leg)
    dumped = adapter.dump_python(leg)
    assert dumped["instrument"] == Instrument.US_SECTOR_ETF

    validated_leg = Leg.model_validate({"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0})
    assert validated_leg.instrument == Instrument.US_SECTOR_ETF


# Tests for new TradeProposal model (formerly TradePlan)
def test_trade_proposal_schema_valid() -> None: # Renamed function
    """Test valid TradeProposal."""
    data = {
        "action": "HOLD",
        "rationale": "Market is neutral.",
        "latent_risk": 0.25,
        "confidence": 0.8,
        "legs": None
    }
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal) # Renamed Type
    proposal = adapter.validate_python(data) # Renamed variable
    assert proposal.action == "HOLD"
    assert proposal.rationale == "Market is neutral."
    assert proposal.latent_risk == 0.25
    assert proposal.confidence == 0.8
    assert proposal.legs is None

    expected_dump = data.copy()
    assert adapter.dump_python(proposal) == expected_dump

    leg_data_for_proposal = {"instrument": "MES", "direction": "LONG", "size": 1.0, "limit_price": 3000.0}
    data_with_leg = {
        "action": "ENTER",
        "rationale": "Signal detected.",
        "latent_risk": 0.1,
        "confidence": 0.95,
        "legs": [leg_data_for_proposal]
    }
    proposal_with_leg = adapter.validate_python(data_with_leg) # Renamed variable
    assert proposal_with_leg.action == "ENTER"
    assert proposal_with_leg.legs is not None
    # Create a new variable that mypy knows is not None
    legs_list = proposal_with_leg.legs
    assert len(legs_list) == 1
    assert legs_list[0].instrument == Instrument.MES

    expected_dump_with_leg = data_with_leg.copy()
    expected_dump_with_leg["legs"] = [Leg.model_validate(leg_data_for_proposal).model_dump(by_alias=False)]

    dumped_proposal_with_leg = adapter.dump_python(proposal_with_leg) # Renamed variable
    assert dumped_proposal_with_leg["action"] == expected_dump_with_leg["action"]
    assert dumped_proposal_with_leg["rationale"] == expected_dump_with_leg["rationale"]
    assert dumped_proposal_with_leg["latent_risk"] == expected_dump_with_leg["latent_risk"]
    assert dumped_proposal_with_leg["confidence"] == expected_dump_with_leg["confidence"]

    dumped_legs = dumped_proposal_with_leg.get("legs") # Use .get() for safety, though validated above
    assert isinstance(dumped_legs, list)
    assert len(dumped_legs) == 1
    assert dumped_legs[0] == Leg.model_validate(leg_data_for_proposal).model_dump(by_alias=False)


def test_trade_proposal_schema_invalid() -> None: # Renamed function
    """Test invalid TradeProposal."""
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal) # Renamed Type
    with pytest.raises(ValidationError, match="Field required"): # Rationale missing
        adapter.validate_python({"action": "ENTER", "latent_risk": 0.1, "confidence": 0.9})
    with pytest.raises(ValidationError, match="Field required"): # Action missing
        adapter.validate_python({"rationale": "Some reason", "latent_risk": 0.1, "confidence": 0.9})
    with pytest.raises(ValidationError, match="Field required"): # latent_risk missing
        adapter.validate_python({"action": "HOLD", "rationale": "Some reason", "confidence": 0.9})
    with pytest.raises(ValidationError, match="Field required"): # confidence missing
        adapter.validate_python({"action": "HOLD", "rationale": "Some reason", "latent_risk": 0.1})

    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        adapter.validate_python({"action": "HOLD", "rationale": "reason", "latent_risk": 0.1, "confidence": 1.1})
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python({"action": "HOLD", "rationale": "reason", "latent_risk": 0.1, "confidence": -0.1})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        adapter.validate_python({
            "action": "EXIT", "rationale": "Done", "latent_risk":0.1, "confidence":0.7, "extra": "bad"
        })

def test_trade_proposal_examples_in_field() -> None: # Renamed function
    """Test that the examples for action are correctly in the schema."""
    action_field = TradeProposal.model_fields["action"] # Renamed Type
    assert action_field.examples == ["HOLD", "ENTER", "EXIT"]
