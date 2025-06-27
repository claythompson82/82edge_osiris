"""Tests for AZR Planner Pydantic schemas."""

import pytest
from pydantic import TypeAdapter, ValidationError
from datetime import datetime, timezone

from azr_planner.schemas import (
    PlanningContext,
    TradeProposal,
    TradePlan,
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
        "latentRisk": 0.5,
        "legs": [
            {"instrument": "MES", "direction": "LONG", "size": 10.5, "limit_price": 3000.0,},
            {"instrument": "M2K", "direction": "SHORT", "size": 5.0,},
        ],
    }
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    proposal = adapter.validate_python(data)
    assert proposal.latent_risk == data["latentRisk"]
    assert len(proposal.legs) == 2
    assert proposal.legs[0].instrument == Instrument.MES
    assert proposal.legs[0].direction == Direction.LONG
    assert proposal.legs[0].size == 10.5
    assert proposal.legs[0].limit_price == 3000.0
    assert proposal.legs[1].instrument == Instrument.M2K
    assert proposal.legs[1].direction == Direction.SHORT
    assert proposal.legs[1].size == 5.0
    assert proposal.legs[1].limit_price is None

    expected_dump_field_names = {
        "latent_risk": 0.5,
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.5, "limit_price": 3000.0,},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None,},
        ],
    }
    assert adapter.dump_python(proposal, by_alias=False) == expected_dump_field_names

    expected_dump_aliases = {
        "latentRisk": 0.5,
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.5, "limit_price": 3000.0,},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None,},
        ],
    }
    dumped_by_alias = adapter.dump_python(proposal, by_alias=True)
    assert dumped_by_alias["latentRisk"] == expected_dump_aliases["latentRisk"]
    # Comparing legs directly as their structure is expected to be the same here
    assert dumped_by_alias["legs"] == expected_dump_aliases["legs"]


def test_trade_proposal_invalid() -> None:
    """Test invalid TradeProposal."""
    adapter: TypeAdapter[TradeProposal] = TypeAdapter(TradeProposal)
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python({"latentRisk": -0.1, "legs": []})
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python(
            {"latentRisk": 0.0, "legs": [{"instrument": "MES", "direction": "LONG"}]}
        )
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        adapter.validate_python(
            {"latentRisk": 0.0, "legs": [{"instrument": "MES", "direction": "LONG", "size": 0}]}
        )
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python(
            {"latentRisk": 0.0, "legs": [{"instrument": "MES", "direction": "LONG", "size": 1, "limit_price": -100}]}
        )

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
    "latentRisk": 0.05,
    "legs": [
        {"instrument": "MES", "direction": "LONG", "size": 10, "limit_price": 4500.0},
        {"instrument": "M2K", "direction": "SHORT", "size": 5}
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
        "latent_risk": SAMPLE_PAYLOAD_TRADE_PROPOSAL["latentRisk"],
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.0, "limit_price": 4500.0},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None}
        ]
    }
    assert adapter.dump_python(proposal, by_alias=False) == expected_dump_field_names

    expected_dump_aliases = {
        "latentRisk": SAMPLE_PAYLOAD_TRADE_PROPOSAL["latentRisk"],
        "legs": [
            {"instrument": Instrument.MES, "direction": Direction.LONG, "size": 10.0, "limit_price": 4500.0},
            {"instrument": Instrument.M2K, "direction": Direction.SHORT, "size": 5.0, "limit_price": None}
        ]
    }
    assert adapter.dump_python(proposal, by_alias=True) == expected_dump_aliases


def test_instrument_us_sector_etf() -> None:
    leg_data = {"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0}
    leg = Leg.model_validate(leg_data)
    assert leg.instrument == Instrument.US_SECTOR_ETF

    adapter: TypeAdapter[Leg] = TypeAdapter(Leg)
    dumped = adapter.dump_python(leg)
    assert dumped["instrument"] == Instrument.US_SECTOR_ETF

    validated_leg = Leg.model_validate({"instrument": "US_SECTOR_ETF", "direction": "LONG", "size": 1.0})
    assert validated_leg.instrument == Instrument.US_SECTOR_ETF


# Tests for new TradePlan model
def test_trade_plan_valid() -> None:
    """Test valid TradePlan."""
    # Now includes new fields: latent_risk, confidence, legs
    data = {
        "action": "HOLD",
        "rationale": "Market is neutral.",
        "latent_risk": 0.25,
        "confidence": 0.8,
        "legs": None # Explicitly None for a HOLD action perhaps
    }
    adapter: TypeAdapter[TradePlan] = TypeAdapter(TradePlan)
    plan = adapter.validate_python(data)
    assert plan.action == "HOLD"
    assert plan.rationale == "Market is neutral."
    assert plan.latent_risk == 0.25
    assert plan.confidence == 0.8
    assert plan.legs is None

    expected_dump = data.copy() # dump should match input if no complex types/aliases
    assert adapter.dump_python(plan) == expected_dump

    # Test with a leg
    leg_data_for_plan = {"instrument": "MES", "direction": "LONG", "size": 1.0, "limit_price": 3000.0}
    data_with_leg = {
        "action": "ENTER",
        "rationale": "Signal detected.",
        "latent_risk": 0.1,
        "confidence": 0.95,
        "legs": [leg_data_for_plan]
    }
    plan_with_leg = adapter.validate_python(data_with_leg)
    assert plan_with_leg.action == "ENTER"
    assert plan_with_leg.legs is not None # Assertion to satisfy mypy before len()
    assert len(plan_with_leg.legs) == 1
    assert plan_with_leg.legs[0].instrument == Instrument.MES

    # Dump for plan_with_leg
    # Need to convert Leg in expected_dump to its dumped form (enum members)
    expected_dump_with_leg = data_with_leg.copy()
    expected_dump_with_leg["legs"] = [Leg.model_validate(leg_data_for_plan).model_dump(by_alias=False)] # Use model_dump for consistent Leg representation

    # adapter.dump_python(plan_with_leg) will have Leg object, not dict.
    # For direct comparison, either dump the leg in expected, or compare field by field.
    dumped_plan_with_leg = adapter.dump_python(plan_with_leg)
    assert dumped_plan_with_leg["action"] == expected_dump_with_leg["action"]
    assert dumped_plan_with_leg["rationale"] == expected_dump_with_leg["rationale"]
    assert dumped_plan_with_leg["latent_risk"] == expected_dump_with_leg["latent_risk"]
    assert dumped_plan_with_leg["confidence"] == expected_dump_with_leg["confidence"]
    assert isinstance(dumped_plan_with_leg["legs"], list)
    assert len(dumped_plan_with_leg["legs"]) == 1
    # Compare the dumped leg with the expected dumped leg
    # This assumes Leg model_dump(by_alias=False) matches what TypeAdapter(TradePlan).dump_python() produces for nested legs.
    # TypeAdapter.dump_python typically returns dicts for nested models if the parent is dumped to python types.
    assert dumped_plan_with_leg["legs"][0] == Leg.model_validate(leg_data_for_plan).model_dump(by_alias=False)


def test_trade_plan_invalid() -> None:
    """Test invalid TradePlan."""
    adapter: TypeAdapter[TradePlan] = TypeAdapter(TradePlan)
    # Missing rationale (latent_risk, confidence also missing)
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python({"action": "ENTER", "latent_risk": 0.1, "confidence": 0.9})
    # Missing action
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python({"rationale": "Some reason", "latent_risk": 0.1, "confidence": 0.9})
    # Missing latent_risk
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python({"action": "HOLD", "rationale": "Some reason", "confidence": 0.9})
    # Missing confidence
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_python({"action": "HOLD", "rationale": "Some reason", "latent_risk": 0.1})
    # Confidence out of bounds
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        adapter.validate_python({"action": "HOLD", "rationale": "reason", "latent_risk": 0.1, "confidence": 1.1})
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        adapter.validate_python({"action": "HOLD", "rationale": "reason", "latent_risk": 0.1, "confidence": -0.1})

    # Extra field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        adapter.validate_python({
            "action": "EXIT", "rationale": "Done", "latent_risk":0.1, "confidence":0.7, "extra": "bad"
        })

def test_trade_plan_examples_in_field() -> None:
    """Test that the examples for action are correctly in the schema."""
    action_field = TradePlan.model_fields["action"]
    assert action_field.examples == ["HOLD", "ENTER", "EXIT"]
