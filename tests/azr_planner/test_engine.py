"""Tests for AZR Planner engine."""

import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st, assume
from datetime import datetime, timezone
from typing import Dict, Any # Added Dict, Any

from azr_planner.engine import generate_plan
from azr_planner.schemas import PlanningContext, TradePlan, Instrument, Direction, Leg

# Strategy to generate a dictionary that can be passed to PlanningContext.model_validate
st_planning_context_data = st.fixed_dictionaries({
    "timestamp": st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2040, 1, 1),
        timezones=st.just(timezone.utc)
    ),
    "equityCurve": st.lists(st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=30, max_size=30),
    "volSurface": st.fixed_dictionaries({
        Instrument.MES.value: st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False)
    }),
    "riskFreeRate": st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
})


@pytest.fixture
def sample_planning_context_data() -> Dict[str, Any]: # Changed dict to Dict[str, Any]
    """Provides a valid sample input dictionary for PlanningContext."""
    return {
        "timestamp": datetime.now(timezone.utc),
        "equityCurve": [100.0 + i for i in range(35)],
        "volSurface": {"MES": 0.15, "M2K": 0.20},
        "riskFreeRate": 0.02,
    }

from unittest.mock import patch, MagicMock # Added MagicMock

# ... (other imports)

@patch('azr_planner.engine.calculate_latent_risk')
def test_generate_plan_action_enter(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
    """Test generate_plan returns 'ENTER' when latent risk <= 0.3."""
    mock_calculate_latent_risk.return_value = 0.25

    ctx = PlanningContext.model_validate(sample_planning_context_data)
    trade_plan = generate_plan(ctx)

    assert isinstance(trade_plan, TradePlan)
    assert trade_plan.action == "ENTER"
    assert trade_plan.rationale == "Latent risk within threshold."
    assert trade_plan.latent_risk == 0.25
    assert trade_plan.confidence == 1.0
    assert trade_plan.legs is not None
    assert len(trade_plan.legs) == 1
    leg = trade_plan.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.LONG
    assert leg.size == 1.0
    assert leg.limit_price is None

@patch('azr_planner.engine.calculate_latent_risk')
def test_generate_plan_action_hold(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
    """Test generate_plan returns 'HOLD' when latent risk > 0.3."""
    mock_calculate_latent_risk.return_value = 0.35

    ctx = PlanningContext.model_validate(sample_planning_context_data)
    trade_plan = generate_plan(ctx)

    assert isinstance(trade_plan, TradePlan)
    assert trade_plan.action == "HOLD"
    assert trade_plan.rationale == "Latent risk exceeds threshold."
    assert trade_plan.latent_risk == 0.35
    assert trade_plan.confidence == 0.5
    assert trade_plan.legs is None


@given(data=st_planning_context_data)
def test_property_generate_plan_structure_and_logic(data: Dict[str, Any]) -> None: # Changed dict to Dict[str, Any]
    """
    Property test for generate_plan.
    Verifies the output structure and basic logic based on latent risk.
    Uses the actual placeholder calculate_latent_risk from math_utils.
    """
    try:
        ctx_input = PlanningContext.model_validate(data)
    except Exception:
        assume(False)
        return

    trade_plan = generate_plan(ctx_input)

    assert isinstance(trade_plan, TradePlan)
    assert trade_plan.action in ["ENTER", "HOLD"]
    assert isinstance(trade_plan.rationale, str)
    assert isinstance(trade_plan.latent_risk, float)
    assert 0.0 <= trade_plan.latent_risk <= 1.0
    assert isinstance(trade_plan.confidence, float)
    assert 0.0 <= trade_plan.confidence <= 1.0

    if trade_plan.action == "ENTER":
        assert trade_plan.rationale == "Latent risk within threshold."
        assert trade_plan.latent_risk <= 0.3
        assert trade_plan.confidence == 1.0
        assert trade_plan.legs is not None
        assert len(trade_plan.legs) == 1
        leg = trade_plan.legs[0]
        assert leg.instrument == Instrument.MES
        assert leg.direction == Direction.LONG
        assert leg.size == 1.0
    elif trade_plan.action == "HOLD":
        assert trade_plan.rationale == "Latent risk exceeds threshold."
        assert trade_plan.latent_risk > 0.3
        assert trade_plan.confidence == 0.5
        assert trade_plan.legs is None


def test_planning_context_instantiation_with_aliases() -> None:
    """Test PlanningContext can be instantiated using aliases and field names."""
    data_with_aliases = {
        "timestamp": datetime.now(timezone.utc),
        "equityCurve": [1.0] * 30,
        "volSurface": {"MES": 0.2},
        "riskFreeRate": 0.01,
    }
    ctx_aliases = PlanningContext.model_validate(data_with_aliases)
    assert ctx_aliases.equity_curve == data_with_aliases["equityCurve"]
    assert ctx_aliases.vol_surface == data_with_aliases["volSurface"]
    assert ctx_aliases.risk_free_rate == data_with_aliases["riskFreeRate"]

    data_with_field_names = {
        "timestamp": datetime.now(timezone.utc),
        "equity_curve": [1.0] * 31,
        "vol_surface": {"M2K": 0.3},
        "risk_free_rate": 0.02,
    }
    ctx_names = PlanningContext.model_validate(data_with_field_names)
    assert ctx_names.equity_curve == data_with_field_names["equity_curve"]
    assert ctx_names.vol_surface == data_with_field_names["vol_surface"]
    assert ctx_names.risk_free_rate == data_with_field_names["risk_free_rate"]
