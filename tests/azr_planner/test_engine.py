"""Tests for AZR Planner engine."""

import pytest
import math # Added import math
from unittest.mock import patch
from hypothesis import given, strategies as st, assume
from datetime import datetime, timezone
from typing import Dict, Any, List

from azr_planner.engine import generate_plan
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg # Changed TradePlan

# Strategy to generate a dictionary that can be passed to PlanningContext.model_validate
# Note: volSurface and riskFreeRate are still in PlanningContext but not used by new latent_risk
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
    """Test generate_plan returns 'ENTER' when latent risk < 0.30."""
    test_risk = 0.15
    mock_calculate_latent_risk.return_value = test_risk

    ctx = PlanningContext.model_validate(sample_planning_context_data)
    trade_proposal = generate_plan(ctx) # Renamed variable

    assert isinstance(trade_proposal, TradeProposal) # Check for TradeProposal
    assert trade_proposal.action == "ENTER"
    assert trade_proposal.rationale == "Latent risk is low, favorable for new positions."
    assert trade_proposal.latent_risk == test_risk
    assert trade_proposal.confidence == round(1.0 - test_risk, 3)
    assert trade_proposal.legs is not None
    assert len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.LONG
    assert leg.size == 1.0
    assert leg.limit_price is None

@patch('azr_planner.engine.calculate_latent_risk')
def test_generate_plan_action_hold(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
    """Test generate_plan returns 'HOLD' when 0.30 <= latent risk <= 0.70."""
    test_risk = 0.50
    mock_calculate_latent_risk.return_value = test_risk

    ctx = PlanningContext.model_validate(sample_planning_context_data)
    trade_proposal = generate_plan(ctx) # Renamed variable

    assert isinstance(trade_proposal, TradeProposal) # Check for TradeProposal
    assert trade_proposal.action == "HOLD"
    assert trade_proposal.rationale == "Latent risk is moderate, maintaining current positions."
    assert trade_proposal.latent_risk == test_risk
    assert trade_proposal.confidence == round(1.0 - test_risk, 3)
    assert trade_proposal.legs is None

@patch('azr_planner.engine.calculate_latent_risk')
def test_generate_plan_action_exit(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
    """Test generate_plan returns 'EXIT' when latent risk > 0.70."""
    test_risk = 0.85
    mock_calculate_latent_risk.return_value = test_risk

    ctx = PlanningContext.model_validate(sample_planning_context_data)
    trade_proposal = generate_plan(ctx) # Renamed variable

    assert isinstance(trade_proposal, TradeProposal) # Check for TradeProposal
    assert trade_proposal.action == "EXIT"
    assert trade_proposal.rationale == "Latent risk is high, reducing exposure."
    assert trade_proposal.latent_risk == test_risk
    assert trade_proposal.confidence == round(1.0 - test_risk, 3)
    assert trade_proposal.legs is not None
    assert len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.SHORT # Exit is SHORT
    assert leg.size == 1.0 # Stub size
    assert leg.limit_price is None


@given(data=st_planning_context_data)
def test_property_generate_plan_structure_and_logic(data: Dict[str, Any]) -> None:
    """
    Property test for generate_plan.
    Verifies the output structure and basic logic based on latent risk.
    Uses the actual (new) calculate_latent_risk from math_utils.
    """
    try:
        # The equityCurve from st_planning_context_data has min_size=30, max_size=30.
        # The new latent_risk function can handle shorter series, but some calculations
        # (like 30-day vol) are specific to that length.
        # For this property test, ensuring len >= 30 is good for full calculation path.
        assume(len(data["equityCurve"]) >= 30)
        ctx_input = PlanningContext.model_validate(data)
    except Exception:
        assume(False) # Skip invalid context data
        return

    trade_proposal = generate_plan(ctx_input)

    assert isinstance(trade_proposal, TradeProposal)
    assert trade_proposal.action in ["ENTER", "HOLD", "EXIT"] # Added EXIT
    assert isinstance(trade_proposal.rationale, str)
    assert isinstance(trade_proposal.latent_risk, float)
    assert 0.0 <= trade_proposal.latent_risk <= 1.0, "Latent risk out of bounds"

    expected_confidence = round(1.0 - trade_proposal.latent_risk, 3)
    assert math.isclose(trade_proposal.confidence, expected_confidence), \
        f"Confidence {trade_proposal.confidence} not matching 1-risk ({expected_confidence})"
    assert 0.0 <= trade_proposal.confidence <= 1.0, "Confidence out of bounds"

    current_latent_risk = trade_proposal.latent_risk

    if current_latent_risk < 0.30:
        assert trade_proposal.action == "ENTER"
        assert trade_proposal.rationale == "Latent risk is low, favorable for new positions."
        assert trade_proposal.legs is not None
        assert len(trade_proposal.legs) == 1
        leg = trade_proposal.legs[0]
        assert leg.instrument == Instrument.MES
        assert leg.direction == Direction.LONG
        assert leg.size == 1.0
    elif current_latent_risk <= 0.70: # 0.30 <= latent_risk <= 0.70
        assert trade_proposal.action == "HOLD"
        assert trade_proposal.rationale == "Latent risk is moderate, maintaining current positions."
        assert trade_proposal.legs is None
    else: # latent_risk > 0.70
        assert trade_proposal.action == "EXIT"
        assert trade_proposal.rationale == "Latent risk is high, reducing exposure."
        assert trade_proposal.legs is not None
        assert len(trade_proposal.legs) == 1
        leg = trade_proposal.legs[0]
        assert leg.instrument == Instrument.MES
        assert leg.direction == Direction.SHORT
        assert leg.size == 1.0


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
