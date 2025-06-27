"""Tests for AZR Planner engine."""

import pytest
import math # Added import math
from unittest.mock import patch
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import DrawFn # For type hinting draw in composite strategies
from datetime import datetime, timezone
from typing import Dict, Any, List

from azr_planner.engine import generate_plan
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg # Changed TradePlan

from azr_planner.engine import ASSUMED_EMA_LONG_PERIOD, ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK # Import constants

# --- Helper function to generate HLC data ---
def _generate_hlc_data(num_periods: int, start_price: float = 100.0, daily_change: float = 0.1, spread: float = 0.5) -> List[tuple[float, float, float]]:
    data = []
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + abs(daily_change * math.sin(i*0.1)) # Add some variation
        low = current_close - spread - abs(daily_change * math.cos(i*0.1))
        close = (high + low) / 2 + (math.sin(i*0.5) * spread*0.1) # Add noise
        # Ensure H >= L and C is within H-L, H > L
        low = min(low, high - 0.01) # Ensure low is strictly less than high
        close = max(min(close, high), low)

        data.append((round(high,2), round(low,2), round(close,2)))
        current_close = close
    return data

# --- Updated Hypothesis strategy for PlanningContext ---
# Minimum data points for daily_history_hlc and daily_volume,
# considering ATR(14) needs 15 HLC, EMAs might need more, Kelly lookbacks might need more.
MIN_HISTORY_POINTS = max(ASSUMED_EMA_LONG_PERIOD, ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK, 15) + 5 # Add buffer

# More direct HLC tuple generation to avoid excessive filtering
@st.composite
def st_single_hlc_tuple(draw: DrawFn) -> tuple[float, float, float]:
    # Draw three initial floats
    f1 = draw(st.floats(min_value=1.0, max_value=1999.0)) # Ensure high can be higher
    f2 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f3 = draw(st.floats(min_value=1.0, max_value=1999.0))

    # Determine High, Low, Close
    s = sorted([f1, f2, f3])
    low, c_candidate, high = s[0], s[1], s[2]

    # Ensure High > Low
    if high <= low:
        high = low + draw(st.floats(min_value=0.1, max_value=1.0)) # Ensure high is strictly greater

    # Close can be anywhere between Low and High (inclusive for this simplified version)
    close = draw(st.floats(min_value=low, max_value=high))

    return round(high, 2), round(low, 2), round(close, 2)

st_hlc_tuples = st.lists(
    st_single_hlc_tuple(),
    min_size=MIN_HISTORY_POINTS, max_size=MIN_HISTORY_POINTS + 20
)

st_volume_data = st.lists(st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False), min_size=MIN_HISTORY_POINTS, max_size=MIN_HISTORY_POINTS + 20)

st_current_legs = st.lists(
    st.builds(
        Leg,
        instrument=st.sampled_from(Instrument),
        direction=st.sampled_from(Direction),
        size=st.floats(min_value=0.1, max_value=100.0),
        limit_price=st.one_of(st.none(), st.floats(min_value=0.01, max_value=2000.0))
    ), max_size=3
)

st_planning_context_data_new = st.fixed_dictionaries({
    "timestamp": st.datetimes( # Hypothesis expects naive datetimes for min/max_value by default
        min_value=datetime(2023, 1, 1),
        max_value=datetime(2024, 1, 1),
        timezones=st.just(timezone.utc) # Generate UTC aware datetimes using naive bounds
    ),
    "equityCurve": st.lists(st.floats(min_value=1000.0, max_value=1e6), min_size=30, max_size=60),
    "dailyHistoryHLC": st_hlc_tuples,
    "dailyVolume": st.one_of(st.none(), st_volume_data), # Optional
    "currentPositions": st.one_of(st.none(), st_current_legs), # Optional
    "volSurface": st.fixed_dictionaries({
        Instrument.MES.value: st.floats(min_value=0.05, max_value=0.8) # Simplified for now
    }),
    "riskFreeRate": st.floats(min_value=0.0, max_value=0.2),
}).map(lambda d: { # Ensure dailyVolume has same length as dailyHistoryHLC if not None
    **d,
    "dailyVolume": [v for v, _ in zip(d["dailyVolume"], d["dailyHistoryHLC"])] if isinstance(d["dailyVolume"], list) and isinstance(d["dailyHistoryHLC"], list) else None
    # dailyHistoryHLC is guaranteed to be a list by st_hlc_tuples strategy, isinstance check is for mypy's benefit
})


@pytest.fixture
def sample_planning_context_data_new() -> Dict[str, Any]:
    """Provides a valid sample input dictionary for the new PlanningContext."""
    num_points = MIN_HISTORY_POINTS + 5
    hlc_data = _generate_hlc_data(num_periods=num_points)
    return {
        "timestamp": datetime.now(timezone.utc),
        "equityCurve": [10000.0 + i*10 for i in range(35)],
        "dailyHistoryHLC": hlc_data,
        "dailyVolume": [10000 + i*100 for i in range(num_points)],
        "currentPositions": [
            {"instrument": "MES", "direction": "LONG", "size": 2.0, "limit_price": 4500.0}
        ],
        "volSurface": {"MES": 0.15, "M2K": 0.20},
        "riskFreeRate": 0.02,
    }

# --- Comment out old tests based on latent_risk driven logic ---
# These tests are no longer valid for the new engine which uses signal-based logic.
# They are kept here for reference or if latent_risk logic is partially re-introduced.

# from unittest.mock import patch, MagicMock

# @patch('azr_planner.engine.calculate_latent_risk')
# def test_generate_plan_action_enter(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
#     """Test generate_plan returns 'ENTER' when latent risk < 0.30."""
#     # ... old test logic ...

# @patch('azr_planner.engine.calculate_latent_risk')
# def test_generate_plan_action_hold(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
#     """Test generate_plan returns 'HOLD' when 0.30 <= latent risk <= 0.70."""
#     # ... old test logic ...

# @patch('azr_planner.engine.calculate_latent_risk')
# def test_generate_plan_action_exit(mock_calculate_latent_risk: MagicMock, sample_planning_context_data: Dict[str, Any]) -> None:
#     """Test generate_plan returns 'EXIT' when latent risk > 0.70."""
#     # ... old test logic ...

# @given(data=st_planning_context_data) # Old strategy
# def test_property_generate_plan_structure_and_logic_OLD(data: Dict[str, Any]) -> None: # Renamed
#     """
#     Property test for generate_plan.
#     Verifies the output structure and basic logic based on latent risk.
#     Uses the actual (new) calculate_latent_risk from math_utils.
#     """
#     # ... old test logic ...


from hypothesis import HealthCheck, settings as hypothesis_settings # For HealthCheck

# --- New Property Test for the Refactored Engine ---
@given(data=st_planning_context_data_new)
@hypothesis_settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=50) # Adjusted from 100 to 50
def test_property_generate_plan_new_engine_basic_structure(data: Dict[str, Any]) -> None:
    """
    Property test for the new generate_plan engine.
    Verifies basic output structure and validity.
    Actual decision logic is placeholder, so this test focuses on schema compliance
    and presence of new fields.
    """
    # Ensure dailyVolume, if present, matches length of dailyHistoryHLC
    if data["dailyVolume"] is not None:
        assume(len(data["dailyVolume"]) == len(data["dailyHistoryHLC"]))

    try:
        ctx_input = PlanningContext.model_validate(data)
    except Exception as e:
        # Useful for debugging failing hypothesis strategies
        # print(f"Failed to validate PlanningContext with data: {data}, error: {e}")
        assume(False) # Skip invalid context data that Pydantic should catch
        return

    trade_proposal = generate_plan(ctx_input)

    assert isinstance(trade_proposal, TradeProposal)
    assert trade_proposal.action in ["ENTER", "HOLD", "EXIT", "ADJUST"] # ADJUST might be used by placeholder
    assert isinstance(trade_proposal.rationale, str) and len(trade_proposal.rationale) > 0

    assert isinstance(trade_proposal.confidence, float)
    assert 0.0 <= trade_proposal.confidence <= 1.0, "Confidence out of bounds"

    # Check new optional fields are either None or of correct type
    if trade_proposal.signal_value is not None:
        assert isinstance(trade_proposal.signal_value, float)
    if trade_proposal.atr_value is not None:
        assert isinstance(trade_proposal.atr_value, float)
        assert trade_proposal.atr_value >= 0 or math.isnan(trade_proposal.atr_value) # ATR can be NaN if not enough data
    if trade_proposal.kelly_fraction_value is not None:
        assert isinstance(trade_proposal.kelly_fraction_value, float)
        assert trade_proposal.kelly_fraction_value >= 0
    if trade_proposal.target_position_size is not None:
        assert isinstance(trade_proposal.target_position_size, float)
        # Size can be 0 if Kelly is 0 or negative
        assert trade_proposal.target_position_size >= 0

    if trade_proposal.legs is not None:
        assert isinstance(trade_proposal.legs, list)
        for leg in trade_proposal.legs:
            assert isinstance(leg, Leg)
            assert leg.size > 0

    # If action is ENTER, there should typically be legs, unless target size is zero for some reason
    if trade_proposal.action == "ENTER":
        if trade_proposal.target_position_size is not None and trade_proposal.target_position_size > 0 :
             assert trade_proposal.legs is not None and len(trade_proposal.legs) > 0, "ENTER action should have legs if target size > 0"
        # else: can be ENTER with 0 size if it's closing out an opposite position (not handled by current simple engine)

    # If action is HOLD, legs should usually be None
    if trade_proposal.action == "HOLD":
        # Legs might exist if HOLDing an existing position, but current engine logic might set to None.
        # This needs refinement based on actual engine logic for HOLD.
        # For now, the placeholder engine might set legs=None for HOLD.
        pass


def test_planning_context_instantiation_with_new_fields(sample_planning_context_data_new: Dict[str, Any]) -> None:
    """Test PlanningContext can be instantiated with new fields and aliases."""
    # Test with aliases (which are the same as field names for new fields for now)
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    assert ctx.equity_curve == sample_planning_context_data_new["equityCurve"]
    assert ctx.daily_history_hlc == sample_planning_context_data_new["dailyHistoryHLC"]
    assert ctx.daily_volume == sample_planning_context_data_new["dailyVolume"]
    assert ctx.current_positions is not None # In sample data
    assert len(ctx.current_positions) == 1
    assert ctx.current_positions[0].instrument == Instrument.MES


# Example of a more specific test if engine logic were finalized
# def test_generate_plan_strong_buy_signal_flat_portfolio(sample_planning_context_data_new):
#     """ Test that a strong buy signal on a flat portfolio generates an ENTER LONG proposal. """
#     data = sample_planning_context_data_new.copy()
      # Manipulate dailyHistoryHLC to create a strong upward trend for EMA crossover
#     hlc_data = []
#     price = 100
#     for i in range(MIN_HISTORY_POINTS):
#         price += 0.5 # Create rising prices
#         hlc_data.append((price + 0.5, price - 0.5, price))
#     data["dailyHistoryHLC"] = hlc_data
#     data["currentPositions"] = None # Flat portfolio
      # Potentially mock parts of math_utils if direct data manipulation is too complex
#     ctx = PlanningContext.model_validate(data)
#     proposal = generate_plan(ctx)
#     assert proposal.action == "ENTER"
#     assert proposal.legs is not None
#     assert len(proposal.legs) == 1
#     assert proposal.legs[0].direction == Direction.LONG
#     assert proposal.signal_value is not None and proposal.signal_value > some_strong_threshold
#     assert proposal.confidence > 0.5 # Expect reasonable confidence
#     assert proposal.target_position_size is not None and proposal.target_position_size > 0
#     assert proposal.kelly_fraction_value is not None and proposal.kelly_fraction_value > 0
#     assert proposal.atr_value is not None

# The old test_planning_context_instantiation_with_aliases is mostly covered by the new one
# and the old one used fields that are no longer sufficient for PlanningContext.
# So, I'm focusing on test_planning_context_instantiation_with_new_fields.
# I'll remove the very old test_planning_context_instantiation_with_aliases.

# Find the old test to remove:
# def test_planning_context_instantiation_with_aliases() -> None:
# Was at the end of the file previously.
# The new test `test_planning_context_instantiation_with_new_fields` replaces its intent
# for the updated schema.
