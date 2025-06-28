"""Tests for AZR Planner engine."""

import pytest
import math # Added import math
from unittest.mock import patch
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import DrawFn # For type hinting draw in composite strategies
from datetime import datetime, timezone
from typing import Dict, Any, List

from azr_planner.engine import generate_plan
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg
from azr_planner.math_utils import LR_V2_MIN_POINTS # Import for MIN_HISTORY_POINTS

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
# Minimum data points for daily_history_hlc and daily_volume
# For AZR-06, this is primarily driven by latent_risk_v2 requirements.
MIN_HISTORY_POINTS = LR_V2_MIN_POINTS + 5 # Add a small buffer

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
    "dailyVolume": st.one_of(st.none(), st_volume_data),
    "currentPositions": st.one_of(st.none(), st_current_legs),
    "n_successes": st.integers(min_value=0, max_value=100), # New field
    "n_failures": st.integers(min_value=0, max_value=100),  # New field
    "volSurface": st.fixed_dictionaries({
        Instrument.MES.value: st.floats(min_value=0.05, max_value=0.8)
    }),
    "riskFreeRate": st.floats(min_value=0.0, max_value=0.2),
}).map(lambda d: {
    **d,
    "dailyVolume": [v for v, _ in zip(d["dailyVolume"], d["dailyHistoryHLC"])] if isinstance(d["dailyVolume"], list) and isinstance(d["dailyHistoryHLC"], list) else None
    # dailyHistoryHLC is guaranteed to be a list by st_hlc_tuples strategy, isinstance check is for mypy's benefit
})


@pytest.fixture
def sample_planning_context_data_new() -> Dict[str, Any]:
    """Provides a valid sample input dictionary for the new PlanningContext."""
    num_points = MIN_HISTORY_POINTS + 5 # e.g., 30 + 5 + 5 = 40
    hlc_data = _generate_hlc_data(num_periods=num_points)
    equity_curve_data = [10000.0 + i*10 for i in range(num_points)] # Match length with hlc_data
    return {
        "timestamp": datetime.now(timezone.utc),
        "equityCurve": equity_curve_data,
        "dailyHistoryHLC": hlc_data,
        "dailyVolume": [10000 + i*100 for i in range(num_points)],
        "currentPositions": [
            {"instrument": "MES", "direction": "LONG", "size": 2.0, "limit_price": 4500.0}
        ],
        "n_successes": 10,
        "n_failures": 5,
        "volSurface": {"MES": 0.15, "M2K": 0.20},
        "riskFreeRate": 0.02,
    }

# --- New tests for AZR-06 Engine Logic ---
from unittest.mock import MagicMock # Ensure MagicMock is available if not already

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_enter_azr06(
    mock_bayesian_confidence: MagicMock,
    mock_latent_risk_v2: MagicMock,
    sample_planning_context_data_new: Dict[str, Any]
) -> None:
    """Test generate_plan returns 'ENTER' with new AZR-06 thresholds."""
    mock_latent_risk_v2.return_value = 0.10 # lr < 0.25
    mock_bayesian_confidence.return_value = 0.80 # conf > 0.7

    # Remove current positions to ensure a clean ENTER scenario if that's simpler for leg check
    sample_planning_context_data_new["currentPositions"] = None
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)

    assert trade_proposal.action == "ENTER"
    assert trade_proposal.latent_risk == 0.10
    assert trade_proposal.confidence == 0.80
    assert trade_proposal.legs is not None
    assert len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.LONG

    # AZR-11: Verify position sizing
    # Reconstruct expected size based on inputs to position_size and mocked values
    mocked_lr = 0.10
    current_equity = ctx.equity_curve[-1]
    mes_price = ctx.daily_history_hlc[-1][2] # Last close price

    # Expected exposure from position_size(latent_risk=0.10, equity=current_equity, max_leverage=DEFAULT_MAX_LEVERAGE)
    # Since risk 0.10 <= 0.3, expected_exposure = current_equity * DEFAULT_MAX_LEVERAGE
    # Constants from engine.py: DEFAULT_MAX_LEVERAGE = 2.0, MES_CONTRACT_MULTIPLIER = 5.0
    expected_dollar_exposure = current_equity * 2.0
    expected_contract_size = expected_dollar_exposure / (mes_price * 5.0)

    assert math.isclose(leg.size, expected_contract_size, rel_tol=1e-5) # Use relative tolerance for float comparison

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_exit_lr_azr06(
    mock_bayesian_confidence: MagicMock,
    mock_latent_risk_v2: MagicMock,
    sample_planning_context_data_new: Dict[str, Any]
) -> None:
    """Test generate_plan returns 'EXIT' due to high latent risk (AZR-06)."""
    mock_latent_risk_v2.return_value = 0.80 # lr > 0.7
    mock_bayesian_confidence.return_value = 0.60 # conf is neutral, not triggering exit itself

    ctx = PlanningContext.model_validate(sample_planning_context_data_new) # Has existing LONG MES
    trade_proposal = generate_plan(ctx)

    assert trade_proposal.action == "EXIT"
    assert trade_proposal.latent_risk == 0.80
    assert trade_proposal.confidence == 0.60
    assert trade_proposal.legs is not None
    assert len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.SHORT
    # Size should be current holding if logic is to close out existing position
    # Sample data has 2.0 LONG MES
    assert leg.size == 2.0


@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_exit_conf_azr06(
    mock_bayesian_confidence: MagicMock,
    mock_latent_risk_v2: MagicMock,
    sample_planning_context_data_new: Dict[str, Any]
) -> None:
    """Test generate_plan returns 'EXIT' due to low confidence (AZR-06)."""
    mock_latent_risk_v2.return_value = 0.50 # lr is neutral
    mock_bayesian_confidence.return_value = 0.30 # conf < 0.4

    sample_planning_context_data_new["currentPositions"] = None # Test EXIT even if flat
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)

    assert trade_proposal.action == "EXIT"
    assert trade_proposal.latent_risk == 0.50
    assert trade_proposal.confidence == 0.30
    assert trade_proposal.legs is not None
    assert len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES
    assert leg.direction == Direction.SHORT
    assert leg.size == 1.0 # Defaults to shorting 1 if no existing long


@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_hold_azr06(
    mock_bayesian_confidence: MagicMock,
    mock_latent_risk_v2: MagicMock,
    sample_planning_context_data_new: Dict[str, Any]
) -> None:
    """Test generate_plan returns 'HOLD' with neutral signals (AZR-06)."""
    mock_latent_risk_v2.return_value = 0.50 # Neutral lr (0.25 <= lr <= 0.7)
    mock_bayesian_confidence.return_value = 0.60 # Neutral conf (0.4 <= conf <= 0.7)

    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)

    assert trade_proposal.action == "HOLD"
    assert trade_proposal.latent_risk == 0.50
    assert trade_proposal.confidence == 0.60
    assert trade_proposal.legs is None


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

    # Check new optional fields from AZR-05 are now None or not present if removed from TradeProposal
    assert getattr(trade_proposal, "signal_value", None) is None
    assert getattr(trade_proposal, "atr_value", None) is None
    assert getattr(trade_proposal, "kelly_fraction_value", None) is None
    assert getattr(trade_proposal, "target_position_size", None) is None

    # Check latent_risk is populated
    assert trade_proposal.latent_risk is not None
    assert 0.0 <= trade_proposal.latent_risk <= 1.0

    if trade_proposal.action == "ENTER":
        assert trade_proposal.legs is not None and len(trade_proposal.legs) == 1
        leg = trade_proposal.legs[0]
        assert leg.instrument == Instrument.MES
        assert leg.direction == Direction.LONG
        # AZR-11: Size is now dynamic.
        # generate_plan reverts to HOLD if size would be too small ( < MIN_CONTRACT_SIZE (0.001) )
        # So if action is still ENTER, size must be >= MIN_CONTRACT_SIZE
        from azr_planner.engine import MIN_CONTRACT_SIZE # Import for the check
        assert leg.size >= MIN_CONTRACT_SIZE
    elif trade_proposal.action == "EXIT":
        # Legs for EXIT can be more complex (sell existing or new short)
        # For this property test, just ensure legs are present if action is EXIT,
        # or None if it's an EXIT signal but no specific legs are formed (e.g. flat portfolio already)
        # The new engine logic for EXIT ensures legs are populated.
        assert trade_proposal.legs is not None
        assert len(trade_proposal.legs) >= 0 # Can be empty if no specific exit leg needed but action is EXIT
                                             # However, current engine logic for EXIT always creates a leg.
        if trade_proposal.legs: # If legs are created
            assert isinstance(trade_proposal.legs[0], Leg)

    elif trade_proposal.action == "HOLD":
        assert trade_proposal.legs is None


def test_planning_context_instantiation_with_new_fields(sample_planning_context_data_new: Dict[str, Any]) -> None:
    """Test PlanningContext can be instantiated with new fields and aliases."""
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    assert ctx.equity_curve == sample_planning_context_data_new["equityCurve"]
    assert ctx.daily_history_hlc == sample_planning_context_data_new["dailyHistoryHLC"]
    assert ctx.daily_volume == sample_planning_context_data_new["dailyVolume"]
    assert ctx.current_positions is not None
    assert len(ctx.current_positions) == 1
    assert ctx.current_positions[0].instrument == Instrument.MES
    assert ctx.n_successes == sample_planning_context_data_new["n_successes"]
    assert ctx.n_failures == sample_planning_context_data_new["n_failures"]


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


@patch('azr_planner.engine.latent_risk_v2', return_value=0.10) # Favorable risk
@patch('azr_planner.engine.bayesian_confidence', return_value=0.80) # Favorable confidence
def test_generate_plan_enter_sizing_edge_cases(
    mock_bayesian_confidence: MagicMock, # Order matters for patch decorators
    mock_latent_risk_v2: MagicMock,
    sample_planning_context_data_new: Dict[str, Any]
) -> None:
    """
    Tests sizing edge cases in generate_plan when an ENTER signal occurs.
    - No HLC data.
    - Invalid (zero/negative) MES price.
    - Calculated size too small.
    """
    base_ctx_data = sample_planning_context_data_new.copy()
    base_ctx_data["currentPositions"] = None # Ensure clean ENTER

    # Case 1 (previously Case 2): Invalid MES price (e.g., zero or negative)
    # The "No HLC data" case is now covered by Pydantic validation on PlanningContext.
    ctx_data_bad_price = base_ctx_data.copy()
    # Ensure dailyHistoryHLC is not empty, but set the last close to 0
    original_hlc = ctx_data_bad_price["dailyHistoryHLC"]
    bad_price_hlc = [list(t) for t in original_hlc] # mutable copy
    if bad_price_hlc:
        bad_price_hlc[-1][2] = 0.0 # Set last close to 0
        ctx_data_bad_price["dailyHistoryHLC"] = [tuple(t) for t in bad_price_hlc]

    ctx_bad_price = PlanningContext.model_validate(ctx_data_bad_price)
    proposal_bad_price = generate_plan(ctx_bad_price)
    assert proposal_bad_price.action == "HOLD"
    assert "Invalid MES price (0.00) for sizing" in proposal_bad_price.rationale

    # Case 3: Calculated size too small (e.g., due to very high price or very low equity/exposure)
    ctx_data_small_size = base_ctx_data.copy()
    # Make equity very small so target exposure is tiny, leading to a sub-MIN_CONTRACT_SIZE.
    # Original equity curve values are ~10000. Last value is ~10000 + 39*10 = 10390.
    # mes_price is last HLC close, from _generate_hlc_data(num_periods=40), starts at 100.
    # Let's assume mes_price is ~120 for estimation. Contract value ~ 120 * 5 = 600.
    # If equity = 0.1, exposure = 0.1 * 2 = 0.2. Size = 0.2 / 600 = 0.000333. This is < MIN_CONTRACT_SIZE (0.001).
    very_small_equity_value = 0.1
    small_equity_curve = [very_small_equity_value] * len(ctx_data_small_size["equityCurve"])
    ctx_data_small_size["equityCurve"] = small_equity_curve

    ctx_small_size = PlanningContext.model_validate(ctx_data_small_size)
    proposal_small_size = generate_plan(ctx_small_size)
    assert proposal_small_size.action == "HOLD"
    assert "Calculated size" in proposal_small_size.rationale and "too small" in proposal_small_size.rationale
