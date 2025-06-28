"""Tests for AZR Planner engine."""

import pytest
import math
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, assume, HealthCheck, settings as hypothesis_settings
from hypothesis.strategies import DrawFn
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from azr_planner.engine import generate_plan, MIN_CONTRACT_SIZE
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg
from azr_planner.math_utils import LR_V2_MIN_POINTS

# --- Helper function to generate HLC data ---
def _generate_hlc_data(num_periods: int, start_price: float = 100.0, daily_change: float = 0.1, spread: float = 0.5) -> List[Tuple[float, float, float]]:
    data: List[Tuple[float, float, float]] = []
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + abs(daily_change * math.sin(i*0.1))
        low = current_close - spread - abs(daily_change * math.cos(i*0.1))
        close = (high + low) / 2 + (math.sin(i*0.5) * spread*0.1)
        low = min(low, high - 0.01)
        close = max(min(close, high), low)
        data.append((round(high,2), round(low,2), round(close,2)))
        current_close = close
    return data

MIN_HISTORY_POINTS = LR_V2_MIN_POINTS + 5

@st.composite
def st_single_hlc_tuple(draw: DrawFn) -> Tuple[float, float, float]:
    f1 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f2 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f3 = draw(st.floats(min_value=1.0, max_value=1999.0))
    s = sorted([f1, f2, f3])
    low, _, high = s[0], s[1], s[2]
    if high <= low: high = low + draw(st.floats(min_value=0.1, max_value=1.0))
    close = draw(st.floats(min_value=low, max_value=high))
    return round(high, 2), round(low, 2), round(close, 2)

st_hlc_tuples = st.lists(st_single_hlc_tuple(), min_size=MIN_HISTORY_POINTS, max_size=MIN_HISTORY_POINTS + 20)
st_volume_data = st.lists(st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False), min_size=MIN_HISTORY_POINTS, max_size=MIN_HISTORY_POINTS + 20)
st_current_legs = st.lists(
    st.builds( Leg, instrument=st.sampled_from(Instrument), direction=st.sampled_from(Direction),
               size=st.floats(min_value=0.1, max_value=100.0),
               limit_price=st.one_of(st.none(), st.floats(min_value=0.01, max_value=2000.0))), max_size=3)

st_planning_context_data_new = st.fixed_dictionaries({
    "timestamp": st.datetimes(min_value=datetime(2023, 1, 1), max_value=datetime(2024, 1, 1), timezones=st.just(timezone.utc)),
    "equityCurve": st.lists(st.floats(min_value=1000.0, max_value=1e6), min_size=LR_V2_MIN_POINTS, max_size=LR_V2_MIN_POINTS +30),
    "dailyHistoryHLC": st_hlc_tuples,
    "dailyVolume": st.one_of(st.none(), st_volume_data),
    "currentPositions": st.one_of(st.none(), st_current_legs),
    "n_successes": st.integers(min_value=0, max_value=100),
    "n_failures": st.integers(min_value=0, max_value=100),
    "volSurface": st.fixed_dictionaries({ Instrument.MES.value: st.floats(min_value=0.05, max_value=0.8) }),
    "riskFreeRate": st.floats(min_value=0.0, max_value=0.2),
}).map(lambda d: {
    **d, "dailyVolume": [v for v, _ in zip(d["dailyVolume"], d["dailyHistoryHLC"])] if isinstance(d["dailyVolume"], list) and isinstance(d["dailyHistoryHLC"], list) else None
})

@pytest.fixture
def sample_planning_context_data_new() -> Dict[str, Any]:
    num_points = MIN_HISTORY_POINTS + 5
    hlc_data = _generate_hlc_data(num_periods=num_points)
    equity_curve_data = [10000.0 + i*10 for i in range(num_points)]
    return {
        "timestamp": datetime.now(timezone.utc), "equityCurve": equity_curve_data, "dailyHistoryHLC": hlc_data,
        "dailyVolume": [float(10000 + i*100) for i in range(num_points)],
        "currentPositions": [ Leg(instrument=Instrument.MES, direction=Direction.LONG, size=2.0, limit_price=4500.0).model_dump() ],
        "n_successes": 10, "n_failures": 5, "volSurface": {"MES": 0.15, "M2K": 0.20}, "riskFreeRate": 0.02,
    }

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_enter_azr06(
    mock_bayesian_confidence: MagicMock, mock_latent_risk_v2: MagicMock, sample_planning_context_data_new: Dict[str, Any]
) -> None:
    mock_latent_risk_v2.return_value = 0.10
    mock_bayesian_confidence.return_value = 0.80
    sample_planning_context_data_new["currentPositions"] = None
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)
    assert trade_proposal.action == "ENTER"
    assert trade_proposal.latent_risk == 0.10; assert trade_proposal.confidence == 0.80
    assert trade_proposal.legs is not None and len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES; assert leg.direction == Direction.LONG
    current_equity = ctx.equity_curve[-1]; mes_price = ctx.daily_history_hlc[-1][2]
    from azr_planner.engine import DEFAULT_MAX_LEVERAGE, MES_CONTRACT_MULTIPLIER # Import engine constants
    expected_dollar_exposure = current_equity * DEFAULT_MAX_LEVERAGE
    expected_contract_size = expected_dollar_exposure / (mes_price * MES_CONTRACT_MULTIPLIER)
    assert math.isclose(leg.size, expected_contract_size, rel_tol=1e-5)

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_exit_lr_azr06(
    mock_bayesian_confidence: MagicMock, mock_latent_risk_v2: MagicMock, sample_planning_context_data_new: Dict[str, Any]
) -> None:
    mock_latent_risk_v2.return_value = 0.80; mock_bayesian_confidence.return_value = 0.60
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)
    assert trade_proposal.action == "EXIT"; assert trade_proposal.latent_risk == 0.80
    assert trade_proposal.confidence == 0.60; assert trade_proposal.legs is not None and len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES; assert leg.direction == Direction.SHORT; assert leg.size == 2.0

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_exit_conf_azr06(
    mock_bayesian_confidence: MagicMock, mock_latent_risk_v2: MagicMock, sample_planning_context_data_new: Dict[str, Any]
) -> None:
    mock_latent_risk_v2.return_value = 0.50; mock_bayesian_confidence.return_value = 0.30
    sample_planning_context_data_new["currentPositions"] = None
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)
    assert trade_proposal.action == "EXIT"; assert trade_proposal.latent_risk == 0.50
    assert trade_proposal.confidence == 0.30; assert trade_proposal.legs is not None and len(trade_proposal.legs) == 1
    leg = trade_proposal.legs[0]
    assert leg.instrument == Instrument.MES; assert leg.direction == Direction.SHORT; assert leg.size == 1.0

@patch('azr_planner.engine.latent_risk_v2')
@patch('azr_planner.engine.bayesian_confidence')
def test_generate_plan_action_hold_azr06(
    mock_bayesian_confidence: MagicMock, mock_latent_risk_v2: MagicMock, sample_planning_context_data_new: Dict[str, Any]
) -> None:
    mock_latent_risk_v2.return_value = 0.50; mock_bayesian_confidence.return_value = 0.60
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    trade_proposal = generate_plan(ctx)
    assert trade_proposal.action == "HOLD"; assert trade_proposal.latent_risk == 0.50
    assert trade_proposal.confidence == 0.60; assert trade_proposal.legs is None

@given(data=st_planning_context_data_new)
@hypothesis_settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=50)
def test_property_generate_plan_new_engine_basic_structure(data: Dict[str, Any]) -> None:
    if data["dailyVolume"] is not None: assume(len(data["dailyVolume"]) == len(data["dailyHistoryHLC"]))
    try: ctx_input = PlanningContext.model_validate(data)
    except Exception: assume(False); return
    trade_proposal = generate_plan(ctx_input)
    assert isinstance(trade_proposal, TradeProposal)
    assert trade_proposal.action in ["ENTER", "HOLD", "EXIT"]
    assert isinstance(trade_proposal.rationale, str) and len(trade_proposal.rationale) > 0
    assert isinstance(trade_proposal.confidence, float) and 0.0 <= trade_proposal.confidence <= 1.0
    assert getattr(trade_proposal, "signal_value", None) is None
    assert trade_proposal.latent_risk is not None and 0.0 <= trade_proposal.latent_risk <= 1.0
    if trade_proposal.action == "ENTER":
        assert trade_proposal.legs is not None and len(trade_proposal.legs) == 1
        leg = trade_proposal.legs[0]
        assert leg.instrument == Instrument.MES; assert leg.direction == Direction.LONG
        assert leg.size >= MIN_CONTRACT_SIZE
    elif trade_proposal.action == "EXIT":
        assert trade_proposal.legs is not None
        if trade_proposal.legs: assert isinstance(trade_proposal.legs[0], Leg)
    elif trade_proposal.action == "HOLD":
        assert trade_proposal.legs is None

def test_planning_context_instantiation_with_new_fields(sample_planning_context_data_new: Dict[str, Any]) -> None:
    ctx = PlanningContext.model_validate(sample_planning_context_data_new)
    assert ctx.equity_curve == sample_planning_context_data_new["equityCurve"]
    assert ctx.daily_history_hlc == sample_planning_context_data_new["dailyHistoryHLC"]
    assert ctx.daily_volume == sample_planning_context_data_new["dailyVolume"]
    assert ctx.current_positions is not None and len(ctx.current_positions) == 1
    assert ctx.current_positions[0].instrument == Instrument.MES
    assert ctx.n_successes == sample_planning_context_data_new["n_successes"] # field name
    assert ctx.n_failures == sample_planning_context_data_new["n_failures"]   # field name

@patch('azr_planner.engine.latent_risk_v2', return_value=0.10)
@patch('azr_planner.engine.bayesian_confidence', return_value=0.80)
def test_generate_plan_enter_sizing_edge_cases(
    mock_bayesian_confidence: MagicMock, mock_latent_risk_v2: MagicMock, sample_planning_context_data_new: Dict[str, Any]
) -> None:
    base_ctx_data = sample_planning_context_data_new.copy()
    base_ctx_data["currentPositions"] = None

    ctx_data_bad_price = base_ctx_data.copy()
    original_hlc = ctx_data_bad_price["dailyHistoryHLC"]
    bad_price_hlc = [list(t) for t in original_hlc]
    if bad_price_hlc: # Check if list is not empty
        bad_price_hlc[-1][2] = 0.0 # Correctly assign to list element's index
        ctx_data_bad_price["dailyHistoryHLC"] = [tuple(t) for t in bad_price_hlc]

    ctx_bad_price = PlanningContext.model_validate(ctx_data_bad_price)
    proposal_bad_price = generate_plan(ctx_bad_price)
    assert proposal_bad_price.action == "HOLD"
    assert "Invalid MES price (0.00) for sizing" in proposal_bad_price.rationale

    ctx_data_small_size = base_ctx_data.copy()
    very_small_equity_value = 0.1
    small_equity_curve = [very_small_equity_value] * len(ctx_data_small_size["equityCurve"])
    ctx_data_small_size["equityCurve"] = small_equity_curve
    ctx_small_size = PlanningContext.model_validate(ctx_data_small_size)
    proposal_small_size = generate_plan(ctx_small_size)
    assert proposal_small_size.action == "HOLD"
    assert "Calculated size" in proposal_small_size.rationale and "too small" in proposal_small_size.rationale
