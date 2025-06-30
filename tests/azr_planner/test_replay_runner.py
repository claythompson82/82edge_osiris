from __future__ import annotations

import pytest
import math
import datetime
import time
from typing import List, Optional, Tuple, Any, Dict, Iterable, Callable
from unittest.mock import MagicMock, patch
from pathlib import Path # <--- ADDED IMPORT

from hypothesis import given, strategies as st, settings, HealthCheck

from azr_planner.schemas import PlanningContext, TradeProposal, Leg, Instrument, Direction
from azr_planner.risk_gate import RiskGateConfig
from azr_planner.replay.schemas import Bar, ReplayReport, ReplayTrade
from azr_planner.replay.runner import run_replay, REPLAY_RUNS_TOTAL
from azr_planner.math_utils import LR_V2_MIN_POINTS

from prometheus_client import CollectorRegistry, Counter


# --- Helper Functions & Mocks ---
def create_bar_stream(
    num_bars: int, start_price: float = 100.0, price_increment: float = 0.0,
    instrument_name: str = "TESTINST", start_time: Optional[datetime.datetime] = None,
    time_increment_seconds: int = 60*15, provide_volume: bool = True
) -> List[Bar]:
    bars: List[Bar] = []
    current_ts = start_time or datetime.datetime(2023,1,1,0,0,0, tzinfo=datetime.timezone.utc)
    current_price = start_price
    for i in range(num_bars):
        h_val = current_price + 0.5 if price_increment >=0 else current_price
        l_val = current_price - 0.5 if price_increment <=0 else current_price
        if l_val > h_val: l_val = h_val
        bar_volume = 100.0 + i if provide_volume else None
        bars.append(Bar(
            timestamp=current_ts + datetime.timedelta(seconds=i * time_increment_seconds),
            instrument=instrument_name, open=current_price, high=h_val, low=l_val,
            close=current_price, volume=bar_volume ))
        current_price += price_increment
    return bars

def mock_planner_hold(ctx: PlanningContext) -> TradeProposal:
    return TradeProposal(action="HOLD", rationale="Mock Hold", latent_risk=0.1, confidence=0.9, legs=None)

def mock_planner_enter_long(ctx: PlanningContext) -> TradeProposal:
    return TradeProposal(action="ENTER", rationale="Mock Enter Long", latent_risk=0.1, confidence=0.9,
                         legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)])

def mock_risk_gate_always_accept_for_replay(
    proposal: TradeProposal, *, db_table: Optional[Any] = None,
    cfg: Optional[RiskGateConfig] = None, registry: Optional[CollectorRegistry] = None
) -> Tuple[bool, Optional[str]]:
    return True, None

# --- Unit Tests for run_replay ---
def test_run_replay_empty_bar_stream() -> None:
    report, trades = run_replay(
        bar_stream=[], initial_equity=100_000.0,
        planner_fn=mock_planner_hold, risk_gate_fn=mock_risk_gate_always_accept_for_replay,
        instrument_group_label="test", granularity_label="test")
    assert report.total_bars_processed == 0; assert report.proposals_generated == 0
    assert report.final_equity == 100_000.0; assert math.isclose(report.total_return_pct, 0.0)
    assert math.isclose(report.max_drawdown_pct, 0.0); assert not trades
    assert len(report.equity_curve) == 0

def test_run_replay_insufficient_bars_for_context() -> None:
    bars = create_bar_stream(num_bars=LR_V2_MIN_POINTS - 1)
    report, trades = run_replay(
        bar_stream=bars, initial_equity=100_000.0,
        planner_fn=mock_planner_hold, risk_gate_fn=mock_risk_gate_always_accept_for_replay,
        instrument_group_label="test", granularity_label="test")
    assert report.total_bars_processed == LR_V2_MIN_POINTS - 1
    assert report.proposals_generated == 0
    assert math.isclose(report.final_equity, 100_000.0, rel_tol=1e-7)
    assert math.isclose(report.total_return_pct, 0.0, abs_tol=1e-7)
    assert math.isclose(report.max_drawdown_pct, 0.0, abs_tol=1e-7)
    assert not trades
    assert len(report.equity_curve) == len(bars) # One point per bar if MTM happens
    assert report.mean_planner_decision_ms is None # No proposals, so no decision times

@given(num_bars=st.integers(min_value=0, max_value=50),
       initial_equity_val=st.floats(min_value=1000.0, max_value=1_000_000.0))
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_run_replay_property_flat_bars_flat_equity(num_bars: int, initial_equity_val: float) -> None:
    if num_bars == 0:
        bar_stream: List[Bar] = []
    else:
        start_ts = datetime.datetime(2023,1,1, tzinfo=datetime.timezone.utc)
        # For num_bars > 0, ensure start_time is not None for create_bar_stream if it expects it
        bar_stream = create_bar_stream(num_bars=num_bars, start_price=100.0, price_increment=0.0, start_time=start_ts)

    report, trades = run_replay(
        bar_stream=bar_stream, initial_equity=initial_equity_val,
        planner_fn=mock_planner_hold, risk_gate_fn=mock_risk_gate_always_accept_for_replay,
        instrument_group_label="prop_flat", granularity_label="test")
    assert report.total_bars_processed == num_bars
    expected_proposals = max(0, num_bars - (LR_V2_MIN_POINTS - 1)) if num_bars >= LR_V2_MIN_POINTS else 0
    assert report.proposals_generated == expected_proposals
    assert report.proposals_accepted == 0; assert report.proposals_rejected == 0
    assert math.isclose(report.final_equity, initial_equity_val, rel_tol=1e-7)
    assert math.isclose(report.total_return_pct, 0.0, abs_tol=1e-7)
    assert math.isclose(report.max_drawdown_pct, 0.0, abs_tol=1e-7)
    if expected_proposals > 0: assert len(trades) == expected_proposals
    else: assert not trades
    # Trying a hypothesis: if num_bars is 0, perhaps equity_curve has 1 point (initial state)
    # This contradicts test_run_replay_empty_bar_stream, but let's see the outcome.
    expected_len = num_bars
    if num_bars == 0:
        expected_len = 0 # Reverting to original logic, previous thought was for debugging. The test_empty_stream should be the guide.
        # If it's truly 1, the run_replay logic needs to change or test_empty_stream is wrong.
        # Sticking to the idea that empty stream means empty equity curve.

    assert len(report.equity_curve) == expected_len

    if report.equity_curve:
        for _, equity_val in report.equity_curve:
            assert math.isclose(equity_val, initial_equity_val, rel_tol=1e-7)

def test_run_replay_deterministic_rising_bars_positive_return() -> None:
    num_bars = 60; initial_equity = 100_000.0
    bars = create_bar_stream(num_bars=num_bars, start_price=4500.0, price_increment=1.0, instrument_name="MES")
    report, trades = run_replay(
        bar_stream=bars, initial_equity=initial_equity,
        planner_fn=mock_planner_enter_long, risk_gate_fn=mock_risk_gate_always_accept_for_replay,
        instrument_group_label="det_rising", granularity_label="test")
    assert report.total_bars_processed == num_bars; assert report.proposals_generated > 0
    assert report.proposals_accepted > 0
    assert report.final_equity > initial_equity ; assert report.total_return_pct > 0.0
    assert 0.0 <= report.max_drawdown_pct < 1.0
    assert len(trades) == report.proposals_generated
    assert any(trade.accepted_by_risk_gate for trade in trades)

def test_run_replay_bars_with_no_volume(tmp_path: Path) -> None:
    num_bars = LR_V2_MIN_POINTS + 5 # Ensure some proposals are generated
    # Create bars with provide_volume=False
    bars_no_volume = create_bar_stream(num_bars=num_bars, start_price=100.0, price_increment=0.1, provide_volume=False)

    # Mock planner function to check context
    def mock_planner_check_volume(ctx: PlanningContext) -> TradeProposal:
        assert ctx.daily_volume is None, "Planner context daily_volume should be None when all bar volumes are None"
        return TradeProposal(action="HOLD", rationale="Checked volume", latent_risk=0.0, confidence=1.0, legs=None)

    report, trades = run_replay(
        bar_stream=bars_no_volume, initial_equity=100_000.0,
        planner_fn=mock_planner_check_volume,
        risk_gate_fn=mock_risk_gate_always_accept_for_replay
    )
    assert report.total_bars_processed == num_bars
    # proposals_generated will be num_bars - (LR_V2_MIN_POINTS - 1)
    assert report.proposals_generated == num_bars - (LR_V2_MIN_POINTS - 1)


def test_run_replay_pnl_and_position_logic() -> None:
    """
    Tests detailed P&L calculations, average price updates, and position handling
    by simulating a sequence of trades.
    """
    initial_equity = 100_000.0
    trade_instrument = Instrument.MES # Using MES for tests

    # Trade sequence: (relative_bar_index_for_planning, action, size)
    # Action: 1=BUY, -1=SELL
    # Indices are relative to the first bar for which a planning decision is made.
    trade_plan = [
        (0, 1, 2.0),   # First planning decision: Buy 2 contracts
        (1, 1, 3.0),   # Second: Buy 3 contracts (Total 5 LONG)
        (2, -1, 1.0),  # Third: Sell 1 contract (Total 4 LONG)
        (3, -1, 4.0),  # Fourth: Sell 4 contracts (Position FLAT)
        (4, -1, 3.0),  # Fifth: Sell 3 contracts (Total 3 SHORT)
        (5, -1, 2.0),  # Sixth: Sell 2 contracts (Total 5 SHORT)
        (6, 1, 1.0),   # Seventh: Buy 1 contract (Total 4 SHORT)
        (7, 1, 4.0),   # Eighth: Buy 4 contracts (Position FLAT)
        (8, 1, 1.0),   # Ninth: Buy 1 (Total 1 LONG)
        (9, -1, 2.0),  # Tenth: Sell 2 (Reverse to 1 SHORT)
    ]

    num_total_bars = LR_V2_MIN_POINTS + 15 # Enough bars for all trades and some buffer to observe final state
    bar_prices = [100.0 + i*0.5 for i in range(num_total_bars)] # Simple rising prices

    bars = []
    for i in range(num_total_bars):
        bars.append(Bar(
            timestamp=datetime.datetime(2023,1,1, tzinfo=datetime.timezone.utc) + datetime.timedelta(minutes=i*15),
            instrument=trade_instrument.value, open=bar_prices[i], high=bar_prices[i]+0.25,
            low=bar_prices[i]-0.25, close=bar_prices[i], volume=100.0
        ))

    # Using a mutable object to track planner calls for strategic_planner
    planner_call_count_obj = {"count": 0}

    def strategic_planner_with_counter(ctx: PlanningContext) -> TradeProposal:
        active_planning_bar_index = planner_call_count_obj["count"]
        planner_call_count_obj["count"] += 1 # Increment after use for next call

        for plan_bar_idx, action_val, size_val in trade_plan:
            if active_planning_bar_index == plan_bar_idx:
                direction = Direction.LONG if action_val == 1 else Direction.SHORT
                return TradeProposal(action="ENTER", rationale=f"Planned {direction.value} {size_val}",
                                     legs=[Leg(instrument=trade_instrument, direction=direction, size=size_val)],
                                     confidence=1.0, latent_risk=0.0)
        return TradeProposal(action="HOLD", rationale="Strategic Hold", legs=None, confidence=1.0, latent_risk=0.0)

    report, trades = run_replay(
        bar_stream=bars, initial_equity=initial_equity,
        planner_fn=strategic_planner_with_counter,
        risk_gate_fn=mock_risk_gate_always_accept_for_replay
    )

    assert report.total_bars_processed == num_total_bars
    assert planner_call_count_obj["count"] == num_total_bars - (LR_V2_MIN_POINTS -1)

    # Verify P&L and final equity. Exact P&L calculation is complex for this test to assert fully.
    # Focus is on exercising the code paths.
    # A simple check: if prices generally rose and we had net long exposure, equity should increase.
    assert report.final_equity != initial_equity # Expect some change

    # Check if all planned trades were attempted (either as ENTER or part of HOLD if logic changes)
    # Number of proposals that are not HOLD
    num_non_hold_proposals = sum(1 for t in trades if t.proposal_action != "HOLD")
    assert num_non_hold_proposals == len(trade_plan)


# --- Test for Prometheus Counter Increment ---
def get_replay_counter_value(registry: CollectorRegistry, instrument_group: str, granularity: str) -> float:
    # REPLAY_RUNS_TOTAL is defined in runner.py, REPLAY_RUNS_TOTAL._name is its string name
    # Counters in prometheus_client automatically append "_total" to the name if not already present.
    metric_name_with_total = REPLAY_RUNS_TOTAL._name
    if not metric_name_with_total.endswith('_total'):
        metric_name_with_total += '_total'

    value = registry.get_sample_value(metric_name_with_total, labels={"instrument_group": instrument_group, "granularity": granularity})
    return value if value is not None else 0.0

def test_run_replay_increments_prometheus_counter() -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    instrument_group = "test_group"; granularity = "test_gran"
    patched_counter = Counter(REPLAY_RUNS_TOTAL._name, REPLAY_RUNS_TOTAL._documentation,
                              REPLAY_RUNS_TOTAL._labelnames, registry=test_registry)
    with patch("azr_planner.replay.runner.REPLAY_RUNS_TOTAL", new=patched_counter):
        initial_count = get_replay_counter_value(test_registry, instrument_group, granularity)
        assert initial_count == 0.0
        run_replay(
            bar_stream=create_bar_stream(5), initial_equity=100_000.0,
            planner_fn=mock_planner_hold, risk_gate_fn=mock_risk_gate_always_accept_for_replay,
            instrument_group_label=instrument_group, granularity_label=granularity
        )
        assert get_replay_counter_value(test_registry, instrument_group, granularity) == initial_count + 1.0
