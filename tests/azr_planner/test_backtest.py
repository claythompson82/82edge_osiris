from __future__ import annotations

import math
import pytest
from typing import List, Dict, Any, Optional, Tuple, cast
import numpy as np
from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, assume, HealthCheck, settings
from hypothesis.strategies import DrawFn
from unittest.mock import patch

from azr_planner.backtest.metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate_and_pnl_stats
)
from azr_planner.backtest.schemas import DailyTrade, SingleBacktestReport
from azr_planner.schemas import PlanningContext, Instrument, Direction, Leg, TradeProposal
from azr_planner.math_utils import LR_V2_MIN_POINTS
from azr_planner.backtest.core import run_backtest, _get_fill_price
from azr_planner.datasets import load_sp500_sample

# --- Tests for calculate_cagr ---
def test_calculate_cagr_basic() -> None:
    cagr_val = calculate_cagr([100.0] * 126 + [121.0] * 127, num_trading_days_per_year=252)
    assert cagr_val is not None
    assert math.isclose(cagr_val, 0.21)
    cagr_2yr = calculate_cagr([100.0] + [110.0]*252 + [121.0]*252, num_trading_days_per_year=252)
    assert cagr_2yr is not None
    assert math.isclose(cagr_2yr, 0.10, abs_tol=1e-3)

def test_calculate_cagr_edge_cases() -> None:
    assert calculate_cagr([], 252) is None
    assert calculate_cagr([100.0], 252) is None
    assert calculate_cagr([100.0, 100.0], 252) == 0.0
    assert calculate_cagr([0.0, 100.0], 252) is None
    assert calculate_cagr([-10.0, 10.0], 252) is None

# --- Tests for calculate_max_drawdown ---
def test_calculate_max_drawdown_basic() -> None:
    equity_curve = [100.0, 120.0, 90.0, 110.0, 80.0, 100.0]
    mdd = calculate_max_drawdown(equity_curve)
    assert math.isclose(mdd, 1/3)

def test_calculate_max_drawdown_no_drawdown() -> None:
    assert calculate_max_drawdown([100.0, 110.0, 120.0]) == 0.0

def test_calculate_max_drawdown_edge_cases() -> None:
    assert calculate_max_drawdown([]) == 0.0
    assert calculate_max_drawdown([100.0]) == 0.0

# --- Test for metrics_consistency (CAGR & MDD) ---
def test_metrics_consistency_cagr_mdd() -> None:
    days_per_year = 252
    y0, y1, y2, y3 = 100.0, 120.0, 96.0, 115.2
    eq_curve_synthetic: List[float] = []
    eq_curve_synthetic.extend(np.linspace(y0, y1, days_per_year + 1).tolist())
    eq_curve_synthetic.extend(np.linspace(y1, y2, days_per_year + 1).tolist()[1:])
    eq_curve_synthetic.extend(np.linspace(y2, y3, days_per_year + 1).tolist()[1:])
    cagr = calculate_cagr(eq_curve_synthetic, days_per_year)
    assert cagr is not None
    assert math.isclose(cagr, (y3/y0)**(1/3.0) - 1, abs_tol=1e-4)
    mdd = calculate_max_drawdown(eq_curve_synthetic)
    assert math.isclose(mdd, 0.2, abs_tol=1e-4)

# --- Tests for calculate_sharpe_ratio ---
def test_calculate_sharpe_ratio_basic() -> None:
    returns = [0.01, -0.005, 0.015, 0.002, -0.003] * 50
    sharpe = calculate_sharpe_ratio(returns, 0.02, 250)
    assert sharpe is not None
    np.random.seed(42)
    test_returns = (np.random.randn(252) * 0.01 + 0.001078).tolist()
    sharpe_calculated = calculate_sharpe_ratio(test_returns, 0.02, 252)
    assert sharpe_calculated is not None
    fixed_returns = [0.01] * 10 + [-0.01] * 10
    sharpe_fixed = calculate_sharpe_ratio(fixed_returns, 0.0, 252)
    sharpe_fixed_val = sharpe_fixed if sharpe_fixed is not None else 0.0
    assert math.isclose(sharpe_fixed_val, 0.0, abs_tol=1e-9)

def test_calculate_sharpe_ratio_edge_cases() -> None:
    assert calculate_sharpe_ratio([], 0.02) is None
    assert calculate_sharpe_ratio([0.01], 0.02) is None
    assert calculate_sharpe_ratio([0.01, 0.01, 0.01], 0.02) is None

# --- Tests for calculate_sortino_ratio ---
def test_calculate_sortino_ratio_basic() -> None:
    returns = [0.01, -0.005, 0.015, 0.002, -0.003] * 50
    sortino = calculate_sortino_ratio(returns, 0.02, 0.01, 250)
    assert sortino is not None
    all_positive_returns = [0.05, 0.06, 0.07]
    assert calculate_sortino_ratio(all_positive_returns, 0.0, 0.0) is None

def test_calculate_sortino_ratio_edge_cases() -> None:
    assert calculate_sortino_ratio([], 0.02) is None
    assert calculate_sortino_ratio([0.01], 0.02) is None
    assert calculate_sortino_ratio([0.01, 0.01, 0.01], 0.02, 0.01) is None

# --- Tests for calculate_win_rate_and_pnl_stats ---
def test_calculate_win_rate_and_pnl_stats_basic() -> None:
    now = datetime.now(timezone.utc)
    trades = [
        DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=10.0),
        DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=-5.0),
        DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=20.0),
        DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=-2.0),
        DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=0.0),
    ]
    stats = calculate_win_rate_and_pnl_stats(trades)
    assert stats["totalTrades"] == 5
    assert stats["winningTrades"] == 2
    assert stats["losingTrades"] == 2
    assert stats["winRate"] is not None and math.isclose(stats["winRate"], 0.4)
    assert stats["avgWinPnl"] is not None and math.isclose(stats["avgWinPnl"], 15.0)
    assert stats["avgLossPnl"] is not None and math.isclose(stats["avgLossPnl"], -3.5)
    assert stats["avgTradePnl"] is not None and math.isclose(stats["avgTradePnl"], 4.6)
    assert stats["profitFactor"] is not None and math.isclose(stats["profitFactor"], 30.0/7.0)

def test_calculate_win_rate_and_pnl_stats_edge_cases() -> None:
    empty_stats = calculate_win_rate_and_pnl_stats([])
    for key, val in empty_stats.items():
        if key.endswith("Trades"): assert val == 0
        else: assert val is None
    now = datetime.now(timezone.utc)
    all_wins = [DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=10.0)]
    stats_all_wins = calculate_win_rate_and_pnl_stats(all_wins)
    assert stats_all_wins["winRate"] == 1.0
    assert stats_all_wins["profitFactor"] == float('inf')
    all_losses = [DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=-10.0)]
    stats_all_losses = calculate_win_rate_and_pnl_stats(all_losses)
    assert stats_all_losses["winRate"] == 0.0
    assert stats_all_losses["profitFactor"] == 0.0
    no_pnl_trades = [DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=None)]
    stats_no_pnl = calculate_win_rate_and_pnl_stats(no_pnl_trades)
    assert stats_no_pnl["totalTrades"] == 0
    assert stats_no_pnl["winRate"] is None
    zero_pnl_trades = [DailyTrade(timestamp=now, instrument=Instrument.MES, direction=Direction.LONG, size=1.0, fill_price=100.0, pnl=0.0)]
    stats_zero_pnl = calculate_win_rate_and_pnl_stats(zero_pnl_trades)
    assert stats_zero_pnl["totalTrades"] == 1
    assert stats_zero_pnl["winningTrades"] == 0
    assert stats_zero_pnl["losingTrades"] == 0
    assert stats_zero_pnl["winRate"] == 0.0
    assert stats_zero_pnl["profitFactor"] is None

# --- Tests for Backtest Core Logic ---
def _generate_hlc_data_fixture(num_periods: int, start_price: float = 100.0, daily_change: float = 0.1, spread: float = 0.5) -> List[Tuple[float, float, float]]:
    data:List[Tuple[float,float,float]]=[]; current_close=start_price
    for i in range(num_periods):
        h=current_close+spread+abs(daily_change*math.sin(i*0.1)); l=current_close-spread-abs(daily_change*math.cos(i*0.1))
        c=(h+l)/2+(math.sin(i*0.5)*spread*0.1); l=min(l,h-0.01); c=max(min(c,h),l)
        data.append((round(h,2),round(l,2),round(c,2))); current_close=c
    return data

MIN_HISTORY_POINTS_FOR_FIXTURE = LR_V2_MIN_POINTS + 5

@pytest.fixture
def sample_planning_context_data_new()->Dict[str,Any]:
    num_points=MIN_HISTORY_POINTS_FOR_FIXTURE; hlc_data=_generate_hlc_data_fixture(num_periods=num_points)
    eq_curve_data=[10000.0+i*10 for i in range(num_points)]
    return {"timestamp":datetime.now(timezone.utc),"equityCurve":eq_curve_data,"dailyHistoryHLC":hlc_data,
            "dailyVolume":[float(10000+i*100) for i in range(num_points)],
            "currentPositions":[Leg(instrument=Instrument.MES,direction=Direction.LONG,size=2.0,limit_price=4500.0).model_dump()],
            "nSuccesses":10,"nFailures":5,"volSurface":{"MES":0.15,"M2K":0.20},"riskFreeRate":0.02} # Using aliases

@st.composite
def st_planning_context_list(draw: DrawFn) -> List[PlanningContext]:
    num_days=draw(st.integers(min_value=LR_V2_MIN_POINTS+2,max_value=LR_V2_MIN_POINTS+10))
    prices:List[float]=[100.0]
    # The error "Incompatible types in assignment (expression has type "float", variable has type "int")" for prices.append
    # was likely because MyPy got confused earlier. `prices` is correctly List[float].
    for _ in range(num_days-1+LR_V2_MIN_POINTS-1): prices.append(abs(prices[-1]+draw(st.floats(min_value=-2.0,max_value=2.0))))
    contexts:List[PlanningContext]=[]
    for i in range(num_days):
        if i+LR_V2_MIN_POINTS > len(prices): break
        eq_win=prices[i:i+LR_V2_MIN_POINTS]; hlc_win=[(p,p,p) for p in eq_win]
        ts=datetime(2023,1,1,tzinfo=timezone.utc)+timedelta(days=i+LR_V2_MIN_POINTS-1)
        contexts.append(PlanningContext(timestamp=ts,equity_curve=eq_win,daily_history_hlc=hlc_win,
                        daily_volume=None,current_positions=None,vol_surface={"MES":0.2},risk_free_rate=0.01,
                        nSuccesses=draw(st.integers(0,10)),nFailures=draw(st.integers(0,10))))
    assume(len(contexts)>=2); return contexts

@given(contexts_iter=st_planning_context_list())
@settings(deadline=None,suppress_health_check=[HealthCheck.too_slow,HealthCheck.filter_too_much],max_examples=10)
def test_run_backtest_property_basic_execution_and_series_length(contexts_iter:List[PlanningContext])->None:
    report:SingleBacktestReport=run_backtest(contexts_iter); assert isinstance(report,SingleBacktestReport)
    exp_steps=max(0,len(contexts_iter)-1); assert len(report.daily_results)==exp_steps
    assert len(report.equity_curve)==exp_steps+1
    assert report.latent_risk_series is not None and len(report.latent_risk_series)==exp_steps
    assert report.confidence_series is not None and len(report.confidence_series)==exp_steps

def test_run_backtest_insufficient_contexts()->None:
    with pytest.raises(ValueError,match="Context iterator must yield at least two PlanningContexts"): run_backtest([])
    eq=[100.0+i for i in range(LR_V2_MIN_POINTS)]; hlc=[(v,v,v) for v in eq]; ts=datetime(2023,1,1,tzinfo=timezone.utc)
    ctx_manual=[PlanningContext(timestamp=ts,equity_curve=eq,daily_history_hlc=hlc,daily_volume=None,current_positions=None,
                              vol_surface={"MES":0.2},risk_free_rate=0.01,nSuccesses=1,nFailures=0)]
    with pytest.raises(ValueError,match="Context iterator must yield at least two PlanningContexts"): run_backtest(ctx_manual)

def test_run_backtest_missing_fill_price(sample_planning_context_data_new:Dict[str,Any])->None:
    class MockPC:
        equity_curve:Optional[List[float]]
        timestamp:datetime
        def __init__(self,c:Optional[List[float]],t:datetime): self.equity_curve=c;self.timestamp=t
    assert _get_fill_price(cast(PlanningContext,MockPC(None,datetime.now(timezone.utc)))) is None
    assert _get_fill_price(cast(PlanningContext,MockPC([],datetime.now(timezone.utc)))) is None
    d1_data=sample_planning_context_data_new.copy(); d1_data['timestamp']=datetime(2023,1,1,tzinfo=timezone.utc)
    d1=PlanningContext.model_validate(d1_data)
    d2_data=sample_planning_context_data_new.copy(); d2_data['timestamp']=datetime(2023,1,2,tzinfo=timezone.utc)
    d2=PlanningContext.model_validate(d2_data)
    with patch('azr_planner.backtest.core._get_fill_price',return_value=None) as mock_gf:
        rpt=run_backtest([d1,d2]); mock_gf.assert_called_once_with(d2)
        assert len(rpt.daily_results)==1
        assert rpt.equity_curve[1]==rpt.equity_curve[0]
        assert rpt.daily_results[0].portfolio_state_after_trades.daily_pnl==0.0

def test_run_backtest_trade_logic_cover_short(sample_planning_context_data_new:Dict[str,Any],monkeypatch:pytest.MonkeyPatch)->None:
    mock_lr=0.10; mock_conf=0.80
    from azr_planner.engine import DEFAULT_MAX_LEVERAGE,MES_CONTRACT_MULTIPLIER,MIN_CONTRACT_SIZE
    init_short_size=2.0; d1_data=sample_planning_context_data_new.copy()
    d1_data['timestamp']=datetime(2023,1,1,tzinfo=timezone.utc)
    d1_data['currentPositions']=[Leg(instrument=Instrument.MES,direction=Direction.SHORT,size=init_short_size).model_dump()]
    d1=PlanningContext.model_validate(d1_data)
    eq_d1=d1.equity_curve[-1]; price_d1=d1.daily_history_hlc[-1][2]
    from azr_planner.position import position_size as actual_pos_sizer
    exp_dollar_exp=actual_pos_sizer(latent_risk=mock_lr,equity=eq_d1,max_leverage=DEFAULT_MAX_LEVERAGE)
    prop_buy_contracts=0.0
    if price_d1>0 and (price_d1*MES_CONTRACT_MULTIPLIER)>0: prop_buy_contracts=exp_dollar_exp/(price_d1*MES_CONTRACT_MULTIPLIER)

    def mock_gen_plan(ctx:PlanningContext) -> TradeProposal:
        if prop_buy_contracts>=MIN_CONTRACT_SIZE:
            return TradeProposal(action="ENTER",rationale="Mock",latent_risk=mock_lr,confidence=mock_conf,
                                 legs=[Leg(instrument=Instrument.MES,direction=Direction.LONG,size=prop_buy_contracts)])
        return TradeProposal(action="HOLD",rationale="Mock small",latent_risk=mock_lr,confidence=mock_conf,legs=None)
    monkeypatch.setattr('azr_planner.backtest.core.generate_plan',mock_gen_plan)

    d2_data=sample_planning_context_data_new.copy(); d2_data['timestamp']=datetime(2023,1,2,tzinfo=timezone.utc)
    fill_p_d2=d2_data['dailyHistoryHLC'][-1][2]; eq_curve_d2=list(d2_data['equityCurve']); eq_curve_d2[-1]=fill_p_d2
    d2_data['equityCurve']=eq_curve_d2; d2=PlanningContext.model_validate(d2_data)

    report=run_backtest([d1,d2]); assert len(report.daily_results)==1; daily_res=report.daily_results[0]
    tot_size_bought=sum(t.size for t in daily_res.trades_executed if t.instrument==Instrument.MES and t.direction==Direction.LONG)

    if prop_buy_contracts>=MIN_CONTRACT_SIZE:
        assert len(daily_res.trades_executed)>0
        assert math.isclose(tot_size_bought,prop_buy_contracts,rel_tol=1e-5)
        final_pos=daily_res.portfolio_state_after_trades.positions.get(Instrument.MES,0.0)
        assert math.isclose(final_pos,prop_buy_contracts,rel_tol=1e-5)

        opening_long_trade_found = False
        if daily_res.trades_executed:
            if prop_buy_contracts > init_short_size: # Corrected: prop_buy_contracts vs init_short_size
                opening_long_trade_found = any(t.direction == Direction.LONG and t.pnl is None for t in daily_res.trades_executed)
            else:
                opening_long_trade_found = not any(t.direction == Direction.LONG and t.pnl is None for t in daily_res.trades_executed)

        if prop_buy_contracts > 1e-7 :
             if prop_buy_contracts > init_short_size: # Corrected: prop_buy_contracts vs init_short_size
                 assert opening_long_trade_found, "New long portion of trade should have no PNL."
             elif daily_res.trades_executed:
                 assert not opening_long_trade_found, "Cover-only long trade should have PNL."
    else:
        assert not daily_res.trades_executed
        assert daily_res.portfolio_state_after_trades.positions.get(Instrument.MES,0.0)==0.0

def test_run_backtest_smoke_sp500_sample()->None:
    try:smpl_ctxs=load_sp500_sample()
    except FileNotFoundError:pytest.skip("sp500_sample.csv not found, skipping.");return
    if not smpl_ctxs or len(smpl_ctxs)<2:pytest.skip(f"Not enough contexts from sp500_sample.csv. Skipping.");return
    rpt:SingleBacktestReport=run_backtest(smpl_ctxs);assert isinstance(rpt,SingleBacktestReport)
    assert rpt.equity_curve[-1]>rpt.initial_cash*0.95
    assert rpt.metrics.max_drawdown is not None and rpt.metrics.max_drawdown>=0.0
    if rpt.metrics.total_trades>0:
        assert rpt.metrics.win_rate is not None;assert rpt.metrics.profit_factor is not None
    assert len(rpt.daily_results)==len(smpl_ctxs)-1
    if rpt.daily_results:
        assert rpt.daily_results[0].trade_proposal is not None
        assert rpt.daily_results[0].portfolio_state_after_trades is not None


def test_run_backtest_pnl_with_default_multiplier(
    sample_planning_context_data_new: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Tests P&L calculation in run_backtest using DEFAULT_MULTIPLIER.
    1. Enter US_SECTOR_ETF on Day 1.
    2. Exit US_SECTOR_ETF on Day 2, triggering P&L calculation.
    """
    target_instrument = Instrument.US_SECTOR_ETF
    trade_size = 2.0

    # --- Mock generate_plan to control decisions over two days ---
    planner_decisions = [
        TradeProposal(action="ENTER", rationale="Enter ETF", latent_risk=0.1, confidence=0.8,
                      legs=[Leg(instrument=target_instrument, direction=Direction.LONG, size=trade_size)]),
        TradeProposal(action="EXIT", rationale="Exit ETF", latent_risk=0.5, confidence=0.5,
                      legs=[Leg(instrument=target_instrument, direction=Direction.SHORT, size=trade_size)])
    ]
    call_count = 0
    def mock_planner_sequential(ctx: PlanningContext) -> TradeProposal:
        nonlocal call_count
        decision = planner_decisions[call_count]
        call_count += 1
        return decision
    monkeypatch.setattr('azr_planner.backtest.core.generate_plan', mock_planner_sequential)

    # --- Prepare contexts for 3 days (2 decision steps) ---
    ctx_data = sample_planning_context_data_new.copy()

    # Day 1 Context (decision to ENTER)
    ctx_d1_data = ctx_data.copy()
    ctx_d1_data['timestamp'] = datetime(2023, 1, 1, tzinfo=timezone.utc)
    ctx_d1_data['currentPositions'] = None
    ctx_d1 = PlanningContext.model_validate(ctx_d1_data)

    # Day 2 Context (fill for Day 1's ENTER, decision to EXIT)
    ctx_d2_data = ctx_data.copy()
    ctx_d2_data['timestamp'] = datetime(2023, 1, 2, tzinfo=timezone.utc)
    # Ensure equity curve has a value for fill price for Day 1's trade
    # Let fill price for ENTER be the last close of day 2's HLC
    enter_fill_price = ctx_d2_data['dailyHistoryHLC'][-1][2]
    temp_eq_d2 = list(ctx_d2_data['equityCurve']); temp_eq_d2[-1] = enter_fill_price
    ctx_d2_data['equityCurve'] = temp_eq_d2
    ctx_d2 = PlanningContext.model_validate(ctx_d2_data)

    # Day 3 Context (fill for Day 2's EXIT)
    ctx_d3_data = ctx_data.copy()
    ctx_d3_data['timestamp'] = datetime(2023, 1, 3, tzinfo=timezone.utc)
    # Let fill price for EXIT be the last close of day 3's HLC
    exit_fill_price = ctx_d3_data['dailyHistoryHLC'][-1][2]
    temp_eq_d3 = list(ctx_d3_data['equityCurve']); temp_eq_d3[-1] = exit_fill_price
    ctx_d3_data['equityCurve'] = temp_eq_d3
    ctx_d3 = PlanningContext.model_validate(ctx_d3_data)

    contexts = [ctx_d1, ctx_d2, ctx_d3]
    report = run_backtest(contexts)

    assert len(report.daily_results) == 2 # Two decision steps

    # Check Day 1 results (ENTER executed)
    day1_trades = report.daily_results[0].trades_executed
    assert len(day1_trades) == 1
    assert day1_trades[0].instrument == target_instrument
    assert day1_trades[0].direction == Direction.LONG
    assert day1_trades[0].size == trade_size
    assert day1_trades[0].fill_price == enter_fill_price
    assert day1_trades[0].pnl is None # Opening trade

    # Check Day 2 results (EXIT executed, PNL calculated)
    day2_trades = report.daily_results[1].trades_executed
    assert len(day2_trades) == 1
    assert day2_trades[0].instrument == target_instrument
    assert day2_trades[0].direction == Direction.SHORT # This is the sell to close the long
    assert day2_trades[0].size == trade_size
    assert day2_trades[0].fill_price == exit_fill_price
    assert day2_trades[0].pnl is not None # Closing trade, should have PNL

    # Expected PNL = (exit_fill_price - enter_fill_price) * trade_size * DEFAULT_MULTIPLIER (1.0)
    from azr_planner.backtest.core import DEFAULT_MULTIPLIER # Default is 1.0
    expected_pnl = (exit_fill_price - enter_fill_price) * trade_size * DEFAULT_MULTIPLIER
    assert math.isclose(day2_trades[0].pnl, expected_pnl, rel_tol=1e-9)

    # Check final portfolio state (should be flat for target_instrument)
    assert report.daily_results[1].portfolio_state_after_trades.positions.get(target_instrument, 0.0) == 0.0


def test_backtest_skips_when_zero_position_size(
    sample_planning_context_data_new: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    If `position_size` (via generate_plan) effectively returns 0 for an ENTER action,
    the backtester should not open trades, and report metrics should stay neutral.
    """
    # Mock azr_planner.position.position_size to always return 0.0
    # This function is called by engine.generate_plan.
    monkeypatch.setattr('azr_planner.engine.position_size', lambda latent_risk, equity, max_leverage: 0.0)

    # Prepare minimal contexts for run_backtest
    ctx_d1_data = sample_planning_context_data_new.copy()
    ctx_d1_data['timestamp'] = datetime(2023, 1, 1, tzinfo=timezone.utc)
    ctx_d1_data['currentPositions'] = None # Start flat for a clean ENTER attempt
    # Ensure generate_plan would normally ENTER by setting favorable lr/conf if its direct inputs were used.
    # However, generate_plan calls latent_risk_v2 and bayesian_confidence itself.
    # We need latent_risk_v2 to return low risk and bayesian_confidence high confidence.

    # Let's mock these too, so generate_plan initially decides to ENTER
    # This mock will affect the `lr` and `conf` values inside `generate_plan`
    monkeypatch.setattr('azr_planner.engine.latent_risk_v2', lambda equity_curve: 0.10) # Low risk
    monkeypatch.setattr('azr_planner.engine.bayesian_confidence', lambda wins, losses: 0.80) # High confidence

    ctx_d1 = PlanningContext.model_validate(ctx_d1_data)

    # Context for Day 2 (for fill prices) - its content beyond timestamp isn't critical if no trade happens
    ctx_d2_data = sample_planning_context_data_new.copy()
    ctx_d2_data['timestamp'] = datetime(2023, 1, 2, tzinfo=timezone.utc)
    ctx_d2 = PlanningContext.model_validate(ctx_d2_data)

    contexts = [ctx_d1, ctx_d2]
    report = run_backtest(contexts)

    assert len(report.daily_results) == 1 # One decision step occurred
    daily_result = report.daily_results[0]

    # generate_plan should have decided ENTER, but then position_size (mocked to 0)
    # would lead to calculated_size = 0. Then generate_plan changes action to HOLD.
    assert daily_result.trade_proposal is not None
    assert daily_result.trade_proposal.action == "HOLD" # Crucial check
    assert not daily_result.trades_executed # No trades executed

    # Check overall report metrics
    assert report.metrics.total_trades == 0
    # Total return should be 0 if initial_cash == final_equity
    assert math.isclose(report.final_equity, report.initial_cash, rel_tol=1e-9)
    assert report.metrics.max_drawdown == 0.0
    # Sharpe and Sortino would be None or NaN if returns are all zero
    if report.metrics.sharpe_ratio is not None: # Can be NaN which is not None
        assert math.isnan(report.metrics.sharpe_ratio) or math.isclose(report.metrics.sharpe_ratio, 0.0)

    # Equity curve should be flat
    assert all(math.isclose(val, report.initial_cash) for val in report.equity_curve)
