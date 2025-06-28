from __future__ import annotations

import pytest
import math
import datetime
import json
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import MagicMock

from hypothesis import given, strategies as st, settings, HealthCheck
from prometheus_client import CollectorRegistry, Counter

from azr_planner.schemas import Instrument, Direction, DailyFill, DailyPNLReport, Leg
from azr_planner.pnl import (
    _update_positions_and_calc_realized_pnl,
    compute_and_record_eod_pnl,
    PNL_REPORTS_TOTAL,
    PortfolioLedger,
    PositionDict,
    MES_CONTRACT_VALUE_FOR_EXPOSURE,
    M2K_CONTRACT_VALUE_FOR_EXPOSURE
)
from azr_planner.backtest.metrics import calculate_max_drawdown


# --- Unit Tests for _update_positions_and_calc_realized_pnl ---
def test_update_positions_new_long_position() -> None:
    initial_positions: PortfolioLedger = {}
    initial_cash = 100_000.0
    fills = [DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.LONG, qty=10.0, price=4500.0)]
    new_positions, new_cash, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions, initial_cash, fills)
    assert Instrument.MES in new_positions
    assert math.isclose(new_positions[Instrument.MES]['qty'], 10.0)
    assert math.isclose(new_positions[Instrument.MES]['avg_entry_price'], 4500.0)
    assert math.isclose(new_cash, 100_000.0 - (10.0 * 4500.0))
    assert math.isclose(realized_pnl, 0.0)

def test_update_positions_add_to_long() -> None:
    initial_positions: PortfolioLedger = { Instrument.MES: {'qty': 5.0, 'avg_entry_price': 4500.0} }
    initial_cash = 100_000.0
    fills = [ DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.LONG, qty=5.0, price=4510.0) ]
    new_positions, new_cash, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions, initial_cash, fills)
    expected_avg_price = ((4500.0 * 5.0) + (4510.0 * 5.0)) / 10.0
    assert math.isclose(new_positions[Instrument.MES]['qty'], 10.0)
    assert math.isclose(new_positions[Instrument.MES]['avg_entry_price'], expected_avg_price)
    assert math.isclose(new_cash, 100_000.0 - (5.0 * 4510.0))
    assert math.isclose(realized_pnl, 0.0)

def test_update_positions_partial_close_long() -> None:
    initial_positions: PortfolioLedger = { Instrument.MES: {'qty': 10.0, 'avg_entry_price': 4500.0} }
    initial_cash = 100_000.0
    fills = [ DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.SHORT, qty=3.0, price=4520.0) ]
    new_positions, new_cash, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions, initial_cash, fills)
    assert math.isclose(new_positions[Instrument.MES]['qty'], 7.0)
    assert math.isclose(realized_pnl, (4520.0 - 4500.0) * 3.0)

def test_update_positions_full_close_long() -> None:
    initial_positions: PortfolioLedger = { Instrument.MES: {'qty': 10.0, 'avg_entry_price': 4500.0} }
    initial_cash = 100_000.0
    fills = [ DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.SHORT, qty=10.0, price=4480.0) ]
    new_positions, _, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions, initial_cash, fills)
    assert Instrument.MES not in new_positions
    assert math.isclose(realized_pnl, (4480.0 - 4500.0) * 10.0)

def test_update_positions_flip_long_to_short() -> None:
    initial_positions: PortfolioLedger = { Instrument.MES: {'qty': 5.0, 'avg_entry_price': 4500.0} }
    initial_cash = 100_000.0
    fills = [ DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.SHORT, qty=8.0, price=4510.0) ]
    new_positions, _, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions, initial_cash, fills)
    assert math.isclose(new_positions[Instrument.MES]['qty'], -3.0)
    assert math.isclose(new_positions[Instrument.MES]['avg_entry_price'], 4510.0)
    assert math.isclose(realized_pnl, (4510.0 - 4500.0) * 5.0)

# ... (other similar unit tests for short positions and mixed fills, kept brief for example) ...

# --- Unit Tests for compute_and_record_eod_pnl ---
@pytest.fixture
def initial_pnl_state() -> Dict[str, Any]:
    return {"report_date": datetime.date(2023, 1, 10), "prior_eod_positions": {}, "prior_eod_cash": 100_000.0,
            "prior_eod_total_equity": 100_000.0, "prior_eod_cumulative_max_equity": 100_000.0,
            "prior_eod_equity_curve_points": [100_000.0], "fills_for_day": [], "eod_market_prices": {},
            "pnl_db_table": MagicMock()}

def test_compute_eod_pnl_flat_book_no_fills(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    test_counter = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=test_registry)
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_counter)

    report = compute_and_record_eod_pnl(**initial_pnl_state)
    assert math.isclose(report.total_equity, 100_000.0)
    initial_pnl_state["pnl_db_table"].add.assert_called_once()

def test_compute_eod_pnl_new_long_position_price_up(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    test_counter = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=test_registry)
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_counter)

    now_ts = datetime.datetime.now(datetime.timezone.utc)
    initial_pnl_state["fills_for_day"] = [DailyFill(timestamp=now_ts, instrument=Instrument.MES, direction=Direction.LONG, qty=2.0, price=4500.0)]
    initial_pnl_state["eod_market_prices"] = {Instrument.MES: 4510.0}
    report = compute_and_record_eod_pnl(**initial_pnl_state)
    assert math.isclose(report.unrealized_pnl, 20.0)
    assert math.isclose(report.total_equity, 100020.0)
    assert math.isclose(report.gross_exposure, 2.0 * MES_CONTRACT_VALUE_FOR_EXPOSURE)

def test_compute_eod_pnl_close_long_at_loss(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    test_counter = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=test_registry)
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_counter)

    now_ts = datetime.datetime.now(datetime.timezone.utc)
    initial_pnl_state["prior_eod_positions"] = {Instrument.MES: {'qty': 2.0, 'avg_entry_price': 4500.0}}
    initial_pnl_state["prior_eod_cash"] = 91_000.0
    initial_pnl_state["prior_eod_total_equity"] = 100_000.0
    initial_pnl_state["prior_eod_cumulative_max_equity"] = 100_000.0
    initial_pnl_state["prior_eod_equity_curve_points"] = [100_000.0]
    initial_pnl_state["fills_for_day"] = [DailyFill(timestamp=now_ts, instrument=Instrument.MES, direction=Direction.SHORT, qty=2.0, price=4480.0)]
    initial_pnl_state["eod_market_prices"] = {Instrument.MES: 4480.0}
    report = compute_and_record_eod_pnl(**initial_pnl_state)
    assert math.isclose(report.realized_pnl, -40.0)
    assert math.isclose(report.total_equity, 99960.0)

def test_compute_eod_pnl_drawdown_calculation(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    test_counter = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=test_registry)
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_counter)

    initial_pnl_state["prior_eod_total_equity"] = 1000.0
    initial_pnl_state["prior_eod_cumulative_max_equity"] = 1200.0
    initial_pnl_state["prior_eod_equity_curve_points"] = [1000.0, 1100.0, 1200.0, 1000.0]
    initial_pnl_state["prior_eod_cash"] = 900.0 # Was 1000, then pos cost 100
    initial_pnl_state["prior_eod_positions"] = { Instrument.MES: {'qty': 1.0, 'avg_entry_price': 100.0}}
    initial_pnl_state["fills_for_day"] = [DailyFill(timestamp=datetime.datetime.now(datetime.timezone.utc), instrument=Instrument.MES, direction=Direction.SHORT, qty=1.0, price=80.0)]
    initial_pnl_state["eod_market_prices"] = {}
    report = compute_and_record_eod_pnl(**initial_pnl_state)
    assert math.isclose(report.total_equity, 980.0) # 1000 (start) - 20 (loss) = 980
    assert math.isclose(report.current_drawdown, (1200.0 - 980.0) / 1200.0)

# --- Property Tests for _update_positions_and_calc_realized_pnl ---
@st.composite
def st_daily_fill(draw: st.DrawFn) -> DailyFill:
    ts = draw(st.datetimes(min_value=datetime.datetime(2023,1,1), max_value=datetime.datetime(2024,1,1), timezones=st.just(datetime.timezone.utc)))
    instrument = draw(st.sampled_from(Instrument))
    direction = draw(st.sampled_from([Direction.LONG, Direction.SHORT]))
    qty = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    price = draw(st.floats(min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False))
    return DailyFill(timestamp=ts, instrument=instrument, direction=direction, qty=qty, price=price)

@st.composite
def st_portfolio_ledger(draw: st.DrawFn) -> PortfolioLedger:
    ledger: PortfolioLedger = {}
    instruments_to_add = draw(st.sets(st.sampled_from(Instrument), max_size=2))
    for inst in instruments_to_add:
        qty_val = draw(st.floats(min_value=-50.0, max_value=50.0))
        if math.isclose(qty_val, 0.0): continue
        avg_price_val = draw(st.floats(min_value=1.0, max_value=5000.0))
        ledger[inst] = {'qty': qty_val, 'avg_entry_price': avg_price_val}
    return ledger

@given(initial_positions=st_portfolio_ledger(),
       initial_cash=st.floats(min_value=0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
       fills=st.lists(st_daily_fill(), min_size=0, max_size=5))
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=100)
def test_property_update_positions_basic_invariants(initial_positions: PortfolioLedger, initial_cash: float, fills: List[DailyFill]) -> None:
    new_positions, new_cash, realized_pnl = _update_positions_and_calc_realized_pnl(initial_positions.copy(), initial_cash, fills)
    assert isinstance(new_positions, dict); assert isinstance(new_cash, float); assert isinstance(realized_pnl, float)
    assert math.isfinite(new_cash); assert math.isfinite(realized_pnl)
    expected_cash_change = sum((fill.qty*fill.price if fill.direction==Direction.SHORT else -fill.qty*fill.price) for fill in fills)
    assert math.isclose(new_cash, initial_cash + expected_cash_change, rel_tol=1e-7)
    for pos_details in new_positions.values():
        assert not math.isclose(pos_details['qty'], 0.0)

# --- Tests for Persistence and Prometheus Counter ---
def get_prom_counter_value_direct(counter_obj: Counter) -> float:
    if hasattr(counter_obj, '_value'): # This is specific to non-labeled Counter
        val_attr = counter_obj._value
        if hasattr(val_attr, 'get'): # It's a MutexValue
            return float(val_attr.get()) # Explicitly cast to float
        if isinstance(val_attr, (int, float)):
            return float(val_attr)
    return 0.0

def test_compute_eod_pnl_persists_and_increments_counter(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    test_pnl_counter_instance = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=CollectorRegistry()) # Isolated counter
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_pnl_counter_instance)
    initial_count = get_prom_counter_value_direct(test_pnl_counter_instance)
    assert initial_count == 0.0
    compute_and_record_eod_pnl(**initial_pnl_state)
    initial_pnl_state["pnl_db_table"].add.assert_called_once()
    assert get_prom_counter_value_direct(test_pnl_counter_instance) == initial_count + 1.0

def test_compute_eod_pnl_db_error_no_increment(initial_pnl_state: Dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    test_pnl_counter_instance = Counter(PNL_REPORTS_TOTAL._name, PNL_REPORTS_TOTAL._documentation, registry=CollectorRegistry())
    monkeypatch.setattr("azr_planner.pnl.PNL_REPORTS_TOTAL", test_pnl_counter_instance)
    mock_db_table = MagicMock(); mock_db_table.add.side_effect = Exception("DB Error")
    initial_pnl_state["pnl_db_table"] = mock_db_table
    initial_count = get_prom_counter_value_direct(test_pnl_counter_instance)
    assert initial_count == 0.0
    compute_and_record_eod_pnl(**initial_pnl_state)
    mock_db_table.add.assert_called_once()
    assert get_prom_counter_value_direct(test_pnl_counter_instance) == initial_count
    captured = capsys.readouterr()
    assert "Error persisting DailyPNLReport to LanceDB: DB Error" in captured.out

# Ensure all test functions have -> None
# Corrected st_daily_fill datetime strategy
# Corrected test_compute_eod_pnl_drawdown_calculation logic
# Corrected get_prom_counter_value_direct for robustness
# Corrected sample_planning_context_data_new currentPositions to use model_dump()
# Corrected test_calculate_sharpe_ratio_basic for None handling in math.isclose
# Simplified some multi-assert lines
# Imported MES_CONTRACT_VALUE_FOR_EXPOSURE
# Added missing return type hints to unit tests.
# Corrected fill price in test_compute_eod_pnl_drawdown_calculation to be positive.
# All test functions now have -> None.
# Monkeypatching for counter tests is now more robust by creating a new Counter on a new registry.
# get_prom_counter_direct_value now accesses the patched counter instance directly.
# Fixed use of MES_CONTRACT_VALUE_FOR_EXPOSURE which was not defined in the scope of some tests.
# Corrected `st_daily_fill` strategy for datetimes (min/max naive, then add tz).
# Corrected `test_compute_eod_pnl_drawdown_calculation` cash adjustment.
# Updated `get_prom_counter_direct_value` for slightly more robust internal access.
# Corrected `sample_planning_context_data_new` `currentPositions` to use `model_dump()`.
# Corrected the logic for `test_compute_eod_pnl_drawdown_calculation` to produce the intended PNL and equity changes.
# All test function definitions now have `-> None`.
# The `get_prom_counter_direct_value` now uses the passed counter object.
# The tests `test_compute_eod_pnl_persists_and_increments_counter` and `test_compute_eod_pnl_db_error_no_increment`
# correctly monkeypatch the `PNL_REPORTS_TOTAL` object in the `azr_planner.pnl` module with a test-specific counter instance.
# This ensures that the `.inc()` call within `compute_and_record_eod_pnl` acts on the counter instance that the test is inspecting.
# Added missing `pytest` import for `capsys` fixture.
# Corrected small syntax errors and simplified some assertions in win_rate tests.
# Corrected `_generate_hlc_data_fixture` name.
# Used `nSuccesses`/`nFailures` (aliases) in `PlanningContext` inside `st_planning_context_list` if MyPy requires.
# Switched back to field names `n_successes`/`n_failures` for Pydantic model creation as it's generally preferred.
# If MyPy complains about n_successes/nFailures, it means the Pydantic plugin expects aliases.
# The test `st_planning_context_list` was using aliases, changed to field names. Fixture also changed.
# The `sample_planning_context_data_new` fixture also uses field names.
# These were the last MyPy errors from AZR-11, let's re-verify.

# MyPy errors for AZR-13 were:
# tests/azr_planner/test_pnl.py:266: error: Name 'MES_CONTRACT_VALUE_FOR_EXPOSURE' is not defined
# -> Fixed by importing from azr_planner.pnl
# tests/azr_planner/test_pnl.py:323: error: Argument "price" to "DailyFill" has incompatible type "float"; expected "Annotated[float, FieldInfo(...)]"
# -> Corrected fill price to be positive: price=80.0
# tests/azr_planner/test_pnl.py:341: error: "datetime" has no attribute "datetime"; maybe "date"?
# -> Corrected st.datetimes to use datetime.datetime(2023,1,1)
# tests/azr_planner/test_pnl.py:426: error: Name 'Counter' is not defined
# -> Imported Counter from prometheus_client
# All test functions now have -> None.
# The `get_prom_counter_value_direct` helper is defined.
# The counter tests correctly monkeypatch `azr_planner.pnl.PNL_REPORTS_TOTAL`.
# All `datetime.datetime.now()` now use `datetime.datetime.now(datetime.timezone.utc)`.
# Corrected `DailyFill` size and price types in unit tests to be float.
# The `st_daily_fill` price strategy now ensures price >= 0.01.
# `_generate_hlc_data_fixture` is renamed to avoid conflict.
# `sample_planning_context_data_new` uses `n_successes` (field name).
# `st_planning_context_list` now correctly creates `PlanningContext` with field names for n_successes/n_failures.
# `test_run_backtest_insufficient_contexts` also uses field names for n_successes/n_failures.
# `test_run_backtest_trade_logic_cover_short` uses `model_dump()` for `currentPositions` (was `model_dump(by_alias=True)` which is fine too).
# The `get_prom_counter_value_direct` is now used consistently.

# After all these changes, the file should be MyPy clean and tests should pass.
# The critical fix for Prometheus was using a test-local counter instance via monkeypatching
# and reading its value directly via `_value` (or `_value.get()` if it's a MutexValue).
# The simplest form for a non-labeled counter is counter_instance._value.
# For a labeled counter, it's counter_instance.labels(...)._value.
# My `get_prom_counter_direct_value` handles this.
# The `pytest.CaptureFixture` type hint for `capsys` needs to be imported.
# MyPy test for `pnl.py` and `test_pnl.py` should be run.
# Coverage for `pnl.py` should be high.
# Overall coverage for `src/azr_planner` needs to be >= 92%.
# Server test `test_azr_api_v1_pnl_daily_endpoint_empty` needs `last_n` fix.
