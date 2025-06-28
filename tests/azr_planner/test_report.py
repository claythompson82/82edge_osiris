from __future__ import annotations

import datetime
import math
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

import pytest # For fixture if needed, and tempfile usage
from hypothesis import given, strategies as st, settings, HealthCheck

# Assuming azr_planner is importable
from azr_planner.backtest.report import generate_html_report
from azr_planner.backtest.schemas import WalkForwardBacktestReport # For creating dummy report instances

# --- Strategies for Hypothesis ---

@st.composite
def st_walk_forward_report_data(draw: st.DrawFn) -> Dict[str, Any]:
    """Generates data similar to WalkForwardBacktestReport.model_dump()."""
    data = {
        "report_generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "from_date": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=draw(st.integers(30, 365)))).isoformat(),
        "to_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mean_sharpe": draw(st.one_of(st.none(), st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))),
        "worst_drawdown": draw(st.one_of(st.none(), st.floats(min_value=0, max_value=1.0, allow_nan=False))),
        "total_return": draw(st.one_of(st.none(), st.floats(min_value=-1.0, max_value=10.0, allow_nan=False))),
        "trades": draw(st.integers(min_value=0, max_value=1000)),
        "win_rate": draw(st.one_of(st.none(), st.floats(min_value=0, max_value=1.0, allow_nan=False))),
    }
    return data

@st.composite
def st_equity_curve_data(draw: st.DrawFn, min_size: int = 10, max_size: int = 200) -> Tuple[List[datetime.datetime], List[float]]:
    """Generates a list of timestamps and corresponding equity values."""
    size = draw(st.integers(min_value=min_size, max_value=max_size)) # Changed to min_value, max_value
    start_date = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    timestamps = [start_date + datetime.timedelta(days=i) for i in range(size)]

    current_equity = 100_000.0
    equity_values = [current_equity]
    for _ in range(size - 1):
        change = draw(st.floats(min_value=-0.05, max_value=0.05)) # Daily change factor
        current_equity *= (1 + change)
        current_equity = max(1.0, current_equity) # Ensure equity stays positive for simplicity here
        equity_values.append(current_equity)

    # Ensure equity_values has 'size' elements if loop is for size-1
    if len(equity_values) < size and size > 0 : # Should only happen if size = 0 or 1 initially
        while len(equity_values) < size: equity_values.append(equity_values[-1])
    elif size == 0:
        return [],[]


    return timestamps, equity_values

@st.composite
def st_trade_pnls(draw: st.DrawFn, max_trades: int = 100) -> List[float]:
    """Generates a list of trade P&Ls."""
    num_trades = draw(st.integers(min_value=0, max_value=max_trades))
    return draw(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=num_trades, max_size=num_trades))


# --- Functional Test ---

def test_generate_html_report_functional() -> None: # Added return type
    """
    Functional test: generate a report to a temp file and assert
    out_path.exists() and that the file contains key phrases.
    """
    dummy_wf_report_data: Dict[str, Any] = {
        "report_generated_at": datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc),
        "from_date": datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        "to_date": datetime.datetime(2023, 12, 31, tzinfo=datetime.timezone.utc),
        "mean_sharpe": 1.5,
        "worst_drawdown": 0.15, # 15%
        "total_return": 0.25,   # 25%
        "trades": 100,
        "win_rate": 0.60,       # 60%
    }

    num_points = 100
    start_ts = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    dummy_timestamps = [start_ts + datetime.timedelta(days=i) for i in range(num_points)]
    dummy_strategy_equity = [float(100000 + i*100) for i in range(num_points)] # Cast to float
    dummy_bnh_equity = [float(100000 + i*80) for i in range(num_points)]      # Cast to float
    dummy_pnls = [float(p) for p in ([10, -5, 20, 15, -10] * 20)]             # Cast to float

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_report.html"

        generate_html_report(
            walk_forward_report_data=dummy_wf_report_data,
            strategy_equity_curve=dummy_strategy_equity,
            strategy_equity_timestamps=dummy_timestamps,
            buy_and_hold_equity_curve=dummy_bnh_equity,
            buy_and_hold_timestamps=dummy_timestamps, # Assume same timestamps for this test
            trade_pnls=dummy_pnls,
            out_path=out_path
        )

        assert out_path.exists()
        content = out_path.read_text()
        assert "Walk-Forward Backtest Report" in content
        assert "Mean Sharpe Ratio" in content
        assert "1.50" in content # Sharpe value
        assert "Worst Drawdown" in content
        assert "15.00%" in content # Drawdown value
        assert "Equity Curve" in content
        assert 'img src="data:image/png;base64,' in content # Check for embedded image
        assert "Trade P&L Histogram" in content
        assert "100" in content # Trade count

# --- Property Test ---

@given(
    wf_report_data=st_walk_forward_report_data(),
    strategy_eq_data=st_equity_curve_data(min_size=60, max_size=150), # Equity curve >= 60 points
    bnh_eq_data=st_equity_curve_data(min_size=60, max_size=150),       # Must align with strategy for plotting
    trade_pnls_list=st_trade_pnls(max_trades=50)
)
@settings(
    deadline=None, # Matplotlib can be slow
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    max_examples=30 # Keep low due to file I/O and plotting
)
def test_generate_html_report_property_no_exceptions(
    wf_report_data: Dict[str, Any],
    strategy_eq_data: Tuple[List[datetime.datetime], List[float]],
    bnh_eq_data: Tuple[List[datetime.datetime], List[float]],
    trade_pnls_list: List[float]
) -> None:
    """
    Property test: generate_html_report with random (but validly structured)
    data should not raise exceptions.
    Equity curve >= 60 points.
    """
    strat_ts, strat_eq = strategy_eq_data

    # For simplicity in this property test, make B&H align with strategy curve length if generated independently
    # Or, ensure the generation strategy for bnh_eq_data uses the same size as strat_eq_data
    # Here, we'll just use the strategy timestamps for B&H and regenerate B&H values if lengths differ.
    # A more robust way is to have st_equity_curve_data generate a single size and use it for both.

    # Crude alignment: if bnh_eq_data is different length, we can't really plot it meaningfully
    # against strategy curve without proper date alignment. The test focuses on not crashing.
    # So, we can pass None for B&H if alignment is hard, or make them same length.

    bnh_ts, bnh_eq = bnh_eq_data
    aligned_bnh_eq = None
    aligned_bnh_ts = None

    if len(strat_ts) == len(bnh_ts): # If they happen to be same length, use them
        aligned_bnh_ts = bnh_ts
        aligned_bnh_eq = bnh_eq
    elif strat_ts: # If strategy curve exists, create a dummy B&H of same length
        aligned_bnh_ts = strat_ts
        first_price = strat_eq[0] if strat_eq else 1.0
        if first_price == 0: first_price = 1.0 # Avoid division by zero
        aligned_bnh_eq = [eq / first_price * 100000 for eq in strat_eq] # Dummy B&H based on strat shape


    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "property_test_report.html"
        try:
            generate_html_report(
                walk_forward_report_data=wf_report_data,
                strategy_equity_curve=strat_eq if strat_ts else None,
                strategy_equity_timestamps=strat_ts if strat_ts else None,
                buy_and_hold_equity_curve=aligned_bnh_eq if aligned_bnh_ts else None,
                buy_and_hold_timestamps=aligned_bnh_ts if aligned_bnh_ts else None,
                trade_pnls=trade_pnls_list if trade_pnls_list else None, # Pass None if empty list
                out_path=out_path
            )
            assert out_path.exists() # Basic check that file was created
            # Further content checks could be added but might make test brittle
        except Exception as e:
            pytest.fail(f"generate_html_report raised an exception: {e} with inputs:\n"
                        f"WF Report: {wf_report_data}\n"
                        f"Strategy TS: {len(strat_ts) if strat_ts else 'None'} points\n"
                        f"Strategy EQ: {len(strat_eq) if strat_eq else 'None'} points\n"
                        f"B&H TS: {len(aligned_bnh_ts) if aligned_bnh_ts else 'None'} points\n"
                        f"B&H EQ: {len(aligned_bnh_eq) if aligned_bnh_eq else 'None'} points\n"
                        f"P&Ls: {len(trade_pnls_list) if trade_pnls_list else 'None'} items")

# Test with completely missing optional data for charts
def test_generate_html_report_missing_chart_data() -> None: # Added return type
    dummy_wf_report_data: Dict[str, Any] = {
        "report_generated_at": datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc),
        "from_date": datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        "to_date": datetime.datetime(2023, 12, 31, tzinfo=datetime.timezone.utc),
        "mean_sharpe": 0.5, "worst_drawdown": 0.1, "total_return": 0.05, "trades": 10, "win_rate": 0.5
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "missing_data_report.html"
        generate_html_report(
            walk_forward_report_data=dummy_wf_report_data,
            strategy_equity_curve=None,
            strategy_equity_timestamps=None,
            buy_and_hold_equity_curve=None,
            buy_and_hold_timestamps=None,
            trade_pnls=None,
            out_path=out_path
        )
        assert out_path.exists()
        content = out_path.read_text()
        assert "Equity curve data not available or mismatched." in content
        assert "No trade P&L data available for histogram." in content
        assert "Mean Sharpe Ratio" in content # Metrics table should still be there
