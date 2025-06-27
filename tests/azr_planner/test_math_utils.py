"""Tests for AZR Planner math_utils."""

import pytest
import math
from azr_planner.math_utils import (
    calculate_mean,
    calculate_std_dev,
    ema,
    rolling_volatility,
    latent_risk,
)
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict
import pandas as pd

# --- Existing tests (calculate_mean, calculate_std_dev) ---
def test_calculate_mean() -> None:
    """Test calculate_mean function."""
    assert calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
    assert calculate_mean([10.0]) == 10.0
    assert calculate_mean([-1.0, 1.0]) == 0.0
    assert calculate_mean([1.5, 2.5, 3.5]) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="Input list cannot be empty"):
        calculate_mean([])

def test_calculate_std_dev() -> None:
    """Test calculate_std_dev function (sample standard deviation)."""
    assert calculate_std_dev([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))
    assert calculate_std_dev([1.0, 3.0]) == pytest.approx(math.sqrt(2.0))
    assert calculate_std_dev([5.0, 5.0, 5.0, 5.0]) == 0.0
    assert calculate_std_dev([-1.0, -2.0, -3.0]) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Standard deviation requires at least two data points."):
        calculate_std_dev([1.0])
    with pytest.raises(ValueError, match="Standard deviation requires at least two data points."):
        calculate_std_dev([])

# --- Strategies for Hypothesis for new functions ---
st_price_like_series = st.lists(
    st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=100
)
st_return_like_series = st.lists(
    st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=100
)
st_span_or_window = st.integers(min_value=1, max_value=50)

# --- Tests for ema ---
@given(series=st_price_like_series, span=st_span_or_window)
@settings(max_examples=50, deadline=None)
def test_ema_hypothesis(series: List[float], span: int) -> None:
    assume(len(series) > 0 and span > 0)
    assume(span <= len(series) * 2 + 5)

    result = ema(series, span)
    assert isinstance(result, float)

    pd_series = pd.Series(series)
    expected_ema_series = pd_series.ewm(span=span, adjust=False).mean()

    if expected_ema_series.empty or pd.isna(expected_ema_series.iloc[-1]):
        assert math.isnan(result)
    else:
        assert math.isclose(result, expected_ema_series.iloc[-1])

def test_ema_known_values() -> None:
    series = [2.0, 4.0, 6.0, 8.0]
    span = 3
    assert math.isclose(ema(series, span), 6.25)

def test_ema_edge_cases() -> None:
    assert math.isclose(ema([10.0], 5), 10.0)
    assert math.isclose(ema([10.0, 12.0], 1), 12.0)
    with pytest.raises(ValueError, match="Series cannot be empty"):
        ema([], 5)
    with pytest.raises(ValueError, match="Span must be positive"):
        ema([1.0, 2.0], 0)

# --- Tests for rolling_volatility ---
@given(series=st_return_like_series, window=st_span_or_window)
@settings(max_examples=50, deadline=None)
def test_rolling_volatility_hypothesis(series: List[float], window: int) -> None:
    assume(len(series) > 0 and window > 0)

    result = rolling_volatility(series, window)
    assert isinstance(result, float)

    if len(series) < window:
        assert math.isnan(result)
    elif window == 1:
        # pandas .rolling(window=1).std() with default ddof=1 results in NaN
        assert math.isnan(result)
    else: # len(series) >= window and window > 1
        pd_series = pd.Series(series)
        expected_std = pd_series.rolling(window=window).std().iloc[-1] # Default ddof=1
        if pd.isna(expected_std):
             # This can happen if all values in the window are identical
             assert math.isclose(result, 0.0) # Annualized vol of 0 std dev is 0
        else:
            assert not math.isnan(result)
            assert result >= 0.0
            expected_vol = expected_std * math.sqrt(252)
            assert math.isclose(result, expected_vol, rel_tol=1e-7)

def test_rolling_volatility_known_values() -> None:
    series = [0.01, -0.01, 0.01, -0.01, 0.01]
    window = 5
    expected_std = pd.Series(series).rolling(window=window).std().iloc[-1]
    expected_annualized_vol = expected_std * math.sqrt(252)
    assert math.isclose(rolling_volatility(series, window), expected_annualized_vol)

    series_flat_non_zero = [0.005] * 10
    assert math.isclose(rolling_volatility(series_flat_non_zero, 5), 0.0)

    series_flat_zero = [0.0] * 10
    assert math.isclose(rolling_volatility(series_flat_zero, 5), 0.0)

def test_rolling_volatility_edge_cases() -> None:
    assert math.isnan(rolling_volatility([0.01, 0.02], 5)) # window > len(series)
    assert math.isnan(rolling_volatility([0.01], 1))      # window = 1

    with pytest.raises(ValueError, match="Window must be positive"):
        rolling_volatility([0.01, 0.02], 0)
    with pytest.raises(ValueError, match="Series cannot be empty"): # Corrected check order in source
        rolling_volatility([], 1)


# --- Tests for latent_risk (placeholder) ---
st_equity_curve = st.lists(st.floats(min_value=1.0, max_value=2000.0, allow_nan=False, allow_infinity=False), min_size=30, max_size=100)
st_vol_surface = st.dictionaries(
    keys=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu', 'Nd'))),
    values=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_size=0, max_size=10
)
st_risk_free_rate_for_lr = st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False)

@given(ec=st_equity_curve, vs=st_vol_surface, rfr=st_risk_free_rate_for_lr)
@settings(max_examples=50, deadline=None)
def test_latent_risk_placeholder_properties(ec: List[float], vs: Dict[str, float], rfr: float) -> None:
    assume(len(ec) >= 30)
    risk = latent_risk(ec, vs, rfr)
    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0, "Latent risk must be clamped between 0 and 1."

def test_latent_risk_edge_cases() -> None:
    assert latent_risk([100.0] * 10, {"MES": 0.2}, 0.01) == 1.0

    risk_empty_vs = latent_risk([100.0 + i/10.0 for i in range(50)], {}, 0.01)
    assert 0.0 <= risk_empty_vs <= 1.0

    risk_flat_ec = latent_risk([100.0] * 50, {"MES": 0.2}, 0.01)
    expected_risk_flat_ec = ((0.0 + (0.2 / 0.5) * 0.5) / 2.0) + 0.0
    assert math.isclose(risk_flat_ec, expected_risk_flat_ec )

    risk_high_vs = latent_risk([100.0 + i/10.0 for i in range(50)], {"MES": 2.0, "SPY": 1.8}, 0.01)
    assert 0.0 < risk_high_vs <= 1.0

    risk_high_rfr = latent_risk([100.0 + i/10.0 for i in range(50)], {"MES": 0.2}, 0.10)
    risk_low_rfr = latent_risk([100.0 + i/10.0 for i in range(50)], {"MES": 0.2}, 0.01)
    assert risk_high_rfr >= risk_low_rfr


# Docstring checks
assert calculate_mean.__doc__ is not None
assert calculate_std_dev.__doc__ is not None
assert ema.__doc__ is not None
assert rolling_volatility.__doc__ is not None
assert latent_risk.__doc__ is not None
