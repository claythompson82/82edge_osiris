"""Tests for azr_planner.math_utils."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, strategies as st
from hypothesis.strategies import DrawFn # For type hinting draw in composite strategies
from pydantic import ValidationError
from pydantic import TypeAdapter

from azr_planner.math_utils import (
    calculate_mean,
    calculate_std_dev,
    ema,
    rolling_volatility,
    latent_risk, # Old v1 latent risk
    atr,
    kelly_fraction,
    bayesian_confidence,
    latent_risk_v2, # New v2 latent risk
    LR_V2_MIN_POINTS # Constant for latent_risk_v2
)
from azr_planner.schemas import TradeProposal
from unittest.mock import patch # For mocking in latent_risk_v2 tests
from hypothesis import HealthCheck # For suppressing health checks


# Helper to generate HLC data for ATR tests
def _generate_hlc_data(num_periods: int, start_price: float = 100.0, daily_change: float = 1.0, spread: float = 0.5) -> List[tuple[float, float, float]]:
    data = []
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + (daily_change if i % 2 == 0 else 0)
        low = current_close - spread - (daily_change if i % 2 != 0 else 0)
        close = (high + low) / 2
        if low > high: low = high - 0.01
        if close > high: close = high
        if close < low: close = low
        data.append((high, low, close))
        current_close = close
    return data

@st.composite
def st_single_hlc_tuple_for_math_tests(draw: DrawFn) -> tuple[float, float, float]:
    f1 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f2 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f3 = draw(st.floats(min_value=1.0, max_value=1999.0))
    s = sorted([f1, f2, f3])
    low, _, high = s[0], s[1], s[2]
    if high <= low:
        high = low + draw(st.floats(min_value=1e-6, max_value=1.0))
    close = draw(st.floats(min_value=low, max_value=high))
    return round(high, 2), round(low, 2), round(close, 2)

st_hlc_data = st.lists(
    st_single_hlc_tuple_for_math_tests(),
    min_size=2,
    max_size=100
)

def test_calculate_mean() -> None:
    assert calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
    with pytest.raises(ValueError): calculate_mean([])

def test_calculate_std_dev() -> None:
    assert calculate_std_dev([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))
    with pytest.raises(ValueError): calculate_std_dev([1.0])

st_price_like = st.lists(st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
st_return_like = st.lists(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
st_window = st.integers(min_value=1, max_value=50)

@given(series=st_price_like, span=st_window)
@settings(max_examples=50, deadline=None)
def test_ema_property(series: List[float], span: int) -> None:
    assume(span > 0)
    result = ema(series, span)
    expected = pd.Series(series).ewm(span=span, adjust=False).mean().iloc[-1]
    if math.isnan(expected): assert math.isnan(result)
    else: assert math.isclose(result, expected, rel_tol=1e-12)

def test_ema_known() -> None:
    assert math.isclose(ema([2.0, 4.0, 6.0, 8.0], 3), 6.25)

@given(series=st_return_like, window=st_window)
@settings(max_examples=50, deadline=None)
def test_rolling_vol_property(series: List[float], window: int) -> None:
    assume(window > 0)
    result = rolling_volatility(series, window)
    if len(series) < window or window == 1: assert math.isnan(result)
    else:
        expected = (pd.Series(series).rolling(window=window).std(ddof=1).iloc[-1] * math.sqrt(252))
        if math.isnan(expected): assert math.isclose(result, 0.0)
        else: assert math.isclose(result, expected, rel_tol=1e-7)

def test_rolling_vol_edge() -> None:
    assert math.isnan(rolling_volatility([0.01, 0.02], 5))

def test_latent_risk_scenarios() -> None: # For old latent_risk
    assert latent_risk([100.0] * (LR_V2_MIN_POINTS -1)) == 1.0
    assert math.isclose(latent_risk([100.0] * 40), 0.0)

def test_trade_proposal_bounds() -> None:
    adapter = TypeAdapter(TradeProposal)
    valid_new = {"action": "ENTER", "rationale": "stub", "confidence": 0.9, "legs": [{"instrument": "MES", "direction": "LONG", "size": 1.0}], "latent_risk": 0.25}
    tp = adapter.validate_python(valid_new)
    assert tp.latent_risk == 0.25
    for field in ("action", "rationale", "confidence"):
        with pytest.raises(ValidationError, match=field): adapter.validate_python({**valid_new, field: None})
    with pytest.raises(ValidationError, match="confidence"): adapter.validate_python({**valid_new, "confidence": -0.1})
    with pytest.raises(ValidationError, match="legs.0.size"): adapter.validate_python({**valid_new, "legs": [{"instrument": "MES", "direction": "LONG"}]})

def test_atr_known_values() -> None:
    hlc_short = [(10.0,8.0,9.0),(11.0,9.0,10.0),(12.0,10.0,11.0),(13.0,11.0,12.0),(14.0,12.0,13.0)]
    assert math.isclose(atr(hlc_short, window=3), 2.0)

def test_atr_edge_cases() -> None:
    with pytest.raises(ValueError): atr(_generate_hlc_data(5), window=0)
    assert math.isnan(atr(_generate_hlc_data(13), window=14))

@given(hlc_data=st_hlc_data, window=st_window)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=50)
def test_atr_property(hlc_data: List[tuple[float,float,float]], window: int) -> None:
    assume(window > 0 and len(hlc_data) >= window + 1 and len(hlc_data) >= 2)
    num_true_ranges = len(hlc_data) - 1
    assume(num_true_ranges >= window)
    result = atr(hlc_data, window)
    assert isinstance(result, float)
    if num_true_ranges >= window: assert not math.isnan(result) and result >= 0
    else: assert math.isnan(result)

def test_kelly_fraction_basic() -> None:
    assert math.isclose(kelly_fraction(mu=0.10, sigma=0.20), 2.5)

def test_kelly_fraction_edge_cases() -> None:
    assert math.isclose(kelly_fraction(mu=0.10, sigma=0.0), 0.0)

@given(mu=st.floats(min_value=-1.0,max_value=1.0,allow_nan=False,allow_infinity=False), sigma=st.one_of(st.floats(min_value=1e-6,max_value=2.0,allow_nan=False,allow_infinity=False), st.floats(min_value=-0.5,max_value=0.0,allow_nan=False,allow_infinity=False)))
@settings(max_examples=100, deadline=None) # Removed type: ignore here
def test_kelly_fraction_property(mu: float, sigma: float) -> None:
    result = kelly_fraction(mu, sigma)
    assert 0.0 <= result or math.isnan(result)
    if sigma > 1e-7 and mu > 0 :
        expected = mu / (sigma**2)
        assert math.isclose(result, expected)
    elif sigma <=0 or mu <=0:
        assert math.isclose(result, 0.0)

# Latent Risk v2 Tests
st_equity_curve_lr_v2_valid = st.lists(st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False), min_size=LR_V2_MIN_POINTS, max_size=LR_V2_MIN_POINTS + 30)
st_equity_curve_lr_v2_short = st.lists(st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=LR_V2_MIN_POINTS - 1)

def test_latent_risk_v2_edge_cases() -> None:
    assert latent_risk_v2([100.0] * (LR_V2_MIN_POINTS - 1)) == 1.0
    assert latent_risk_v2([]) == 1.0
    assert math.isclose(latent_risk_v2([100.0] * (LR_V2_MIN_POINTS + 5)), 0.0)
    with patch('azr_planner.math_utils.rolling_volatility', return_value=float('nan')):
        assert latent_risk_v2([100.0 + (i*0.01) for i in range(LR_V2_MIN_POINTS + 5)]) == 1.0
    with patch('azr_planner.math_utils._calculate_shannon_entropy', return_value=float('nan')):
        assert latent_risk_v2([100.0 + (i*0.01) for i in range(LR_V2_MIN_POINTS + 5)]) == 1.0

@given(equity_curve=st.one_of(st_equity_curve_lr_v2_valid, st_equity_curve_lr_v2_short))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_latent_risk_v2_output_range(equity_curve: List[float]) -> None:
    risk = latent_risk_v2(equity_curve)
    assert 0.0 <= risk <= 1.0
    if len(equity_curve) < LR_V2_MIN_POINTS: assert risk == 1.0

# Bayesian Confidence Tests
def test_bayesian_confidence_basic_cases() -> None:
    assert math.isclose(bayesian_confidence(wins=0, losses=0), 3.0 / 7.0)
    assert math.isclose(bayesian_confidence(wins=10, losses=0), 13.0 / 17.0)

def test_bayesian_confidence_custom_prior() -> None:
    assert math.isclose(bayesian_confidence(wins=0, losses=0, alpha=1.0, beta=1.0), 0.5)

def test_bayesian_confidence_edge_cases_inputs() -> None:
    with pytest.raises(ValueError): bayesian_confidence(wins=-1, losses=0)
    with pytest.raises(ValueError): bayesian_confidence(wins=0, losses=0, alpha=0, beta=1)

@given(wins=st.integers(min_value=0,max_value=1000), losses=st.integers(min_value=0,max_value=1000), alpha=st.floats(min_value=0.1,max_value=10.0), beta=st.floats(min_value=0.1,max_value=10.0))
@settings(max_examples=100, deadline=None) # Removed type: ignore here
def test_bayesian_confidence_property(wins: int, losses: int, alpha: float, beta: float) -> None:
    confidence = bayesian_confidence(wins, losses, alpha, beta)
    assert 0.0 <= confidence <= 1.0
    prior_mean = alpha / (alpha + beta)
    if wins == 0 and losses == 0: assert math.isclose(confidence, prior_mean)

# Docstring check loop
# Ensure all functions imported for testing are also checked for docstrings if they are part of the public API.
# LR_V2_MIN_POINTS is a constant, not a function, so it's excluded.
# The old `latent_risk` is also checked if it's still imported.
functions_to_check_for_docstrings = [
    calculate_mean, calculate_std_dev, ema, rolling_volatility,
    latent_risk, atr, kelly_fraction, bayesian_confidence, latent_risk_v2
]
for fn_obj in functions_to_check_for_docstrings:
    assert fn_obj.__doc__, f"{fn_obj.__name__} is missing a docstring"
