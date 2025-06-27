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
    latent_risk,
    atr,
    kelly_fraction,
    bayesian_confidence,
    latent_risk_v2, # Added latent_risk_v2 to imports
)
from azr_planner.schemas import TradeProposal


# Helper to generate HLC data for ATR tests
def _generate_hlc_data(num_periods: int, start_price: float = 100.0, daily_change: float = 1.0, spread: float = 0.5) -> List[tuple[float, float, float]]:
    data = []
    current_close = start_price
    for i in range(num_periods):
        high = current_close + spread + (daily_change if i % 2 == 0 else 0) # Add some variation
        low = current_close - spread - (daily_change if i % 2 != 0 else 0)
        close = (high + low) / 2
        # Ensure H >= L and C is within H-L
        if low > high: low = high - 0.01
        if close > high: close = high
        if close < low: close = low
        data.append((high, low, close))
        current_close = close
    return data

# More direct HLC tuple generation to avoid excessive filtering for st_hlc_data
@st.composite
def st_single_hlc_tuple_for_math_tests(draw: DrawFn) -> tuple[float, float, float]:
    # Draw three initial floats
    f1 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f2 = draw(st.floats(min_value=1.0, max_value=1999.0))
    f3 = draw(st.floats(min_value=1.0, max_value=1999.0))

    s = sorted([f1, f2, f3])
    low, _, high = s[0], s[1], s[2] # Use _ for c_candidate as close is drawn separately

    if high <= low: # Ensure High > Low
        high = low + draw(st.floats(min_value=1e-6, max_value=1.0)) # Ensure high is strictly greater

    close = draw(st.floats(min_value=low, max_value=high)) # Close can be anywhere between Low and High

    return round(high, 2), round(low, 2), round(close, 2)

st_hlc_data = st.lists(
    st_single_hlc_tuple_for_math_tests(),
    min_size=2,
    max_size=100
)

# ──────────────────────────────────────────────────────────────────────────────
#  Basic helpers (mean, std-dev)
# ──────────────────────────────────────────────────────────────────────────────


def test_calculate_mean() -> None:
    assert calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
    assert calculate_mean([10.0]) == 10.0
    assert calculate_mean([-1.0, 1.0]) == 0.0
    assert calculate_mean([1.5, 2.5, 3.5]) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        calculate_mean([])


def test_calculate_std_dev() -> None:
    assert calculate_std_dev([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))
    assert calculate_std_dev([1.0, 3.0]) == pytest.approx(math.sqrt(2.0))
    assert calculate_std_dev([5.0, 5.0, 5.0, 5.0]) == 0.0
    with pytest.raises(ValueError):
        calculate_std_dev([1.0])
    with pytest.raises(ValueError):
        calculate_std_dev([])


# ──────────────────────────────────────────────────────────────────────────────
#  Hypothesis strategies
# ──────────────────────────────────────────────────────────────────────────────

st_price_like = st.lists(
    st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=100,
)
st_return_like = st.lists(
    st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=100,
)
st_window = st.integers(min_value=1, max_value=50)

# ──────────────────────────────────────────────────────────────────────────────
#  EMA
# ──────────────────────────────────────────────────────────────────────────────


@given(series=st_price_like, span=st_window)
@settings(max_examples=50, deadline=None)
def test_ema_property(series: List[float], span: int) -> None:
    assume(span > 0)
    result = ema(series, span)
    pd_series = pd.Series(series)
    expected = pd_series.ewm(span=span, adjust=False).mean().iloc[-1]
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert math.isclose(result, expected, rel_tol=1e-12)


def test_ema_known() -> None:
    assert math.isclose(ema([2.0, 4.0, 6.0, 8.0], 3), 6.25)
    assert math.isclose(ema([10.0], 5), 10.0)
    with pytest.raises(ValueError):
        ema([], 3)
    with pytest.raises(ValueError):
        ema([1.0, 2.0], 0)


# ──────────────────────────────────────────────────────────────────────────────
#  Rolling volatility
# ──────────────────────────────────────────────────────────────────────────────


@given(series=st_return_like, window=st_window)
@settings(max_examples=50, deadline=None)
def test_rolling_vol_property(series: List[float], window: int) -> None:
    assume(window > 0)
    result = rolling_volatility(series, window)
    if len(series) < window or window == 1:
        assert math.isnan(result)
    else:
        expected = (
            pd.Series(series).rolling(window=window).std(ddof=1).iloc[-1] * math.sqrt(252)
        )
        if math.isnan(expected):
            assert math.isclose(result, 0.0)
        else:
            assert math.isclose(result, expected, rel_tol=1e-7)


def test_rolling_vol_edge() -> None:
    assert math.isnan(rolling_volatility([0.01, 0.02], 5))
    assert math.isnan(rolling_volatility([0.01], 1))
    with pytest.raises(ValueError):
        rolling_volatility([], 2)
    with pytest.raises(ValueError):
        rolling_volatility([0.01], 0)


# ──────────────────────────────────────────────────────────────────────────────
#  Latent-risk — unit scenarios
# ──────────────────────────────────────────────────────────────────────────────


def test_latent_risk_scenarios() -> None:
    # 1) Very short / empty → clamp to 1
    assert latent_risk([100.0, 100.1, 100.2]) == 1.0
    assert latent_risk([]) == 1.0

    # 2) Flat 40-day curve → 0
    assert math.isclose(latent_risk([100.0] * 40), 0.0)

    # 3) Gentle linear rise still produces high σ_a → risk≈1
    rising = [100.0 + 0.1 * i for i in range(40)]
    assert math.isclose(latent_risk(rising), 1.0, abs_tol=1e-6)

    # 4) High-volatility mix → risk ≥ 0.5
    volatile = (
        [100.0, 120.0, 80.0, 110.0, 90.0, 130.0, 70.0, 100.0, 140.0, 60.0]
        * 4
    )
    assert latent_risk(volatile) >= 0.5

    # 5) Large sustained draw-down → 1
    drawdown = [200.0] * 15 + [100.0] * 25
    assert math.isclose(latent_risk(drawdown), 1.0)

    # 6) Random-walk entropy smoke test
    np.random.seed(42)
    walk = [100.0]
    for _ in range(59):
        walk.append(walk[-1] * (1 + np.random.uniform(-0.03, 0.03)))
    rw_risk = latent_risk(walk)
    assert 0.0 <= rw_risk <= 1.0

    # 7) Exactly 30 points (minimum for 30-day vol) → still 1.0
    min_len = [100.0 + 0.01 * i for i in range(30)]
    assert math.isclose(latent_risk(min_len), 1.0, abs_tol=1e-6)

    # 8) 29 points (missing 30-day window) → clamp to 1
    almost = [100.0 + 0.01 * i for i in range(29)]
    assert math.isclose(latent_risk(almost), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  TradeProposal schema sanity
# ──────────────────────────────────────────────────────────────────────────────


def test_trade_proposal_bounds() -> None:
    adapter = TypeAdapter(TradeProposal)

    # happy path
    valid_new = {
        "action": "ENTER",
        "rationale": "stub",
        "confidence": 0.9, # latent_risk is now optional
        "legs": [{"instrument": "MES", "direction": "LONG", "size": 1.0}], # size must be float
        "latent_risk": 0.25, # Can still be provided
        "signal_value": 0.5,
        "atr_value": 1.2,
        "kelly_fraction_value": 0.1,
        "target_position_size": 1.0
    }
    tp = adapter.validate_python(valid_new)
    assert tp.latent_risk == 0.25 # Check if it was parsed

    # missing required (action, rationale, confidence are required)
    for field in ("action", "rationale", "confidence"):
        bad = {**valid_new}
        bad.pop(field)
        with pytest.raises(ValidationError, match=field): # Check that the error message mentions the field
            adapter.validate_python(bad)

    # bounds for confidence (latent_risk is optional, its bounds are checked by Pydantic if provided)
    # Check confidence bounds
    low_conf = {**valid_new, "confidence": -0.1}
    hi_conf = {**valid_new, "confidence": 1.1}
    with pytest.raises(ValidationError, match="confidence"):
        adapter.validate_python(low_conf)
    with pytest.raises(ValidationError, match="confidence"):
        adapter.validate_python(hi_conf)

    # Check latent_risk bounds if provided
    low_lr = {**valid_new, "latent_risk": -0.1}
    hi_lr = {**valid_new, "latent_risk": 1.1}
    with pytest.raises(ValidationError, match="latent_risk"): # Pydantic should complain about latent_risk
        adapter.validate_python(low_lr)
    with pytest.raises(ValidationError, match="latent_risk"):
        adapter.validate_python(hi_lr)

    # Check other new optional fields' bounds if provided (e.g. atr_value >= 0)
    bad_atr = {**valid_new, "atr_value": -0.1}
    with pytest.raises(ValidationError, match="atr_value"):
         adapter.validate_python(bad_atr)

    bad_kelly = {**valid_new, "kelly_fraction_value": -0.1}
    with pytest.raises(ValidationError, match="kelly_fraction_value"):
        adapter.validate_python(bad_kelly)

    bad_target_size = {**valid_new, "target_position_size": -1.0}
    with pytest.raises(ValidationError, match="target_position_size"):
        adapter.validate_python(bad_target_size)

    # legs – missing size
    bad_legs = {**valid_new, "legs": [{"instrument": "MES", "direction": "LONG"}]} # Use valid_new
    with pytest.raises(ValidationError, match=r"legs\.0\.size"):
        adapter.validate_python(bad_legs)


# ──────────────────────────────────────────────────────────────────────────────
#  ATR (Average True Range)
# ──────────────────────────────────────────────────────────────────────────────

def test_atr_known_values() -> None:
    """Test ATR with a known sequence from online examples (e.g., StockCharts)."""
    # Data from https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr
    # Day H   L   C   TR  ATR_14
    # 1   23.32 22.78 22.81 -   -
    # 2   22.94 22.50 22.63 0.44
    # ...
    # For simplicity, using pandas_ta to verify a short sequence
    # Note: pandas_ta might use slightly different smoothing for the first few values.
    # Our implementation uses ewm(alpha=1/N, adjust=False) which is Wilder's smoothing.
    hlc = [ # H, L, C
        (23.32, 22.78, 22.81), (22.94, 22.50, 22.63), (23.13, 22.63, 23.06),
        (23.00, 22.38, 22.50), (22.81, 22.25, 22.75), (23.06, 22.69, 22.88),
        (23.38, 22.88, 23.25), (23.50, 23.00, 23.13), (23.38, 22.81, 22.88),
        (23.25, 22.94, 23.13), (23.56, 23.19, 23.44), (23.50, 23.00, 23.06),
        (23.19, 22.81, 22.88), (23.13, 22.63, 22.75), (23.00, 22.50, 22.94) # 15 days for 14 TRs + 1 ATR
    ]
    # Expected TRs:
    # TR1 = 0.44 (H[1]-L[1]=0.44, H[1]-C[0]=0.13, L[1]-C[0]=-0.31 -> abs=0.31. max=0.44)
    # TR2 = 0.50 (H[2]-L[2]=0.50, H[2]-C[1]=0.50, L[2]-C[1]=0.00. max=0.50)
    # ...
    # A full 14-period ATR calculation is complex to do by hand here.
    # Let's use a shorter, verifiable example.
    # Data for 5 days, window = 3
    hlc_short = [
        (10.0, 8.0, 9.0),    # Day 1
        (11.0, 9.0, 10.0),   # Day 2. Prev C=9. TR = max(11-9, abs(11-9), abs(9-9)) = max(2,2,0) = 2
        (12.0, 10.0, 11.0),  # Day 3. Prev C=10. TR = max(12-10, abs(12-10), abs(10-10)) = max(2,2,0) = 2
        (13.0, 11.0, 12.0),  # Day 4. Prev C=11. TR = max(13-11, abs(13-11), abs(11-11)) = max(2,2,0) = 2
        (14.0, 12.0, 13.0),  # Day 5. Prev C=12. TR = max(14-12, abs(14-12), abs(12-12)) = max(2,2,0) = 2
    ]
    # TRs = [2.0, 2.0, 2.0, 2.0]
    # ATR(3) on these TRs:
    # First ATR(3) = (2+2+2)/3 = 2. (This is the SMA of first 3 TRs)
    # Our current EWM method:
    # TRs: [2, 2, 2, 2]. Window=3. Alpha=1/3.
    # Val1: 2
    # Val2: (2 * (1-1/3)) + (2 * 1/3) = 2
    # Val3: (2 * (1-1/3)) + (2 * 1/3) = 2 (this would be the first output if min_periods=3)
    # Val4: (2 * (1-1/3)) + (2 * 1/3) = 2
    # So, for constant TRs, ATR should be that constant.
    assert math.isclose(atr(hlc_short, window=3), 2.0)

    # Example with varying TRs
    hlc_var = [
        (10.0, 8.0, 9.0),    # Day 1
        (11.0, 9.0, 10.0),   # Day 2. Prev C=9. TR = 2.0
        (12.5, 10.0, 12.0), # Day 3. Prev C=10. TR = max(2.5, abs(2.5), abs(0)) = 2.5
        (13.0, 10.5, 11.0), # Day 4. Prev C=12. TR = max(2.5, abs(1), abs(1.5)) = 2.5
        (15.0, 12.0, 14.5)  # Day 5. Prev C=11. TR = max(3, abs(4), abs(1)) = 4.0
    ]
    # TRs = [2.0, 2.5, 2.5, 4.0]. Window = 3. Alpha = 1/3.
    # ewm(adjust=False):
    # atr1 = 2
    # atr2 = 2 * (2/3) + 2.5 * (1/3) = 4/3 + 2.5/3 = 6.5/3 = 2.1666...
    # atr3 = (6.5/3)*(2/3) + 2.5*(1/3) = 13/9 + 2.5/3 = 13/9 + 7.5/9 = 20.5/9 = 2.2777... (this is output if min_periods=3)
    # atr4 = (20.5/9)*(2/3) + 4*(1/3) = 41/27 + 4/3 = 41/27 + 36/27 = 77/27 = 2.85185...
    # Our function uses min_periods=window for ewm. So for window=3, it needs 3 TR values.
    # TRs = [2, 2.5, 2.5, 4]. window=3.
    # 1st value is 2.5 (avg of first 3 TRs [2, 2.5, 2.5] / 3 = 7/3 = 2.333 - if using SMA for first)
    # Pandas ewm(alpha=1/3, adjust=False, min_periods=3) applied to [2, 2.5, 2.5, 4]:
    # s = pd.Series([2, 2.5, 2.5, 4])
    # s.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    # 0    NaN
    # 1    NaN
    # 2    2.277778  <-- (2 * (2/3)^2 + 2.5 * (2/3)*(1/3) + 2.5 * (1/3)) -- this is not how ewm(adjust=False) works iteratively.
    # Iterative ewm(adjust=False) with seed:
    # y_0 = x_0
    # y_t = (1-alpha)*y_{t-1} + alpha*x_t
    # x = [2, 2.5, 2.5, 4]
    # y0 = 2
    # y1 = (2/3)*2 + (1/3)*2.5 = 4/3 + 2.5/3 = 6.5/3 = 2.1666...
    # y2 = (2/3)*(6.5/3) + (1/3)*2.5 = 13/9 + 2.5/3 = 20.5/9 = 2.2777... (This is the value after 3 TRs)
    # y3 = (2/3)*(20.5/9) + (1/3)*4 = 41/27 + 4/3 = 77/27 = 2.85185... (This is the value after 4 TRs)
    # The function returns the last value, so 2.85185...
    assert math.isclose(atr(hlc_var, window=3), 2.8518518518518517)


def test_atr_edge_cases() -> None:
    with pytest.raises(ValueError, match="Window must be positive"):
        atr(_generate_hlc_data(5), window=0)
    with pytest.raises(ValueError, match="Window must be positive"):
        atr(_generate_hlc_data(5), window=-1)
    with pytest.raises(ValueError, match="Input data list cannot be empty"):
        atr([], window=14)

    # Not enough data for window
    assert math.isnan(atr(_generate_hlc_data(13), window=14)) # Needs 14 TRs, so 15 HLC days. Here 13 HLC -> 12 TRs.
    assert math.isnan(atr(_generate_hlc_data(14), window=14)) # 14 HLC -> 13 TRs.
    assert not math.isnan(atr(_generate_hlc_data(15), window=14)) # 15 HLC -> 14 TRs. Should produce a value.

    # Minimal data
    assert math.isnan(atr([(1.0,1.0,1.0)], window=1)) # Needs 2 HLC for 1 TR.
    minimal_hlc = [(10.0,8.0,9.0), (11.0,9.0,10.0)] # 1 TR value = 2.0
    assert math.isclose(atr(minimal_hlc, window=1), 2.0) # ATR(1) of [2.0] is 2.0

    # Data with no price movement (TR=0)
    flat_hlc = [(10.0,10.0,10.0)] * 20
    assert math.isclose(atr(flat_hlc, window=14), 0.0)

    # Data with gaps (large TRs)
    gappy_hlc = [
        (10.0, 8.0, 9.0), (20.0, 18.0, 19.0) # Prev C=9. High=20, Low=18. H-L=2. H-PC=11. L-PC=9. TR = 11.0
    ] * 15 # Repeat this pattern, TRs will be [11.0, 11.0, ...]
    assert math.isclose(atr(gappy_hlc, window=3), 11.0)


from hypothesis import HealthCheck # For suppressing health checks

@given(hlc_data=st_hlc_data, window=st_window)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=50)
def test_atr_property(hlc_data: List[tuple[float,float,float]], window: int) -> None:
    assume(window > 0)
    assume(len(hlc_data) >= window +1) # Ensure enough data for at least one ATR calculation
                                      # (window TRs needed, so window+1 HLC points)
                                      # Our current code returns NaN if len(hlc_data) < window + 1 if window > 1
                                      # For window=1, len(hlc_data) >= 2
    assume(len(hlc_data) >= 2) # General minimum for any TR calculation

    # Correct assumption for enough data for our ATR implementation:
    # We need `window` True Range values.
    # To get `k` TR values, we need `k+1` HLC data points.
    # So, for `window` TRs, we need `window+1` HLC data points.
    # The ewm in pandas with min_periods=window will output NaN if fewer than `window` TRs are fed.
    num_true_ranges = len(hlc_data) - 1
    assume(num_true_ranges >= window)


    result = atr(hlc_data, window)

    assert isinstance(result, float)
    if num_true_ranges >= window:
        assert not math.isnan(result), "ATR should not be NaN with sufficient data"
        assert result >= 0, "ATR should be non-negative"
    else:
        assert math.isnan(result), "ATR should be NaN with insufficient data"


# ──────────────────────────────────────────────────────────────────────────────
#  Kelly Fraction
# ──────────────────────────────────────────────────────────────────────────────

def test_kelly_fraction_basic() -> None:
    assert math.isclose(kelly_fraction(mu=0.10, sigma=0.20), 0.10 / (0.20**2)) # 0.10 / 0.04 = 2.5
    assert math.isclose(kelly_fraction(mu=0.05, sigma=0.10), 0.05 / (0.10**2)) # 0.05 / 0.01 = 5.0
    assert math.isclose(kelly_fraction(mu=0.20, sigma=0.50), 0.20 / (0.50**2)) # 0.20 / 0.25 = 0.8

    # mu = 0 -> fraction = 0
    assert math.isclose(kelly_fraction(mu=0.0, sigma=0.20), 0.0)

    # mu < 0 -> fraction = 0 (as per our implementation choice)
    assert math.isclose(kelly_fraction(mu=-0.05, sigma=0.20), 0.0)


def test_kelly_fraction_edge_cases() -> None:
    # sigma = 0 -> fraction = 0
    assert math.isclose(kelly_fraction(mu=0.10, sigma=0.0), 0.0)
    assert math.isclose(kelly_fraction(mu=0.10, sigma=-0.1), 0.0) # sigma <= 0

    # Large mu, small sigma (large fraction)
    assert math.isclose(kelly_fraction(mu=0.5, sigma=0.01), 0.5 / (0.01**2)) # 0.5 / 0.0001 = 5000.0


@given(
    mu=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    # Adjust sigma to avoid extremely small values that square to zero, while still testing positive values.
    # Also test behavior for sigma <= 0 explicitly.
    sigma=st.one_of(
        st.floats(min_value=1e-6, max_value=2.0, allow_nan=False, allow_infinity=False), # Smallest positive sigma
        st.floats(min_value=-0.5, max_value=0.0, allow_nan=False, allow_infinity=False) # Non-positive sigma
    )
)
@settings(max_examples=100, deadline=None) # Increased examples for one_of
def test_kelly_fraction_property(mu: float, sigma: float) -> None:
    result = kelly_fraction(mu, sigma)

    assert isinstance(result, float)
    assert not math.isnan(result)
    assert not math.isinf(result)
    assert result >= 0.0 # Kelly fraction should be non-negative as per implementation

    if sigma <= 0:
        assert math.isclose(result, 0.0)
    elif mu <= 0: # mu > 0 is already covered by sigma > 0 and result >= 0
        assert math.isclose(result, 0.0)
    else: # mu > 0 and sigma > 0
        expected = mu / (sigma**2)
        assert math.isclose(result, expected)


# -----------------------------------------------------------------------------#
#  Keep docstrings
# -----------------------------------------------------------------------------#
for fn in (
    calculate_mean,
    calculate_std_dev,
    ema,
    rolling_volatility,
    latent_risk,
    atr,
    kelly_fraction,
    bayesian_confidence,
    latent_risk_v2,
):
    assert fn.__doc__, f"{fn.__name__} is missing a docstring"


# ──────────────────────────────────────────────────────────────────────────────
# Latent Risk v2
# ──────────────────────────────────────────────────────────────────────────────
from azr_planner.math_utils import latent_risk_v2, LR_V2_MIN_POINTS
from unittest.mock import patch

# Strategy for equity curve for latent_risk_v2 tests
st_equity_curve_lr_v2_valid = st.lists(
    st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False), # More stable prices
    min_size=LR_V2_MIN_POINTS, # Ensure enough points for default calculation
    max_size=LR_V2_MIN_POINTS + 30
)
st_equity_curve_lr_v2_short = st.lists(
    st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=LR_V2_MIN_POINTS - 1
)


def test_latent_risk_v2_edge_cases() -> None:
    # Test with insufficient data
    assert latent_risk_v2([100.0] * (LR_V2_MIN_POINTS - 1)) == 1.0
    assert latent_risk_v2([]) == 1.0
    assert latent_risk_v2([100.0, 101.0]) == 1.0

    # Test with flat curve (expect low risk, exact value depends on component contributions)
    # Volatility = 0, Drawdown = 0, Entropy of zero returns = 0
    # So raw_risk = 0.45*(0/SIGMA_TGT) + 0.35*(0/DD_TGT) + 0.2*(0/H_TGT) = 0
    assert math.isclose(latent_risk_v2([100.0] * (LR_V2_MIN_POINTS + 5)), 0.0)

    # Test with NaN in components - should default to 1.0
    # Mock internal rolling_volatility to return NaN to test this path
    with patch('azr_planner.math_utils.rolling_volatility', return_value=float('nan')):
        equity_data = [100.0 + (i*0.01) for i in range(LR_V2_MIN_POINTS + 5)] # Gently rising to have non-zero returns
        assert latent_risk_v2(equity_data) == 1.0

    # Mock _calculate_shannon_entropy to return NaN
    with patch('azr_planner.math_utils._calculate_shannon_entropy', return_value=float('nan')):
        equity_data = [100.0 + (i*0.01) for i in range(LR_V2_MIN_POINTS + 5)]
        assert latent_risk_v2(equity_data) == 1.0

    # Test with a curve that would produce a large drawdown
    dd_curve = [100.0] * LR_V2_MIN_POINTS + [50.0] * 5 # Sustained large drawdown
    # Max DD here is 0.5. dd_term = 0.5 / 0.333 approx 1.5.
    # Other terms might be small if vol/entropy of the 100s part is low.
    # Example, if vol=0, entropy=0 for first part, then risk = 0.35 * (0.5 / 0.333) which is > 0.5
    # This should not be 1.0 unless other components also max out, or the combined sum > 1 before clamp.
    # Let's check it's high, e.g. > 0.5 (0.35 * 1.5 = 0.525)
    assert latent_risk_v2(dd_curve) > 0.5

@given(equity_curve=st.one_of(st_equity_curve_lr_v2_valid, st_equity_curve_lr_v2_short))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_latent_risk_v2_output_range(equity_curve: List[float]):
    risk = latent_risk_v2(equity_curve)
    assert 0.0 <= risk <= 1.0, f"Risk {risk} out of bounds [0,1] for curve: {equity_curve}"
    if len(equity_curve) < LR_V2_MIN_POINTS:
        assert risk == 1.0, "Risk should be 1.0 for insufficient data"


# ──────────────────────────────────────────────────────────────────────────────
#  Bayesian Confidence
# ──────────────────────────────────────────────────────────────────────────────

def test_bayesian_confidence_basic_cases() -> None: # Already has -> None, error was for a different one.
    # Default prior: alpha=3, beta=4. Prior mean = 3 / (3+4) = 3/7 approx 0.42857
    # No observations
    assert math.isclose(bayesian_confidence(wins=0, losses=0), 3.0 / 7.0)

    # Only wins
    # (10 wins + 3 alpha) / (10 wins + 0 losses + 3 alpha + 4 beta) = 13 / (10 + 7) = 13/17
    assert math.isclose(bayesian_confidence(wins=10, losses=0), 13.0 / 17.0)

    # Only losses
    # (0 wins + 3 alpha) / (0 wins + 10 losses + 3 alpha + 4 beta) = 3 / (10 + 7) = 3/17
    assert math.isclose(bayesian_confidence(wins=0, losses=10), 3.0 / 17.0)

    # Equal wins and losses
    # (10 wins + 3 alpha) / (10 wins + 10 losses + 3 alpha + 4 beta) = 13 / (20 + 7) = 13/27
    assert math.isclose(bayesian_confidence(wins=10, losses=10), 13.0 / 27.0)

    # Check confidence approaches 1 with many wins, few losses
    # (1000 + 3) / (1000 + 0 + 3 + 4) = 1003 / 1007
    assert math.isclose(bayesian_confidence(wins=1000, losses=0), 1003.0 / 1007.0)
    assert bayesian_confidence(wins=1000, losses=0) > 0.99

    # Check confidence approaches 0 with many losses, few wins
    # (0 + 3) / (0 + 1000 + 3 + 4) = 3 / 1007
    assert math.isclose(bayesian_confidence(wins=0, losses=1000), 3.0 / 1007.0)
    assert bayesian_confidence(wins=0, losses=1000) < 0.01


def test_bayesian_confidence_custom_prior() -> None: # Added -> None
    # Custom prior: alpha=1, beta=1 (Laplace smoothing / uniform prior)
    # Prior mean = 1 / (1+1) = 0.5
    assert math.isclose(bayesian_confidence(wins=0, losses=0, alpha=1.0, beta=1.0), 0.5)

    # (10 wins + 1 alpha) / (10 wins + 0 losses + 1 alpha + 1 beta) = 11 / 12
    assert math.isclose(bayesian_confidence(wins=10, losses=0, alpha=1.0, beta=1.0), 11.0 / 12.0)


def test_bayesian_confidence_edge_cases_inputs() -> None: # Added -> None
    # Negative wins/losses should raise ValueError
    with pytest.raises(ValueError, match="Number of wins cannot be negative."):
        bayesian_confidence(wins=-1, losses=0)
    with pytest.raises(ValueError, match="Number of losses cannot be negative."):
        bayesian_confidence(wins=0, losses=-1)

    # Non-positive alpha/beta should raise ValueError
    with pytest.raises(ValueError, match="Alpha parameter must be positive."):
        bayesian_confidence(wins=0, losses=0, alpha=0, beta=1)
    with pytest.raises(ValueError, match="Beta parameter must be positive."):
        bayesian_confidence(wins=0, losses=0, alpha=1, beta=0)
    with pytest.raises(ValueError, match="Alpha parameter must be positive."):
        bayesian_confidence(wins=0, losses=0, alpha=-1, beta=1)
    with pytest.raises(ValueError, match="Beta parameter must be positive."):
        bayesian_confidence(wins=0, losses=0, alpha=1, beta=-1)


@given(
    wins=st.integers(min_value=0, max_value=1000),
    losses=st.integers(min_value=0, max_value=1000),
    alpha=st.floats(min_value=0.1, max_value=10.0), # Ensure positive alpha
    beta=st.floats(min_value=0.1, max_value=10.0)   # Ensure positive beta
)
@settings(max_examples=100, deadline=None)
def test_bayesian_confidence_property(wins: int, losses: int, alpha: float, beta: float) -> None:
    confidence = bayesian_confidence(wins, losses, alpha, beta)
    assert 0.0 <= confidence <= 1.0

    # If only wins, confidence should be > prior mean (alpha / (alpha+beta))
    # If only losses, confidence should be < prior mean
    # If wins/losses are proportional to alpha/beta, confidence should be near prior mean
    prior_mean = alpha / (alpha + beta)
    if wins > 0 and losses == 0:
        assert confidence > prior_mean or math.isclose(confidence, prior_mean) # Can be close if alpha is large
    if losses > 0 and wins == 0:
        assert confidence < prior_mean or math.isclose(confidence, prior_mean) # Can be close if beta is large
    if wins == 0 and losses == 0:
        assert math.isclose(confidence, prior_mean)
