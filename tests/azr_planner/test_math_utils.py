"""Tests for azr_planner.math_utils."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, strategies as st
from pydantic import ValidationError
from pydantic import TypeAdapter

from azr_planner.math_utils import (
    calculate_mean,
    calculate_std_dev,
    ema,
    rolling_volatility,
    latent_risk,
)
from azr_planner.schemas import TradeProposal

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
    valid = {
        "action": "ENTER",
        "rationale": "stub",
        "latent_risk": 0.25,
        "confidence": 0.9,
        "legs": [{"instrument": "MES", "direction": "LONG", "size": 1}],
    }
    tp = adapter.validate_python(valid)
    assert tp.latent_risk == 0.25

    # missing required
    for field in ("action", "rationale", "latent_risk", "confidence"):
        bad = {**valid}
        bad.pop(field)
        with pytest.raises(ValidationError):
            adapter.validate_python(bad)

    # bounds
    for k in ("latent_risk", "confidence"):
        low = {**valid, k: -0.1}
        hi = {**valid, k: 1.1}
        with pytest.raises(ValidationError):
            adapter.validate_python(low)
        with pytest.raises(ValidationError):
            adapter.validate_python(hi)

    # legs – missing size
    bad_legs = {**valid, "legs": [{"instrument": "MES", "direction": "LONG"}]}
    with pytest.raises(ValidationError, match=r"legs\.0\.size.*Field required"):
        adapter.validate_python(bad_legs)


# -----------------------------------------------------------------------------#
#  Keep docstrings
# -----------------------------------------------------------------------------#
for fn in (
    calculate_mean,
    calculate_std_dev,
    ema,
    rolling_volatility,
    latent_risk,
):
    assert fn.__doc__, f"{fn.__name__} is missing a docstring"
