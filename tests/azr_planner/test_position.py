from __future__ import annotations

import math
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from azr_planner.position import position_size

# --- Unit Tests for position_size ---

@pytest.mark.parametrize(
    "latent_risk, equity, max_leverage, expected_exposure",
    [
        # Region 1: risk <= 0.3 (Full exposure)
        (0.0, 100_000.0, 2.0, 200_000.0),
        (0.15, 100_000.0, 2.0, 200_000.0),
        (0.3, 100_000.0, 2.0, 200_000.0),
        (0.2, 50_000.0, 1.0, 50_000.0), # Different equity/leverage

        # Region 2: 0.3 < risk < 0.7 (Linear taper)
        # Mid-point: risk = 0.5 (should be half of max_exposure)
        # Max exposure = 100_000 * 2 = 200_000. Half = 100_000
        (0.5, 100_000.0, 2.0, 100_000.0),
        # Closer to 0.3: risk = 0.4. Exposure = ME * (1 - (0.4-0.3)/0.4) = ME * (1 - 0.1/0.4) = ME * 0.75
        # ME * (1 - (risk - 0.3) / (0.7 - 0.3)) = ME * (1 - (risk - 0.3) / 0.4)
        # For risk = 0.4: 200_000 * (1 - (0.1/0.4)) = 200_000 * 0.75 = 150_000
        (0.4, 100_000.0, 2.0, 150_000.0),
        # Closer to 0.7: risk = 0.6. Exposure = ME * (1 - (0.6-0.3)/0.4) = ME * (1 - 0.3/0.4) = ME * 0.25
        (0.6, 100_000.0, 2.0, 50_000.0),
        # Test with risk slightly above 0.3
        (0.30001, 100_000.0, 2.0, 200_000.0 * (1 - (0.00001 / 0.4))), # Almost full
        # Test with risk slightly below 0.7
        (0.69999, 100_000.0, 2.0, 200_000.0 * (1 - (0.39999 / 0.4))), # Almost zero

        # Region 3: risk >= 0.7 (Zero exposure)
        (0.7, 100_000.0, 2.0, 0.0),
        (0.85, 100_000.0, 2.0, 0.0),
        (1.0, 100_000.0, 2.0, 0.0),

        # Edge cases for risk clamping
        (-0.1, 100_000.0, 2.0, 200_000.0), # risk < 0 treated as 0
        (1.1, 100_000.0, 2.0, 0.0),      # risk > 1 treated as 1 (which is >= 0.7)

        # Edge cases for equity and leverage
        (0.2, 0.0, 2.0, 0.0),          # Zero equity
        (0.2, -1000.0, 2.0, 0.0),      # Negative equity
        (0.2, 100_000.0, 0.0, 0.0),    # Zero leverage
        (0.2, 100_000.0, -1.0, 0.0),   # Negative leverage
        (0.2, 100_000.0, 1.0, 100_000.0),# Max leverage = 1
    ]
)
def test_position_size_unit_cases(
    latent_risk: float, equity: float, max_leverage: float, expected_exposure: float
) -> None:
    """Unit tests for position_size covering specific scenarios."""
    calculated_exposure = position_size(latent_risk, equity, max_leverage)
    assert math.isclose(calculated_exposure, expected_exposure, abs_tol=1e-9), \
        f"Risk: {latent_risk}, Equity: {equity}, Leverage: {max_leverage} -> Expected: {expected_exposure}, Got: {calculated_exposure}"

# --- Property Test for position_size ---

@given(
    latent_risk=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False), # Test clamping
    equity=st.floats(min_value=-100_000, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    max_leverage=st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_position_size_properties(latent_risk: float, equity: float, max_leverage: float) -> None:
    """
    Property tests for position_size:
    1. Output exposure is within [0, equity * max_leverage].
    2. Output exposure is monotonically non-increasing with latent_risk.
    """
    calculated_exposure = position_size(latent_risk, equity, max_leverage)

    # Property 1: Output exposure bounds
    if equity > 0 and max_leverage > 0:
        max_possible_exposure = equity * max_leverage
        assert 0.0 <= calculated_exposure <= max_possible_exposure + 1e-9 # Add tolerance for float math
        # Ensure it's exactly 0 if it should be (e.g. risk >= 0.7)
        clamped_risk_for_check = min(max(latent_risk, 0.0), 1.0)
        if clamped_risk_for_check >= 0.7:
            assert math.isclose(calculated_exposure, 0.0, abs_tol=1e-9)
    else: # If equity or leverage is non-positive, exposure must be 0
        assert math.isclose(calculated_exposure, 0.0, abs_tol=1e-9)

    # Property 2: Monotonically non-increasing with risk
    # Test by checking a slightly higher risk results in lower or equal exposure
    # (only if current exposure is not already zero, and equity/leverage are positive)
    if equity > 0 and max_leverage > 0 and calculated_exposure > 1e-9: # Avoid testing if already at zero
        delta = 0.01
        # Ensure higher_risk is still within a reasonable range for comparison, esp. around 0.7
        higher_risk = min(latent_risk + delta, 1.0)
        if higher_risk > latent_risk: # Only if higher_risk is actually higher
             exposure_at_higher_risk = position_size(higher_risk, equity, max_leverage)
             assert exposure_at_higher_risk <= calculated_exposure + 1e-9 # Add tolerance


@given(
    equity=st.floats(min_value=1, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    max_leverage=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(deadline=None)
def test_position_size_monotonicity_strict(equity: float, max_leverage: float) -> None:
    """
    Stricter test for monotonicity across the relevant risk range [0,1].
    """
    import random # Import Python's random module
    risks = sorted([0.0, 0.1, 0.29, 0.3, 0.31, 0.4, 0.5, 0.6, 0.69, 0.7, 0.71, 0.9, 1.0] + \
                   [random.uniform(0,1) for _ in range(10)]) # Use random.uniform

    exposures = [position_size(r, equity, max_leverage) for r in sorted(list(set(risks)))]

    for i in range(len(exposures) - 1):
        assert exposures[i+1] <= exposures[i] + 1e-9 # Check non-increasing with tolerance
