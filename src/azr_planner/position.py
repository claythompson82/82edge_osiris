from __future__ import annotations

import math

def position_size(
    latent_risk: float,
    equity: float,
    max_leverage: float = 2.0
) -> float:
    """
    Calculates the target dollar exposure based on latent risk, equity, and max leverage.

    The sizing is piece-wise linear:
    - Risk <= 0.3: Full target exposure (+1.0 * equity * max_leverage).
    - 0.3 < Risk < 0.7: Linearly tapers exposure from full to zero.
    - Risk >= 0.7: Zero exposure.

    The calculated exposure is implicitly clamped by the definition, as max exposure
    is equity * max_leverage and min is 0. The function handles positive exposure;
    direction (long/short) is determined by the trading strategy.

    Args:
        latent_risk: The calculated latent risk, expected to be between 0 and 1.
        equity: The current equity available for trading.
        max_leverage: The maximum leverage allowed on the equity.

    Returns:
        The calculated target dollar exposure. Returns 0.0 if equity or max_leverage
        is not positive, or if risk is too high.
    """
    if equity <= 0 or max_leverage <= 0:
        return 0.0

    # Clamp latent_risk to the [0, 1] range for safety, though inputs should ideally be validated.
    clamped_risk = min(max(latent_risk, 0.0), 1.0)

    max_dollar_exposure = equity * max_leverage
    target_exposure: float

    if clamped_risk <= 0.3:
        target_exposure = max_dollar_exposure
    elif clamped_risk < 0.7: # Linearly taper from full exposure at risk=0.3 to zero at risk=0.7
        # y = y1 + m * (x - x1)
        # y1 = max_dollar_exposure at x1 = 0.3
        # y2 = 0 at x2 = 0.7
        # m = (y2 - y1) / (x2 - x1) = (0 - max_dollar_exposure) / (0.7 - 0.3)
        # m = -max_dollar_exposure / 0.4
        slope = -max_dollar_exposure / 0.4
        target_exposure = max_dollar_exposure + slope * (clamped_risk - 0.3)
    else: # risk >= 0.7
        target_exposure = 0.0

    # Ensure exposure is not negative due to floating point issues if risk is very near 0.7
    # and ensure it does not exceed max_dollar_exposure (already handled by logic if inputs are valid).
    return max(0.0, min(target_exposure, max_dollar_exposure))
