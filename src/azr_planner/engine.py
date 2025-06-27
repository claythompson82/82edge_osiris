"""Core trade planning engine for AZR Planner."""

import math
import numpy as np
import pandas as pd
from typing import Optional

from .schemas import PlanningContext, TradeProposal, Leg, Instrument, Direction
from .math_utils import (
    atr as calculate_atr,
    kelly_fraction as calculate_kelly_fraction,
    ema as calculate_ema,
    # latent_risk as calculate_latent_risk # Keep if needed for comparison or phased rollout
)

# --- Constants and Assumptions (pending Design Doc §4) ---
# Signal Generation
ASSUMED_EMA_SHORT_PERIOD = 12 # Placeholder
ASSUMED_EMA_LONG_PERIOD = 26  # Placeholder
ASSUMED_SIGNAL_EMA_PERIOD = 9 # Placeholder for MACD-like signal line
# Position Sizing
ASSUMED_KELLY_MU_LOOKBACK = 60 # Placeholder: lookback for returns mean for Kelly
ASSUMED_KELLY_SIGMA_LOOKBACK = 60 # Placeholder: lookback for returns std dev for Kelly
KELLY_ATR_CAP_MULTIPLIER = 3.0
# Confidence
ASSUMED_CONFIDENCE_SIGNAL_STD_LOOKBACK = 30 # Placeholder: lookback for signal std dev for confidence
# Action Thresholds (example, will depend on signal properties)
ASSUMED_ENTER_THRESHOLD_STRONG = 0.5 # Placeholder: e.g. normalized signal > 0.5
ASSUMED_ENTER_THRESHOLD_WEAK = 0.1 # Placeholder
ASSUMED_EXIT_THRESHOLD_STRONG = -0.5 # Placeholder
ASSUMED_EXIT_THRESHOLD_WEAK = -0.1 # Placeholder


def _sigmoid(x: float) -> float:
    """Helper for sigmoid function."""
    return 1 / (1 + math.exp(-x))

def generate_plan(ctx: PlanningContext) -> TradeProposal:
    """
    Generates a trade proposal based on signal generation, position sizing, and confidence.
    Logic to be guided by AZR Planner Design PDF §4.
    """
    # --- Initialize output fields ---
    action: str = "HOLD" # Default action
    rationale: str = "No clear signal or conditions not met."
    legs: Optional[list[Leg]] = None
    calculated_confidence: float = 0.5 # Default confidence

    signal_value_out: Optional[float] = None
    atr_value_out: Optional[float] = None
    kelly_fraction_out: Optional[float] = None
    target_position_size_out: Optional[float] = None
    # current_latent_risk_out: Optional[float] = calculate_latent_risk(ctx.equity_curve) # If still needed

    # --- 1. Preparations ---
    # Ensure we have enough data from context
    if len(ctx.daily_history_hlc) < max(ASSUMED_EMA_LONG_PERIOD, ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK, 15): # 15 for ATR(14)
        rationale = "Insufficient historical data for calculation."
        return TradeProposal(
            action="HOLD", # Or a specific "ERROR" action if defined
            rationale=rationale,
            # latent_risk=current_latent_risk_out,
            confidence=0.0, # Low confidence due to insufficient data
            legs=None
        )

    close_prices = [hlc[2] for hlc in ctx.daily_history_hlc]

    # --- 2. Calculate ATR ---
    # ATR calculation needs HLC data.
    atr_value = calculate_atr(ctx.daily_history_hlc, window=14) # Standard ATR(14)
    atr_value_out = atr_value if not math.isnan(atr_value) else None

    if atr_value_out is None or atr_value_out == 0: # ATR is zero if prices are flat, can cause issues if used as divisor
        rationale = "ATR calculation resulted in None or zero, cannot proceed with sizing."
        # Consider if this should be a HOLD or a more critical error/flag.
        # If ATR is critical for any position sizing or risk, and it's zero/None, may not be safe to trade.
        return TradeProposal(action="HOLD", rationale=rationale, confidence=0.1, atr_value=atr_value_out)


    # --- 3. Signal Generation (EMA x vol-adjust) ---
    # TODO(AZR-05): Implement full signal generation logic from Design Doc §4.
    # This includes specific EMA periods, volume adjustment, and potentially other indicators.
    # Placeholder: Using a simple MACD-like signal for now.

    # Ensure close_prices are sufficient for EMA calculations
    if len(close_prices) < ASSUMED_EMA_LONG_PERIOD:
        rationale = "Insufficient close prices for EMA calculation."
        return TradeProposal(action="HOLD", rationale=rationale, confidence=0.1, atr_value=atr_value_out)

    ema_short = calculate_ema(close_prices, span=ASSUMED_EMA_SHORT_PERIOD)
    ema_long = calculate_ema(close_prices, span=ASSUMED_EMA_LONG_PERIOD)

    if math.isnan(ema_short) or math.isnan(ema_long):
        rationale = "EMA calculation resulted in NaN."
        return TradeProposal(action="HOLD", rationale=rationale, confidence=0.1, atr_value=atr_value_out)

    # Example: MACD line
    macd_line = ema_short - ema_long

    # Example: Signal line (EMA of MACD line)
    # Need a series of MACD values to calculate its EMA.
    # For simplicity, let's assume the raw macd_line is the "signal" for now.
    # A proper implementation would calculate historical MACD values first.
    # This is a major simplification.
    current_signal_value = macd_line
    signal_value_out = current_signal_value

    # --- 4. Position Sizing (Kelly fraction w/ cap 3 × ATR) ---
    # TODO(AZR-05): Implement full position sizing logic from Design Doc §4.
    # This includes specific derivation of mu and sigma for Kelly,
    # and the precise mechanism for the "cap 3 x ATR".
    # Placeholder: Using simple historical returns for Kelly inputs. ATR cap not implemented.

    # Calculate returns
    if len(close_prices) < max(ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK) + 1:
        rationale = "Insufficient data for Kelly fraction inputs."
        # Potentially use signal but no sizing, or just HOLD
    else:
        returns = pd.Series(close_prices).pct_change().dropna().tolist()
        if len(returns) >= max(ASSUMED_KELLY_MU_LOOKBACK, ASSUMED_KELLY_SIGMA_LOOKBACK):
            # Ensure results from numpy are explicitly float for type checker
            mu_returns = float(np.mean(returns[-ASSUMED_KELLY_MU_LOOKBACK:]))
            sigma_returns = float(np.std(returns[-ASSUMED_KELLY_SIGMA_LOOKBACK:]))

            if sigma_returns > 0: # Avoid division by zero
                # Assuming mu_returns is excess return for Kelly. If not, ctx.risk_free_rate should be used.
                # For now, let's assume mu_returns is okay.
                raw_kelly_fraction = calculate_kelly_fraction(mu=mu_returns, sigma=sigma_returns)
                kelly_fraction_out = raw_kelly_fraction

                # Position size based on Kelly (e.g., fraction of equity).
                # This part is highly dependent on how Kelly fraction is translated to size.
                # Let's assume Kelly fraction applies to a nominal account size (e.g. 100,000)
                # and then we derive number of contracts. This needs proper definition.
                # For now, let's say target_position_size is the Kelly fraction itself,
                # and capping logic will be applied later or is part of "how Kelly fraction translates to size".

                # ASSUMPTION: Kelly fraction is a multiplier for a base position size.
                # Let base size be 1 unit (e.g. 1 contract).
                # This is a placeholder for actual position sizing logic.
                position_size_kelly = raw_kelly_fraction * 1.0 # Example: 1 unit scaled by Kelly

                # ATR based cap: "cap 3 x ATR"
                # How ATR translates to a position size cap is not defined.
                # Assumption: ATR is a monetary value. Cap is also monetary. Or ATR normalizes something.
                # If signal is positive, consider entering. If negative, consider exiting.
                # This is a very simplified placeholder for position sizing.
                # Let's assume for now the target_position_size_out is just the Kelly-derived one,
                # and the capping would be applied if we knew how.
                target_position_size_out = position_size_kelly # Placeholder, cap not applied yet.
            else:
                kelly_fraction_out = 0.0 # Sigma is zero, no conviction
                target_position_size_out = 0.0
        else:
            rationale = "Not enough return data points for Kelly calculation after processing."
            # Fallback or hold.

    # --- 5. Confidence Calculation (sigmoid(signal / σ)) ---
    # TODO(AZR-05): Implement proper confidence calculation from Design Doc §4.
    # This includes the definition and calculation of σ (sigma for the signal).
    # Placeholder: Using ATR as a proxy for signal volatility for normalization.
    # For now, using a dummy sigma or a fixed confidence.

    # Simplified: if signal is strong, higher confidence. This is not per spec.
    # A more proper way would be to get a series of `current_signal_value` over time.
    # For now, let's compute a proxy for signal volatility if possible, or use a fixed confidence.
    # As a placeholder, let's use a fixed confidence or a simple scaling of signal.
    if signal_value_out is not None:
        # Example: Normalize signal by some factor (e.g., recent ATR or signal std dev)
        # Placeholder for sigma_signal:
        sigma_signal_proxy = atr_value_out if atr_value_out and atr_value_out > 0 else 1.0
        if sigma_signal_proxy == 0: sigma_signal_proxy = 1.0 # Avoid division by zero

        normalized_signal = signal_value_out / sigma_signal_proxy
        calculated_confidence = _sigmoid(normalized_signal) # Apply sigmoid to normalized signal
        calculated_confidence = round(calculated_confidence, 3)
    else:
        calculated_confidence = 0.1 # Low confidence if no signal

    # --- 6. Action Determination & Leg Population ---
    # TODO(AZR-05): Implement full action determination logic from Design Doc §4.
    # This includes how signals, position sizes, confidence, and current_positions translate
    # into ENTER, HOLD, EXIT actions and corresponding leg populations.
    # Placeholder: Simplified logic based on basic thresholds and current holdings.

    # Example simplified logic:
    # Assume target_position_size_out is the desired number of contracts (long if positive, short if negative, but Kelly usually for long).
    # Let's assume signal_value_out drives direction, target_position_size_out drives magnitude.

    current_holding_size = 0.0
    current_direction = None
    if ctx.current_positions:
        for leg in ctx.current_positions:
            if leg.instrument == Instrument.MES: # Assuming MES for now
                if leg.direction == Direction.LONG:
                    current_holding_size += leg.size
                    if current_direction is None: current_direction = Direction.LONG
                elif leg.direction == Direction.SHORT:
                    current_holding_size -= leg.size # Netting off, simple sum for now
                    if current_direction is None: current_direction = Direction.SHORT

    # Default action is HOLD, rationale already set.
    if target_position_size_out is not None and target_position_size_out > 0 and signal_value_out is not None:
        # Positive signal, positive target size -> consider LONG
        if signal_value_out > ASSUMED_ENTER_THRESHOLD_WEAK: # Some positive signal
            if current_holding_size <= 0: # Not long or flat/short
                action = "ENTER"
                rationale = f"Positive signal ({signal_value_out:.2f}), target size {target_position_size_out:.2f}. Entering LONG."
                legs = [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=target_position_size_out)]
            elif current_holding_size > 0 and abs(current_holding_size - target_position_size_out) > 0.1: # Already long, but size differs
                action = "ADJUST" # Or treat as ENTER/EXIT to simplify
                rationale = f"Positive signal ({signal_value_out:.2f}). Adjusting LONG position from {current_holding_size} to {target_position_size_out:.2f}."
                # This would require more complex leg generation (e.g. new leg for diff, or replace existing)
                # For now, let's simplify: if target is different, we just propose the target.
                # The interpretation of how to achieve that is for execution.
                # Or, if simplifying, only enter if flat.
                legs = [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=target_position_size_out)]
                action = "ENTER" # Re-classifying ADJUST as ENTER for simplicity of current schema actions
            else: # Already long and size is similar
                action = "HOLD"
                rationale = f"Positive signal ({signal_value_out:.2f}), but already holding similar LONG position."

        # Negative signal -> consider EXIT if holding LONG
        elif signal_value_out < ASSUMED_EXIT_THRESHOLD_WEAK: # Some negative signal
            if current_holding_size > 0: # Currently long
                action = "EXIT"
                rationale = f"Negative signal ({signal_value_out:.2f}). Exiting current LONG position of {current_holding_size}."
                legs = [Leg(instrument=Instrument.MES, direction=Direction.SHORT, size=current_holding_size)] # Exit entire position
            # If signal is negative and target_position_size_out is also suggesting short, logic would be here.
            # Kelly fraction as implemented is usually for long bets (mu > 0).
            # So target_position_size_out > 0 means a long position.
            # If we want to handle short based on negative signal, sizing needs different rule or interpretation.
            else:
                action = "HOLD"
                rationale = f"Negative signal ({signal_value_out:.2f}), but no current LONG position to exit."
        else: # Signal is weak / neutral
            action = "HOLD"
            rationale = f"Neutral signal ({signal_value_out:.2f}). Holding current position."

    elif target_position_size_out is not None and target_position_size_out == 0 and signal_value_out is not None:
        # Kelly suggests no position
        if current_holding_size > 0 and signal_value_out < ASSUMED_ENTER_THRESHOLD_WEAK : # If holding long and signal not strong positive
            action = "EXIT"
            rationale = f"Kelly fraction is zero and signal is not strong positive. Exiting current LONG position of {current_holding_size}."
            legs = [Leg(instrument=Instrument.MES, direction=Direction.SHORT, size=current_holding_size)]
        elif current_holding_size < 0 and signal_value_out > ASSUMED_EXIT_THRESHOLD_WEAK: # If holding short and signal not strong negative
             action = "EXIT"
             rationale = f"Kelly fraction is zero and signal is not strong negative. Exiting current SHORT position of {abs(current_holding_size)}."
             legs = [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=abs(current_holding_size))]
        else:
            action = "HOLD"
            rationale = "Kelly fraction is zero, no new positions. Holding current state."

    # Final check on rationale if it's still the default "No clear signal..."
    if rationale == "No clear signal or conditions not met." and signal_value_out is not None:
        rationale = f"Signal ({signal_value_out:.2f}) and Kelly ({kelly_fraction_out}) did not meet thresholds for action. Holding."


    return TradeProposal(
        action=action,
        rationale=rationale,
        # latent_risk=current_latent_risk_out, # If calculating
        confidence=calculated_confidence,
        legs=legs,
        signal_value=signal_value_out,
        atr_value=atr_value_out,
        kelly_fraction_value=kelly_fraction_out,
        target_position_size=target_position_size_out,
        # expected_pnl_per_unit=... # Not calculated yet
    )
