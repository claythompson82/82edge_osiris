"""Core trade planning engine for AZR Planner."""

from .schemas import PlanningContext, TradeProposal, Leg, Instrument, Direction # Changed TradePlan to TradeProposal
from .math_utils import latent_risk as calculate_latent_risk # Using the new latent_risk function

def generate_plan(ctx: PlanningContext) -> TradeProposal: # Return type changed to TradeProposal
    """
    Generates a trade proposal based on the computed latent risk.
    Actions ('ENTER', 'HOLD', 'EXIT') are determined by risk thresholds.
    Legs are populated based on the action.
    """

    # 1. Calculate latent risk using the updated math_utils.latent_risk
    # The new latent_risk function only requires equity_curve.
    current_latent_risk = calculate_latent_risk(
        equity_curve=ctx.equity_curve # Pass only the equity curve
    )

    # 2. Calculate confidence
    # confidence = 1 – latent_risk (linear for now)
    # Return round(confidence, 3).
    current_confidence = round(1.0 - current_latent_risk, 3)

    # 3. Determine action based on latent risk thresholds (§3.4)
    action: str
    rationale: str
    legs: list[Leg] | None = None # Initialize legs, default to None or empty list

    if current_latent_risk < 0.30:
        action = "ENTER"
        rationale = "Latent risk is low, favorable for new positions."
        # For ENTER: one MES micro-future, direction = BUY (LONG), qty = 1.
        legs = [
            Leg(
                instrument=Instrument.MES,
                direction=Direction.LONG,
                size=1.0,
                limit_price=None # Assuming no limit price for this stub
            )
        ]
    elif current_latent_risk <= 0.70: # 0.30 <= latent_risk <= 0.70
        action = "HOLD"
        rationale = "Latent risk is moderate, maintaining current positions."
        legs = None # Or [] as per §3.5 "legs =[]"
    else: # latent_risk > 0.70
        action = "EXIT"
        rationale = "Latent risk is high, reducing exposure."
        # For EXIT: close all MES legs, direction = SELL (SHORT), qty equals current open size (stub qty=1).
        legs = [
            Leg(
                instrument=Instrument.MES,
                direction=Direction.SHORT,
                size=1.0, # Stub quantity
                limit_price=None # Assuming no limit price for this stub
            )
        ]

    return TradeProposal(
        action=action,
        rationale=rationale,
        latent_risk=current_latent_risk,
        confidence=current_confidence,
        legs=legs
    )
