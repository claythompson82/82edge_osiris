"""Core trade planning engine for AZR Planner."""

import math
# import numpy as np # Not directly used in this version of engine
# import pandas as pd # Not directly used in this version of engine
from typing import Optional, cast

from .schemas import PlanningContext, TradeProposal, Leg, Instrument, Direction
from .position import position_size # AZR-11: Import position_size
from .math_utils import (
    latent_risk_v2,
    bayesian_confidence,
)

# Constants for action thresholds (final §4.5)
ENTER_LR_THRESHOLD = 0.25
ENTER_CONF_THRESHOLD = 0.7
EXIT_LR_THRESHOLD = 0.7
EXIT_CONF_THRESHOLD = 0.4

# AZR-11: Constants for position sizing
DEFAULT_MAX_LEVERAGE = 2.0
MES_CONTRACT_MULTIPLIER = 5.0 # Standard multiplier for MES futures
MIN_CONTRACT_SIZE = 0.001 # Smallest practical contract size to trade, to avoid dust


def generate_plan(ctx: PlanningContext) -> TradeProposal:
    """
    Generates a trade proposal based on Latent Risk v2 and Bayesian Confidence.
    Includes risk-adjusted position sizing for ENTER actions.
    Uses action thresholds from AZR Planner Design PDF §4.5.
    """

    lr = latent_risk_v2(ctx.equity_curve)
    conf = bayesian_confidence(wins=ctx.n_successes, losses=ctx.n_failures)

    action: str
    rationale: str
    legs: Optional[list[Leg]] = None

    # Determine action based on thresholds
    if lr < ENTER_LR_THRESHOLD and conf > ENTER_CONF_THRESHOLD:
        action = "ENTER"
        rationale = f"Favorable conditions: Latent Risk ({lr:.2f}) < {ENTER_LR_THRESHOLD} and Confidence ({conf:.2f}) > {ENTER_CONF_THRESHOLD}."

        # AZR-11: Position Sizing Logic for ENTER action
        current_equity = ctx.equity_curve[-1] # Assuming last point of equity_curve is current equity

        # Pydantic validation ensures daily_history_hlc has min_length=15, so it's never None or empty.
        mes_price = ctx.daily_history_hlc[-1][2] # Last close price for MES

        if mes_price <= 0:
            action = "HOLD"
            rationale += f" Invalid MES price ({mes_price:.2f}) for sizing, holding."
            legs = None
        else:
            target_dollar_exposure = position_size(
                    latent_risk=lr,
                    equity=current_equity,
                    max_leverage=DEFAULT_MAX_LEVERAGE
                )
            # This block was previously over-indented. Corrected now.
            contract_value = mes_price * MES_CONTRACT_MULTIPLIER
            # Assuming mes_price > 0 (checked before) and MES_CONTRACT_MULTIPLIER > 0,
            # contract_value will be > 0. So, direct calculation.
            calculated_size = target_dollar_exposure / contract_value

            if calculated_size >= MIN_CONTRACT_SIZE:
                legs = [
                    Leg(
                        instrument=Instrument.MES,
                        direction=Direction.LONG, # Assuming ENTER means LONG for MES as per original logic
                        size=calculated_size
                    )
                ]
                rationale += f" Sized to {calculated_size:.2f} contracts for MES."
            else:
                # If calculated size is too small, either don't trade or it implies HOLD.
                action = "HOLD"
                rationale += f" Calculated size ({calculated_size:.4f}) too small, holding."
                legs = None

    elif lr > EXIT_LR_THRESHOLD or conf < EXIT_CONF_THRESHOLD:
        action = "EXIT"
        rationale = f"Unfavorable conditions: Latent Risk ({lr:.2f}) > {EXIT_LR_THRESHOLD} or Confidence ({conf:.2f}) < {EXIT_CONF_THRESHOLD}."

        current_mes_long_qty = 0.0 # Initialize as float
        if ctx.current_positions:
            for leg_item in ctx.current_positions: # Changed loop var name to avoid conflict
                if leg_item.instrument == Instrument.MES and leg_item.direction == Direction.LONG:
                    current_mes_long_qty += leg_item.size

        if current_mes_long_qty > 0:
            legs = [
                Leg(
                    instrument=Instrument.MES,
                    direction=Direction.SHORT,
                    size=current_mes_long_qty
                )
            ]
        else:
            legs = [
                Leg(
                    instrument=Instrument.MES,
                    direction=Direction.SHORT,
                    size=1.0
                )
            ]
            rationale += " Proposing to short 1 MES contract."
    else:
        action = "HOLD"
        rationale = f"Neutral conditions: Latent Risk ({lr:.2f}), Confidence ({conf:.2f}). Holding positions."
        legs = None

    return TradeProposal(
        action=action,
        rationale=rationale,
        latent_risk=round(lr, 3),
        confidence=round(conf, 3),
        legs=legs
    )

from typing import Dict, Any # Added for type hint
# from .backtest.runner import run_walk_forward # Moved import to local scope

def backtest_strategy(ctx: PlanningContext, window_days: int = 30) -> Dict[str, Any]:
    """
    Runs a walk-forward backtest for the strategy defined by the engine
    and returns the results as a dictionary.
    """
    from .backtest.runner import run_walk_forward # Local import to break cycle
    report = run_walk_forward(full_history_ctx=ctx, window_days=window_days)
    return report.model_dump()
