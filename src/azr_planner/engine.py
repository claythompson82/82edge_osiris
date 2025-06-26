"""Core trade planning engine for AZR Planner."""

from typing import List # Required for List type hint, even if not used in stub body
from .schemas import PlanningContext, TradeProposal, Leg # Leg needed for TradeProposal

def generate_plan(ctx: PlanningContext) -> TradeProposal:
    """
    Generates a trade proposal based on the given planning context.

    For this initial scaffolding, it returns a deterministic dummy output
    with latent_risk = 0.0 and an empty list of legs.
    """
    return TradeProposal(latent_risk=0.0, legs=[])
