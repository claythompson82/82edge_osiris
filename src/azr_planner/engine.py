"""Core trade planning engine for AZR Planner."""

from .schemas import PlanningContext, TradePlan, Leg, Instrument, Direction
from .math_utils import latent_risk as calculate_latent_risk # Alias to avoid confusion if needed

def generate_plan(ctx: PlanningContext) -> TradePlan:
    """
    Generates a trade plan based on the computed latent risk.
    If latent risk <= 0.3, an "ENTER" plan with one MES leg is proposed.
    Otherwise, a "HOLD" plan is proposed.
    """

    # 1. Calculate latent risk
    # Ensure all necessary fields from PlanningContext are passed to calculate_latent_risk
    # calculate_latent_risk(equity_curve: List[float], vol_surface: Dict[str, float], risk_free_rate: float)
    # PlanningContext fields: equity_curve, vol_surface, risk_free_rate
    # The schema ensures these are present and correctly typed.

    current_latent_risk = calculate_latent_risk(
        equity_curve=ctx.equity_curve,
        vol_surface=ctx.vol_surface,
        risk_free_rate=ctx.risk_free_rate
    )

    # 2. Determine action based on latent risk
    if current_latent_risk <= 0.3:
        # Create a single leg: "MES" buy 1 contract
        # Placeholder for limit_price, assuming it's optional or can be None
        enter_leg = Leg(
            instrument=Instrument.MES,
            direction=Direction.LONG,
            size=1.0, # Buy 1 contract
            limit_price=None # Or some calculated price; stubbing with None
        )
        return TradePlan(
            action="ENTER",
            rationale="Latent risk within threshold.",
            latent_risk=current_latent_risk,
            confidence=1.0, # Stub confidence for "ENTER"
            legs=[enter_leg]
        )
    else:
        return TradePlan(
            action="HOLD",
            rationale="Latent risk exceeds threshold.",
            latent_risk=current_latent_risk,
            confidence=0.5, # Stub confidence for "HOLD"
            legs=None # Or empty list: []
        )
