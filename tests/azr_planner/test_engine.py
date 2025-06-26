"""Tests for AZR Planner engine."""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timezone

from azr_planner.engine import generate_plan
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Leg # Added Leg

# Define a strategy for PlanningContext
# Generate instrument string values directly from the enum
st_instrument_values = st.sampled_from([inst.value for inst in Instrument])

st_planning_context = st.builds(
    PlanningContext,
    timestamp=st.datetimes(
        min_value=datetime(2000, 1, 1), # Naive datetime for min_value
        max_value=datetime(2040, 1, 1), # Naive datetime for max_value
        timezones=st.just(timezone.utc) # Specify UTC timezone strategy
    ),
    equity_curve=st.lists(
        st.floats(min_value=-1e6, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=100
    ),
    vol_surface=st.dictionaries(
        keys=st_instrument_values, # Use string values for keys
        values=st.floats(min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=len(Instrument) # Max size based on number of instruments
    ),
    risk_free_rate=st.one_of(
        st.none(),
        st.floats(min_value=-0.05, max_value=0.2,  allow_nan=False, allow_infinity=False)
    )
)


def test_generate_plan_stub_output() -> None:
    """Test the stub output of generate_plan."""
    dummy_ctx_data = {
        "timestamp": datetime.now(timezone.utc),
        "equity_curve": [100.0] * 30,
        "vol_surface": {"MES": 0.15},
        "risk_free_rate": 0.01,
    }
    ctx = PlanningContext.model_validate(dummy_ctx_data)
    proposal = generate_plan(ctx)

    assert isinstance(proposal, TradeProposal)
    assert proposal.latent_risk == 0.0
    assert proposal.legs == []

@given(ctx=st_planning_context)
def test_property_generate_plan_invariants(ctx: PlanningContext) -> None:
    """
    Property test for generate_plan.
    Verifies invariants:
    - latentRisk is always 0.0 (for the stub).
    - legs is always an empty list (for the stub).
    - Output is always a valid TradeProposal model.
    """
    proposal = generate_plan(ctx)

    assert isinstance(proposal, TradeProposal)
    assert proposal.latent_risk == 0.0, "Latent risk should be 0.0 for the stub implementation."
    assert proposal.legs == [], "Legs should be empty for the stub implementation."
    # Pydantic model validation for TradeProposal (latent_risk >= 0) is implicitly checked by construction.

def test_generate_plan_with_minimal_context() -> None:
    """Test generate_plan with the most minimal valid PlanningContext."""
    minimal_ctx_data = {
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "equity_curve": [float(i) for i in range(30)],
    }
    ctx = PlanningContext.model_validate(minimal_ctx_data)

    proposal = generate_plan(ctx)

    assert isinstance(proposal, TradeProposal)
    assert proposal.latent_risk == 0.0
    assert proposal.legs == []

def test_generate_plan_with_all_optional_fields() -> None:
    """Test generate_plan with all optional fields provided in PlanningContext."""
    full_ctx_data = {
        "timestamp": datetime(2024, 7, 30, 12, 30, 0, tzinfo=timezone.utc),
        "equity_curve": [100.0 + i/10.0 for i in range(50)],
        "vol_surface": {
            Instrument.MES.value: 0.18, # Use enum values as keys
            Instrument.M2K.value: 0.22,
            Instrument.US_SECTOR_ETF.value: 0.15,
            Instrument.ETH_OPT.value: 0.88
        },
        "risk_free_rate": 0.025
    }
    ctx = PlanningContext.model_validate(full_ctx_data)

    proposal = generate_plan(ctx)

    assert isinstance(proposal, TradeProposal)
    assert proposal.latent_risk == 0.0
    assert proposal.legs == []
    assert ctx.risk_free_rate == 0.025
    assert ctx.vol_surface[Instrument.ETH_OPT.value] == 0.88
