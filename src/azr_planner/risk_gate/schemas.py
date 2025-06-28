from __future__ import annotations

from pydantic import BaseModel, Field

class RiskGateConfig(BaseModel):
    """
    Configuration for the Risk Gate.
    Specifies thresholds for various risk checks.
    """
    max_latent_risk: float = Field(
        default=0.35,
        description="Maximum allowed latent risk. Proposals exceeding this will be rejected."
    )
    min_confidence: float = Field(
        default=0.60,
        description="Minimum required confidence. Proposals below this will be rejected."
    )
    max_position_usd: float = Field(
        default=25_000.0,
        description="Maximum allowed total USD value of the proposed position."
    )

    # Allow model to be used as a hashable type, e.g. in dicts if needed
    # model_config = {"frozen": True} # Not strictly needed unless used as dict keys

# Contract values (as per AZR-12 spec for the stub)
# These are simplified fixed values for specific instruments for now.
# A more advanced system would fetch current prices and multipliers.
MES_CONTRACT_VALUE: float = 50.0
M2K_CONTRACT_VALUE: float = 100.0
# For other instruments, their contribution to max_position_usd check will be 0
# if not explicitly defined here and handled by a default in the core logic.
DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE: float = 0.0


# AZR-13: Schema for LanceDB record of accepted trade proposals
import datetime # Ensure datetime is imported for type hint
from typing import Optional # Ensure Optional is imported

class AcceptedTradeProposalRecord(BaseModel):
    """
    Schema for storing accepted trade proposals in LanceDB.
    """
    timestamp: datetime.datetime = Field(description="Timestamp when the proposal was accepted and recorded.")
    action: str = Field(description="Proposed trading action (e.g., ENTER, EXIT, HOLD).")
    latent_risk: Optional[float] = Field(description="Calculated latent risk score at the time of proposal.")
    confidence: float = Field(description="Calculated confidence score at the time of proposal.")
    legs_json: str = Field(description="JSON string representation of the trade proposal legs.")
    # Optional: Add proposal_id if there's a unique ID associated with TradeProposal
    # proposal_id: Optional[str] = None
