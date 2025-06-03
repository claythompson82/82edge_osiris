"""
Osiris • Proof-able Reward Module
================================
Implements the composite reward R* described in *“PAC-Bayesian Sharpe Ratio
Uplift Estimator for Curriculum Gain Provability”* (§ 3.2) and
*“Provable Curriculum RL Integrating DGM with AZR Self-Play”* (§ 4.1).

Notation (from papers):
    Rs : Sharpe-ratio uplift term                 (Eq. 2)
    Rt : Temporal-consistency penalty             (Eq. 3)
    Re : Exposure decay / risk budget term        (Eq. 4)
    Rh : Human-override multiplier (0.0–2.0)      (Implementation Note)

Composite reward:
    R* = Rh · (Rs  −  λ_t Rt  −  λ_e Re)          (Eq. 5)

λ_t and λ_e are tunable weights (default 0.2, 0.1).
Rh = 1.0 unless human feedback overwrites confidence.

This file only exposes **proofable_reward** (called by TAC + Orchestrator).

All heavy lifting (Sharpe, drawdown, etc.) will be filled in by Jules.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TypedDict, Optional

# --------------------------------------------------------------------------- #
# ✨ SCHEMA TYPES – kept ultra-minimal so this module has *zero* heavy deps.   #
# --------------------------------------------------------------------------- #
class TradeJSON(TypedDict):
    """Subset of a Phi-3 trade-proposal JSON needed for reward calc."""
    ticker: str
    side: str           # "LONG" | "SHORT"
    entry_price: float
    exit_price: Optional[float]
    entry_ts: str       # ISO-8601
    exit_ts: Optional[str]
    pnl_pct: Optional[float]
    confidence: float   # 0-1

class MarketSnapshot(TypedDict):
    """Any intraday features we need (OHLC, volume, VIX, ...)"""
    price_series: list[float]   # chronological close prices
    ts_series: list[str]        # ISO-timestamps aligned with price_series

class HumanFeedback(TypedDict, total=False):
    """Optional human label pushed via /feedback/phi3/"""
    override_confidence: float  # 0-1
    comment: str

# --------------------------------------------------------------------------- #
# 🔑 MAIN ENTRY POINT                                                         #
# --------------------------------------------------------------------------- #
def proofable_reward(
    trade: TradeJSON,
    market: MarketSnapshot,
    feedback: Optional[HumanFeedback] = None,
    *,
    lambda_t: float = 0.20,
    lambda_e: float = 0.10,
) -> float:
    """
    Compute R* (Eq. 5) for a completed trade *or* a rolling in-flight update.

    Parameters
    ----------
    trade      : parsed Phi-3 proposal with at least pnl_% computed
    market     : aligned market data covering trade duration
    feedback   : optional human corrections / ratings
    lambda_t   : weight for temporal consistency penalty
    lambda_e   : weight for exposure decay penalty

    Returns
    -------
    float
        Reward signal ∈ ℝ, positive favours agent behaviour.

    TODO(Jules)
    -----------
    1. Implement _sharpe_uplift()           -> Rs
    2. Implement _temporal_consistency()    -> Rt
    3. Implement _exposure_decay()          -> Re
    4. Combine with Rh (human factor).
    5. Add unit tests in tests/test_reward.py
    """
    # -- STEP 0: Human multiplier ------------------------------------------- #
    Rh = feedback.get("override_confidence", 1.0) if feedback else 1.0
    Rh = max(0.0, min(2.0, Rh))  # hard-clamp

    # -- STEP 1: core terms (placeholders) ---------------------------------- #
    Rs = _sharpe_uplift(trade, market)            # Eq. 2
    Rt = _temporal_consistency(trade, market)     # Eq. 3
    Re = _exposure_decay(trade, market)           # Eq. 4

    # -- STEP 2: composite --------------------------------------------------- #
    reward_star = Rh * (Rs - lambda_t * Rt - lambda_e * Re)
    return reward_star


# --------------------------------------------------------------------------- #
# 🔧 INTERNAL UTILITIES – Jules will flesh these out                          #
# --------------------------------------------------------------------------- #

def _sharpe_uplift(trade: TradeJSON, market: MarketSnapshot) -> float:
    """Eq. 2: ΔSharpe between agent PnL and benchmark (e.g., SPY)."""
    # TODO(Jules): implement; may import numpy/pandas inside function
    return 0.0

def _temporal_consistency(trade: TradeJSON, market: MarketSnapshot) -> float:
    """Eq. 3: penalises jittery exits vs. stated horizon."""
    # TODO(Jules)
    return 0.0

def _exposure_decay(trade: TradeJSON, market: MarketSnapshot) -> float:
    """Eq. 4: discourages over-concentration in same asset/direction."""
    # TODO(Jules)
    return 0.0
