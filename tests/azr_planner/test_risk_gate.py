from __future__ import annotations

import math
import pytest
import datetime
import json
from typing import List, Optional, Dict, Any, Tuple
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import MagicMock

from prometheus_client import CollectorRegistry # For test isolation

from azr_planner.schemas import TradeProposal, Leg, Instrument, Direction
from azr_planner.risk_gate import accept, RiskGateConfig
# Import counter names for use with get_sample_value
from azr_planner.risk_gate.core import _ACCEPT_COUNTER_NAME, _REJECT_COUNTER_NAME
from azr_planner.risk_gate.schemas import MES_CONTRACT_VALUE, M2K_CONTRACT_VALUE, DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE, AcceptedTradeProposalRecord
from azr_planner.risk_gate.core import NUMERIC_COMPARISON_TOLERANCE


# --- Factory Helper for TradeProposal ---
def create_trade_proposal(
    latent_risk: Optional[float] = 0.1,
    confidence: Optional[float] = 0.8,
    legs: Optional[List[Leg]] = None,
    action: str = "ENTER",
    rationale: str = "Test proposal"
) -> TradeProposal:
    if legs is None:
        legs = [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)]

    clamped_latent_risk: Optional[float] = None
    if latent_risk is not None:
        clamped_latent_risk = max(0.0, min(1.0, latent_risk))

    processed_confidence: float = confidence if confidence is not None else 0.8
    processed_confidence = max(0.0, min(1.0, processed_confidence))

    return TradeProposal(
        action=action, rationale=rationale,
        latent_risk=clamped_latent_risk, confidence=processed_confidence, legs=legs
    )

# --- Unit Tests (each gets a fresh registry for counter tests) ---
def test_accept_default_config_pass() -> None:
    proposal = create_trade_proposal(latent_risk=0.2, confidence=0.7,
        legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=100.0)])
    accepted, reason = accept(proposal, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None

def test_reject_high_latent_risk() -> None:
    cfg = RiskGateConfig(max_latent_risk=0.35)
    proposal = create_trade_proposal(latent_risk=0.35 + NUMERIC_COMPARISON_TOLERANCE * 1.1)
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "high_risk"
    proposal_at_thresh = create_trade_proposal(latent_risk=0.35)
    accepted_at, _ = accept(proposal_at_thresh, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted_at is True

def test_reject_low_confidence() -> None:
    cfg = RiskGateConfig(min_confidence=0.60)
    proposal = create_trade_proposal(confidence=0.60 - NUMERIC_COMPARISON_TOLERANCE * 1.1)
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "low_confidence"
    proposal_at_thresh = create_trade_proposal(confidence=0.60)
    accepted_at, _ = accept(proposal_at_thresh, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted_at is True

def test_reject_position_limit_exceeded_mes() -> None:
    cfg = RiskGateConfig(max_position_usd=25_000.0)
    size_just_over = (cfg.max_position_usd / MES_CONTRACT_VALUE) + 1.0
    proposal = create_trade_proposal(legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=size_just_over)])
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "position_limit"
    size_at_thresh = cfg.max_position_usd / MES_CONTRACT_VALUE
    proposal_at = create_trade_proposal(legs=[Leg(instrument=Instrument.MES, direction=Direction.LONG, size=size_at_thresh)])
    accepted_at, reason_at = accept(proposal_at, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted_at is True, f"Failed at threshold: Total USD: {size_at_thresh * MES_CONTRACT_VALUE}, Max: {cfg.max_position_usd}, Reason: {reason_at}"

def test_reject_position_limit_exceeded_m2k() -> None:
    cfg = RiskGateConfig(max_position_usd=25_000.0)
    size_just_over = (cfg.max_position_usd / M2K_CONTRACT_VALUE) + 1.0
    proposal = create_trade_proposal(legs=[Leg(instrument=Instrument.M2K, direction=Direction.SHORT, size=size_just_over)])
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "position_limit"

def test_position_limit_multiple_legs() -> None:
    cfg = RiskGateConfig(max_position_usd=25_000.0)
    proposal_pass = create_trade_proposal(legs=[
        Leg(instrument=Instrument.MES, direction=Direction.LONG, size=200.0),
        Leg(instrument=Instrument.M2K, direction=Direction.SHORT, size=150.0)])
    accepted, reason = accept(proposal_pass, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None
    proposal_fail = create_trade_proposal(legs=[
        Leg(instrument=Instrument.MES, direction=Direction.LONG, size=200.0),
        Leg(instrument=Instrument.M2K, direction=Direction.SHORT, size=151.0)])
    accepted, reason = accept(proposal_fail, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "position_limit"

def test_position_limit_other_instruments_ignored() -> None:
    cfg = RiskGateConfig(max_position_usd=25_000.0)
    proposal = create_trade_proposal(legs=[
        Leg(instrument=Instrument.US_SECTOR_ETF, direction=Direction.LONG, size=1_000_000.0),
        Leg(instrument=Instrument.ETH_OPT, direction=Direction.LONG, size=100.0)])
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None

def test_first_failure_reason_priority() -> None:
    cfg = RiskGateConfig(max_latent_risk=0.3, min_confidence=0.7, max_position_usd=1000.0)
    proposal = create_trade_proposal(latent_risk=0.4, confidence=0.6,
                                     legs=[Leg(instrument=Instrument.MES, size=100.0, direction=Direction.LONG)])
    accepted, reason = accept(proposal, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "high_risk"
    proposal_conf_fail_first = create_trade_proposal(latent_risk=0.2, confidence=0.6,
                                                     legs=[Leg(instrument=Instrument.MES, size=100.0, direction=Direction.LONG)])
    accepted, reason = accept(proposal_conf_fail_first, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "low_confidence"
    proposal_pos_fail_only = create_trade_proposal(latent_risk=0.2, confidence=0.8,
                                                   legs=[Leg(instrument=Instrument.MES, size=100.0, direction=Direction.LONG)])
    accepted, reason = accept(proposal_pos_fail_only, cfg=cfg, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is False
    assert reason == "position_limit"

def test_accept_no_legs_passes_position_check() -> None:
    proposal = create_trade_proposal(legs=None, action="HOLD")
    accepted, reason = accept(proposal, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None
    proposal_empty_legs = create_trade_proposal(legs=[], action="HOLD")
    accepted, reason = accept(proposal_empty_legs, db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None

def test_accept_none_latent_risk() -> None:
    proposal_none_risk = create_trade_proposal(latent_risk=None, confidence=0.8)
    accepted, reason = accept(proposal_none_risk, cfg=RiskGateConfig(max_latent_risk=0.1), db_table=MagicMock(), registry=CollectorRegistry())
    assert accepted is True
    assert reason is None

# --- AZR-13: Tests for DB Persistence and Prometheus Metrics ---
def get_metric_value_from_registry(registry: CollectorRegistry, metric_name: str, labels: Optional[Dict[str, str]] = None) -> float:
    """Helper to get current value of a Prometheus metric from a specific registry."""
    query_labels = labels if labels is not None else {}
    value = registry.get_sample_value(metric_name, labels=query_labels)
    return value if value is not None else 0.0

def test_accept_logs_to_db_and_increments_accept_counter() -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    mock_db_table = MagicMock()
    proposal_legs = [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)]
    proposal = create_trade_proposal(latent_risk=0.1, confidence=0.9, legs=proposal_legs)

    initial_accept_count = get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME)

    accepted, reason = accept(proposal, db_table=mock_db_table, cfg=None, registry=test_registry)

    assert accepted is True
    assert reason is None
    mock_db_table.add.assert_called_once()
    added_data_list = mock_db_table.add.call_args[0][0]
    assert len(added_data_list) == 1
    added_record_dict = added_data_list[0]
    record_obj = AcceptedTradeProposalRecord.model_validate(added_record_dict)
    assert record_obj.action == proposal.action
    assert record_obj.latent_risk == proposal.latent_risk
    assert record_obj.confidence == proposal.confidence
    assert isinstance(record_obj.timestamp, datetime.datetime)
    expected_legs_as_dicts = [leg.model_dump() for leg in proposal_legs]
    assert json.loads(record_obj.legs_json) == expected_legs_as_dicts
    assert get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME) == initial_accept_count + 1.0

@pytest.mark.parametrize("rejection_scenario, reason_key, proposal_override_dict", [
    ("high_risk", "high_risk", {"latent_risk": 0.9}),
    ("low_confidence", "low_confidence", {"confidence": 0.1}),
    ("position_limit", "position_limit", {"legs": [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1000.0)]}),
])
def test_reject_increments_reject_counter(
    rejection_scenario: str, reason_key: str, proposal_override_dict: Dict[str, Any]
) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    mock_db_table = MagicMock()
    base_proposal_fields = {"latent_risk": 0.1, "confidence": 0.8,
                            "legs": [Leg(instrument=Instrument.MES, direction=Direction.LONG, size=1.0)]}
    current_proposal_fields = {**base_proposal_fields, **proposal_override_dict}
    if "legs" in proposal_override_dict and isinstance(proposal_override_dict["legs"], list):
         current_proposal_fields["legs"] = proposal_override_dict["legs"]
    proposal = create_trade_proposal(**current_proposal_fields)

    initial_reject_count = get_metric_value_from_registry(test_registry, _REJECT_COUNTER_NAME, labels={"reason": reason_key})
    initial_accept_count = get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME)

    accepted, reason = accept(proposal, db_table=mock_db_table, cfg=None, registry=test_registry)

    assert accepted is False
    assert reason == reason_key
    mock_db_table.add.assert_not_called()
    assert get_metric_value_from_registry(test_registry, _REJECT_COUNTER_NAME, labels={"reason": reason_key}) == initial_reject_count + 1.0
    assert get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME) == initial_accept_count

# --- Hypothesis Property Test ---
@st.composite
def st_legs(draw: st.DrawFn, max_legs: int = 3) -> List[Leg]:
    num_legs = draw(st.integers(min_value=0, max_value=max_legs))
    legs_list: List[Leg] = []
    for _ in range(num_legs):
        instrument = draw(st.sampled_from(Instrument))
        size = draw(st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False))
        direction = draw(st.sampled_from(Direction))
        legs_list.append(Leg(instrument=instrument, direction=direction, size=size))
    return legs_list

@st.composite
def st_trade_proposal_for_risk_gate(draw: st.DrawFn) -> TradeProposal:
    latent_risk = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    legs = draw(st_legs())
    return TradeProposal(action="ENTER", rationale="Hypothesis generated", latent_risk=latent_risk, confidence=confidence, legs=legs)

st_risk_gate_config = st.builds(
    RiskGateConfig,
    max_latent_risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    max_position_usd=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
)

@given(proposal=st_trade_proposal_for_risk_gate(), config_input=st.one_of(st.none(), st_risk_gate_config))
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large], max_examples=50)
def test_accept_property_based(proposal: TradeProposal, config_input: Optional[RiskGateConfig]) -> None:
    test_registry = CollectorRegistry(auto_describe=True)
    cfg_to_use = config_input if config_input is not None else RiskGateConfig()

    initial_accept_val = get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME)
    initial_reject_vals = {r: get_metric_value_from_registry(test_registry, _REJECT_COUNTER_NAME, labels={"reason": r})
                           for r in ["high_risk", "low_confidence", "position_limit"]}

    expected_accept = True; expected_reason = None
    if proposal.latent_risk is not None and (proposal.latent_risk - cfg_to_use.max_latent_risk) > NUMERIC_COMPARISON_TOLERANCE:
        expected_accept = False; expected_reason = "high_risk"
    if expected_accept and (cfg_to_use.min_confidence - proposal.confidence) > NUMERIC_COMPARISON_TOLERANCE:
        expected_accept = False; expected_reason = "low_confidence"
    if expected_accept and proposal.legs:
        current_pos_usd = sum(abs(leg.size * (MES_CONTRACT_VALUE if leg.instrument == Instrument.MES else M2K_CONTRACT_VALUE if leg.instrument == Instrument.M2K else DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE)) for leg in proposal.legs)
        if (current_pos_usd - cfg_to_use.max_position_usd) > NUMERIC_COMPARISON_TOLERANCE:
            expected_accept = False; expected_reason = "position_limit"

    actual_accepted, actual_reason = accept(proposal, cfg=cfg_to_use, db_table=MagicMock(), registry=test_registry)

    assert actual_accepted == expected_accept
    assert actual_reason == expected_reason

    if expected_accept:
        assert get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME) == initial_accept_val + 1.0
    else:
        assert get_metric_value_from_registry(test_registry, _ACCEPT_COUNTER_NAME) == initial_accept_val
        if expected_reason:
            assert get_metric_value_from_registry(test_registry, _REJECT_COUNTER_NAME, labels={"reason": expected_reason}) == initial_reject_vals.get(expected_reason, 0) + 1.0
            for r, val in initial_reject_vals.items():
                if r != expected_reason:
                    assert get_metric_value_from_registry(test_registry, _REJECT_COUNTER_NAME, labels={"reason": r}) == val
