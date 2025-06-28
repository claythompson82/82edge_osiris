from __future__ import annotations

import math
from typing import Tuple, Optional, Any, Dict # AZR-13: Added Any for db_table type hint, Dict for _counters_per_registry

from prometheus_client import Counter, CollectorRegistry, REGISTRY as DEFAULT_REGISTRY

from azr_planner.schemas import TradeProposal, Instrument, Leg
from .schemas import RiskGateConfig, MES_CONTRACT_VALUE, M2K_CONTRACT_VALUE, DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE, AcceptedTradeProposalRecord
# AcceptedTradeProposalRecord needs to be imported for the db logging part.

# Tolerance for floating point comparisons
NUMERIC_COMPARISON_TOLERANCE = 1e-9

# AZR-13: Prometheus Metrics Helper Functions for Lazy Initialization
_ACCEPT_COUNTER_NAME = 'azr_riskgate_accept_total'
_REJECT_COUNTER_NAME = 'azr_riskgate_reject_total'

# Store created counters in a dictionary to ensure they are singletons per registry
_counters_per_registry: Dict[CollectorRegistry, Dict[str, Counter]] = {}

def _get_or_create_accept_counter(registry: CollectorRegistry) -> Counter:
    if registry not in _counters_per_registry:
        _counters_per_registry[registry] = {}

    metric = _counters_per_registry[registry].get(_ACCEPT_COUNTER_NAME)
    if metric is None:
        metric = Counter(
            _ACCEPT_COUNTER_NAME,
            'Total number of trade proposals accepted by the risk gate',
            registry=registry # Register with the specific registry
        )
        _counters_per_registry[registry][_ACCEPT_COUNTER_NAME] = metric

    if not isinstance(metric, Counter): # Should not happen if logic is correct
        raise TypeError(f"Metric {_ACCEPT_COUNTER_NAME} in registry is not a Counter.")
    return metric

def _get_or_create_reject_counter(registry: CollectorRegistry) -> Counter:
    if registry not in _counters_per_registry:
        _counters_per_registry[registry] = {}

    metric = _counters_per_registry[registry].get(_REJECT_COUNTER_NAME)
    if metric is None:
        metric = Counter(
            _REJECT_COUNTER_NAME,
            'Total number of trade proposals rejected by the risk gate',
            ['reason'],
            registry=registry # Register with the specific registry
        )
        _counters_per_registry[registry][_REJECT_COUNTER_NAME] = metric

    if not isinstance(metric, Counter): # Should not happen
        raise TypeError(f"Metric {_REJECT_COUNTER_NAME} in registry is not a Counter.")
    return metric


def get_instrument_contract_value(instrument: Instrument) -> float:
    """Returns the fixed contract value for known instruments."""
    if instrument == Instrument.MES:
        return MES_CONTRACT_VALUE
    elif instrument == Instrument.M2K:
        return M2K_CONTRACT_VALUE
    return DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE

def accept(
    proposal: TradeProposal,
    *,
    db_table: Optional[Any] = None,
    cfg: RiskGateConfig | None = None,
    registry: Optional[CollectorRegistry] = None
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a given trade proposal is acceptable based on risk gate rules.
    Increments Prometheus counters (using provided or default registry)
    and logs accepted proposals to LanceDB if table is provided.
    """
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    accept_counter = _get_or_create_accept_counter(active_registry)
    reject_counter = _get_or_create_reject_counter(active_registry)

    config_to_use = cfg if cfg is not None else RiskGateConfig()

    # Rule 1: Latent Risk
    if proposal.latent_risk is not None and \
       (proposal.latent_risk - config_to_use.max_latent_risk) > NUMERIC_COMPARISON_TOLERANCE:
        reject_counter.labels(reason="high_risk").inc()
        return False, "high_risk"

    # Rule 2: Confidence (proposal.confidence is non-Optional)
    if (config_to_use.min_confidence - proposal.confidence) > NUMERIC_COMPARISON_TOLERANCE:
        reject_counter.labels(reason="low_confidence").inc()
        return False, "low_confidence"

    # Rule 3: Position Size Limit
    if proposal.legs:
        total_proposed_position_usd = 0.0
        for leg in proposal.legs:
            contract_value = get_instrument_contract_value(leg.instrument)
            total_proposed_position_usd += abs(leg.size * contract_value)

        if (total_proposed_position_usd - config_to_use.max_position_usd) > NUMERIC_COMPARISON_TOLERANCE:
            reject_counter.labels(reason="position_limit").inc()
            return False, "position_limit"

    # If all checks passed:
    accept_counter.inc()

    if db_table is not None:
        import datetime
        import json
        # from .schemas import AcceptedTradeProposalRecord # Already imported at module level

        try:
            # Ensure legs are dumped correctly for JSON serialization
            legs_as_dicts = [leg.model_dump() for leg in proposal.legs] if proposal.legs else []

            record = AcceptedTradeProposalRecord(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                action=proposal.action,
                latent_risk=proposal.latent_risk,
                confidence=proposal.confidence,
                legs_json=json.dumps(legs_as_dicts)
            )
            db_table.add([record.model_dump()])
        except Exception as e:
            print(f"Error persisting accepted trade proposal to LanceDB: {e}")
            # Potentially log this error more formally
            pass

    return True, None
