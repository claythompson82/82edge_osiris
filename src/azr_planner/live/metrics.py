from __future__ import annotations

from prometheus_client import Counter, Gauge

AZR_LIVE_TRADES_TOTAL = Counter(
    'azr_live_trades_total',
    'Total number of live paper trades executed',
    ['instrument', 'action']  # e.g., instrument="MES", action="ENTER_LONG" / "EXIT_SHORT" etc.
)

AZR_LIVE_OPEN_RISK = Gauge(
    'azr_live_open_risk',
    'Current estimated open risk in live paper trading, typically value of open positions.',
    # Could add more labels like 'symbol_group' if needed in future.
    ['instrument'] # e.g., instrument="MES"
)

# Example usage for AZR_LIVE_TRADES_TOTAL:
# AZR_LIVE_TRADES_TOTAL.labels(instrument="MES", action="ENTER_LONG").inc()

# Example usage for AZR_LIVE_OPEN_RISK:
# AZR_LIVE_OPEN_RISK.labels(instrument="MES").set(current_mes_position_value_at_risk)
