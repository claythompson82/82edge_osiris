from __future__ import annotations

import pytest
import datetime

from azr_planner.live.schemas import LiveConfig, LivePosition, LivePnl

def test_live_config_instantiation() -> None:
    """Test basic instantiation and default values for LiveConfig."""
    config = LiveConfig(symbol="MESU24", initial_equity=100000.0)
    assert config.symbol == "MESU24"
    assert config.initial_equity == 100000.0
    assert config.max_risk_per_trade_pct == 0.01 # Default
    assert config.max_drawdown_pct_account == 0.10 # Default

    # Test with explicit risk params
    config_explicit = LiveConfig(
        symbol="MNQZ23",
        initial_equity=50000.0,
        max_risk_per_trade_pct=0.02,
        max_drawdown_pct_account=0.05
    )
    assert config_explicit.symbol == "MNQZ23"
    assert config_explicit.initial_equity == 50000.0
    assert config_explicit.max_risk_per_trade_pct == 0.02
    assert config_explicit.max_drawdown_pct_account == 0.05

    # Test validation (e.g., initial_equity >= 0)
    with pytest.raises(ValueError): # Pydantic uses ValueError for validation errors
        LiveConfig(symbol="TEST", initial_equity=-100.0)
    with pytest.raises(ValueError):
        LiveConfig(symbol="TEST", initial_equity=100.0, max_risk_per_trade_pct=1.1) # > 1.0
    with pytest.raises(ValueError):
        LiveConfig(symbol="TEST", initial_equity=100.0, max_drawdown_pct_account=-0.1) # < 0


def test_live_position_instantiation() -> None:
    """Test basic instantiation and default values for LivePosition."""
    pos = LivePosition(
        instrument="MES",
        quantity=2.0,
        average_entry_price=4500.50
    )
    assert pos.instrument == "MES"
    assert pos.quantity == 2.0
    assert pos.average_entry_price == 4500.50
    assert pos.unrealized_pnl == 0.0 # Default
    assert pos.realized_pnl_session == 0.0 # Default

    pos_with_pnl = LivePosition(
        instrument="EURUSD",
        quantity=-10000.0,
        average_entry_price=1.0850,
        unrealized_pnl=-50.0,
        realized_pnl_session=120.0
    )
    assert pos_with_pnl.instrument == "EURUSD"
    assert pos_with_pnl.quantity == -10000.0
    assert pos_with_pnl.average_entry_price == 1.0850
    assert pos_with_pnl.unrealized_pnl == -50.0
    assert pos_with_pnl.realized_pnl_session == 120.0


def test_live_pnl_instantiation() -> None:
    """Test basic instantiation for LivePnl."""
    now = datetime.datetime.now(datetime.timezone.utc)
    pnl_report = LivePnl(
        timestamp=now,
        total_equity=105000.0,
        session_realized_pnl=1000.0,
        session_unrealized_pnl=-500.0,
        open_positions_count=2
    )
    assert pnl_report.timestamp == now
    assert pnl_report.total_equity == 105000.0
    assert pnl_report.session_realized_pnl == 1000.0
    assert pnl_report.session_unrealized_pnl == -500.0
    assert pnl_report.open_positions_count == 2

    # Test validation (e.g., open_positions_count >= 0)
    with pytest.raises(ValueError):
        LivePnl(timestamp=now, total_equity=0, session_realized_pnl=0,
                session_unrealized_pnl=0, open_positions_count=-1)
