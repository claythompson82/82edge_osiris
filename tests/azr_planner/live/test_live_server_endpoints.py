from __future__ import annotations

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import math
import importlib
import os

from fastapi.testclient import TestClient
from fastapi import FastAPI # For type hint if needed, and for app.state
from typing import Iterator, Tuple, cast # Added cast

# Import specific schemas for response validation if needed
from azr_planner.live.schemas import LivePosition, LivePnl

# It's better to import the specific things needed from tasks, blotter, etc.
# rather than importing the whole module if we are asserting against their state.
# We will access these via client.app.state now.
# from azr_planner.live.tasks import live_config as module_live_config # Removed
# from azr_planner.live.tasks import blotter as module_blotter # Removed
# from azr_planner.live.tasks import live_trading_task as module_live_trading_main_task_in_tasks # Keep for fixture check


@pytest.fixture(scope="function")
def test_client_live_trading(
    monkeypatch: pytest.MonkeyPatch
) -> Iterator[Tuple[TestClient, AsyncMock, AsyncMock]]: # Added type hints
    """
    Provides a TestClient for an app where live trading is managed by startup/shutdown.
    Ensures OSIRIS_TEST=1 and reloads server module to apply this.
    Patches live_paper_trading_loop to prevent actual indefinite run.
    """
    monkeypatch.setenv("OSIRIS_TEST", "1")

    # Ensure server module is loaded fresh with OSIRIS_TEST=1
    import osiris.server
    importlib.reload(osiris.server)
    app_under_test = osiris.server.app

    # Mock the core loop to prevent it from actually running, just check if called
    mock_loop_async = AsyncMock(return_value=None)

    # Patch where start_live_trading_task will look for live_paper_trading_loop
    with patch("azr_planner.live.tasks.live_paper_trading_loop", new=mock_loop_async) as mock_loop_patch_obj, \
         patch("azr_planner.live.tasks.stop_live_trading_task", autospec=True) as mock_stop_task_patch_obj:

        # TestClient context manager handles startup and shutdown events on app_under_test
        with TestClient(app_under_test) as client:
            yield client, mock_loop_patch_obj, mock_stop_task_patch_obj

        # After TestClient exits, shutdown events should have fired.
        # Assert stop_live_trading_task was called if the task was considered started.
        # The osiris.server._live_trading_main_task global holds the task.
        if hasattr(osiris.server, '_live_trading_main_task') and osiris.server._live_trading_main_task is not None:
            mock_stop_task_patch_obj.assert_called_once()
        elif mock_loop_patch_obj.called: # If loop was started (or attempted), stop should be called
             mock_stop_task_patch_obj.assert_called_once()


@pytest.mark.asyncio
async def test_live_trading_startup_behavior(
    test_client_live_trading: Tuple[TestClient, AsyncMock, AsyncMock]
) -> None: # Added type hints
    """Test that the live trading loop is initiated on startup."""
    client, mock_loop_patch, _ = test_client_live_trading
    # Allow startup events to complete
    await asyncio.sleep(0.05) # Increased delay slightly
    assert mock_loop_patch.called, "live_paper_trading_loop should have been called via start_live_trading_task on app startup."

@pytest.mark.asyncio
async def test_get_live_positions_empty(
    test_client_live_trading: Tuple[TestClient, AsyncMock, AsyncMock]
) -> None: # Added type hints
    """Test /positions endpoint when no positions exist."""
    client, _, _ = test_client_live_trading
    await asyncio.sleep(0.01) # Ensure startup tasks have a moment

    response = client.get("/azr_api/v1/live/positions")
    assert response.status_code == 200
    assert response.json() == []

@pytest.mark.asyncio
async def test_get_live_pnl_initial(
    test_client_live_trading: Tuple[TestClient, AsyncMock, AsyncMock]
) -> None: # Added type hints
    """Test /pnl endpoint for initial state."""
    client, _, _ = test_client_live_trading
    await asyncio.sleep(0.01)

    response = client.get("/azr_api/v1/live/pnl")
    assert response.status_code == 200
    pnl_data = response.json()

    app_state_config = cast(FastAPI, client.app).state.live_config
    assert app_state_config is not None, "LiveConfig not found in app.state"
    assert pnl_data["total_equity"] == app_state_config.initial_equity
    assert pnl_data["session_realized_pnl"] == 0.0
    assert pnl_data["session_unrealized_pnl"] == 0.0
    assert pnl_data["open_positions_count"] == 0
    assert "timestamp" in pnl_data

@pytest.mark.asyncio
async def test_endpoints_with_active_position(
    test_client_live_trading: Tuple[TestClient, AsyncMock, AsyncMock]
) -> None: # Added type hints
    """Test /positions and /pnl endpoints when there's an active position."""
    client, _, _ = test_client_live_trading
    await asyncio.sleep(0.01)

    actual_blotter = cast(FastAPI, client.app).state.live_blotter
    actual_config = cast(FastAPI, client.app).state.live_config
    assert actual_blotter is not None, "Blotter should be initialized by startup task and found in app.state"
    assert actual_config is not None, "LiveConfig should be initialized by startup task and found in app.state"

    from azr_planner.schemas import Instrument, Direction # Core enums

    # Manually execute a trade and MTM via the blotter instance
    trade_instrument = Instrument.MES
    entry_price = 4600.0
    mtm_price = 4605.0
    trade_size = 2.0
    pnl_multiplier = actual_blotter._get_pnl_multiplier(trade_instrument.value)

    await actual_blotter.execute_trade(trade_instrument, Direction.LONG, trade_size, entry_price)
    await actual_blotter.mark_to_market(trade_instrument.value, mtm_price)

    # Test /positions
    response_pos = client.get("/azr_api/v1/live/positions")
    assert response_pos.status_code == 200
    positions = response_pos.json()
    assert len(positions) == 1
    pos1 = positions[0]
    assert pos1["instrument"] == trade_instrument.value
    assert pos1["quantity"] == trade_size
    assert math.isclose(pos1["average_entry_price"], entry_price)
    expected_unrealized_pnl = (mtm_price - entry_price) * trade_size * pnl_multiplier
    assert math.isclose(pos1["unrealized_pnl"], expected_unrealized_pnl)
    assert math.isclose(pos1["realized_pnl_session"], 0.0)

    # Test /pnl
    response_pnl = client.get("/azr_api/v1/live/pnl")
    assert response_pnl.status_code == 200
    pnl_data = response_pnl.json()

    assert math.isclose(pnl_data["session_realized_pnl"], 0.0)
    assert math.isclose(pnl_data["session_unrealized_pnl"], expected_unrealized_pnl)
    assert pnl_data["open_positions_count"] == 1
    expected_total_equity = actual_config.initial_equity + expected_unrealized_pnl # Since realized is 0
    assert math.isclose(pnl_data["total_equity"], expected_total_equity)

@pytest.mark.asyncio
async def test_endpoints_after_realized_pnl(
    test_client_live_trading: Tuple[TestClient, AsyncMock, AsyncMock]
) -> None: # Added type hints
    """Test /positions and /pnl after some P&L has been realized."""
    client, _, _ = test_client_live_trading
    await asyncio.sleep(0.01)

    actual_blotter = cast(FastAPI, client.app).state.live_blotter
    actual_config = cast(FastAPI, client.app).state.live_config
    assert actual_blotter is not None, "Blotter not found in app.state"
    assert actual_config is not None, "LiveConfig not found in app.state"

    from azr_planner.schemas import Instrument, Direction
    pnl_multiplier_mes = actual_blotter._get_pnl_multiplier(Instrument.MES.value)

    # Sequence: Buy 1 MES @ 4700, MTM @ 4702. Sell 0.5 MES @ 4710. MTM remaining 0.5 MES @ 4712.
    await actual_blotter.execute_trade(Instrument.MES, Direction.LONG, 1.0, 4700.0)
    await actual_blotter.mark_to_market(Instrument.MES.value, 4702.0) # Unrealized: (4702-4700)*1*mult = 2*mult

    await actual_blotter.execute_trade(Instrument.MES, Direction.SHORT, 0.5, 4710.0)
    # Realized PNL: (4710-4700)*0.5*mult = 10*0.5*mult = 5*mult
    realized_pnl_expected = (4710.0 - 4700.0) * 0.5 * pnl_multiplier_mes

    await actual_blotter.mark_to_market(Instrument.MES.value, 4712.0)
    # Remaining pos: 0.5 long at 4700. MTM price 4712.
    # Unrealized: (4712-4700)*0.5*mult = 12*0.5*mult = 6*mult
    unrealized_pnl_expected = (4712.0 - 4700.0) * 0.5 * pnl_multiplier_mes

    # Test /pnl
    response_pnl = client.get("/azr_api/v1/live/pnl")
    assert response_pnl.status_code == 200
    pnl_data = response_pnl.json()

    assert math.isclose(pnl_data["session_realized_pnl"], realized_pnl_expected)
    assert math.isclose(pnl_data["session_unrealized_pnl"], unrealized_pnl_expected)
    assert pnl_data["open_positions_count"] == 1 # Remaining 0.5 MES

    expected_total_equity = actual_config.initial_equity + realized_pnl_expected + unrealized_pnl_expected
    assert math.isclose(pnl_data["total_equity"], expected_total_equity)

    # Test /positions
    response_pos = client.get("/azr_api/v1/live/positions")
    assert response_pos.status_code == 200
    positions = response_pos.json()
    assert len(positions) == 1
    pos1 = positions[0]
    assert pos1["instrument"] == Instrument.MES.value
    assert math.isclose(pos1["quantity"], 0.5) # Remaining quantity
    assert math.isclose(pos1["average_entry_price"], 4700.0) # Avg price of original long
    assert math.isclose(pos1["unrealized_pnl"], unrealized_pnl_expected)
    assert math.isclose(pos1["realized_pnl_session"], realized_pnl_expected) # Realized PNL for this instrument's trades this session
