from __future__ import annotations

import pytest
import asyncio # For async functions
import math

from azr_planner.live.blotter import Blotter, PNL_MULTIPLIERS, DEFAULT_PNL_MULTIPLIER
from azr_planner.live.schemas import LivePosition, LivePnl
from azr_planner.schemas import Instrument, Direction # Core enums

# Synchronous fixture, as Blotter instantiation is synchronous.
@pytest.fixture
def new_blotter() -> Blotter: # Changed from async def
    return Blotter(initial_equity=100_000.0)

@pytest.mark.asyncio
async def test_blotter_initialization() -> None:
    blotter = Blotter(initial_equity=50000.0)
    assert blotter.initial_equity == 50000.0
    assert blotter.current_cash == 50000.0
    assert blotter.session_realized_pnl_total == 0.0
    assert not blotter.positions

    with pytest.raises(ValueError, match="Initial equity must be positive."):
        Blotter(initial_equity=0)
    with pytest.raises(ValueError, match="Initial equity must be positive."):
        Blotter(initial_equity=-100.0)

@pytest.mark.asyncio
async def test_execute_trade_invalid_inputs(new_blotter: Blotter) -> None:
    with pytest.raises(ValueError, match="Trade size must be positive."):
        await new_blotter.execute_trade(Instrument.MES, Direction.LONG, 0, 4500)
    with pytest.raises(ValueError, match="Trade size must be positive."):
        await new_blotter.execute_trade(Instrument.MES, Direction.LONG, -1, 4500)
    with pytest.raises(ValueError, match="Trade price cannot be negative."):
        await new_blotter.execute_trade(Instrument.MES, Direction.LONG, 1, -100)

@pytest.mark.asyncio
async def test_execute_trade_open_long_position(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    price = 4500.0
    size = 2.0
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)

    await new_blotter.execute_trade(instrument, Direction.LONG, size, price)

    assert instrument.value in new_blotter.positions
    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == size
    assert pos.average_entry_price == price
    assert pos.unrealized_pnl == 0.0 # MTM not called yet after this trade
    assert pos.realized_pnl_session == 0.0

    expected_cash_change = - (size * price * pnl_multiplier)
    assert math.isclose(new_blotter.current_cash, new_blotter.initial_equity + expected_cash_change)
    assert new_blotter.session_realized_pnl_total == 0.0

@pytest.mark.asyncio
async def test_execute_trade_open_short_position(new_blotter: Blotter) -> None:
    instrument = Instrument.M2K # Changed from MNQ
    price = 2000.0 # Adjusted price for M2K context
    size = 1.0
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)

    await new_blotter.execute_trade(instrument, Direction.SHORT, size, price)

    assert instrument.value in new_blotter.positions
    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == -size
    assert pos.average_entry_price == price
    assert pos.realized_pnl_session == 0.0

    expected_cash_change = size * price * pnl_multiplier # Cash increases on short sell
    assert math.isclose(new_blotter.current_cash, new_blotter.initial_equity + expected_cash_change)
    assert new_blotter.session_realized_pnl_total == 0.0

@pytest.mark.asyncio
async def test_execute_trade_add_to_long(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)

    # Initial long
    await new_blotter.execute_trade(instrument, Direction.LONG, 2.0, 4500.0)
    cash_after_first_trade = new_blotter.current_cash

    # Add to long
    await new_blotter.execute_trade(instrument, Direction.LONG, 3.0, 4510.0)

    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == 5.0 # 2.0 + 3.0
    expected_avg_price = ((4500.0 * 2.0) + (4510.0 * 3.0)) / 5.0
    assert math.isclose(pos.average_entry_price, expected_avg_price)

    expected_cash_change_second_trade = - (3.0 * 4510.0 * pnl_multiplier)
    assert math.isclose(new_blotter.current_cash, cash_after_first_trade + expected_cash_change_second_trade)
    assert new_blotter.session_realized_pnl_total == 0.0 # No closing trades

@pytest.mark.asyncio
async def test_execute_trade_add_to_short(new_blotter: Blotter) -> None:
    instrument = Instrument.M2K # Changed from MNQ
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)

    await new_blotter.execute_trade(instrument, Direction.SHORT, 1.0, 2000.0) # Adjusted price
    cash_after_first_trade = new_blotter.current_cash

    await new_blotter.execute_trade(instrument, Direction.SHORT, 2.0, 14990.0)

    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == -3.0 # -1.0 - 2.0
    # Initial trade was 1.0 short @ 2000.0. Adding 2.0 short @ 14990.0
    expected_avg_price = ((2000.0 * 1.0) + (14990.0 * 2.0)) / 3.0
    assert math.isclose(pos.average_entry_price, expected_avg_price)

    expected_cash_change_second_trade = (2.0 * 14990.0 * pnl_multiplier)
    assert math.isclose(new_blotter.current_cash, cash_after_first_trade + expected_cash_change_second_trade)
    assert new_blotter.session_realized_pnl_total == 0.0

@pytest.mark.asyncio
async def test_execute_trade_partially_close_long(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    await new_blotter.execute_trade(instrument, Direction.LONG, 5.0, 4500.0) # Avg price 4500

    # Partially close by selling 2 contracts at 4520.0
    closing_price = 4520.0
    closing_size = 2.0
    await new_blotter.execute_trade(instrument, Direction.SHORT, closing_size, closing_price)

    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == 3.0 # 5.0 - 2.0
    assert math.isclose(pos.average_entry_price, 4500.0) # Avg price of remaining long unchanged

    expected_realized_pnl = (closing_price - 4500.0) * closing_size * pnl_multiplier
    assert math.isclose(new_blotter.session_realized_pnl_total, expected_realized_pnl)
    assert math.isclose(pos.realized_pnl_session, expected_realized_pnl)

@pytest.mark.asyncio
async def test_execute_trade_fully_close_long(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    await new_blotter.execute_trade(instrument, Direction.LONG, 2.0, 4500.0)

    # Fully close by selling 2 contracts at 4490.0 (a loss)
    closing_price = 4490.0
    closing_size = 2.0
    await new_blotter.execute_trade(instrument, Direction.SHORT, closing_size, closing_price)

    assert instrument.value not in new_blotter.positions # Position should be removed
    expected_realized_pnl = (closing_price - 4500.0) * closing_size * pnl_multiplier
    assert math.isclose(new_blotter.session_realized_pnl_total, expected_realized_pnl)

@pytest.mark.asyncio
async def test_execute_trade_reverse_long_to_short(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    initial_long_size = 2.0
    initial_entry_price = 4500.0
    await new_blotter.execute_trade(instrument, Direction.LONG, initial_long_size, initial_entry_price)

    # Reverse to short by selling 5 contracts at 4510.0
    # This closes 2 long, opens 3 short.
    reversing_price = 4510.0
    reversing_sell_size = 5.0
    await new_blotter.execute_trade(instrument, Direction.SHORT, reversing_sell_size, reversing_price)

    assert instrument.value in new_blotter.positions
    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == -3.0 # 2 (long) - 5 (sell) = -3 (short)
    assert math.isclose(pos.average_entry_price, reversing_price) # Avg price of new short leg

    expected_realized_pnl_on_close = (reversing_price - initial_entry_price) * initial_long_size * pnl_multiplier
    assert math.isclose(new_blotter.session_realized_pnl_total, expected_realized_pnl_on_close)
    assert math.isclose(pos.realized_pnl_session, expected_realized_pnl_on_close)

@pytest.mark.asyncio
async def test_execute_trade_reverse_short_to_long(new_blotter: Blotter) -> None:
    instrument = Instrument.M2K # Changed from MNQ
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    initial_short_size = 1.0
    initial_entry_price = 2000.0 # Adjusted price
    await new_blotter.execute_trade(instrument, Direction.SHORT, initial_short_size, initial_entry_price)

    reversing_price = 1990.0 # Buy back lower (profit on short) - Adjusted price
    reversing_buy_size = 3.0 # Cover 1 short, open 2 long
    await new_blotter.execute_trade(instrument, Direction.LONG, reversing_buy_size, reversing_price)

    assert instrument.value in new_blotter.positions
    pos = new_blotter.positions[instrument.value]
    assert pos.quantity == 2.0 # -1 (short) + 3 (buy) = 2 (long)
    assert math.isclose(pos.average_entry_price, reversing_price)

    expected_realized_pnl_on_cover = (initial_entry_price - reversing_price) * initial_short_size * pnl_multiplier
    assert math.isclose(new_blotter.session_realized_pnl_total, expected_realized_pnl_on_cover)
    assert math.isclose(pos.realized_pnl_session, expected_realized_pnl_on_cover)

@pytest.mark.asyncio
async def test_mark_to_market_long_position(new_blotter: Blotter) -> None:
    instrument = Instrument.MES
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    await new_blotter.execute_trade(instrument, Direction.LONG, 2.0, 4500.0)

    mtm_price = 4510.0
    await new_blotter.mark_to_market(instrument.value, mtm_price)

    pos = new_blotter.positions[instrument.value]
    expected_unrealized_pnl = (mtm_price - 4500.0) * 2.0 * pnl_multiplier
    assert math.isclose(pos.unrealized_pnl, expected_unrealized_pnl)

@pytest.mark.asyncio
async def test_mark_to_market_short_position(new_blotter: Blotter) -> None:
    instrument = Instrument.M2K # Changed from MNQ
    pnl_multiplier = PNL_MULTIPLIERS.get(instrument.value, DEFAULT_PNL_MULTIPLIER)
    await new_blotter.execute_trade(instrument, Direction.SHORT, 1.0, 2000.0) # Adjusted price

    mtm_price = 2010.0 # Price moved against short - Adjusted price
    await new_blotter.mark_to_market(instrument.value, mtm_price)

    pos = new_blotter.positions[instrument.value]
    expected_unrealized_pnl = (2000.0 - mtm_price) * 1.0 * pnl_multiplier # (entry - current) * size
    assert math.isclose(pos.unrealized_pnl, expected_unrealized_pnl)

@pytest.mark.asyncio
async def test_mark_to_market_no_position(new_blotter: Blotter) -> None:
    await new_blotter.mark_to_market("NONEXISTENT_SYMBOL", 100.0)
    # No error should occur, and positions should remain empty
    assert not new_blotter.positions

@pytest.mark.asyncio
async def test_get_current_positions(new_blotter: Blotter) -> None:
    positions = await new_blotter.get_current_positions()
    assert not positions # Initially empty

    await new_blotter.execute_trade(Instrument.MES, Direction.LONG, 1.0, 4500)
    await new_blotter.execute_trade(Instrument.M2K, Direction.SHORT, 2.0, 2000) # Changed MNQ to M2K, adjusted price

    positions = await new_blotter.get_current_positions()
    assert len(positions) == 2
    symbols = {p.instrument for p in positions}
    assert Instrument.MES.value in symbols
    assert Instrument.M2K.value in symbols # Changed MNQ to M2K

@pytest.mark.asyncio
async def test_get_current_pnl_no_positions(new_blotter: Blotter) -> None:
    pnl_report = await new_blotter.get_current_pnl() # No market prices needed as no positions
    assert pnl_report.total_equity == new_blotter.initial_equity
    assert pnl_report.session_realized_pnl == 0.0
    assert pnl_report.session_unrealized_pnl == 0.0
    assert pnl_report.open_positions_count == 0

@pytest.mark.asyncio
async def test_get_current_pnl_with_positions(new_blotter: Blotter) -> None:
    # Trade 1: MES Long
    await new_blotter.execute_trade(Instrument.MES, Direction.LONG, 2.0, 4500.0) # Cost basis updated
    await new_blotter.mark_to_market(Instrument.MES.value, 4510.0) # MES Unrealized PNL = (4510-4500)*2*5 = 100

    # Trade 2: M2K Short
    await new_blotter.execute_trade(Instrument.M2K, Direction.SHORT, 1.0, 2000.0) # Changed MNQ to M2K, adjusted price
    await new_blotter.mark_to_market(Instrument.M2K.value, 1980.0) # M2K Unrealized PNL = (2000-1980)*1*5 = 100 (M2K mult is 5)

    # Trade 3: Close part of MES for realized PNL
    # Sell 1 MES at 4515. Realized PNL = (4515-4500)*1*5 = 75. Remaining 1 MES long at 4500.
    await new_blotter.execute_trade(Instrument.MES, Direction.SHORT, 1.0, 4515.0)

    # Update MTM for remaining MES position with a new price
    await new_blotter.mark_to_market(Instrument.MES.value, 4520.0) # Rem 1 MES Unrealized PNL = (4520-4500)*1*5 = 100
                                                                # (Original unrealized was 100 for 2 lots, now 100 for 1 lot)
                                                                # This MTM call updates the specific LivePosition object.

    # Get PNL
    pnl_report = await new_blotter.get_current_pnl()

    expected_session_realized_pnl = 75.0
    assert math.isclose(pnl_report.session_realized_pnl, expected_session_realized_pnl)

    # Expected unrealized PNL:
    # MES: 1 lot long from 4500, current MTM price 4520. Unrealized = (4520 - 4500) * 1 * 5 = 100
    # M2K: 1 lot short from 2000, current MTM price 1980. Unrealized = (2000 - 1980) * 1 * 5 = 100
    expected_session_unrealized_pnl = 100.0 + 100.0
    assert math.isclose(pnl_report.session_unrealized_pnl, expected_session_unrealized_pnl)

    assert pnl_report.open_positions_count == 2 # 1 MES, 1 M2K

    # Cash = Initial - CostBasisBought + ProceedsSold.
    # Initial Cash = 100_000
    # 1. Buy 2 MES @ 4500 (cost 2*4500*5 = 45000): Cash = 100_000 - 45000 = 55_000
    # 2. Sell 1 M2K @ 2000 (proceeds 1*2000*5 = 10000): Cash = 55_000 + 10000 = 65_000
    # 3. Sell 1 MES @ 4515 (proceeds 1*4515*5 = 22575): Cash = 65_000 + 22575 = 87_575
    # This cash calculation reflects realized P&L implicitly.
    assert math.isclose(new_blotter.current_cash, 87575.0)
        # The primary definition of total_equity is initial + realized + unrealized.
        # The blotter's internal calculation via current_cash should align if all logic is correct.
        # The second assertion is the definitional one.
    assert math.isclose(pnl_report.total_equity, new_blotter.initial_equity + expected_session_realized_pnl + expected_session_unrealized_pnl)
