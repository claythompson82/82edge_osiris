from __future__ import annotations

import asyncio
import datetime
import math
from typing import Dict, List, Optional

from azr_planner.schemas import Instrument, Direction
from .schemas import LivePosition, LivePnl

PNL_MULTIPLIERS: Dict[str, float] = {
    Instrument.MES.value: 5.0,
    Instrument.M2K.value: 5.0,
    Instrument.US_SECTOR_ETF.value: 1.0,
    Instrument.ETH_OPT.value: 1.0,
}
DEFAULT_PNL_MULTIPLIER = 1.0

class Blotter:
    def __init__(self, initial_equity: float):
        if initial_equity <= 0:
            raise ValueError("Initial equity must be positive.")

        self.initial_equity: float = initial_equity
        self.current_cash: float = initial_equity
        self.session_realized_pnl_total: float = 0.0
        self.positions: Dict[str, LivePosition] = {} # Keyed by instrument.value (string)
        self.lock = asyncio.Lock()

    def _get_pnl_multiplier(self, instrument_symbol: str) -> float:
        return PNL_MULTIPLIERS.get(instrument_symbol, DEFAULT_PNL_MULTIPLIER)

    async def execute_trade(
        self, instrument: Instrument, direction: Direction, size: float, price: float
    ) -> None:
        if size <= 0:
            raise ValueError("Trade size must be positive.")
        if price < 0: # Price can be 0 for some instruments, but not negative.
            raise ValueError("Trade price cannot be negative.")

        instrument_symbol = instrument.value
        pnl_multiplier = self._get_pnl_multiplier(instrument_symbol)

        # qty_traded represents the change in position: positive for buy/long, negative for sell/short
        qty_traded_signed = size if direction == Direction.LONG else -size

        async with self.lock:
            current_pos_obj = self.positions.get(instrument_symbol)

            current_qty_on_book = current_pos_obj.quantity if current_pos_obj else 0.0
            current_avg_entry_price = current_pos_obj.average_entry_price if current_pos_obj else 0.0
            instrument_session_realized_pnl = current_pos_obj.realized_pnl_session if current_pos_obj else 0.0

            final_qty_on_book = current_qty_on_book + qty_traded_signed

            # Determine final average price and realized P&L
            final_avg_price = price # Default for new or flipped positions

            if current_pos_obj: # Existing position being modified
                if (current_qty_on_book * qty_traded_signed < 0): # Opposite trade (closing/reducing opposite or flipping)
                    qty_closed = min(abs(current_qty_on_book), abs(qty_traded_signed))
                    pnl_per_unit = 0.0
                    if current_qty_on_book > 0: # Closing a long
                        pnl_per_unit = price - current_avg_entry_price
                    else: # Closing a short (current_qty_on_book < 0)
                        pnl_per_unit = current_avg_entry_price - price

                    realized_pnl_this_trade = pnl_per_unit * qty_closed * pnl_multiplier
                    instrument_session_realized_pnl += realized_pnl_this_trade
                    self.session_realized_pnl_total += realized_pnl_this_trade

                    if not math.isclose(final_qty_on_book, 0.0) and (final_qty_on_book * current_qty_on_book < 0): # Flipped
                        final_avg_price = price # New leg starts at current price
                    elif not math.isclose(final_qty_on_book, 0.0): # Partially closed, side not flipped
                        final_avg_price = current_avg_entry_price # Avg price remains
                    # If fully closed (final_qty_on_book is 0), final_avg_price is irrelevant for position but set to price

                elif not math.isclose(qty_traded_signed, 0.0): # Same direction trade (adding to position)
                    if not math.isclose(final_qty_on_book, 0.0):
                         final_avg_price = ((current_avg_entry_price * abs(current_qty_on_book)) + \
                                           (price * abs(qty_traded_signed))) / abs(final_qty_on_book)
                    else: # Should not happen if adding to existing unless qty_traded_signed was 0
                        final_avg_price = price # pragma: no cover
            # else (current_pos_obj is None): opening new position, final_avg_price is already `price`

            # Update cash: Net cash effect of the trade
            # Buying reduces cash, selling increases cash
            trade_value_effect = abs(qty_traded_signed) * price * pnl_multiplier
            if qty_traded_signed > 0: # Net buy
                self.current_cash -= trade_value_effect
            else: # Net sell
                self.current_cash += trade_value_effect

            # Update positions dictionary
            if math.isclose(final_qty_on_book, 0.0):
                if instrument_symbol in self.positions:
                    del self.positions[instrument_symbol]
            else:
                self.positions[instrument_symbol] = LivePosition(
                    instrument=instrument_symbol,
                    quantity=final_qty_on_book,
                    average_entry_price=final_avg_price,
                    unrealized_pnl=0.0, # MTM will update this; if flipped, old unrealized is gone.
                    realized_pnl_session=instrument_session_realized_pnl
                )

    async def mark_to_market(self, instrument_symbol: str, current_price: float) -> None:
        if current_price < 0:
            return

        pnl_multiplier = self._get_pnl_multiplier(instrument_symbol)
        async with self.lock:
            if instrument_symbol in self.positions:
                pos = self.positions[instrument_symbol]
                unrealized_pnl = 0.0
                if pos.quantity > 0: # Long
                    unrealized_pnl = (current_price - pos.average_entry_price) * pos.quantity * pnl_multiplier
                elif pos.quantity < 0: # Short
                    unrealized_pnl = (pos.average_entry_price - current_price) * abs(pos.quantity) * pnl_multiplier

                # Update LivePosition with new unrealized P&L
                self.positions[instrument_symbol] = LivePosition(
                    instrument=pos.instrument,
                    quantity=pos.quantity,
                    average_entry_price=pos.average_entry_price,
                    unrealized_pnl=unrealized_pnl, # This is the update
                    realized_pnl_session=pos.realized_pnl_session
                )

    async def get_current_positions(self) -> List[LivePosition]:
        async with self.lock:
            return list(self.positions.values())

    async def get_current_pnl(self) -> LivePnl:
        async with self.lock:
            session_unrealized_pnl_total = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_equity = self.initial_equity + self.session_realized_pnl_total + session_unrealized_pnl_total

            return LivePnl(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                total_equity=total_equity,
                session_realized_pnl=self.session_realized_pnl_total,
                session_unrealized_pnl=session_unrealized_pnl_total,
                open_positions_count=len(self.positions)
            )
