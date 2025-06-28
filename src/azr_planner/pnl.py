from __future__ import annotations

import datetime
import math
import json # For handling legs_json if needed by a persistence schema, though pnl.py itself might not persist directly
from typing import List, Dict, Tuple, Optional, Any

from prometheus_client import Counter

from azr_planner.schemas import Instrument, Direction, DailyFill, DailyPNLReport, Leg # Leg for position structure if needed
from azr_planner.backtest.metrics import calculate_max_drawdown # Corrected import path
# Assuming fixed contract values for exposure calculation as per AZR-12 risk gate, or use actual prices.
# For P&L, actual prices are used. For exposure, it's value-based.
# Let's define exposure constants here if needed, or get them from a shared place.
# For now, using fixed multipliers for MES/M2K for exposure as a simplification, similar to risk_gate.
# This might need refinement if exposure should be purely price * qty.
# The prompt says "size x contract_value" for exposure, implying fixed values for contract_value.
MES_CONTRACT_VALUE_FOR_EXPOSURE = 50.0
M2K_CONTRACT_VALUE_FOR_EXPOSURE = 100.0
DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE_FOR_EXPOSURE = 0.0 # Or price * qty if value is not fixed


# --- Prometheus Counter ---
PNL_REPORTS_TOTAL = Counter(
    'azr_pnl_reports_total',
    'Total number of Daily PNL Reports persisted'
)

# --- Position Ledger Structure ---
# Dict[Instrument, {'qty': float, 'avg_entry_price': float, 'realized_pnl_today': float}]
# 'realized_pnl_today' can be accumulated here per instrument as fills are processed.
PositionDict = Dict[str, float] # e.g. {'qty': 10.0, 'avg_entry_price': 100.0}
PortfolioLedger = Dict[Instrument, PositionDict]


# --- Helper Functions ---

def _get_instrument_value_for_exposure(instrument: Instrument) -> float:
    """Returns the fixed contract value for known instruments for exposure calculation."""
    if instrument == Instrument.MES:
        return MES_CONTRACT_VALUE_FOR_EXPOSURE
    elif instrument == Instrument.M2K:
        return M2K_CONTRACT_VALUE_FOR_EXPOSURE
    return DEFAULT_OTHER_INSTRUMENT_CONTRACT_VALUE_FOR_EXPOSURE


def _update_positions_and_calc_realized_pnl(
    current_positions: PortfolioLedger,
    current_cash: float,
    fills_for_day: List[DailyFill]
) -> Tuple[PortfolioLedger, float, float]:
    """
    Updates positions based on daily fills and calculates realized P&L for the day.
    Returns the new positions ledger, updated cash, and total realized P&L for the day.
    """
    updated_positions = {inst: pos.copy() for inst, pos in current_positions.items()} # Deep copy for modification
    todays_realized_pnl = 0.0
    updated_cash = current_cash

    for fill in fills_for_day:
        instrument = fill.instrument
        fill_qty = fill.qty
        fill_price = fill.price

        # Update cash
        if fill.direction == Direction.LONG: # BUY
            updated_cash -= fill_qty * fill_price # Cash out
        elif fill.direction == Direction.SHORT: # SELL
            updated_cash += fill_qty * fill_price # Cash in

        pos = updated_positions.get(instrument)

        if pos is None: # New position
            pos = {'qty': 0.0, 'avg_entry_price': 0.0}
            if fill.direction == Direction.LONG: # BUY
                pos['qty'] = fill_qty
                pos['avg_entry_price'] = fill_price
            elif fill.direction == Direction.SHORT: # SELL (initiating a short)
                pos['qty'] = -fill_qty
                pos['avg_entry_price'] = fill_price
            updated_positions[instrument] = pos
        else: # Existing position
            current_qty = pos['qty']
            avg_price = pos['avg_entry_price']

            realized_pnl_from_this_fill = 0.0

            if fill.direction == Direction.LONG: # BUY
                if current_qty < 0: # Covering a short position
                    qty_to_cover = min(fill_qty, abs(current_qty))
                    realized_pnl_from_this_fill = (avg_price - fill_price) * qty_to_cover # PNL per unit for short
                    pos['qty'] += qty_to_cover
                    fill_qty -= qty_to_cover # Remaining fill qty, if any

                if fill_qty > 0: # Opening new long or adding to existing long
                    if pos['qty'] > 0: # Adding to existing long
                        pos['avg_entry_price'] = ((avg_price * pos['qty']) + (fill_price * fill_qty)) / (pos['qty'] + fill_qty)
                    else: # New long portion after cover, or initial new long
                        pos['avg_entry_price'] = fill_price
                    pos['qty'] += fill_qty

            elif fill.direction == Direction.SHORT: # SELL
                if current_qty > 0: # Closing a long position
                    qty_to_sell = min(fill_qty, current_qty)
                    realized_pnl_from_this_fill = (fill_price - avg_price) * qty_to_sell # PNL per unit for long
                    pos['qty'] -= qty_to_sell
                    fill_qty -= qty_to_sell

                if fill_qty > 0: # Opening new short or adding to existing short
                    if pos['qty'] < 0: # Adding to existing short
                        pos['avg_entry_price'] = ((avg_price * abs(pos['qty'])) + (fill_price * fill_qty)) / (abs(pos['qty']) + fill_qty)
                    else: # New short portion after close, or initial new short
                        pos['avg_entry_price'] = fill_price
                    pos['qty'] -= fill_qty

            todays_realized_pnl += realized_pnl_from_this_fill

            if math.isclose(pos['qty'], 0.0):
                del updated_positions[instrument]
            else:
                updated_positions[instrument] = pos

    return updated_positions, updated_cash, todays_realized_pnl


def compute_and_record_eod_pnl(
    report_date: datetime.date,
    prior_eod_positions: PortfolioLedger,
    prior_eod_cash: float,
    prior_eod_total_equity: float,
    prior_eod_cumulative_max_equity: float,
    prior_eod_equity_curve_points: List[float],
    fills_for_day: List[DailyFill],
    eod_market_prices: Dict[Instrument, float],
    pnl_db_table: Optional[Any] # Opaque type for LanceDB table handle
) -> DailyPNLReport:
    """
    Computes daily P&L, updates portfolio state, and creates a DailyPNLReport.
    Optionally persists the report to a LanceDB table and increments Prometheus counter.
    """

    # 1. Update positions based on today's fills and calculate realized P&L for the day
    current_day_eod_positions, current_cash, realized_pnl_today = \
        _update_positions_and_calc_realized_pnl(prior_eod_positions, prior_eod_cash, fills_for_day)

    # 2. Calculate Unrealized P&L and Net Position Value for EOD positions
    unrealized_pnl_eod = 0.0
    net_position_value_eod = 0.0
    gross_exposure_eod = 0.0
    net_exposure_eod = 0.0

    for instrument, pos_details in current_day_eod_positions.items():
        qty = pos_details['qty']
        avg_entry_price = pos_details['avg_entry_price']
        eod_price = eod_market_prices.get(instrument)

        if eod_price is None:
            # Handle missing EOD price: Cannot calculate UPL or value.
            # Option: Use last known price, or log error and skip. For now, skip.
            # This implies this position won't contribute to net_position_value or UPL.
            # Consider if this position should still contribute to exposure based on avg_entry_price.
            # For simplicity, if no EOD price, its value and UPL are 0 for this calculation.
            continue

        instrument_value_at_eod = qty * eod_price # This is signed value
        net_position_value_eod += instrument_value_at_eod

        # Unrealized P&L calculation
        if qty > 0: # Long position
            unrealized_pnl_eod += (eod_price - avg_entry_price) * qty
        elif qty < 0: # Short position
            unrealized_pnl_eod += (avg_entry_price - eod_price) * abs(qty)

        # Exposure calculation using fixed contract values for simplicity as per prompt context
        # This might differ from true market value used for net_position_value if multipliers are involved
        # For now, using a simplified exposure based on the problem description for risk gate (size * fixed_contract_value)
        # A more accurate exposure would be abs(qty * eod_price * actual_multiplier_if_any)
        instrument_exposure_value = abs(qty * _get_instrument_value_for_exposure(instrument))
        gross_exposure_eod += instrument_exposure_value
        net_exposure_eod += qty * _get_instrument_value_for_exposure(instrument) # Signed

    # 3. Calculate Total Equity at EOD
    total_equity_eod = current_cash + net_position_value_eod

    # 4. Update Equity Curve
    updated_equity_curve_points = prior_eod_equity_curve_points + [total_equity_eod]

    # 5. Calculate Drawdown
    new_cumulative_max_equity = max(prior_eod_cumulative_max_equity, total_equity_eod)
    current_drawdown = 0.0
    if new_cumulative_max_equity > 0: # Avoid division by zero; drawdown is 0 if peak is 0 or negative
        drawdown_val = (new_cumulative_max_equity - total_equity_eod) / new_cumulative_max_equity
        current_drawdown = max(0.0, drawdown_val) # Drawdown cannot be negative

    # 6. Clamping values
    clamp_limit = 1e9
    realized_pnl_today = max(-clamp_limit, min(realized_pnl_today, clamp_limit))
    unrealized_pnl_eod = max(-clamp_limit, min(unrealized_pnl_eod, clamp_limit))
    net_position_value_eod = max(-clamp_limit, min(net_position_value_eod, clamp_limit))
    current_cash = max(-clamp_limit, min(current_cash, clamp_limit)) # Cash can be very negative
    total_equity_eod = max(-clamp_limit, min(total_equity_eod, clamp_limit)) # Equity can be very negative
    gross_exposure_eod = max(0.0, min(gross_exposure_eod, clamp_limit)) # Gross exposure non-negative
    net_exposure_eod = max(-clamp_limit, min(net_exposure_eod, clamp_limit))
    # cumulative_max_equity and current_drawdown are handled by their logic.

    # 7. Create Report
    report = DailyPNLReport(
        date=report_date,
        realized_pnl=realized_pnl_today,
        unrealized_pnl=unrealized_pnl_eod,
        net_position_value=net_position_value_eod,
        cash=current_cash,
        total_equity=total_equity_eod,
        gross_exposure=gross_exposure_eod,
        net_exposure=net_exposure_eod,
        cumulative_max_equity=new_cumulative_max_equity,
        current_drawdown=current_drawdown,
        equity_curve_points=updated_equity_curve_points
    )

    # 8. Persistence and Metrics
    if pnl_db_table is not None:
        try:
            # Assuming pnl_db_table is a LanceDB table object
            pnl_db_table.add([report.model_dump()])
            PNL_REPORTS_TOTAL.inc()
        except Exception as e:
            # Log error appropriately in a real system
            print(f"Error persisting DailyPNLReport to LanceDB: {e}")
            # Decide if this should raise or just log. For now, just log.
            pass

    return report
