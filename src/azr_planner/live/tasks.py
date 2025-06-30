from __future__ import annotations

import asyncio
import datetime
import logging
from collections import deque
from typing import Optional, Deque, List, Any # Added List, Any

from fastapi import FastAPI # For type hinting app in start/stop functions

# AZR Planner core components
from azr_planner.engine import generate_plan as default_planner_fn
from azr_planner.risk_gate import accept as default_risk_gate_fn
from azr_planner.risk_gate.schemas import RiskGateConfig
from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg
from azr_planner.math_utils import LR_V2_MIN_POINTS # Lookback for planning context

# Live trading components
from .blotter import Blotter
from .stream import AbstractBarStream, MockWebSocketBarStream
from .schemas import LiveConfig
from .metrics import AZR_LIVE_TRADES_TOTAL, AZR_LIVE_OPEN_RISK

# Bar type from replay module for consistency
from azr_planner.replay.schemas import Bar


logger = logging.getLogger(__name__)

# Module-level state for the live trading task
# These will be initialized in start_live_trading_task
live_config: Optional[LiveConfig] = None
blotter: Optional[Blotter] = None
bar_stream_client: Optional[AbstractBarStream] = None
live_trading_task: Optional[asyncio.Task[None]] = None # Added type parameter
shutdown_event: asyncio.Event = asyncio.Event()


async def live_paper_trading_loop() -> None: # Added return type
    """
    Main loop for live paper trading.
    Streams bars, generates plans, checks risk, and executes trades into the blotter.
    """
    global blotter, bar_stream_client, live_config, shutdown_event

    if not blotter or not bar_stream_client or not live_config:
        logger.error("Live trading loop started without proper initialization.")
        return

    logger.info(f"Live paper trading loop starting for symbol: {live_config.symbol} with equity: {live_config.initial_equity}")

    # Context for planning
    bar_window: Deque[Bar] = deque(maxlen=LR_V2_MIN_POINTS)

    # Risk gate configuration. For now, use default RiskGateConfig.
    # LiveConfig parameters like max_risk_per_trade_pct and max_drawdown_pct_account
    # are not directly used by the current default_risk_gate_fn's RiskGateConfig schema.
    # This would require enhancing RiskGateConfig and the risk_gate.accept logic.
    risk_config = RiskGateConfig()

    instrument_enum = Instrument(live_config.symbol.upper()) # Assuming symbol matches Instrument enum value

    try:
        async for current_bar in bar_stream_client.stream():
            if shutdown_event.is_set():
                logger.info("Shutdown event received, exiting live trading loop.")
                break

            if current_bar.instrument.upper() != live_config.symbol.upper():
                logger.warning(f"Received bar for unexpected instrument {current_bar.instrument}, expecting {live_config.symbol}. Skipping.")
                continue

            # 1. Mark to market
            await blotter.mark_to_market(instrument_symbol=instrument_enum.value, current_price=current_bar.close)

            # 2. Update planning context window
            bar_window.append(current_bar)

            if len(bar_window) < LR_V2_MIN_POINTS:
                continue # Not enough data for planning yet

            # 3. Prepare PlanningContext
            # Similar to replay runner, adapt as needed
            planner_equity_curve = [b.close for b in bar_window] # Simplified equity curve for planner
            planner_hlc = [(b.high, b.low, b.close) for b in bar_window]

            temp_planner_volume: List[float] = []
            all_volumes_present = True
            if any(b.volume is not None for b in bar_window):
                for b_in_window in bar_window:
                    if b_in_window.volume is None:
                        all_volumes_present = False; break
                    temp_planner_volume.append(b_in_window.volume)
                planner_volume: Optional[List[float]] = temp_planner_volume if all_volumes_present else None
            else:
                planner_volume = None

            current_positions_list: List[Leg] = []
            # Fetch current positions from blotter to pass to planner
            # This part needs care: blotter positions are LivePosition, planner needs List[Leg]
            blotter_positions = await blotter.get_current_positions()
            for live_pos in blotter_positions:
                if live_pos.instrument.upper() == instrument_enum.value: # Filter for the traded symbol
                    direction = Direction.LONG if live_pos.quantity > 0 else Direction.SHORT
                    current_positions_list.append(Leg(instrument=instrument_enum, direction=direction, size=abs(live_pos.quantity)))

            # Simplified vol surface and risk-free rate for now
            vol_surface_data = {instrument_enum.value: 0.20}
            risk_free_rate_val = 0.02

            planning_ctx = PlanningContext(
                timestamp=current_bar.timestamp,
                equity_curve=planner_equity_curve,
                daily_history_hlc=planner_hlc,
                daily_volume=planner_volume,
                current_positions=current_positions_list if current_positions_list else None,
                vol_surface=vol_surface_data,
                risk_free_rate=risk_free_rate_val,
                nSuccesses=50, nFailures=10 # Example values
            )

            # 4. Generate Plan
            proposal: TradeProposal = default_planner_fn(planning_ctx)

            # 5. Risk Gate
            # Note: default_risk_gate_fn may require a DB table and registry, which are None here.
            # This might need adjustment or a live-specific risk gate variant.
            accepted, reason = default_risk_gate_fn(proposal, cfg=risk_config, db_table=None, registry=None)

            # 6. Execute Trade if accepted and not HOLD
            if accepted and proposal.action != "HOLD" and proposal.legs:
                for leg in proposal.legs:
                    if leg.instrument == instrument_enum: # Ensure trading only the configured symbol
                        await blotter.execute_trade(
                            instrument=leg.instrument,
                            direction=leg.direction,
                            size=leg.size,
                            price=current_bar.close # Execute at current bar's close
                        )
                        logger.info(f"Executed trade: {leg.direction.value} {leg.size} {leg.instrument.value} @ {current_bar.close}")

                        # Prometheus metrics
                        # proposal.action is already a str (e.g. "ENTER", "EXIT", "HOLD")
                        action_label = f"{proposal.action}_{leg.direction.value}"
                        AZR_LIVE_TRADES_TOTAL.labels(instrument=leg.instrument.value, action=action_label).inc() # Use .value for enum's string value
            elif not accepted:
                logger.info(f"Trade proposal rejected by risk gate: {reason}. Proposal: {proposal.action}")

            # Update open risk gauge (simplified: total value of position)
            # This needs to fetch the latest position quantity after potential trade
            instrument_symbol_str = instrument_enum.value # This is already the string value, e.g., "MES"
            final_pos_for_bar = blotter.positions.get(instrument_symbol_str)
            if final_pos_for_bar:
                open_value = abs(final_pos_for_bar.quantity * current_bar.close * blotter._get_pnl_multiplier(instrument_symbol_str))
                AZR_LIVE_OPEN_RISK.labels(instrument=instrument_symbol_str).set(open_value)
            else:
                AZR_LIVE_OPEN_RISK.labels(instrument=instrument_symbol_str).set(0)

    except asyncio.CancelledError:
        logger.info("Live trading loop was cancelled.")
    except Exception as e:
        logger.error(f"Error in live paper trading loop: {e}", exc_info=True)
    finally:
        logger.info("Live paper trading loop finished.")
        # Clean up or final logging if needed


async def start_live_trading_task(app: FastAPI, config: LiveConfig) -> None: # Added return type
    """Initializes and starts the live paper trading background task."""
    global live_config, blotter, bar_stream_client, live_trading_task, shutdown_event

    if live_trading_task and not live_trading_task.done():
        logger.warning("Live trading task already running.")
        return

    live_config = config
    blotter = Blotter(initial_equity=config.initial_equity)
    # For now, MockWebSocketBarStream is used. Real implementation would use a live feed.
    bar_stream_client = MockWebSocketBarStream(symbol=config.symbol)
    shutdown_event.clear() # Clear event for this new session

    logger.info(f"Starting live paper trading task for {config.symbol}.")
    live_trading_task = asyncio.create_task(live_paper_trading_loop())

    # Store instances on app state if needed by other parts of the app directly
    app.state.live_blotter = blotter
    app.state.live_config = config
    app.state.live_trading_task = live_trading_task


async def stop_live_trading_task() -> None: # Added return type
    """Stops the live paper trading background task."""
    global live_trading_task, shutdown_event

    if live_trading_task and not live_trading_task.done():
        logger.info("Stopping live paper trading task...")
        shutdown_event.set() # Signal the loop to stop
        try:
            await asyncio.wait_for(live_trading_task, timeout=10.0) # Wait for graceful exit
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for live trading loop to stop, cancelling task.")
            live_trading_task.cancel()
            try:
                await live_trading_task # Await cancellation
            except asyncio.CancelledError:
                logger.info("Live trading task cancelled successfully.")
        except Exception as e: # Catch any other exceptions during task shutdown
             logger.error(f"Exception while stopping live trading task: {e}", exc_info=True)

        live_trading_task = None
        logger.info("Live paper trading task stopped.")
    else:
        logger.info("Live paper trading task not running or already stopped.")

    # Clear globals
    # live_config = None # Keep config for potential restart? Or clear?
    # blotter = None
    # bar_stream_client = None
    # shutdown_event.clear() # Ready for next start, or handled in start_task? Already cleared in start.
