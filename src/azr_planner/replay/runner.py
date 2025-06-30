from __future__ import annotations

import time
import datetime
import math
import uuid
import json
from typing import Iterable, List, Optional, Tuple, Callable, Dict, Any
from collections import deque

from prometheus_client import Counter, CollectorRegistry

from azr_planner.schemas import PlanningContext, TradeProposal, Instrument, Direction, Leg
from azr_planner.engine import generate_plan as default_planner_fn
from azr_planner.risk_gate import accept as default_risk_gate_fn, RiskGateConfig
from azr_planner.backtest.metrics import calculate_max_drawdown
from azr_planner.math_utils import LR_V2_MIN_POINTS

from .schemas import Bar, ReplayTrade, ReplayReport

REPLAY_RUNS_TOTAL = Counter(
    'azr_replay_runs_total',
    'Total number of replay runs completed',
    ['instrument_group', 'granularity']
)

Portfolio = Dict[Instrument, Dict[str, float]]

def run_replay(
    bar_stream: Iterable[Bar],
    initial_equity: float,
    planner_fn: Callable[[PlanningContext], TradeProposal],
    # The actual default_risk_gate_fn takes keyword args: db_table, cfg, registry
    # So, the Callable should reflect this if we want strictness, or use Callable[..., Out]
    # For now, using a more generic Callable and relying on runtime compatibility + type:ignore at call site.
    risk_gate_fn: Callable[..., Tuple[bool, Optional[str]]],
    risk_gate_config: Optional[RiskGateConfig] = None,
    instrument_group_label: str = "unknown_instrument_group",
    granularity_label: str = "unknown_granularity"
) -> Tuple[ReplayReport, List[ReplayTrade]]:

    REPLAY_RUNS_TOTAL.labels(instrument_group=instrument_group_label, granularity=granularity_label).inc()
    replay_start_wall_time = datetime.datetime.now(datetime.timezone.utc)

    current_positions: Portfolio = {}
    current_cash = initial_equity
    current_total_equity = initial_equity
    equity_curve_points: List[Tuple[datetime.datetime, float]] = []

    replay_trades: List[ReplayTrade] = []
    total_bars_processed = 0; proposals_generated = 0; proposals_accepted = 0; proposals_rejected = 0
    decision_times_ms: List[float] = []
    context_lookback_size = LR_V2_MIN_POINTS
    bar_window: deque[Bar] = deque(maxlen=context_lookback_size)
    first_bar_timestamp: Optional[datetime.datetime] = None
    last_bar_timestamp: Optional[datetime.datetime] = None
    active_risk_gate_config = risk_gate_config if risk_gate_config is not None else RiskGateConfig()
    PNL_MULTIPLIER_SIMPLIFIED = 1.0

    for current_bar in bar_stream:
        total_bars_processed += 1
        if first_bar_timestamp is None:
            first_bar_timestamp = current_bar.timestamp
            equity_curve_points.append((current_bar.timestamp, initial_equity))

        last_bar_timestamp = current_bar.timestamp
        bar_window.append(current_bar)

        current_mtm_value_of_open_positions = 0.0
        for instr_held_enum, pos_details in current_positions.items():
            price_for_mtm = pos_details['avg_price']
            if current_bar.instrument == str(instr_held_enum.value):
                 price_for_mtm = current_bar.close
            current_mtm_value_of_open_positions += pos_details['qty'] * price_for_mtm * PNL_MULTIPLIER_SIMPLIFIED

        equity_at_decision_point = current_cash + current_mtm_value_of_open_positions

        if len(bar_window) < context_lookback_size:
            if total_bars_processed > 0:
                if not equity_curve_points or equity_curve_points[-1][0] != current_bar.timestamp:
                     equity_curve_points.append((current_bar.timestamp, equity_at_decision_point))
                else:
                    equity_curve_points[-1] = (current_bar.timestamp, equity_at_decision_point)
            continue

        planner_equity_curve = [b.close for b in bar_window]
        planner_hlc = [(b.high, b.low, b.close) for b in bar_window]

        temp_planner_volume: List[float] = []
        all_volumes_present = True
        # Check if any bar in window has volume to begin with
        if bar_window and any(b.volume is not None for b in bar_window):
            for b_in_window in bar_window:
                if b_in_window.volume is None: # If any specific bar is missing volume, then all_volumes_present is false
                    all_volumes_present = False; break
                temp_planner_volume.append(b_in_window.volume)
            planner_volume: Optional[List[float]] = temp_planner_volume if all_volumes_present else None
        else:
            planner_volume = None # No bar had volume, or bar_window empty (though caught by lookback check)

        context_current_positions_list: List[Leg] = []
        for instr_enum, pos_data in current_positions.items():
            qty = pos_data['qty']
            if not math.isclose(qty, 0.0):
                direction = Direction.LONG if qty > 0 else Direction.SHORT
                context_current_positions_list.append(Leg(instrument=instr_enum, direction=direction, size=abs(qty)))

        vol_surface_data = {str(inst.value): 0.2 for inst in Instrument}
        vol_surface_data[current_bar.instrument] = 0.2

        planning_ctx = PlanningContext(
            timestamp=current_bar.timestamp, equity_curve=planner_equity_curve,
            daily_history_hlc=planner_hlc, daily_volume=planner_volume, # planner_hlc is List[Tuple[float,float,float]]
            current_positions=context_current_positions_list if context_current_positions_list else None,
            vol_surface=vol_surface_data,
            risk_free_rate=0.02,
            nSuccesses=50, nFailures=10
        )

        plan_start_time = time.perf_counter()
        proposal = planner_fn(planning_ctx)
        decision_ms = (time.perf_counter() - plan_start_time) * 1000
        decision_times_ms.append(decision_ms)
        proposals_generated += 1

        accepted, reason = risk_gate_fn(proposal, db_table=None, cfg=active_risk_gate_config, registry=None)

        simulated_exec_price: Optional[float] = None
        pnl_from_this_trade_event: float = 0.0

        if accepted and proposal.legs:
            proposals_accepted += 1
            for leg in proposal.legs:
                simulated_exec_price = current_bar.close
                instr_key = leg.instrument
                pos = current_positions.get(instr_key, {'qty': 0.0, 'avg_price': 0.0})
                current_leg_qty = pos['qty']; avg_leg_price = pos['avg_price']
                trade_qty_for_leg = leg.size

                if leg.direction == Direction.LONG:
                    current_cash -= trade_qty_for_leg * simulated_exec_price * PNL_MULTIPLIER_SIMPLIFIED
                    if current_leg_qty < 0:
                        qty_covered = min(trade_qty_for_leg, abs(current_leg_qty))
                        pnl_from_this_trade_event += (avg_leg_price - simulated_exec_price) * qty_covered * PNL_MULTIPLIER_SIMPLIFIED
                        pos['qty'] += qty_covered; trade_qty_for_leg -= qty_covered
                    if trade_qty_for_leg > 0:
                        new_qty_long = pos['qty'] + trade_qty_for_leg
                        pos['avg_price'] = ((avg_leg_price * pos['qty']) + (simulated_exec_price * trade_qty_for_leg)) / new_qty_long if not math.isclose(new_qty_long, 0.0) and pos['qty'] > 0 else simulated_exec_price
                        pos['qty'] = new_qty_long
                elif leg.direction == Direction.SHORT:
                    current_cash += trade_qty_for_leg * simulated_exec_price * PNL_MULTIPLIER_SIMPLIFIED
                    if current_leg_qty > 0:
                        qty_closed = min(trade_qty_for_leg, current_leg_qty)
                        pnl_from_this_trade_event += (simulated_exec_price - avg_leg_price) * qty_closed * PNL_MULTIPLIER_SIMPLIFIED
                        pos['qty'] -= qty_closed; trade_qty_for_leg -= qty_closed
                    if trade_qty_for_leg > 0:
                        new_qty_short = pos['qty'] - trade_qty_for_leg
                        pos['avg_price'] = ((avg_leg_price * abs(pos['qty'])) + (simulated_exec_price * trade_qty_for_leg)) / abs(new_qty_short) if not math.isclose(new_qty_short, 0.0) and pos['qty'] < 0 else simulated_exec_price
                        pos['qty'] = new_qty_short

                if math.isclose(pos['qty'], 0.0):
                    if instr_key in current_positions: del current_positions[instr_key]
                else: current_positions[instr_key] = pos
        elif not accepted: proposals_rejected += 1

        final_mtm_value_of_day = sum(
            pos_data['qty'] * (current_bar.close if str(instr_key_pos.value) == current_bar.instrument else pos_data['avg_price']) * PNL_MULTIPLIER_SIMPLIFIED
            for instr_key_pos, pos_data in current_positions.items()
        )
        current_total_equity = current_cash + final_mtm_value_of_day

        if not equity_curve_points or equity_curve_points[-1][0] != current_bar.timestamp:
             equity_curve_points.append((current_bar.timestamp, current_total_equity))
        else:
            equity_curve_points[-1] = (current_bar.timestamp, current_total_equity)

        replay_trades.append(ReplayTrade(
            bar_timestamp=current_bar.timestamp, instrument_id=current_bar.instrument,
            proposal_action=proposal.action,
            proposal_legs_json=json.dumps([leg.model_dump() for leg in proposal.legs]) if proposal.legs else "[]",
            accepted_by_risk_gate=accepted, rejection_reason=reason,
            simulated_execution_price=simulated_exec_price,
            pnl_from_trade=pnl_from_this_trade_event if not math.isclose(pnl_from_this_trade_event, 0.0) else None,
            planner_decision_ms=decision_ms ))

    replay_end_wall_time = datetime.datetime.now(datetime.timezone.utc)
    replay_duration_seconds = (replay_end_wall_time - replay_start_wall_time).total_seconds()
    final_equity = equity_curve_points[-1][1] if equity_curve_points else initial_equity
    total_return_pct = (final_equity / initial_equity - 1.0) if not math.isclose(initial_equity, 0.0) else 0.0
    equity_values_only = [eq_pt[1] for eq_pt in equity_curve_points]
    max_dd_pct = calculate_max_drawdown(equity_values_only) if equity_values_only else 0.0
    mean_decision_ms_val = sum(decision_times_ms) / len(decision_times_ms) if decision_times_ms else None

    return ReplayReport(
        source_file=str(getattr(bar_stream, 'name', 'Stream')),
        replay_start_time_utc=replay_start_wall_time, replay_end_time_utc=replay_end_wall_time,
        replay_duration_seconds=replay_duration_seconds, data_start_timestamp=first_bar_timestamp,
        data_end_timestamp=last_bar_timestamp, total_bars_processed=total_bars_processed,
        proposals_generated=proposals_generated, proposals_accepted=proposals_accepted,
        proposals_rejected=proposals_rejected, initial_equity=initial_equity, final_equity=final_equity,
        total_return_pct=total_return_pct, max_drawdown_pct=max_dd_pct,
        mean_planner_decision_ms=mean_decision_ms_val, equity_curve=equity_curve_points
    ), replay_trades
