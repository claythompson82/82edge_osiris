from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List, Dict, Any # For type hints

# Ensure azr_planner is on PYTHONPATH or installed
from azr_planner.replay.loader import load_bars
from azr_planner.replay.runner import run_replay
from azr_planner.replay.schemas import ReplayReport, ReplayTrade, Bar # Bar for type hint
from azr_planner.engine import generate_plan as azr_default_planner # Default planner
from azr_planner.risk_gate import accept as azr_default_risk_gate, RiskGateConfig # Default risk gate and config


def main() -> None:
    parser = argparse.ArgumentParser(description="AZR Planner Historical Replay Harness.")
    parser.add_argument(
        "--bars",
        type=Path,
        required=True,
        help="Path to the input bar data file (CSV, Parquet, .gz supported)."
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="azr",
        choices=["azr"], # For now, only default 'azr' planner is supported
        help="Planner to use (default: 'azr'). Currently only 'azr' is implemented."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the JSONL file containing ReplayTrade records. ReplayReport summary will be printed to stdout."
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=100_000.0,
        help="Initial equity for the replay simulation (default: 100,000.0)."
    )
    # Potentially add arguments for RiskGateConfig overrides, instrument labels, etc. later

    args = parser.parse_args()

    if not args.bars.exists():
        print(f"Error: Bar data file not found at {args.bars}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    if args.out.parent and not args.out.parent.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting replay for: {args.bars}")
    print(f"Outputting ReplayTrade records to: {args.out}")
    print(f"Initial Equity: {args.initial_equity:,.2f}")

    try:
        bar_stream = load_bars(args.bars)

        # For now, planner and risk_gate are fixed to defaults
        planner_to_use = azr_default_planner
        risk_gate_to_use = azr_default_risk_gate
        # Using default RiskGateConfig, can be made configurable via CLI args later
        risk_gate_config_to_use = RiskGateConfig()

        # Determine instrument_group and granularity for Prometheus labels (simplified)
        # This could be more sophisticated, e.g. derived from filename or bar data.
        instrument_group_label = "cli_replay_group"
        granularity_label = "cli_replay_granularity"
        if "15min" in args.bars.name.lower(): # Basic inference
            granularity_label = "15min"
        elif "daily" in args.bars.name.lower() or "1d" in args.bars.name.lower():
            granularity_label = "1D"

        # Try to get a representative instrument from the first few bars for the group label
        # This requires iterating part of the stream, so make a copy or re-open if needed.
        # For simplicity, we might just use a generic label or part of the filename.
        # Let's use a generic one for now.
        # A better way would be for load_bars to also return some metadata.

        # Re-open stream for actual run if we consumed part of it for metadata
        # Or, pass the first bar's instrument if loader can peek.
        # For now, assume bar_stream is fresh.

        report, replay_trades = run_replay(
            bar_stream=bar_stream,
            initial_equity=args.initial_equity,
            planner_fn=planner_to_use,
            risk_gate_fn=risk_gate_to_use,
            risk_gate_config=risk_gate_config_to_use,
            instrument_group_label=instrument_group_label, # Simplified label
            granularity_label=granularity_label # Simplified label
        )

        print("\n--- Replay Report Summary ---")
        # Print a user-friendly summary of the report
        print(f"Replay ID: {report.replay_id}")
        print(f"Source File: {report.source_file}") # Will be "Stream" or "IterableInput" from runner
        print(f"Duration: {report.replay_duration_seconds:.2f}s")
        print(f"Data Period: {report.data_start_timestamp} to {report.data_end_timestamp}")
        print(f"Bars Processed: {report.total_bars_processed}")
        print(f"Proposals: Generated={report.proposals_generated}, Accepted={report.proposals_accepted}, Rejected={report.proposals_rejected}")
        print(f"Initial Equity: {report.initial_equity:,.2f}")
        print(f"Final Equity: {report.final_equity:,.2f}")
        print(f"Total Return: {report.total_return_pct:.2%}")
        print(f"Max Drawdown: {report.max_drawdown_pct:.2%}")
        if report.mean_planner_decision_ms is not None:
            print(f"Avg. Planner Decision Time: {report.mean_planner_decision_ms:.2f} ms")

        # Writing ReplayTrade records to JSONL
        with open(args.out, 'w', encoding='utf-8') as f_out:
            for trade_record in replay_trades:
                # Pydantic's model_dump_json() is convenient here
                f_out.write(trade_record.model_dump_json() + "\n")

        print(f"\nFull ReplayTrade records written to: {args.out}")
        # Optionally, save the full ReplayReport object as JSON too
        report_out_path = args.out.with_suffix(".report.json")
        with open(report_out_path, 'w', encoding='utf-8') as f_report_out:
            f_report_out.write(report.model_dump_json(indent=2))
        print(f"Full ReplayReport JSON saved to: {report_out_path}")


    except FileNotFoundError: # Already checked by arg.bars.exists(), but as fallback
        print(f"Error: Input bar data file not found at {args.bars}", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        print(f"ValueError during replay: {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during replay: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure PYTHONPATH includes src if running directly for azr_planner imports
    # e.g., export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    # This is typically handled by `pip install -e .` for editable installs.
    main()
