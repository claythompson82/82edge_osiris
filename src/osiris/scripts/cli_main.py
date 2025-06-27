# osiris/scripts/cli_main.py

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import json # For report dumping

# --- Placeholder for original chrono logic if it needs to be preserved ---
# (Original chrono imports and functions were here)
# For AZR-08, focusing on adding planner backtest CLI.
# Original chrono logic can be reinstated later if needed, possibly as its own subcommand.

def run_planner_backtest(args: argparse.Namespace) -> None:
    """Handles the 'planner backtest' command."""
    # Imports are done here to avoid loading heavy planner/backtest modules
    # if another CLI command is run.
    from azr_planner.datasets import load_sp500_sample
    from azr_planner.backtest.core import run_backtest

    print(f"Running AZR Planner backtest...")
    print(f"Dataset: {args.dataset}")

    contexts = None
    if args.dataset == "sp500_sample":
        try:
            contexts = load_sp500_sample()
            if not contexts:
                print(f"Error: Dataset '{args.dataset}' loaded no contexts. Check CSV file and loader logic.")
                sys.exit(1) # Exit if dataset loading fails critically
        except FileNotFoundError:
            print(f"Error: Dataset file for '{args.dataset}' not found. Please ensure 'src/azr_planner/datasets/sp500_sample.csv' exists.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading dataset '{args.dataset}': {e}")
            sys.exit(1)
    else:
        print(f"Error: Unsupported dataset '{args.dataset}'. Currently only 'sp500_sample' is supported.")
        sys.exit(1)

    print(f"Loaded {len(contexts)} PlanningContexts for backtest.")
    if len(contexts) < 2:
         print(f"Error: Not enough PlanningContexts ({len(contexts)}) to run a backtest (min 2 required).")
         sys.exit(1)

    print(f"Starting backtest simulation...")
    try:
        report = run_backtest(contexts)
    except Exception as e:
        print(f"Error during backtest execution: {e}")
        # Consider more detailed error logging here or re-raising
        sys.exit(1)

    output_path = Path(args.out)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(by_alias=True, indent=2))
        print(f"Backtest report successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving backtest report to {output_path}: {e}")
        sys.exit(1)


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Osiris CLI tool.")
    # Make command_group optional to allow calling the script without any args to print help
    subparsers = parser.add_subparsers(dest="command_group", help="Main command group (e.g., planner)")

    # --- Planner Subcommands ---
    planner_parser = subparsers.add_parser("planner", help="AZR Planner related commands")
    planner_subparsers = planner_parser.add_subparsers(dest="planner_command", help="Planner action")

    # `planner backtest` command
    backtest_parser = planner_subparsers.add_parser("backtest", help="Run a planner backtest")
    backtest_parser.add_argument(
        "--dataset",
        type=str,
        default="sp500_sample",
        choices=["sp500_sample"],
        help="Dataset to use for the backtest (default: sp500_sample)"
    )
    backtest_parser.add_argument(
        "--out",
        type=str,
        default="backtest_report.json",
        help="Output file for the backtest report (default: backtest_report.json)"
    )
    backtest_parser.set_defaults(func=run_planner_backtest)

    # --- (Optional) Add other command groups like 'chrono' here ---
    # chrono_parser = subparsers.add_parser("chrono", help="Chrono stack commands (placeholder)")
    # chrono_parser.set_defaults(func=lambda args: print("Chrono CLI placeholder."))

    # If no arguments provided to script (e.g. "python -m osiris.scripts.cli_main"), print help.
    # The `argv` passed to `parse_args` will be empty in this case.
    # If `sys.argv[1:]` is empty, `parse_args` with empty list will result in `args`
    # where `command_group` is None.
    parsed_args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if hasattr(parsed_args, 'func') and parsed_args.func:
        parsed_args.func(parsed_args)
    elif hasattr(parsed_args, 'command_group') and parsed_args.command_group == "planner" and not getattr(parsed_args, 'planner_command', None):
        planner_parser.print_help()
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    # The set_start_method("spawn") was for torch.multiprocessing used by original chrono logic.
    # If that logic is not being run by default, this might not be strictly necessary here.
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method("spawn")
    # except RuntimeError:
    #     pass # Already set or not applicable
    cli_main()
