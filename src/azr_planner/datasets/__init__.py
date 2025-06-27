from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from azr_planner.schemas import PlanningContext
from azr_planner.math_utils import LR_V2_MIN_POINTS

# Define the path to the CSV file relative to this datasets package
# This assumes the CSV is in the same directory as this __init__.py
_MODULE_DIR = Path(__file__).parent
SP500_SAMPLE_CSV_PATH = _MODULE_DIR / "sp500_sample.csv"

def _generate_dummy_hlc_from_closes(closes: List[float]) -> List[tuple[float, float, float]]:
    """Generates dummy HLC data where H=L=C for simplicity."""
    return [(c, c, c) for c in closes]

def load_sp500_sample() -> List[PlanningContext]:
    """
    Loads the sp500_sample.csv dataset and transforms it into a list of PlanningContext objects.
    Uses rolling windows for equity_curve and daily_history_hlc.
    """
    contexts: List[PlanningContext] = []

    if not SP500_SAMPLE_CSV_PATH.exists():
        # In a real scenario, might raise an error or log a warning.
        # For now, if file doesn't exist (e.g. not yet placed by user/CI), return empty.
        # However, the spec implies it's already committed.
        print(f"Warning: {SP500_SAMPLE_CSV_PATH} not found. Returning empty context list.")
        return contexts

    data_rows: List[Dict[str, Any]] = []
    try:
        with open(SP500_SAMPLE_CSV_PATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Assuming 'Timestamp' or 'timestamp' and 'Close' or 'close' columns
                    ts_key = 'Timestamp' if 'Timestamp' in row else 'timestamp'
                    close_key = 'Close' if 'Close' in row else 'close'

                    dt_obj = datetime.fromisoformat(row[ts_key].replace('Z', '+00:00'))
                    # Ensure it's offset-aware (UTC)
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    else:
                        dt_obj = dt_obj.astimezone(timezone.utc)

                    data_rows.append({
                        "timestamp": dt_obj,
                        "close": float(row[close_key])
                    })
                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to parsing error: {row}, error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {SP500_SAMPLE_CSV_PATH} not found during read attempt.")
        return contexts # Should have been caught by .exists() earlier, but defensive.
    except Exception as e:
        print(f"An unexpected error occurred reading {SP500_SAMPLE_CSV_PATH}: {e}")
        return contexts


    if len(data_rows) < LR_V2_MIN_POINTS:
        print(f"Warning: Not enough data rows ({len(data_rows)}) in CSV for lookback {LR_V2_MIN_POINTS}. Returning empty context list.")
        return contexts

    # Generate PlanningContext objects using a rolling window
    # Number of contexts will be len(data_rows) - LR_V2_MIN_POINTS + 1
    for i in range(len(data_rows) - LR_V2_MIN_POINTS + 1):
        window_data = data_rows[i : i + LR_V2_MIN_POINTS]

        current_timestamp = window_data[-1]["timestamp"] # Timestamp of the current context is the end of the window

        # equity_curve for PlanningContext is the list of close prices in the window
        equity_curve_window = [row["close"] for row in window_data]

        # daily_history_hlc also uses the same window of close prices
        daily_hlc_window = _generate_dummy_hlc_from_closes(equity_curve_window)

        # Dummy values for other fields
        vol_surface_dummy = {"MES": 0.20, "M2K": 0.25} # Example
        risk_free_rate_dummy = 0.02

        try:
            context = PlanningContext( # type: ignore[call-arg]
                timestamp=current_timestamp,
                equity_curve=equity_curve_window,
                daily_history_hlc=daily_hlc_window,
                daily_volume=None,
                current_positions=None,
                n_successes=0,
                n_failures=0,
                vol_surface=vol_surface_dummy,
                risk_free_rate=risk_free_rate_dummy
            )
            contexts.append(context)
        except Exception as e: # Catch potential Pydantic validation errors or others
            print(f"Error creating PlanningContext for window ending {current_timestamp}: {e}")
            continue

    if not contexts:
        print(f"No PlanningContext objects were generated from {SP500_SAMPLE_CSV_PATH}.")

    return contexts

__all__ = ["load_sp500_sample", "SP500_SAMPLE_CSV_PATH"]
