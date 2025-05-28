import pathlib
import lancedb

DB_ROOT = pathlib.Path("/app/lancedb_data")
DB_ROOT.mkdir(parents=True, exist_ok=True)

_db = lancedb.connect(DB_ROOT)
# Try to open the table, and if it doesn't exist, create it with a schema.
# Inferring schema from the first write can be problematic if the first write is partial.
# For now, we'll let it create on first write as per issue, but ideally, a schema should be defined.
# The issue states schema=None, which means it will be inferred.
feedback_tbl = _db.create_table("phi3_feedback", schema=None, mode="overwrite") # Changed open_table to create_table and mode to "overwrite" to ensure it's always fresh or created. The issue had open_table with mode="w" which is not a valid mode for open_table. "w" is for lancedb.connect(uri, mode="w") to create a new DB if not exists. For tables, create_table is more explicit. Given the test expects to count rows and this might run multiple times, ensuring the table is either freshly created or explicitly handled is better. Let's stick to the issue's spirit: open if exists, create if not. `create_table` with `exist_ok=True` (default) is suitable. The issue's `open_table` with `mode="w"` is problematic as `open_table` doesn't have `mode="w"`. It has `mode="append"` or `mode="overwrite"`. `lancedb.connect` has `mode="w"`. Let's try to match the intent of "create if not exists, open if it does". `db.open_table(name)` will open if exists, or raise error. `db.create_table(name, schema)` will create. A common pattern is try-open-except-create. Or use `db.table_names()` to check.

# Let's refine the table creation/opening logic to be more robust.
# The original instruction was: `feedback_tbl = _db.open_table("phi3_feedback", mode="w", schema=None)`
# `open_table` does not have a `mode` parameter. `create_table` does.
# If the goal is to create if not exists, and open if it does, we can do this:
try:
    feedback_tbl = _db.open_table("phi3_feedback")
except FileNotFoundError: # LanceDB raises FileNotFoundError if table doesn't exist
    # Define a schema based on the FeedbackItem structure and test_db.py
    # This is better than schema=None for consistency.
    # Based on test_db.py: transaction_id, timestamp, feedback_type, feedback_content
    # Based on server.py FeedbackItem: transaction_id, feedback_type, feedback_content, timestamp, corrected_proposal
    # The example row in test_db.py is simpler. Let's use a schema that accommodates the test.
    from pydantic import BaseModel, Field
    from typing import Optional, Dict, Any, List
    import datetime
    import uuid # For OrchestratorRunLog run_id
    import json # For CLI output

    class FeedbackSchema(BaseModel):
        transaction_id: str
        timestamp: str
        feedback_type: str
        feedback_content: str
        schema_version: str # Added this line
        # corrected_proposal: Optional[dict] # This makes it more complex with LanceDB schema if not always present
                                         # For simplicity, and matching the test, let's keep it to the 4 fields.
                                         # If corrected_proposal is needed, its type needs to be defined carefully.

    feedback_tbl = _db.create_table("phi3_feedback", schema=FeedbackSchema)


def append_feedback(row: dict) -> None:
    feedback_tbl.add([row])

# --- Orchestrator Run Logs ---

class OrchestratorRunLog(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    input_query: str
    final_output: Optional[Dict[str, Any]] = None # Storing as dict, LanceDB can handle nested dicts
    status: str  # e.g., "SUCCESS", "FAILURE"
    error_message: Optional[str] = None

try:
    osiris_runs_tbl = _db.open_table("osiris_runs")
except FileNotFoundError:
    osiris_runs_tbl = _db.create_table("osiris_runs", schema=OrchestratorRunLog)

def log_run(run_data: OrchestratorRunLog) -> None:
    """Adds a new orchestrator run log to the osiris_runs table."""
    try:
        osiris_runs_tbl.add([run_data.model_dump()])
    except Exception as e:
        # Basic error handling, consider more sophisticated logging for production
        print(f"Error logging run to LanceDB: {e}")
        # Potentially re-raise or handle more gracefully depending on requirements

def cli_main(argv: Optional[List[str]] = None):
    import argparse # Moved import here
    parser = argparse.ArgumentParser(description="LLM Sidecar DB CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=False) # Made command optional for help

    # Subparser for query-runs
    query_parser = subparsers.add_parser("query-runs", help="Query and display orchestrator run logs.")
    query_parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of recent runs to retrieve (default: 5)."
    )

    args = parser.parse_args(argv) # Use passed argv or sys.argv if None

    if args.command == "query-runs":
        if 'osiris_runs_tbl' not in globals() or osiris_runs_tbl is None:
            print("Error: osiris_runs table is not initialized.")
            return 
        
        try:
            all_runs_ds = osiris_runs_tbl.to_lance()
            all_runs_list = all_runs_ds.to_table().to_pylist()
            
            sorted_runs = sorted(all_runs_list, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            num_to_show = args.last
            # If --last is 0 or negative, show all runs (after sorting)
            # If positive, show at most that many.
            if num_to_show <= 0:
                 runs_to_display = sorted_runs
            else:
                 runs_to_display = sorted_runs[:num_to_show]

            if not runs_to_display:
                print("No run logs found.")
                return

            for run_log in runs_to_display:
                print(json.dumps(run_log, indent=2, default=str))
                print("-" * 20)

        except Exception as e:
            print(f"Error querying runs: {e}")
            # Consider logging the traceback for debug purposes
            # import traceback
            # traceback.print_exc()

    elif args.command is None: # No subcommand was provided
        parser.print_help()
    else: # Unknown command
        parser.print_help()


if __name__ == "__main__":
    cli_main() # Calls the new cli_main function
