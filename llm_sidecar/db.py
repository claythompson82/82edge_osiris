import pathlib
import lancedb
from lancedb.pydantic import LanceModel # Import LanceModel
import datetime
import uuid
from typing import Optional, Dict, Any, List
from pydantic import Field # BaseModel is replaced by LanceModel

# --- Configuration ---
DB_ROOT = pathlib.Path("/app/lancedb_data")
# Ensure DB_ROOT directory exists
DB_ROOT.mkdir(parents=True, exist_ok=True)

# --- Pydantic Schemas ---
def get_current_utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

class Phi3FeedbackSchema(LanceModel):
    transaction_id: str
    timestamp: str = Field(default_factory=get_current_utc_iso)
    feedback_type: str
    feedback_content: str  # Can be JSON string
    schema_version: str = "1.0"
    corrected_proposal: Optional[str] = None # Changed from Dict[str, Any] to str

class OrchestratorRunSchema(LanceModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=get_current_utc_iso)
    input_query: str
    final_output: Optional[str] = None # Changed from Dict[str, Any] to str
    status: str  # e.g., "SUCCESS", "FAILURE"
    error_message: Optional[str] = None

class HermesScoreSchema(LanceModel):
    proposal_id: uuid.UUID
    timestamp: str = Field(default_factory=get_current_utc_iso)
    score: float
    reasoning: Optional[str] = None

# --- Database Connection and Table Initialization ---
_db = lancedb.connect(DB_ROOT)

TABLE_SCHEMAS = {
    "phi3_feedback": Phi3FeedbackSchema,
    "orchestrator_runs": OrchestratorRunSchema,
    "hermes_scores": HermesScoreSchema,
}

# Store table objects in a dictionary
_tables: Dict[str, Any] = {}

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    for table_name, schema_model in TABLE_SCHEMAS.items(): # schema_model is now a LanceModel
        try:
            table = _db.open_table(table_name)
        except ValueError: # Catch ValueError, as lancedb seems to raise this
            table = _db.create_table(table_name, schema=schema_model) # Pass LanceModel directly
        _tables[table_name] = table
    # For compatibility with old global variable names, if needed elsewhere (though should be refactored)
    global feedback_tbl, osiris_runs_tbl, hermes_scores_tbl
    feedback_tbl = _tables.get("phi3_feedback")
    osiris_runs_tbl = _tables.get("orchestrator_runs")
    hermes_scores_tbl = _tables.get("hermes_scores")


# --- Generic Table Helper ---
def add_to_table(table_name: str, data: LanceModel) -> None: # data is now a LanceModel instance
    """
    Adds a Pydantic model (LanceModel) instance to the specified table after validation.
    The actual async safety depends on LanceDB's capabilities with asyncio.
    For now, this is a standard synchronous function.
    """
    if table_name not in _tables:
        raise ValueError(f"Table '{table_name}' not initialized. Call init_db() first.")

    table = _tables[table_name]
    # Validate data with the schema (Pydantic does this on instantiation)
    # Convert Pydantic model to dict for LanceDB
    try:
        table.add([data.model_dump()])
    except Exception as e:
        # Handle potential LanceDB errors (e.g., schema mismatch if validation is bypassed)
        print(f"Error adding data to table '{table_name}': {e}")
        # Potentially re-raise or handle more gracefully

# --- Specific Data Logging Functions ---
def append_feedback(feedback_data: Phi3FeedbackSchema) -> None:
    """Logs feedback data."""
    add_to_table("phi3_feedback", feedback_data)

def log_run(run_data: OrchestratorRunSchema) -> None:
    """Logs orchestrator run data."""
    add_to_table("orchestrator_runs", run_data)

def log_hermes_score(score_data: HermesScoreSchema) -> None:
    """Logs Hermes evaluation scores."""
    add_to_table("hermes_scores", score_data)


# --- CLI ---
# (Keeping the CLI part for now, may need adjustments if table var names changed)
def cli_main(argv: Optional[List[str]] = None):
    import argparse
    import json # Moved import here
    import sys # For explicit stdout/stderr and exit

    parser = argparse.ArgumentParser(description="LLM Sidecar DB CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=False)

    query_parser = subparsers.add_parser("query-runs", help="Query and display orchestrator run logs.")
    query_parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of recent runs to retrieve (default: 5)."
    )
    query_parser.add_argument(
        "--table",
        type=str,
        default="orchestrator_runs",
        choices=list(_tables.keys()), # Use initialized table names
        help="Table to query (default: orchestrator_runs)."
    )


    args = parser.parse_args(argv)

    if not _tables: # Check if tables are loaded
        print("Database not initialized. Running init_db()...")
        init_db()
        if not _tables: # Still no tables after init
             print("Failed to initialize database tables. Exiting.")
             return


    if args.command == "query-runs":
        table_to_query = _tables.get(args.table)
        if table_to_query is None:
            print(f"Error: Table '{args.table}' is not initialized or does not exist.")
            return
        
        try:
            all_items_ds = table_to_query.to_lance()
            all_items_list = all_items_ds.to_table().to_pylist()
            
            # Assuming 'timestamp' field exists for sorting; make this robust if not all tables have it
            sorted_items = sorted(all_items_list, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            num_to_show = args.last
            if num_to_show <= 0:
                 items_to_display = sorted_items
            else:
                 items_to_display = sorted_items[:num_to_show]

            if not items_to_display:
                print(f"No logs found in '{args.table}'.")
                return

            for item_log in items_to_display:
                print(json.dumps(item_log, indent=2, default=str))
                print("-" * 20)

        except Exception as e:
            print(f"Error querying table '{args.table}': {e}")
            import traceback
            traceback.print_exc()


    elif args.command is None: # No subcommand was provided
        parser.print_help(sys.stdout) # Print help to stdout
        sys.exit(0) # Exit with 0
    # The final 'else' for unknown command is redundant,
    # as argparse default behavior is to print error and exit(2) for unknown commands.

# Initialize the database and tables when the module is loaded
init_db()

if __name__ == "__main__":
    cli_main()
