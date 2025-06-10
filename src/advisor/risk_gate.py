import pathlib
import lancedb
from pydantic import BaseModel, Field
import datetime
import uuid
from typing import Dict, Optional

# --- Configuration ---
# Default LanceDB path, can be overridden if specific config is passed to functions.
DEFAULT_LANCEDB_PATH = "/app/lancedb_data"


# --- Pydantic Models ---
class AdviceLog(BaseModel):
    # Using default_factory for fields that should be auto-generated if not provided
    advice_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    proposal: Dict  # The actual proposal dict from phi-3
    accepted: bool
    reason: str
    nav_before_trade: float
    daily_pnl_before_trade: float
    # Optional: Add schema version if you anticipate future changes
    # schema_version: str = "1.0"


# --- LanceDB Table Management ---
_db_connections = {}  # Cache for DB connections: {path: db_instance}
_table_instances = (
    {}
)  # Cache for table instances: {path_and_table_name: table_instance}


def get_db_connection(db_path_str: str = DEFAULT_LANCEDB_PATH):
    if db_path_str not in _db_connections:
        db_path = pathlib.Path(db_path_str)
        db_path.mkdir(parents=True, exist_ok=True)
        _db_connections[db_path_str] = lancedb.connect(db_path)
    return _db_connections[db_path_str]


def init_advice_table(
    db_path_str: str = DEFAULT_LANCEDB_PATH, table_name: str = "advice"
):
    table_key = f"{db_path_str}_{table_name}"
    if table_key in _table_instances:
        return _table_instances[table_key]

    db = get_db_connection(db_path_str)
    try:
        tbl = db.open_table(table_name)
    except FileNotFoundError:
        tbl = db.create_table(table_name, schema=AdviceLog)
    _table_instances[table_key] = tbl
    return tbl


def log_decision(
    advice_log_entry: AdviceLog,
    db_path_str: str = DEFAULT_LANCEDB_PATH,
    table_name: str = "advice",
):
    """Adds a new advice log entry to the specified table."""
    tbl = init_advice_table(db_path_str, table_name)  # Ensures table is initialized
    try:
        tbl.add([advice_log_entry.model_dump()])
    except Exception as e:
        # Basic error handling, consider more sophisticated logging for production
        print(f"Error logging advice to LanceDB table '{table_name}': {e}")
        # Potentially re-raise or handle more gracefully


# --- Risk Gate Logic ---
def accept(
    proposal: Dict, current_nav: float, daily_pnl: float, risk_config: Dict
) -> Dict:
    """
    Evaluates a trade proposal against configured risk rules.

    Args:
        proposal: The trade proposal dictionary. Expected to contain at least
                  'quantity' and 'price_estimate' to calculate proposed value.
                  Example: {"ticker": "XYZ", "action": "BUY", "quantity": 100, "price_estimate": 150.0, ...}
        current_nav: Current Net Asset Value.
        daily_pnl: Current daily cumulative Profit and Loss.
        risk_config: Dictionary with risk parameters:
                     {'max_position_size_pct': float, 'max_daily_loss_pct': float}
                     Example: {"max_position_size_pct": 0.01, "max_daily_loss_pct": 0.02}

    Returns:
        A dictionary {"accepted": bool, "reason": str}.
    """
    if not proposal or not isinstance(proposal, dict):
        return {"accepted": False, "reason": "Invalid or empty proposal."}

    quantity = proposal.get("quantity")
    price_estimate = proposal.get("price_estimate")

    if quantity is None or price_estimate is None:
        return {
            "accepted": False,
            "reason": "Proposal missing 'quantity' or 'price_estimate' for risk assessment.",
        }

    try:
        quantity = float(quantity)
        price_estimate = float(price_estimate)
    except ValueError:
        return {
            "accepted": False,
            "reason": "'quantity' or 'price_estimate' are not valid numbers.",
        }

    if current_nav <= 0:  # NAV should be positive
        return {
            "accepted": False,
            "reason": f"Current NAV is non-positive ({current_nav}). Cannot assess risk.",
        }

    proposed_value = quantity * price_estimate
    max_pos_size_pct = risk_config.get("max_position_size_pct", 0.01)  # Default 1%
    max_daily_loss_pct = risk_config.get("max_daily_loss_pct", 0.02)  # Default 2%

    # Rule 1: Position Size
    # Ensure proposed_value is positive for this check if side matters (e.g. shorting gives negative value)
    # For simplicity, using absolute value for position sizing relative to NAV.
    position_size_nav_pct = abs(proposed_value) / current_nav
    if position_size_nav_pct >= max_pos_size_pct:
        return {
            "accepted": False,
            "reason": f"Position size ({position_size_nav_pct*100:.2f}% of NAV) exceeds max ({max_pos_size_pct*100:.2f}% NAV). Proposed value: {proposed_value:.2f}, NAV: {current_nav:.2f}.",
        }

    # Rule 2: Daily P&L
    # P&L is a loss if negative. Rule: P&L / NAV > -max_daily_loss_pct
    # This means current P&L percentage should not be worse than the negative threshold.
    current_pnl_pct = daily_pnl / current_nav
    if (
        current_pnl_pct <= -max_daily_loss_pct
    ):  # If P&L is already -3% and limit is -2%, reject.
        return {
            "accepted": False,
            "reason": f"Daily P&L ({current_pnl_pct*100:.2f}% of NAV) has breached the max daily loss limit (-{max_daily_loss_pct*100:.2f}% NAV). P&L: {daily_pnl:.2f}, NAV: {current_nav:.2f}.",
        }

    # All rules passed
    return {"accepted": True, "reason": "Proposal meets risk criteria."}


if __name__ == "__main__":
    # --- Example Usage & Basic Test ---
    print("Running example usage of advisor.risk_gate.py...")

    # Mock data
    mock_proposal_ok = {
        "ticker": "TEST",
        "action": "BUY",
        "quantity": 10,
        "price_estimate": 100.0,
        "run_id": "run_123",
    }  # value = 1000
    mock_proposal_too_large = {
        "ticker": "TEST",
        "action": "BUY",
        "quantity": 20,
        "price_estimate": 100.0,
        "run_id": "run_124",
    }  # value = 2000

    mock_nav = 10000.0
    mock_pnl_ok = 100.0  # Positive P&L
    mock_pnl_breached = -300.0  # P&L is -3% of NAV (10000 * -0.02 = -200 limit)

    mock_risk_config = {
        "max_position_size_pct": 0.15,
        "max_daily_loss_pct": 0.02,
    }  # Max 15% position, max 2% daily loss

    # Initialize table (uses default path)
    advice_table = init_advice_table()
    print(f"Advice table '{advice_table.name}' initialized/opened.")

    # Test 1: Proposal OK
    print("\n--- Test 1: Proposal OK ---")
    decision1 = accept(mock_proposal_ok, mock_nav, mock_pnl_ok, mock_risk_config)
    print(f"Decision 1: {decision1}")
    if decision1["accepted"]:
        log_entry1 = AdviceLog(
            run_id=mock_proposal_ok["run_id"],
            proposal=mock_proposal_ok,
            accepted=decision1["accepted"],
            reason=decision1["reason"],
            nav_before_trade=mock_nav,
            daily_pnl_before_trade=mock_pnl_ok,
        )
        log_decision(log_entry1)
        print(f"Logged decision for {log_entry1.advice_id}")

    # Test 2: Proposal too large
    print("\n--- Test 2: Proposal too large ---")
    decision2 = accept(mock_proposal_too_large, mock_nav, mock_pnl_ok, mock_risk_config)
    print(f"Decision 2: {decision2}")
    if decision2["accepted"]:  # Should not happen for this test
        log_entry2 = AdviceLog(
            run_id=mock_proposal_too_large["run_id"],
            proposal=mock_proposal_too_large,
            accepted=decision2["accepted"],
            reason=decision2["reason"],
            nav_before_trade=mock_nav,
            daily_pnl_before_trade=mock_pnl_ok,
        )
        log_decision(log_entry2)
        print(f"Logged decision for {log_entry2.advice_id}")
    else:
        # Still log the rejected advice
        log_entry2_rejected = AdviceLog(
            run_id=mock_proposal_too_large.get(
                "run_id", str(uuid.uuid4())
            ),  # Ensure run_id
            proposal=mock_proposal_too_large,
            accepted=decision2["accepted"],
            reason=decision2["reason"],
            nav_before_trade=mock_nav,
            daily_pnl_before_trade=mock_pnl_ok,
        )
        log_decision(log_entry2_rejected)
        print(f"Logged rejected decision for {log_entry2_rejected.advice_id}")

    # Test 3: P&L breached
    print("\n--- Test 3: P&L breached ---")
    decision3 = accept(mock_proposal_ok, mock_nav, mock_pnl_breached, mock_risk_config)
    print(f"Decision 3: {decision3}")
    # Log this decision as well
    log_entry3_rejected = AdviceLog(
        run_id=mock_proposal_ok.get("run_id", str(uuid.uuid4())),  # Ensure run_id
        proposal=mock_proposal_ok,
        accepted=decision3["accepted"],
        reason=decision3["reason"],
        nav_before_trade=mock_nav,
        daily_pnl_before_trade=mock_pnl_breached,
    )
    log_decision(log_entry3_rejected)
    print(f"Logged rejected decision for {log_entry3_rejected.advice_id}")

    # Verify by querying the table (optional, basic check)
    try:
        print(
            f"\n--- Current contents of '{advice_table.name}' table (first 5 entries): ---"
        )
        # Querying the table; you might need to install pyarrow for to_pandas() or to_arrow()
        # For basic viewing, to_lance().to_table().to_pylist() is fine
        retrieved_data = advice_table.to_lance().to_table().to_pylist()
        for i, row in enumerate(retrieved_data):
            if i >= 5:
                print(f"...and {len(retrieved_data) - 5} more rows.")
                break
            print(row)
        if not retrieved_data:
            print("Table is empty or query failed.")
    except Exception as e:
        print(f"Error querying table for verification: {e}")
        print(
            "You might need to install 'pyarrow' (pip install pyarrow) to properly query LanceDB tables for verification."
        )

    print("\nExample usage finished.")
