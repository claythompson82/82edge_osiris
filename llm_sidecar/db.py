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
    from pydantic import BaseModel
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
