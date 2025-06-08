import json
import lancedb
import os
import pyarrow as pa
from typing import Any, Dict, List

# Define the target pyarrow schema for the migrated table.
# This ensures all columns have the correct, final data type.
final_schema = pa.schema(
    [
        pa.field("transaction_id", pa.string()),
        pa.field("timestamp", pa.string()),
        pa.field("feedback_type", pa.string()),
        pa.field("feedback_content", pa.string()),
        pa.field("corrected_proposal", pa.string()),
        pa.field("schema_version", pa.string()),
    ]
)


def migrate_data():
    """
    Connects to the LanceDB database, reads all records from the phi3_feedback table,
    updates them in memory to conform to schema version 1.0, and replaces the old
    table with a new one that uses a strict, final pyarrow schema.
    """
    db_path = os.getenv("LANCEDB_URI", "/app/lancedb_data")
    original_table_name = "phi3_feedback"

    print(
        f"Starting migration for table '{original_table_name}' in database '{db_path}'..."
    )

    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        print(f"FATAL: Could not connect to LanceDB at {db_path}. Error: {e}")
        return

    # Step 1: Attempt to open the original table.
    try:
        original_table = db.open_table(original_table_name)
        print(f"Successfully opened original table '{original_table_name}'.")
    except Exception as e:
        print(
            f"INFO: Original table '{original_table_name}' not found. No migration needed. Error: {e}"
        )
        return

    # Step 2: Read all records from the original table into memory.
    try:
        records = original_table.search().to_list()
        if not records:
            print("INFO: Table is empty. No migration needed.")
            return
        print(f"Read {len(records)} records from '{original_table_name}'.")
    except Exception as e:
        print(f"FATAL: Could not read records from '{original_table_name}'. Error: {e}")
        return

    # Step 3: Iterate through records in memory, normalizing their structure.
    processed_records: List[Dict[str, Any]] = []
    for record in records:
        record_dict = dict(record)

        # Ensure schema_version is present and set to "1.0".
        if record_dict.get("schema_version") is None or record_dict.get("schema_version") != "1.0":
            record_dict["schema_version"] = "1.0"

        # Ensure 'corrected_proposal' is a JSON string.
        cp = record_dict.get("corrected_proposal")
        if isinstance(cp, dict):
            record_dict["corrected_proposal"] = json.dumps(cp)
        elif cp is None:
            record_dict["corrected_proposal"] = "{}"

        # Ensure 'feedback_content' is a string.
        fc = record_dict.get("feedback_content")
        if not isinstance(fc, str):
            record_dict["feedback_content"] = str(fc)

        processed_records.append(record_dict)

    processed_count = len(processed_records)
    print(f"Processed {processed_count} records for migration.")

    # Step 4: Drop the original table to prepare for replacement.
    try:
        db.drop_table(original_table_name)
        print(f"Successfully deleted original table '{original_table_name}'.")
    except Exception as e:
        print(f"FATAL: Error deleting original table '{original_table_name}'. Error: {e}")
        return

    # Step 5: Re-create the table with the same name, now using the processed data
    # and enforcing the final, strict schema.
    try:
        db.create_table(
            original_table_name,
            data=processed_records,
            schema=final_schema,
            mode="overwrite",
        )
        print(
            f"Successfully created migrated table '{original_table_name}' with new schema and {processed_count} records."
        )
    except Exception as e:
        print(f"FATAL: Error creating migrated table '{original_table_name}'. Error: {e}")
        return

    print("Migration complete.")


def main():
    """Entry point for command-line usage."""
    db_dir = os.getenv("LANCEDB_URI", "/app/lancedb_data")
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"Created database directory: {db_dir}")
    migrate_data()


if __name__ == "__main__":
    main()