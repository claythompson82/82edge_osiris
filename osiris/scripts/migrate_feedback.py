import json
import lancedb
import os
import pyarrow as pa
from typing import Any, Dict, List, Optional


# Define the target pyarrow schema for the migrated table
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
    db_path = "/app/lancedb_data"
    original_table_name = "phi3_feedback"
    temp_table_name = "phi3_feedback_migrated"

    processed_count = 0

    print(
        f"Starting migration for table '{original_table_name}' in database '{db_path}'..."
    )

    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        print(f"Error: Could not connect to LanceDB at {db_path}. {e}")
        return

    # 1. Attempt to open the original table
    try:
        original_table = db.open_table(original_table_name)
        print(f"Successfully opened original table '{original_table_name}'.")
    except (
        Exception
    ) as e:  # LanceDB often raises generic Exception or OS-level errors like FileNotFoundError
        print(
            f"Error: Original table '{original_table_name}' not found or could not be opened. {e}"
        )
        print("Migration cannot proceed without the original table.")
        return

    # 2. Read all records from the original table
    try:
        records = original_table.search().to_list()
        print(f"Read {len(records)} records from '{original_table_name}'.")
    except Exception as e:
        print(f"Error: Could not read records from '{original_table_name}'. {e}")
        return

    # 3. Iterate and update records
    processed_records: List[Dict[str, Any]] = []
    for record in records:
        record_dict = dict(record)
        if record_dict.get("schema_version") is None:
            record_dict["schema_version"] = "1.0"

        cp = record_dict.get("corrected_proposal")
        if isinstance(cp, dict):
            record_dict["corrected_proposal"] = json.dumps(cp)

        fc = record_dict.get("feedback_content")
        if not isinstance(fc, str):
            record_dict["feedback_content"] = json.dumps(fc)

        processed_records.append(record_dict)

    processed_count = len(processed_records)
    print(f"Processed {processed_count} records. Added 'schema_version' where missing.")

    # 4. Delete the original table if it exists
    try:
        if original_table_name in db.table_names():
            db.drop_table(original_table_name)
            print(f"Deleted original table '{original_table_name}'.")
    except Exception as e:
        print(f"Error deleting original table '{original_table_name}'. {e}")
        return

    # 5. Create the migrated table with the defined schema
    try:
        db.create_table(
            original_table_name,
            data=processed_records,
            schema=final_schema,
            mode="overwrite",
        )
        print(
            f"Created migrated table '{original_table_name}' with new schema and {processed_count} records."
        )
    except Exception as e:
        print(f"Error creating migrated table '{original_table_name}'. {e}")
        return

    print(
        f"Migration complete. {processed_count} records were processed and migrated to the new schema in '{original_table_name}'."
    )


def main():
    """Entry point for command-line usage."""
    migrate_data()


if __name__ == "__main__":
    # Ensure the lancedb_data directory exists, similar to db.py, for consistency if script is run standalone
    # and the main app hasn't created it yet (though for a migration, it's expected to exist).
    db_dir = "/app/lancedb_data"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created database directory: {db_dir}")

    main()
