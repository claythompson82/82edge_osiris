import lancedb
import os
from pydantic import BaseModel
from typing import Any, Optional, Dict, List

# Define the target schema for the table, including schema_version
class FeedbackSchemaWithVersion(BaseModel):
    transaction_id: str
    timestamp: str  # Assuming ISO format string
    feedback_type: str
    feedback_content: Any  # Using Any for flexibility as in original task
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str

def migrate_data():
    db_path = "/app/lancedb_data"
    original_table_name = "phi3_feedback"
    temp_table_name = "phi3_feedback_migrated"
    
    processed_count = 0

    print(f"Starting migration for table '{original_table_name}' in database '{db_path}'...")

    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        print(f"Error: Could not connect to LanceDB at {db_path}. {e}")
        return

    # 1. Attempt to open the original table
    try:
        original_table = db.open_table(original_table_name)
        print(f"Successfully opened original table '{original_table_name}'.")
    except Exception as e: # LanceDB often raises generic Exception or OS-level errors like FileNotFoundError
        print(f"Error: Original table '{original_table_name}' not found or could not be opened. {e}")
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
        # Ensure record is a mutable dict
        record_dict = dict(record)
        if record_dict.get("schema_version") is None:
            record_dict["schema_version"] = "1.0"
        processed_records.append(record_dict)
    
    processed_count = len(processed_records)
    print(f"Processed {processed_count} records. Added 'schema_version' where missing.")

    # 4. If the temporary table already exists, delete it.
    try:
        if temp_table_name in db.table_names():
            db.drop_table(temp_table_name)
            print(f"Deleted existing temporary table '{temp_table_name}'.")
    except Exception as e:
        print(f"Error deleting existing temporary table '{temp_table_name}'. {e}")
        return

    # 5. Create the new temporary table with the defined schema
    try:
        temp_table = db.create_table(temp_table_name, schema=FeedbackSchemaWithVersion)
        print(f"Created new temporary table '{temp_table_name}' with target schema.")
    except Exception as e:
        print(f"Error creating temporary table '{temp_table_name}'. {e}")
        # Attempt to clean up if records were processed but table creation failed
        if processed_count > 0 and temp_table_name in db.table_names():
             db.drop_table(temp_table_name)
        return

    # 6. Add all processed records to this temporary table
    if processed_records:
        try:
            temp_table.add(processed_records)
            print(f"Added {len(processed_records)} records to '{temp_table_name}'.")
        except Exception as e:
            print(f"Error adding records to temporary table '{temp_table_name}'. {e}")
            # Clean up by dropping the temp table if add fails
            db.drop_table(temp_table_name)
            return
    else:
        print("No records to add to the temporary table.")

    # 7. Delete the original table
    try:
        if original_table_name in db.table_names():
            db.drop_table(original_table_name)
            print(f"Deleted original table '{original_table_name}'.")
        else:
            print(f"Original table '{original_table_name}' not found for deletion, perhaps already deleted or renamed.")
    except Exception as e:
        print(f"Error deleting original table '{original_table_name}'. {e}")
        # At this point, we have data in temp_table. Critical decision here.
        # For safety, maybe we should not proceed with rename if old table deletion fails.
        # However, if the old table was already gone, that's okay.
        # The check `original_table_name in db.table_names()` handles the "already gone" case.
        # If it exists and deletion fails, that's a more serious issue.
        return


    # 8. Rename the temporary table to the original table name
    try:
        db.rename_table(temp_table_name, original_table_name)
        print(f"Renamed temporary table '{temp_table_name}' to '{original_table_name}'.")
    except Exception as e:
        print(f"Error renaming temporary table '{temp_table_name}' to '{original_table_name}'. {e}")
        print(f"CRITICAL: Data is now in '{temp_table_name}'. Manual intervention may be required.")
        return

    print(f"Migration complete. {processed_count} records were processed and migrated to the new schema in '{original_table_name}'.")

if __name__ == "__main__":
    # Ensure the lancedb_data directory exists, similar to db.py, for consistency if script is run standalone
    # and the main app hasn't created it yet (though for a migration, it's expected to exist).
    db_dir = "/app/lancedb_data"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created database directory: {db_dir}")
    
    migrate_data()
