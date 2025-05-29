# scripts/verify_advice.py
import lancedb
import pathlib
import logging
import os
import sys # To exit with error code

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VERIFY_ADVICE - %(levelname)s - %(message)s')

LANCEDB_PATH = "./lancedb_data" # Relative to project root, where docker-compose mounts it
ADVICE_TABLE_NAME = "advice" # As defined in advisor/risk_gate.py

def check_advice_entries():
    logging.info(f"Starting advice verification. Checking LanceDB path: {LANCEDB_PATH}, Table: {ADVICE_TABLE_NAME}")

    db_path = pathlib.Path(LANCEDB_PATH)
    if not db_path.exists() or not db_path.is_dir():
        logging.error(f"LanceDB directory not found at {LANCEDB_PATH}. Cannot verify advice.")
        sys.exit(1) # Exit with error

    try:
        db = lancedb.connect(db_path)
        table_names = db.table_names()
        logging.info(f"Available tables in LanceDB: {table_names}")

        if ADVICE_TABLE_NAME not in table_names:
            logging.error(f"'{ADVICE_TABLE_NAME}' table not found in LanceDB. Current tables: {table_names}")
            sys.exit(1) # Exit with error
        
        advice_table = db.open_table(ADVICE_TABLE_NAME)
        row_count = len(advice_table) # This counts all rows in the table.
        
        logging.info(f"Found {row_count} entries in the '{ADVICE_TABLE_NAME}' table.")

        if row_count >= 1:
            logging.info("Verification successful: At least one advice entry found.")
            sys.exit(0) # Success
        else:
            logging.error("Verification failed: No advice entries found in the table.")
            sys.exit(1) # Error
            
    except Exception as e:
        logging.error(f"An error occurred during advice verification: {e}", exc_info=True)
        sys.exit(1) # Error

if __name__ == "__main__":
    check_advice_entries()
