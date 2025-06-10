#!/usr/bin/env python3

import lancedb
import argparse
import sys
import os
import time

# Attempt to import the AdviceLog schema if it's defined and needed by LanceDB
# This is often only required if you're creating a table or inserting data
# with a specific schema object. For opening and counting, it might not be.
# from aistore.experimental.llm_policy.policy_common.pipeline_components.persister import AdviceLog # Adjust path as needed


def main():
    parser = argparse.ArgumentParser(description="Check for advice entries in LanceDB.")
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the LanceDB database directory.",
    )
    parser.add_argument(
        "--table-name", type=str, default="advice", help="Name of the table to check."
    )
    parser.add_argument(
        "--min-entries", type=int, default=1, help="Minimum number of entries expected."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retries to check for advice."
    )
    parser.add_argument(
        "--retry-delay", type=int, default=10, help="Delay in seconds between retries."
    )

    args = parser.parse_args()

    print(f"Attempting to connect to LanceDB at: {args.db_path}")

    if not os.path.exists(args.db_path):
        print(f"Error: LanceDB path '{args.db_path}' does not exist.")
        sys.exit(1)
    if not os.listdir(args.db_path):  # Check if the directory is empty
        print(
            f"Error: LanceDB path '{args.db_path}' is empty. Table '{args.table_name}' likely does not exist."
        )
        sys.exit(1)

    for attempt in range(args.retries):
        try:
            print(
                f"Attempt {attempt + 1}/{args.retries} to connect and check table '{args.table_name}'..."
            )
            # Connect to LanceDB
            # The URI for a local LanceDB is simply its path.
            db = lancedb.connect(args.db_path)

            table_names = db.table_names()
            print(f"Available tables: {table_names}")

            if args.table_name not in table_names:
                print(
                    f"Table '{args.table_name}' not found in database. Available: {table_names}"
                )
                if attempt < args.retries - 1:
                    print(f"Retrying in {args.retry_delay} seconds...")
                    time.sleep(args.retry_delay)
                    continue
                else:
                    print("Max retries reached. Table not found.")
                    sys.exit(1)

            tbl = db.open_table(args.table_name)
            count = tbl.count_rows()

            print(f"Found {count} entries in table '{args.table_name}'.")

            if count >= args.min_entries:
                print(
                    f"Success: Found {count} entries, which is >= minimum expectation of {args.min_entries}."
                )
                sys.exit(0)  # Success
            else:
                print(
                    f"Failure: Found {count} entries, but expected at least {args.min_entries}."
                )
                if attempt < args.retries - 1:
                    print(f"Retrying in {args.retry_delay} seconds...")
                    time.sleep(args.retry_delay)
                else:
                    print("Max retries reached. Condition not met.")
                    sys.exit(1)  # Failure

        except Exception as e:
            print(f"Error connecting to LanceDB or querying table: {e}")
            if attempt < args.retries - 1:
                print(f"Retrying in {args.retry_delay} seconds...")
                time.sleep(args.retry_delay)
            else:
                print("Max retries reached. Error during DB operation.")
                sys.exit(1)  # Failure after retries

    # Should not be reached if logic is correct
    sys.exit(1)


if __name__ == "__main__":
    main()
