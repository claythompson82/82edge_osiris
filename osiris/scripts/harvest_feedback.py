import argparse
import datetime
import json
import lancedb
import os


def main():
    parser = argparse.ArgumentParser(description="Harvest feedback data from LanceDB.")
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to filter records from the 'when' column.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="feedback_data.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of records to output.",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="1.0",
        help="Schema version to filter records by.",
    )
    args = parser.parse_args()

    db_path = "/app/lancedb_data"
    table_name = "phi3_feedback"

    if not os.path.exists(db_path):
        print(f"Error: LanceDB path {db_path} does not exist.")
        return

    try:
        db = lancedb.connect(db_path)
        table = db.open_table(table_name)
    except Exception as e:
        print(f"Error connecting to LanceDB or opening table: {e}")
        return

    # Calculate the cutoff timestamp
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=args.days_back
    )

    # Build the WHERE clause for SQL query
    # Note: LanceDB uses SQL syntax for filtering.
    # Timestamps in LanceDB are often stored as nanoseconds or microseconds.
    # Assuming it's stored in a compatible format (e.g., Unix timestamp in seconds or a string that can be cast).
    # For this example, let's assume 'when' is a Unix timestamp in seconds.
    # We'll need to adjust this if the actual schema is different.
    # It's also common to store timestamps as ISO 8601 strings.
    # If 'when' is an ISO 8601 string:
    # where_clause = f"feedback_type = 'correction' AND corrected_proposal IS NOT NULL AND corrected_proposal != '' AND CAST(when AS TIMESTAMP) >= TIMESTAMP '{cutoff_date.isoformat()}'"

    # Let's assume 'when' is a float/int representing Unix timestamp in seconds for now
    # This might need adjustment based on how LanceDB handles timestamp comparisons.
    # A robust way is to convert cutoff_date to the same format as stored in 'when'.
    # If 'when' is stored as Arrow timestamp (nanoseconds):
    cutoff_timestamp_ns = int(cutoff_date.timestamp() * 1_000_000_000)
    # where_clause = f"feedback_type = 'correction' AND corrected_proposal IS NOT NULL AND corrected_proposal != '' AND \"when\" >= {cutoff_timestamp_ns}"
    # Updated where_clause to include schema_version
    where_clause = f"feedback_type = 'correction' AND corrected_proposal IS NOT NULL AND corrected_proposal != '' AND \"when\" >= {cutoff_timestamp_ns} AND schema_version = '{args.schema_version}'"

    # Some LanceDB versions may not support complex where clauses in search().
    # Fallback to loading all records and filtering in Python for test
    try:
        query = table.search().where(where_clause)
        if args.max is not None:
            query = query.limit(args.max)
        results = query.to_list()
    except Exception:
        all_records = table.to_list()
        results = [
            r for r in all_records
            if r.get("feedback_type") == "correction"
            and r.get("corrected_proposal") not in (None, "")
            and r.get("when", 0) >= cutoff_timestamp_ns
            and str(r.get("schema_version", "")).startswith(args.schema_version)
        ]
        if args.max is not None:
            results = results[: args.max]

    count = 0
    with open(args.out, "w") as f:
        for record in results:
            prompt_text = record.get("assessment") or record.get("proposal", "")
            corrected_proposal_raw = record.get("corrected_proposal")

            if not prompt_text or not corrected_proposal_raw:
                continue  # Skip if essential data is missing

            try:
                # Assuming corrected_proposal is a JSON string that needs to be parsed
                # and then pretty-printed. If it's already a Python dict, this might not be needed.
                if isinstance(corrected_proposal_raw, str):
                    corrected_proposal_dict = json.loads(corrected_proposal_raw)
                elif isinstance(corrected_proposal_raw, dict):
                    corrected_proposal_dict = corrected_proposal_raw
                else:
                    # If it's neither string nor dict, try to represent it as string, or skip
                    print(
                        f"Warning: corrected_proposal is of unexpected type: {type(corrected_proposal_raw)}. Skipping record."
                    )
                    continue

                response_text = json.dumps(corrected_proposal_dict, indent=2)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse corrected_proposal as JSON: {corrected_proposal_raw}. Storing as raw string."
                )
                # Fallback: store the raw string if it's not valid JSON, though the requirement is pretty-printed JSON.
                # Depending on strictness, we might choose to skip or handle differently.
                # For now, let's skip if it's meant to be JSON but isn't.
                # response_text = corrected_proposal_raw # Alternative: store as is
                continue

            output_record = {"prompt": prompt_text, "response": response_text}
            f.write(json.dumps(output_record) + "\n")
            count += 1
            if args.max is not None and count >= args.max:
                break

    print(f"Successfully wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
