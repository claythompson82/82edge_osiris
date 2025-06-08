import argparse
import datetime
import json
import lancedb
import os


def main():
    """
    Parses command-line arguments to harvest specific feedback data from a LanceDB
    table and writes the results to a JSONL file.
    """
    parser = argparse.ArgumentParser(description="Harvest feedback data from LanceDB.")
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to filter records from.",
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
        help="Schema version prefix to filter records by (e.g., '1.0' matches '1.0' and '1.0.1').",
    )
    args = parser.parse_args()

    db_path = os.getenv("LANCEDB_URI", "/app/lancedb_data")
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

    # Calculate the cutoff timestamp in nanoseconds for the WHERE clause.
    cutoff_date = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(days=args.days_back)
    cutoff_timestamp_ns = int(cutoff_date.timestamp() * 1_000_000_000)

    # Build the WHERE clause for the SQL-like query.
    where_clause = (
        "feedback_type = 'correction' "
        "AND corrected_proposal IS NOT NULL AND corrected_proposal != '' "
        f"AND when >= {cutoff_timestamp_ns} "
        f"AND schema_version LIKE '{args.schema_version}%'"
    )

    # Try to filter using LanceDB's engine first, with a fallback to Python filtering
    # for compatibility with older LanceDB versions.
    try:
        query = table.search().where(where_clause)
        if args.max is not None:
            query = query.limit(args.max)
        results = query.to_list()
    except Exception as query_error:
        print(f"Warning: LanceDB query failed ('{query_error}'). Falling back to manual Python filtering.")
        all_records = table.to_arrow().to_pylist()
        results = []
        for r in all_records:
            if (
                r.get("feedback_type") == "correction"
                and r.get("corrected_proposal") not in (None, "")
                and r.get("when", 0) >= cutoff_timestamp_ns
                and str(r.get("schema_version", "")).startswith(args.schema_version)
            ):
                results.append(r)
        if args.max is not None:
            results = results[: args.max]


    # Process and write the filtered records to the output file.
    count = 0
    with open(args.out, "w") as f:
        for record in results:
            prompt_text = (
                record.get("assessment")
                or record.get("proposal")
                or record.get("feedback_content", "")
            )
            corrected_proposal_raw = record.get("corrected_proposal")

            if not corrected_proposal_raw:
                continue

            # Ensure the corrected proposal is a properly formatted JSON string.
            try:
                if isinstance(corrected_proposal_raw, str):
                    corrected_proposal_dict = json.loads(corrected_proposal_raw)
                elif isinstance(corrected_proposal_raw, dict):
                    corrected_proposal_dict = corrected_proposal_raw
                else:
                    print(f"Warning: Skipping record with unexpected proposal type: {type(corrected_proposal_raw)}")
                    continue
                response_text = json.dumps(corrected_proposal_dict, indent=2)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse corrected_proposal JSON. Skipping record.")
                continue

            # Construct the final output record.
            output_record = {
                "transaction_id": record.get("transaction_id"),
                "schema_version": record.get("schema_version"), # Keep schema version for provenance
                "prompt": prompt_text,
                "response": response_text,
            }
            f.write(json.dumps(output_record) + "\n")
            count += 1

    print(f"Successfully wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()