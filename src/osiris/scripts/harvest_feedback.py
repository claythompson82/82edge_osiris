# src/osiris/scripts/harvest_feedback.py

import argparse
import json
import datetime
from pathlib import Path

import lancedb


def record_matches_filter(
    record: dict,
    cutoff_ns: int,
    schema_version: str = None,
    feedback_type: str = "correction",
) -> bool:
    """
    Return True if `record` passes the given filters:
    - 'when' timestamp >= cutoff_ns
    - schema_version startswith given prefix (if any)
    - feedback_type exactly matches (default = "correction")
    """
    # time filter
    if record.get("when", 0) < cutoff_ns:
        return False
    # version prefix filter
    if schema_version and not record.get("schema_version", "").startswith(schema_version):
        return False
    # type filter
    if feedback_type and record.get("feedback_type") != feedback_type:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Harvest phi3_feedback → JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--schema-version",
        help="Only include records whose schema_version starts with this prefix (e.g. '1.0')",
    )
    parser.add_argument(
        "--feedback-type",
        help="Only include records with this exact feedback_type (overrides default 'correction')",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        help="Only include records from the last N days",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to the DB in the directory of the output file
    db = lancedb.connect(out_path.parent)

    # Fetch the phi3_feedback table (support both real LanceDB and test DummyDB)
    try:
        # DummyDB and some clients expose .table()
        table = db.table("phi3_feedback")
    except Exception:
        # Fallback for real LanceDBConnection
        table = db.open_table("phi3_feedback")

    # Pull all rows into an Arrow table and then into Python dicts
    arrow_tbl = table.to_arrow()
    all_rows = arrow_tbl.to_pylist()

    # Determine if we need to filter at all
    use_filter = bool(args.schema_version or args.days_back or args.feedback_type)

    # Prepare cutoff timestamp in nanoseconds
    cutoff_ns = 0
    if args.days_back is not None:
        cutoff = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=args.days_back)
        )
        cutoff_ns = int(cutoff.timestamp() * 1e9)

    # Write out JSONL
    with open(out_path, "w") as out_f:
        for rec in all_rows:
            if use_filter:
                # Always default to filtering "correction" unless user provided --feedback-type
                ft = args.feedback_type if args.feedback_type is not None else "correction"
                if not record_matches_filter(
                    rec,
                    cutoff_ns=cutoff_ns,
                    schema_version=args.schema_version,
                    feedback_type=ft,
                ):
                    continue
            out_f.write(json.dumps(rec))
            out_f.write("\n")


if __name__ == "__main__":
    main()
