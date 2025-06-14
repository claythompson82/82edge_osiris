"""
harvest_feedback.py

Extracts feedback records from the phi3_feedback table and writes them as JSONL.
Includes filtering by schema version, feedback_type, and days_back (for tests and data pipeline use).

Exports:
    - record_matches_filter: For test_harvest_filter_function.py and others.
"""
import argparse
import json
import os
import datetime
from typing import Optional, Any, Dict

try:
    from llm_sidecar import db as lls_db
except ImportError:
    # Fallback for some test environments
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../llm_sidecar')))
    import db as lls_db

def record_matches_filter(
    record: Dict[str, Any],
    schema_version: Optional[str] = None,
    feedback_type: Optional[str] = None,
    days_back: Optional[int] = None,
    now_utc: Optional[datetime.datetime] = None
) -> bool:
    """
    Returns True if record matches all filter criteria.
    - schema_version: String prefix match (e.g., "1.0" matches "1.0.1").
    - feedback_type: Exact match.
    - days_back: Only include records newer than now - days_back.
    """
    if schema_version:
        rec_ver = str(record.get("schema_version", ""))
        # Accepts prefix matches (e.g., "1.0" will match "1.0.1" and "1.0")
        if not rec_ver.startswith(schema_version):
            return False
    if feedback_type:
        if record.get("feedback_type") != feedback_type:
            return False
    if days_back is not None:
        # Try several timestamp fields (int nanoseconds "when", or ISO "timestamp")
        now_utc = now_utc or datetime.datetime.now(datetime.timezone.utc)
        threshold = now_utc - datetime.timedelta(days=days_back)
        # Handle nanosecond integer timestamps
        when = record.get("when")
        if when:
            # convert to datetime (assume UTC)
            try:
                ts = datetime.datetime.utcfromtimestamp(int(when) / 1e9).replace(tzinfo=datetime.timezone.utc)
                if ts < threshold:
                    return False
            except Exception:
                pass  # fallback to ISO below
        elif "timestamp" in record:
            try:
                ts = datetime.datetime.fromisoformat(record["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                if ts < threshold:
                    return False
            except Exception:
                pass  # If unparseable, let it through (for legacy/test records)
    return True

def main():
    parser = argparse.ArgumentParser(description="Harvest phi3_feedback to JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--schema-version", help="Only include schema version (prefix match, e.g., '1.0')")
    parser.add_argument("--feedback-type", help="Only include given feedback_type")
    parser.add_argument("--days-back", type=int, help="Only include records from the last N days")
    args = parser.parse_args()

    tbl = lls_db._tables["phi3_feedback"]
    df = tbl.to_pandas()

    # Apply filters if requested
    if not df.empty:
        records = df.to_dict(orient="records")
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        filtered_records = [
            rec for rec in records
            if record_matches_filter(
                rec,
                schema_version=args.schema_version,
                feedback_type=args.feedback_type,
                days_back=args.days_back,
                now_utc=now_utc,
            )
        ]
    else:
        filtered_records = []

    # Write the filtered (or all, if no filters) records to the output file
    with open(args.out, "w") as f:
        for rec in filtered_records:
            json.dump(rec, f, default=str)
            f.write("\n")

if __name__ == "__main__":
    main()
